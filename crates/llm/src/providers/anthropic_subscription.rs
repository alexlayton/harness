//! Unofficial Claude subscription provider, ported from pi-sub-anthropic/omp.
//! MIT attribution: see THIRD_PARTY_NOTICES.md. Never falls back to API billing.
use crate::dialects::anthropic_subscription::{self as wire, SubscriptionParser};
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider};
use auth::AnthropicSubscriptionAuth;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::sync::Arc;

/// Supported metadata, not proof of entitlement or live availability. Context
/// is conservatively limited to 200K; OAuth does not advertise the 1M beta.
pub const MODELS: &[&str] = &[
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
];
/// Separate provider identity keeps existing Anthropic routes unchanged.
#[derive(Clone)]
pub struct AnthropicSubscriptionProvider {
    auth: Arc<AnthropicSubscriptionAuth>,
    endpoint: String,
}
impl AnthropicSubscriptionProvider {
    /// Construct with a shared refresh lifecycle.
    pub fn new(auth: Arc<AnthropicSubscriptionAuth>) -> Self {
        Self {
            auth,
            endpoint: "https://api.anthropic.com".into(),
        }
    }
    /// Override the origin for offline protocol fixtures.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into().trim_end_matches('/').into();
        self
    }
}
#[cfg(test)]
#[path = "anthropic_subscription_tests.rs"]
mod tests;

fn headers(thinking: bool, agent: bool) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in [
        ("accept", "application/json"),
        ("content-type", "application/json"),
        ("connection", "keep-alive"),
        ("anthropic-version", "2023-06-01"),
        ("anthropic-dangerous-direct-browser-access", "true"),
        (
            "user-agent",
            "claude-cli/2.1.280 (external, claude-desktop)",
        ),
        ("x-app", "cli"),
        ("x-stainless-lang", "js"),
        ("x-stainless-os", "Linux"),
        ("x-stainless-package-version", "0.94.0"),
        ("x-stainless-retry-count", "0"),
        ("x-stainless-runtime", "node"),
        ("x-stainless-runtime-version", "v26.3.0"),
        ("x-stainless-timeout", "600"),
    ] {
        h.insert(HeaderName::from_static(k), HeaderValue::from_static(v));
    }
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "x86" => "x86",
        other => other,
    };
    h.insert("x-stainless-arch", HeaderValue::from_str(arch).unwrap());
    let mut betas = if agent {
        "claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,advanced-tool-use-2025-11-20".to_owned()
    } else {
        "interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,structured-outputs-2025-12-15".to_owned()
    };
    if agent {
        if thinking {
            betas.push_str(",effort-2025-11-24");
        }
        betas.push_str(",fallback-credit-2026-06-01");
    }
    h.insert("anthropic-beta", HeaderValue::from_str(&betas).unwrap());
    h.insert(
        "x-client-request-id",
        HeaderValue::from_str(&uuid::Uuid::new_v4().to_string()).unwrap(),
    );
    h
}
#[async_trait::async_trait]
impl Provider for AnthropicSubscriptionProvider {
    fn name(&self) -> &'static str {
        "anthropic-claude"
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Ok(MODELS
            .iter()
            .map(|id| ModelInfo {
                id: (*id).into(),
                name: Some((*id).into()),
                context_length: Some(200000),
            })
            .collect())
    }
    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        if !MODELS.contains(&req.model.as_str()) {
            return Err(LlmError::Parse(
                "unsupported Anthropic subscription model (see list_models)".into(),
            ));
        }
        let bytes = wire::serialize(req)?;
        let thinking = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .is_some_and(|v| v.get("thinking").is_some());
        let c = self
            .auth
            .ensure_valid()
            .await
            .map_err(|e| LlmError::Auth(e.to_string()))?;
        // Transport details must not leak bearer values via reqwest errors.
        let response = crate::http::streaming_client()
            .post(format!("{}/v1/messages", self.endpoint))
            .headers(headers(thinking, true))
            .bearer_auth(&c.access)
            .body(bytes)
            .send()
            .await
            .map_err(|_| {
                LlmError::Stream("Anthropic subscription network request failed".into())
            })?;
        let response = crate::http::check_status_with_secret(response, &c.access).await?;
        Ok(crate::dialects::drive_parser_stream(
            crate::sse::stream_response(response),
            SubscriptionParser::new(req),
            &c.access,
        ))
    }
}
