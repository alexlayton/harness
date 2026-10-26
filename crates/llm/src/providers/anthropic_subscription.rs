//! Unofficial Claude subscription provider, ported from pi-sub-anthropic/omp.
//! MIT attribution: see THIRD_PARTY_NOTICES.md. Never falls back to API billing.
use crate::dialects::anthropic_subscription::{self as wire, SubscriptionParser};
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider};
use auth::AnthropicSubscriptionAuth;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::sync::Arc;

/// Supported model IDs, not proof of entitlement or live availability.
/// Context limits are model-specific; see `context_length` below.
/// One Claude subscription model. This table is the only source for the
/// catalogue, the context limits, and the thinking capabilities that the
/// dialect applies; the lists were previously repeated (and had drifted)
/// across the provider and the dialect.
pub(crate) struct SubscriptionModel {
    pub id: &'static str,
    /// Model capacity, separate from account entitlement. `None` where no
    /// verified metadata exists; limits are not guessed.
    pub context_length: Option<u64>,
    /// Uses adaptive thinking (`Off` is unsupported).
    pub adaptive_thinking: bool,
    /// Accepts the `max` adaptive effort.
    pub max_effort: bool,
}

const fn model(
    id: &'static str,
    context_length: Option<u64>,
    adaptive_thinking: bool,
    max_effort: bool,
) -> SubscriptionModel {
    SubscriptionModel {
        id,
        context_length,
        adaptive_thinking,
        max_effort,
    }
}

const M: Option<u64> = Some(1_000_000);
const K200: Option<u64> = Some(200_000);

/// Supported models, newest first.
pub(crate) const SUBSCRIPTION_MODELS: &[SubscriptionModel] = &[
    model("claude-opus-5-5", M, true, true),
    model("claude-sonnet-5-5", None, true, true),
    model("claude-haiku-5-5", None, true, true),
    model("claude-fable-5-1", M, true, true),
    model("claude-opus-5", M, true, true),
    model("claude-sonnet-5", M, true, true),
    model("claude-fable-5", M, true, true),
    model("claude-opus-4-8", M, true, true),
    model("claude-opus-4-7", M, true, true),
    model("claude-sonnet-4-6", M, true, false),
    model("claude-opus-4-6", M, true, true),
    model("claude-opus-4-5", K200, false, false),
    model("claude-sonnet-4-5", K200, false, false),
    model("claude-haiku-4-5", K200, false, false),
];

/// Metadata for a supported model id.
pub(crate) fn subscription_model(id: &str) -> Option<&'static SubscriptionModel> {
    SUBSCRIPTION_MODELS.iter().find(|model| model.id == id)
}

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
        Ok(SUBSCRIPTION_MODELS
            .iter()
            .map(|model| ModelInfo {
                id: model.id.into(),
                name: Some(model.id.into()),
                context_length: model.context_length,
            })
            .collect())
    }
    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        if subscription_model(&req.model).is_none() {
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
        match self.stream_with(req, &bytes, thinking, &c).await {
            // An unexpired token can still be revoked server-side. Refresh
            // once (single-flight across handles) and repeat the request.
            Err(LlmError::Http { status: 401, .. }) => {
                let c = self
                    .auth
                    .refresh_rejected(&c.access)
                    .await
                    .map_err(|e| LlmError::Auth(e.to_string()))?;
                self.stream_with(req, &bytes, thinking, &c).await
            }
            result => result,
        }
    }
}

impl AnthropicSubscriptionProvider {
    async fn stream_with(
        &self,
        req: &CompletionRequest,
        bytes: &[u8],
        thinking: bool,
        c: &auth::AnthropicSubscriptionCredential,
    ) -> Result<EventStream, LlmError> {
        // Connect and timeout failures must stay `Network` so `with_retry`
        // repeats them. A reqwest error carries the URL but never request
        // headers, so the bearer value cannot leak through it; the URL is
        // still removed because the endpoint can be user-configured.
        let response = crate::http::streaming_client()
            .post(format!("{}/v1/messages", self.endpoint))
            .headers(headers(thinking, true))
            .bearer_auth(&c.access)
            .body(bytes.to_vec())
            .send()
            .await
            .map_err(|error| LlmError::Network(error.without_url()))?;
        let response = crate::http::check_status_with_secret(response, &c.access).await?;
        Ok(crate::dialects::drive_parser_stream(
            crate::sse::stream_response(response),
            SubscriptionParser::new(req),
            &c.access,
        ))
    }
}
