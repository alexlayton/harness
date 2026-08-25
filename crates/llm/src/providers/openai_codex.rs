//! ChatGPT Codex subscription provider (SSE transport only).
use crate::dialects::openai_codex_responses::OpenAiCodexResponsesClient;
use crate::http::HttpClient;
use crate::{
    CompletionRequest, EventStream, LlmError, ModelInfo, Provider, SubscriptionUsage,
    SubscriptionUsageWindow,
};
use auth::OpenAiCodexAuth;
use reqwest::header::{ACCEPT, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use std::sync::Arc;

pub const PROVIDER_NAME: &str = "openai-codex";
pub const CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/wham";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodexModel {
    pub id: &'static str,
    pub name: &'static str,
    pub context_length: u64,
}
/// Local catalog is available before sign-in so `/model` stays useful offline.
pub const CODEX_MODELS: &[CodexModel] = &[
    CodexModel {
        id: "gpt-5.3-codex-spark",
        name: "GPT-5.3 Codex Spark",
        context_length: 1_000_000,
    },
    CodexModel {
        id: "gpt-5.4",
        name: "GPT-5.4",
        context_length: 1_000_000,
    },
    CodexModel {
        id: "gpt-5.4-mini",
        name: "GPT-5.4 mini",
        context_length: 400_000,
    },
    CodexModel {
        id: "gpt-5.5",
        name: "GPT-5.5",
        context_length: 1_000_000,
    },
    CodexModel {
        id: "gpt-5.6-luna",
        name: "GPT-5.6 Luna",
        context_length: 1_050_000,
    },
    CodexModel {
        id: "gpt-5.6-sol",
        name: "GPT-5.6 Sol",
        context_length: 1_050_000,
    },
    CodexModel {
        id: "gpt-5.6-terra",
        name: "GPT-5.6 Terra",
        context_length: 1_050_000,
    },
];
#[derive(Clone)]
pub struct OpenAiCodexProvider {
    auth: Arc<OpenAiCodexAuth>,
    endpoint: String,
}
impl std::fmt::Debug for OpenAiCodexProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCodexProvider")
            .field("endpoint", &self.endpoint)
            .field("auth", &self.auth)
            .finish()
    }
}
impl OpenAiCodexProvider {
    pub fn new(auth: Arc<OpenAiCodexAuth>) -> Self {
        Self {
            auth,
            endpoint: CODEX_ENDPOINT.into(),
        }
    }
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}
#[async_trait::async_trait]
impl Provider for OpenAiCodexProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }
    async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
        if !CODEX_MODELS.iter().any(|model| model.id == request.model) {
            return Err(LlmError::Parse(format!(
                "OpenAI Codex model `{}` is not in the supported model catalog",
                request.model
            )));
        }
        let credential = self
            .auth
            .ensure_valid()
            .await
            .map_err(|error| LlmError::Auth(error.to_string()))?;
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        headers.insert(
            HeaderName::from_static("chatgpt-account-id"),
            HeaderValue::from_str(&credential.account_id)
                .map_err(|_| LlmError::Auth("invalid Codex account id".into()))?,
        );
        headers.insert(
            HeaderName::from_static("originator"),
            HeaderValue::from_static("harness"),
        );
        headers.insert(
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static("harness/0.1"),
        );
        headers.insert(
            HeaderName::from_static("openai-beta"),
            HeaderValue::from_static("responses=experimental"),
        );
        OpenAiCodexResponsesClient::with_headers(&self.endpoint, credential.access.clone(), headers)
            .stream(request)
            .await
            .map_err(|error| redact(error, &credential.access))
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Ok(CODEX_MODELS
            .iter()
            .map(|model| ModelInfo {
                id: model.id.into(),
                name: Some(model.name.into()),
                context_length: Some(model.context_length),
            })
            .collect())
    }

    async fn subscription_usage(&self) -> Result<Option<SubscriptionUsage>, LlmError> {
        let credential = self
            .auth
            .ensure_valid()
            .await
            .map_err(|error| LlmError::Auth(error.to_string()))?;
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("chatgpt-account-id"),
            HeaderValue::from_str(&credential.account_id)
                .map_err(|_| LlmError::Auth("invalid Codex account id".into()))?,
        );
        headers.insert(
            HeaderName::from_static("originator"),
            HeaderValue::from_static("harness"),
        );
        headers.insert(
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static("harness/0.1"),
        );
        let client = HttpClient::with_headers(&self.endpoint, credential.access.clone(), headers);
        let response = client
            .get("/usage")
            .await
            .map_err(|error| redact(error, &credential.access))?;
        let body = response
            .text()
            .await
            .map_err(LlmError::Network)
            .map_err(|error| redact(error, &credential.access))?;
        parse_usage_body(&body).map(Some)
    }
}

#[derive(Debug, Deserialize)]
struct CodexUsageResponse {
    plan_type: Option<String>,
    rate_limit: Option<CodexRateLimit>,
    additional_rate_limits: Option<Vec<CodexAdditionalRateLimit>>,
}

#[derive(Debug, Deserialize)]
struct CodexAdditionalRateLimit {
    limit_name: String,
    rate_limit: Option<CodexRateLimit>,
}

#[derive(Debug, Deserialize)]
struct CodexRateLimit {
    #[serde(default)]
    limit_reached: bool,
    primary_window: Option<CodexUsageWindow>,
    secondary_window: Option<CodexUsageWindow>,
}

#[derive(Debug, Deserialize)]
struct CodexUsageWindow {
    used_percent: u16,
    limit_window_seconds: u64,
    reset_after_seconds: Option<u64>,
    reset_at: Option<i64>,
}

fn parse_usage_body(body: &str) -> Result<SubscriptionUsage, LlmError> {
    let response: CodexUsageResponse = serde_json::from_str(body)
        .map_err(|error| LlmError::Parse(format!("invalid Codex usage response: {error}")))?;
    let mut windows = Vec::new();
    if let Some(rate_limit) = response.rate_limit {
        append_codex_windows(&mut windows, None, rate_limit);
    }
    for additional in response.additional_rate_limits.unwrap_or_default() {
        if let Some(rate_limit) = additional.rate_limit {
            append_codex_windows(&mut windows, Some(&additional.limit_name), rate_limit);
        }
    }
    if windows.is_empty() {
        return Err(LlmError::Parse(
            "invalid Codex usage response: no rate-limit windows".into(),
        ));
    }
    Ok(SubscriptionUsage {
        plan: response.plan_type,
        windows,
    })
}

fn append_codex_windows(
    output: &mut Vec<SubscriptionUsageWindow>,
    group: Option<&str>,
    rate_limit: CodexRateLimit,
) {
    let status = rate_limit.limit_reached.then(|| "limit reached".into());
    for (fallback, window) in [
        ("primary", rate_limit.primary_window),
        ("secondary", rate_limit.secondary_window),
    ] {
        let Some(window) = window else {
            continue;
        };
        let label = codex_window_label(window.limit_window_seconds, fallback);
        output.push(SubscriptionUsageWindow {
            label: group.map_or(label.clone(), |group| format!("{group} · {label}")),
            used_percent: window.used_percent,
            status: status.clone(),
            resets_at: window.reset_at.map(|value| value.to_string()),
            resets_after_seconds: window.reset_after_seconds,
        });
    }
}

fn codex_window_label(seconds: u64, fallback: &str) -> String {
    const HOUR: u64 = 60 * 60;
    const DAY: u64 = 24 * HOUR;
    const WEEK: u64 = 7 * DAY;
    const MONTH: u64 = 30 * DAY;
    match seconds {
        WEEK => "weekly".into(),
        MONTH => "monthly".into(),
        DAY => "daily".into(),
        value if value > 0 && value % WEEK == 0 => count_label(value / WEEK, "week"),
        value if value > 0 && value % DAY == 0 => count_label(value / DAY, "day"),
        value if value > 0 && value % HOUR == 0 => count_label(value / HOUR, "hour"),
        _ => fallback.into(),
    }
}

fn count_label(count: u64, unit: &str) -> String {
    format!("{count} {unit}{}", if count == 1 { "" } else { "s" })
}

fn redact(error: LlmError, secret: &str) -> LlmError {
    match error {
        LlmError::Http { status, body } => LlmError::Http {
            status,
            body: body.replace(secret, "[redacted]"),
        },
        LlmError::Network(error) => LlmError::Network(error),
        LlmError::Stream(message) => LlmError::Stream(message.replace(secret, "[redacted]")),
        LlmError::Parse(message) => LlmError::Parse(message.replace(secret, "[redacted]")),
        LlmError::Auth(message) => LlmError::Auth(message.replace(secret, "[redacted]")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_codex_rate_limit_windows() {
        let usage = parse_usage_body(
            r#"{
                "plan_type":"plus",
                "rate_limit": {
                    "allowed":true,
                    "limit_reached":false,
                    "primary_window": {
                        "used_percent":23,
                        "limit_window_seconds":604800,
                        "reset_after_seconds":553554,
                        "reset_at":1788204958
                    },
                    "secondary_window":null
                },
                "additional_rate_limits":null
            }"#,
        )
        .unwrap();

        assert_eq!(usage.plan.as_deref(), Some("plus"));
        assert_eq!(usage.windows.len(), 1);
        assert_eq!(usage.windows[0].label, "weekly");
        assert_eq!(usage.windows[0].used_percent, 23);
        assert_eq!(usage.windows[0].resets_after_seconds, Some(553_554));
        assert_eq!(usage.windows[0].resets_at.as_deref(), Some("1788204958"));
    }

    #[test]
    fn parses_primary_secondary_and_additional_windows() {
        let usage = parse_usage_body(
            r#"{
                "plan_type":"pro",
                "rate_limit": {
                    "limit_reached":true,
                    "primary_window":{"used_percent":100,"limit_window_seconds":18000,"reset_after_seconds":60,"reset_at":1000},
                    "secondary_window":{"used_percent":42,"limit_window_seconds":604800,"reset_after_seconds":120,"reset_at":2000}
                },
                "additional_rate_limits":[{
                    "limit_name":"code review",
                    "rate_limit":{"primary_window":{"used_percent":8,"limit_window_seconds":3600,"reset_after_seconds":30,"reset_at":3000}}
                }]
            }"#,
        )
        .unwrap();

        assert_eq!(
            usage
                .windows
                .iter()
                .map(|window| window.label.as_str())
                .collect::<Vec<_>>(),
            ["5 hours", "weekly", "code review · 1 hour"]
        );
        assert_eq!(usage.windows[0].status.as_deref(), Some("limit reached"));
        assert_eq!(usage.windows[2].status, None);
    }

    #[test]
    fn rejects_codex_response_without_windows() {
        assert!(parse_usage_body(r#"{"plan_type":"plus","rate_limit":null}"#).is_err());
    }
}
