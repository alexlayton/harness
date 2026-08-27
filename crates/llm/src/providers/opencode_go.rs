use crate::dialects::anthropic::AnthropicMessagesClient;
use crate::dialects::openai_chat::OpenAiChatClient;
use crate::dialects::openai_responses::OpenAiResponsesClient;
use crate::http::HttpClient;
use crate::{
    CompletionRequest, EventStream, LlmError, ModelInfo, Provider, ReasoningPolicy,
    SubscriptionUsage, SubscriptionUsageWindow,
};
use serde::Deserialize;

pub const BASE_URL: &str = "https://opencode.ai/zen/go/v1";

pub const RESPONSES_MODELS: &[&str] = &["gpt-5.6-luna", "grok-4.6"];
pub const MESSAGES_MODELS: &[&str] = &[
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.8-max",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
];
pub const CHAT_MODELS: &[&str] = &[
    "grok-4.5",
    "glm-5.2",
    "glm-5.1",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "hy3",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Responses,
    Messages,
    Chat,
}

pub fn dialect_for_model(model: &str) -> Dialect {
    if RESPONSES_MODELS.contains(&model) {
        Dialect::Responses
    } else if MESSAGES_MODELS.contains(&model) {
        Dialect::Messages
    } else {
        // Unknown/new Go models are most likely added to the OpenAI-compatible
        // endpoint, so Chat is the safe forward-compatible fallback.
        Dialect::Chat
    }
}

#[derive(Clone)]
pub struct OpenCodeGoProvider {
    pub chat: OpenAiChatClient,
    pub responses: OpenAiResponsesClient,
    pub messages: AnthropicMessagesClient,
    usage: HttpClient,
}

impl OpenCodeGoProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            chat: OpenAiChatClient::new(BASE_URL, api_key.clone()),
            responses: OpenAiResponsesClient::new(BASE_URL, api_key.clone()),
            messages: AnthropicMessagesClient::new(BASE_URL, api_key.clone()),
            usage: HttpClient::new(BASE_URL, api_key),
        }
    }
}

#[async_trait::async_trait]
impl Provider for OpenCodeGoProvider {
    fn name(&self) -> &str {
        "opencode-go"
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        let dialect = dialect_for_model(&req.model);
        if matches!(req.reasoning, ReasoningPolicy::Effort(_)) && dialect != Dialect::Responses {
            return Err(LlmError::Parse(format!(
                "OpenCode Go model `{}` does not expose portable reasoning effort on its {:?} endpoint; use `auto` or `off`",
                req.model, dialect
            )));
        }
        match dialect {
            Dialect::Responses => self.responses.stream(req).await,
            Dialect::Messages => self.messages.stream(req).await,
            Dialect::Chat => self.chat.stream(req).await,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.chat.list_models().await
    }

    async fn subscription_usage(&self) -> Result<Option<SubscriptionUsage>, LlmError> {
        let response = self.usage.get("/usage").await?;
        let body = response.text().await.map_err(LlmError::Network)?;
        parse_usage_body(&body).map(Some)
    }
}

#[derive(Debug, Deserialize)]
struct UsageResponse {
    usage: UsageWindows,
}

#[derive(Debug, Deserialize)]
struct UsageWindows {
    rolling: UsageWindow,
    weekly: UsageWindow,
    monthly: UsageWindow,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageWindow {
    status: String,
    percent: u16,
    resets_at: Option<String>,
}

fn parse_usage_body(body: &str) -> Result<SubscriptionUsage, LlmError> {
    let response: UsageResponse = serde_json::from_str(body)
        .map_err(|error| LlmError::Parse(format!("invalid OpenCode Go usage response: {error}")))?;
    let UsageWindows {
        rolling,
        weekly,
        monthly,
    } = response.usage;
    Ok(SubscriptionUsage {
        plan: Some("Go".into()),
        windows: [
            ("rolling", rolling),
            ("weekly", weekly),
            ("monthly", monthly),
        ]
        .into_iter()
        .map(|(label, window)| SubscriptionUsageWindow {
            label: label.into(),
            used_percent: window.percent,
            status: Some(window.status),
            resets_at: window.resets_at,
            resets_after_seconds: None,
        })
        .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_every_documented_model_and_falls_back() {
        for model in RESPONSES_MODELS {
            assert_eq!(dialect_for_model(model), Dialect::Responses);
        }
        for model in MESSAGES_MODELS {
            assert_eq!(dialect_for_model(model), Dialect::Messages);
        }
        for model in CHAT_MODELS {
            assert_eq!(dialect_for_model(model), Dialect::Chat);
        }
        assert_eq!(dialect_for_model("new-model"), Dialect::Chat);
    }

    #[test]
    fn parses_all_subscription_windows() {
        let usage = parse_usage_body(
            r#"{
                "usage": {
                    "rolling": {"status":"ok","percent":0,"resetsAt":"2026-08-25T14:48:50.201Z"},
                    "weekly": {"status":"ok","percent":12,"resetsAt":"2026-08-31T00:00:00.201Z"},
                    "monthly": {"status":"limited","percent":54,"resetsAt":"2026-09-14T10:36:14.201Z"}
                }
            }"#,
        )
        .unwrap();

        assert_eq!(usage.plan.as_deref(), Some("Go"));
        assert_eq!(usage.windows.len(), 3);
        assert_eq!(usage.windows[0].label, "rolling");
        assert_eq!(usage.windows[1].used_percent, 12);
        assert_eq!(usage.windows[2].status.as_deref(), Some("limited"));
        assert_eq!(
            usage.windows[2].resets_at.as_deref(),
            Some("2026-09-14T10:36:14.201Z")
        );
    }

    #[test]
    fn rejects_incomplete_subscription_usage() {
        let error = parse_usage_body(r#"{"usage":{"rolling":{}}}"#).unwrap_err();
        assert!(error.to_string().contains("OpenCode Go usage response"));
    }
}
