use crate::dialects::anthropic::AnthropicMessagesClient;
use crate::dialects::openai_chat::OpenAiChatClient;
use crate::dialects::openai_responses::OpenAiResponsesClient;
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider};

pub const BASE_URL: &str = "https://opencode.ai/zen/go/v1";

pub const RESPONSES_MODELS: &[&str] = &["gpt-5.6-luna"];
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
}

impl OpenCodeGoProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            chat: OpenAiChatClient::new(BASE_URL, api_key.clone()),
            responses: OpenAiResponsesClient::new(BASE_URL, api_key.clone()),
            messages: AnthropicMessagesClient::new(BASE_URL, api_key),
        }
    }
}

#[async_trait::async_trait]
impl Provider for OpenCodeGoProvider {
    fn name(&self) -> &str {
        "opencode-go"
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        match dialect_for_model(&req.model) {
            Dialect::Responses => self.responses.stream(req).await,
            Dialect::Messages => self.messages.stream(req).await,
            Dialect::Chat => self.chat.stream(req).await,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.chat.list_models().await
    }
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
}
