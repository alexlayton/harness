use crate::dialects::openai_chat::{ChatReasoningFormat, OpenAiChatClient};
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider};
use reqwest::header::{HeaderMap, HeaderValue};

pub const BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Clone)]
pub struct OpenRouterProvider {
    pub chat: OpenAiChatClient,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert("X-OpenRouter-Title", HeaderValue::from_static("harness"));
        Self {
            chat: OpenAiChatClient::with_headers(BASE_URL, api_key, headers)
                .with_reasoning_format(ChatReasoningFormat::OpenRouter),
        }
    }
}

#[async_trait::async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        self.chat.stream(req).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.chat.list_models().await
    }
}
