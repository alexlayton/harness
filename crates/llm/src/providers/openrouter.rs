use crate::dialects::openai_chat::OpenAiChatClient;
use crate::retry::with_retry;
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider, RetryCallback};
use reqwest::header::{HeaderMap, HeaderValue};
use std::sync::Arc;

pub const BASE_URL: &str = "https://openrouter.ai/api/v1";

pub fn parse_models_response(body: &str) -> Result<Vec<ModelInfo>, LlmError> {
    crate::dialects::openai_chat::parse_models_body(body)
}

#[derive(Clone)]
pub struct OpenRouterProvider {
    pub chat: OpenAiChatClient,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert("X-OpenRouter-Title", HeaderValue::from_static("harness"));
        Self {
            chat: OpenAiChatClient::with_headers(BASE_URL, api_key, headers),
        }
    }

    async fn stream_once(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        self.chat.stream(req).await
    }

    pub async fn stream_with_callback(
        &self,
        req: &CompletionRequest,
        on_retry: RetryCallback,
    ) -> Result<EventStream, LlmError> {
        let callback = on_retry.clone();
        with_retry(
            || async { self.stream_once(req).await },
            move |attempt, error| {
                tracing::warn!(attempt, error = %error, "retrying OpenRouter request");
                callback(attempt, error);
            },
        )
        .await
    }
}

#[async_trait::async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        self.stream_with_callback(req, Arc::new(|_, _| {})).await
    }

    async fn stream_with_retry(
        &self,
        req: &CompletionRequest,
        on_retry: RetryCallback,
    ) -> Result<EventStream, LlmError> {
        self.stream_with_callback(req, on_retry).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.chat.list_models().await
    }
}
