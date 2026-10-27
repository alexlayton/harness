//! User-defined endpoints that speak the OpenAI Chat Completions API, such as
//! LM Studio, Ollama, vLLM, or a self-hosted gateway.

use crate::dialects::openai_chat::OpenAiChatClient;
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider};
use reqwest::header::HeaderMap;

/// A named OpenAI-compatible endpoint configured by the user.
///
/// The endpoint is generic, so no provider-specific request extensions are
/// sent; the plain Chat Completions shape is the only shared contract.
#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    name: String,
    chat: OpenAiChatClient,
    context_window: Option<u64>,
}

impl OpenAiCompatibleProvider {
    /// Build a provider for `base_url`. An empty `api_key` sends no
    /// `Authorization` header, which local servers commonly expect.
    /// `context_window` is the planning fallback for models whose `/models`
    /// entry omits a context length (or when that request fails), because
    /// local servers rarely report one.
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        headers: HeaderMap,
        context_window: Option<u64>,
    ) -> Self {
        Self {
            name: name.into(),
            chat: OpenAiChatClient::with_headers(base_url, api_key, headers),
            context_window,
        }
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        self.chat.stream(req).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.chat
            .list_models()
            .await
            .map_err(|error| error.redacted(self.chat.api_key()))
    }

    fn context_window_hint(&self) -> Option<u64> {
        self.context_window
    }
}
