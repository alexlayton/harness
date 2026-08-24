//! ChatGPT Codex subscription provider (SSE transport only).
use crate::dialects::openai_codex_responses::OpenAiCodexResponsesClient;
use crate::{CompletionRequest, EventStream, LlmError, ModelInfo, Provider};
use auth::OpenAiCodexAuth;
use reqwest::header::{ACCEPT, HeaderMap, HeaderName, HeaderValue};
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
