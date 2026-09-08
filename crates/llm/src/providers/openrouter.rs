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
        self.chat
            .list_models()
            .await
            .map_err(|error| error.redacted(self.chat.api_key()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ReasoningPolicy};
    use futures_util::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn provider_sse_errors_redact_the_active_key() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture listener");
        let address = listener.local_addr().expect("fixture address");
        let secret = "openrouter-stream-sentinel";
        let body = format!("data: {{\"error\":{{\"message\":\"upstream echoed {secret}\"}}}}\n\n");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept fixture request");
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write fixture response");
        });

        let provider = OpenRouterProvider {
            chat: OpenAiChatClient::new(format!("http://{address}"), secret),
        };
        let request = CompletionRequest {
            model: "fixture-model".into(),
            system: None,
            messages: vec![Message::user("hello")],
            tools: Vec::new(),
            max_tokens: None,
            temperature: None,
            reasoning: ReasoningPolicy::Auto,
            session_id: None,
        };
        let mut stream = provider
            .stream(&request)
            .await
            .expect("stream construction");
        let error = stream
            .next()
            .await
            .expect("fixture should yield an error")
            .expect_err("provider error payload must not succeed");
        let rendered = error.to_string();
        assert!(matches!(error, LlmError::Stream(_)));
        assert!(!rendered.contains(secret), "secret leaked in {rendered}");
        assert!(
            rendered.contains("[redacted]"),
            "redaction missing in {rendered}"
        );
    }
}
