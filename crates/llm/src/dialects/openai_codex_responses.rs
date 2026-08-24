//! Wire translation for ChatGPT's Codex Responses endpoint.
//!
//! This is intentionally separate from the public OpenAI Responses API: the
//! subscription service uses a different endpoint and requires Codex-specific
//! request fields even though it streams familiar Responses SSE events.
use crate::dialects::openai_responses::{
    ResponsesParser, convert_input as base_convert_input, convert_tools,
};
use crate::http::HttpClient;
use crate::sse::stream_response;
use crate::{CompletionRequest, EventStream, LlmError};
use futures_util::StreamExt;
use reqwest::header::HeaderMap;
use serde_json::{Value, json};

#[derive(Clone)]
pub struct OpenAiCodexResponsesClient {
    http: HttpClient,
}
impl OpenAiCodexResponsesClient {
    pub fn with_headers(
        base_url: impl Into<String>,
        access_token: impl Into<String>,
        headers: HeaderMap,
    ) -> Self {
        Self {
            http: HttpClient::with_headers(base_url, access_token, headers),
        }
    }
    pub async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
        let response = self
            .http
            .post_json("/responses", &build_request_body(request))
            .await?;
        Ok(event_stream(stream_response(response)))
    }
}
/// Convert neutral harness history to the subset accepted by Codex.
pub fn build_request_body(request: &CompletionRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "input": convert_input(&request.messages),
        "tools": convert_tools(&request.tools),
        "parallel_tool_calls": true,
        "store": false,
        "stream": true,
        "reasoning": { "effort": "medium", "summary": "auto" },
        "include": ["reasoning.encrypted_content"],
    });
    if let Some(system) = &request.system {
        body["instructions"] = Value::String(system.clone());
    }
    if let Some(max) = request.max_tokens {
        body["max_output_tokens"] = json!(max);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    body
}

/// Replay only Codex-owned opaque items. Other provider metadata is ignored
/// rather than leaking a foreign wire format into this endpoint.
pub fn convert_input(messages: &[crate::Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        // Convert one neutral message at a time so opaque response items stay
        // adjacent to the assistant turn which produced them.
        input.extend(base_convert_input(std::slice::from_ref(message)));
        for content in &message.content {
            if let crate::Content::Opaque { provider, data } = content
                && provider == "openai-codex"
            {
                input.push(data.clone());
            }
        }
    }
    input
}
fn event_stream(mut sse: crate::sse::SseStream) -> EventStream {
    Box::pin(async_stream::try_stream! {
        let mut parser = ResponsesParser::new();
        while let Some(event) = sse.next().await {
            let event = event?;
            // Codex sends encrypted reasoning on completed output items. Keep
            // the complete item opaque so a later tool follow-up can replay it.
            if let Ok(value) = serde_json::from_str::<Value>(&event.data)
                && value.get("type").and_then(Value::as_str) == Some("response.output_item.done")
                && value.get("item").and_then(|item| item.get("encrypted_content")).is_some()
            {
                yield crate::StreamEvent::OpaqueState { provider: "openai-codex".into(), data: value["item"].clone() };
            }
            for value in parser.parse_event(&event)? { yield value; }
        }
        if !parser.is_done() { for value in parser.finish()? { yield value; } }
    })
}
