use crate::as_u64;
use crate::sse::{SseEvent, stream_response};
use crate::{
    CompletionRequest, Content, EventStream, LlmError, Message, Role, StreamEvent, ToolCall,
    ToolDefinition, Usage,
};
use futures_util::StreamExt;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct AnthropicMessagesClient {
    http: Client,
    pub base_url: String,
    pub api_key: String,
    extra_headers: HeaderMap,
    /// Copilot's backend expects only `Authorization: Bearer`; when set, the
    /// `x-api-key` header is omitted.  Defaults to the dual-header behaviour
    /// used by the Anthropic API and zen-compatible proxies.
    bearer_only: bool,
}

impl AnthropicMessagesClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::with_headers(base_url, api_key, HeaderMap::new())
    }

    pub fn with_headers(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
    ) -> Self {
        Self::internal(base_url, api_key, extra_headers, false)
    }

    /// Construct a client that sends only `Authorization: Bearer <key>` and
    /// omits `x-api-key`.  Required for the GitHub Copilot backend, which
    /// rejects requests carrying both headers.
    pub fn with_bearer_only_headers(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
    ) -> Self {
        Self::internal(base_url, api_key, extra_headers, true)
    }

    fn internal(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
        bearer_only: bool,
    ) -> Self {
        Self {
            http: crate::http::streaming_client(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            extra_headers,
            bearer_only,
        }
    }

    pub async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        tracing::debug!(provider = "anthropic-messages", model = %req.model, "starting stream request");
        let response = self
            .http
            .post(format!("{}/messages", self.base_url))
            .headers(self.headers())
            .json(&build_request_body(req))
            .send()
            .await
            .map_err(LlmError::Network)?;
        let response = crate::http::check_status(response).await?;
        Ok(event_stream(stream_response(response)))
    }

    fn headers(&self) -> HeaderMap {
        let mut headers = self.extra_headers.clone();
        if let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", self.api_key)) {
            headers.insert(AUTHORIZATION, value);
        }
        if !self.bearer_only
            && let Ok(value) = HeaderValue::from_str(&self.api_key)
        {
            headers.insert(HeaderName::from_static("x-api-key"), value);
        }
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static("2023-06-01"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }
}

pub fn build_request_body(req: &CompletionRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), Value::String(req.model.clone()));
    body.insert("max_tokens".into(), json!(req.max_tokens.unwrap_or(32_768)));
    let system = system_text(req);
    if !system.is_empty() {
        body.insert("system".into(), Value::String(system));
    }
    body.insert(
        "messages".into(),
        Value::Array(convert_messages(&req.messages)),
    );
    if !req.tools.is_empty() {
        body.insert("tools".into(), Value::Array(convert_tools(&req.tools)));
    }
    if let Some(temperature) = req.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    // Deliberately do not send the extended thinking parameter.  A zen proxy
    // may still emit thinking blocks for a model that enables them itself.
    body.insert("stream".into(), Value::Bool(true));
    Value::Object(body)
}

fn system_text(req: &CompletionRequest) -> String {
    let mut parts = Vec::new();
    if let Some(system) = &req.system {
        parts.push(system.clone());
    }
    for message in &req.messages {
        if message.role == Role::System {
            let text: String = message
                .content
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if !text.is_empty() {
                parts.push(text);
            }
        }
    }
    parts.join("\n\n")
}

/// Convert internal messages to Anthropic's alternating role/block format.
/// In particular, a run of local Tool messages is represented by one user
/// message, as required by the Messages API.
pub fn convert_messages(messages: &[Message]) -> Vec<Value> {
    let mut result = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for message in messages {
        if message.role == Role::Tool {
            for content in &message.content {
                if let Content::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                } = content
                {
                    pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                        "is_error": is_error,
                    }));
                }
            }
            continue;
        }

        flush_tool_results(&mut result, &mut pending_tool_results);
        match message.role {
            Role::System => {}
            Role::User => {
                let blocks: Vec<Value> = message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(json!({ "type": "text", "text": text })),
                        _ => None,
                    })
                    .collect();
                result.push(json!({ "role": "user", "content": blocks }));
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                for content in &message.content {
                    match content {
                        Content::Text(text) => blocks.push(json!({ "type": "text", "text": text })),
                        Content::ToolCall(call) => blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        })),
                        Content::Reasoning(_) | Content::ToolResult { .. } => {}
                    }
                }
                result.push(json!({ "role": "assistant", "content": blocks }));
            }
            Role::Tool => unreachable!(),
        }
    }
    flush_tool_results(&mut result, &mut pending_tool_results);
    result
}

fn flush_tool_results(result: &mut Vec<Value>, pending: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    result.push(json!({ "role": "user", "content": std::mem::take(pending) }));
}

pub fn convert_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect()
}

#[derive(Debug, Default)]
pub struct AnthropicParser {
    blocks: BTreeMap<u64, BlockKind>,
    tools: BTreeMap<u64, PartialToolUse>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    stop_reason: Option<String>,
    done: bool,
    current_index: Option<u64>,
}

#[derive(Debug)]
enum BlockKind {
    Text,
    Thinking,
    Tool,
}

#[derive(Debug, Default)]
struct PartialToolUse {
    id: String,
    name: String,
    arguments: String,
}

impl AnthropicParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse_event(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, LlmError> {
        self.parse_payload(&event.data)
    }

    pub fn parse_payload(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        let value: Value = serde_json::from_str(payload)
            .map_err(|error| LlmError::Parse(format!("Anthropic SSE payload: {error}")))?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "content_block_start" => {
                let index = value.get("index").and_then(as_u64).unwrap_or(0);
                self.current_index = Some(index);
                let block = value.get("content_block").unwrap_or(&Value::Null);
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        self.blocks.insert(index, BlockKind::Tool);
                        self.tools.insert(
                            index,
                            PartialToolUse {
                                id: block
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .into(),
                                name: block
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .into(),
                                arguments: String::new(),
                            },
                        );
                    }
                    Some("thinking") => {
                        self.blocks.insert(index, BlockKind::Thinking);
                    }
                    _ => {
                        self.blocks.insert(index, BlockKind::Text);
                    }
                }
                Ok(Vec::new())
            }
            "content_block_delta" => {
                let index = value
                    .get("index")
                    .and_then(as_u64)
                    .or(self.current_index)
                    .unwrap_or(0);
                let delta = value.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => Ok(delta
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .map(|value| vec![StreamEvent::TextDelta(value.into())])
                        .unwrap_or_default()),
                    Some("thinking_delta") => Ok(delta
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .map(|value| vec![StreamEvent::ReasoningDelta(value.into())])
                        .unwrap_or_default()),
                    Some("input_json_delta") => {
                        if let Some(tool) = self.tools.get_mut(&index)
                            && let Some(fragment) =
                                delta.get("partial_json").and_then(Value::as_str)
                        {
                            tool.arguments.push_str(fragment);
                        }
                        Ok(Vec::new())
                    }
                    _ => Ok(Vec::new()),
                }
            }
            "content_block_stop" => {
                let index = value
                    .get("index")
                    .and_then(as_u64)
                    .or(self.current_index)
                    .unwrap_or(0);
                self.finish_tool(index)
            }
            "message_start" => {
                self.input_tokens = value
                    .get("message")
                    .and_then(|message| message.get("usage"))
                    .and_then(|usage| usage.get("input_tokens"))
                    .and_then(as_u64);
                Ok(Vec::new())
            }
            "message_delta" => {
                let delta = value.get("delta").unwrap_or(&Value::Null);
                self.stop_reason = delta
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.output_tokens = value
                    .get("usage")
                    .and_then(|usage| usage.get("output_tokens"))
                    .and_then(as_u64)
                    .or(self.output_tokens);
                Ok(Vec::new())
            }
            "message_stop" => {
                let mut output = Vec::new();
                let indices: Vec<u64> = self.tools.keys().copied().collect();
                for index in indices {
                    output.extend(self.finish_tool(index)?);
                }
                self.done = true;
                output.push(StreamEvent::Done {
                    stop_reason: self.stop_reason.clone(),
                    usage: Some(Usage {
                        input_tokens: self.input_tokens.unwrap_or(0),
                        output_tokens: self.output_tokens.unwrap_or(0),
                        cached_tokens: None,
                        reasoning_tokens: None,
                        cost: None,
                    }),
                });
                Ok(output)
            }
            "error" => Err(LlmError::Stream(error_message(&value, "Anthropic error"))),
            "ping" => Ok(Vec::new()),
            _ => {
                tracing::trace!(event_type = kind, "ignored Anthropic SSE event");
                Ok(Vec::new())
            }
        }
    }

    fn finish_tool(&mut self, index: u64) -> Result<Vec<StreamEvent>, LlmError> {
        if !matches!(self.blocks.get(&index), Some(BlockKind::Tool)) {
            return Ok(Vec::new());
        }
        let Some(tool) = self.tools.remove(&index) else {
            return Ok(Vec::new());
        };
        let arguments = if tool.arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&tool.arguments).map_err(|error| {
                LlmError::Parse(format!("invalid Anthropic tool arguments: {error}"))
            })?
        };
        Ok(vec![StreamEvent::ToolCallComplete(ToolCall {
            id: tool.id,
            name: tool.name,
            arguments,
        })])
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Emit a fallback `Done` when the SSE stream ended without a
    /// `message_stop` (e.g. a proxy dropped the connection after the last text
    /// delta). Mirrors `ChatStreamParser::finish` / `ResponsesParser::finish` so
    /// the agent loop always receives a terminal `Done` and the turn's usage is
    /// accounted for rather than dropped.
    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.done {
            return Ok(Vec::new());
        }
        let mut output = Vec::new();
        let indices: Vec<u64> = self.tools.keys().copied().collect();
        for index in indices {
            output.extend(self.finish_tool(index)?);
        }
        self.done = true;
        output.push(StreamEvent::Done {
            stop_reason: self.stop_reason.clone(),
            usage: Some(Usage {
                input_tokens: self.input_tokens.unwrap_or(0),
                output_tokens: self.output_tokens.unwrap_or(0),
                cached_tokens: None,
                reasoning_tokens: None,
                cost: None,
            }),
        });
        Ok(output)
    }
}

fn error_message(value: &Value, fallback: &str) -> String {
    value
        .get("error")
        .and_then(value_message)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .unwrap_or(fallback)
        .to_owned()
}

fn value_message(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("message").and_then(Value::as_str))
}

fn event_stream(mut sse: crate::sse::SseStream) -> EventStream {
    let stream = async_stream::try_stream! {
        let mut parser = AnthropicParser::new();
        while let Some(event) = sse.next().await {
            let event = event?;
            for item in parser.parse_event(&event)? {
                yield item;
            }
        }
        if !parser.is_done() {
            for item in parser.finish()? {
                yield item;
            }
        }
    };
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_only_omits_x_api_key() {
        let default = AnthropicMessagesClient::new("https://api.anthropic.com", "secret");
        let headers = default.headers();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer secret");
        assert_eq!(headers.get("x-api-key").unwrap(), "secret");

        let bearer_only = AnthropicMessagesClient::with_bearer_only_headers(
            "https://api.githubcopilot.com",
            "secret",
            HeaderMap::new(),
        );
        let headers = bearer_only.headers();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer secret");
        assert!(headers.get("x-api-key").is_none());
    }

    #[test]
    fn groups_consecutive_tool_results() {
        let messages = vec![
            Message::user("use bash"),
            Message::assistant(vec![Content::ToolCall(ToolCall {
                id: "a".into(),
                name: "bash".into(),
                arguments: json!({"command":"true"}),
            })]),
            Message::tool_result("a", "one", false),
            Message::tool_result("b", "two", true),
        ];
        let wire = convert_messages(&messages);
        assert_eq!(wire.len(), 3);
        assert_eq!(wire[2]["role"], "user");
        assert_eq!(wire[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(wire[2]["content"][1]["is_error"], true);
    }

    #[test]
    fn parses_text_thinking_tool_and_usage() {
        let mut parser = AnthropicParser::new();
        parser
            .parse_payload(r#"{"type":"message_start","message":{"usage":{"input_tokens":5}}}"#)
            .unwrap();
        parser
            .parse_payload(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
            )
            .unwrap();
        assert_eq!(parser.parse_payload(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#).unwrap(), vec![StreamEvent::TextDelta("hello".into())]);
        parser
            .parse_payload(
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking"}}"#,
            )
            .unwrap();
        assert_eq!(parser.parse_payload(r#"{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#).unwrap(), vec![StreamEvent::ReasoningDelta("hmm".into())]);
        parser.parse_payload(r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t","name":"read","input":{}}}"#).unwrap();
        parser.parse_payload(r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#).unwrap();
        parser.parse_payload(r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"x\"}"}}"#).unwrap();
        let call = parser
            .parse_payload(r#"{"type":"content_block_stop","index":2}"#)
            .unwrap();
        assert!(matches!(&call[0], StreamEvent::ToolCallComplete(call) if call.name == "read"));
        parser.parse_payload(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#).unwrap();
        let done = parser.parse_payload(r#"{"type":"message_stop"}"#).unwrap();
        assert!(matches!(
            &done[0],
            StreamEvent::Done {
                usage: Some(Usage {
                    input_tokens: 5,
                    output_tokens: 7,
                    ..
                }),
                ..
            }
        ));
    }

    #[test]
    fn finish_emits_done_with_usage_when_stream_ends_without_message_stop() {
        let mut parser = AnthropicParser::new();
        parser
            .parse_payload(r#"{"type":"message_start","message":{"usage":{"input_tokens":5}}}"#)
            .unwrap();
        parser
            .parse_payload(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
            )
            .unwrap();
        parser
            .parse_payload(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
            )
            .unwrap();
        parser
            .parse_payload(r#"{"type":"content_block_stop","index":0}"#)
            .unwrap();

        assert!(!parser.is_done());
        let done = parser.finish().unwrap();
        assert_eq!(done.len(), 1);
        assert!(matches!(
            &done[0],
            StreamEvent::Done {
                usage: Some(Usage {
                    input_tokens: 5,
                    output_tokens: 0,
                    ..
                }),
                ..
            }
        ));
        assert!(parser.is_done());
        assert!(parser.finish().unwrap().is_empty());
    }
}
