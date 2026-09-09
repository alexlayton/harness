use crate::as_u64;
use crate::dialects::openai_reasoning_effort;
use crate::http::HttpClient;
use crate::sse::{SseEvent, stream_response};
use crate::{
    CompletionRequest, Content, EventStream, LlmError, Message, ModelInfo, ReasoningPolicy, Role,
    StreamEvent, ToolCall, ToolDefinition, Usage,
};
use reqwest::header::HeaderMap;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashSet};

/// Provider-specific reasoning extension for OpenAI-compatible Chat APIs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatReasoningFormat {
    /// Do not speculate about extensions on a generic compatible endpoint.
    #[default]
    None,
    /// OpenRouter's unified `reasoning` object.
    OpenRouter,
}

#[derive(Clone)]
pub struct OpenAiChatClient {
    http: HttpClient,
    reasoning_format: ChatReasoningFormat,
}

impl OpenAiChatClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: HttpClient::new(base_url, api_key),
            reasoning_format: ChatReasoningFormat::None,
        }
    }

    pub fn with_headers(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
    ) -> Self {
        Self {
            http: HttpClient::with_headers(base_url, api_key, extra_headers),
            reasoning_format: ChatReasoningFormat::None,
        }
    }

    /// Enable a documented provider extension without affecting generic Chat
    /// endpoints that may reject unknown fields.
    pub fn with_reasoning_format(mut self, format: ChatReasoningFormat) -> Self {
        self.reasoning_format = format;
        self
    }

    pub async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        tracing::debug!(provider = "openai-chat", model = %req.model, "starting stream request");
        let response = self
            .http
            .post_json(
                "/chat/completions",
                &build_request_body_with_reasoning(req, self.reasoning_format),
            )
            .await?;
        Ok(event_stream(stream_response(response), &self.http.api_key))
    }

    pub(crate) fn api_key(&self) -> &str {
        &self.http.api_key
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let response = self.http.get("/models").await?;
        let body = response.text().await.map_err(LlmError::Network)?;
        parse_models_body(&body).map_err(|error| error.redacted(self.api_key()))
    }
}

/// Build an OpenAI-compatible chat request without making a network call.
pub fn build_request_body(req: &CompletionRequest) -> Value {
    build_request_body_with_reasoning(req, ChatReasoningFormat::None)
}

/// Build a request with a provider's documented reasoning extension.
pub fn build_request_body_with_reasoning(
    req: &CompletionRequest,
    reasoning_format: ChatReasoningFormat,
) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), Value::String(req.model.clone()));

    let mut messages = Vec::new();
    if let Some(system) = &req.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.extend(convert_messages(&req.messages));
    body.insert("messages".into(), Value::Array(messages));

    if !req.tools.is_empty() {
        body.insert("tools".into(), Value::Array(convert_tools(&req.tools)));
        body.insert("tool_choice".into(), Value::String("auto".into()));
    }
    if let Some(max_tokens) = req.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = req.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if reasoning_format == ChatReasoningFormat::OpenRouter {
        match req.reasoning {
            ReasoningPolicy::Off => {
                body.insert("reasoning".into(), json!({ "enabled": false }));
            }
            // Omission preserves the model/provider default and matches the
            // pre-setting OpenRouter request shape.
            ReasoningPolicy::Auto => {}
            ReasoningPolicy::Effort(_) => {
                body.insert(
                    "reasoning".into(),
                    json!({ "effort": openai_reasoning_effort(req.reasoning) }),
                );
            }
        }
    }
    body.insert("stream".into(), Value::Bool(true));
    body.insert("stream_options".into(), json!({ "include_usage": true }));
    Value::Object(body)
}

pub fn convert_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect()
}

pub fn convert_messages(messages: &[Message]) -> Vec<Value> {
    let mut wire = Vec::new();
    for message in messages {
        match message.role {
            Role::System => {
                let text = text_content(&message.content);
                if !text.is_empty() {
                    wire.push(json!({ "role": "system", "content": text }));
                }
            }
            Role::User => {
                let text = text_content(&message.content);
                wire.push(json!({ "role": "user", "content": text }));
            }
            Role::Assistant => {
                let mut text = String::new();
                let mut calls = Vec::new();
                for item in &message.content {
                    match item {
                        Content::Text(value) => text.push_str(value),
                        Content::ToolCall(call) => calls.push(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": stringify_arguments(&call.arguments),
                            }
                        })),
                        Content::Reasoning(_)
                        | Content::Opaque { .. }
                        | Content::ToolResult { .. } => {}
                    }
                }
                let mut value = Map::new();
                value.insert("role".into(), Value::String("assistant".into()));
                value.insert(
                    "content".into(),
                    if text.is_empty() {
                        Value::Null
                    } else {
                        Value::String(text)
                    },
                );
                if !calls.is_empty() {
                    value.insert("tool_calls".into(), Value::Array(calls));
                }
                wire.push(Value::Object(value));
            }
            Role::Tool => {
                for item in &message.content {
                    if let Content::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                    } = item
                    {
                        let content = if *is_error {
                            format!("Error: {content}")
                        } else {
                            content.clone()
                        };
                        wire.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content,
                        }));
                    }
                }
            }
        }
    }
    wire
}

fn text_content(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|item| match item {
            Content::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn stringify_arguments(arguments: &Value) -> String {
    serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into())
}

pub fn parse_models_body(body: &str) -> Result<Vec<ModelInfo>, LlmError> {
    let value: Value =
        serde_json::from_str(body).map_err(|error| LlmError::Parse(error.to_string()))?;
    let items = value
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| value.as_array())
        .ok_or_else(|| LlmError::Parse("models response has no data array".into()))?;
    Ok(items
        .iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?.to_owned();
            Some(ModelInfo {
                id,
                name: item.get("name").and_then(Value::as_str).map(str::to_owned),
                context_length: item.get("context_length").and_then(as_u64),
            })
        })
        .collect())
}

#[derive(Debug, Default)]
pub struct ChatStreamParser {
    calls: BTreeMap<u64, PartialToolCall>,
    stop_reason: Option<String>,
    done: bool,
    calls_flushed: bool,
    /// Call IDs already emitted in this response.  IDs are unique per
    /// assistant response; a repeated ID is a provider error, not a second
    /// call, and surfaces as `LlmError::Parse` before execution.
    seen_ids: HashSet<String>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChatStreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse_event(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, LlmError> {
        self.parse_payload(&event.data)
    }

    pub fn parse_payload(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        // A terminal payload is sometimes followed by a transport-level
        // duplicate (or is passed to the parser directly after a usage
        // terminal).  Do not allow either case to produce another terminal
        // event or process data after the turn has ended.
        if self.done {
            return Ok(Vec::new());
        }
        if payload.trim() == "[DONE]" {
            // The documented Chat terminator has no usage of its own.  It is
            // nevertheless a complete protocol terminal: validate and emit
            // every pending call before recording one usage-less Done event.
            return self.complete(None);
        }
        let value: Value = serde_json::from_str(payload)
            .map_err(|error| LlmError::Parse(format!("chat SSE payload: {error}")))?;
        let mut output = Vec::new();

        if let Some(error_value) = value.get("error")
            && !error_value.is_null()
        {
            // A mid-stream provider error payload (e.g. `{"error": ...}`)
            // is a terminal failure, not a silent empty chunk: surface it
            // as `LlmError::Stream` so partial output stays an error and
            // the agent's recovery path handles it.
            let detail = error_value
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error_value.as_str())
                .unwrap_or("provider error");
            return Err(LlmError::Stream(format!("chat error: {detail}")));
        }

        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
            for (choice_index, choice) in choices.iter().enumerate() {
                if choice_index > 0 {
                    break; // v1 requests one completion
                }
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        output.push(StreamEvent::TextDelta(text.to_owned()));
                    }
                    if let Some(reasoning) = delta.get("reasoning").and_then(Value::as_str)
                        && !reasoning.is_empty()
                    {
                        output.push(StreamEvent::ReasoningDelta(reasoning.to_owned()));
                    }
                    if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str)
                        && !reasoning.is_empty()
                    {
                        output.push(StreamEvent::ReasoningDelta(reasoning.to_owned()));
                    }
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                        for (fallback_index, item) in tool_calls.iter().enumerate() {
                            self.accumulate_tool_call(item, fallback_index as u64);
                        }
                    }
                }
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(reason.to_owned());
                    output.extend(self.flush_calls()?);
                }
            }
        }

        if let Some(usage_value) = value.get("usage")
            && !usage_value.is_null()
        {
            output.extend(self.complete(Some(parse_usage(usage_value)?))?);
        }
        Ok(output)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.done {
            return Ok(Vec::new());
        }
        // Clean transport EOF without a protocol terminal event is a
        // truncated stream, not a successful turn: surface it as
        // `LlmError::Stream` so the agent's recovery path (persist partial
        // text, emit a diagnostic, retry or fail loudly) handles it.  Never
        // manufacture `Done` and never finalize unfinished tool calls here.
        self.done = true;
        Err(LlmError::Stream(
            "chat stream ended without a terminal event (expected [DONE], a finish_reason, or a usage chunk)"
                .into(),
        ))
    }

    fn complete(&mut self, usage: Option<Usage>) -> Result<Vec<StreamEvent>, LlmError> {
        let mut output = self.flush_calls()?;
        output.push(StreamEvent::Done {
            stop_reason: self.stop_reason.clone(),
            usage,
        });
        self.done = true;
        Ok(output)
    }

    fn accumulate_tool_call(&mut self, item: &Value, fallback_index: u64) {
        let index = item.get("index").and_then(as_u64).unwrap_or(fallback_index);
        let call = self.calls.entry(index).or_default();
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            // IDs normally arrive whole in the first delta; append when a
            // fragment continues a partial ID so split transport chunks
            // still assemble, while repeated identical IDs stay idempotent.
            if call.id.is_empty() {
                call.id = id.to_owned();
            } else if id != call.id && !call.id.ends_with(id) {
                call.id.push_str(id);
            }
        }
        if let Some(function) = item.get("function") {
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                call.name.push_str(name);
            }
            if let Some(arguments) = function.get("arguments") {
                if let Some(arguments) = arguments.as_str() {
                    call.arguments.push_str(arguments);
                } else if !arguments.is_null() {
                    call.arguments.push_str(&arguments.to_string());
                }
            }
        }
    }

    fn flush_calls(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.calls_flushed && self.calls.is_empty() {
            return Ok(Vec::new());
        }
        // Validate every call before emitting any: a malformed call fails
        // the response with `LlmError::Parse` (handled by the agent's
        // malformed-tool recovery) and no `ToolCallComplete` is emitted.
        // Keep the pending map intact until validation succeeds so a failed
        // terminal cannot turn the malformed call into an apparent success
        // if the parser is inspected again.
        let mut validated = Vec::with_capacity(self.calls.len());
        for call in self.calls.values() {
            let id = call.id.trim();
            if id.is_empty() {
                return Err(LlmError::Parse(
                    "chat tool call is missing a call ID".into(),
                ));
            }
            let name = call.name.trim();
            if name.is_empty() {
                return Err(LlmError::Parse(format!(
                    "chat tool call {id} is missing a name"
                )));
            }
            if self.seen_ids.contains(id) || validated.iter().any(|(seen, _)| seen == id) {
                return Err(LlmError::Parse(format!("duplicate chat tool call ID {id}")));
            }
            let arguments = if call.arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&call.arguments).map_err(|error| {
                    LlmError::Parse(format!("invalid tool arguments for {name}: {error}"))
                })?
            };
            validated.push((
                id.to_owned(),
                ToolCall {
                    id: id.to_owned(),
                    name: name.to_owned(),
                    arguments,
                },
            ));
        }
        self.calls_flushed = true;
        self.calls.clear();
        let mut result = Vec::with_capacity(validated.len());
        for (id, call) in validated {
            self.seen_ids.insert(id);
            result.push(StreamEvent::ToolCallComplete(call));
        }
        Ok(result)
    }
}

fn parse_usage(value: &Value) -> Result<Usage, LlmError> {
    Ok(Usage {
        input_tokens: value.get("prompt_tokens").and_then(as_u64).unwrap_or(0),
        output_tokens: value.get("completion_tokens").and_then(as_u64).unwrap_or(0),
        cached_tokens: value
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(as_u64),
        reasoning_tokens: value
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(as_u64),
        cost: value.get("cost").and_then(Value::as_f64),
    })
}

impl super::StreamParser for ChatStreamParser {
    fn parse_event(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, LlmError> {
        Self::parse_event(self, event)
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        Self::finish(self)
    }
}

fn event_stream(sse: crate::sse::SseStream, secret: &str) -> EventStream {
    super::drive_parser_stream(sse, ChatStreamParser::new(), secret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> CompletionRequest {
        CompletionRequest {
            model: "demo".into(),
            system: Some("be concise".into()),
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![Content::Text("hello".into())],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![
                        Content::Text("ok".into()),
                        Content::Reasoning("hidden".into()),
                        Content::ToolCall(ToolCall {
                            id: "c1".into(),
                            name: "read".into(),
                            arguments: json!({"path":"x"}),
                        }),
                    ],
                },
                Message::tool_result("c1", "result", true),
            ],
            tools: vec![ToolDefinition {
                name: "read".into(),
                description: "read a file".into(),
                parameters: json!({"type":"object"}),
            }],
            max_tokens: None,
            temperature: None,
            reasoning: ReasoningPolicy::Auto,
            session_id: None,
        }
    }

    #[test]
    fn openrouter_serializes_reasoning_without_affecting_generic_chat() {
        let mut req = request();
        req.reasoning = ReasoningPolicy::Off;
        let generic = build_request_body(&req);
        assert!(generic.get("reasoning").is_none());

        let off = build_request_body_with_reasoning(&req, ChatReasoningFormat::OpenRouter);
        assert_eq!(off["reasoning"]["enabled"], false);

        req.reasoning = ReasoningPolicy::Auto;
        let auto = build_request_body_with_reasoning(&req, ChatReasoningFormat::OpenRouter);
        assert!(auto.get("reasoning").is_none());

        req.reasoning = ReasoningPolicy::Effort(crate::ReasoningEffort::High);
        let high = build_request_body_with_reasoning(&req, ChatReasoningFormat::OpenRouter);
        assert_eq!(high["reasoning"]["effort"], "high");
    }

    #[test]
    fn converts_all_message_content() {
        let messages = convert_messages(&request().messages);
        assert_eq!(messages[0]["content"], "hello");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(messages[2]["content"], "Error: result");
        assert!(messages[1]["content"] == "ok");
    }

    #[test]
    fn parses_interleaved_text_reasoning_and_parallel_calls() {
        let mut parser = ChatStreamParser::new();
        let mut events = parser
            .parse_payload(r#"{"choices":[{"delta":{"content":"hi","reasoning":"think"}}]}"#)
            .unwrap();
        events.extend(parser.parse_payload(r#"{"choices":[{"delta":{"reasoning_content":" more","tool_calls":[{"index":1,"id":"b","function":{"name":"bash","arguments":"{\"command\":"}}]}}]}"#).unwrap());
        events.extend(parser.parse_payload(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{\"path\":\"x\"}"}},{"index":1,"function":{"arguments":"\"echo\"}"}}]},"finish_reason":"tool_calls"}]}"#).unwrap());
        events.extend(parser.parse_payload(r#"{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":1}}}"#).unwrap());
        assert!(events.contains(&StreamEvent::TextDelta("hi".into())));
        assert!(events.contains(&StreamEvent::ReasoningDelta("think".into())));
        assert!(events.contains(&StreamEvent::ReasoningDelta(" more".into())));
        assert!(
            events.iter().any(
                |event| matches!(event, StreamEvent::ToolCallComplete(call) if call.id == "a")
            )
        );
        assert!(
            events.iter().any(
                |event| matches!(event, StreamEvent::ToolCallComplete(call) if call.id == "b")
            )
        );
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done { usage: Some(_), .. })
        ));
    }

    #[test]
    fn terminal_usage_chunk_split_across_chunks_succeeds() {
        // The terminal usage chunk split across two `push_bytes` calls
        // still terminates with `Done`: the decoder buffers the partial
        // line and only `process_line`s on `\n`. Split *between* SSE
        // lines: the first chunk holds a complete data line, the second
        // the blank dispatch line.
        use crate::sse::SseParser;
        let mut sse = SseParser::new();
        let first = r#"{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#;
        let payload = format!("data: {first}\n");
        assert!(sse.push_bytes(payload.as_bytes()).unwrap().is_empty());
        let events = sse.push_bytes(b"\n").unwrap();
        assert_eq!(events.len(), 1);
        let mut parser = ChatStreamParser::new();
        let done = parser.parse_event(&events[0]).unwrap();
        assert!(matches!(&done[0], StreamEvent::Done { .. }), "got {done:?}");
        assert!(parser.done);
        assert!(parser.parse_payload("[DONE]").unwrap().is_empty());
    }

    #[test]
    fn valid_text_only_done_terminator_emits_one_terminal_event() {
        let mut parser = ChatStreamParser::new();
        let mut events = parser
            .parse_payload(r#"{"choices":[{"delta":{"content":"hi"}}]}"#)
            .unwrap();
        events.extend(parser.parse_payload("[DONE]").unwrap());
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("hi".into()),
                StreamEvent::Done {
                    stop_reason: None,
                    usage: None,
                },
            ]
        );
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn text_then_eof_without_terminator_fails() {
        // Text followed by clean EOF with no `[DONE]`, finish_reason, or
        // usage chunk is a truncated stream: `finish` is a `Stream` error.
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(r#"{"choices":[{"delta":{"content":"hi"}}]}"#)
            .unwrap();
        let error = parser.finish().unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
    }

    #[test]
    fn partial_tool_call_then_eof_fails_and_emits_no_completed_call() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{\"path\":"}}]}}]}"#,
            )
            .unwrap();
        let error = parser.finish().unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
        assert!(parser.seen_ids.is_empty());
    }

    #[test]
    fn rejects_missing_id() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read","arguments":"{}"}}]}}]}"#,
            )
            .unwrap();
        let error = parser.parse_payload("[DONE]").unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
        assert!(!parser.done);
        assert!(parser.seen_ids.is_empty());
    }

    #[test]
    fn rejects_missing_name() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"arguments":"{}"}}]}}]}"#,
            )
            .unwrap();
        let error = parser.parse_payload("[DONE]").unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
        assert!(!parser.done);
    }

    #[test]
    fn rejects_partial_json_arguments_on_done() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{\"path\":"}}]}}]}"#,
            )
            .unwrap();
        let error = parser.parse_payload("[DONE]").unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
        assert!(!parser.done);
        assert!(parser.seen_ids.is_empty());
    }

    #[test]
    fn valid_tool_call_followed_directly_by_done_flushes_once() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"read","arguments":"{\"path\":\"x\"}"}}]}}]}"#,
            )
            .unwrap();
        let events = parser.parse_payload("[DONE]").unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::ToolCallComplete(ToolCall {
                    id: "call-1".into(),
                    name: "read".into(),
                    arguments: json!({"path": "x"}),
                }),
                StreamEvent::Done {
                    stop_reason: None,
                    usage: None,
                },
            ]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Done { .. }))
                .count(),
            1
        );
        assert!(parser.parse_payload("[DONE]").unwrap().is_empty());
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn rejects_blank_tool_call_id_and_name_on_done() {
        for (id, name) in [("  ", "read"), ("call-1", "   ")] {
            let mut parser = ChatStreamParser::new();
            let payload = format!(
                r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"{id}","function":{{"name":"{name}","arguments":"{{}}"}}}}]}}}}]}}"#
            );
            parser.parse_payload(&payload).unwrap();
            let error = parser.parse_payload("[DONE]").unwrap_err();
            assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
        }
    }

    #[test]
    fn rejects_duplicate_ids_in_parallel_calls() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"dup","function":{"name":"read","arguments":"{}"}},{"index":1,"id":"dup","function":{"name":"bash","arguments":"{}"}}]}}]}"#,
            )
            .unwrap();
        let error = parser
            .parse_payload(r#"{"choices":[{"finish_reason":"tool_calls"}]}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
    }

    #[test]
    fn fragmented_valid_ids_and_names_assemble() {
        let mut parser = ChatStreamParser::new();
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"ca","function":{"name":"re"}}]}}]}"#,
            )
            .unwrap();
        // A continued fragment appends to the partial ID/name.
        parser
            .parse_payload(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"ll-1","function":{"name":"ad","arguments":"{}"}}]}}]}"#,
            )
            .unwrap();
        let mut events = Vec::new();
        events.extend(
            parser
                .parse_payload(r#"{"choices":[{"finish_reason":"tool_calls"}]}"#)
                .unwrap(),
        );
        assert!(
            events.iter().any(|event| matches!(event, StreamEvent::ToolCallComplete(call) if call.id == "call-1" && call.name == "read")),
            "got {events:?}"
        );
    }

    #[test]
    fn stream_error_after_partial_output_remains_an_error() {
        // Partial text followed by a mid-stream provider error payload is
        // a failure, not a truncated-but-usable turn: the error surfaces
        // as `LlmError::Stream` (agent recovery path) instead of an empty
        // `Ok` that a later EOF would turn into a confusing truncation.
        let mut parser = ChatStreamParser::new();
        let events = parser
            .parse_payload(r#"{"choices":[{"delta":{"content":"hi"}}]}"#)
            .unwrap();
        assert_eq!(events, vec![StreamEvent::TextDelta("hi".into())]);
        let error = parser
            .parse_payload(r#"{"error":{"message":"boom","type":"server_error"}}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
    }

    #[test]
    fn parser_errors_are_redacted_only_at_the_stream_boundary() {
        let secret = "parser-stream-sentinel";
        let mut parser = ChatStreamParser::new();
        let error = parser
            .parse_payload(&format!(
                r#"{{"error":{{"message":"upstream echoed {secret}"}}}}"#
            ))
            .unwrap_err();
        // The wire parser retains the provider's diagnostic; the shared
        // stream adapter is the security boundary for returned EventStreams.
        assert!(error.to_string().contains(secret));
        let rendered = error.redacted(secret).to_string();
        assert!(!rendered.contains(secret));
        assert!(rendered.contains("[redacted]"));
    }
}
