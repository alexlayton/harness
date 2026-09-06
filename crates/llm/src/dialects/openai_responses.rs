use crate::as_u64;
use crate::dialects::openai_reasoning_effort;
use crate::http::HttpClient;
use crate::sse::{SseEvent, stream_response};
use crate::{
    CompletionRequest, Content, EventStream, LlmError, Message, ReasoningPolicy, Role, StreamEvent,
    ToolCall, ToolDefinition, Usage,
};
use reqwest::header::HeaderMap;
use serde_json::{Map, Value, json};

#[derive(Clone)]
pub struct OpenAiResponsesClient {
    http: HttpClient,
}

impl OpenAiResponsesClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: HttpClient::new(base_url, api_key),
        }
    }

    pub fn with_headers(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
    ) -> Self {
        Self {
            http: HttpClient::with_headers(base_url, api_key, extra_headers),
        }
    }

    pub async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        tracing::debug!(provider = "openai-responses", model = %req.model, "starting stream request");
        // Some zen-compatible proxies reject the Responses reasoning field even
        // for models that otherwise speak Responses.  This narrow fallback is
        // intentionally separate from transient retry handling.
        let response = match self
            .http
            .post_json("/responses", &build_request_body(req, req.reasoning))
            .await
        {
            Ok(response) => response,
            // `Auto` is deliberately best-effort so compatible proxies retain
            // the old fallback. Explicit user choices are never silently
            // discarded.
            Err(error)
                if req.reasoning == ReasoningPolicy::Auto
                    && matches!(&error, LlmError::Http {
                        status: 400,
                        body,
                        ..
                    } if body.to_ascii_lowercase().contains("reasoning")) =>
            {
                tracing::debug!("Responses endpoint rejected reasoning; retrying without it");
                self.http
                    .post_json("/responses", &build_request_body(req, ReasoningPolicy::Off))
                    .await?
            }
            Err(error) => return Err(error),
        };
        Ok(event_stream(stream_response(response)))
    }
}

pub fn build_request_body(req: &CompletionRequest, reasoning: ReasoningPolicy) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), Value::String(req.model.clone()));
    if let Some(system) = &req.system {
        body.insert("instructions".into(), Value::String(system.clone()));
    }
    body.insert("input".into(), Value::Array(convert_input(&req.messages)));
    if !req.tools.is_empty() {
        body.insert("tools".into(), Value::Array(convert_tools(&req.tools)));
        body.insert("tool_choice".into(), Value::String("auto".into()));
    }
    match reasoning {
        ReasoningPolicy::Off => {}
        ReasoningPolicy::Auto => {
            body.insert("reasoning".into(), json!({ "summary": "auto" }));
        }
        ReasoningPolicy::Effort(_) => {
            body.insert(
                "reasoning".into(),
                json!({ "effort": openai_reasoning_effort(reasoning), "summary": "auto" }),
            );
        }
    }
    if let Some(max_tokens) = req.max_tokens {
        body.insert("max_output_tokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = req.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    body.insert("store".into(), Value::Bool(false));
    body.insert("stream".into(), Value::Bool(true));
    Value::Object(body)
}

pub fn convert_input(messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::new();
    for message in messages {
        match message.role {
            Role::System => {}
            Role::User => {
                for text in text_parts(&message.content) {
                    items.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": text }]
                    }));
                }
            }
            Role::Assistant => {
                let text: String = message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if !text.is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }]
                    }));
                }
                for content in &message.content {
                    if let Content::ToolCall(call) = content {
                        items.push(json!({
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": stringify_arguments(&call.arguments),
                        }));
                    }
                }
            }
            Role::Tool => {
                for content in &message.content {
                    if let Content::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                    } = content
                    {
                        let output = if *is_error {
                            format!("Error: {content}")
                        } else {
                            content.clone()
                        };
                        items.push(json!({
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": output,
                        }));
                    }
                }
            }
        }
    }
    items
}

fn text_parts(content: &[Content]) -> impl Iterator<Item = &str> {
    content.iter().filter_map(|content| match content {
        Content::Text(text) => Some(text.as_str()),
        _ => None,
    })
}

pub fn convert_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect()
}

fn stringify_arguments(arguments: &Value) -> String {
    serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into())
}

#[derive(Debug, Default)]
pub struct ResponsesParser {
    done: bool,
    /// Call IDs already observed in this response. IDs are unique per
    /// assistant response; malformed calls are held until the terminal event
    /// so an earlier valid call cannot escape before a later invalid one is
    /// detected.
    seen_ids: std::collections::HashSet<String>,
    pending_calls: Vec<ToolCall>,
}

impl ResponsesParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse_event(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, LlmError> {
        self.parse_payload(&event.data)
    }

    pub fn parse_payload(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        let value: Value = serde_json::from_str(payload)
            .map_err(|error| LlmError::Parse(format!("Responses SSE payload: {error}")))?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "response.output_text.delta" => Ok(value
                .get("delta")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|value| vec![StreamEvent::TextDelta(value.to_owned())])
                .unwrap_or_default()),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => Ok(value
                .get("delta")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|value| vec![StreamEvent::ReasoningDelta(value.to_owned())])
                .unwrap_or_default()),
            "response.output_item.done" => {
                let item = value.get("item").unwrap_or(&Value::Null);
                if item.get("type").and_then(Value::as_str) != Some("function_call") {
                    return Ok(Vec::new());
                }
                let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or("");
                let arguments = if arguments.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(arguments).map_err(|error| {
                        LlmError::Parse(format!("invalid Responses tool arguments: {error}"))
                    })?
                };
                // Missing or blank IDs/names cannot be replayed to the
                // provider, so they fail with `LlmError::Parse` (handled by
                // the agent's malformed-tool recovery) instead of becoming
                // synthetic IDs.  No `ToolCallComplete` is emitted.
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        LlmError::Parse("Responses tool call is missing a call ID".into())
                    })?
                    .to_owned();
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        LlmError::Parse(format!("Responses tool call {id} is missing a name"))
                    })?
                    .to_owned();
                if self.seen_ids.contains(&id) {
                    return Err(LlmError::Parse(format!(
                        "duplicate Responses tool call ID {id}"
                    )));
                }
                self.seen_ids.insert(id.clone());
                self.pending_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
                Ok(Vec::new())
            }
            "response.completed" | "response.incomplete" => {
                self.done = true;
                let response = value.get("response").unwrap_or(&Value::Null);
                let usage = response.get("usage").map(parse_usage).transpose()?;
                // Preserve the terminal status verbatim: `completed` and
                // each `incomplete` reason (e.g. `max_output_tokens`) are
                // normal stop reasons the agent records, not errors (see
                // the all-normal rationale below).  The
                // `incomplete_details.reason` (when present) is surfaced by
                // mapping it into the stop reason so it survives in
                // `Done` even though no separate event carries it.
                let mut stop_reason = response
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                // Every `incomplete` reason is a normal stop: the provider
                // delivered a complete terminal event (status + reason +
                // usage), so the turn records it in `Done` rather than
                // retrying. Only transport-level problems (`Stream`), HTTP
                // failures, and auth errors are retryable — see
                // `LlmError::is_retryable`.
                if kind == "response.incomplete" {
                    let detail = response
                        .get("incomplete_details")
                        .and_then(|details| details.get("reason"))
                        .and_then(Value::as_str);
                    stop_reason = Some(match (stop_reason, detail) {
                        (Some(status), Some(reason)) => format!("{status}: {reason}"),
                        (Some(status), None) => status,
                        (None, Some(reason)) => format!("incomplete: {reason}"),
                        (None, None) => "incomplete".to_owned(),
                    });
                }
                let mut output = self
                    .pending_calls
                    .drain(..)
                    .map(StreamEvent::ToolCallComplete)
                    .collect::<Vec<_>>();
                output.push(StreamEvent::Done { stop_reason, usage });
                Ok(output)
            }
            "response.failed" => Err(LlmError::Stream(error_message(&value, "response failed"))),
            "error" => Err(LlmError::Stream(error_message(&value, "Responses error"))),
            _ => {
                tracing::trace!(event_type = kind, "ignored Responses SSE event");
                Ok(Vec::new())
            }
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Clean transport EOF without `response.completed`/`response.incomplete`
    /// is a truncated stream, not a successful turn: surface it as
    /// `LlmError::Stream` so the agent's recovery path handles it.  Never
    /// manufacture `Done` and never finalize unfinished tool calls here.
    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.done {
            return Ok(Vec::new());
        }
        self.done = true;
        Err(LlmError::Stream(
            "Responses stream ended without a terminal event (expected response.completed or response.incomplete)"
                .into(),
        ))
    }
}

fn error_message(value: &Value, fallback: &str) -> String {
    value
        .get("error")
        .and_then(value_message)
        .or_else(|| {
            value
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(value_message)
        })
        .or_else(|| value.get("message").and_then(Value::as_str))
        .unwrap_or(fallback)
        .to_owned()
}

fn value_message(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("message").and_then(Value::as_str))
}

fn parse_usage(value: &Value) -> Result<Usage, LlmError> {
    Ok(Usage {
        input_tokens: value.get("input_tokens").and_then(as_u64).unwrap_or(0),
        output_tokens: value.get("output_tokens").and_then(as_u64).unwrap_or(0),
        cached_tokens: value
            .get("input_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(as_u64),
        reasoning_tokens: value
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(as_u64),
        cost: value.get("cost").and_then(Value::as_f64),
    })
}

impl super::StreamParser for ResponsesParser {
    fn parse_event(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, LlmError> {
        Self::parse_event(self, event)
    }

    fn is_done(&self) -> bool {
        Self::is_done(self)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        Self::finish(self)
    }
}

fn event_stream(sse: crate::sse::SseStream) -> EventStream {
    super::drive_parser_stream(sse, ResponsesParser::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(reasoning: ReasoningPolicy) -> CompletionRequest {
        CompletionRequest {
            model: "demo".into(),
            system: None,
            messages: vec![Message::user("hello")],
            tools: Vec::new(),
            max_tokens: None,
            temperature: None,
            reasoning,
            session_id: None,
        }
    }

    #[test]
    fn serializes_reasoning_policy() {
        let off = build_request_body(&request(ReasoningPolicy::Off), ReasoningPolicy::Off);
        assert!(off.get("reasoning").is_none());

        let auto = build_request_body(&request(ReasoningPolicy::Auto), ReasoningPolicy::Auto);
        assert_eq!(auto["reasoning"]["summary"], "auto");
        assert!(auto["reasoning"].get("effort").is_none());

        let maximum = ReasoningPolicy::Effort(crate::ReasoningEffort::Maximum);
        let explicit = build_request_body(&request(maximum), maximum);
        assert_eq!(explicit["reasoning"]["effort"], "xhigh");
        assert_eq!(explicit["reasoning"]["summary"], "auto");
    }

    #[test]
    fn converts_input_items() {
        let messages = vec![
            Message::user("hello"),
            Message::assistant(vec![
                Content::Text("sure".into()),
                Content::ToolCall(ToolCall {
                    id: "call-1".into(),
                    name: "bash".into(),
                    arguments: json!({"command":"pwd"}),
                }),
            ]),
            Message::tool_result("call-1", "output", false),
        ];
        let input = convert_input(&messages);
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[3]["type"], "function_call_output");
    }

    #[test]
    fn finish_without_terminal_event_is_a_stream_error() {
        // Clean EOF with no `response.completed`/`response.incomplete` is
        // a truncated stream, not success: `finish` returns
        // `LlmError::Stream` and emits no completed calls.
        let mut parser = ResponsesParser::new();
        assert!(!parser.is_done());
        let error = parser.finish().unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
        assert!(parser.is_done());
        // A second finish after the error is a no-op.
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn text_then_eof_fails_and_partial_tool_call_emits_nothing() {
        let mut parser = ResponsesParser::new();
        parser
            .parse_payload(r#"{"type":"response.output_text.delta","delta":"hi"}"#)
            .unwrap();
        let error = parser.finish().unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
    }

    #[test]
    fn held_partial_tool_call_emits_nothing_at_eof() {
        // A completed `function_call` item is held (no `ToolCallComplete`
        // escapes before the terminal event). Clean EOF first must surface
        // `Stream` (never `Done`); `finish` never drains held calls.
        let mut parser = ResponsesParser::new();
        let held = parser.parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"held-1","name":"read","arguments":"{\"path\":\"x\"}"}}"#).unwrap();
        assert!(
            !held
                .iter()
                .any(|event| matches!(event, StreamEvent::ToolCallComplete(_))),
            "held call escaped before terminal: {held:?}"
        );
        let error = parser.finish().unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
        assert!(parser.is_done());
        // A second finish after the error is a no-op, still no `Done`.
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn stream_error_after_partial_output_remains_an_error() {
        // Partial text followed by a provider failure payload stays an
        // error: `response.failed` surfaces `LlmError::Stream` (agent
        // recovery path) instead of ever reaching a terminal `Done`.
        let mut parser = ResponsesParser::new();
        let events = parser
            .parse_payload(r#"{"type":"response.output_text.delta","delta":"hi"}"#)
            .unwrap();
        assert_eq!(events, vec![StreamEvent::TextDelta("hi".into())]);
        let error = parser
            .parse_payload(r#"{"type":"response.failed","response":{"status":"failed","error":{"message":"boom"}}}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)), "got {error:?}");
        assert!(!parser.is_done());
    }

    #[test]
    fn incomplete_terminal_event_succeeds_with_reason_and_usage() {
        // All `incomplete` reasons are normal stops: the provider sent a
        // complete terminal event, so the turn records `Done` (with reason
        // and usage) instead of retrying. Only transport/HTTP/auth
        // failures are retryable.
        let mut parser = ResponsesParser::new();
        let done = parser.parse_payload(r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":10,"output_tokens":20}}}"#).unwrap();
        assert!(matches!(
            &done[0],
            StreamEvent::Done {
                stop_reason: Some(reason),
                usage: Some(Usage { input_tokens: 10, output_tokens: 20, .. }),
            } if reason.contains("max_output_tokens")
        ));
        assert!(parser.is_done());
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn finish_is_noop_after_completed() {
        let mut parser = ResponsesParser::new();
        parser
            .parse_payload(r#"{"type":"response.completed","response":{"status":"completed"}}"#)
            .unwrap();
        assert!(parser.is_done());
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn terminal_event_split_across_transport_chunks_succeeds() {
        // The `response.completed` terminal split mid-object across two
        // `push_bytes` calls still terminates: the decoder buffers the
        // partial line and only `process_line`s on `\n`, so dialect
        // parsing sees the reassembled payload.
        use crate::sse::SseParser;
        let mut sse = SseParser::new();
        // Split *between* SSE lines (after the `data:` line's `\n`): the
        // first chunk holds a complete data line, the second the blank
        // dispatch line. Byte-halving one `push_bytes` call would split
        // inside the JSON string instead (an interior `\n` is a line
        // separator, and half a `\n\n` terminator dispatches nothing).
        let first = r#"{"type":"response.completed","response":{"status":"completed"}}"#;
        let payload = format!("data: {first}\n");
        assert!(sse.push_bytes(payload.as_bytes()).unwrap().is_empty());
        let events = sse.push_bytes(b"\n").unwrap();
        assert_eq!(events.len(), 1, "reassembled terminal: {events:?}");
        let mut parser = ResponsesParser::new();
        let done = parser.parse_event(&events[0]).unwrap();
        assert!(matches!(&done[0], StreamEvent::Done { .. }), "got {done:?}");
        assert!(parser.is_done());
    }
    #[test]
    fn parses_response_events() {
        let mut parser = ResponsesParser::new();
        assert_eq!(
            parser
                .parse_payload(r#"{"type":"response.output_text.delta","delta":"hi"}"#)
                .unwrap(),
            vec![StreamEvent::TextDelta("hi".into())]
        );
        assert_eq!(
            parser
                .parse_payload(
                    r#"{"type":"response.reasoning_summary_text.delta","delta":"think"}"#
                )
                .unwrap(),
            vec![StreamEvent::ReasoningDelta("think".into())]
        );
        let calls = parser.parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"c","name":"read","arguments":"{\"path\":\"x\"}"}}"#).unwrap();
        assert!(calls.is_empty(), "calls are held until the terminal event");
        let done = parser.parse_payload(r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":4,"output_tokens_details":{"reasoning_tokens":1}}}}"#).unwrap();
        assert!(matches!(&done[0], StreamEvent::ToolCallComplete(call) if call.name == "read"));
        assert!(matches!(
            &done[1],
            StreamEvent::Done {
                usage: Some(Usage {
                    input_tokens: 3,
                    reasoning_tokens: Some(1),
                    ..
                }),
                ..
            }
        ));
    }

    #[test]
    fn rejects_missing_id() {
        let mut parser = ResponsesParser::new();
        let error = parser
            .parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","name":"read","arguments":"{}"}}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
    }

    #[test]
    fn rejects_blank_id_and_missing_name() {
        let mut parser = ResponsesParser::new();
        let error = parser
            .parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"  ","name":"read","arguments":"{}"}}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
        let mut parser = ResponsesParser::new();
        let error = parser
            .parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"c1","arguments":"{}"}}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
    }

    #[test]
    fn rejects_duplicate_ids_in_parallel_calls() {
        let mut parser = ResponsesParser::new();
        parser
            .parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"dup","name":"read","arguments":"{}"}}"#)
            .unwrap();
        let error = parser
            .parse_payload(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"dup","name":"bash","arguments":"{}"}}"#)
            .unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
    }
}
