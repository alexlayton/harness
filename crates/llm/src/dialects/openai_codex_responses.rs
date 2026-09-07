//! Wire translation for ChatGPT's Codex Responses endpoint.
//!
//! This is intentionally separate from the public OpenAI Responses API: the
//! subscription service uses a different endpoint and requires Codex-specific
//! request fields even though it streams familiar Responses SSE events.
use crate::dialects::openai_reasoning_effort;
use crate::dialects::openai_responses::{
    ResponsesParser, convert_input as base_convert_input, convert_tools,
};
use crate::http::HttpClient;
use crate::sse::stream_response;
use crate::{CompletionRequest, EventStream, LlmError, ReasoningPolicy};
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
    });
    match request.reasoning {
        ReasoningPolicy::Off => {}
        ReasoningPolicy::Auto => {
            // Preserve the subscription endpoint's historical default.
            body["reasoning"] = json!({ "effort": "medium", "summary": "auto" });
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
        ReasoningPolicy::Effort(_) => {
            body["reasoning"] = json!({
                "effort": openai_reasoning_effort(request.reasoning),
                "summary": "auto"
            });
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
    }
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

/// Convert neutral harness history to the subset accepted by Codex.
///
/// Assistant content is serialized in one ordered pass so Codex opaque
/// reasoning state stays in its original position relative to the function
/// calls it precedes (the Responses base converter emits text first, then
/// all calls, which would reorder interleaved opaque items).  Only
/// Codex-owned opaque items are replayed; foreign-provider state is
/// omitted rather than leaking another wire format into this endpoint.
pub fn convert_input(messages: &[crate::Message]) -> Vec<Value> {
    use crate::{Content, Role};
    let mut input = Vec::new();
    for message in messages {
        match message.role {
            Role::Assistant => {
                // Ordered pass: text block first (matching base behavior),
                // then each tool call / Codex opaque item in content order.
                let text: String = message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if !text.is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }]
                    }));
                }
                for content in &message.content {
                    match content {
                        Content::ToolCall(call) => input.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into()),
                        })),
                        Content::Opaque { provider, data } if provider == "openai-codex" => {
                            input.push(data.clone());
                        }
                        _ => {}
                    }
                }
            }
            _ => {
                // Non-assistant messages keep base conversion (which skips
                // opaque items entirely); Codex opaque items only ever ride
                // on assistant turns.
                input.extend(base_convert_input(std::slice::from_ref(message)));
            }
        }
    }
    input
}

/// An output item that must remain in wire order until the shared Responses
/// parser releases function calls at the terminal event.
#[derive(Debug)]
enum CodexOutputItem {
    Opaque(Value),
    ToolCall,
}

/// Codex shares the Responses terminal contract but keeps encrypted
/// reasoning items as opaque replay state. Function calls are deliberately
/// held by [`ResponsesParser`] until the terminal event, so Codex keeps an
/// ordered marker for every output item and releases both kinds together.
struct CodexParser {
    inner: ResponsesParser,
    pending_items: Vec<CodexOutputItem>,
}

impl CodexParser {
    fn new() -> Self {
        Self {
            inner: ResponsesParser::new(),
            pending_items: Vec::new(),
        }
    }

    fn flush_terminal(
        &mut self,
        events: Vec<crate::StreamEvent>,
    ) -> Result<Vec<crate::StreamEvent>, crate::LlmError> {
        let mut calls = Vec::new();
        let mut other = Vec::new();
        let mut terminal = Vec::new();
        for event in events {
            match event {
                crate::StreamEvent::ToolCallComplete(call) => calls.push(call),
                done @ crate::StreamEvent::Done { .. } => terminal.push(done),
                event => other.push(event),
            }
        }

        let expected_calls = self
            .pending_items
            .iter()
            .filter(|item| matches!(item, CodexOutputItem::ToolCall))
            .count();
        if calls.len() != expected_calls {
            return Err(crate::LlmError::Stream(
                "Codex output items and parsed tool calls diverged".into(),
            ));
        }

        let mut calls = calls.into_iter();
        let mut output =
            Vec::with_capacity(self.pending_items.len() + other.len() + terminal.len());
        for item in std::mem::take(&mut self.pending_items) {
            match item {
                CodexOutputItem::Opaque(data) => output.push(crate::StreamEvent::OpaqueState {
                    provider: "openai-codex".into(),
                    data,
                }),
                CodexOutputItem::ToolCall => {
                    let call = calls.next().ok_or_else(|| {
                        crate::LlmError::Stream(
                            "Codex output items and parsed tool calls diverged".into(),
                        )
                    })?;
                    output.push(crate::StreamEvent::ToolCallComplete(call));
                }
            }
        }
        debug_assert!(calls.next().is_none());
        output.extend(other);
        output.extend(terminal);
        Ok(output)
    }
}

impl super::StreamParser for CodexParser {
    fn parse_event(
        &mut self,
        event: &crate::sse::SseEvent,
    ) -> Result<Vec<crate::StreamEvent>, crate::LlmError> {
        // Classify before delegating, but append the marker only after the
        // shared parser accepts the item. Function calls take precedence over
        // encrypted content so malformed calls cannot bypass validation.
        let item = serde_json::from_str::<Value>(&event.data)
            .ok()
            .filter(|value| {
                value.get("type").and_then(Value::as_str) == Some("response.output_item.done")
            })
            .and_then(|value| {
                let item = value.get("item")?;
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    Some(CodexOutputItem::ToolCall)
                } else if item.get("encrypted_content").is_some() {
                    Some(CodexOutputItem::Opaque(item.clone()))
                } else {
                    None
                }
            });

        let events = self.inner.parse_event(event)?;
        if self.inner.is_done() {
            return self.flush_terminal(events);
        }
        if let Some(item) = item {
            self.pending_items.push(item);
        }
        Ok(events)
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    fn finish(&mut self) -> Result<Vec<crate::StreamEvent>, crate::LlmError> {
        // Do not release pending opaque state on EOF: the shared parser has
        // not observed a valid terminal event, so replaying it would create a
        // durable partial turn.
        self.inner.finish()
    }
}

fn event_stream(sse: crate::sse::SseStream) -> EventStream {
    super::drive_parser_stream(sse, CodexParser::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ReasoningEffort, StreamEvent};

    fn request(reasoning: ReasoningPolicy) -> CompletionRequest {
        CompletionRequest {
            model: "gpt-test".into(),
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
    fn reasoning_policy_controls_codex_fields() {
        let off = build_request_body(&request(ReasoningPolicy::Off));
        assert!(off.get("reasoning").is_none());
        assert!(off.get("include").is_none());

        let auto = build_request_body(&request(ReasoningPolicy::Auto));
        assert_eq!(auto["reasoning"]["effort"], "medium");
        assert_eq!(auto["include"][0], "reasoning.encrypted_content");

        let maximum =
            build_request_body(&request(ReasoningPolicy::Effort(ReasoningEffort::Maximum)));
        assert_eq!(maximum["reasoning"]["effort"], "xhigh");
    }

    #[test]
    fn incomplete_terminal_event_succeeds_with_reason_and_usage() {
        // Codex shares the Responses terminal contract: `response.incomplete`
        // is a handled terminal event preserving reason, status, and usage.
        use crate::dialects::StreamParser;
        let mut parser = CodexParser::new();
        let done = StreamParser::parse_event(
            &mut parser,
            &crate::sse::SseEvent {
                event: None,
                data: r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"usage":{"input_tokens":7,"output_tokens":3}}}"#.into(),
            },
        )
        .unwrap();
        assert!(
            matches!(&done[0], StreamEvent::Done { stop_reason: Some(reason), usage: Some(usage) }
                if reason.contains("content_filter") && usage.input_tokens == 7 && usage.output_tokens == 3),
            "got {done:?}"
        );
        assert!(parser.is_done());
    }

    #[test]
    fn codex_stream_parser_preserves_interleaved_output_item_order() {
        // The SSE parser and Codex wrapper are exercised together here. Calls
        // are held by the shared Responses parser, but opaque items use the
        // same ordered queue so terminal output still matches provider order.
        use crate::dialects::StreamParser;
        let opaque = |id: &str, encrypted: &str| {
            serde_json::json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": id,
                    "encrypted_content": encrypted,
                },
            })
            .to_string()
        };
        let call = |id: &str, name: &str, path: &str| {
            serde_json::json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": serde_json::json!({"path": path}).to_string(),
                },
            })
            .to_string()
        };
        let wire = [
            opaque("rs_1", "enc1"),
            call("c1", "read", "one"),
            opaque("rs_2", "enc2"),
            call("c2", "read", "two"),
            serde_json::json!({
                "type": "response.completed",
                "response": {"status": "completed"},
            })
            .to_string(),
        ]
        .into_iter()
        .map(|payload| format!("data: {payload}\n\n"))
        .collect::<String>();
        let events = crate::sse::parse_events(&wire);
        let mut parser = CodexParser::new();
        let mut output = Vec::new();
        for (index, event) in events.iter().enumerate() {
            let parsed = StreamParser::parse_event(&mut parser, event).unwrap();
            if index < 4 {
                assert!(
                    parsed.is_empty(),
                    "item escaped before terminal: {parsed:?}"
                );
            }
            output.extend(parsed);
        }

        assert_eq!(
            output,
            vec![
                StreamEvent::OpaqueState {
                    provider: "openai-codex".into(),
                    data: serde_json::json!({
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "enc1",
                    }),
                },
                StreamEvent::ToolCallComplete(crate::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "one"}),
                }),
                StreamEvent::OpaqueState {
                    provider: "openai-codex".into(),
                    data: serde_json::json!({
                        "type": "reasoning",
                        "id": "rs_2",
                        "encrypted_content": "enc2",
                    }),
                },
                StreamEvent::ToolCallComplete(crate::ToolCall {
                    id: "c2".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "two"}),
                }),
                StreamEvent::Done {
                    stop_reason: Some("completed".into()),
                    usage: None,
                },
            ]
        );
    }

    #[test]
    fn codex_opaque_state_preserves_order_before_multiple_tool_calls() {
        // Opaque Codex reasoning items stay in original content order
        // relative to the function calls they precede; foreign-provider
        // state is omitted by `convert_input`.
        use crate::{Content, Role};
        let opaque = |n: u64| serde_json::json!({"type": "reasoning", "id": format!("rs_{n}"), "encrypted_content": format!("enc{n}")});
        let messages = vec![crate::Message {
            role: Role::Assistant,
            content: vec![
                Content::Opaque {
                    provider: "openai-codex".into(),
                    data: opaque(1),
                },
                Content::ToolCall(crate::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({}),
                }),
                Content::Opaque {
                    provider: "other-provider".into(),
                    data: opaque(9),
                },
                Content::Opaque {
                    provider: "openai-codex".into(),
                    data: opaque(2),
                },
                Content::ToolCall(crate::ToolCall {
                    id: "c2".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({}),
                }),
            ],
        }];
        let input = convert_input(&messages);
        let rendered: Vec<String> = input.iter().map(|item| item.to_string()).collect();
        assert!(
            !rendered.iter().any(|item| item.contains("rs_9")),
            "foreign state leaked: {rendered:?}"
        );
        // Exact content order: rs_1, c1, rs_2, c2 — interleaved as authored.
        // (Call IDs serialize as `call_id`, so match the encrypted payload
        // marker plus the call_id value to stay unambiguous.)
        let positions = ["enc1", "\"c1\"", "enc2", "\"c2\""].map(|needle| {
            rendered
                .iter()
                .position(|item| item.contains(needle))
                .unwrap()
        });
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "order violated: {rendered:?}"
        );
    }

    #[test]
    fn codex_rejects_malformed_tool_calls_through_shared_parser() {
        // Codex delegates validation to `ResponsesParser`: a missing call ID
        // fails with `LlmError::Parse` before any queued opaque state or
        // `ToolCallComplete` can escape, so malformed-tool recovery remains
        // atomic for the whole response.
        use crate::LlmError;
        use crate::dialects::StreamParser;
        let mut parser = CodexParser::new();
        let opaque = StreamParser::parse_event(
            &mut parser,
            &crate::sse::SseEvent {
                event: None,
                data: serde_json::json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "enc1",
                    },
                })
                .to_string(),
            },
        )
        .unwrap();
        assert!(
            opaque.is_empty(),
            "opaque state escaped before terminal: {opaque:?}"
        );
        let error = StreamParser::parse_event(
            &mut parser,
            &crate::sse::SseEvent {
                event: None,
                data: r#"{"type":"response.output_item.done","item":{"type":"function_call","name":"read","arguments":"{}"}}"#.into(),
            },
        )
        .unwrap_err();
        assert!(matches!(error, LlmError::Parse(_)), "got {error:?}");
    }
}
