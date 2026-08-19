use crate::as_u64;
use crate::http::HttpClient;
use crate::sse::{SseEvent, stream_response};
use crate::{
    CompletionRequest, Content, EventStream, LlmError, Message, Role, StreamEvent, ToolCall,
    ToolDefinition, Usage,
};
use futures_util::StreamExt;
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

    pub fn request_body(&self, req: &CompletionRequest) -> Value {
        build_request_body(req, req.reasoning)
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
            Err(error)
                if req.reasoning
                    && matches!(&error, LlmError::Http { status: 400, body } if body.to_ascii_lowercase().contains("reasoning")) =>
            {
                tracing::debug!("Responses endpoint rejected reasoning; retrying without it");
                self.http
                    .post_json("/responses", &build_request_body(req, false))
                    .await?
            }
            Err(error) => return Err(error),
        };
        Ok(event_stream(stream_response(response)))
    }
}

pub fn build_request_body(req: &CompletionRequest, include_reasoning: bool) -> Value {
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
    if include_reasoning && req.reasoning {
        body.insert("reasoning".into(), json!({ "summary": "auto" }));
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
}

pub fn parse_payload(
    payload: &str,
    parser: &mut ResponsesParser,
) -> Result<Vec<StreamEvent>, LlmError> {
    parser.parse_payload(payload)
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
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("response-call")
                    .to_owned();
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                Ok(vec![StreamEvent::ToolCallComplete(ToolCall {
                    id,
                    name,
                    arguments,
                })])
            }
            "response.completed" => {
                self.done = true;
                let response = value.get("response").unwrap_or(&Value::Null);
                let usage = response.get("usage").map(parse_usage).transpose()?;
                let stop_reason = response
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                Ok(vec![StreamEvent::Done { stop_reason, usage }])
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

    /// Emit a fallback `Done` event when the SSE stream ended without a
    /// `response.completed` (e.g. a proxy dropped the connection after the last
    /// text delta).  Mirrors `ChatStreamParser::finish` so the agent loop always
    /// receives `TurnFinished` rather than staying busy forever.
    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.done {
            return Ok(Vec::new());
        }
        self.done = true;
        Ok(vec![StreamEvent::Done {
            stop_reason: None,
            usage: None,
        }])
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

fn event_stream(mut sse: crate::sse::SseStream) -> EventStream {
    let stream = async_stream::try_stream! {
        let mut parser = ResponsesParser::new();
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
    fn finish_emits_fallback_done_only_once() {
        let mut parser = ResponsesParser::new();
        assert!(!parser.is_done());
        let events = parser.finish().unwrap();
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::Done {
                stop_reason: None,
                usage: None
            }]
        ));
        // A second finish (or a later completed event) is a no-op.
        assert!(parser.finish().unwrap().is_empty());
        assert!(parser.is_done());
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
        assert!(matches!(&calls[0], StreamEvent::ToolCallComplete(call) if call.name == "read"));
        let done = parser.parse_payload(r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":4,"output_tokens_details":{"reasoning_tokens":1}}}}"#).unwrap();
        assert!(matches!(
            &done[0],
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
}
