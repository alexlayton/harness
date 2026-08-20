use crate::as_u64;
use crate::http::HttpClient;
use crate::sse::{SseEvent, stream_response};
use crate::{
    CompletionRequest, Content, EventStream, LlmError, Message, ModelInfo, Role, StreamEvent,
    ToolCall, ToolDefinition, Usage,
};
use futures_util::StreamExt;
use reqwest::header::HeaderMap;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct OpenAiChatClient {
    http: HttpClient,
}

impl OpenAiChatClient {
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
        tracing::debug!(provider = "openai-chat", model = %req.model, "starting stream request");
        let response = self
            .http
            .post_json("/chat/completions", &build_request_body(req))
            .await?;
        Ok(event_stream(stream_response(response)))
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let response = self.http.get("/models").await?;
        let body = response.text().await.map_err(LlmError::Network)?;
        parse_models_body(&body)
    }
}

/// Build an OpenAI-compatible chat request without making a network call.
pub fn build_request_body(req: &CompletionRequest) -> Value {
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
                        Content::Reasoning(_) | Content::ToolResult { .. } => {}
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
        if payload.trim() == "[DONE]" {
            return self.finish();
        }
        let value: Value = serde_json::from_str(payload)
            .map_err(|error| LlmError::Parse(format!("chat SSE payload: {error}")))?;
        let mut output = Vec::new();

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
            output.extend(self.flush_calls()?);
            output.push(StreamEvent::Done {
                stop_reason: self.stop_reason.clone(),
                usage: Some(parse_usage(usage_value)?),
            });
            self.done = true;
        }
        Ok(output)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.done {
            return Ok(Vec::new());
        }
        let mut output = self.flush_calls()?;
        output.push(StreamEvent::Done {
            stop_reason: self.stop_reason.clone(),
            usage: None,
        });
        self.done = true;
        Ok(output)
    }

    fn accumulate_tool_call(&mut self, item: &Value, fallback_index: u64) {
        let index = item.get("index").and_then(as_u64).unwrap_or(fallback_index);
        let call = self.calls.entry(index).or_default();
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            call.id = id.to_owned();
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
        if self.calls_flushed {
            return Ok(Vec::new());
        }
        self.calls_flushed = true;
        let mut result = Vec::new();
        for (_, call) in std::mem::take(&mut self.calls) {
            let arguments = if call.arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&call.arguments).map_err(|error| {
                    LlmError::Parse(format!("invalid tool arguments for {}: {error}", call.name))
                })?
            };
            result.push(StreamEvent::ToolCallComplete(ToolCall {
                id: if call.id.is_empty() {
                    tracing::warn!(
                        name = %call.name,
                        "tool call streamed without an id; generated synthetic id that cannot be matched in later turns"
                    );
                    format!("call-{}", result.len())
                } else {
                    call.id
                },
                name: call.name,
                arguments,
            }));
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

fn event_stream(mut sse: crate::sse::SseStream) -> EventStream {
    let stream = async_stream::try_stream! {
        let mut parser = ChatStreamParser::new();
        while let Some(event) = sse.next().await {
            let event = event?;
            for item in parser.parse_event(&event)? {
                yield item;
            }
        }
        if !parser.done {
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
            reasoning: true,
        }
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
}
