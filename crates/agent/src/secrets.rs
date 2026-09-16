//! Exact-value secret masking at the agent/provider boundary.
//!
//! Secret values stay in memory. Provider requests, provider text, tool calls,
//! and tool results use stable named placeholders instead. Opaque provider
//! continuation items that contain an exact secret are dropped because their
//! authenticated data cannot be changed safely. This is privacy hygiene rather
//! than a sandbox: local tools can still read process state and access the
//! network.

use futures_util::Stream;
use llm::{
    CompletionRequest, Content, EventStream, LlmError, Message, ModelInfo, Provider, StreamEvent,
    SubscriptionUsage, ToolCall,
};
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Minimum configured value length. Short exact values cause broad accidental
/// replacement in source code, paths, and natural-language output.
pub const MIN_SECRET_BYTES: usize = 8;

#[derive(Clone)]
struct SecretEntry {
    name: String,
    value: String,
    placeholder: String,
}

/// Immutable map from configured secret values to provider-safe placeholders.
#[derive(Clone, Default)]
pub struct SecretMasker {
    entries: Arc<[SecretEntry]>,
}

impl fmt::Debug for SecretMasker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SecretMasker")
            .field("names", &self.names().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl SecretMasker {
    /// Build a stable map from logical names and resolved values.
    ///
    /// Values are sorted longest-first so one secret that is a prefix of
    /// another cannot expose the longer value through partial replacement.
    pub fn new(entries: impl IntoIterator<Item = (String, String)>) -> Result<Self, String> {
        let mut seen_names = HashSet::new();
        let mut seen_values = HashSet::new();
        let mut resolved = Vec::new();
        for (name, value) in entries {
            if !seen_names.insert(name.clone()) {
                return Err(format!("duplicate secret environment variable `{name}`"));
            }
            if value.len() < MIN_SECRET_BYTES {
                return Err(format!(
                    "secret environment variable `{name}` must contain at least {MIN_SECRET_BYTES} bytes"
                ));
            }
            if !seen_values.insert(value.clone()) {
                continue;
            }
            let placeholder = format!("{{{{harness-secret:{name}}}}}");
            resolved.push(SecretEntry {
                name,
                value,
                placeholder,
            });
        }
        resolved.sort_by(|left, right| {
            right
                .value
                .len()
                .cmp(&left.value.len())
                .then_with(|| left.name.cmp(&right.name))
        });
        Ok(Self {
            entries: resolved.into(),
        })
    }

    /// Whether no values are configured.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Configured logical names. Values are deliberately not exposed.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.name.as_str())
    }

    /// Replace every exact configured value with its stable placeholder.
    pub fn mask_text(&self, text: &str) -> String {
        self.replace_entries(text, false)
    }

    /// Restore known placeholders in text used only for approved local tools.
    pub fn restore_text(&self, text: &str) -> String {
        self.replace_entries(text, true)
    }

    fn contains_secret(&self, text: &str) -> bool {
        self.entries.iter().any(|entry| text.contains(&entry.value))
    }

    fn json_contains_secret(&self, value: &Value) -> bool {
        match value {
            Value::String(text) => self.contains_secret(text),
            Value::Array(values) => values.iter().any(|value| self.json_contains_secret(value)),
            Value::Object(values) => values
                .iter()
                .any(|(key, value)| self.contains_secret(key) || self.json_contains_secret(value)),
            Value::Null | Value::Bool(_) | Value::Number(_) => false,
        }
    }

    fn replace_entries(&self, text: &str, restore: bool) -> String {
        let mut output = String::with_capacity(text.len());
        let mut cursor = 0;
        while cursor < text.len() {
            let next = self
                .entries
                .iter()
                .filter_map(|entry| {
                    let needle = if restore {
                        &entry.placeholder
                    } else {
                        &entry.value
                    };
                    text[cursor..]
                        .find(needle)
                        .map(|offset| (cursor + offset, needle, entry))
                })
                .min_by_key(|(position, _, _)| *position);
            let Some((position, needle, entry)) = next else {
                output.push_str(&text[cursor..]);
                break;
            };
            output.push_str(&text[cursor..position]);
            output.push_str(if restore {
                &entry.value
            } else {
                &entry.placeholder
            });
            cursor = position + needle.len();
        }
        output
    }

    /// Mask JSON strings recursively, including object keys.
    pub fn mask_json(&self, value: &Value) -> Value {
        self.transform_json(value, false)
    }

    /// Restore known placeholders in JSON strings recursively, including keys.
    pub fn restore_json(&self, value: &Value) -> Value {
        self.transform_json(value, true)
    }

    fn transform_json(&self, value: &Value, restore: bool) -> Value {
        match value {
            Value::String(text) => Value::String(if restore {
                self.restore_text(text)
            } else {
                self.mask_text(text)
            }),
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .map(|value| self.transform_json(value, restore))
                    .collect(),
            ),
            Value::Object(values) => Value::Object(
                values
                    .iter()
                    .map(|(key, value)| {
                        let key = if restore {
                            self.restore_text(key)
                        } else {
                            self.mask_text(key)
                        };
                        (key, self.transform_json(value, restore))
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    /// Mask provider-visible message content without touching opaque
    /// provider-owned continuation state.
    pub fn mask_message(&self, message: &Message) -> Message {
        Message {
            role: message.role.clone(),
            content: message
                .content
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(Content::Text(self.mask_text(text))),
                    Content::Reasoning(text) => Some(Content::Reasoning(self.mask_text(text))),
                    // Opaque continuation data must not be modified because
                    // providers can authenticate its exact bytes. Dropping a
                    // contaminated item is safer than persisting plaintext or
                    // sending corrupted continuation state back.
                    Content::Opaque { data, .. } if self.json_contains_secret(data) => None,
                    Content::Opaque { provider, data } => Some(Content::Opaque {
                        provider: provider.clone(),
                        data: data.clone(),
                    }),
                    Content::ToolCall(call) => Some(Content::ToolCall(self.mask_tool_call(call))),
                    Content::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                    } => Some(Content::ToolResult {
                        tool_call_id: tool_call_id.clone(),
                        content: self.mask_text(content),
                        is_error: *is_error,
                    }),
                })
                .collect(),
        }
    }

    /// Mask a tool call while keeping its protocol identity unchanged.
    pub fn mask_tool_call(&self, call: &ToolCall) -> ToolCall {
        ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: self.mask_json(&call.arguments),
        }
    }

    /// Mask a completed tool result before it reaches hooks, history, or disk.
    pub fn mask_tool_output(&self, output: tools::ToolOutput) -> tools::ToolOutput {
        tools::ToolOutput {
            content: self.mask_text(&output.content),
            is_error: output.is_error,
            summary: self.mask_text(&output.summary),
        }
    }

    /// Return a masked clone of a provider request.
    pub fn mask_request(&self, request: &CompletionRequest) -> CompletionRequest {
        CompletionRequest {
            model: request.model.clone(),
            system: request.system.as_deref().map(|text| self.mask_text(text)),
            messages: request
                .messages
                .iter()
                .map(|message| self.mask_message(message))
                .collect(),
            tools: request
                .tools
                .iter()
                .map(|tool| llm::ToolDefinition {
                    name: tool.name.clone(),
                    description: self.mask_text(&tool.description),
                    parameters: self.mask_json(&tool.parameters),
                })
                .collect(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            reasoning: request.reasoning,
            session_id: request.session_id.clone(),
        }
    }

    fn mask_error(&self, mut error: LlmError) -> LlmError {
        for entry in self.entries.iter() {
            error = error.redacted(&entry.value);
        }
        error
    }
}

/// Incremental exact-value masker for provider deltas. It retains only a
/// suffix that could become a secret after the next delta arrives.
struct StreamingMasker {
    masker: Arc<SecretMasker>,
    pending: String,
}

impl StreamingMasker {
    fn new(masker: Arc<SecretMasker>) -> Self {
        Self {
            masker,
            pending: String::new(),
        }
    }

    fn push(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let retained = self.retained_suffix_len();
        let emit_len = self.pending.len().saturating_sub(retained);
        let emitted = self.masker.mask_text(&self.pending[..emit_len]);
        self.pending.drain(..emit_len);
        emitted
    }

    fn finish(&mut self) -> String {
        self.masker
            .mask_text(std::mem::take(&mut self.pending).as_str())
    }

    fn retained_suffix_len(&self) -> usize {
        let mut retained = 0;
        for entry in self.masker.entries.iter() {
            let prefix_lengths = entry
                .value
                .char_indices()
                .skip(1)
                .map(|(index, _)| index)
                .chain(std::iter::once(entry.value.len()));
            for prefix_len in prefix_lengths {
                if self.pending.ends_with(&entry.value[..prefix_len]) {
                    retained = retained.max(prefix_len);
                }
            }
        }
        retained
    }
}

struct MaskedEventStream {
    inner: EventStream,
    masker: Arc<SecretMasker>,
    text: StreamingMasker,
    reasoning: StreamingMasker,
    queued: VecDeque<Result<StreamEvent, LlmError>>,
    inner_done: bool,
}

impl MaskedEventStream {
    fn new(inner: EventStream, masker: Arc<SecretMasker>) -> Self {
        Self {
            inner,
            text: StreamingMasker::new(masker.clone()),
            reasoning: StreamingMasker::new(masker.clone()),
            masker,
            queued: VecDeque::new(),
            inner_done: false,
        }
    }

    fn queue_flush(&mut self) {
        let text = self.text.finish();
        if !text.is_empty() {
            self.queued.push_back(Ok(StreamEvent::TextDelta(text)));
        }
        let reasoning = self.reasoning.finish();
        if !reasoning.is_empty() {
            self.queued
                .push_back(Ok(StreamEvent::ReasoningDelta(reasoning)));
        }
    }
}

impl Stream for MaskedEventStream {
    type Item = Result<StreamEvent, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(item) = self.queued.pop_front() {
                return Poll::Ready(Some(item));
            }
            if self.inner_done {
                return Poll::Ready(None);
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.queue_flush();
                    self.inner_done = true;
                }
                Poll::Ready(Some(Ok(StreamEvent::TextDelta(delta)))) => {
                    let masked = self.text.push(&delta);
                    if !masked.is_empty() {
                        return Poll::Ready(Some(Ok(StreamEvent::TextDelta(masked))));
                    }
                }
                Poll::Ready(Some(Ok(StreamEvent::ReasoningDelta(delta)))) => {
                    let masked = self.reasoning.push(&delta);
                    if !masked.is_empty() {
                        return Poll::Ready(Some(Ok(StreamEvent::ReasoningDelta(masked))));
                    }
                }
                Poll::Ready(Some(Ok(StreamEvent::ToolCallComplete(call)))) => {
                    return Poll::Ready(Some(Ok(StreamEvent::ToolCallComplete(
                        self.masker.mask_tool_call(&call),
                    ))));
                }
                Poll::Ready(Some(Ok(event @ StreamEvent::Done { .. }))) => {
                    self.queue_flush();
                    self.queued.push_back(Ok(event));
                }
                Poll::Ready(Some(Ok(StreamEvent::OpaqueState { provider, data }))) => {
                    if !self.masker.json_contains_secret(&data) {
                        return Poll::Ready(Some(Ok(StreamEvent::OpaqueState { provider, data })));
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    self.queue_flush();
                    let error = self.masker.mask_error(error);
                    self.queued.push_back(Err(error));
                }
            }
        }
    }
}

/// Provider decorator that masks all outbound request text and inbound
/// displayable content. It deliberately leaves provider-owned opaque state
/// unchanged.
struct MaskingProvider {
    inner: Arc<dyn Provider>,
    masker: Arc<SecretMasker>,
}

/// Attach exact-value masking to a provider.
pub fn mask_provider(inner: Arc<dyn Provider>, masker: Arc<SecretMasker>) -> Arc<dyn Provider> {
    if masker.is_empty() {
        inner
    } else {
        Arc::new(MaskingProvider { inner, masker })
    }
}

#[async_trait::async_trait]
impl Provider for MaskingProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
        let request = self.masker.mask_request(request);
        let stream = self
            .inner
            .stream(&request)
            .await
            .map_err(|error| self.masker.mask_error(error))?;
        Ok(Box::pin(MaskedEventStream::new(
            stream,
            self.masker.clone(),
        )))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.inner
            .list_models()
            .await
            .map_err(|error| self.masker.mask_error(error))
    }

    async fn subscription_usage(&self) -> Result<Option<SubscriptionUsage>, LlmError> {
        self.inner
            .subscription_usage()
            .await
            .map_err(|error| self.masker.mask_error(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{StreamExt, stream};
    use std::sync::Mutex;

    fn masker() -> SecretMasker {
        SecretMasker::new([
            ("SHORT".into(), "abcdefgh".into()),
            ("LONG".into(), "abcdefgh-extra".into()),
        ])
        .unwrap()
    }

    #[test]
    fn masks_longest_values_first_and_restores_known_placeholders() {
        let masker = masker();
        let masked = masker.mask_text("abcdefgh-extra / abcdefgh");
        assert_eq!(masked, "{{harness-secret:LONG}} / {{harness-secret:SHORT}}");
        assert_eq!(masker.restore_text(&masked), "abcdefgh-extra / abcdefgh");
        assert_eq!(
            masker.restore_text("{{harness-secret:UNKNOWN}}"),
            "{{harness-secret:UNKNOWN}}"
        );
    }

    #[test]
    fn masks_and_restores_nested_json_values_and_keys() {
        let masker = masker();
        let original = serde_json::json!({
            "abcdefgh": [
                "prefix abcdefgh-extra suffix",
                {"key-abcdefgh-extra": "abcdefgh"}
            ]
        });
        let masked = masker.mask_json(&original);
        let serialized = masked.to_string();
        assert!(!serialized.contains("abcdefgh"), "{serialized}");
        assert!(serialized.contains("harness-secret:SHORT"));
        assert!(serialized.contains("harness-secret:LONG"));
        assert_eq!(masker.restore_json(&masked), original);
    }

    #[test]
    fn streaming_masker_holds_a_secret_split_across_deltas() {
        let masker = Arc::new(masker());
        for split in 1.."abcdefgh-extra".len() {
            let mut stream = StreamingMasker::new(masker.clone());
            let first = stream.push(&"abcdefgh-extra"[..split]);
            let second = stream.push(&"abcdefgh-extra"[split..]);
            let last = stream.finish();
            let output = format!("{first}{second}{last}");
            assert_eq!(output, "{{harness-secret:LONG}}", "split {split}");
        }
    }

    #[test]
    fn debug_never_contains_values() {
        let debug = format!("{:?}", masker());
        assert!(debug.contains("SHORT"));
        assert!(!debug.contains("abcdefgh"));
    }

    struct RecordingProvider {
        request: Arc<Mutex<Option<CompletionRequest>>>,
    }

    #[async_trait::async_trait]
    impl Provider for RecordingProvider {
        fn name(&self) -> &str {
            "recording"
        }

        async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
            *self.request.lock().unwrap() = Some(request.clone());
            Ok(Box::pin(stream::iter([
                Ok(StreamEvent::TextDelta("abcdefgh".into())),
                Ok(StreamEvent::TextDelta("-extra".into())),
                Ok(StreamEvent::ToolCallComplete(ToolCall {
                    id: "call".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"command": "echo abcdefgh-extra"}),
                })),
                Ok(StreamEvent::OpaqueState {
                    provider: "recording".into(),
                    data: serde_json::json!({"state": "abcdefgh-extra"}),
                }),
                Ok(StreamEvent::OpaqueState {
                    provider: "recording".into(),
                    data: serde_json::json!({"state": "safe"}),
                }),
                Ok(StreamEvent::Done {
                    usage: None,
                    stop_reason: None,
                }),
            ])))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn provider_masks_requests_and_split_stream_content() {
        let recorded = Arc::new(Mutex::new(None));
        let provider = mask_provider(
            Arc::new(RecordingProvider {
                request: recorded.clone(),
            }),
            Arc::new(masker()),
        );
        let request = CompletionRequest {
            model: "demo".into(),
            system: Some("system abcdefgh-extra".into()),
            messages: vec![
                Message::user("user abcdefgh-extra"),
                Message::assistant(vec![
                    Content::Opaque {
                        provider: "recording".into(),
                        data: serde_json::json!({"state": "abcdefgh-extra"}),
                    },
                    Content::Opaque {
                        provider: "recording".into(),
                        data: serde_json::json!({"state": "safe"}),
                    },
                ]),
            ],
            tools: Vec::new(),
            max_tokens: None,
            temperature: None,
            reasoning: llm::ReasoningPolicy::Auto,
            session_id: None,
        };

        let mut response = provider.stream(&request).await.unwrap();
        let mut text = String::new();
        let mut tool_call = None;
        let mut opaque = Vec::new();
        while let Some(event) = response.next().await {
            match event.unwrap() {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::ToolCallComplete(call) => tool_call = Some(call),
                StreamEvent::OpaqueState { data, .. } => opaque.push(data),
                _ => {}
            }
        }

        let sent = recorded.lock().unwrap().clone().unwrap();
        assert!(sent.system.unwrap().contains("{{harness-secret:LONG}}"));
        assert!(matches!(
            &sent.messages[0].content[0],
            Content::Text(text) if text.contains("{{harness-secret:LONG}}")
        ));
        assert_eq!(
            sent.messages[1].content,
            vec![Content::Opaque {
                provider: "recording".into(),
                data: serde_json::json!({"state": "safe"}),
            }]
        );
        assert_eq!(opaque, vec![serde_json::json!({"state": "safe"})]);
        assert_eq!(text, "{{harness-secret:LONG}}");
        assert_eq!(
            tool_call.unwrap().arguments["command"],
            "echo {{harness-secret:LONG}}"
        );
    }
}
