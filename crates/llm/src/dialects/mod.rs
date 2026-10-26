pub mod anthropic;
pub mod anthropic_subscription;
pub mod openai_chat;
pub mod openai_codex_responses;
pub mod openai_responses;

pub use anthropic::AnthropicMessagesClient;
pub use openai_chat::OpenAiChatClient;
pub use openai_responses::OpenAiResponsesClient;

/// Largest total SSE payload accepted for one response (64 MiB). Far above
/// any real model answer; it only stops runaway or hostile streams.
const MAX_STREAM_BYTES: usize = 64 * 1024 * 1024;

/// One SSE transport loop shared by every dialect adapter (CLEANUP-3).
///
/// The four `event_stream` adapters duplicated this exact shape and diverged
/// only in parser type, the done predicate, and (for Codex) an opaque-state
/// side channel. `Parser` captures the common contract so transport
/// adaptation stays shared while payload parsing remains per-dialect: each
/// adapter implements `parse_event`/`is_done`/`finish` on its own parser
/// and delegates the loop here.
pub(crate) trait StreamParser {
    fn parse_event(
        &mut self,
        event: &crate::sse::SseEvent,
    ) -> Result<Vec<crate::StreamEvent>, crate::LlmError>;
    fn is_done(&self) -> bool;
    fn finish(&mut self) -> Result<Vec<crate::StreamEvent>, crate::LlmError>;
}

/// Drive any [`StreamParser`] over a shared [`SseStream`], yielding parsed
/// events until the parser reports its protocol terminal (or EOF without
/// one, which the parser surfaces as `LlmError::StreamInterrupted`).  The returned
/// boundary redacts the active credential from both transport and parser
/// errors, which are produced after the async request has already returned.
pub(crate) fn drive_parser_stream(
    mut sse: crate::sse::SseStream,
    mut parser: impl StreamParser + Send + 'static,
    secret: &str,
) -> crate::EventStream {
    use futures_util::StreamExt;
    let stream: crate::EventStream = Box::pin(async_stream::try_stream! {
        let mut received = 0usize;
        while let Some(event) = sse.next().await {
            let event = event?;
            // Each SSE event is bounded, but parsers retain state across
            // events (tool arguments, pending calls, opaque items). Bound the
            // whole response so many small events cannot grow it without
            // limit.
            received = received.saturating_add(event.data.len());
            if received > MAX_STREAM_BYTES {
                Err(crate::LlmError::Stream(format!(
                    "response stream exceeds {MAX_STREAM_BYTES} bytes"
                )))?;
            }
            for item in parser.parse_event(&event)? {
                yield item;
            }
            if parser.is_done() {
                break;
            }
        }
        if !parser.is_done() {
            for item in parser.finish()? {
                yield item;
            }
        }
    });
    crate::provider::redact_stream(stream, secret)
}

/// Serialize tool-call arguments into the JSON string the OpenAI-style wire
/// formats expect. Serializing a `Value` cannot realistically fail; the `{}`
/// fallback only keeps the request well-formed so one odd value cannot abort
/// the whole turn.
pub(crate) fn stringify_arguments(arguments: &serde_json::Value) -> String {
    serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into())
}

/// Human-readable message carried by an error value that is either a bare
/// string or an object with a string `message` field.
pub(crate) fn value_message(value: &serde_json::Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("message").and_then(serde_json::Value::as_str))
}

/// Best human-readable message for an in-band error event.
///
/// `errors` are the candidate error values in priority order (dialects nest
/// the error object at different depths, as for [`in_band_error`]). When
/// none of them carries a message, the event's own top-level `message` and
/// then `fallback` are used, so the user always sees some reason.
pub(crate) fn error_message(
    errors: &[Option<&serde_json::Value>],
    event: &serde_json::Value,
    fallback: &str,
) -> String {
    errors
        .iter()
        .flatten()
        .find_map(|error| value_message(error))
        .or_else(|| event.get("message").and_then(serde_json::Value::as_str))
        .unwrap_or(fallback)
        .to_owned()
}

/// Build the error for a clean transport EOF that arrived before the
/// dialect's protocol terminal event.  The answer is incomplete, but a new
/// request can succeed, so it is a transient [`crate::LlmError::StreamInterrupted`].
pub(crate) fn truncated_stream(message: &str) -> crate::LlmError {
    crate::LlmError::StreamInterrupted(message.to_owned())
}

/// Classify an in-band provider error payload.  Overload and server faults
/// are transient (a new request can succeed) and become
/// [`crate::LlmError::StreamInterrupted`]; all other errors, such as invalid
/// requests or context overflow, stay a permanent [`crate::LlmError::Stream`]
/// so they are reported and not repeated.
///
/// `payloads` are the JSON objects that can carry a `type` or `code` field
/// (dialects nest the error object at different depths).
pub(crate) fn in_band_error(
    payloads: &[Option<&serde_json::Value>],
    message: String,
) -> crate::LlmError {
    if payloads
        .iter()
        .flatten()
        .any(|payload| is_transient_error_payload(payload))
    {
        crate::LlmError::StreamInterrupted(message)
    } else {
        crate::LlmError::Stream(message)
    }
}

fn is_transient_error_payload(payload: &serde_json::Value) -> bool {
    use serde_json::Value;
    const TRANSIENT: &[&str] = &[
        "overloaded_error",
        "overloaded",
        "api_error",
        "server_error",
        "internal_error",
        "internal_server_error",
        "service_unavailable",
        "timeout",
        "timeout_error",
    ];
    ["type", "code"].iter().any(|key| match payload.get(key) {
        Some(Value::String(code)) => TRANSIENT.contains(&code.as_str()),
        Some(Value::Number(code)) => code
            .as_u64()
            .is_some_and(|code| (500..=599).contains(&code)),
        _ => false,
    })
}

/// OpenAI-compatible wire spelling for a portable explicit effort.
pub(crate) const fn openai_reasoning_effort(
    reasoning: crate::ReasoningPolicy,
) -> Option<&'static str> {
    match reasoning {
        crate::ReasoningPolicy::Effort(crate::ReasoningEffort::Minimal) => Some("minimal"),
        crate::ReasoningPolicy::Effort(crate::ReasoningEffort::Low) => Some("low"),
        crate::ReasoningPolicy::Effort(crate::ReasoningEffort::Medium) => Some("medium"),
        crate::ReasoningPolicy::Effort(crate::ReasoningEffort::High) => Some("high"),
        crate::ReasoningPolicy::Effort(crate::ReasoningEffort::Maximum) => Some("xhigh"),
        crate::ReasoningPolicy::Auto | crate::ReasoningPolicy::Off => None,
    }
}
