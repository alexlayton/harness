pub mod anthropic;
pub mod openai_chat;
pub mod openai_codex_responses;
pub mod openai_responses;

pub use anthropic::AnthropicMessagesClient;
pub use openai_chat::OpenAiChatClient;
pub use openai_responses::OpenAiResponsesClient;

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
/// one, which the parser surfaces as `LlmError::Stream`).  The returned
/// boundary redacts the active credential from both transport and parser
/// errors, which are produced after the async request has already returned.
pub(crate) fn drive_parser_stream(
    mut sse: crate::sse::SseStream,
    mut parser: impl StreamParser + Send + 'static,
    secret: &str,
) -> crate::EventStream {
    use futures_util::StreamExt;
    let stream: crate::EventStream = Box::pin(async_stream::try_stream! {
        while let Some(event) = sse.next().await {
            let event = event?;
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
