pub mod anthropic;
pub mod openai_chat;
pub mod openai_codex_responses;
pub mod openai_responses;

pub use anthropic::AnthropicMessagesClient;
pub use openai_chat::OpenAiChatClient;
pub use openai_responses::OpenAiResponsesClient;

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
