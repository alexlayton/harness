//! Token estimation for cut-point selection.
//!
//! Precision is deliberately irrelevant here: the estimator only *selects a
//! cut point with slack*, so a ~4 bytes/token heuristic is more than enough.
//! The compaction *trigger* uses exact provider-reported numbers
//! (`usage.input_tokens` + `output_tokens`) when available, and this same
//! provider-context estimate for both durable and in-memory histories before
//! the first usage report.

/// Rough bytes per token. UTF-8 safe in the sense that we count bytes, never
/// split characters.
pub const BYTES_PER_TOKEN: u64 = 4;

/// Estimate the token count of a byte payload.
pub fn estimate_tokens(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(BYTES_PER_TOKEN)
}

/// Estimate the token count of a text payload.
pub fn estimate_text_tokens(text: &str) -> u64 {
    estimate_tokens(text.len())
}

/// Estimate the provider-context size from the inputs used to build a
/// [`llm::CompletionRequest`]. Unlike transcript-only estimates, this counts
/// the system prompt, full tool schemas, compaction summaries, opaque
/// continuation state, and complete tool results.
///
/// This is intentionally provider-neutral and conservative: dialects may
/// omit a field or add wire overhead, but undercounting a large live result is
/// more damaging than a small overestimate when deciding whether to compact.
pub fn estimate_provider_context_tokens(
    system: Option<&str>,
    tools: &[llm::ToolDefinition],
    messages: &[llm::Message],
) -> u64 {
    let mut bytes = system.map_or(0, str::len);
    for tool in tools {
        bytes = bytes
            .saturating_add(tool.name.len())
            .saturating_add(tool.description.len())
            .saturating_add(json_len(&tool.parameters));
    }
    for message in messages {
        bytes = bytes.saturating_add(role_len(&message.role));
        for content in &message.content {
            bytes = bytes.saturating_add(content_len(content));
        }
    }
    estimate_tokens(bytes)
}

fn role_len(role: &llm::Role) -> usize {
    match role {
        llm::Role::System => 6,
        llm::Role::User => 4,
        llm::Role::Assistant => 9,
        llm::Role::Tool => 4,
    }
}

fn content_len(content: &llm::Content) -> usize {
    match content {
        llm::Content::Text(text) => text.len(),
        // Reasoning deltas are local display state; provider dialects do not
        // replay them as ordinary context.
        llm::Content::Reasoning(_) => 0,
        llm::Content::Opaque { provider, data } => provider.len().saturating_add(json_len(data)),
        llm::Content::ToolCall(call) => call
            .id
            .len()
            .saturating_add(call.name.len())
            .saturating_add(json_len(&call.arguments)),
        llm::Content::ToolResult {
            tool_call_id,
            content,
            is_error: _,
        } => tool_call_id.len().saturating_add(content.len()),
    }
}

fn json_len(value: &serde_json::Value) -> usize {
    serde_json::to_string(value).map_or(0, |value| value.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_ceil_of_bytes_over_four() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(4), 1);
        assert_eq!(estimate_tokens(5), 2);
        assert_eq!(estimate_tokens(100), 25);
    }

    #[test]
    fn provider_estimate_includes_request_prefix_and_full_tool_results() {
        let small = estimate_provider_context_tokens(
            Some("system"),
            &[llm::ToolDefinition {
                name: "read".into(),
                description: "read files".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            &[llm::Message::user("hello")],
        );
        let large_result = llm::Message::tool_result("call-1", "x".repeat(20_000), false);
        let large = estimate_provider_context_tokens(
            Some("system"),
            &[llm::ToolDefinition {
                name: "read".into(),
                description: "read files".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            &[llm::Message::user("hello"), large_result],
        );
        assert!(large > small);
    }

    #[test]
    fn multibyte_utf8_counts_bytes_not_chars() {
        // "é" is two bytes; a 4-char string of them is 8 bytes → 2 tokens.
        assert_eq!(estimate_text_tokens("éééé"), 2);
    }
}
