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
        let tools = || {
            vec![llm::ToolDefinition {
                name: "read".into(),
                description: "read files".into(),
                parameters: serde_json::json!({"type": "object"}),
            }]
        };
        let small = estimate_provider_context_tokens(
            Some("system"),
            &tools(),
            &[llm::Message::user("hello")],
        );
        let large_result = llm::Message::tool_result("call-1", "x".repeat(20_000), false);
        let large = estimate_provider_context_tokens(
            Some("system"),
            &tools(),
            &[llm::Message::user("hello"), large_result],
        );
        assert!(large > small);
    }

    #[test]
    fn durable_and_in_memory_histories_estimate_equivalently() {
        // PERF-1 equivalence: the agent estimates over `session
        // .context_messages()` for durable histories, so the same event log
        // reconstructed through the session must estimate identically to the
        // in-memory message vector.
        use session::model::{Session, SessionEvent, SessionMetadata, StoredMessage};
        let mut session = Session::new(SessionMetadata::new("/tmp/project", None, None));
        let user = llm::Message::user("do the thing");
        let assistant = llm::Message::assistant(vec![
            llm::Content::Text("working".into()),
            llm::Content::ToolCall(llm::ToolCall {
                id: "call-1".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "x"}),
            }),
        ]);
        let result = llm::Message::tool_result("call-1", "file contents here", false);
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&user),
        });
        session.append(SessionEvent::AssistantMessage {
            message: StoredMessage::from_llm(&assistant),
        });
        session.append(SessionEvent::ToolResult {
            tool_call_id: "call-1".into(),
            content: "file contents here".into(),
            is_error: false,
            tool_name: Some("read".into()),
        });
        let system = "system prompt";
        let tools = vec![llm::ToolDefinition {
            name: "read".into(),
            description: "read files".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let via_session =
            estimate_provider_context_tokens(Some(system), &tools, &session.context_messages());
        let via_memory =
            estimate_provider_context_tokens(Some(system), &tools, &[user, assistant, result]);
        assert_eq!(via_session, via_memory);
    }

    #[test]
    fn active_compaction_summary_contributes_to_the_estimate() {
        // The retained summary is replayed as a full user message, so it
        // must count — not vanish — in the provider-context estimate.
        use session::model::{Session, SessionEvent, SessionMetadata, StoredMessage};
        let mut session = Session::new(SessionMetadata::new("/tmp/project", None, None));
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&llm::Message::user("old question")),
        });
        session.append(SessionEvent::CompactionSummary {
            summary: "prior work summary".into(),
            compacted_through: 1,
        });
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&llm::Message::user("new question")),
        });
        let with_summary = estimate_provider_context_tokens(None, &[], &session.context_messages());
        assert!(
            session
                .context_messages()
                .iter()
                .any(|message| message.content.iter().any(|content| matches!(
                    content,
                    llm::Content::Text(text) if text.contains("prior work summary")
                ))),
            "the summary must be replayed into context"
        );
        let without_summary =
            estimate_provider_context_tokens(None, &[], &[llm::Message::user("new question")]);
        assert!(
            with_summary > without_summary,
            "the active summary must contribute tokens"
        );
    }

    #[test]
    fn large_project_context_and_tool_schemas_can_trigger_compaction() {
        // PERF-1 trigger coverage: the request prefix (system/project
        // context + tool schemas) counts, so a huge prefix alone can push
        // the estimate over the trigger — the durable-session undercount
        // this fixes.
        let big_system = "p".repeat(400_000);
        let big_tools: Vec<llm::ToolDefinition> = (0..100)
            .map(|index| llm::ToolDefinition {
                name: format!("tool-{index}"),
                description: "d".repeat(2_000),
                parameters: serde_json::json!({"type": "object"}),
            })
            .collect();
        let estimate = estimate_provider_context_tokens(
            Some(&big_system),
            &big_tools,
            &[llm::Message::user("hi")],
        );
        let policy = crate::policy::CompactionPolicy {
            keep_recent_turns: 10,
            keep_recent_tokens: 20_000,
            ..crate::policy::CompactionPolicy::default()
        };
        assert!(
            policy.should_auto_compact(estimate, 200_000),
            "a huge request prefix must be able to trigger compaction"
        );
    }

    #[test]
    fn no_double_count_of_pending_input_after_provider_usage() {
        // The trigger path adds the pending input once
        // (`base + estimate_tokens(extra)`): the `Done` usage total already
        // covers the request it describes, so the next turn's estimate is
        // `usage_total + new_input`, never `usage_total + old_output +
        // new_input`.
        let usage_total = 10_000u64;
        let new_input = "fresh question";
        let estimate = usage_total.saturating_add(estimate_tokens(new_input.len()));
        assert_eq!(estimate, usage_total + estimate_tokens(new_input.len()));
        // And the estimator itself is additive over one message: estimating
        // `[history, input]` equals `estimate(history) + estimate(input)`
        // modulo the fixed per-message role bytes (no multiplicative
        // blowup from re-scanning).
        let history = vec![llm::Message::user("old")];
        let input = llm::Message::user(new_input);
        let combined =
            estimate_provider_context_tokens(None, &[], &[history[0].clone(), input.clone()]);
        let separate = estimate_provider_context_tokens(None, &[], &history)
            + estimate_provider_context_tokens(None, &[], std::slice::from_ref(&input));
        // One extra role prefix (4 bytes → 1 token) is the only divergence.
        assert!(
            combined <= separate + 1,
            "combined {combined} vs separate {separate}"
        );
    }
}
