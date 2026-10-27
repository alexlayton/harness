//! LLM summarization with a deterministic fallback.
//!
//! The summarizer builds a structured context checkpoint from the events the
//! planner selected, calling the *same* provider/model as the conversation
//! (v1 decision: no separate summarizer model). Compaction must never
//! dead-end: on any provider/stream error — or no output — we fall back to the
//! deterministic summarizer.

use crate::plan::CompactionPlan;
use crate::policy::CompactionPolicy;
use crate::policy::DEFAULT_TOOL_RESULT_CHARS;
use crate::serialize::{
    OMISSION_MARKER, extract_file_operations, format_file_operations, serialize_events,
    truncate_bytes,
};
use futures_util::StreamExt;
use llm::{CompletionRequest, Message, Provider, ReasoningPolicy, StreamEvent, Usage};
use tokio_util::sync::CancellationToken;

/// System prompt: marks the task as summarization and forbids continuing the
/// conversation.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
You are a context summarization assistant. Your task is to read a conversation \
between a user and an AI assistant, then produce a structured summary following \
the exact format below.

Do NOT continue the conversation. Do NOT respond to any questions in the \
conversation. ONLY output the structured summary.";

/// The structured summary format requested from the model.
pub const SUMMARIZATION_PROMPT: &str = "\
The text above is a session transcript to summarize. Create a structured \
context checkpoint that another model can use to continue the work. Use this \
EXACT format:

## Goal
[What is the user trying to accomplish? Multiple items if the session covers \
different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements the user mentioned]
- [Or \"(none)\" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks / changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [brief rationale]

## Next Steps
1. [ordered list of what should happen next]

## Critical Context
- [Data, examples, code/function names, exact file paths, or error messages \
needed to continue]
- [Or \"(none)\" if not applicable]

Keep each section concise. Preserve exact file paths, function names, command \
invocations, and error messages verbatim where they matter.";

/// The result of a summarization attempt.
#[derive(Clone, Debug, PartialEq)]
pub enum SummaryOutcome {
    /// Successful LLM summary plus optional provider usage (recorded by the agent).
    Model { text: String, usage: Option<Usage> },
    /// Deterministic fallback. `usage` is set when the summarizer request
    /// ran (and was billed) but its answer could not be used, for example
    /// when it was empty.
    Deterministic { text: String, usage: Option<Usage> },
    /// The caller cancelled compaction before a summary was completed.
    /// Cancellation is not a summarizer failure and must not be persisted.
    Cancelled,
}

/// Generate a summary for `plan` using the conversation's provider/model with
/// a deterministic fallback on any failure. `cancel` aborts the summarizer
/// request (the caller then persists nothing, so no half-written state).
/// `session_id` scopes the request to the conversation being summarized so
/// providers with per-conversation accounting bill it to the right place;
/// `None` leaves the request unscoped.
pub async fn summarize(
    provider: &dyn Provider,
    model: &str,
    plan: &CompactionPlan,
    policy: &CompactionPolicy,
    session_id: Option<&str>,
    cancel: &CancellationToken,
) -> SummaryOutcome {
    if cancel.is_cancelled() {
        return SummaryOutcome::Cancelled;
    }
    match model_summarize(provider, model, plan, policy, session_id, cancel).await {
        Ok((text, usage)) => {
            if cancel.is_cancelled() {
                SummaryOutcome::Cancelled
            } else {
                SummaryOutcome::Model {
                    text: append_file_lists(text, plan, policy.max_summary_bytes),
                    usage,
                }
            }
        }
        Err(_) if cancel.is_cancelled() => SummaryOutcome::Cancelled,
        Err((error, usage)) => {
            tracing::warn!(error = %error, "LLM summarization failed; using deterministic fallback");
            SummaryOutcome::Deterministic {
                text: deterministic_summary(plan, policy),
                usage,
            }
        }
    }
}

/// Instruction added when an earlier summary exists, so its still-relevant
/// content is carried forward instead of replaced by the new span alone.
const MERGE_PREVIOUS_PROMPT: &str = "\
The <previous-summary> above describes earlier work in this session. Merge \
it into the new checkpoint: keep its goals, decisions, progress, and \
critical context that still apply, update items the new conversation \
changed, and do not drop earlier information only because it is not \
repeated in the new conversation.\n\n";

/// One-shot LLM summarization. Errors and cancellation both return `Err`, so
/// the caller can fall back without distinguishing them. The error carries
/// the usage of a request that completed but whose answer was unusable.
async fn model_summarize(
    provider: &dyn Provider,
    model: &str,
    plan: &CompactionPlan,
    policy: &CompactionPolicy,
    session_id: Option<&str>,
    cancel: &CancellationToken,
) -> Result<(String, Option<Usage>), (llm::LlmError, Option<Usage>)> {
    let serialized = serialize_events(
        &plan.to_summarize,
        policy.max_summary_input_bytes,
        DEFAULT_TOOL_RESULT_CHARS,
    );
    let mut prompt = String::from("<conversation>\n");
    prompt.push_str(if serialized.text.is_empty() {
        "(no prior conversation material)"
    } else {
        &serialized.text
    });
    if serialized.truncated && policy.max_summary_input_bytes > OMISSION_MARKER.len() {
        prompt.push('\n');
        prompt.push_str(OMISSION_MARKER);
    }
    prompt.push_str("\n</conversation>\n\n");
    if let Some(previous) = &plan.previous_summary {
        prompt.push_str("<previous-summary>\n");
        prompt.push_str(previous);
        prompt.push_str("\n</previous-summary>\n\n");
        prompt.push_str(MERGE_PREVIOUS_PROMPT);
    }
    prompt.push_str(SUMMARIZATION_PROMPT);

    let request = CompletionRequest {
        model: model.to_owned(),
        system: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![Message::user(prompt)],
        tools: Vec::new(),
        max_tokens: Some(summary_max_tokens(policy)),
        temperature: None,
        reasoning: ReasoningPolicy::Off,
        session_id: session_id.map(str::to_owned),
    };

    // Same initial-request retry policy as conversation turns: one transient
    // 429/5xx must not replace a model summary with the fallback.
    let on_retry: llm::RetryCallback = std::sync::Arc::new(|attempt, error| {
        tracing::warn!(attempt, error = %error, "retrying summarizer request");
    });
    let mut stream = tokio::select! {
        result = provider.stream_with_retry(&request, on_retry) => result.map_err(|error| (error, None))?,
        _ = cancel.cancelled() => {
            return Err((llm::LlmError::Stream("summarization cancelled".into()), None));
        }
    };
    let mut text = String::new();
    let mut usage = None;
    loop {
        tokio::select! {
            next = stream.next() => {
                let Some(next) = next else { break };
                match next {
                    Ok(StreamEvent::TextDelta(delta)) => {
                        append_with_limit(&mut text, &delta, policy.max_summary_bytes);
                    }
                    Ok(StreamEvent::Done { usage: found, .. }) => {
                        usage = found;
                        // Done is the protocol boundary; do not wait for an
                        // HTTP stream that may remain open after completion.
                        break;
                    }
                    Ok(StreamEvent::ReasoningDelta(_))
                    | Ok(StreamEvent::OpaqueState { .. })
                    | Ok(StreamEvent::ToolCallComplete(_)) => {}
                    Err(error) => return Err((error, usage)),
                }
            }
            _ = cancel.cancelled() => {
                return Err((llm::LlmError::Stream("summarization cancelled".into()), usage));
            }
        }
    }
    if text.trim().is_empty() {
        // The request completed and was billed; keep its usage.
        return Err((
            llm::LlmError::Stream("summarizer produced no output".into()),
            usage,
        ));
    }
    Ok((text, usage))
}

/// Append a provider delta without allowing a long or malicious stream to
/// defeat the configured final-summary memory budget.
fn append_with_limit(target: &mut String, delta: &str, max_bytes: usize) {
    if target.len() >= max_bytes {
        return;
    }
    let remaining = max_bytes - target.len();
    target.push_str(truncate_bytes(delta, remaining));
}

/// A capped `max_tokens` budget for the summarizer: large enough for the full
/// structured output, small enough not to eat the reserve.
fn summary_max_tokens(policy: &CompactionPolicy) -> u32 {
    let derived = (policy.max_summary_bytes as u32) / 4;
    derived.clamp(512, 4096)
}

/// Append the deterministic `<files-read>` / `<files-modified>` sections to a
/// generated summary.
fn append_file_lists(summary: String, plan: &CompactionPlan, max_summary_bytes: usize) -> String {
    let operations = extract_file_operations(&plan.to_summarize);
    let lists = format_file_operations(&operations);
    if lists.is_empty() {
        return truncate_bytes(&summary, max_summary_bytes).to_owned();
    }
    let combined = format!("{summary}{lists}");
    truncate_bytes(&combined, max_summary_bytes).to_owned()
}

/// Deterministic fallback, capped at `max_summary_bytes`.
///
/// The newest material wins. File lists may use a quarter of the budget,
/// and the previous summary at most half of what remains (keeping its newest
/// part); the new span gets the rest, newest events first, and any unused
/// space goes back to the previous summary. Before, the previous summary was
/// placed first and the result was cut from the end, so repeated fallbacks
/// dropped every new span. Never splits a UTF-8 code point.
fn deterministic_summary(plan: &CompactionPlan, policy: &CompactionPolicy) -> String {
    const HEADER: &str = "This is generated context, not a verbatim transcript.\n";
    const PREVIOUS_OPEN: &str = "<previous-generated-summary>\n";
    const PREVIOUS_CLOSE: &str = "\n</previous-generated-summary>\n";
    const EARLIER_OMITTED: &str = "[... earlier summary omitted ...]\n";
    const NO_MATERIAL: &str = "(no conversation material)";

    let budget = policy.max_summary_bytes;
    let files = format_file_operations(&extract_file_operations(&plan.to_summarize));
    let files = truncate_bytes(&files, budget / 4);
    let mut available = budget.saturating_sub(HEADER.len() + files.len());

    let previous = plan
        .previous_summary
        .as_deref()
        .filter(|previous| !previous.is_empty());
    let previous_overhead = PREVIOUS_OPEN.len() + PREVIOUS_CLOSE.len() + EARLIER_OMITTED.len();
    let previous_reserve = previous.map_or(0, |previous| {
        (previous.len() + previous_overhead).min(available / 2)
    });
    let transcript_budget = available
        .saturating_sub(previous_reserve)
        .min(policy.max_summary_input_bytes);
    let serialized = serialize_events(
        &plan.to_summarize,
        transcript_budget,
        DEFAULT_TOOL_RESULT_CHARS,
    );
    // `serialize_events` reserves room for "\n" + marker inside its budget.
    let marker = serialized.truncated && transcript_budget > OMISSION_MARKER.len();
    let transcript_len = if serialized.text.is_empty() {
        NO_MATERIAL.len()
    } else {
        serialized.text.len()
    } + if marker { OMISSION_MARKER.len() + 1 } else { 0 };
    available = available.saturating_sub(transcript_len);

    let mut text = String::from(HEADER);
    if let Some(previous) = previous {
        let room = available.saturating_sub(previous_overhead);
        if room > 0 {
            text.push_str(PREVIOUS_OPEN);
            if previous.len() > room {
                text.push_str(EARLIER_OMITTED);
                text.push_str(tail_bytes(previous, room));
            } else {
                text.push_str(previous);
            }
            text.push_str(PREVIOUS_CLOSE);
        }
    }
    if serialized.text.is_empty() {
        text.push_str(NO_MATERIAL);
    } else {
        text.push_str(&serialized.text);
    }
    if marker {
        text.push('\n');
        text.push_str(OMISSION_MARKER);
    }
    text.push_str(files);
    // Tiny budgets cannot hold the fixed parts; stay a hard limit anyway.
    truncate_bytes(&text, budget).to_owned()
}

/// Return the longest UTF-8 suffix that fits within `max_bytes`.
fn tail_bytes(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan_compaction;
    use crate::policy::CompactionPolicy;
    use async_trait::async_trait;
    use futures_util::stream;
    use llm::{EventStream, ModelInfo};
    use serde_json::json;
    use session::model::{Session, SessionEvent, SessionMetadata, StoredMessage, StoredToolCall};

    fn push_user(session: &mut Session, text: &str) {
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&Message::user(text)),
        });
    }

    fn push_assistant(session: &mut Session, text: &str) {
        session.append(SessionEvent::AssistantMessage {
            message: StoredMessage::from_llm(&Message::assistant(vec![llm::Content::Text(
                text.into(),
            )])),
        });
    }

    fn build_plan() -> CompactionPlan {
        let mut session = Session::new(SessionMetadata::new("/tmp/project", None, None));
        for index in 0..10 {
            push_user(&mut session, &format!("question {index}"));
            push_assistant(&mut session, &"a".repeat(4_000));
            session.append(SessionEvent::ToolCall {
                call: StoredToolCall {
                    id: format!("call-{index}"),
                    name: "read".into(),
                    arguments: json!({ "path": "src/lib.rs" }),
                },
            });
            session.append(SessionEvent::ToolResult {
                tool_call_id: format!("call-{index}"),
                content: "b".repeat(4_000),
                is_error: false,
                tool_name: Some("read".into()),
            });
        }
        let policy = CompactionPolicy {
            keep_recent_turns: 4,
            keep_recent_tokens: 20_000,
            ..CompactionPolicy::default()
        };
        plan_compaction(&session, &policy, 500_000).unwrap()
    }

    struct ScriptProvider {
        events: Vec<Result<StreamEvent, String>>,
    }

    #[async_trait]
    impl Provider for ScriptProvider {
        fn name(&self) -> &str {
            "script"
        }
        async fn stream(&self, _req: &CompletionRequest) -> Result<EventStream, llm::LlmError> {
            let events = self.events.clone();
            Ok(Box::pin(stream::iter(
                events
                    .into_iter()
                    .map(|event| event.map_err(llm::LlmError::Stream)),
            )))
        }
        async fn list_models(&self) -> Result<Vec<ModelInfo>, llm::LlmError> {
            Ok(Vec::new())
        }
    }

    struct HangingSummaryProvider;

    #[async_trait]
    impl Provider for HangingSummaryProvider {
        fn name(&self) -> &str {
            "hanging-summary"
        }

        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, llm::LlmError> {
            std::future::pending::<()>().await;
            unreachable!("the provider stream should be cancelled first")
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, llm::LlmError> {
            Ok(Vec::new())
        }
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn model_summary_returns_text_and_usage() {
        let runtime = runtime();
        let plan = build_plan();
        let provider = ScriptProvider {
            events: vec![
                Ok(StreamEvent::TextDelta("## Goal\nFix bugs".into())),
                Ok(StreamEvent::TextDelta("\n## Progress".into())),
                Ok(StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(Usage {
                        input_tokens: 100,
                        output_tokens: 20,
                        ..Usage::default()
                    }),
                }),
            ],
        };
        let cancel = CancellationToken::new();
        let outcome = runtime.block_on(summarize(
            &provider,
            "demo",
            &plan,
            &CompactionPolicy::default(),
            None,
            &cancel,
        ));
        match outcome {
            SummaryOutcome::Model { text, usage } => {
                assert!(text.contains("## Goal"));
                assert_eq!(usage.unwrap().output_tokens, 20);
                // File lists derived from the read tool calls are appended.
                assert!(text.contains("<files-read>"));
                assert!(text.contains("src/lib.rs"));
            }
            other => panic!("expected model outcome, got {other:?}"),
        }
    }

    #[test]
    fn done_finishes_summary_without_waiting_for_stream_end() {
        let provider = ScriptProvider {
            events: vec![
                Ok(StreamEvent::TextDelta("Summary is complete".into())),
                Ok(StreamEvent::Done {
                    stop_reason: None,
                    usage: Some(Usage {
                        output_tokens: 7,
                        ..Usage::default()
                    }),
                }),
                Err("transport failed after Done".into()),
            ],
        };
        let outcome = runtime().block_on(summarize(
            &provider,
            "demo",
            &build_plan(),
            &CompactionPolicy::default(),
            None,
            &CancellationToken::new(),
        ));
        assert!(
            matches!(outcome, SummaryOutcome::Model { usage: Some(usage), .. } if usage.output_tokens == 7)
        );
    }

    #[test]
    fn final_summary_budget_includes_file_operation_lists() {
        let plan = build_plan();
        let policy = CompactionPolicy {
            max_summary_bytes: 128,
            ..CompactionPolicy::default()
        };
        let summary = deterministic_summary(&plan, &policy);
        assert!(summary.len() <= policy.max_summary_bytes);
    }

    #[test]
    fn repeated_fallbacks_keep_the_newest_span() {
        // A previous summary as large as the budget used to fill it, so the
        // new span was cut away on every fallback.
        let mut plan = build_plan();
        plan.previous_summary = Some("p".repeat(8 * 1024));
        let policy = CompactionPolicy {
            max_summary_bytes: 2 * 1024,
            ..CompactionPolicy::default()
        };
        let last = &plan.to_summarize.last().unwrap().event;
        let newest = crate::serialize::serialize_record(last, DEFAULT_TOOL_RESULT_CHARS);
        let newest = truncate_bytes(&newest, 40).to_owned();
        let summary = deterministic_summary(&plan, &policy);
        assert!(summary.len() <= policy.max_summary_bytes);
        assert!(summary.contains(&newest), "{summary}");
        assert!(summary.contains("<previous-generated-summary>"));
    }

    #[test]
    fn stream_error_falls_back_to_deterministic() {
        let runtime = runtime();
        let plan = build_plan();
        let provider = ScriptProvider {
            events: vec![Err("boom".into())],
        };
        let cancel = CancellationToken::new();
        let outcome = runtime.block_on(summarize(
            &provider,
            "demo",
            &plan,
            &CompactionPolicy::default(),
            None,
            &cancel,
        ));
        match outcome {
            SummaryOutcome::Deterministic { text, .. } => {
                assert!(text.contains("generated context"));
                assert!(text.contains("[User]: question"));
            }
            other => panic!("expected deterministic outcome, got {other:?}"),
        }
    }

    #[test]
    fn repeated_compaction_fallback_keeps_earlier_summary_in_provider_history() {
        let runtime = runtime();
        let policy = CompactionPolicy {
            keep_recent_turns: 2,
            max_summary_bytes: 512,
            ..CompactionPolicy::default()
        };
        let mut session = Session::new(SessionMetadata::new("/tmp/project", None, None));
        for index in 0..4 {
            push_user(&mut session, &format!("question {index}"));
            push_assistant(&mut session, "reply");
        }
        let first = plan_compaction(&session, &policy, 100_000).unwrap();
        let cancel = CancellationToken::new();
        let initial = ScriptProvider {
            events: vec![Ok(StreamEvent::TextDelta(
                "unique earlier fact: café".into(),
            ))],
        };
        let SummaryOutcome::Model { text, .. } =
            runtime.block_on(summarize(&initial, "demo", &first, &policy, None, &cancel))
        else {
            panic!("first summary must use the model");
        };
        session.append(SessionEvent::CompactionSummary {
            summary: text,
            compacted_through: first.boundary,
        });
        for index in 4..7 {
            push_user(&mut session, &format!("question {index}"));
            push_assistant(&mut session, "reply");
        }
        let second = plan_compaction(&session, &policy, 100_000).unwrap();
        let failure = ScriptProvider {
            events: vec![Err("offline".into())],
        };
        let SummaryOutcome::Deterministic { text, .. } =
            runtime.block_on(summarize(&failure, "demo", &second, &policy, None, &cancel))
        else {
            panic!("second summary must fall back");
        };
        assert!(text.len() <= policy.max_summary_bytes);
        session.append(SessionEvent::CompactionSummary {
            summary: text,
            compacted_through: second.boundary,
        });
        let history = session.context_messages();
        assert!(format!("{history:?}").contains("unique earlier fact: café"));
    }

    #[test]
    fn cancellation_interrupts_provider_stream_acquisition() {
        let runtime = runtime();
        let plan = build_plan();
        let provider = HangingSummaryProvider;
        let policy = CompactionPolicy::default();
        let cancel = CancellationToken::new();
        let mut summary = Box::pin(summarize(&provider, "demo", &plan, &policy, None, &cancel));
        let outcome = runtime.block_on(async {
            tokio::select! {
                outcome = &mut summary => panic!("summary completed unexpectedly: {outcome:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => cancel.cancel(),
            }
            summary.await
        });
        assert_eq!(outcome, SummaryOutcome::Cancelled);
    }

    #[test]
    fn cancelled_summarization_does_not_fall_back_or_emit_a_summary() {
        let runtime = runtime();
        let plan = build_plan();
        let provider = ScriptProvider { events: Vec::new() };
        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = runtime.block_on(summarize(
            &provider,
            "demo",
            &plan,
            &CompactionPolicy::default(),
            None,
            &cancel,
        ));
        assert_eq!(outcome, SummaryOutcome::Cancelled);
    }

    #[test]
    fn empty_model_output_falls_back_to_deterministic() {
        let runtime = runtime();
        let plan = build_plan();
        let provider = ScriptProvider {
            events: vec![Ok(StreamEvent::Done {
                stop_reason: Some("stop".into()),
                usage: Some(Usage {
                    output_tokens: 3,
                    ..Usage::default()
                }),
            })],
        };
        let cancel = CancellationToken::new();
        let outcome = runtime.block_on(summarize(
            &provider,
            "demo",
            &plan,
            &CompactionPolicy::default(),
            None,
            &cancel,
        ));
        // The empty answer still cost tokens; the usage must survive.
        assert!(matches!(
            outcome,
            SummaryOutcome::Deterministic { usage: Some(usage), .. } if usage.output_tokens == 3
        ));
    }
}
