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
use crate::serialize::{extract_file_operations, format_file_operations, serialize_events};
use futures_util::StreamExt;
use llm::{
    CompletionRequest, Message, Provider, ReasoningPolicy, StreamEvent, Usage, truncate_utf8,
};
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
    /// Successful LLM summary plus its provider usage (recorded by the agent).
    Model { text: String, usage: Usage },
    /// Deterministic fallback (no provider usage to record).
    Deterministic { text: String },
}

/// Generate a summary for `plan` using the conversation's provider/model with
/// a deterministic fallback on any failure. `cancel` aborts the summarizer
/// request (the caller then persists nothing, so no half-written state).
pub async fn summarize(
    provider: &dyn Provider,
    model: &str,
    plan: &CompactionPlan,
    policy: &CompactionPolicy,
    cancel: &CancellationToken,
) -> SummaryOutcome {
    match model_summarize(provider, model, plan, policy, cancel).await {
        Ok((text, usage)) => SummaryOutcome::Model {
            text: append_file_lists(text, plan),
            usage,
        },
        Err(error) => {
            tracing::warn!(error = %error, "LLM summarization failed; using deterministic fallback");
            SummaryOutcome::Deterministic {
                text: deterministic_summary(plan, policy),
            }
        }
    }
}

/// One-shot LLM summarization. Errors and cancellation both return `Err`, so
/// the caller can fall back without distinguishing them.
async fn model_summarize(
    provider: &dyn Provider,
    model: &str,
    plan: &CompactionPlan,
    policy: &CompactionPolicy,
    cancel: &CancellationToken,
) -> Result<(String, Usage), llm::LlmError> {
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
    prompt.push_str("\n</conversation>\n\n");
    if let Some(previous) = &plan.previous_summary {
        prompt.push_str("<previous-summary>\n");
        prompt.push_str(previous);
        prompt.push_str("\n</previous-summary>\n\n");
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
    };

    let mut stream = provider.stream(&request).await?;
    let mut text = String::new();
    let mut usage = Usage::default();
    loop {
        tokio::select! {
            next = stream.next() => {
                let Some(next) = next else { break };
                match next {
                    Ok(StreamEvent::TextDelta(delta)) => text.push_str(&delta),
                    Ok(StreamEvent::Done { usage: Some(found), .. }) => usage = found,
                    Ok(StreamEvent::Done { .. })
                    | Ok(StreamEvent::ReasoningDelta(_))
                    | Ok(StreamEvent::OpaqueState { .. })
                    | Ok(StreamEvent::ToolCallComplete(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            _ = cancel.cancelled() => {
                return Err(llm::LlmError::Stream("summarization cancelled".into()));
            }
        }
    }
    if text.trim().is_empty() {
        return Err(llm::LlmError::Stream(
            "summarizer produced no output".into(),
        ));
    }
    let text = truncate_utf8(&text, policy.max_summary_bytes);
    Ok((text, usage))
}

/// A capped `max_tokens` budget for the summarizer: large enough for the full
/// structured output, small enough not to eat the reserve.
fn summary_max_tokens(policy: &CompactionPolicy) -> u32 {
    let derived = (policy.max_summary_bytes as u32) / 4;
    derived.clamp(512, 4096)
}

/// Append the deterministic `<files-read>` / `<files-modified>` sections to a
/// generated summary.
fn append_file_lists(summary: String, plan: &CompactionPlan) -> String {
    let operations = extract_file_operations(&plan.to_summarize);
    let lists = format_file_operations(&operations);
    if lists.is_empty() {
        summary
    } else {
        format!("{summary}{lists}")
    }
}

/// Deterministic fallback: a condensed transcript of the summarized span,
/// capped at `max_summary_bytes`, tagged as non-verbatim context.
fn deterministic_summary(plan: &CompactionPlan, policy: &CompactionPolicy) -> String {
    let serialized = serialize_events(
        &plan.to_summarize,
        policy.max_summary_input_bytes,
        DEFAULT_TOOL_RESULT_CHARS,
    );
    let mut text = String::from("This is generated context, not a verbatim transcript.\n");
    if serialized.text.is_empty() {
        text.push_str("(no conversation material)");
    } else {
        text.push_str(&serialized.text);
    }
    let text = truncate_utf8(&text, policy.max_summary_bytes);
    append_file_lists(text, plan)
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
            &cancel,
        ));
        match outcome {
            SummaryOutcome::Model { text, usage } => {
                assert!(text.contains("## Goal"));
                assert_eq!(usage.output_tokens, 20);
                // File lists derived from the read tool calls are appended.
                assert!(text.contains("<files-read>"));
                assert!(text.contains("src/lib.rs"));
            }
            other => panic!("expected model outcome, got {other:?}"),
        }
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
            &cancel,
        ));
        match outcome {
            SummaryOutcome::Deterministic { text } => {
                assert!(text.contains("generated context"));
                assert!(text.contains("[User]: question"));
            }
            other => panic!("expected deterministic outcome, got {other:?}"),
        }
    }

    #[test]
    fn empty_model_output_falls_back_to_deterministic() {
        let runtime = runtime();
        let plan = build_plan();
        let provider = ScriptProvider {
            events: vec![Ok(StreamEvent::Done {
                stop_reason: Some("stop".into()),
                usage: None,
            })],
        };
        let cancel = CancellationToken::new();
        let outcome = runtime.block_on(summarize(
            &provider,
            "demo",
            &plan,
            &CompactionPolicy::default(),
            &cancel,
        ));
        assert!(matches!(outcome, SummaryOutcome::Deterministic { .. }));
    }
}
