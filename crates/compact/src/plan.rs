//! Cut-point selection over session events (pure).
//!
//! The planner decides *where* to cut a session's event log for compaction.
//! It never mutates anything: it returns a [`CompactionPlan`] describing the
//! boundary and the events to summarize; the agent persists the summary.
//!
//! Keep-recent is turn-count primary (last N user turns complete), with the
//! token cap as a backstop: when the last N turns alone exceed
//! `keep_recent_tokens`, the cut moves to an assistant message mid-turn — a
//! "split turn".

use crate::estimate::estimate_tokens;
use crate::policy::CompactionPolicy;
use crate::serialize::serialize_record;
use session::model::{Session, SessionEvent, SessionEventRecord};
use session::model::{events_after_latest_compaction, latest_compaction_boundary};

/// A completed compaction plan: what to summarize, where the new boundary is,
/// and how much context the summary is expected to free.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionPlan {
    /// Sequence boundary a new `CompactionSummary { compacted_through: … }`
    /// event will carry. Everything at or below this sequence (from the
    /// previous compaction's boundary onward) is represented by the summary.
    pub boundary: u64,
    /// Events between the previous compaction's boundary and `boundary` that
    /// become the summarize input. Folded into the next summary on repeated
    /// compaction (iterative compaction).
    pub to_summarize: Vec<SessionEventRecord>,
    /// The most recent previous summarizer output, if any, threaded into the
    /// summarizer so context survives across compactions.
    pub previous_summary: Option<String>,
    /// Estimated context tokens freed by this compaction (whole live region
    /// minus the kept tail). Informational; the trigger uses exact numbers.
    pub estimated_tokens_freed: u64,
}

/// Plan a compaction for `session` under `policy`, or return `None` when
/// there is nothing meaningful to summarize (first turn, no new events since
/// the last compaction, or the active region is already within the keep
/// budget).
///
/// `estimated_context_tokens` is the agent's current estimate of the full
/// context size (exact last-request usage when available). The planner does
/// not trigger on it — the agent decides *when* — but it is accepted so
/// callers can short-circuit on clear no-pressure sessions and so future
/// token-calibration work has a stable hook.
pub fn plan_compaction(
    session: &Session,
    policy: &CompactionPolicy,
    estimated_context_tokens: u64,
) -> Option<CompactionPlan> {
    // Never bother planning when the whole session clearly fits in the keep
    // budget: there is nothing to compress.
    if estimated_context_tokens > 0 && estimated_context_tokens <= policy.keep_recent_tokens {
        return None;
    }

    let (previous_boundary, previous_summary) = match latest_compaction_boundary(&session.events) {
        Some((sequence, boundary)) => {
            let summary = session
                .events
                .iter()
                .find(|record| record.sequence == sequence)
                .and_then(|record| match &record.event {
                    SessionEvent::CompactionSummary { summary, .. } => Some(summary.clone()),
                    _ => None,
                });
            (boundary, summary)
        }
        None => (0, None),
    };

    // The live region = events kept since the most recent compaction boundary
    // (the summary event itself plus everything after it in the same region).
    let active = events_after_latest_compaction(&session.events);
    let live: Vec<&SessionEventRecord> = active
        .iter()
        .copied()
        .filter(|record| !is_compaction(record))
        .collect();

    let cut = choose_cut(&live, policy)?;

    let cut_record = live[cut];
    let boundary = cut_record.sequence.saturating_sub(1);
    if boundary <= previous_boundary {
        // Cutting here would not advance past the previous compaction — either
        // there is no pressure or there is nothing new to summarize.
        return None;
    }

    let to_summarize: Vec<SessionEventRecord> = session
        .events
        .iter()
        .filter(|record| {
            record.sequence > previous_boundary
                && record.sequence <= boundary
                && !is_compaction(record)
        })
        .cloned()
        .collect();
    if to_summarize.is_empty() {
        return None;
    }

    let estimated_tokens_freed = live_tokens(&live, 0).saturating_sub(live_tokens(&live, cut));

    Some(CompactionPlan {
        boundary,
        to_summarize,
        previous_summary,
        estimated_tokens_freed,
    })
}

fn is_compaction(record: &SessionEventRecord) -> bool {
    matches!(record.event, SessionEvent::CompactionSummary { .. })
}

/// Event kinds that are valid cut points. Cutting before a `UserMessage` is a
/// normal turn boundary; before an `AssistantMessage` is a split turn. A
/// standalone `ToolCall` is also a safe cut because its result follows and is
/// kept. We never cut before a `ToolResult` — that would orphan a tool call
/// from its result and produce invalid provider history.
fn is_cut_point(record: &SessionEventRecord) -> bool {
    matches!(
        record.event,
        SessionEvent::UserMessage { .. }
            | SessionEvent::AssistantMessage { .. }
            | SessionEvent::ToolCall { .. }
    )
}

/// Estimated provider-context tokens represented by a live event.
fn event_tokens(record: &SessionEventRecord, max_tool_result_chars: usize) -> u64 {
    let text = serialize_record(&record.event, max_tool_result_chars);
    if text.is_empty() {
        return 0;
    }
    estimate_tokens(text.len()).saturating_add(4)
}

/// Sum of estimated tokens for `live[from..]`.
fn live_tokens(live: &[&SessionEventRecord], from: usize) -> u64 {
    let max_chars = crate::policy::DEFAULT_TOOL_RESULT_CHARS;
    let mut total = 0u64;
    for record in live.iter().skip(from) {
        total = total.saturating_add(event_tokens(record, max_chars));
    }
    total
}

/// Choose the cut index within `live` under the policy, or `None` when there
/// is no pressure (fewer than `keep_recent_turns` complete turns and the whole
/// region is within the token cap).
fn choose_cut(live: &[&SessionEventRecord], policy: &CompactionPolicy) -> Option<usize> {
    if live.is_empty() {
        return None;
    }
    let max_chars = crate::policy::DEFAULT_TOOL_RESULT_CHARS;

    // Prefix-sum of estimated tokens so suffix totals are O(1).
    let mut suffix = vec![0u64; live.len() + 1];
    for i in (0..live.len()).rev() {
        suffix[i] = suffix[i + 1].saturating_add(event_tokens(live[i], max_chars));
    }
    let cut_points: Vec<usize> = (0..live.len()).filter(|&i| is_cut_point(live[i])).collect();

    // --- Phase 1: turn-count primary -------------------------------------
    // Walk backward from the newest event until we have keep_recent_turns
    // complete user turns. The oldest kept event at that point is a user
    // message, so the cut is a normal turn boundary.
    let mut turn_cut: Option<usize> = None;
    let mut turns = 0usize;
    for i in (0..live.len()).rev() {
        if matches!(live[i].event, SessionEvent::UserMessage { .. }) {
            turns = turns.saturating_add(1);
        }
        if turns >= policy.keep_recent_turns {
            turn_cut = Some(i);
            break;
        }
    }

    if let Some(turn_index) = turn_cut {
        if suffix[turn_index] <= policy.keep_recent_tokens {
            return Some(turn_index);
        }
    } else if suffix[0] <= policy.keep_recent_tokens {
        // Fewer than keep_recent_turns in the whole live region and it all
        // fits the budget → nothing to compact.
        return None;
    }

    // --- Phase 2: token backstop -----------------------------------------
    // The last N turns alone exceed keep_recent_tokens (or there are fewer
    // than N turns but we are already over budget). Keep the newest content
    // up to the token cap, snapping to the nearest valid cut point. This can
    // land mid-turn on an assistant message (split turn).
    let mut accumulated = 0u64;
    let mut cut = *cut_points.first()?;
    for i in (0..live.len()).rev() {
        accumulated = accumulated.saturating_add(suffix[i].saturating_sub(suffix[i + 1]));
        if accumulated >= policy.keep_recent_tokens {
            cut = cut_points
                .iter()
                .copied()
                .find(|&candidate| candidate >= i)
                .unwrap_or(cut);
            break;
        }
    }
    Some(cut)
}

/// Estimated tokens currently occupied by the live conversation region
/// (events after the latest compaction), used by the agent for exactness
/// fallback when no request has provided usage yet.
pub fn estimate_live_tokens(session: &Session) -> u64 {
    let active = events_after_latest_compaction(&session.events);
    let live: Vec<&SessionEventRecord> = active
        .iter()
        .copied()
        .filter(|record| !is_compaction(record))
        .collect();
    live_tokens(&live, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm::Message;
    use serde_json::json;
    use session::model::{Session, SessionMetadata, StoredMessage, StoredToolCall};

    fn new_session() -> Session {
        Session::new(SessionMetadata::new("/tmp/project", None, None))
    }

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

    fn push_tool_call(session: &mut Session, id: &str, name: &str) {
        session.append(SessionEvent::ToolCall {
            call: StoredToolCall {
                id: id.into(),
                name: name.into(),
                arguments: json!({}),
            },
        });
    }

    fn push_tool_result(session: &mut Session, id: &str, content: &str) {
        session.append(SessionEvent::ToolResult {
            tool_call_id: id.into(),
            content: content.into(),
            is_error: false,
            tool_name: None,
        });
    }

    fn default_policy() -> CompactionPolicy {
        CompactionPolicy {
            keep_recent_turns: 2,
            keep_recent_tokens: 8_000,
            ..CompactionPolicy::default()
        }
    }

    /// Build several turns of a realistic size so compaction pressure is real.
    fn grow_session(turns: usize, bytes_per_turn: usize) -> Session {
        let mut session = new_session();
        for index in 0..turns {
            push_user(&mut session, &format!("question {index}"));
            push_assistant(&mut session, &"a".repeat(bytes_per_turn));
            push_tool_call(&mut session, &format!("call-{index}"), "read");
            push_tool_result(
                &mut session,
                &format!("call-{index}"),
                &"b".repeat(bytes_per_turn),
            );
        }
        session
    }

    #[test]
    fn cut_lands_on_a_user_turn_boundary() {
        let session = grow_session(6, 2_000);
        let plan = plan_compaction(&session, &default_policy(), 60_000).unwrap();
        // The boundary is immediately before a user message (start of a kept turn).
        let next = session
            .events
            .iter()
            .find(|record| record.sequence == plan.boundary + 1)
            .unwrap();
        assert!(matches!(next.event, SessionEvent::UserMessage { .. }));
        assert!(plan.boundary > 0);
        assert!(plan.boundary < session.events.len() as u64);
        // Exactly keep_recent_turns user turns are kept.
        let kept_turns = session
            .events
            .iter()
            .filter(|record| {
                record.sequence > plan.boundary
                    && matches!(record.event, SessionEvent::UserMessage { .. })
            })
            .count();
        assert_eq!(kept_turns, 2);
        assert!(!plan.to_summarize.is_empty());
        assert!(plan.estimated_tokens_freed > 0);
    }

    #[test]
    fn never_cuts_before_a_tool_result() {
        let session = grow_session(6, 2_000);
        let plan = plan_compaction(&session, &default_policy(), 60_000).unwrap();
        // The kept tail starts at plan.boundary + 1 and must not begin with a
        // ToolResult (which would orphan a summarized tool call).
        let first_kept = session
            .events
            .iter()
            .find(|record| record.sequence == plan.boundary + 1)
            .unwrap();
        assert!(!matches!(first_kept.event, SessionEvent::ToolResult { .. }));
    }

    #[test]
    fn split_turn_when_one_turn_exceeds_the_keep_budget() {
        // One enormous turn that alone exceeds keep_recent_tokens, preceded by
        // some older material. The cut must split the turn: keep its tail,
        // summarize its head + older turns.
        let mut session = new_session();
        push_user(&mut session, "older");
        push_assistant(&mut session, "old answer");
        push_user(&mut session, "big turn");
        push_assistant(&mut session, &"x".repeat(100_000)); // huge assistant text
        push_tool_call(&mut session, "big-call", "bash");
        push_tool_result(&mut session, "big-call", &"y".repeat(100_000)); // huge result

        let policy = CompactionPolicy {
            keep_recent_turns: 1,
            keep_recent_tokens: 20_000,
            ..CompactionPolicy::default()
        };
        let plan = plan_compaction(&session, &policy, 500_000).unwrap();
        let first_kept = session
            .events
            .iter()
            .find(|record| record.sequence == plan.boundary + 1)
            .unwrap();
        // The kept tail must not start with a ToolResult, and must contain the
        // huge tool result (the current turn's tail) while the older "big turn"
        // user message is summarized (sequence <= boundary).
        assert!(matches!(
            first_kept.event,
            SessionEvent::AssistantMessage { .. }
        ));
        let big_user = session
            .events
            .iter()
            .find(|record| {
                matches!(&record.event, SessionEvent::UserMessage { message }
                    if message_text_contains(message, "big turn"))
            })
            .unwrap();
        assert!(
            big_user.sequence <= plan.boundary,
            "the split turn's head is summarized"
        );
    }

    fn message_text_contains(message: &StoredMessage, needle: &str) -> bool {
        message.content.iter().any(|content| match content {
            session::model::StoredContent::Text { text } => text.contains(needle),
            _ => false,
        })
    }

    #[test]
    fn no_op_when_small_session_fits_the_keep_budget() {
        let mut session = new_session();
        push_user(&mut session, "hello");
        push_assistant(&mut session, "hi");
        assert_eq!(
            plan_compaction(&session, &default_policy(), 1_000),
            None,
            "a tiny session has nothing to compact"
        );
    }

    #[test]
    fn no_op_when_estimated_context_already_within_keep_budget() {
        let session = grow_session(3, 1_000);
        // Even though live tokens may exceed the budget, the caller saying the
        // context is small short-circuits the planner.
        assert_eq!(plan_compaction(&session, &default_policy(), 5_000), None);
    }

    #[test]
    fn iterative_compaction_folds_previous_messages_into_next_summary() {
        let mut session = grow_session(4, 2_000);
        let policy = default_policy();
        // First compaction.
        let first = plan_compaction(&session, &policy, 40_000).unwrap();
        session.append(SessionEvent::CompactionSummary {
            summary: "first summary".into(),
            compacted_through: first.boundary,
        });
        // Grow more after the first compaction.
        for index in 4..8 {
            push_user(&mut session, &format!("question {index}"));
            push_assistant(&mut session, &"a".repeat(2_000));
            push_tool_call(&mut session, &format!("call-{index}"), "read");
            push_tool_result(&mut session, &format!("call-{index}"), &"b".repeat(2_000));
        }
        let second = plan_compaction(&session, &policy, 60_000).unwrap();
        assert!(second.boundary > first.boundary);
        assert_eq!(
            second.previous_summary.as_deref(),
            Some("first summary"),
            "iterative compaction threads the previous summary through"
        );
        // Events between the first boundary and the second boundary are folded
        // into the next summary.
        assert!(second.to_summarize.iter().any(|record| {
            record.sequence > first.boundary && record.sequence <= second.boundary
        }));
    }

    #[test]
    fn boundary_advances_strictly_past_previous_boundary() {
        let mut session = grow_session(4, 2_000);
        let policy = default_policy();
        let first = plan_compaction(&session, &policy, 40_000).unwrap();
        session.append(SessionEvent::CompactionSummary {
            summary: "first".into(),
            compacted_through: first.boundary,
        });
        // No new events: a repeated plan must be a no-op.
        assert!(plan_compaction(&session, &policy, 40_000).is_none());
    }

    #[test]
    fn tool_call_is_a_valid_cut_point() {
        let mut session = new_session();
        push_user(&mut session, "u1");
        push_assistant(&mut session, "a1");
        for id in ["c1", "c2", "c3"] {
            push_tool_call(&mut session, id, "read");
            push_tool_result(&mut session, id, &".rand".repeat(5_000));
        }
        push_user(&mut session, "u2 (just arrived, must be kept)");
        push_assistant(&mut session, "a2");
        let policy = CompactionPolicy {
            keep_recent_turns: 1,
            keep_recent_tokens: 12_000,
            ..CompactionPolicy::default()
        };
        let plan = plan_compaction(&session, &policy, 100_000).unwrap();
        assert!(plan.boundary >= 1);
        // The newest user message is always kept.
        let newest = session.events.last().unwrap();
        assert!(newest.sequence > plan.boundary);
    }

    #[test]
    fn estimate_live_tokens_covers_all_active_events() {
        let mut session = grow_session(6, 4_000);
        let estimated = estimate_live_tokens(&session);
        assert!(estimated > 0);
        // After a compaction, live region shrinks dramatically.
        let plan = plan_compaction(&session, &default_policy(), estimated).unwrap();
        session.append(SessionEvent::CompactionSummary {
            summary: "s".into(),
            compacted_through: plan.boundary,
        });
        let after = estimate_live_tokens(&session);
        assert!(after < estimated);
    }
}
