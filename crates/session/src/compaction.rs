//! Deterministic, local compaction primitives.
//!
//! The compactor never rewrites or deletes old events.  It appends a summary
//! with an explicit sequence boundary; context reconstruction then uses the
//! summary plus events after that boundary.  A model-assisted summarizer can
//! be added behind this API later without changing the storage format.

use crate::error::Result;
use crate::model::{Session, SessionEvent, SessionEventRecord, StoredContent};

/// Policy for the first local compactor.  Counts are deliberately based on
/// durable events/messages rather than provider-specific tokenizers.
#[derive(Clone, Debug)]
pub struct CompactionPolicy {
    /// Compact when more than this many message-like events are active.
    pub max_messages: usize,
    /// Keep this many recent user turns and their complete event tails.
    pub retain_messages: usize,
    /// Bound the generated summary by UTF-8 bytes.
    pub max_summary_bytes: usize,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            max_messages: 40,
            retain_messages: 12,
            max_summary_bytes: 12 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionResult {
    pub summary: String,
    pub compacted_through: u64,
    pub removed_event_count: usize,
    pub retained_event_count: usize,
}

/// Produce a deterministic compaction event payload, if the current active
/// path exceeds the policy.  Existing compaction summaries are treated as a
/// single context item, so calling this repeatedly without new pressure is a
/// no-op.
pub fn deterministic_compaction(
    session: &Session,
    policy: &CompactionPolicy,
) -> Option<CompactionResult> {
    let policy = policy.normalized();
    let latest_compaction = session.events.iter().rev().find_map(|record| {
        if let SessionEvent::CompactionSummary {
            compacted_through, ..
        } = &record.event
        {
            Some((record.sequence, *compacted_through))
        } else {
            None
        }
    });

    let active = if let Some((summary_sequence, boundary)) = latest_compaction {
        session
            .events
            .iter()
            .filter(|record| {
                record.sequence == summary_sequence
                    || (record.sequence > summary_sequence && record.sequence > boundary)
            })
            .collect::<Vec<_>>()
    } else {
        session.events.iter().collect::<Vec<_>>()
    };

    let message_count = active
        .iter()
        .filter(|record| {
            matches!(
                record.event,
                SessionEvent::UserMessage { .. }
                    | SessionEvent::AssistantMessage { .. }
                    | SessionEvent::CompactionSummary { .. }
            )
        })
        .count();
    if message_count <= policy.max_messages {
        return None;
    }

    let user_sequences = active
        .iter()
        .filter_map(|record| {
            matches!(record.event, SessionEvent::UserMessage { .. }).then_some(record.sequence)
        })
        .collect::<Vec<_>>();
    if user_sequences.len() <= policy.retain_messages {
        return None;
    }
    let keep_from_user = user_sequences[user_sequences.len() - policy.retain_messages];

    // Start on a user-turn boundary.  If a malformed/imported path does not
    // have a user event at the desired point, fall back to the first retained
    // active event rather than cutting a tool pair in half.
    let first_retained = active
        .iter()
        .find(|record| record.sequence >= keep_from_user)
        .map(|record| record.sequence)?;
    let boundary = first_retained.saturating_sub(1);
    if latest_compaction.is_some_and(|(_, previous_boundary)| boundary <= previous_boundary) {
        return None;
    }

    let older = session
        .events
        .iter()
        .filter(|record| record.sequence <= boundary)
        .collect::<Vec<_>>();
    let summary = summarize_events(&older, policy.max_summary_bytes);
    let retained_event_count = session
        .events
        .iter()
        .filter(|record| record.sequence > boundary)
        .count();
    Some(CompactionResult {
        summary,
        compacted_through: boundary,
        removed_event_count: older.len(),
        retained_event_count,
    })
}

impl CompactionPolicy {
    fn normalized(&self) -> Self {
        Self {
            max_messages: self.max_messages.max(1),
            retain_messages: self.retain_messages.max(1).min(self.max_messages.max(1)),
            max_summary_bytes: self.max_summary_bytes.max(128),
        }
    }
}

/// Append a deterministic summary through a caller-supplied append function.
/// This keeps the pure compactor independent from the filesystem store.
pub fn append_compaction<F>(
    session: &mut Session,
    policy: &CompactionPolicy,
    mut append: F,
) -> Result<Option<CompactionResult>>
where
    F: FnMut(&mut Session, SessionEvent) -> Result<()>,
{
    let Some(result) = deterministic_compaction(session, policy) else {
        return Ok(None);
    };
    append(
        session,
        SessionEvent::CompactionSummary {
            summary: result.summary.clone(),
            compacted_through: result.compacted_through,
        },
    )?;
    Ok(Some(result))
}

fn summarize_events(events: &[&SessionEventRecord], max_bytes: usize) -> String {
    let mut lines = Vec::<String>::new();
    lines.push("This is generated context, not a verbatim transcript.".into());
    for record in events {
        match &record.event {
            SessionEvent::UserMessage { message } => {
                let text = message_text(message_contents(message));
                if !text.is_empty() {
                    lines.push(format!("User: {text}"));
                }
            }
            SessionEvent::AssistantMessage { message } => {
                let text = message_text(message_contents(message));
                if !text.is_empty() {
                    lines.push(format!("Assistant: {text}"));
                }
            }
            SessionEvent::ToolCall { call } => {
                lines.push(format!("Tool call: {} ({})", call.name, call.id));
            }
            SessionEvent::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let status = if *is_error { "error" } else { "ok" };
                lines.push(format!("Tool result [{status}] {tool_call_id}: {content}"));
            }
            SessionEvent::Reasoning { text } => {
                if !text.is_empty() {
                    lines.push(format!("Reasoning: {text}"));
                }
            }
            SessionEvent::ModelChange { provider, model } => {
                lines.push(format!("Model changed to {provider} · {model}"));
            }
            SessionEvent::Error { message } => lines.push(format!("Error: {message}")),
            SessionEvent::TurnCancelled { reason } => {
                lines.push(format!("Turn cancelled: {reason}"));
            }
            SessionEvent::CompactionSummary { summary, .. } => {
                lines.push(format!("Earlier generated summary: {summary}"));
            }
            SessionEvent::Usage { .. }
            | SessionEvent::MetadataChange { .. }
            | SessionEvent::Unknown { .. } => {}
        }
    }
    cap_utf8(&lines.join("\n"), max_bytes)
}

fn message_contents(message: &crate::model::StoredMessage) -> Vec<String> {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            StoredContent::Text { text } => Some(text.clone()),
            StoredContent::Reasoning { text } => Some(text.clone()),
            StoredContent::ToolResult { content, .. } => Some(content.clone()),
            StoredContent::ToolCall { .. } => None,
        })
        .collect()
}

fn message_text(parts: Vec<String>) -> String {
    parts.join(" ").replace('\n', " ")
}

fn cap_utf8(value: &str, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let suffix = "…";
    let mut end = max_bytes.saturating_sub(suffix.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{suffix}", &value[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SessionEvent, SessionMetadata, StoredMessage};
    use llm::Message;

    #[test]
    fn compaction_is_deterministic_and_has_a_boundary() {
        let mut session = Session::new(SessionMetadata::new("/tmp/project", None, None));
        for index in 0..6 {
            session.append(SessionEvent::UserMessage {
                message: StoredMessage::from_llm(&Message::user(format!("question {index}"))),
            });
            session.append(SessionEvent::AssistantMessage {
                message: StoredMessage::from_llm(&Message::assistant(vec![llm::Content::Text(
                    format!("answer {index}"),
                )])),
            });
        }
        let policy = CompactionPolicy {
            max_messages: 4,
            retain_messages: 2,
            max_summary_bytes: 1_000,
        };
        let first = deterministic_compaction(&session, &policy).unwrap();
        let second = deterministic_compaction(&session, &policy).unwrap();
        assert_eq!(first, second);
        assert!(first.compacted_through > 0);
    }
}
