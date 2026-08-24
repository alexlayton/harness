//! Event → text serialization for the summarizer.
//!
//! Moved from `session::compaction::summarize_events` and hardened: tool
//! results are truncated per-item, the whole transcript is capped from the
//! oldest side (the tail is most relevant), and the output is explicitly
//! marked as a serialized transcript so the summarizer does not try to
//! "reply" to it.

use serde_json::Value;
use session::model::{SessionEvent, SessionEventRecord, StoredContent, StoredMessage};
use std::collections::BTreeSet;

/// One serialized transcript plus whether any material was dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerializedTranscript {
    pub text: String,
    pub truncated: bool,
}

/// Serialize `events` into a flat transcript for the summarizer.
///
/// Events are emitted oldest → newest. The newest material is always kept:
/// when the accumulated bytes exceed `max_input_bytes`, the *oldest* lines are
/// dropped. Tool results are each truncated to `max_tool_result_chars`.
pub fn serialize_events(
    events: &[SessionEventRecord],
    max_input_bytes: usize,
    max_tool_result_chars: usize,
) -> SerializedTranscript {
    // Walk newest → oldest, keeping lines until the budget is exhausted. The
    // budget is measured by the head line count so that even a single
    // oversized (newest) line is retained rather than dropping the tail.
    let mut kept = Vec::<String>::new();
    let mut bytes = 0usize;
    let mut truncated = false;
    for record in events.iter().rev() {
        let line = serialize_record(&record.event, max_tool_result_chars);
        if line.is_empty() {
            continue;
        }
        if !kept.is_empty() && bytes.saturating_add(line.len()) > max_input_bytes {
            truncated = true;
            break;
        }
        bytes = bytes.saturating_add(line.len());
        kept.push(line);
    }
    kept.reverse();
    SerializedTranscript {
        text: kept.join("\n"),
        truncated,
    }
}

/// Serialize one event to its transcript line (empty for events that do not
/// contribute to the summarizer's view, e.g. usage accounting).  `pub(crate)`
/// so the planner can reuse it for per-event token estimation over the live
/// region.
pub(crate) fn serialize_record(event: &SessionEvent, max_tool_result_chars: usize) -> String {
    match event {
        SessionEvent::UserMessage { message } => message_joined_text(message)
            .map(|text| format!("[User]: {text}"))
            .unwrap_or_default(),
        SessionEvent::AssistantMessage { message } => {
            serialize_assistant_message(message, max_tool_result_chars)
        }
        SessionEvent::Reasoning { text } => {
            if text.trim().is_empty() {
                String::new()
            } else {
                format!("[Assistant reasoning]: {text}")
            }
        }
        SessionEvent::ToolCall { call } => format!(
            "[Assistant tool calls]: {}",
            format_tool_call(&call.name, &call.arguments)
        ),
        SessionEvent::ToolResult {
            tool_call_id,
            content,
            is_error,
            ..
        } => {
            if content.trim().is_empty() {
                return String::new();
            }
            let status = if *is_error { "error" } else { "ok" };
            let body = truncate_for_summary(content, max_tool_result_chars);
            format!("[Tool result {status}]: ({tool_call_id}) {body}")
        }
        SessionEvent::CompactionSummary { summary, .. } => {
            if summary.trim().is_empty() {
                String::new()
            } else {
                format!("[Earlier generated summary]: {summary}")
            }
        }
        SessionEvent::ModelChange { provider, model } => {
            format!("[Model changed]: {provider} · {model}")
        }
        SessionEvent::TurnCancelled { reason } => format!("[Turn cancelled]: {reason}"),
        SessionEvent::Error { message } => format!("[Error]: {message}"),
        // These never enter provider context, so they do not matter to the
        // summarizer either.
        SessionEvent::Usage { .. }
        | SessionEvent::MetadataChange { .. }
        | SessionEvent::Unknown { .. } => String::new(),
    }
}

fn serialize_assistant_message(message: &StoredMessage, max_tool_result_chars: usize) -> String {
    let mut sections = Vec::<String>::new();
    let mut reasoning = Vec::<String>::new();
    let mut text = Vec::<String>::new();
    let mut calls = Vec::<String>::new();
    for content in &message.content {
        match content {
            StoredContent::Text { text: value } => text.push(value.clone()),
            StoredContent::Reasoning { text: value } => reasoning.push(value.clone()),
            // Opaque continuation state is intentionally excluded from summary prose.
            StoredContent::Opaque { .. } => {}
            StoredContent::ToolCall {
                name, arguments, ..
            } => {
                calls.push(format_tool_call(name, arguments));
            }
            StoredContent::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                sections.push(serialize_embedded_tool_result(
                    tool_call_id,
                    content,
                    *is_error,
                    max_tool_result_chars,
                ));
            }
        }
    }
    if !reasoning.is_empty() {
        sections.push(format!("[Assistant reasoning]: {}", reasoning.join("\n")));
    }
    if !text.is_empty() {
        sections.push(format!("[Assistant]: {}", text.join("\n")));
    }
    if !calls.is_empty() {
        sections.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
    }
    sections
        .iter()
        .filter(|section| !section.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
}

fn serialize_embedded_tool_result(
    tool_call_id: &str,
    content: &str,
    is_error: bool,
    max_tool_result_chars: usize,
) -> String {
    if content.trim().is_empty() {
        return String::new();
    }
    let status = if is_error { "error" } else { "ok" };
    let body = truncate_for_summary(content, max_tool_result_chars);
    format!("[Tool result {status}]: ({tool_call_id}) {body}")
}

fn message_joined_text(message: &StoredMessage) -> Option<String> {
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            StoredContent::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    (!text.trim().is_empty()).then_some(text)
}

/// Render a tool call as `read(path="foo.txt")` for the transcript.
pub fn format_tool_call(name: &str, arguments: &Value) -> String {
    let rendered = match arguments {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            keys.iter()
                .map(|key| format!("{key}={}", json_arg(&map[*key])))
                .collect::<Vec<_>>()
                .join(", ")
        }
        other => other.to_string(),
    };
    format!("{name}({rendered})")
}

/// A JSON value rendered compactly (strings stay quoted).
fn json_arg(value: &Value) -> String {
    match value {
        Value::String(text) => format!("\"{text}\""),
        other => other.to_string(),
    }
}

/// Truncate a tool result to `max_chars` characters, keeping the beginning
/// and appending a marker with the dropped count.
pub fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let chars = text.chars().count();
    if chars <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars).collect();
    let dropped = chars - max_chars;
    format!("{kept}\n\n[... {dropped} more characters truncated]")
}

/// File-touching operations extracted from tool calls for the summary's
/// structured file lists. Follows the "names + paths only" rule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileOperations {
    /// Paths read but not modified.
    pub read: Vec<String>,
    /// Paths written or edited.
    pub modified: Vec<String>,
}

/// Collect file-touching tool calls (`read` / `write` / `edit`, plus any
/// tool call carrying a `path` argument) from a span of events.
pub fn extract_file_operations(events: &[SessionEventRecord]) -> FileOperations {
    let mut read = BTreeSet::new();
    let mut modified = BTreeSet::new();
    for record in events {
        match &record.event {
            SessionEvent::AssistantMessage { message } => {
                for content in &message.content {
                    if let StoredContent::ToolCall {
                        name, arguments, ..
                    } = content
                    {
                        record_tool_path(name, arguments, &mut read, &mut modified);
                    }
                }
            }
            SessionEvent::ToolCall { call } => {
                record_tool_path(&call.name, &call.arguments, &mut read, &mut modified);
            }
            _ => {}
        }
    }
    FileOperations {
        read: read
            .into_iter()
            .filter(|path| !modified.contains(path))
            .collect(),
        modified: modified.into_iter().collect(),
    }
}

fn record_tool_path(
    name: &str,
    arguments: &Value,
    read: &mut BTreeSet<String>,
    modified: &mut BTreeSet<String>,
) {
    let Some(path) = arguments.get("path").and_then(Value::as_str) else {
        return;
    };
    if path.trim().is_empty() {
        return;
    }
    match name {
        "write" | "edit" => {
            modified.insert(path.to_owned());
        }
        "read" => {
            read.insert(path.to_owned());
        }
        _ => {
            // Any other tool with a path argument is treated as a read.
            read.insert(path.to_owned());
        }
    }
}

/// Render the file lists as trailing summary sections, if any.
pub fn format_file_operations(operations: &FileOperations) -> String {
    let mut sections = Vec::<String>::new();
    if !operations.read.is_empty() {
        sections.push(format!(
            "<files-read>\n{}\n</files-read>",
            operations.read.join("\n")
        ));
    }
    if !operations.modified.is_empty() {
        sections.push(format!(
            "<files-modified>\n{}\n</files-modified>",
            operations.modified.join("\n")
        ));
    }
    if sections.is_empty() {
        return String::new();
    }
    format!("\n\n{}", sections.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm::Message;
    use serde_json::json;
    use session::model::{Session, SessionMetadata, StoredMessage, StoredToolCall};

    fn user(session: &mut Session, text: &str) -> u64 {
        let record = session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&Message::user(text)),
        });
        record.sequence
    }

    fn assistant(session: &mut Session, text: &str) -> u64 {
        let record = session.append(SessionEvent::AssistantMessage {
            message: StoredMessage::from_llm(&Message::assistant(vec![llm::Content::Text(
                text.into(),
            )])),
        });
        record.sequence
    }

    fn tool_result(session: &mut Session, content: &str, is_error: bool) -> u64 {
        let record = session.append(SessionEvent::ToolResult {
            tool_call_id: "call-1".into(),
            content: content.into(),
            is_error,
            tool_name: Some("read".into()),
        });
        record.sequence
    }

    #[test]
    fn serializes_mixed_history_with_markers() {
        let mut session = Session::new(SessionMetadata::new("/tmp", None, None));
        user(&mut session, "hello");
        assistant(&mut session, "working…");
        tool_result(&mut session, "file contents", false);

        let transcript = serialize_events(&session.events, 96 * 1024, 2_000);
        assert!(transcript.text.contains("[User]: hello"));
        assert!(transcript.text.contains("[Assistant]: working…"));
        assert!(
            transcript
                .text
                .contains("[Tool result ok]: (call-1) file contents")
        );
        assert!(!transcript.truncated);
    }

    #[test]
    fn tool_results_are_truncated_per_item() {
        let mut session = Session::new(SessionMetadata::new("/tmp", None, None));
        user(&mut session, "hi");
        tool_result(&mut session, &"x".repeat(5_000), false);
        let transcript = serialize_events(&session.events, 96 * 1024, 100);
        assert!(
            transcript
                .text
                .contains("[... 4900 more characters truncated]")
        );
    }

    #[test]
    fn oldest_material_is_dropped_first_when_over_budget() {
        let mut session = Session::new(SessionMetadata::new("/tmp", None, None));
        user(&mut session, "oldest message that is fairly long");
        assistant(&mut session, "old answer");
        user(&mut session, "newest message");
        assistant(&mut session, "new answer");

        let transcript = serialize_events(&session.events, 40, 2_000);
        assert!(
            transcript.text.contains("new"),
            "the newest tail must survive the cap"
        );
        assert!(
            !transcript.text.contains("oldest"),
            "oldest material must be dropped first"
        );
        assert!(transcript.truncated);
    }

    #[test]
    fn embedded_tool_result_is_truncated_once_with_its_id() {
        let content = "y".repeat(5_000);
        let rendered = serialize_embedded_tool_result("call-9", &content, false, 100);
        assert_eq!(
            rendered,
            format!(
                "[Tool result ok]: (call-9) {}",
                truncate_for_summary(&content, 100)
            ),
            "body truncated once and rendered with its id"
        );

        let error = serialize_embedded_tool_result("call-9", &"boom".repeat(5_000), true, 2);
        assert!(error.starts_with("[Tool result error]:"), "{error}");
    }

    #[test]
    fn embedded_tool_result_in_assistant_message_keeps_id_and_truncates() {
        let message =
            StoredMessage::from_llm(&Message::assistant(vec![llm::Content::ToolResult {
                tool_call_id: "call-7".into(),
                content: "z".repeat(8_000),
                is_error: false,
            }]));
        let rendered = serialize_record(&SessionEvent::AssistantMessage { message }, 50);
        assert!(rendered.contains("(call-7)"), "{rendered}");
        assert!(
            rendered.contains("[... 7950 more characters truncated]"),
            "{rendered}"
        );
    }

    #[test]
    fn format_tool_call_renders_arguments_compactly() {
        assert_eq!(
            format_tool_call("read", &json!({ "path": "foo.txt", "offset": 5 })),
            "read(offset=5, path=\"foo.txt\")"
        );
    }

    #[test]
    fn file_operations_are_extracted_from_tool_calls() {
        let mut session = Session::new(SessionMetadata::new("/tmp", None, None));
        user(&mut session, "read and write");
        session.append(SessionEvent::ToolCall {
            call: StoredToolCall {
                id: "1".into(),
                name: "read".into(),
                arguments: json!({ "path": "a.txt" }),
            },
        });
        session.append(SessionEvent::ToolCall {
            call: StoredToolCall {
                id: "2".into(),
                name: "write".into(),
                arguments: json!({ "path": "b.rs" }),
            },
        });
        session.append(SessionEvent::ToolCall {
            call: StoredToolCall {
                id: "3".into(),
                name: "edit".into(),
                arguments: json!({ "path": "a.txt" }),
            },
        });

        let ops = extract_file_operations(&session.events);
        // a.txt was both read and edited → modified only.
        assert_eq!(ops.read, Vec::<String>::new());
        assert_eq!(ops.modified, vec!["a.txt".to_owned(), "b.rs".to_owned()]);
    }

    #[test]
    fn file_operation_lists_format_as_sections() {
        let ops = FileOperations {
            read: vec!["a.txt".into()],
            modified: vec!["b.rs".into()],
        };
        let rendered = format_file_operations(&ops);
        assert!(rendered.contains("<files-read>"));
        assert!(rendered.contains("a.txt"));
        assert!(rendered.contains("<files-modified>"));
        assert!(rendered.contains("b.rs"));

        assert_eq!(
            format_file_operations(&FileOperations::default()),
            String::new()
        );
    }

    #[test]
    fn truncate_for_summary_keeps_beginning_and_marks_dropped_count() {
        assert_eq!(truncate_for_summary("abc", 10), "abc");
        let rendered = truncate_for_summary("abcdef", 3);
        assert!(rendered.starts_with("abc"));
        assert!(rendered.contains("3 more characters truncated"));
    }
}
