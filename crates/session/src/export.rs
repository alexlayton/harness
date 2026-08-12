//! Canonical JSONL and human-readable exports.

use crate::codec::{encode_header, encode_record};
use crate::error::{Result, SessionError, io_error};
use crate::model::{Session, SessionEvent, SessionEventRecord, StoredContent, StoredMessage};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ExportOptions {
    pub include_reasoning: bool,
    pub include_tool_output: bool,
    pub redact_secrets: bool,
    pub max_tool_output_bytes: Option<usize>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_reasoning: true,
            include_tool_output: true,
            redact_secrets: false,
            max_tool_output_bytes: None,
        }
    }
}

/// Export a complete session to canonical JSONL.  If no destination is
/// supplied, a timestamped file is created in the current directory (never in
/// the hidden state directory).
pub fn export_jsonl(
    session: &Session,
    destination: Option<&Path>,
    options: &ExportOptions,
) -> Result<PathBuf> {
    let destination = destination
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_export_path(session));
    let destination = if destination.is_absolute() {
        destination
    } else {
        std::env::current_dir()
            .map_err(|source| io_error("resolve export directory", ".", source))?
            .join(destination)
    };
    if session
        .path()
        .is_some_and(|path| same_file_path(path, &destination))
    {
        return Err(SessionError::ExportWouldOverwrite(destination));
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|source| io_error("create export directory", parent, source))?;
    let temp = destination.with_extension(format!(
        "{}.tmp-{}",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("jsonl"),
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|source| io_error("create export file", &temp, source))?;
        let header = encode_header(session.header_metadata())?;
        file.write_all(header.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|source| io_error("write export header", &temp, source))?;
        for record in &session.events {
            let transformed = transform_record(record, options);
            let line = encode_record(session.id(), &transformed)?;
            file.write_all(line.as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .map_err(|source| io_error("write export event", &temp, source))?;
        }
        file.flush()
            .and_then(|_| file.sync_all())
            .map_err(|source| io_error("flush export", &temp, source))?;
        fs::rename(&temp, &destination)
            .map_err(|source| io_error("replace export file", &destination, source))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map(|_| destination)
}

/// Export a readable transcript.  This is intentionally separate from the
/// canonical JSONL interchange format.
pub fn export_transcript(session: &Session, destination: Option<&Path>) -> Result<PathBuf> {
    let destination = destination.map(Path::to_path_buf).unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(format!("harness-session-{}.txt", session.id()))
    });
    let destination = if destination.is_absolute() {
        destination
    } else {
        std::env::current_dir()
            .map_err(|source| io_error("resolve transcript directory", ".", source))?
            .join(destination)
    };
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|source| io_error("create transcript directory", parent, source))?;
    let mut output = String::new();
    output.push_str(&format!("Harness session {}\n", session.id()));
    output.push_str(&format!(
        "Workspace: {}\n\n",
        session.metadata.workspace_root.display()
    ));
    for record in &session.events {
        match &record.event {
            SessionEvent::UserMessage { message } => {
                output.push_str("## User\n");
                output.push_str(&message_text(message));
                output.push_str("\n\n");
            }
            SessionEvent::AssistantMessage { message } => {
                output.push_str("## Assistant\n");
                output.push_str(&message_text(message));
                output.push_str("\n\n");
            }
            SessionEvent::ToolCall { call } => {
                output.push_str(&format!(
                    "## Tool call: {}\n{}\n\n",
                    call.name, call.arguments
                ));
            }
            SessionEvent::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                output.push_str(&format!(
                    "## Tool result {}{}\n{}\n\n",
                    tool_call_id,
                    if *is_error { " (error)" } else { "" },
                    content
                ));
            }
            SessionEvent::CompactionSummary { summary, .. } => {
                output.push_str("## Generated summary\n");
                output.push_str(summary);
                output.push_str("\n\n");
            }
            SessionEvent::Reasoning { text } => {
                output.push_str("## Reasoning\n");
                output.push_str(text);
                output.push_str("\n\n");
            }
            SessionEvent::ModelChange { provider, model } => {
                output.push_str(&format!("[Model changed to {provider} · {model}]\n\n"));
            }
            SessionEvent::TurnCancelled { reason } => {
                output.push_str(&format!("[Turn cancelled: {reason}]\n\n"));
            }
            SessionEvent::Error { message } => {
                output.push_str(&format!("[Error: {message}]\n\n"));
            }
            SessionEvent::Usage { .. }
            | SessionEvent::MetadataChange { .. }
            | SessionEvent::Unknown { .. } => {}
        }
    }
    fs::write(&destination, output)
        .map_err(|source| io_error("write transcript", &destination, source))?;
    Ok(destination)
}

fn transform_record(record: &SessionEventRecord, options: &ExportOptions) -> SessionEventRecord {
    let mut transformed = record.clone();
    transformed.event = match &record.event {
        SessionEvent::AssistantMessage { message } => SessionEvent::AssistantMessage {
            message: transform_message(message, options),
        },
        SessionEvent::UserMessage { message } => SessionEvent::UserMessage {
            message: transform_message(message, options),
        },
        SessionEvent::Reasoning { text } if !options.include_reasoning => SessionEvent::Reasoning {
            text: String::new(),
        },
        SessionEvent::CompactionSummary {
            summary,
            compacted_through,
        } if !options.include_reasoning => SessionEvent::CompactionSummary {
            summary: summary
                .lines()
                .filter(|line| !line.trim_start().starts_with("Reasoning:"))
                .collect::<Vec<_>>()
                .join("\n"),
            compacted_through: *compacted_through,
        },
        SessionEvent::ToolResult {
            tool_call_id,
            content,
            is_error,
            tool_name,
        } => SessionEvent::ToolResult {
            tool_call_id: tool_call_id.clone(),
            content: transform_output(content, options),
            is_error: *is_error,
            tool_name: tool_name.clone(),
        },
        SessionEvent::Unknown { kind, data } => SessionEvent::Unknown {
            kind: kind.clone(),
            data: if options.redact_secrets {
                redact_json(data)
            } else {
                data.clone()
            },
        },
        event => event.clone(),
    };
    transformed
}

fn transform_message(message: &StoredMessage, options: &ExportOptions) -> StoredMessage {
    let mut message = message.clone();
    message.content.retain(|content| {
        options.include_reasoning || !matches!(content, StoredContent::Reasoning { .. })
    });
    for content in &mut message.content {
        match content {
            StoredContent::ToolResult { content, .. } => {
                *content = transform_output(content, options)
            }
            StoredContent::Text { text } | StoredContent::Reasoning { text }
                if options.redact_secrets =>
            {
                *text = redact_text(text)
            }
            StoredContent::ToolCall { arguments, .. } if options.redact_secrets => {
                *arguments = redact_json(arguments)
            }
            _ => {}
        }
    }
    message
}

fn transform_output(value: &str, options: &ExportOptions) -> String {
    let mut output = if options.include_tool_output {
        value.to_owned()
    } else {
        "[tool output omitted from export]".into()
    };
    if let Some(max_bytes) = options.max_tool_output_bytes {
        output = cap_utf8(&output, max_bytes);
    }
    if options.redact_secrets {
        output = redact_text(&output);
    }
    output
}

fn redact_text(value: &str) -> String {
    // This is intentionally conservative and dependency-free.  It masks
    // values after common secret keys while leaving normal tool output intact.
    let mut result = value.to_owned();
    for key in [
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
    ] {
        let mut search_from = 0;
        loop {
            let lower = result.to_ascii_lowercase();
            if search_from >= lower.len() {
                break;
            }
            let Some(offset) = lower[search_from..].find(key) else {
                break;
            };
            let start = search_from + offset;
            let after_key = start + key.len();
            let Some(separator) = result[after_key..].find([':', '=']) else {
                search_from = after_key;
                continue;
            };
            let value_start = after_key + separator + 1;
            let value_start = value_start
                + result[value_start..]
                    .find(|character: char| {
                        !character.is_whitespace() && character != '"' && character != '\''
                    })
                    .unwrap_or(0);
            let value_end = result[value_start..]
                .find(|character: char| {
                    character.is_whitespace() || matches!(character, ',' | '}' | ']' | '"' | '\'')
                })
                .map(|end| value_start + end)
                .unwrap_or(result.len());
            if value_start >= value_end {
                search_from = value_start.saturating_add(1);
                continue;
            }
            result.replace_range(value_start..value_end, "[REDACTED]");
            search_from = value_start + "[REDACTED]".len();
        }
    }
    result
}

fn redact_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => serde_json::Value::Object(
            object
                .iter()
                .map(|(key, value)| {
                    let sensitive = [
                        "api_key",
                        "apikey",
                        "authorization",
                        "password",
                        "secret",
                        "token",
                    ]
                    .iter()
                    .any(|candidate| key.eq_ignore_ascii_case(candidate));
                    (
                        key.clone(),
                        if sensitive {
                            serde_json::Value::String("[REDACTED]".into())
                        } else {
                            redact_json(value)
                        },
                    )
                })
                .collect(),
        ),
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(redact_json).collect())
        }
        value => value.clone(),
    }
}

fn message_text(message: &StoredMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            StoredContent::Text { text } | StoredContent::Reasoning { text } => Some(text.as_str()),
            StoredContent::ToolResult { content, .. } => Some(content.as_str()),
            StoredContent::ToolCall { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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

fn default_export_path(session: &Session) -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(format!("harness-session-{}.jsonl", session.id()))
}

fn same_file_path(left: &Path, right: &Path) -> bool {
    left.canonicalize().ok() == right.canonicalize().ok() || left == right
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::decode_session;
    use crate::model::{SessionEvent, SessionMetadata, StoredMessage};
    use llm::Message;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn export_is_loadable_and_can_redact_tool_output() {
        let directory = tempdir().unwrap();
        let mut session = Session::new(SessionMetadata::new(directory.path(), None, None));
        session.append(SessionEvent::ModelChange {
            provider: "mock".into(),
            model: "demo".into(),
        });
        session.append(SessionEvent::Usage {
            usage: crate::model::UsageSummary {
                input_tokens: 4,
                turns: 1,
                ..crate::model::UsageSummary::default()
            },
        });
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&Message::user("hello")),
        });
        session.append(SessionEvent::ToolCall {
            call: crate::model::StoredToolCall {
                id: "call".into(),
                name: "bash".into(),
                arguments: json!({"token": "secret"}),
            },
        });
        session.append(SessionEvent::ToolResult {
            tool_call_id: "call".into(),
            content: "token=secret output".into(),
            is_error: false,
            tool_name: Some("bash".into()),
        });
        let destination = directory.path().join("export.jsonl");
        export_jsonl(
            &session,
            Some(&destination),
            &ExportOptions {
                redact_secrets: true,
                ..ExportOptions::default()
            },
        )
        .unwrap();
        let content = fs::read_to_string(&destination).unwrap();
        assert!(content.contains("[REDACTED]"));
        let loaded = decode_session(&content, &destination).unwrap();
        assert_eq!(loaded.metadata.usage.input_tokens, 4);
        assert_eq!(loaded.metadata.provider.as_deref(), Some("mock"));
    }
}
