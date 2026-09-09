//! Canonical JSONL and human-readable exports.

use crate::codec::{encode_header, encode_record};
use crate::error::{Result, SessionError, io_error};
use crate::model::{Session, SessionEvent, SessionEventRecord, StoredContent, StoredMessage};
use llm::util::truncate_utf8;
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
        // Preserve IDs and numeric fields needed to decode the export, while
        // transforming every free-text header field that may contain copied
        // credentials or workspace-specific secrets.
        let header_metadata = transform_metadata(session.header_metadata(), options);
        let header = encode_header(&header_metadata)?;
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
    // Redaction runs after omission/truncation decisions so no unredacted
    // alternate representation remains: messages, outputs, summaries, and
    // structured payloads are all transformed through one path.
    transformed.event = match &record.event {
        SessionEvent::AssistantMessage { message } => SessionEvent::AssistantMessage {
            message: transform_message(message, options),
        },
        SessionEvent::UserMessage { message } => SessionEvent::UserMessage {
            message: transform_message(message, options),
        },
        SessionEvent::Reasoning { text } => SessionEvent::Reasoning {
            text: transform_reasoning(text, options),
        },
        SessionEvent::ToolCall { call } => SessionEvent::ToolCall {
            call: transform_tool_call(call, options),
        },
        SessionEvent::CompactionSummary {
            summary,
            compacted_through,
        } => SessionEvent::CompactionSummary {
            summary: transform_summary(summary, options),
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
            tool_name: tool_name
                .as_ref()
                .map(|tool_name| transform_text(tool_name, options)),
        },
        SessionEvent::ModelChange { provider, model } => SessionEvent::ModelChange {
            provider: transform_text(provider, options),
            model: transform_text(model, options),
        },
        SessionEvent::MetadataChange { title } => SessionEvent::MetadataChange {
            title: title.as_ref().map(|title| transform_title(title, options)),
        },
        SessionEvent::TurnCancelled { reason } => SessionEvent::TurnCancelled {
            reason: transform_diagnostic(reason, options),
        },
        SessionEvent::Error { message } => SessionEvent::Error {
            message: transform_diagnostic(message, options),
        },
        SessionEvent::Unknown { kind, data } => SessionEvent::Unknown {
            kind: kind.clone(),
            data: transform_unknown(data, options),
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
            StoredContent::Opaque { data, .. } if options.redact_secrets => {
                *data = redact_json(data)
            }
            _ => {}
        }
    }
    message
}

fn transform_metadata(
    metadata: &crate::model::SessionMetadata,
    options: &ExportOptions,
) -> crate::model::SessionMetadata {
    let mut metadata = metadata.clone();
    metadata.title = metadata
        .title
        .as_ref()
        .map(|title| transform_text(title, options));
    metadata.provider = metadata
        .provider
        .as_ref()
        .map(|provider| transform_text(provider, options));
    metadata.model = metadata
        .model
        .as_ref()
        .map(|model| transform_text(model, options));
    let workspace = metadata.workspace_root.to_string_lossy().into_owned();
    metadata.workspace_root = PathBuf::from(transform_text(&workspace, options));
    metadata
}

fn transform_text(value: &str, options: &ExportOptions) -> String {
    if options.redact_secrets {
        redact_text(value)
    } else {
        value.to_owned()
    }
}

fn transform_output(value: &str, options: &ExportOptions) -> String {
    let mut output = if options.include_tool_output {
        value.to_owned()
    } else {
        "[tool output omitted from export]".into()
    };
    if let Some(max_bytes) = options.max_tool_output_bytes {
        output = truncate_utf8(&output, max_bytes);
    }
    if options.redact_secrets {
        output = redact_text(&output);
    }
    output
}

/// Standalone reasoning: omitted unless included, then redacted.
fn transform_reasoning(text: &str, options: &ExportOptions) -> String {
    if !options.include_reasoning {
        return String::new();
    }
    if options.redact_secrets {
        redact_text(text)
    } else {
        text.to_owned()
    }
}

/// Compaction summaries: reasoning lines omitted unless included, then the
/// surviving summary is redacted.
fn transform_summary(summary: &str, options: &ExportOptions) -> String {
    let kept = if options.include_reasoning {
        summary.to_owned()
    } else {
        summary
            .lines()
            .filter(|line| !line.trim_start().starts_with("Reasoning:"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    if options.redact_secrets {
        redact_text(&kept)
    } else {
        kept
    }
}

/// Standalone tool calls carry secret-bearing arguments; IDs and names are
/// structural and preserved so call/result pairing stays valid.
fn transform_tool_call(
    call: &crate::model::StoredToolCall,
    options: &ExportOptions,
) -> crate::model::StoredToolCall {
    if !options.redact_secrets {
        return call.clone();
    }
    crate::model::StoredToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: redact_json(&call.arguments),
    }
}

/// Titles, errors, and cancellation reasons are free text: redact them.
fn transform_title(title: &str, options: &ExportOptions) -> String {
    if options.redact_secrets {
        redact_text(title)
    } else {
        title.to_owned()
    }
}

fn transform_diagnostic(text: &str, options: &ExportOptions) -> String {
    if options.redact_secrets {
        redact_text(text)
    } else {
        text.to_owned()
    }
}

/// Unknown event payloads are structured: recurse so nested secrets are
/// masked while IDs and structure survive.
fn transform_unknown(data: &serde_json::Value, options: &ExportOptions) -> serde_json::Value {
    if options.redact_secrets {
        redact_json(data)
    } else {
        data.clone()
    }
}

/// Best-effort, heuristic secret redaction for free text.
///
/// The string scanner masks values following common secret keys
/// (`token=…`, `"secret": "…"`) while leaving normal tool output intact.
/// Export redaction uses this scanner on every secret-bearing text field,
/// so tests place each sentinel beside a recognized key (as in
/// `token=<value>`).  Bare secrets with no nearby key are out of scope: the
/// exporter never learns an out-of-band secret value to search for.
///
/// It is intentionally dependency-free and conservative, but it is *not* a
/// parser, so treat it as a guardrail rather than a guarantee:
/// - nested quotes or escaped characters (`"secret": "a\\\"b"`) can defeat it;
/// - multi-line values are only masked up to the first line break;
/// - a value containing a comma, brace, bracket, or quote is truncated there.
///
/// Use [`redact_json`] for structured JSON/YAML values.  A future version
/// could replace this with a Tree-sitter or regex-based redactor for
/// structured output.
fn redact_text(value: &str) -> String {
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
            let mut value_start = value_start
                + result[value_start..]
                    .find(|character: char| {
                        !character.is_whitespace() && character != '"' && character != '\''
                    })
                    .unwrap_or(0);
            // Treat the bearer scheme as part of an Authorization value. The
            // old scanner replaced only `Bearer`, leaving the token itself in
            // the export.
            if key.eq_ignore_ascii_case("authorization") {
                let remaining = &result[value_start..];
                if remaining.len() >= 6
                    && remaining.is_char_boundary(6)
                    && remaining[..6].eq_ignore_ascii_case("bearer")
                    && remaining[6..]
                        .chars()
                        .next()
                        .is_some_and(char::is_whitespace)
                {
                    value_start += 6;
                    value_start += result[value_start..]
                        .find(|character: char| !character.is_whitespace())
                        .unwrap_or(0);
                }
            }
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
        serde_json::Value::String(text) => serde_json::Value::String(redact_text(text)),
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
            StoredContent::ToolCall { .. } | StoredContent::Opaque { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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

    /// A unique sentinel in every supported header/event field must vanish
    /// from the redacted export while call/result pairing and inclusion
    /// options keep working.
    #[test]
    fn redaction_covers_every_secret_bearing_field() {
        let directory = tempdir().unwrap();
        let sentinel = "sentinel-secret-9f3a1c";
        let mut session = Session::new(SessionMetadata::new(directory.path(), None, None));
        session.header_metadata.title = Some(format!("title holds token={sentinel}"));
        session.header_metadata.provider = Some(format!("provider=token={sentinel}"));
        session.header_metadata.model = Some(format!("model=token={sentinel}"));
        session.header_metadata.workspace_root =
            PathBuf::from(format!("/workspace/token={sentinel}"));
        session.metadata.title = Some(format!("title holds token={sentinel}"));
        session.append(SessionEvent::MetadataChange {
            title: Some(format!("title holds token={sentinel}")),
        });
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&Message::user(format!(
                "user text carries token={sentinel}"
            ))),
        });
        session.append(SessionEvent::AssistantMessage {
            message: StoredMessage {
                role: crate::model::StoredRole::Assistant,
                content: vec![
                    crate::model::StoredContent::Text {
                        text: format!("assistant text carries token={sentinel}"),
                    },
                    crate::model::StoredContent::Reasoning {
                        text: format!("assistant reasoning carries token={sentinel}"),
                    },
                    crate::model::StoredContent::Opaque {
                        provider: "mock".into(),
                        data: json!({"continuation": format!("token={sentinel}")}),
                    },
                    crate::model::StoredContent::ToolCall {
                        id: "embedded-1".into(),
                        name: "read".into(),
                        arguments: json!({
                            "token": sentinel,
                            "command": format!("curl -H 'Authorization: Bearer {sentinel}'")
                        }),
                    },
                ],
            },
        });
        session.append(SessionEvent::Reasoning {
            text: format!("standalone reasoning carries token={sentinel}"),
        });
        session.append(SessionEvent::ToolCall {
            call: crate::model::StoredToolCall {
                id: "call-1".into(),
                name: "bash".into(),
                arguments: json!({
                    "secret": sentinel,
                    "diagnostic": format!("Authorization: Bearer {sentinel}")
                }),
            },
        });
        session.append(SessionEvent::ToolResult {
            tool_call_id: "embedded-1".into(),
            content: format!("embedded result carries token={sentinel}"),
            is_error: true,
            tool_name: Some("read".into()),
        });
        session.append(SessionEvent::ToolResult {
            tool_call_id: "call-1".into(),
            content: format!("tool output carries token={sentinel}"),
            is_error: false,
            tool_name: Some("bash".into()),
        });
        session.append(SessionEvent::CompactionSummary {
            summary: format!("summary carries token={sentinel}"),
            compacted_through: 1,
        });
        session.append(SessionEvent::TurnCancelled {
            reason: format!("cancel carries token={sentinel}"),
        });
        session.append(SessionEvent::Error {
            message: format!("error carries token={sentinel}"),
        });
        session.append(SessionEvent::Unknown {
            kind: "future_event".into(),
            data: json!({"nested": {"password": sentinel}}),
        });

        let destination = directory.path().join("redacted.jsonl");
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
        assert!(
            !content.contains(sentinel),
            "redacted export must not contain the sentinel anywhere"
        );
        assert!(content.contains("[REDACTED]"));
        // The export still decodes and pairs calls with results.
        let loaded = decode_session(&content, &destination).unwrap();
        let mut calls = std::collections::HashSet::new();
        let mut results = Vec::new();
        for record in &loaded.events {
            match &record.event {
                SessionEvent::ToolCall { call } => {
                    calls.insert(call.id.clone());
                }
                SessionEvent::AssistantMessage { message } => {
                    for item in &message.content {
                        if let crate::model::StoredContent::ToolCall { id, .. } = item {
                            calls.insert(id.clone());
                        }
                    }
                }
                SessionEvent::ToolResult { tool_call_id, .. } => {
                    results.push(tool_call_id.clone());
                }
                _ => {}
            }
        }
        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(calls.contains(result), "orphaned result {result}");
        }

        // Inclusion options still behave: excluding reasoning/tool output
        // keeps the export valid and call/result pairing intact.
        let destination = directory.path().join("redacted-slim.jsonl");
        export_jsonl(
            &session,
            Some(&destination),
            &ExportOptions {
                include_reasoning: false,
                include_tool_output: false,
                redact_secrets: true,
                max_tool_output_bytes: Some(64),
            },
        )
        .unwrap();
        let slim = fs::read_to_string(&destination).unwrap();
        assert!(!slim.contains(sentinel));
        let slim_loaded = decode_session(&slim, &destination).unwrap();
        assert_eq!(slim_loaded.events.len(), loaded.events.len());
    }
}
