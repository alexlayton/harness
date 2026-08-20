//! Canonical JSONL codec for Harness sessions.
//!
//! The codec deliberately keeps the envelope small and human-readable.  A
//! header is followed by one append-only event per line.  Unknown event kinds
//! are retained as `SessionEvent::Unknown` and are emitted with their original
//! kind/data on export.

use crate::error::{Result, SessionError};
use crate::model::{
    EventId, FORMAT_VERSION, Session, SessionEvent, SessionEventRecord, SessionId, SessionMetadata,
    StoredMessage, StoredToolCall, UsageSummary,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Serialize)]
struct HeaderEnvelope<'a> {
    version: u32,
    #[serde(rename = "type")]
    kind: &'static str,
    session_id: String,
    timestamp: &'a str,
    data: &'a SessionMetadata,
}

#[derive(Serialize)]
struct EventEnvelope {
    version: u32,
    #[serde(rename = "type")]
    kind: String,
    session_id: String,
    event_id: String,
    sequence: u64,
    timestamp: String,
    data: Value,
}

#[derive(Deserialize)]
struct RawEnvelope {
    version: Option<u32>,
    #[serde(rename = "type")]
    kind: Option<String>,
    session_id: Option<String>,
    #[serde(default)]
    event_id: Option<String>,
    #[serde(default)]
    sequence: Option<u64>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    data: Value,
}

pub fn encode_header(metadata: &SessionMetadata) -> Result<String> {
    let envelope = HeaderEnvelope {
        version: FORMAT_VERSION,
        kind: "session",
        session_id: metadata.id.to_string(),
        timestamp: &metadata.created_at,
        data: metadata,
    };
    serde_json::to_string(&envelope).map_err(|source| SessionError::Json {
        path: "<memory>".into(),
        line: 1,
        source,
    })
}

pub fn encode_record(session_id: SessionId, record: &SessionEventRecord) -> Result<String> {
    let envelope = EventEnvelope {
        version: FORMAT_VERSION,
        kind: record.event.kind().to_owned(),
        session_id: session_id.to_string(),
        event_id: record.id.to_string(),
        sequence: record.sequence,
        timestamp: record.timestamp.clone(),
        data: event_data(&record.event)?,
    };
    serde_json::to_string(&envelope).map_err(|source| SessionError::Json {
        path: "<memory>".into(),
        line: 1,
        source,
    })
}

/// Serialize a complete session using the canonical representation.
pub fn encode_session(session: &Session) -> Result<String> {
    let mut lines = Vec::with_capacity(session.events.len() + 1);
    lines.push(encode_header(session.header_metadata())?);
    for record in &session.events {
        lines.push(encode_record(session.id(), record)?);
    }
    Ok(format!("{}\n", lines.join("\n")))
}

/// Parse a complete JSONL session held in memory.  Unlike file loading this
/// function treats a malformed final line as an error because the caller has
/// no file-write boundary with which to identify a partial write.
pub fn decode_session(content: &str, source: impl AsRef<Path>) -> Result<Session> {
    decode_session_lines(content, source.as_ref(), false).map(|(session, _)| session)
}

/// Parse a session file's contents.  A malformed, unterminated final line is
/// treated as an interrupted append and ignored.  Any malformed line in the
/// middle, or a malformed line terminated by a newline, remains an error.
pub fn decode_session_file(content: &str, source: impl AsRef<Path>) -> Result<(Session, bool)> {
    decode_session_lines(content, source.as_ref(), true)
}

fn decode_session_lines(
    content: &str,
    source: &Path,
    recover_trailing: bool,
) -> Result<(Session, bool)> {
    let mut lines = content.split_inclusive('\n').collect::<Vec<_>>();
    // The trailing fragment is kept in the normal loop: it is only dropped when
    // its JSON is actually malformed, while a valid final line is accepted.
    let has_unterminated_tail = lines.last().is_some_and(|line| !line.ends_with('\n'));
    if content.is_empty() {
        return Err(SessionError::InvalidLine {
            path: source.to_path_buf(),
            line: 1,
            message: "session file is empty".into(),
        });
    }

    let last_line = content.lines().count();
    let mut session = None::<Session>;
    let mut recovered = false;
    let mut records = Vec::new();

    for (index, raw_line) in lines.drain(..).enumerate() {
        let line_number = index + 1;
        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        if line.trim().is_empty() {
            return Err(SessionError::InvalidLine {
                path: source.to_path_buf(),
                line: line_number,
                message: "blank lines are not valid session records".into(),
            });
        }

        let raw: RawEnvelope = match serde_json::from_str(line) {
            Ok(raw) => raw,
            Err(_) if recover_trailing && has_unterminated_tail && line_number == last_line => {
                recovered = true;
                break;
            }
            Err(source_error) => {
                return Err(SessionError::Json {
                    path: source.to_path_buf(),
                    line: line_number,
                    source: source_error,
                });
            }
        };

        let version = raw.version.ok_or_else(|| SessionError::InvalidLine {
            path: source.to_path_buf(),
            line: line_number,
            message: "missing version".into(),
        })?;
        if version > FORMAT_VERSION {
            return Err(SessionError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let kind = raw.kind.ok_or_else(|| SessionError::InvalidLine {
            path: source.to_path_buf(),
            line: line_number,
            message: "missing type".into(),
        })?;

        if session.is_none() {
            if kind != "session" {
                return Err(SessionError::InvalidLine {
                    path: source.to_path_buf(),
                    line: line_number,
                    message: "first record must have type `session`".into(),
                });
            }
            let session_id = parse_session_id(raw.session_id.as_deref(), source, line_number)?;
            let mut metadata: SessionMetadata =
                serde_json::from_value(raw.data).map_err(|source_error| SessionError::Json {
                    path: source.to_path_buf(),
                    line: line_number,
                    source: source_error,
                })?;
            if metadata.id != session_id {
                return Err(SessionError::InvalidLine {
                    path: source.to_path_buf(),
                    line: line_number,
                    message: format!(
                        "header session_id {session_id} does not match metadata id {}",
                        metadata.id
                    ),
                });
            }
            if metadata.format_version > FORMAT_VERSION {
                return Err(SessionError::UnsupportedVersion {
                    found: metadata.format_version,
                    supported: FORMAT_VERSION,
                });
            }
            // A missing timestamp is tolerated for very early in-memory
            // exports, but current writers always provide one.
            if metadata.created_at.trim().is_empty() {
                metadata.created_at = raw.timestamp.unwrap_or_default();
            }
            if metadata.updated_at.trim().is_empty() {
                metadata.updated_at = metadata.created_at.clone();
            }
            session = Some(Session {
                header_metadata: metadata.clone(),
                metadata,
                events: Vec::new(),
                path: None,
            });
            continue;
        }

        if kind == "session" {
            return Err(SessionError::InvalidLine {
                path: source.to_path_buf(),
                line: line_number,
                message: "session header may only occur on the first line".into(),
            });
        }
        let expected_session_id = session.as_ref().expect("session initialized").id();
        let session_id = parse_session_id(raw.session_id.as_deref(), source, line_number)?;
        if session_id != expected_session_id {
            return Err(SessionError::InvalidLine {
                path: source.to_path_buf(),
                line: line_number,
                message: format!(
                    "event belongs to session {session_id}, expected {expected_session_id}"
                ),
            });
        }
        let event_id = raw
            .event_id
            .as_deref()
            .ok_or_else(|| SessionError::InvalidLine {
                path: source.to_path_buf(),
                line: line_number,
                message: "event is missing event_id".into(),
            })
            .and_then(EventId::parse)?;
        let sequence = raw.sequence.ok_or_else(|| SessionError::InvalidLine {
            path: source.to_path_buf(),
            line: line_number,
            message: "event is missing sequence".into(),
        })?;
        let timestamp = raw.timestamp.ok_or_else(|| SessionError::InvalidLine {
            path: source.to_path_buf(),
            line: line_number,
            message: "event is missing timestamp".into(),
        })?;
        let event = decode_event(&kind, raw.data).map_err(|message| SessionError::InvalidLine {
            path: source.to_path_buf(),
            line: line_number,
            message,
        })?;
        records.push(SessionEventRecord {
            id: event_id,
            sequence,
            timestamp,
            event,
        });
    }

    let mut session = session.ok_or_else(|| SessionError::InvalidLine {
        path: source.to_path_buf(),
        line: 1,
        message: "session file is empty".into(),
    })?;
    crate::model::validate_events(&records)?;
    for record in records {
        session.append_record(record);
    }
    Ok((session, recovered))
}

fn parse_session_id(value: Option<&str>, source: &Path, line: usize) -> Result<SessionId> {
    let value = value.ok_or_else(|| SessionError::InvalidLine {
        path: source.to_path_buf(),
        line,
        message: "missing session_id".into(),
    })?;
    SessionId::parse(value).map_err(|error| SessionError::InvalidLine {
        path: source.to_path_buf(),
        line,
        message: error.to_string(),
    })
}

fn event_data(event: &SessionEvent) -> Result<Value> {
    let value = match event {
        SessionEvent::UserMessage { message } | SessionEvent::AssistantMessage { message } => {
            serde_json::to_value(message)
        }
        SessionEvent::Reasoning { text } => Ok(json!({ "text": text })),
        SessionEvent::ToolCall { call } => serde_json::to_value(call),
        SessionEvent::ToolResult {
            tool_call_id,
            content,
            is_error,
            tool_name,
        } => Ok(json!({
            "tool_call_id": tool_call_id,
            "content": content,
            "is_error": is_error,
            "tool_name": tool_name,
        })),
        SessionEvent::CompactionSummary {
            summary,
            compacted_through,
        } => Ok(json!({
            "summary": summary,
            "compacted_through": compacted_through,
        })),
        SessionEvent::ModelChange { provider, model } => Ok(json!({
            "provider": provider,
            "model": model,
        })),
        SessionEvent::Usage { usage } => serde_json::to_value(usage),
        SessionEvent::MetadataChange { title } => Ok(json!({ "title": title })),
        SessionEvent::TurnCancelled { reason } => Ok(json!({ "reason": reason })),
        SessionEvent::Error { message } => Ok(json!({ "message": message })),
        SessionEvent::Unknown { data, .. } => Ok(data.clone()),
    };
    value.map_err(|source| SessionError::Json {
        path: "<memory>".into(),
        line: 1,
        source,
    })
}

fn decode_event(kind: &str, data: Value) -> std::result::Result<SessionEvent, String> {
    match kind {
        "user_message" => decode_message(data).map(|message| SessionEvent::UserMessage { message }),
        "assistant_message" => {
            decode_message(data).map(|message| SessionEvent::AssistantMessage { message })
        }
        "reasoning" => {
            #[derive(Deserialize)]
            struct Data {
                text: String,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::Reasoning { text: data.text })
                .map_err(|error| error.to_string())
        }
        "tool_call" => serde_json::from_value::<StoredToolCall>(data)
            .map(|call| SessionEvent::ToolCall { call })
            .map_err(|error| error.to_string()),
        "tool_result" => {
            #[derive(Deserialize)]
            struct Data {
                tool_call_id: String,
                content: String,
                #[serde(default)]
                is_error: bool,
                #[serde(default)]
                tool_name: Option<String>,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::ToolResult {
                    tool_call_id: data.tool_call_id,
                    content: data.content,
                    is_error: data.is_error,
                    tool_name: data.tool_name,
                })
                .map_err(|error| error.to_string())
        }
        "compaction" => {
            #[derive(Deserialize)]
            struct Data {
                summary: String,
                compacted_through: u64,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::CompactionSummary {
                    summary: data.summary,
                    compacted_through: data.compacted_through,
                })
                .map_err(|error| error.to_string())
        }
        "model_change" => {
            #[derive(Deserialize)]
            struct Data {
                provider: String,
                model: String,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::ModelChange {
                    provider: data.provider,
                    model: data.model,
                })
                .map_err(|error| error.to_string())
        }
        "usage" => serde_json::from_value::<UsageSummary>(data)
            .map(|usage| SessionEvent::Usage { usage })
            .map_err(|error| error.to_string()),
        "metadata_change" => {
            #[derive(Deserialize)]
            struct Data {
                #[serde(default)]
                title: Option<String>,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::MetadataChange { title: data.title })
                .map_err(|error| error.to_string())
        }
        "turn_cancelled" => {
            #[derive(Deserialize)]
            struct Data {
                reason: String,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::TurnCancelled {
                    reason: data.reason,
                })
                .map_err(|error| error.to_string())
        }
        "error" => {
            #[derive(Deserialize)]
            struct Data {
                message: String,
            }
            serde_json::from_value::<Data>(data)
                .map(|data| SessionEvent::Error {
                    message: data.message,
                })
                .map_err(|error| error.to_string())
        }
        _ => Ok(SessionEvent::Unknown {
            kind: kind.to_owned(),
            data,
        }),
    }
}

fn decode_message(data: Value) -> std::result::Result<StoredMessage, String> {
    serde_json::from_value(data).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SessionEvent, SessionMetadata, StoredMessage};
    use llm::Message;

    #[test]
    fn unterminated_trailing_fragment_is_ignored_but_middle_corruption_is_not() {
        let encoded = encode_session(&Session::new(SessionMetadata::new(
            "/tmp/project",
            Some("p".into()),
            Some("m".into()),
        )))
        .unwrap();
        let corrupted = format!("{encoded}{{");
        let (session, recovered) = decode_session_file(&corrupted, "<memory>").unwrap();
        assert!(recovered);
        assert_eq!(
            session.events,
            Session::new(SessionMetadata::new(
                "/tmp/project",
                Some("p".into()),
                Some("m".into()),
            ))
            .events
        );
        let middle_corruption = encoded.replacen("\n", "\n{\n", 1);
        assert!(decode_session_file(&middle_corruption, "<memory>").is_err());
    }

    #[test]
    fn canonical_round_trip_keeps_unknown_events() {
        let mut session = Session::new(SessionMetadata::new(
            "/tmp/project",
            Some("p".into()),
            Some("m".into()),
        ));
        session.append(SessionEvent::UserMessage {
            message: StoredMessage::from_llm(&Message::user("hello")),
        });
        session.append(SessionEvent::Unknown {
            kind: "future_event".into(),
            data: json!({"value": 1}),
        });
        let encoded = encode_session(&session).unwrap();
        let decoded = decode_session(&encoded, "<memory>").unwrap();
        assert_eq!(decoded.events, session.events);
        assert!(encoded.ends_with('\n'));
    }
}
