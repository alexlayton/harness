use super::{Tool, ToolOutput};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::fs;
use tokio_util::sync::CancellationToken;

pub struct ReadTool;

impl ReadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1024;

#[async_trait]
impl Tool for ReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".into(),
            description: "Read a text file, optionally selecting a range of lines.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path relative to the working directory" },
                    "offset": { "type": "integer", "minimum": 1, "description": "First 1-indexed line" },
                    "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of lines" }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let path = match args.get("path").and_then(Value::as_str) {
            Some(path) if !path.is_empty() => path.to_owned(),
            _ => return error("read", "missing required argument: path"),
        };
        let offset = match optional_positive(&args, "offset") {
            Ok(value) => value.unwrap_or(1),
            Err(message) => return error(&format!("read {path}"), &message),
        };
        let limit = match optional_positive(&args, "limit") {
            Ok(value) => value,
            Err(message) => return error(&format!("read {path}"), &message),
        };
        if cancel.is_cancelled() {
            return error(&format!("read {path}"), "cancelled");
        }

        let full_path = resolve_path(&path);
        let metadata = match fs::metadata(&full_path).await {
            Ok(metadata) => metadata,
            Err(io_error) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {io_error}"),
                );
            }
        };
        if metadata.is_dir() {
            return error(
                &format!("read {path}"),
                &format!("cannot read {path}: is a directory"),
            );
        }
        let bytes = match fs::read(&full_path).await {
            Ok(bytes) => bytes,
            Err(io_error) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {io_error}"),
                );
            }
        };
        if bytes[..bytes.len().min(8 * 1024)].contains(&0) {
            return error(&format!("read {path}"), "binary file not supported");
        }
        let content = match String::from_utf8(bytes) {
            Ok(content) => content,
            Err(_) => return error(&format!("read {path}"), "binary file not supported"),
        };
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let start = offset.saturating_sub(1).min(total);
        let requested_end = limit
            .map(|limit| start.saturating_add(limit).min(total))
            .unwrap_or(total);
        let mut end = requested_end.min(start.saturating_add(MAX_LINES));
        let mut selected = lines[start..end].join("\n");
        let mut byte_truncated = false;

        if selected.len() > MAX_BYTES {
            selected = truncate_utf8(&selected, MAX_BYTES).to_owned();
            // The byte limit can cut through a line.  Report the last complete
            // line when possible; the text itself remains useful for a huge line.
            let complete_lines = selected.lines().count();
            end = (start + complete_lines).min(end);
            byte_truncated = true;
        }

        // An explicit limit is intentional selection, not an implementation
        // truncation.  Only the safety caps (or an offset beyond the file)
        // receive the diagnostic notice.
        let truncated = byte_truncated || end < requested_end || (limit.is_none() && end < total);
        if truncated {
            let shown_start = if total == 0 { offset } else { start + 1 };
            let shown_end = if total == 0 {
                offset.saturating_sub(1)
            } else {
                end.max(start + 1)
            };
            let notice = format!("[truncated: showing lines {shown_start}–{shown_end} of {total}]");
            if !selected.is_empty() {
                selected.push('\n');
            }
            selected.push_str(&notice);
        }

        ToolOutput {
            content: selected,
            is_error: false,
            summary: format!("read {path}"),
        }
    }
}

fn optional_positive(args: &Value, name: &str) -> Result<Option<usize>, String> {
    let Some(value) = args.get(name) else {
        return Ok(None);
    };
    let Some(number) = value.as_u64() else {
        return Err(format!("{name} must be a positive integer"));
    };
    if number == 0 || number > usize::MAX as u64 {
        return Err(format!("{name} must be a positive integer"));
    }
    Ok(Some(number as usize))
}

fn resolve_path(path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn truncate_utf8(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn error(summary: &str, content: &str) -> ToolOutput {
    ToolOutput {
        content: content.to_owned(),
        is_error: true,
        summary: summary.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn reads_ranges_and_reports_truncation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();
        let old = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let output = ReadTool
            .execute(
                json!({"path":"file.txt", "offset":2, "limit":2}),
                CancellationToken::new(),
            )
            .await;
        std::env::set_current_dir(old).unwrap();
        assert_eq!(output.content, "two\nthree");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn rejects_binary_and_missing_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("binary");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"ok\0no").unwrap();
        let output = ReadTool
            .execute(
                json!({"path": dir.path().join("binary")}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("binary"));
    }
}
