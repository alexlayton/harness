use super::file_mutation::with_file_mutation_lock;
use super::{Tool, ToolOutput};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::fs;
use tokio_util::sync::CancellationToken;

pub struct WriteTool;

impl WriteTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".into(),
            description: "Create or fully overwrite a text file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path relative to the working directory" },
                    "content": { "type": "string", "description": "Complete file contents" }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let path = match args.get("path").and_then(Value::as_str) {
            Some(path) if !path.is_empty() => path.to_owned(),
            _ => return error("write", "missing required argument: path"),
        };
        let Some(content) = args.get("content").and_then(Value::as_str) else {
            return error(
                &format!("write {path}"),
                "missing required argument: content",
            );
        };
        if cancel.is_cancelled() {
            return error(&format!("write {path}"), "cancelled");
        }

        let full_path = resolve_path(&path);
        let summary = format!("write {path}");
        let Some(result) = with_file_mutation_lock(&full_path, &cancel, || async {
            if let Some(parent) = full_path.parent()
                && let Err(io_error) = fs::create_dir_all(parent).await
            {
                return Err(format!("cannot create parent directory: {io_error}"));
            }
            fs::write(&full_path, content.as_bytes())
                .await
                .map_err(|io_error| format!("cannot write {path}: {io_error}"))
        })
        .await
        else {
            return error(&summary, "cancelled");
        };
        if let Err(message) = result {
            return error(&summary, &message);
        }
        ToolOutput {
            content: format!("wrote {} bytes to {path}", content.len()),
            is_error: false,
            summary,
        }
    }
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
    use tempfile::tempdir;

    #[tokio::test]
    async fn creates_parents_and_overwrites() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("a/b/file.txt");
        let output = WriteTool
            .execute(
                json!({"path": path, "content":"first"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");
        WriteTool
            .execute(
                json!({"path": path, "content":"second"}),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }
}
