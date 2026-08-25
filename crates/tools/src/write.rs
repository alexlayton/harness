use super::file_mutation::{atomic_write, with_file_mutation_lock};
use super::{
    Tool, ToolOutput, ToolPrompt, ToolSpec, normalize_workspace_root, resolve_workspace_path,
};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::fs;
use tokio_util::sync::CancellationToken;

pub struct WriteTool {
    workspace_root: Option<PathBuf>,
}

impl WriteTool {
    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: Some(normalize_workspace_root(root)),
        }
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
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
            },
            prompt: ToolPrompt::new(
                "Create or overwrite files with complete contents",
                ["Use write only for new files or complete rewrites.".to_owned()],
            ),
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

        let full_path =
            match resolve_workspace_path(&path, self.workspace_root.as_deref(), false).await {
                Ok(path) => path,
                Err(message) => {
                    return error(
                        &format!("write {path}"),
                        &format!("cannot write {path}: {message}"),
                    );
                }
            };
        let summary = format!("write {path}");
        let Some(result) = with_file_mutation_lock(&full_path, &cancel, || async {
            if cancel.is_cancelled() {
                return Err("cancelled".to_owned());
            }
            if let Some(parent) = full_path.parent()
                && let Err(io_error) = fs::create_dir_all(parent).await
            {
                return Err(format!("cannot create parent directory: {io_error}"));
            }
            if cancel.is_cancelled() {
                return Err("cancelled".to_owned());
            }
            atomic_write(&full_path, content.as_bytes(), &cancel)
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
        let tool = WriteTool::with_workspace_root(dir.path());
        let output = tool
            .execute(
                json!({"path": "a/b/file.txt", "content":"first"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/file.txt")).unwrap(),
            "first"
        );
        tool.execute(
            json!({"path": "a/b/file.txt", "content":"second"}),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/file.txt")).unwrap(),
            "second"
        );
    }
}
