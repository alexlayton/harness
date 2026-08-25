use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec};
use crate::find::{DEFAULT_GREP_LIMIT, FileSearchIndex, MAX_GREP_LIMIT, format_grep_output};
use async_trait::async_trait;
use fff_search::GrepMode;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Content search over the same watched FFF index that powers `find`.  Uses
/// fff's ripgrep-compatible engine natively, so the model never needs a
/// `bash` pipeline to grep the workspace.
pub struct GrepTool {
    index: Arc<FileSearchIndex>,
}

impl GrepTool {
    pub fn new(index: Arc<FileSearchIndex>) -> Self {
        Self { index }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "grep".into(),
                description: "Search file contents for a pattern using a fast indexed, ripgrep-compatible engine. Results respect repository ignore rules and are returned as workspace-relative paths with line numbers. Patterns may include fff constraints (e.g. `TODO *.rs` scopes to Rust files) and `path` restricts the search to a directory.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Search pattern; interpreted as regex by default, literal text in plain mode"
                        },
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Optional workspace-relative directory scope. Omit this field to search the entire workspace; when provided it must not be empty."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["regex", "plain", "fuzzy"],
                            "default": "regex",
                            "description": "How to interpret the pattern: regex (default, ripgrep semantics), plain (literal text, fastest), or fuzzy"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_GREP_LIMIT,
                            "default": DEFAULT_GREP_LIMIT,
                            "description": "Maximum number of matches to return"
                        },
                        "context": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                            "default": 0,
                            "description": "Number of context lines before and after each match"
                        }
                    },
                    "required": ["pattern"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Search file contents for a pattern",
                [
                    "Use grep for content search instead of bash grep, ripgrep, or shell pipelines.".to_owned(),
                    "Match lines are formatted path:line:content; context lines use path-line-content.".to_owned(),
                    "Read-only: independent searches may be batched in one response and run concurrently.".to_owned(),
                ],
            ),
        }
    }

    fn concurrency(&self, _args: &Value) -> Concurrency {
        Concurrency::ReadOnly
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let pattern = match args.get("pattern").and_then(Value::as_str) {
            Some(pattern) if !pattern.trim().is_empty() => pattern.trim().to_owned(),
            _ => return error("grep", "pattern must be a non-empty string"),
        };
        let path = match args.get("path") {
            None => None,
            Some(Value::String(path)) if !path.trim().is_empty() => Some(path.clone()),
            Some(_) => return error("grep", "path must be a non-empty string when provided"),
        };
        let mode = match args.get("mode").and_then(Value::as_str).unwrap_or("regex") {
            "regex" => GrepMode::Regex,
            "plain" => GrepMode::PlainText,
            "fuzzy" => GrepMode::Fuzzy,
            other => {
                return error(
                    "grep",
                    &format!("unknown mode `{other}` (expected regex, plain, or fuzzy)"),
                );
            }
        };
        let limit = match args.get("limit") {
            None => DEFAULT_GREP_LIMIT,
            Some(value) => match value.as_u64() {
                Some(value) if value > 0 && value <= MAX_GREP_LIMIT as u64 => value as usize,
                _ => {
                    return error(
                        "grep",
                        &format!(
                            "limit must be a positive integer no greater than {MAX_GREP_LIMIT}"
                        ),
                    );
                }
            },
        };
        let context = match args.get("context") {
            None => 0,
            Some(value) => match value.as_u64() {
                Some(value) if value <= 20 => value as usize,
                _ => return error("grep", "context must be an integer between 0 and 20"),
            },
        };
        if cancel.is_cancelled() {
            return error(&format!("grep {pattern}"), "cancelled");
        }
        let summary = match path.as_deref() {
            Some(path) => format!("grep {pattern} in {path}"),
            None => format!("grep {pattern}"),
        };
        let result = self
            .index
            .grep(pattern.clone(), path, limit, context, mode, cancel)
            .await;
        match result {
            Ok(raw) => {
                let content = format_grep_output(raw, &pattern, limit, MAX_GREP_LIMIT);
                ToolOutput {
                    content,
                    is_error: false,
                    summary,
                }
            }
            Err(message) => error(&summary, &message),
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
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn scope_schema_requires_a_non_empty_string_when_present() {
        let directory = tempdir().unwrap();
        let tool = GrepTool::new(Arc::new(FileSearchIndex::new(directory.path()).unwrap()));
        let parameters = tool.spec().definition.parameters;

        assert_eq!(parameters["properties"]["path"]["minLength"], 1);
        assert!(
            parameters["properties"]["path"]["description"]
                .as_str()
                .unwrap()
                .contains("Omit this field")
        );
    }

    #[tokio::test]
    async fn greps_without_a_shell() {
        let directory = tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::write(
            directory.path().join("src/main.rs"),
            "fn main() {\n    let greeting = \"hello\";\n    println!(\"{greeting}\");\n}\n",
        )
        .unwrap();
        std::fs::write(
            directory.path().join("src/lib.rs"),
            "fn helper() {\n    let greeting = \"hi\";\n}\n",
        )
        .unwrap();

        let tool = GrepTool::new(Arc::new(FileSearchIndex::new(directory.path()).unwrap()));
        let output = tool
            .execute(json!({"pattern": "greeting"}), CancellationToken::new())
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(
            output
                .content
                .contains("src/main.rs:2:let greeting = \"hello\";")
        );
        assert!(
            output
                .content
                .contains("src/lib.rs:2:let greeting = \"hi\";")
        );
        assert_eq!(output.summary, "grep greeting");
    }

    #[tokio::test]
    async fn respects_directory_scope_and_regex_mode() {
        let directory = tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::create_dir_all(directory.path().join("tests")).unwrap();
        std::fs::write(
            directory.path().join("src/main.rs"),
            "fn main() {\n    let x = 1;\n}\n",
        )
        .unwrap();
        std::fs::write(directory.path().join("tests/test.rs"), "let x = 2;\n").unwrap();

        let tool = GrepTool::new(Arc::new(FileSearchIndex::new(directory.path()).unwrap()));
        let scoped = tool
            .execute(
                json!({"pattern": "let [a-z]+ =", "path": "src"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!scoped.is_error, "{}", scoped.content);
        assert!(scoped.content.contains("src/main.rs"));
        assert!(!scoped.content.contains("tests/test.rs"));

        let outside = tool
            .execute(
                json!({"pattern": "let [a-z]+ =", "path": "../outside"}),
                CancellationToken::new(),
            )
            .await;
        assert!(outside.is_error);
        assert!(outside.content.contains("outside"));
    }

    #[tokio::test]
    async fn reports_no_matches_and_unknown_modes() {
        let directory = tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "hello\n").unwrap();
        let tool = GrepTool::new(Arc::new(FileSearchIndex::new(directory.path()).unwrap()));
        let none = tool
            .execute(
                json!({"pattern": "missing-token"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!none.is_error);
        assert!(none.content.contains("No matches"));

        let bad_mode = tool
            .execute(
                json!({"pattern": "hello", "mode": "regexx"}),
                CancellationToken::new(),
            )
            .await;
        assert!(bad_mode.is_error);
        assert!(bad_mode.content.contains("unknown mode"));
    }
}
