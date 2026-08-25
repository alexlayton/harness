use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec};
use crate::find::{FileSearchIndex, MAX_GREP_LIMIT, format_grep_output};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const DEFAULT_LIMIT: usize = 60;
const MAX_PATTERNS: usize = 8;

/// One-pass literal search for several aliases or symbol variants.
pub struct MultiGrepTool {
    index: Arc<FileSearchIndex>,
}

impl MultiGrepTool {
    pub fn new(index: Arc<FileSearchIndex>) -> Self {
        Self { index }
    }
}

#[async_trait]
impl Tool for MultiGrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "multigrep".into(),
                description: "Search once for several literal patterns with OR semantics. Use for aliases, renamed symbols, or spelling variants; use grep for regex or fuzzy search. Results respect repository ignores and path scope.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_PATTERNS,
                            "uniqueItems": true,
                            "items": { "type": "string", "minLength": 1 },
                            "description": "One to eight unique non-empty literal patterns (OR semantics, smart case)"
                        },
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Optional workspace-relative directory scope"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_GREP_LIMIT,
                            "default": DEFAULT_LIMIT,
                            "description": "Hard aggregate match limit"
                        },
                        "context": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                            "default": 0,
                            "description": "Context lines before and after matches"
                        }
                    },
                    "required": ["patterns"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Search multiple literal patterns",
                ["Use multigrep for aliases, renamed symbols, or literal variants.".to_owned()],
            ),
        }
    }

    fn concurrency(&self, _args: &Value) -> Concurrency {
        Concurrency::ReadOnly
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let values = match args.get("patterns").and_then(Value::as_array) {
            Some(values) if !values.is_empty() && values.len() <= MAX_PATTERNS => values,
            _ => return error("multigrep", "patterns must contain 1 to 8 literal strings"),
        };
        let mut seen = HashSet::with_capacity(values.len());
        let mut patterns = Vec::with_capacity(values.len());
        for value in values {
            let Some(pattern) = value.as_str() else {
                return error("multigrep", "every pattern must be a string");
            };
            if pattern.trim().is_empty() {
                return error("multigrep", "patterns must not be empty or whitespace-only");
            }
            if !seen.insert(pattern.to_owned()) {
                return error("multigrep", "patterns must not contain duplicates");
            }
            patterns.push(pattern.to_owned());
        }
        let path = match args.get("path") {
            None => None,
            Some(Value::String(path)) if !path.trim().is_empty() => Some(path.clone()),
            Some(_) => return error("multigrep", "path must be a non-empty string when provided"),
        };
        let limit = match args.get("limit") {
            None => DEFAULT_LIMIT,
            Some(value) => match value.as_u64() {
                Some(value) if value > 0 && value <= MAX_GREP_LIMIT as u64 => value as usize,
                _ => return error("multigrep", "limit must be an integer between 1 and 500"),
            },
        };
        let context = match args.get("context") {
            None => 0,
            Some(value) => match value.as_u64() {
                Some(value) if value <= 20 => value as usize,
                _ => return error("multigrep", "context must be an integer between 0 and 20"),
            },
        };
        let summary = match path.as_deref() {
            Some(path) => format!("multigrep {} patterns in {path}", patterns.len()),
            None => format!("multigrep {} patterns", patterns.len()),
        };
        match self
            .index
            .multi_grep(patterns, path, limit, context, cancel)
            .await
        {
            Ok(raw) => ToolOutput {
                content: format_grep_output(raw, "the requested literals", limit, MAX_GREP_LIMIT),
                is_error: false,
                summary,
            },
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
    use tempfile::tempdir;

    #[tokio::test]
    async fn finds_union_in_one_call_and_rejects_duplicates() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "old_name\nnew_name\nother\n").unwrap();
        let tool = MultiGrepTool::new(Arc::new(FileSearchIndex::new(dir.path()).unwrap()));
        let output = tool
            .execute(
                json!({"patterns": ["old_name", "new_name"]}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("a.txt:1:old_name"));
        assert!(output.content.contains("a.txt:2:new_name"));

        let duplicate = tool
            .execute(
                json!({"patterns": ["same", "same"]}),
                CancellationToken::new(),
            )
            .await;
        assert!(duplicate.is_error);
    }
}
