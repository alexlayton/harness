//! The `subagent` tool: delegates a self-contained task to a nested agent
//! loop and returns its final report.
//!
//! This module only defines the *tool surface* (spec, prompt metadata,
//! concurrency class). The actual runner lives behind [`SubagentRunner`] and
//! is injected by the host — `crates/tools` deliberately cannot depend on the
//! agent loop, mirroring how `ReadTool` receives its allowlist instead of
//! discovering skills itself. A registry without an injected runner still
//! advertises nothing: the tool is only registered when a runner exists, so
//! the model never sees a `subagent` schema it cannot call.

use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// Name under which the subagent tool registers.
pub const SUBAGENT_TOOL_NAME: &str = "subagent";

/// Spawns and drives a nested agent loop for one delegated task. Implemented
/// by the agent crate; kept here as a trait so the tool crate stays
/// dependency-free.
#[async_trait]
pub trait SubagentRunner: Send + Sync {
    /// Run one subagent to completion with a fresh context seeded only by
    /// `prompt`. Returns the subagent's final report text on success, or a
    /// human-readable error message (the caller turns it into an errored
    /// tool result).
    ///
    /// Implementations must treat `cancel` as authoritative: cancellation
    /// should stop the nested loop promptly and return an error mentioning
    /// it, so the parent's interrupt path synthesizes its usual cancelled
    /// tool result.
    async fn run(
        &self,
        description: &str,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<String, String>;
}

/// Extract the two string arguments of the tool. Shared by `execute` and
/// `call_summary` so previews never diverge from what execution reads.
pub(crate) fn parse_args(args: &Value) -> Result<(Option<&str>, Option<&str>), String> {
    let object = args.as_object().ok_or("arguments must be an object")?;
    let description = object.get("description").and_then(Value::as_str);
    let prompt = object.get("prompt").and_then(Value::as_str);
    Ok((description, prompt))
}

/// Build the concise preview used before/while a subagent runs: the
/// description when present, else a capped first line of the prompt.
pub(crate) fn preview(description: Option<&str>, prompt: Option<&str>) -> String {
    if let Some(description) = description.map(str::trim).filter(|d| !d.is_empty()) {
        return description.to_owned();
    }
    const MAX_CHARS: usize = 80;
    let Some(prompt) = prompt.map(str::trim).filter(|p| !p.is_empty()) else {
        return "subagent".to_owned();
    };
    let line = prompt.lines().next().unwrap_or(prompt);
    let mut out: String = line.chars().take(MAX_CHARS).collect();
    if line.chars().count() > MAX_CHARS {
        out.push('…');
    }
    out
}

/// The `subagent` tool. Wraps an injected [`SubagentRunner`]; classification
/// is [`Concurrency::Parallel`] so adjacent calls fan out concurrently while
/// everything else keeps program order around them.
pub struct SubagentTool {
    runner: std::sync::Arc<dyn SubagentRunner>,
}

impl SubagentTool {
    pub fn new(runner: std::sync::Arc<dyn SubagentRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: SUBAGENT_TOOL_NAME.to_owned(),
                description: "Delegate a self-contained task to a fresh subagent that works \
                              autonomously with its own tools and context window, then returns \
                              its final report. Use for parallelizable work across independent \
                              scopes (per-crate sweeps, multi-file research) or to isolate a \
                              large exploratory detour from this conversation."
                    .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "One-line label for the task (shown in the UI and session list), e.g. \"audit crates/tui error paths\"."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Complete, self-contained instructions for the task. The subagent starts with an empty context and cannot see this conversation, so include every path, constraint, and expected outcome it needs."
                        }
                    },
                    "required": ["description", "prompt"]
                }),
            },
            prompt: ToolPrompt::new(
                "delegate a self-contained task to a fresh subagent (own context window); returns its final report",
                vec![
                    "The subagent cannot see this conversation: write `prompt` fully \
                     self-contained, with explicit file paths, constraints, and the exact \
                     deliverable you expect back.",
                    "Batch multiple `subagent` calls in one response when their tasks are \
                     independent (e.g. one per crate/module) — they run concurrently. \
                     Sequence them when one depends on another's result.",
                    "Prefer subagents for breadth (surveys, audits, per-target repetition) \
                     over depth; do the focused edit yourself so you keep full context.",
                ],
            ),
        }
    }

    /// Not read-only, but designed for fan-out: adjacent same-tool calls run
    /// concurrently, bounded by the harness's subagent limit.
    fn concurrency(&self, _args: &Value) -> Concurrency {
        Concurrency::Parallel
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let (description, prompt) = match parse_args(&args) {
            Ok(parsed) => parsed,
            Err(message) => {
                return ToolOutput {
                    content: message,
                    is_error: true,
                    summary: SUBAGENT_TOOL_NAME.to_owned(),
                };
            }
        };
        let Some(prompt) = prompt.map(str::trim).filter(|p| !p.is_empty()) else {
            return ToolOutput {
                content: "`prompt` must be a non-empty string".to_owned(),
                is_error: true,
                summary: SUBAGENT_TOOL_NAME.to_owned(),
            };
        };
        let summary = preview(description, Some(prompt));
        // The description is display metadata; trim it defensively so odd
        // model output cannot smuggle newlines into titles/previews.
        let description = description
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .unwrap_or(summary.trim_end_matches('…'))
            .to_owned();
        match self.runner.run(&description, prompt, cancel).await {
            Ok(report) => ToolOutput {
                content: report,
                is_error: false,
                summary,
            },
            Err(error) => ToolOutput {
                content: format!("subagent failed: {error}"),
                is_error: true,
                summary,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingRunner {
        calls: Mutex<Vec<(String, String)>>,
        reply: Mutex<Option<Result<String, String>>>,
    }

    #[async_trait]
    impl SubagentRunner for RecordingRunner {
        async fn run(
            &self,
            description: &str,
            prompt: &str,
            cancel: CancellationToken,
        ) -> Result<String, String> {
            let _ = cancel;
            self.calls
                .lock()
                .unwrap()
                .push((description.to_owned(), prompt.to_owned()));
            self.reply
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| Ok("report body".to_owned()))
        }
    }

    fn tool() -> SubagentTool {
        SubagentTool::new(std::sync::Arc::new(RecordingRunner::default()))
    }

    #[test]
    fn spec_has_schema_prompt_and_parallel_class() {
        let spec = tool().spec();
        assert_eq!(spec.definition.name, "subagent");
        assert_eq!(
            spec.definition.parameters["required"],
            json!(["description", "prompt"])
        );
        assert!(!spec.prompt.guidelines.is_empty());
        assert!(spec.prompt.snippet.is_some());
        assert_eq!(tool().concurrency(&json!({})), Concurrency::Parallel);
    }

    #[tokio::test]
    async fn executes_runner_and_returns_report() {
        let tool = tool();
        let output = tool
            .execute(
                json!({"description": "audit tui", "prompt": "scan crates/tui"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error);
        assert_eq!(output.content, "report body");
        assert_eq!(output.summary, "audit tui");
    }

    #[tokio::test]
    async fn missing_or_empty_prompt_is_an_errored_result() {
        let tool = tool();
        for args in [json!({}), json!({"prompt": ""}), json!({"prompt": 3})] {
            let output = tool.execute(args, CancellationToken::new()).await;
            assert!(output.is_error, "{output:?}");
        }
    }

    #[tokio::test]
    async fn runner_errors_surface_as_errored_results() {
        let runner = RecordingRunner::default();
        *runner.reply.lock().unwrap() = Some(Err("provider down".to_owned()));
        let tool = SubagentTool::new(std::sync::Arc::new(runner));
        let output = tool
            .execute(
                json!({"description": "x", "prompt": "y"}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("provider down"));
    }

    #[test]
    fn preview_prefers_description_over_prompt() {
        assert_eq!(
            preview(Some("label"), Some("long prompt")),
            "label".to_owned()
        );
        let long = preview(None, Some(&"x".repeat(120)));
        assert_eq!(long.chars().count(), 81);
        assert!(long.ends_with('…'));
        assert_eq!(preview(None, None), "subagent");
    }

    #[test]
    fn call_summary_uses_description() {
        assert_eq!(
            super::super::call_summary(
                "subagent",
                &json!({"description": "audit llm", "prompt": "go"})
            ),
            "subagent: audit llm"
        );
    }
}
