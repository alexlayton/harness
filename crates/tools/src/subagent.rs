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

/// Which capability set one delegation asks for. The mode decides both the
/// child's tool set and the scheduler class: read-only children inspect and
/// report (and fan out concurrently), workspace children get the normal tool
/// set but serialize with other mutating work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubagentMode {
    /// Inspect-and-report only. The child registry exposes no mutating tools
    /// and no shell; adjacent read-only delegations run concurrently.
    ReadOnly,
    /// The normal built-in tool set including edit/write/bash. Because these
    /// children may mutate the shared workspace, each runs exclusively.
    Workspace,
}

impl SubagentMode {
    /// Strict parsing of the optional `mode` argument. Missing defaults to
    /// [`SubagentMode::ReadOnly`] (conservative compatibility: existing
    /// callers keep the parallel, non-mutating behavior); anything present
    /// must match exactly, so empty strings and typos fail closed at the call
    /// site.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None => Ok(Self::ReadOnly),
            Some("read_only") => Ok(Self::ReadOnly),
            Some("workspace") => Ok(Self::Workspace),
            Some(other) => Err(format!(
                "unknown subagent mode `{other}`: expected \"read_only\" or \"workspace\""
            )),
        }
    }

    /// Canonical wire spelling, also used in logs and titles.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::Workspace => "workspace",
        }
    }
}

/// Spawns and drives a nested agent loop for one delegated task. Implemented
/// by the agent crate; kept here as a trait so the tool crate stays
/// dependency-free.
#[async_trait]
pub trait SubagentRunner: Send + Sync {
    /// Run one subagent to completion with a fresh context seeded only by
    /// `prompt`. `mode` selects the child's capability set (and thereby its
    /// scheduler class). Returns the subagent's final report text on success,
    /// or a human-readable error message (the caller turns it into an errored
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
        mode: SubagentMode,
        cancel: CancellationToken,
    ) -> Result<String, String>;
}

/// Extract the string arguments plus the requested mode. Shared by
/// `execute`, `concurrency`, and `call_summary` so previews, scheduling,
/// and execution never diverge on what a call means.
///
/// Extracted arguments: description, prompt, and the parsed (or rejected)
/// mode.
pub(crate) type SubagentArgs<'a> = (
    Option<&'a str>,
    Option<&'a str>,
    Result<SubagentMode, String>,
);

pub(crate) fn parse_args(args: &Value) -> Result<SubagentArgs<'_>, String> {
    let object = args.as_object().ok_or("arguments must be an object")?;
    if let Some(unknown) = object
        .keys()
        .find(|key| !matches!(key.as_str(), "description" | "prompt" | "mode"))
    {
        return Err(format!("unknown subagent argument `{unknown}`"));
    }
    let description = object.get("description").and_then(Value::as_str);
    let prompt = object.get("prompt").and_then(Value::as_str);
    // A present-but-non-string mode is malformed (not merely absent): it
    // must fail closed rather than silently default.
    let mode = match object.get("mode") {
        None => Ok(SubagentMode::ReadOnly),
        Some(Value::String(raw)) => SubagentMode::parse(Some(raw)),
        Some(_) => Err("subagent `mode` must be a string".to_owned()),
    };
    Ok((description, prompt, mode))
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
/// is argument-sensitive: `read_only` (the default) fans out concurrently,
/// `workspace` serializes because it may mutate the shared workspace, and a
/// malformed mode fails closed as exclusive (execute then reports the
/// validation error).
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
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["read_only", "workspace"],
                            "description": "Capability set for this delegation; defaults to read_only. read_only subagents can inspect the repo and report but cannot modify files or run arbitrary commands — they run CONCURRENTLY, so batch independent ones in one response. workspace subagents have the normal tool set including edit/write/bash but run ONE AT A TIME (serialized), so reserve them for delegations that genuinely need to modify or execute."
                        }
                    },
                    "required": ["description", "prompt"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "delegate a self-contained task to a fresh subagent (own context window); returns its final report",
                vec![
                    "The subagent cannot see this conversation: write `prompt` fully \
                     self-contained, with explicit file paths, constraints, and the exact \
                     deliverable you expect back.",
                    "Batch multiple `read_only` `subagent` calls in one response when their tasks \
                     are independent (e.g. one per crate/module) — they run concurrently. \
                     Sequence calls when one depends on another's result.",
                    "Use `mode: \"workspace\"` only when a delegation genuinely needs to modify \
                     files or execute commands. Workspace delegations serialize with other \
                     mutating work, so keep their scopes explicit and non-overlapping.",
                    "Prefer subagents for breadth (surveys, audits, per-target repetition) \
                     over depth; do the focused edit yourself so you keep full context.",
                ],
            ),
        }
    }

    /// Argument-sensitive scheduling: read-only delegations fan out;
    /// workspace delegations (and malformed modes) serialize. A malformed
    /// mode classifies as Exclusive so an invalid call can never join a
    /// concurrent batch; `execute` turns it into an errored result.
    fn concurrency(&self, args: &Value) -> Concurrency {
        match args.get("mode") {
            None => Concurrency::Parallel,
            Some(Value::String(raw)) => match SubagentMode::parse(Some(raw)) {
                Ok(SubagentMode::ReadOnly) => Concurrency::Parallel,
                Ok(SubagentMode::Workspace) | Err(_) => Concurrency::Exclusive,
            },
            // Malformed (non-string) mode fails closed.
            Some(_) => Concurrency::Exclusive,
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let (description, prompt, mode) = match parse_args(&args) {
            Ok(parsed) => parsed,
            Err(message) => {
                return ToolOutput {
                    content: message,
                    is_error: true,
                    summary: SUBAGENT_TOOL_NAME.to_owned(),
                };
            }
        };
        // A malformed mode failed closed at classification time; here it
        // becomes an explicit validation error instead of silently running.
        let Ok(mode) = mode else {
            return ToolOutput {
                content: mode.unwrap_err(),
                is_error: true,
                summary: SUBAGENT_TOOL_NAME.to_owned(),
            };
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
        match self.runner.run(&description, prompt, mode, cancel).await {
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
        calls: Mutex<Vec<(String, String, SubagentMode)>>,
        reply: Mutex<Option<Result<String, String>>>,
    }

    #[async_trait]
    impl SubagentRunner for RecordingRunner {
        async fn run(
            &self,
            description: &str,
            prompt: &str,
            mode: SubagentMode,
            cancel: CancellationToken,
        ) -> Result<String, String> {
            let _ = cancel;
            self.calls
                .lock()
                .unwrap()
                .push((description.to_owned(), prompt.to_owned(), mode));
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
    fn spec_has_schema_prompt_and_mode_sensitive_class() {
        let spec = tool().spec();
        assert_eq!(spec.definition.name, "subagent");
        assert_eq!(
            spec.definition.parameters["required"],
            json!(["description", "prompt"])
        );
        // The schema advertises both modes and documents the default.
        assert_eq!(
            spec.definition.parameters["properties"]["mode"]["enum"],
            json!(["read_only", "workspace"])
        );
        assert!(
            spec.definition.parameters["properties"]["mode"]["description"]
                .as_str()
                .unwrap()
                .contains("defaults to read_only")
        );
        assert!(!spec.prompt.guidelines.is_empty());
        assert!(spec.prompt.snippet.is_some());
        // Missing mode defaults to read_only → parallel; workspace and any
        // malformed mode fail closed to exclusive.
        assert_eq!(tool().concurrency(&json!({})), Concurrency::Parallel);
        assert_eq!(
            tool().concurrency(&json!({"mode": "read_only"})),
            Concurrency::Parallel
        );
        assert_eq!(
            tool().concurrency(&json!({"mode": "workspace"})),
            Concurrency::Exclusive
        );
        assert_eq!(
            tool().concurrency(&json!({"mode": "sudo"})),
            Concurrency::Exclusive
        );
        assert_eq!(
            tool().concurrency(&json!({"mode": 3})),
            Concurrency::Exclusive
        );
    }

    #[test]
    fn mode_parses_strictly_and_defaults_to_read_only() {
        assert_eq!(SubagentMode::parse(None), Ok(SubagentMode::ReadOnly));
        assert!(SubagentMode::parse(Some("")).is_err());
        assert_eq!(
            SubagentMode::parse(Some("read_only")),
            Ok(SubagentMode::ReadOnly)
        );
        assert_eq!(
            SubagentMode::parse(Some("workspace")),
            Ok(SubagentMode::Workspace)
        );
        assert!(SubagentMode::parse(Some("Workspace")).is_err());
        assert!(SubagentMode::parse(Some("readonly")).is_err());
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

    #[tokio::test]
    async fn unknown_mode_errors_without_running_the_runner() {
        let runner = std::sync::Arc::new(RecordingRunner::default());
        let tool = SubagentTool::new(runner.clone());
        let output = tool
            .execute(
                json!({"description": "x", "prompt": "y", "mode": "sudo"}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("unknown subagent mode"));
        assert!(runner.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn mode_is_forwarded_to_the_runner() {
        let runner = std::sync::Arc::new(RecordingRunner::default());
        let tool = SubagentTool::new(runner.clone());
        tool.execute(
            json!({"description": "d", "prompt": "p", "mode": "workspace"}),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(runner.calls.lock().unwrap()[0].2, SubagentMode::Workspace);
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
