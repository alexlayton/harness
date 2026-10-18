//! One-shot agent execution without terminal or process-global frontend setup.

use crate::config::{Cli, Config, ProviderArg, build_provider_with_auths, provider_factory};
use crate::context::project_context_for;
use agent::assembly::AgentBuilder;
use agent::{AgentEvent, InputMessage};
use anyhow::{Context, Result, anyhow, ensure};
use llm::ReasoningPolicy;
use serde::Deserialize;
use session::{SessionCreateOptions, SessionStore};
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tools::{ToolConfig, default_registry};

/// Per-call overrides. Unset settings use the normal Harness configuration.
///
/// Credentials remain in the environment or the normal OAuth store. A call
/// never writes its provider/model overrides back to the configuration file.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunOptions {
    /// Provider name, including the CLI's supported aliases.
    pub provider: Option<String>,
    /// Model identifier for this call.
    pub model: Option<String>,
    /// Portable reasoning preference for this call.
    pub reasoning_effort: Option<ReasoningPolicy>,
    /// Workspace directory; defaults to the caller's current directory.
    pub workspace: Option<PathBuf>,
    /// Disable durable session creation and automatic session compaction.
    pub no_session: bool,
    /// Resume a session by ID, unique prefix, `latest`, or path.
    pub resume: Option<String>,
    /// Disable instruction-file injection for this call.
    pub no_context_files: bool,
}

impl RunOptions {
    /// Check overrides before opening sessions or connecting providers/tools.
    pub fn validate(&self) -> Result<()> {
        if let Some(provider) = &self.provider {
            ensure!(
                ProviderArg::from_name(provider).is_some(),
                "unknown provider `{provider}`"
            );
        }
        if let Some(model) = &self.model {
            ensure!(!model.trim().is_empty(), "model must not be blank");
        }
        if let Some(resume) = &self.resume {
            ensure!(!resume.trim().is_empty(), "resume must not be blank");
            ensure!(!self.no_session, "resume cannot be used with no_session");
        }
        Ok(())
    }
}

/// Run one prompt and return only its final answer, without a trailing newline.
///
/// Configuration, authentication, tools, MCP, subagents, and persistence follow
/// the CLI defaults. This function installs no signal handlers or global log
/// subscriber, and never changes the current directory or environment.
///
/// Cancel `cancel` to stop the call, then keep awaiting this future so session
/// writes and MCP shutdown can finish. Dropping the future requests cancellation
/// but cannot wait for cleanup. A successful tool-only turn returns an empty
/// string; cancellation and unrecovered errors return an error, not partial text.
pub async fn run(prompt: String, options: RunOptions, cancel: CancellationToken) -> Result<String> {
    ensure!(!prompt.trim().is_empty(), "prompt must not be blank");
    options.validate()?;
    let _cancel_on_drop = cancel.clone().drop_guard();
    let cli = Cli {
        provider: options.provider.as_deref().and_then(ProviderArg::from_name),
        model: options.model.clone(),
        reasoning_effort: options.reasoning_effort,
        ..Cli::default()
    };
    let config = Config::resolve(&cli)?;
    let workspace = options
        .workspace
        .clone()
        .map(Ok)
        .unwrap_or_else(std::env::current_dir)
        .context("resolve workspace directory")?;
    let workspace = std::fs::canonicalize(workspace).context("resolve workspace directory")?;
    ensure!(workspace.is_dir(), "workspace must be a directory");
    ensure!(!cancel.is_cancelled(), "Harness run cancelled");
    let provider = build_provider_with_auths(
        &config.provider.to_string(),
        config.copilot_auth.clone(),
        config.codex_auth.clone(),
    )?;
    let context = project_context_for(&workspace, options.no_context_files);
    let tools = default_registry(ToolConfig::new(&workspace, config.rtk))?;
    let mut builder = AgentBuilder::new(
        provider.clone(),
        config.model.clone(),
        tools,
        cancel.clone(),
    )
    .with_reasoning(config.reasoning)
    .with_secret_masker(config.secret_masker.clone())
    .with_compaction(config.compaction.clone())
    .with_subagents(config.subagents, config.rtk)
    .with_mcp_servers(config.mcp_servers.clone())
    .with_project_context(context)
    .with_provider_factory(provider_factory(
        config.copilot_auth.clone(),
        config.codex_auth.clone(),
    ));
    if !options.no_session {
        let store = SessionStore::default_for_workspace(&workspace)?;
        let session = match options.resume {
            Some(selector) => store
                .load(&selector)
                .with_context(|| format!("load session `{selector}`"))?,
            None => store.create(SessionCreateOptions {
                provider: Some(provider.name().to_owned()),
                model: Some(config.model.clone()),
                ..SessionCreateOptions::default()
            })?,
        };
        builder = builder.with_session(store, session);
    }
    let assembled = builder.build().await?;
    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(assembled.run(input_rx, event_tx));
    let sent = input_tx.send(InputMessage::Message(prompt));
    // A closed channel drains this one turn, rather than interrupting its tools.
    drop(input_tx);
    let mut answer = FinalAnswer::default();
    while let Some(event) = event_rx.recv().await {
        answer.observe(event);
    }
    let joined = task.await;
    sent.context("send prompt to agent")?;
    joined.context("Harness agent task failed")?;
    ensure!(!cancel.is_cancelled(), "Harness run cancelled");
    answer.finish()
}

// Match headless output semantics: prose from a tool-bearing model round is
// intermediate, not part of the final answer. Errors can be followed by a
// genuine recovery, but usage/lifecycle events must not erase the failure.
#[derive(Default)]
struct FinalAnswer {
    pending: String,
    text: String,
    active_tools: usize,
    error: Option<String>,
    finished: bool,
}

impl FinalAnswer {
    fn observe(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta(text) => {
                self.pending.push_str(&text);
                self.error = None;
            }
            AgentEvent::ToolCallStarted { .. } => {
                self.pending.clear();
                self.active_tools = self.active_tools.saturating_add(1);
            }
            AgentEvent::ToolCallFinished { .. } => {
                self.active_tools = self.active_tools.saturating_sub(1);
                if self.active_tools == 0 {
                    self.pending.clear();
                }
            }
            AgentEvent::Error(message) => {
                self.pending.clear();
                self.error = Some(message);
            }
            AgentEvent::TurnFinished => {
                if self.active_tools == 0 {
                    self.text = std::mem::take(&mut self.pending);
                }
                self.finished = true;
            }
            _ => {}
        }
    }

    fn finish(self) -> Result<String> {
        if let Some(error) = self.error {
            return Err(anyhow!(error));
        }
        ensure!(self.finished, "Harness stopped before the turn completed");
        Ok(self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_answer_excludes_tool_round_text_and_preserves_failure() {
        let mut answer = FinalAnswer::default();
        answer.observe(AgentEvent::TextDelta("I'll inspect it.".into()));
        answer.observe(AgentEvent::ToolCallStarted {
            call_id: "read-1".into(),
            name: "read".into(),
            summary: "read file".into(),
        });
        answer.observe(AgentEvent::TextDelta("More intermediate text.".into()));
        answer.observe(AgentEvent::ToolCallFinished {
            call_id: "read-1".into(),
            name: "read".into(),
            summary: "read file".into(),
            ok: true,
            duration_ms: 1,
            output: "file contents".into(),
            error: None,
        });
        answer.observe(AgentEvent::TextDelta("Final answer.".into()));
        answer.observe(AgentEvent::TurnFinished);
        assert_eq!(answer.finish().unwrap(), "Final answer.");

        let mut failed = FinalAnswer::default();
        failed.observe(AgentEvent::TextDelta("Partial answer".into()));
        failed.observe(AgentEvent::Error("provider failed".into()));
        failed.observe(AgentEvent::TurnFinished);
        assert_eq!(failed.finish().unwrap_err().to_string(), "provider failed");
        assert!(FinalAnswer::default().finish().is_err());
    }

    #[test]
    fn overrides_default_to_config_and_reject_unknown_or_conflicting_options() {
        let defaults: RunOptions = serde_json::from_str("{}").unwrap();
        assert!(defaults.provider.is_none());
        assert!(defaults.model.is_none());
        assert!(defaults.reasoning_effort.is_none());
        assert!(!defaults.no_session);
        assert!(serde_json::from_str::<RunOptions>(r#"{"modle":"typo"}"#).is_err());
        let conflicting = RunOptions {
            resume: Some("latest".into()),
            no_session: true,
            ..RunOptions::default()
        };
        assert!(conflicting.validate().is_err());
    }
}
