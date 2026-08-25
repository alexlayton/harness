//! Shared construction contract for the TUI, headless, and ACP frontends.
//!
//! Frontends decide where a workspace/session comes from; this builder makes
//! the resulting agent policy identical once those inputs have been resolved.

use crate::agent::{Agent, SubagentLimits};
use crate::config::SubagentPolicy;
use crate::subagent::SubagentRunnerImpl;
use anyhow::{Context, Result};
use auth::CopilotAuth;
use compact::CompactionPolicy;
use llm::Provider;
use session::{Session, SessionStore};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tools::ToolRegistry;

/// Common agent assembly used by every frontend.
///
/// A configured subagent is registered together with its runner in one
/// operation, so a schema can never be advertised without executable support.
pub struct AgentBuilder {
    provider: Arc<dyn Provider>,
    model: String,
    tools: ToolRegistry,
    cancel: CancellationToken,
    project_context: String,
    compaction: CompactionPolicy,
    subagents: SubagentPolicy,
    rtk: bool,
    session: Option<(SessionStore, Session)>,
    copilot_auth: Option<Arc<CopilotAuth>>,
    mcp_servers: Vec<mcp::McpServerConfig>,
}

impl AgentBuilder {
    /// Start assembling an agent from frontend-resolved provider and tools.
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        tools: ToolRegistry,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            provider,
            model: model.into(),
            tools,
            cancel,
            project_context: String::new(),
            compaction: CompactionPolicy::default(),
            subagents: SubagentPolicy::default(),
            rtk: false,
            session: None,
            copilot_auth: None,
            mcp_servers: Vec::new(),
        }
    }

    /// Attach the rendered workspace instructions.
    pub fn with_project_context(mut self, context: impl Into<String>) -> Self {
        self.project_context = context.into();
        self
    }

    /// Attach resolved compaction policy.
    pub fn with_compaction(mut self, policy: CompactionPolicy) -> Self {
        self.compaction = policy;
        self
    }

    /// Configure subagent enablement and concurrency policy.
    pub fn with_subagents(mut self, policy: SubagentPolicy, rtk: bool) -> Self {
        self.subagents = policy;
        self.rtk = rtk;
        self
    }

    /// Attach an optional durable store/session pair.
    pub fn with_session(mut self, store: SessionStore, session: Session) -> Self {
        self.session = Some((store, session));
        self
    }

    /// Attach the shared Copilot authentication handle.
    pub fn with_copilot_auth(mut self, auth: Arc<CopilotAuth>) -> Self {
        self.copilot_auth = Some(auth);
        self
    }

    /// Connect external MCP servers before registering subagents. Subagents
    /// deliberately retain their built-in-only registries.
    pub fn with_mcp_servers(mut self, servers: Vec<mcp::McpServerConfig>) -> Self {
        self.mcp_servers = servers;
        self
    }

    /// Connect optional MCP servers, register optional subagents, and produce
    /// a runtime that keeps external server processes alive for the agent.
    pub async fn build(mut self) -> Result<AssembledAgent> {
        let mcp = if self.mcp_servers.is_empty() {
            None
        } else {
            let runtime = mcp::McpRuntime::connect(
                &self.mcp_servers,
                self.tools.workspace_root(),
                self.cancel.clone(),
            )
            .await
            .context("connect MCP servers")?;
            if let Err(error) = runtime.register_into(&mut self.tools) {
                runtime.shutdown().await;
                return Err(error).context("register MCP tools");
            }
            Some(runtime)
        };
        let parent_session = self.session.as_ref().map(|(_, session)| session.id());
        let search_index = self.tools.file_search_index().cloned();
        let runner = if self.subagents.max_turns > 0 {
            let mut runner = SubagentRunnerImpl::new(
                self.provider.clone(),
                self.model.clone(),
                self.tools.workspace_root().to_path_buf(),
                self.rtk,
                self.project_context.clone(),
                self.subagents,
                self.session.as_ref().map(|(store, _)| store.clone()),
                parent_session,
            );
            if let Some(index) = search_index {
                runner = runner.with_file_search_index(index);
            }
            let runner = Arc::new(runner);
            self.tools
                .register_subagent(runner.clone())
                .context("register subagent tool")?;
            Some(runner)
        } else {
            None
        };

        let mut agent = Agent::new(self.provider, self.tools, self.model, self.cancel)
            .with_project_context(self.project_context)
            .with_compaction(self.compaction)
            .with_subagent_limits(SubagentLimits {
                max_concurrent: self.subagents.max_concurrent,
            });
        if let Some(auth) = self.copilot_auth {
            agent = agent.with_copilot_auth(auth);
        }
        if let Some(runner) = runner {
            agent = agent.with_subagent_runner(runner);
        }
        if let Some((store, session)) = self.session {
            agent = agent.with_session(store, session);
        }
        Ok(AssembledAgent { agent, mcp })
    }
}

/// An agent plus the external MCP runtime whose handles its tools use.
/// Calling [`Self::run`] guarantees an orderly server shutdown after the
/// frontend closes the agent input channel.
pub struct AssembledAgent {
    agent: Agent,
    mcp: Option<mcp::McpRuntime>,
}

impl AssembledAgent {
    /// Run the contained agent, then close every MCP server and reap children.
    pub async fn run(
        self,
        input: tokio::sync::mpsc::UnboundedReceiver<tui::InputMessage>,
        events: tokio::sync::mpsc::UnboundedSender<crate::agent::AgentEvent>,
    ) {
        self.agent.run(input, events).await;
        if let Some(mcp) = self.mcp {
            mcp.shutdown().await;
        }
    }
}
