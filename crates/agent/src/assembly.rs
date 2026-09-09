//! Shared construction contract for the TUI, headless, and ACP frontends.
//!
//! Frontends decide where a workspace/session comes from; this builder makes
//! the resulting agent policy identical once those inputs have been resolved.

use crate::agent::{Agent, AgentEvent, InputMessage, ProviderFactory, SubagentLimits};
use crate::subagent::SubagentRunnerImpl;
use anyhow::{Context, Result};
use compact::CompactionPolicy;
use llm::{Provider, ReasoningPolicy};
use session::{Session, SessionStore};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tools::ToolRegistry;

/// Resolved subagent execution bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubagentPolicy {
    /// Maximum nested request/tool rounds for one delegation; zero disables it.
    pub max_turns: usize,
    /// Maximum concurrently running delegations per turn.
    pub max_concurrent: usize,
}

impl Default for SubagentPolicy {
    fn default() -> Self {
        Self {
            max_turns: 25,
            max_concurrent: 4,
        }
    }
}

/// Common agent assembly used by every frontend.
///
/// A configured subagent is registered together with its runner in one
/// operation, so a schema can never be advertised without executable support.
pub struct AgentBuilder {
    provider: Arc<dyn Provider>,
    model: String,
    reasoning: ReasoningPolicy,
    tools: ToolRegistry,
    cancel: CancellationToken,
    project_context: String,
    compaction: CompactionPolicy,
    subagents: SubagentPolicy,
    rtk: bool,
    session: Option<(SessionStore, Session)>,
    provider_factory: Option<ProviderFactory>,
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
            reasoning: ReasoningPolicy::Auto,
            tools,
            cancel,
            project_context: String::new(),
            compaction: CompactionPolicy::default(),
            subagents: SubagentPolicy::default(),
            rtk: false,
            session: None,
            provider_factory: None,
            mcp_servers: Vec::new(),
        }
    }

    /// Configure reasoning policy for parent and nested model requests.
    pub fn with_reasoning(mut self, reasoning: ReasoningPolicy) -> Self {
        self.reasoning = reasoning;
        self
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

    /// Attach host-owned provider construction for `/model` and `/models`.
    pub fn with_provider_factory(mut self, factory: ProviderFactory) -> Self {
        self.provider_factory = Some(factory);
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
        // Repair a loaded crash tail before starting MCP servers or exposing a
        // live agent. Failing here is preferable to reporting a successful
        // load that predictably quarantines on its first append.
        if let Some((store, session)) = self.session.as_mut() {
            store
                .repair_incomplete_tool_calls(session)
                .context("repair incomplete session tool calls")?;
        }
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
        if self.cancel.is_cancelled() {
            if let Some(mcp) = mcp {
                mcp.shutdown().await;
            }
            anyhow::bail!("agent assembly cancelled");
        }
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
            )
            .with_reasoning(self.reasoning);
            if let Some(index) = search_index {
                runner = runner.with_file_search_index(index);
            }
            let runner = Arc::new(runner);
            if let Err(error) = self.tools.register_subagent(runner.clone()) {
                if let Some(mcp) = mcp {
                    mcp.shutdown().await;
                }
                return Err(error).context("register subagent tool");
            }
            Some(runner)
        } else {
            None
        };

        let assembly_cancel = self.cancel.clone();
        let mut agent = Agent::new(self.provider, self.tools, self.model, self.cancel)
            .with_reasoning(self.reasoning)
            .with_project_context(self.project_context)
            .with_compaction(self.compaction)
            .with_subagent_limits(SubagentLimits {
                max_concurrent: self.subagents.max_concurrent,
            });
        if let Some(factory) = self.provider_factory {
            agent = agent.with_provider_factory(factory);
        }
        if let Some(runner) = runner {
            agent = agent.with_subagent_runner(runner);
        }
        if let Some((store, session)) = self.session {
            agent = agent.with_session(store, session);
        }
        if assembly_cancel.is_cancelled() {
            if let Some(mcp) = mcp {
                mcp.shutdown().await;
            }
            anyhow::bail!("agent assembly cancelled");
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
        input: tokio::sync::mpsc::UnboundedReceiver<InputMessage>,
        events: tokio::sync::mpsc::UnboundedSender<AgentEvent>,
    ) {
        self.agent.run(input, events).await;
        if let Some(mcp) = self.mcp {
            mcp.shutdown().await;
        }
    }

    /// Shut down MCP services when assembly completes but the frontend cannot
    /// take ownership of the resulting agent.
    pub async fn shutdown(self) {
        if let Some(mcp) = self.mcp {
            mcp.shutdown().await;
        }
    }
}
