//! ACP frontend: expose the harness agent to editors over stdio
//! ([Agent Client Protocol](https://agentclientprotocol.com/)).
//!
//! This is the third frontend beside the TUI and [`headless`](crate::headless):
//! it drives the exact same `Agent` stack and translates between ACP wire
//! messages and harness events. Tools execute immediately with no approval
//! step — the same semantics as every other frontend — so we simply never
//! send `session/request_permission`.
//!
//! One ACP session is one `(ToolRegistry, SessionStore, Agent)` triple rooted
//! at the request's `cwd`. Our own session id doubles as the opaque ACP
//! `SessionId`, so `session/load` maps 1:1 onto `SessionStore::load`.
//!
//! Deliberately unsupported: permission gating, auth over ACP
//! (`authenticate` answers with instructions to sign in interactively),
//! transcript replay on `session/load` (history is intact on disk and in the
//! agent context; the editor shows an empty transcript until the next turn),
//! HTTP/SSE/MCP-over-ACP transports, mid-session model switching. ACP-provided
//! stdio MCP servers are supported per session. Unhandled requests fall through
//! to the SDK default of method-not-found.
//!
//! Stdout ownership is inverted here: stdout carries JSON-RPC only. Tracing
//! stays behind `HARNESS_LOG` (file-only), and this module never writes to
//! stderr either — editors surface child stderr as agent noise.

use crate::config::{Config, ProviderArg, provider_factory};
use crate::context::project_context_for;
use agent::assembly::AgentBuilder;
use agent::{AgentEvent, InputMessage};
use agent_client_protocol::schema::v1::{
    AgentCapabilities, AuthenticateRequest, CancelNotification, ContentBlock, ContentChunk,
    DeleteSessionRequest, DeleteSessionResponse, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, McpServer,
    NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse,
    SessionId, SessionInfo, SessionMode, SessionModeId, SessionModeState, SessionNotification,
    SessionUpdate, StopReason, ToolCall as AcToolCall, ToolCallContent, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind, UsageUpdate,
};
use agent_client_protocol::schema::{ProtocolVersion, v1};
use agent_client_protocol::{
    Agent as AgentRole, ConnectTo, ConnectionTo, Error as AcError, Responder, Result as AcResult,
    Stdio, on_receive_notification, on_receive_request,
};
use anyhow::{Context as _, Result};
use auth::CopilotAuth;
use llm::Provider;
use session::{SessionCreateOptions, SessionStore};
use std::collections::{HashMap, hash_map::Entry};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tools::{ToolConfig, ToolRegistry, default_registry};

/// The single mode advertised for every session: harness has no permission
/// levels to switch between (tools always run), so `set_mode` stays
/// unsupported while the mode list still gives editors something to display.
const MODE_ID: &str = "work";

/// Everything needed to drive one live session.
struct SessionHandle {
    /// Commands for the agent's run loop.
    input_tx: mpsc::UnboundedSender<InputMessage>,
    /// Cancellation owned by this ACP session, rather than shared across all
    /// editor sessions.
    cancel: CancellationToken,
    /// The agent task must stop before its session file is removed.
    agent_task: JoinHandle<()>,
    /// The forwarder must also stop before a deleted session can emit late
    /// protocol updates.
    forwarder_task: JoinHandle<()>,
}

/// A prompt request parked until its turn ends. Stored in
/// [`AcpState::in_flight`] because the per-session forwarder — not the request
/// handler — observes `TurnFinished`, and `session/cancel` needs to mark it.
struct InFlight {
    responder: Responder<PromptResponse>,
    /// Set by `session/cancel` before the interrupt lands; distinguishes a
    /// cancelled stop from a natural end when the turn finishes.
    cancelled: bool,
    /// Terminal agent failures are retained until `TurnFinished`, because a
    /// recoverable provider error may be followed by a successful retry.
    error: Option<String>,
}

/// The slice of [`AcpState`] the forwarder tasks share. Keeping it separate
/// from the full state makes the forwarder's dependencies explicit and lets
/// each task hold one small `Arc` instead of pinning provider/config alive.
struct PromptTracker {
    in_flight: Mutex<HashMap<String, InFlight>>,
}

impl PromptTracker {
    /// Resolve the pending prompt for a session, if any, and answer the
    /// editor. Called by the forwarder on `TurnFinished` (with a stop reason)
    /// and again on agent-task death (with `None`) so a crashed turn fails
    /// the JSON-RPC request instead of hanging it open forever.
    fn resolve(&self, session_id: &str, stop_reason: Option<StopReason>) {
        let Some(entry) = self.in_flight.lock().unwrap().remove(session_id) else {
            return;
        };
        let response = if entry.cancelled {
            Ok(PromptResponse::new(StopReason::Cancelled))
        } else {
            match (entry.error, stop_reason) {
                (Some(error), _) => Err(AcError::internal_error().data(error)),
                (None, Some(reason)) => Ok(PromptResponse::new(reason)),
                (None, None) => Err(AcError::internal_error()
                    .data("agent task ended before the turn completed".to_owned())),
            }
        };
        let _ = entry.responder.respond_with_result(response);
    }

    /// Record an agent error for the current prompt. It is resolved at the
    /// turn boundary so recoverable provider errors can still succeed.
    fn mark_error(&self, session_id: &str, message: &str) {
        if let Some(entry) = self.in_flight.lock().unwrap().get_mut(session_id) {
            entry.error = Some(format!("agent turn failed: {message}"));
        }
    }

    /// Clear a previously observed error only when a new assistant text delta
    /// proves the provider recovered; unrelated metadata must not do this.
    fn clear_error(&self, session_id: &str) {
        if let Some(entry) = self.in_flight.lock().unwrap().get_mut(session_id) {
            entry.error = None;
        }
    }

    /// Mark a pending prompt as cancelled. Returns false when nothing is in
    /// flight, in which case `session/cancel` has nothing to interrupt.
    fn mark_cancelled(&self, session_id: &str) -> bool {
        match self.in_flight.lock().unwrap().get_mut(session_id) {
            Some(entry) => {
                entry.cancelled = true;
                true
            }
            None => false,
        }
    }

    fn is_cancelled(&self, session_id: &str) -> bool {
        self.in_flight
            .lock()
            .unwrap()
            .get(session_id)
            .is_some_and(|entry| entry.cancelled)
    }
}

/// Shared adapter state for the lifetime of one stdio connection.
struct AcpState {
    /// Directory holding the session store's workspace groups. Resolved once
    /// at startup (from `HARNESS_SESSION_DIR`/`HARNESS_STATE_DIR` env or the
    /// default under home), then threaded through every handler so tests can
    /// pin an isolated temp dir without touching the process-global env.
    session_root: std::path::PathBuf,
    /// Resolved once from CLI/config at startup; reused for every session.
    provider: Arc<dyn Provider>,
    config: Config,
    copilot_auth: Option<Arc<CopilotAuth>>,
    no_context_files: bool,
    sessions: Mutex<HashMap<String, SessionHandle>>,
    prompts: Arc<PromptTracker>,
}

impl AcpState {
    fn cancel_session(&self, session_id: &str) {
        // Keep the same sessions → prompts lock order as prompt submission and
        // deletion, preventing a cancel/delete race from deadlocking.
        let sessions = self.sessions.lock().unwrap();
        let Some(handle) = sessions.get(session_id) else {
            return;
        };
        if !self.prompts.mark_cancelled(session_id) {
            // No prompt in flight: nothing meaningful to cancel.
            return;
        }
        if handle.input_tx.send(InputMessage::Interrupt).is_err() {
            tracing::warn!(session = %session_id, "cancel arrived after the agent stopped");
        }
    }
}

/// Entry point for `harness acp`: serve one ACP agent over stdio until the
/// client disconnects.
pub async fn run(
    provider: Arc<dyn Provider>,
    config: Config,
    copilot_auth: Option<Arc<CopilotAuth>>,
    no_context_files: bool,
) -> Result<ExitCode> {
    // Copilot's first login is a device flow needing a browser and a human;
    // over ACP the only honest answer is to point at the interactive CLI.
    if config.provider == ProviderArg::GithubCopilot
        && let Some(auth) = &copilot_auth
        && matches!(auth.credential(), Ok(None))
    {
        anyhow::bail!(
            "Copilot is not authenticated yet: run `harness login github-copilot` \
             (the credential persists in ~/.config/harness/auth.json)"
        );
    }
    serve(
        provider,
        config,
        copilot_auth,
        no_context_files,
        session::default_session_dir(),
        Stdio::new(),
    )
    .await
}

/// [`run`] without the Copilot pre-flight, so tests can drive the connection
/// with a mock provider and no real credentials. The `transport` is hoisted
/// out so tests can connect over an in-memory duplex pair instead of the real
/// process stdio.
async fn serve<C>(
    provider: Arc<dyn Provider>,
    config: Config,
    copilot_auth: Option<Arc<CopilotAuth>>,
    no_context_files: bool,
    session_root: std::path::PathBuf,
    transport: C,
) -> Result<ExitCode>
where
    C: ConnectTo<agent_client_protocol::Agent> + 'static,
{
    let state = Arc::new(AcpState {
        session_root,
        provider,
        config,
        copilot_auth,
        no_context_files,
        sessions: Mutex::new(HashMap::new()),
        prompts: Arc::new(PromptTracker {
            in_flight: Mutex::new(HashMap::new()),
        }),
    });

    // Handlers run inside the SDK dispatch loop and block message processing,
    // so none of them await a prompt turn: `session/prompt` parks its
    // responder in `prompts` and the per-session forwarder resolves it.
    //
    // Handlers are async closures (the builder takes `AsyncFnMut`) that
    // respond inline; long work lives in the spawned agent/forwarder tasks.
    let connection_result = AgentRole
        .builder()
        .name("harness")
        .on_receive_request(
            async |request: InitializeRequest,
                   responder: Responder<InitializeResponse>,
                   _cx: ConnectionTo<agent_client_protocol::Client>| {
                let _ = responder.respond(initialize_response(&request));
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async |_request: AuthenticateRequest,
                   responder: Responder<v1::AuthenticateResponse>,
                   _cx: ConnectionTo<agent_client_protocol::Client>| {
                let _ = responder.respond_with_error(
                    AcError::auth_required().data(
                        "harness does not support auth over ACP; run `harness login github-copilot` \
                     or `harness login openai-codex`, or export the provider API key"
                            .to_owned(),
                    ),
                );
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |request: NewSessionRequest,
                      responder: Responder<NewSessionResponse>,
                      cx: ConnectionTo<agent_client_protocol::Client>| {
                    new_session(request, responder, cx, state.clone())
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |request: LoadSessionRequest,
                      responder: Responder<LoadSessionResponse>,
                      cx: ConnectionTo<agent_client_protocol::Client>| {
                    load_session(request, responder, cx, state.clone())
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async |request: ListSessionsRequest,
                   responder: Responder<ListSessionsResponse>,
                   _cx: ConnectionTo<agent_client_protocol::Client>| {
                let _ = responder.respond(list_sessions(&request, &state));
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |request: DeleteSessionRequest,
                      responder: Responder<DeleteSessionResponse>,
                      _cx: ConnectionTo<agent_client_protocol::Client>| {
                    delete_session(request, responder, state.clone())
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                move |request: PromptRequest,
                      responder: Responder<PromptResponse>,
                      _cx: ConnectionTo<agent_client_protocol::Client>| {
                    prompt(request, responder, state.clone())
                }
            },
            on_receive_request!(),
        )
        .on_receive_notification(
            {
                let state = state.clone();
                async move |notification: CancelNotification,
                            _cx: ConnectionTo<agent_client_protocol::Client>| {
                    state.cancel_session(notification.session_id.0.as_ref());
                    Ok(())
                }
            },
            on_receive_notification!(),
        )
        .connect_to(transport)
        .await
        .context("run ACP connection");

    // Disconnect is a lifecycle boundary too: dropping JoinHandles would
    // detach agents and leave providers/MCP servers able to append after the
    // ACP transport is gone. Cancel and await every owned session before the
    // frontend returns.
    let handles = {
        let mut sessions = state.sessions.lock().unwrap();
        std::mem::take(&mut *sessions)
            .into_values()
            .collect::<Vec<_>>()
    };
    for handle in handles {
        shutdown_session(handle).await;
    }
    connection_result?;
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------
// Pure translation: ACP wire types <-> harness types. Free and
// side-effect-free so they unit-test without a connection.
// ---------------------------------------------------------------------

/// Reply to `initialize`: always answer with v1 (the only version this
/// adapter implements — echoing an older client version back would claim
/// support we don't have), advertise exactly what this frontend supports,
/// and offer no auth methods (see the `authenticate` handler).
fn initialize_response(_request: &InitializeRequest) -> InitializeResponse {
    InitializeResponse::new(ProtocolVersion::V1)
        .agent_info(v1::Implementation::new(
            "harness",
            env!("CARGO_PKG_VERSION"),
        ))
        .agent_capabilities(
            AgentCapabilities::default()
                .load_session(true)
                .prompt_capabilities(PromptCapabilities::default().embedded_context(true))
                .session_capabilities(
                    v1::SessionCapabilities::default()
                        .list(v1::SessionListCapabilities::default())
                        .delete(v1::SessionDeleteCapabilities::default()),
                ),
        )
}

fn static_mode_state() -> SessionModeState {
    SessionModeState::new(
        SessionModeId::new(MODE_ID),
        vec![SessionMode::new(MODE_ID, "work")],
    )
}

/// Flatten prompt content blocks into the single text string the agent loop
/// consumes. `Text` passes through verbatim; embedded resources are inlined
/// as fenced context. Resource links are noted rather than fetched — the
/// model can `read` the referenced path itself if it needs the contents.
fn flatten_prompt(blocks: &[ContentBlock]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(text) => parts.push(text.text.clone()),
            ContentBlock::ResourceLink(link) => {
                parts.push(format!("[attached file: {}]({})", link.name, link.uri));
            }
            ContentBlock::Resource(resource) => match &resource.resource {
                v1::EmbeddedResourceResource::TextResourceContents(text) => {
                    parts.push(format!("```\n{}\n```", text.text.trim_end()));
                }
                v1::EmbeddedResourceResource::BlobResourceContents(blob) => {
                    parts.push(format!(
                        "[embedded binary resource: {} ({} bytes)]",
                        blob.uri,
                        blob.blob.len()
                    ));
                }
                _ => {}
            },
            _ => {}
        }
    }
    parts.join("\n")
}

/// Map a harness tool name to the ACP tool category editors use for icons.
fn tool_kind(name: &str) -> ToolKind {
    match name {
        "read" => ToolKind::Read,
        "edit" | "write" => ToolKind::Edit,
        "bash" => ToolKind::Execute,
        "find" | "grep" => ToolKind::Search,
        _ => ToolKind::Other,
    }
}

/// Tool-call id correlation. `AgentEvent` carries the harness call id, so
/// starts and finishes are correlated by exact key — never by tool name or
/// FIFO position, which would be ambiguous for concurrent `subagent` calls
/// that all share one name.
#[derive(Default)]
struct ToolCallIds {
    in_flight: HashMap<String, ToolCallId>,
}

impl ToolCallIds {
    fn start(&mut self, call_id: &str, name: &str, summary: &str) -> AcToolCall {
        let id = ToolCallId::new(uuid::Uuid::new_v4().to_string());
        self.in_flight.insert(call_id.to_owned(), id.clone());
        AcToolCall::new(id, summary).kind(tool_kind(name))
    }

    fn finish(&mut self, call_id: &str) -> Option<ToolCallId> {
        self.in_flight.remove(call_id)
    }
}

/// Translate one agent event into at most one ACP session update. Events
/// without an ACP counterpart (auth UX, retries, compaction notices, …) map
/// to `None`; they stay visible in `HARNESS_LOG`. Context occupancy comes from
/// the agent's dedicated `ContextUsageUpdated` event, while cumulative billing
/// usage is intentionally not used as a context-size proxy.
fn translate_event(
    event: &AgentEvent,
    ids: &mut ToolCallIds,
    _context_window: u64,
) -> Option<SessionUpdate> {
    match event {
        AgentEvent::TextDelta(delta) => Some(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            ContentBlock::from(delta.as_str()),
        ))),
        AgentEvent::ReasoningDelta(delta) => Some(SessionUpdate::AgentThoughtChunk(
            ContentChunk::new(ContentBlock::from(delta.as_str())),
        )),
        AgentEvent::ToolCallStarted {
            call_id,
            name,
            summary,
        } => Some(SessionUpdate::ToolCall(ids.start(call_id, name, summary))),
        AgentEvent::ToolCallFinished {
            call_id,
            name,
            summary,
            ok,
            duration_ms: _,
            output,
            error,
        } => {
            // Correlate by the exact harness call id. A miss is an internal
            // invariant violation (finish without start); log it loudly but
            // still mint a fresh id so the editor gets *some* terminal
            // update instead of a dangling card.
            let tool_call_id = match ids.finish(call_id) {
                Some(id) => id,
                None => {
                    tracing::warn!(call_id = %call_id, name = %name, "tool finish without matching start (invariant violation)");
                    ToolCallId::new(uuid::Uuid::new_v4().to_string())
                }
            };
            // The error text is the informative payload on failure; otherwise
            // attach the full output so editors can expand it.
            let text = error.clone().unwrap_or_else(|| output.clone());
            let fields = ToolCallUpdateFields::new()
                .title(summary.clone())
                .status(if *ok {
                    ToolCallStatus::Completed
                } else {
                    ToolCallStatus::Failed
                })
                .content(vec![ToolCallContent::from(text)]);
            Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                tool_call_id,
                fields,
            )))
        }
        // UsageUpdated is cumulative billing telemetry. It must not be
        // interpreted as current context occupancy and is therefore consumed
        // only by the forwarder's cost bookkeeping.
        AgentEvent::UsageUpdated { .. } => None,
        AgentEvent::ContextUsageUpdated {
            used_tokens,
            max_tokens,
        } => Some(SessionUpdate::UsageUpdate(UsageUpdate::new(
            *used_tokens,
            *max_tokens,
        ))),
        AgentEvent::Error(message) => Some(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            ContentBlock::from(format!("error: {message}")),
        ))),
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Request handler bodies. Each responds exactly once and returns quickly;
// anything long runs inside the spawned agent/forwarder tasks.
// ---------------------------------------------------------------------

/// Map any error into a JSON-RPC error response.
fn respond_anyhow<T: agent_client_protocol::JsonRpcResponse>(
    responder: Responder<T>,
    error: anyhow::Error,
) -> AcResult<()> {
    let _ = responder.respond_with_error(AcError::internal_error().data(error.to_string()));
    Ok(())
}

/// Reject malformed or unsupported session-provided MCP configuration without
/// pretending the agent itself failed internally.
fn respond_invalid_params<T: agent_client_protocol::JsonRpcResponse>(
    responder: Responder<T>,
    error: anyhow::Error,
) -> AcResult<()> {
    let _ = responder.respond_with_error(AcError::invalid_params().data(error.to_string()));
    Ok(())
}

async fn new_session(
    request: NewSessionRequest,
    responder: Responder<NewSessionResponse>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    tokio::spawn(async move {
        if let Err(error) = new_session_inner(request, responder, connection, state).await {
            tracing::error!(error = %error, "ACP new-session task failed");
        }
    });
    Ok(())
}

async fn new_session_inner(
    request: NewSessionRequest,
    responder: Responder<NewSessionResponse>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    let mcp_servers = match acp_mcp_servers(&request.mcp_servers) {
        Ok(servers) => servers,
        Err(error) => return respond_invalid_params(responder, error),
    };
    let (store, tools) = match build_session_stack(&request.cwd, &state.session_root).await {
        Ok(stack) => stack,
        Err(error) => return respond_anyhow(responder, error),
    };
    let session = match store.create(SessionCreateOptions {
        provider: Some(state.provider.name().to_owned()),
        model: Some(state.config.model.clone()),
        ..SessionCreateOptions::default()
    }) {
        Ok(session) => session,
        Err(error) => return respond_anyhow(responder, error.into()),
    };
    let id = session.id().to_string();
    match spawn_agent(
        &state,
        store,
        tools,
        session,
        connection,
        id.clone(),
        mcp_servers,
    )
    .await
    {
        Ok(()) => {
            tracing::info!(session = %id, cwd = %request.cwd.display(), "ACP session created");
            let _ = responder
                .respond(NewSessionResponse::new(SessionId::from(id)).modes(static_mode_state()));
            Ok(())
        }
        Err(error) => respond_anyhow(responder, error),
    }
}

async fn load_session(
    request: LoadSessionRequest,
    responder: Responder<LoadSessionResponse>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    tokio::spawn(async move {
        if let Err(error) = load_session_inner(request, responder, connection, state).await {
            tracing::error!(error = %error, "ACP load-session task failed");
        }
    });
    Ok(())
}

async fn load_session_inner(
    request: LoadSessionRequest,
    responder: Responder<LoadSessionResponse>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    let raw_id = request.session_id.0.to_string();
    if raw_id.trim() != raw_id {
        return respond_invalid_params(
            responder,
            anyhow::anyhow!("ACP session IDs must be an exact UUID"),
        );
    }
    let parsed_id = match session::SessionId::parse(&raw_id) {
        Ok(id) => id,
        Err(error) => return respond_invalid_params(responder, anyhow::Error::new(error)),
    };
    let id = parsed_id.to_string();
    if state.sessions.lock().unwrap().contains_key(&id) {
        return respond_invalid_params(
            responder,
            anyhow::anyhow!("session `{id}` is already loaded"),
        );
    }
    let mcp_servers = match acp_mcp_servers(&request.mcp_servers) {
        Ok(servers) => servers,
        Err(error) => return respond_invalid_params(responder, error),
    };
    let (store, tools) = match build_session_stack(&request.cwd, &state.session_root).await {
        Ok(stack) => stack,
        Err(error) => return respond_anyhow(responder, error),
    };
    let session = match store.open(&parsed_id) {
        Ok(session) => session,
        Err(error) => {
            return respond_anyhow(
                responder,
                anyhow::Error::new(error).context(format!("load session `{id}`")),
            );
        }
    };
    match spawn_agent(
        &state,
        store,
        tools,
        session,
        connection,
        id.clone(),
        mcp_servers,
    )
    .await
    {
        Ok(()) => {
            tracing::info!(session = %id, cwd = %request.cwd.display(), "ACP session loaded");
            // Documented limitation: no transcript replay notifications. The
            // editor renders an empty transcript until the next turn; the
            // full history is intact on disk and in the agent's context.
            let _ = responder.respond(LoadSessionResponse::default().modes(static_mode_state()));
            Ok(())
        }
        Err(error) => respond_anyhow(responder, error),
    }
}

fn list_sessions(request: &ListSessionsRequest, state: &AcpState) -> ListSessionsResponse {
    // Sessions are grouped per workspace on disk; list for the requested cwd,
    // falling back to the process cwd when the client omits it. Listing
    // failures degrade to an empty page rather than failing the request.
    let cwd = request.cwd.clone().unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
    });
    let entries = SessionStore::new(&state.session_root, &cwd)
        .and_then(|store| store.list())
        .unwrap_or_default();
    ListSessionsResponse::new(
        entries
            .into_iter()
            .map(|entry| {
                SessionInfo::new(entry.id.to_string(), entry.workspace_root)
                    .title(entry.title)
                    .updated_at(entry.updated_at)
            })
            .collect(),
    )
}

const SESSION_TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

async fn delete_session(
    request: DeleteSessionRequest,
    responder: Responder<DeleteSessionResponse>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    let id = request.session_id.0.to_string();
    // Reject deletion while a prompt is active. This keeps the protocol
    // response tied to a live session and avoids deleting a file whose agent
    // is still allowed to append a turn.
    let handle = {
        let mut sessions = state.sessions.lock().unwrap();
        if state.prompts.in_flight.lock().unwrap().contains_key(&id) {
            let _ = responder.respond_with_error(AcError::invalid_request().data(
                "cannot delete a session while a prompt is running; cancel it first".to_owned(),
            ));
            return Ok(());
        }
        sessions.remove(&id)
    };
    if let Some(handle) = handle {
        shutdown_session(handle).await;
    }
    if let Err(error) = delete_session_everywhere(&id, &state.session_root) {
        return respond_anyhow(responder, error);
    }
    tracing::info!(session = %id, "ACP session deleted");
    let _ = responder.respond(DeleteSessionResponse::new());
    Ok(())
}

/// Stop both owners of an ACP session before its durable file is removed.
/// Hanging tasks are aborted after one shared deadline, not one timeout per
/// task, so a broken provider cannot delay deletion indefinitely.
async fn shutdown_session(handle: SessionHandle) {
    let SessionHandle {
        input_tx,
        cancel,
        mut agent_task,
        mut forwarder_task,
    } = handle;
    cancel.cancel();
    drop(input_tx);
    let stopped = tokio::time::timeout(SESSION_TASK_TIMEOUT, async {
        let _ = (&mut agent_task).await;
        let _ = (&mut forwarder_task).await;
    })
    .await
    .is_ok();
    if !stopped {
        agent_task.abort();
        forwarder_task.abort();
        let _ = agent_task.await;
        let _ = forwarder_task.await;
    }
}

/// `session/delete` supplies only an id, not a cwd, so scan every workspace
/// group under the session root. A missing file counts as success.
fn delete_session_everywhere(id: &str, session_root: &std::path::Path) -> Result<()> {
    let parsed = session::SessionId::parse(id).map_err(anyhow::Error::msg)?;
    let file_name = format!("{parsed}.jsonl");
    let entries = match std::fs::read_dir(session_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(anyhow::Error::new(error).context("list session root")),
    };
    for entry in entries {
        let entry = entry.context("read session root entry")?;
        let workspace_dir = entry.path();
        let metadata = match std::fs::metadata(&workspace_dir) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(anyhow::Error::new(error)
                    .context(format!("inspect `{}`", workspace_dir.display())));
            }
        };
        if !metadata.is_dir() {
            continue;
        }
        let path = workspace_dir.join(&file_name);
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => {
                std::fs::remove_file(&path)
                    .with_context(|| format!("delete `{}`", path.display()))?;
                let lock_path = path.with_extension("jsonl.lock");
                match std::fs::remove_file(&lock_path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(anyhow::Error::new(error)
                            .context(format!("delete `{}`", lock_path.display())));
                    }
                }
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(
                    anyhow::Error::new(error).context(format!("inspect `{}`", path.display()))
                );
            }
        }
    }
    Ok(())
}

async fn prompt(
    request: PromptRequest,
    responder: Responder<PromptResponse>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    let session_id = request.session_id.0.to_string();
    // Exactly one prompt in flight per session: a concurrent second prompt
    // would interleave two conversations into one history. Acquire locks in
    // sessions → prompts order, matching deletion and cancellation, so a
    // prompt cannot slip in while deletion checks activity.
    let sessions = state.sessions.lock().unwrap();
    let Some(handle) = sessions.get(&session_id) else {
        let _ = responder.respond_with_error(
            AcError::invalid_params().data(format!("unknown session `{session_id}`")),
        );
        return Ok(());
    };
    let mut in_flight = state.prompts.in_flight.lock().unwrap();
    if in_flight.contains_key(&session_id) {
        drop(in_flight);
        drop(sessions);
        let _ = responder.respond_with_error(
            AcError::invalid_request()
                .data("a prompt is already running for this session; cancel it first".to_owned()),
        );
        return Ok(());
    }
    in_flight.insert(
        session_id.clone(),
        InFlight {
            responder,
            cancelled: false,
            error: None,
        },
    );
    let input_tx = handle.input_tx.clone();
    drop(in_flight);
    drop(sessions);

    let text = flatten_prompt(&request.prompt);
    if text.trim().is_empty() {
        // The agent ignores blank messages; answer directly instead of
        // leaving a prompt parked forever.
        state
            .prompts
            .resolve(&session_id, Some(StopReason::EndTurn));
        return Ok(());
    }
    if input_tx.send(InputMessage::Message(text)).is_err() {
        state.prompts.resolve(&session_id, None);
        return Ok(());
    }
    // The turn now runs in the agent task; the forwarder resolves the parked
    // responder on `TurnFinished` (or agent death). Returning without
    // responding is safe: the SDK keeps the request open until this
    // responder answers.
    Ok(())
}

// ---------------------------------------------------------------------
// Session assembly + event forwarding
// ---------------------------------------------------------------------

/// Build the `(store, registry)` pair for a session root. This mirrors what
/// `main.rs` assembles for the TUI, rooted at the request's `cwd` instead of
/// the process cwd. `rtk` stays off: it is a local shell-output preference
/// from the developer's config file, and editor sessions should not depend on
/// it being installed.
async fn build_session_stack(
    cwd: &std::path::Path,
    session_root: &std::path::Path,
) -> Result<(SessionStore, ToolRegistry)> {
    let cwd = cwd.to_path_buf();
    let session_root = session_root.to_path_buf();
    let joined = tokio::time::timeout(
        SESSION_TASK_TIMEOUT,
        tokio::task::spawn_blocking(move || {
            let workspace_root = std::fs::canonicalize(&cwd)
                .with_context(|| format!("resolve session cwd `{}`", cwd.display()))?;
            let store = SessionStore::new(&session_root, &workspace_root)?;
            let tools = default_registry(ToolConfig::new(&workspace_root, false))?;
            Ok::<_, anyhow::Error>((store, tools))
        }),
    )
    .await
    .context("build ACP session stack timed out")?;
    joined.context("build ACP session stack task failed")?
}

/// Convert ACP's session-local stdio declarations without retaining ACP wire
/// types outside this frontend. HTTP, SSE, and MCP-over-ACP are rejected
/// rather than silently omitted.
fn acp_mcp_servers(servers: &[McpServer]) -> Result<Vec<mcp::McpServerConfig>> {
    let servers = servers
        .iter()
        .map(|server| match server {
            McpServer::Stdio(server) => Ok(mcp::McpServerConfig {
                name: server.name.clone(),
                transport: mcp::McpTransportConfig::Stdio {
                    command: server.command.clone(),
                    args: server.args.clone(),
                    env: server
                        .env
                        .iter()
                        .map(|entry| (entry.name.clone(), entry.value.clone()))
                        .collect(),
                },
            }),
            McpServer::Http(server) => anyhow::bail!(
                "MCP server `{}` requests HTTP, which this Harness build does not support",
                server.name
            ),
            McpServer::Sse(server) => anyhow::bail!(
                "MCP server `{}` requests SSE, which this Harness build does not support",
                server.name
            ),
            _ => anyhow::bail!("requested MCP transport is unsupported"),
        })
        .collect::<Result<Vec<_>>>()?;
    mcp::McpConfig {
        servers: servers.clone(),
    }
    .validate()?;
    Ok(servers)
}

/// Insert one assembled session without replacing an already-live handle.
///
/// Assembly deliberately happens outside the sessions mutex because it can
/// initialize tools, MCP servers, and an agent. The final occupied/vacant
/// decision must nevertheless be one locked operation: a duplicate receives
/// its own assembled handle back so the caller can stop both of its tasks.
fn try_register_session(
    state: &AcpState,
    acp_session_id: String,
    handle: SessionHandle,
) -> std::result::Result<(), SessionHandle> {
    let mut sessions = state.sessions.lock().unwrap();
    match sessions.entry(acp_session_id) {
        Entry::Vacant(entry) => {
            entry.insert(handle);
            Ok(())
        }
        Entry::Occupied(_) => Err(handle),
    }
}

/// Discard an assembled session that never became the registered owner.
/// Abort the forwarder before the agent can close its event channel: a
/// forwarder that exits normally calls `PromptTracker::resolve`, and its
/// session id may belong to the original live agent that won registration.
async fn discard_session(handle: SessionHandle) {
    let SessionHandle {
        input_tx,
        cancel,
        agent_task,
        forwarder_task,
    } = handle;
    cancel.cancel();
    drop(input_tx);
    forwarder_task.abort();
    agent_task.abort();
    let _ = forwarder_task.await;
    let _ = agent_task.await;
}

/// Spawn the agent task plus its event forwarder and register the session's
/// input channel under `acp_session_id`. The forwarder owns everything
/// event-shaped: notification translation and prompt-turn resolution. The
/// resulting task handles are retained by `SessionHandle` so lifecycle
/// operations can stop both owners before touching session files.
async fn spawn_agent(
    state: &AcpState,
    store: SessionStore,
    tools: ToolRegistry,
    session: session::Session,
    connection: ConnectionTo<agent_client_protocol::Client>,
    acp_session_id: String,
    mcp_servers: Vec<mcp::McpServerConfig>,
) -> Result<()> {
    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let project_context = project_context_for(tools.workspace_root(), state.no_context_files);

    let builder = AgentBuilder::new(
        state.provider.clone(),
        state.config.model.clone(),
        tools,
        cancel.clone(),
    )
    .with_reasoning(state.config.reasoning)
    .with_compaction(state.config.compaction.clone())
    .with_subagents(state.config.subagents, false)
    .with_mcp_servers(mcp_servers)
    .with_project_context(project_context)
    .with_session(store, session)
    .with_provider_factory(provider_factory(
        state.copilot_auth.clone(),
        state.config.codex_auth.clone(),
    ));
    let agent = tokio::time::timeout(SESSION_TASK_TIMEOUT, builder.build())
        .await
        .context("build ACP agent timed out")??;
    let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

    let forwarder_task = tokio::spawn(forward_events(
        event_rx,
        connection,
        state.prompts.clone(),
        acp_session_id.clone(),
    ));

    let handle = SessionHandle {
        input_tx,
        cancel,
        agent_task,
        forwarder_task,
    };
    match try_register_session(state, acp_session_id.clone(), handle) {
        Ok(()) => Ok(()),
        Err(handle) => {
            // A duplicate load can race another request during assembly. Keep
            // the original live session and stop the newly assembled pair.
            discard_session(handle).await;
            anyhow::bail!("session `{acp_session_id}` is already loaded");
        }
    }
}

/// Consume one session's agent events until the agent task exits: translate
/// each into a `session/update` notification and resolve the parked prompt on
/// `TurnFinished`. Its task handle is owned by `SessionHandle`, and deletion
/// waits for it after cancelling the session.
async fn forward_events(
    mut event_rx: mpsc::UnboundedReceiver<AgentEvent>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    prompts: Arc<PromptTracker>,
    acp_session_id: String,
) {
    let mut ids = ToolCallIds::default();
    // Billing is cumulative and only attached to the next authoritative
    // context update; it never determines the context occupancy itself.
    let mut cumulative_cost = None;
    while let Some(event) = event_rx.recv().await {
        match &event {
            AgentEvent::TurnFinished => {
                let reason = if prompts.is_cancelled(&acp_session_id) {
                    StopReason::Cancelled
                } else {
                    StopReason::EndTurn
                };
                prompts.resolve(&acp_session_id, Some(reason));
            }
            AgentEvent::Error(message) => {
                prompts.mark_error(&acp_session_id, message);
            }
            AgentEvent::TextDelta(_) => {
                prompts.clear_error(&acp_session_id);
            }
            AgentEvent::UsageUpdated { cost, .. } => {
                cumulative_cost = cost.parse::<f64>().ok();
            }
            _ => {}
        }
        let update = if let AgentEvent::ContextUsageUpdated {
            used_tokens,
            max_tokens,
        } = &event
        {
            let update = UsageUpdate::new(*used_tokens, *max_tokens);
            Some(SessionUpdate::UsageUpdate(match cumulative_cost {
                Some(cost) => update.cost(v1::Cost::new(cost, "USD")),
                None => update,
            }))
        } else {
            translate_event(&event, &mut ids, 0)
        };
        if let Some(update) = update {
            let notification =
                SessionNotification::new(SessionId::from(acp_session_id.clone()), update);
            // A send error means the client connection is gone; stop feeding
            // it. The agent notices on its own at the next turn boundary.
            if connection.send_notification(notification).is_err() {
                break;
            }
        }
    }
    // Agent task gone: fail any still-parked prompt so the editor is not left
    // waiting on a turn that will never finish. This is unconditional because
    // completed prompts are removed at their own turn boundary; a lifetime
    // completion flag can strand a later prompt.
    prompts.resolve(&acp_session_id, None);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Cli, FileConfig};
    use agent_client_protocol::{ByteStreams, Client as ClientRole, ConnectTo, JsonRpcResponse};
    use async_trait::async_trait;
    use futures_util::stream;
    use llm::{CompletionRequest, EventStream, LlmError, ModelInfo, StreamEvent, Usage};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    // ------------------------------------------------------------------
    // Pure translation
    // ------------------------------------------------------------------

    fn text_block(s: &str) -> ContentBlock {
        ContentBlock::from(s)
    }

    #[test]
    fn flatten_keeps_text_verbatim_and_joins_with_newlines() {
        let blocks = vec![text_block("first"), text_block("second")];
        assert_eq!(flatten_prompt(&blocks), "first\nsecond");
    }

    #[test]
    fn flatten_inlines_text_resources_as_fenced_context() {
        let blocks = vec![ContentBlock::Resource(v1::EmbeddedResource::new(
            v1::EmbeddedResourceResource::TextResourceContents(v1::TextResourceContents::new(
                "let x = 1;\n",
                "file:///tmp/x.rs",
            )),
        ))];
        assert_eq!(flatten_prompt(&blocks), "```\nlet x = 1;\n```");
    }

    #[test]
    fn flatten_notes_resource_links_without_fetching() {
        let blocks = vec![ContentBlock::ResourceLink(v1::ResourceLink::new(
            "notes.txt",
            "file:///tmp/notes.txt",
        ))];
        let flat = flatten_prompt(&blocks);
        assert!(flat.contains("attached file"), "{flat}");
        assert!(flat.contains("file:///tmp/notes.txt"), "{flat}");
        assert!(
            !flat.contains("contents of notes"),
            "must not invent content"
        );
    }

    #[test]
    fn tool_kinds_map_to_acp_categories() {
        assert_eq!(tool_kind("read"), ToolKind::Read);
        assert_eq!(tool_kind("edit"), ToolKind::Edit);
        assert_eq!(tool_kind("write"), ToolKind::Edit);
        assert_eq!(tool_kind("bash"), ToolKind::Execute);
        assert_eq!(tool_kind("find"), ToolKind::Search);
        assert_eq!(tool_kind("grep"), ToolKind::Search);
        assert_eq!(tool_kind("unknown"), ToolKind::Other);
    }

    #[test]
    fn concurrent_same_name_calls_correlate_by_harness_call_id() {
        let mut ids = ToolCallIds::default();
        // Two `subagent` calls: same tool name, similar summaries, distinct
        // harness call ids — exactly the case FIFO-by-name guessing breaks
        // on.
        let started_first = match translate_event(
            &AgentEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "subagent".into(),
                summary: "subagent: audit agent".into(),
            },
            &mut ids,
            0,
        ) {
            Some(SessionUpdate::ToolCall(call)) => call,
            other => panic!("expected tool call, got {other:?}"),
        };
        let started_second = match translate_event(
            &AgentEvent::ToolCallStarted {
                call_id: "call-2".into(),
                name: "subagent".into(),
                summary: "subagent: audit tools".into(),
            },
            &mut ids,
            0,
        ) {
            Some(SessionUpdate::ToolCall(call)) => call,
            other => panic!("expected tool call, got {other:?}"),
        };
        assert_ne!(started_first.tool_call_id, started_second.tool_call_id);

        // Finishing the second first must target the second ACP card, not
        // whichever call started earlier.
        let finished = translate_event(
            &AgentEvent::ToolCallFinished {
                call_id: "call-2".into(),
                name: "subagent".into(),
                summary: "subagent: audit tools".into(),
                ok: true,
                duration_ms: 3,
                output: "tools report".into(),
                error: None,
            },
            &mut ids,
            0,
        );
        let Some(SessionUpdate::ToolCallUpdate(update)) = finished else {
            panic!("expected tool call update, got {finished:?}");
        };
        assert_eq!(update.tool_call_id, started_second.tool_call_id);
        assert_eq!(update.fields.status, Some(ToolCallStatus::Completed));

        // The remaining in-flight call is the first one's.
        let finished = translate_event(
            &AgentEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "subagent".into(),
                summary: "subagent: audit agent".into(),
                ok: false,
                duration_ms: 1,
                output: String::new(),
                error: Some("boom".into()),
            },
            &mut ids,
            0,
        );
        let Some(SessionUpdate::ToolCallUpdate(update)) = finished else {
            panic!("expected tool call update, got {finished:?}");
        };
        assert_eq!(update.tool_call_id, started_first.tool_call_id);
        assert_eq!(update.fields.status, Some(ToolCallStatus::Failed));
        let Some(Some(ToolCallContent::Content(content))) =
            update.fields.content.map(|mut c| c.pop())
        else {
            panic!("expected content on the update");
        };
        assert!(
            matches!(content.content, ContentBlock::Text(ref t) if t.text == "boom"),
            "error text should be surfaced"
        );

        // Nothing stays in flight after both finishes.
        assert!(ids.in_flight.is_empty());
    }

    #[test]
    fn finish_without_start_is_an_invariant_violation_but_still_updates() {
        let mut ids = ToolCallIds::default();
        let finished = translate_event(
            &AgentEvent::ToolCallFinished {
                call_id: "ghost".into(),
                name: "read".into(),
                summary: "read a.rs".into(),
                ok: true,
                duration_ms: 1,
                output: "x".into(),
                error: None,
            },
            &mut ids,
            0,
        );
        // Defensive fallback: the editor still receives a terminal update;
        // normal execution never takes this path.
        assert!(matches!(finished, Some(SessionUpdate::ToolCallUpdate(_))));
    }

    #[test]
    fn context_usage_is_authoritative_and_billing_usage_is_not_occupancy() {
        let billing = AgentEvent::UsageUpdated {
            input_tokens: 100,
            output_tokens: 20,
            cached_tokens: 10,
            reasoning_tokens: 5,
            cost: "0.5".into(),
        };
        let context = AgentEvent::ContextUsageUpdated {
            used_tokens: 95,
            max_tokens: 128_000,
        };
        let mut ids = ToolCallIds::default();
        assert!(translate_event(&billing, &mut ids, 0).is_none());
        let update = match translate_event(&context, &mut ids, 0) {
            Some(SessionUpdate::UsageUpdate(update)) => update,
            other => panic!("expected context usage update, got {other:?}"),
        };
        assert_eq!(update.used, 95);
        assert_eq!(update.size, 128_000);
    }

    #[test]
    fn non_protocol_events_translate_to_nothing() {
        let events = [
            AgentEvent::TurnFinished,
            AgentEvent::Retrying {
                attempt: 1,
                message: "transient".into(),
            },
            AgentEvent::Notice("model list failed".into()),
            AgentEvent::SessionChanged {
                id: "s".into(),
                title: None,
                loaded: false,
            },
            AgentEvent::CompactionFinished {
                compacted_through: 3,
                summary_bytes: 100,
                auto: true,
                reason: agent::CompactionReason::Auto,
            },
        ];
        for event in &events {
            let mut ids = ToolCallIds::default();
            assert!(
                translate_event(event, &mut ids, 1000).is_none(),
                "{event:?} has no ACP counterpart"
            );
        }
    }

    #[test]
    fn initialize_negotiates_down_to_v1_and_advertises_capabilities() {
        let request = InitializeRequest::new(ProtocolVersion::V1);
        let response = initialize_response(&request);
        assert_eq!(response.protocol_version, ProtocolVersion::V1);
        assert!(response.agent_capabilities.load_session);
        assert!(
            response
                .agent_capabilities
                .prompt_capabilities
                .embedded_context
        );
        assert!(
            response
                .agent_capabilities
                .session_capabilities
                .list
                .is_some()
        );
        assert!(
            response
                .agent_capabilities
                .session_capabilities
                .delete
                .is_some()
        );
        assert!(response.auth_methods.is_empty(), "no auth over ACP");

        // A client asking for a different version gets v1 back either way:
        // echoing an older version would claim support we don't have.
        let newer = InitializeRequest::new(ProtocolVersion::from(9u16));
        assert_eq!(
            initialize_response(&newer).protocol_version,
            ProtocolVersion::V1
        );
    }

    // ------------------------------------------------------------------
    // Integration: full connection over an in-memory transport with a
    // scripted provider. Follows the SDK's own test shape (LocalSet +
    // spawn_local) because handler futures are not required to be Send.
    // ------------------------------------------------------------------

    /// Provider serving canned scripts per call, mirroring the mock used by
    /// the agent loop tests.
    struct ScriptProvider {
        calls: AtomicUsize,
        scripts: Vec<Vec<Result<StreamEvent, String>>>,
    }

    impl ScriptProvider {}

    struct HangingProvider;

    struct TaskDropFlag(Arc<AtomicUsize>);

    impl Drop for TaskDropFlag {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[derive(Clone)]
    struct TestAssemblyFlags {
        agent_dropped: Arc<AtomicUsize>,
        forwarder_dropped: Arc<AtomicUsize>,
        used: Arc<AtomicUsize>,
    }

    fn test_assembly_flags() -> TestAssemblyFlags {
        TestAssemblyFlags {
            agent_dropped: Arc::new(AtomicUsize::new(0)),
            forwarder_dropped: Arc::new(AtomicUsize::new(0)),
            used: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Build task owners that behave like an assembled agent but expose their
    /// lifetime to the registration race test.
    fn test_session_handle(flags: &TestAssemblyFlags) -> SessionHandle {
        let (input_tx, mut input_rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let agent_marker = TaskDropFlag(flags.agent_dropped.clone());
        let used = flags.used.clone();
        let agent_task = tokio::spawn(async move {
            let _agent_marker = agent_marker;
            if input_rx.recv().await.is_some() {
                used.fetch_add(1, Ordering::SeqCst);
            }
        });

        let forwarder_marker = TaskDropFlag(flags.forwarder_dropped.clone());
        let forwarder_task = tokio::spawn(async move {
            let _forwarder_marker = forwarder_marker;
            std::future::pending::<()>().await;
        });

        SessionHandle {
            input_tx,
            cancel,
            agent_task,
            forwarder_task,
        }
    }

    fn test_acp_state() -> Arc<AcpState> {
        Arc::new(AcpState {
            session_root: PathBuf::new(),
            provider: Arc::new(HangingProvider),
            config: acp_config(),
            copilot_auth: None,
            no_context_files: true,
            sessions: Mutex::new(HashMap::new()),
            prompts: Arc::new(PromptTracker {
                in_flight: Mutex::new(HashMap::new()),
            }),
        })
    }

    async fn register_test_assembly(
        state: Arc<AcpState>,
        session_id: String,
        barrier: Arc<tokio::sync::Barrier>,
        handle: SessionHandle,
    ) -> anyhow::Result<()> {
        // Both assembled agents reach the registration point together. The
        // winner is intentionally scheduler-dependent; only the locked entry
        // operation decides which one owns the session.
        barrier.wait().await;
        match try_register_session(&state, session_id.clone(), handle) {
            Ok(()) => Ok(()),
            Err(handle) => {
                discard_session(handle).await;
                anyhow::bail!("session `{session_id}` is already loaded");
            }
        }
    }

    /// Two completed assemblies racing to register one id must leave exactly
    /// one owner in the map. The losing agent and forwarder are both aborted,
    /// while the winner's input channel remains usable.
    #[tokio::test]
    async fn concurrent_session_registration_keeps_original_and_cleans_loser() {
        let state = test_acp_state();
        let session_id = "same-session".to_owned();
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let left_flags = test_assembly_flags();
        let right_flags = test_assembly_flags();

        let left = tokio::spawn(register_test_assembly(
            state.clone(),
            session_id.clone(),
            barrier.clone(),
            test_session_handle(&left_flags),
        ));
        let right = tokio::spawn(register_test_assembly(
            state.clone(),
            session_id.clone(),
            barrier,
            test_session_handle(&right_flags),
        ));
        let left_result = left.await.unwrap();
        let right_result = right.await.unwrap();

        assert_eq!(
            left_result.is_ok() as usize + right_result.is_ok() as usize,
            1
        );
        assert_eq!(state.sessions.lock().unwrap().len(), 1);

        let (winner, loser) = if left_result.is_ok() {
            (&left_flags, &right_flags)
        } else {
            (&right_flags, &left_flags)
        };
        assert_eq!(loser.agent_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(loser.forwarder_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(winner.agent_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(winner.forwarder_dropped.load(Ordering::SeqCst), 0);

        // The registered/original handle was not replaced by the duplicate.
        let input_tx = state
            .sessions
            .lock()
            .unwrap()
            .get(&session_id)
            .expect("the winning session remains registered")
            .input_tx
            .clone();
        input_tx.send(InputMessage::Interrupt).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while winner.used.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the original session input channel remains usable");

        let handle = state.sessions.lock().unwrap().remove(&session_id).unwrap();
        discard_session(handle).await;
        assert_eq!(winner.agent_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(winner.forwarder_dropped.load(Ordering::SeqCst), 1);
    }

    #[async_trait]
    impl Provider for HangingProvider {
        fn name(&self) -> &str {
            "hang"
        }
        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            Ok(Box::pin(stream::pending()))
        }
        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    #[async_trait]
    impl Provider for ScriptProvider {
        fn name(&self) -> &str {
            "script"
        }
        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            let index = self.calls.fetch_add(1, Ordering::SeqCst);
            let script = self.scripts.get(index).cloned().unwrap_or_default();
            Ok(Box::pin(stream::iter(
                script
                    .into_iter()
                    .map(|step| step.map_err(LlmError::Stream)),
            )))
        }
        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(vec![ModelInfo {
                id: "demo".into(),
                name: Some("Demo".into()),
                context_length: Some(4096),
            }])
        }
    }

    fn acp_config() -> Config {
        Config::resolve_from_file(
            &Cli::default(),
            &FileConfig {
                provider: Some("opencode-go".into()),
                ..FileConfig::default()
            },
            PathBuf::from("/tmp/harness-acp-config.toml"),
            |_| Some("secret".into()),
        )
        .unwrap()
    }

    /// Collect `session/update` notifications arriving on the client side of
    /// an in-memory connection, then run `main` with the agent-side handle.
    async fn run_client_side<R>(
        transport: impl ConnectTo<agent_client_protocol::Client>,
        main: impl AsyncFnOnce(
            ConnectionTo<agent_client_protocol::Agent>,
            mpsc::UnboundedReceiver<SessionUpdate>,
        ) -> AcResult<R>,
    ) -> R {
        let (update_tx, update_rx) = mpsc::unbounded_channel();
        ClientRole
            .builder()
            .on_receive_notification(
                async move |notification: SessionNotification, _cx| {
                    let _ = update_tx.send(notification.update);
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(transport, async move |cx| main(cx, update_rx).await)
            .await
            .expect("client connection")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn initialize_new_prompt_stream_end_turn_over_one_connection() {
        use tokio::task::LocalSet;
        let local = LocalSet::new();
        local
            .run_until(async {
                let workspace = tempdir().unwrap();
                let DuplexPair {
                    server_reader,
                    server_writer,
                    client_reader,
                    client_writer,
                } = duplex_pair();

                let provider = Arc::new(ScriptProvider {
                    calls: AtomicUsize::new(0),
                    scripts: vec![vec![
                        Ok(StreamEvent::TextDelta("Hello ".into())),
                        Ok(StreamEvent::ReasoningDelta("thinking".into())),
                        Ok(StreamEvent::TextDelta("world".into())),
                        Ok(StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: Some(Usage {
                                input_tokens: 10,
                                output_tokens: 2,
                                cached_tokens: None,
                                reasoning_tokens: None,
                                cost: Some(0.25),
                            }),
                        }),
                    ]],
                });

                // Server side: our ACP frontend. `serve` takes an explicit
                // session root so the test never touches real ~/.harness/sessions
                // nor mutates the process-global HARNESS_SESSION_DIR env (which
                // would race between tests running in parallel threads).
                let session_root = workspace.path().join("sessions");
                let server_session_root = session_root.clone();
                tokio::task::spawn_local({
                    let provider: Arc<dyn Provider> = provider.clone();
                    async move {
                        let config = acp_config();
                        // `serve` is rooted on this test's half of the duplex
                        // pair rather than process stdio, so the two peers
                        // actually talk to each other.
                        let _ = serve(
                            provider,
                            config,
                            None,
                            true,
                            server_session_root,
                            ByteStreams::new(server_writer, server_reader),
                        )
                        .await
                        .inspect_err(|error| eprintln!("server error: {error:#}"));
                    }
                });
                let result = run_client_side(
                    ByteStreams::new(client_writer, client_reader),
                    async |cx: ConnectionTo<agent_client_protocol::Agent>,
                           updates: mpsc::UnboundedReceiver<SessionUpdate>| {
                        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                            .block_task()
                            .await
                            .expect("initialize");
                        let new_session = cx
                            .send_request(NewSessionRequest::new(workspace.path()))
                            .block_task()
                            .await
                            .expect("session/new");
                        let session_id = new_session.session_id.clone();

                        let response = cx
                            .send_request(PromptRequest::new(
                                session_id.clone(),
                                vec![text_block("say hello")],
                            ))
                            .block_task()
                            .await
                            .expect("prompt");

                        Ok((session_id, response, updates))
                    },
                )
                .await;

                let (session_id, response, mut updates) = result;
                assert_eq!(response.stop_reason, StopReason::EndTurn);

                let mut seen = Vec::new();
                while let Ok(update) = updates.try_recv() {
                    seen.push(update);
                }
                assert!(
                    seen.iter().any(|update| matches!(
                        update,
                        SessionUpdate::AgentMessageChunk(chunk)
                            if matches!(&chunk.content, ContentBlock::Text(t) if t.text == "Hello ")
                    )),
                    "missing streamed text chunk: {seen:?}"
                );
                assert!(
                    seen.iter()
                        .any(|update| matches!(update, SessionUpdate::AgentThoughtChunk(_))),
                    "reasoning deltas must stream as thought chunks: {seen:?}"
                );
                assert!(
                    !seen.iter().any(|update| matches!(
                        update,
                        SessionUpdate::ToolCall(_) | SessionUpdate::ToolCallUpdate(_)
                    )),
                    "a text-only turn must not emit tool calls: {seen:?}"
                );

                // HARNESS-1 is covered by a dedicated test below
                // (`mid_stream_provider_error_fails_the_prompt`); the load
                // round-trip needs the connection here, so keep this test
                // to the happy path + persistence.

                // The turn was persisted under our session id. The first
                // connection owns the live agent, so use a fresh connection
                // to exercise the real `session/load` path.
                drop(updates);
                let DuplexPair {
                    server_reader,
                    server_writer,
                    client_reader,
                    client_writer,
                } = duplex_pair();
                let server_session_root = session_root.clone();
                let provider: Arc<dyn Provider> = provider.clone();
                tokio::task::spawn_local(async move {
                    let _ = serve(
                        provider,
                        acp_config(),
                        None,
                        true,
                        server_session_root,
                        ByteStreams::new(server_writer, server_reader),
                    )
                    .await
                    .inspect_err(|error| eprintln!("load server error: {error:#}"));
                });
                run_client_side(
                    ByteStreams::new(client_writer, client_reader),
                    async |cx: ConnectionTo<agent_client_protocol::Agent>,
                           _updates: mpsc::UnboundedReceiver<SessionUpdate>| {
                        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                            .block_task()
                            .await
                            .expect("initialize load connection");
                        cx.send_request(LoadSessionRequest::new(
                            session_id.clone(),
                            workspace.path(),
                        ))
                        .block_task()
                        .await
                        .expect("session/load");
                        Ok(())
                    },
                )
                .await;
            })
            .await;
    }

    /// A mid-stream provider error must surface as a failed prompt — with
    /// the error diagnostic visible — never as a successful blank EndTurn.
    /// The scripted turn emits text, then a stream error, then the agent
    /// closes the turn with its own Error event + TurnFinished.
    #[tokio::test(flavor = "current_thread")]
    async fn mid_stream_provider_error_fails_the_prompt() {
        use tokio::task::LocalSet;
        let local = LocalSet::new();
        local
            .run_until(async {
                let workspace = tempdir().unwrap();
                let DuplexPair {
                    server_reader,
                    server_writer,
                    client_reader,
                    client_writer,
                } = duplex_pair();

                let provider = Arc::new(ScriptProvider {
                    calls: AtomicUsize::new(0),
                    scripts: vec![vec![
                        Ok(StreamEvent::TextDelta("partial".into())),
                        Err("provider exploded".into()),
                    ]],
                });
                let session_root = workspace.path().join("sessions");
                tokio::task::spawn_local({
                    let provider: Arc<dyn Provider> = provider.clone();
                    async move {
                        let _ = serve(
                            provider,
                            acp_config(),
                            None,
                            true,
                            session_root,
                            ByteStreams::new(server_writer, server_reader),
                        )
                        .await
                        .inspect_err(|error| eprintln!("server error: {error:#}"));
                    }
                });
                let (response, mut updates) = run_client_side(
                    ByteStreams::new(client_writer, client_reader),
                    async |cx: ConnectionTo<agent_client_protocol::Agent>,
                           updates: mpsc::UnboundedReceiver<SessionUpdate>| {
                        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                            .block_task()
                            .await
                            .expect("initialize");
                        let new_session = cx
                            .send_request(NewSessionRequest::new(workspace.path()))
                            .block_task()
                            .await
                            .expect("session/new");
                        let prompt_result = cx
                            .send_request(PromptRequest::new(
                                new_session.session_id.clone(),
                                vec![text_block("fail please")],
                            ))
                            .block_task()
                            .await;
                        Ok((prompt_result, updates))
                    },
                )
                .await;
                assert!(
                    response.is_err(),
                    "a failed turn must not resolve as successful blank output"
                );
                let mut seen = Vec::new();
                while let Ok(update) = updates.try_recv() {
                    seen.push(update);
                }
                assert!(
                    seen.iter().any(|update| matches!(
                        update,
                        SessionUpdate::AgentMessageChunk(chunk)
                            if matches!(&chunk.content, ContentBlock::Text(t) if t.text.contains("provider exploded"))
                    )),
                    "the error diagnostic must be visible: {seen:?}"
                );
            })
            .await;
    }

    /// A prompt parked against a provider whose stream never yields keeps the
    /// turn in flight until the client sends `session/cancel`; the interrupt
    /// must resolve the parked `PromptResponse` as `Cancelled` rather than
    /// hanging the editor's request forever.
    #[tokio::test(flavor = "current_thread")]
    async fn cancel_interrupts_pending_turn_and_cancels_prompt() {
        use tokio::task::LocalSet;
        let local = LocalSet::new();
        local
            .run_until(async {
                let workspace = tempdir().unwrap();
                let DuplexPair {
                    server_reader,
                    server_writer,
                    client_reader,
                    client_writer,
                } = duplex_pair();

                let session_root = workspace.path().join("sessions");
                tokio::task::spawn_local({
                    async move {
                        let config = acp_config();
                        let _ = serve(
                            Arc::new(HangingProvider),
                            config,
                            None,
                            true,
                            session_root,
                            ByteStreams::new(server_writer, server_reader),
                        )
                        .await
                        .inspect_err(|error| eprintln!("server error: {error:#}"));
                    }
                });
                let (_, response, mut updates) = run_client_side(
                    ByteStreams::new(client_writer, client_reader),
                    async |cx: ConnectionTo<agent_client_protocol::Agent>,
                           updates: mpsc::UnboundedReceiver<SessionUpdate>| {
                        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                            .block_task()
                            .await
                            .expect("initialize");
                        let new_session = cx
                            .send_request(NewSessionRequest::new(workspace.path()))
                            .block_task()
                            .await
                            .expect("session/new");
                        let session_id = new_session.session_id.clone();

                        // Park a prompt whose turn can only ever end by interrupt,
                        // then cancel it over the same connection.
                        let prompt = cx.send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![text_block("stay pending")],
                        ));
                        cx.send_notification(CancelNotification::new(session_id.clone()))?;
                        let response = prompt.block_task().await.expect("cancelled prompt");

                        Ok((session_id, response, updates))
                    },
                )
                .await;

                assert_eq!(response.stop_reason, StopReason::Cancelled);
                // The turn never streamed (the hang provider yields nothing), so
                // whatever ephemeral updates arrived, the prompt was still
                // resolved rather than hung open.
                let mut seen = Vec::new();
                while let Ok(update) = updates.try_recv() {
                    seen.push(update);
                }
                assert!(
                    !seen.iter().any(|update| matches!(
                        update,
                        SessionUpdate::AgentMessageChunk(_) | SessionUpdate::AgentThoughtChunk(_)
                    )),
                    "a cancelled turn must not deliver assistant text: {seen:?}"
                );
            })
            .await;
    }

    /// Concrete duplex pair types (impl-trait type aliases are unstable).
    struct DuplexPair {
        server_reader:
            futures_util::io::BufReader<tokio_util::compat::Compat<tokio::io::DuplexStream>>,
        server_writer: tokio_util::compat::Compat<tokio::io::DuplexStream>,
        client_reader:
            futures_util::io::BufReader<tokio_util::compat::Compat<tokio::io::DuplexStream>>,
        client_writer: tokio_util::compat::Compat<tokio::io::DuplexStream>,
    }

    fn duplex_pair() -> DuplexPair {
        let (client_writer, server_reader) = tokio::io::duplex(64 * 1024);
        let (server_writer, client_reader) = tokio::io::duplex(64 * 1024);
        use tokio_util::compat::{TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};
        DuplexPair {
            server_reader: futures_util::io::BufReader::new(server_reader.compat()),
            server_writer: server_writer.compat_write(),
            client_reader: futures_util::io::BufReader::new(client_reader.compat()),
            client_writer: client_writer.compat_write(),
        }
    }

    // Keep unused imports referenced when only unit tests run.
    #[allow(dead_code)]
    fn _assert_traits() {
        fn is_response<T: JsonRpcResponse>() {}
        is_response::<PromptResponse>();
    }
}
