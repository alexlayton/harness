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
//! MCP-over-ACP, mid-session model switching. Unhandled requests fall through
//! to the SDK default of method-not-found.
//!
//! Stdout ownership is inverted here: stdout carries JSON-RPC only. Tracing
//! stays behind `HARNESS_LOG` (file-only), and this module never writes to
//! stderr either — editors surface child stderr as agent noise.

use crate::agent::{Agent, AgentEvent};
use crate::config::{Config, ProviderArg};
use crate::project_context_for;
use crate::tools::{ToolConfig, ToolRegistry, default_registry};
use agent_client_protocol::schema::v1::{
    AgentCapabilities, AuthenticateRequest, CancelNotification, ContentBlock, ContentChunk,
    DeleteSessionRequest, DeleteSessionResponse, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse,
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
use std::collections::HashMap;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

/// The single mode advertised for every session: harness has no permission
/// levels to switch between (tools always run), so `set_mode` stays
/// unsupported while the mode list still gives editors something to display.
const MODE_ID: &str = "work";

/// Everything needed to drive one live session.
struct SessionHandle {
    /// Commands for the agent's run loop. Held open for the session's
    /// lifetime; dropping the last sender lets the agent task wind down.
    input_tx: mpsc::UnboundedSender<InputMessage>,
}

/// A prompt request parked until its turn ends. Stored in
/// [`AcpState::in_flight`] because the per-session forwarder — not the request
/// handler — observes `TurnFinished`, and `session/cancel` needs to mark it.
struct InFlight {
    responder: Responder<PromptResponse>,
    /// Set by `session/cancel` before the interrupt lands; distinguishes a
    /// cancelled stop from a natural end when the turn finishes.
    cancelled: bool,
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
        let response = match stop_reason {
            Some(reason) => Ok(PromptResponse::new(reason)),
            None => Err(AcError::internal_error()
                .data("agent task ended before the turn completed".to_owned())),
        };
        let _ = entry.responder.respond_with_result(response);
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
    /// Process-lifetime token handed to every `Agent::new`; never fired by
    /// this frontend. Turns are interrupted with `InputMessage::Interrupt`,
    /// mirroring how the TUI cancels a turn without killing the run loop.
    app_cancel: CancellationToken,
    sessions: Mutex<HashMap<String, SessionHandle>>,
    prompts: Arc<PromptTracker>,
}

impl AcpState {
    fn cancel_session(&self, session_id: &str) {
        // Mark the pending prompt first so its finishing `TurnFinished`
        // resolves as `Cancelled`, then deliver the native interrupt.
        if !self.prompts.mark_cancelled(session_id) {
            // No prompt in flight: nothing meaningful to cancel.
            return;
        }
        if let Some(handle) = self.sessions.lock().unwrap().get(session_id)
            && handle.input_tx.send(InputMessage::Interrupt).is_err()
        {
            tracing::warn!(session = %session_id, "cancel arrived after the agent stopped");
        }
    }
}

/// Entry point for `harness --acp`: serve one ACP agent over stdio until the
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
            "Copilot is not authenticated yet: run `harness` interactively once to log in \
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
        app_cancel: CancellationToken::new(),
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
    AgentRole
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
                        "harness does not support auth over ACP; run `harness` interactively once \
                     to log in (Copilot) or export the provider API key environment variable"
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
        .context("run ACP connection")?;
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

/// Tool-call id correlation. `AgentEvent` carries no LLM call id, so the
/// adapter mints a UUID per started call and matches the finished event back
/// by name in FIFO order — the agent reports starts and finishes in program
/// order, which holds for both serial and batched dispatch it performs.
#[derive(Default)]
struct ToolCallIds {
    in_flight: Vec<(String, ToolCallId)>,
}

impl ToolCallIds {
    fn start(&mut self, name: &str, summary: &str) -> AcToolCall {
        let id = ToolCallId::new(uuid::Uuid::new_v4().to_string());
        self.in_flight.push((name.to_owned(), id.clone()));
        AcToolCall::new(id, summary).kind(tool_kind(name))
    }

    fn finish(&mut self, name: &str) -> Option<ToolCallId> {
        let index = self.in_flight.iter().position(|(tool, _)| tool == name)?;
        Some(self.in_flight.remove(index).1)
    }
}

/// Translate one agent event into at most one ACP session update. Events
/// without an ACP counterpart (auth UX, retries, compaction notices, …) map
/// to `None`; they stay visible in `HARNESS_LOG`. `context_window == 0`
/// (unknown until the provider reports usage or a model catalog) suppresses
/// usage updates rather than advertise a wrong window size.
fn translate_event(
    event: &AgentEvent,
    ids: &mut ToolCallIds,
    context_window: u64,
) -> Option<SessionUpdate> {
    match event {
        AgentEvent::TextDelta(delta) => Some(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            ContentBlock::from(delta.as_str()),
        ))),
        AgentEvent::ReasoningDelta(delta) => Some(SessionUpdate::AgentThoughtChunk(
            ContentChunk::new(ContentBlock::from(delta.as_str())),
        )),
        AgentEvent::ToolCallStarted { name, summary } => {
            Some(SessionUpdate::ToolCall(ids.start(name, summary)))
        }
        AgentEvent::ToolCallFinished {
            name,
            summary,
            ok,
            duration_ms: _,
            output,
            error,
        } => {
            let tool_call_id = ids
                .finish(name)
                .unwrap_or_else(|| ToolCallId::new(uuid::Uuid::new_v4().to_string()));
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
        AgentEvent::UsageUpdated { .. } if context_window == 0 => None,
        AgentEvent::UsageUpdated {
            input_tokens,
            output_tokens,
            cached_tokens,
            reasoning_tokens: _,
            cost,
        } => Some(SessionUpdate::UsageUpdate(
            UsageUpdate::new(
                input_tokens
                    .saturating_add(*output_tokens)
                    .saturating_sub(*cached_tokens),
                context_window,
            )
            .cost(v1::Cost::new(cost.parse().unwrap_or(0.0), "USD")),
        )),
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

async fn new_session(
    request: NewSessionRequest,
    responder: Responder<NewSessionResponse>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    let (store, tools) = match build_session_stack(&request.cwd, &state.session_root) {
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
    match spawn_agent(&state, store, tools, session, connection, id.clone()) {
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
    let (store, tools) = match build_session_stack(&request.cwd, &state.session_root) {
        Ok(stack) => stack,
        Err(error) => return respond_anyhow(responder, error),
    };
    let id = request.session_id.0.to_string();
    let session = match store.load(&id) {
        Ok(session) => session,
        Err(error) => {
            return respond_anyhow(
                responder,
                anyhow::Error::new(error).context(format!("load session `{id}`")),
            );
        }
    };
    match spawn_agent(&state, store, tools, session, connection, id.clone()) {
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

async fn delete_session(
    request: DeleteSessionRequest,
    responder: Responder<DeleteSessionResponse>,
    state: Arc<AcpState>,
) -> AcResult<()> {
    let id = request.session_id.0.to_string();
    // Dropping the live handle closes the agent's input channel, so the run
    // loop exits after any in-flight turn instead of fighting the deletion.
    let _ = state.sessions.lock().unwrap().remove(&id);
    if let Err(error) = delete_session_everywhere(&id, &state.session_root) {
        return respond_anyhow(responder, error);
    }
    tracing::info!(session = %id, "ACP session deleted");
    let _ = responder.respond(DeleteSessionResponse::new());
    Ok(())
}

/// `session/delete` supplies only an id, not a cwd, so scan every workspace
/// group under the session root. A missing file counts as success.
fn delete_session_everywhere(id: &str, session_root: &std::path::Path) -> Result<()> {
    let parsed = session::SessionId::parse(id).map_err(anyhow::Error::msg)?;
    let file_name = format!("{parsed}.jsonl");
    let Ok(entries) = std::fs::read_dir(session_root) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let workspace_dir = entry.path();
        if !workspace_dir.is_dir() {
            continue;
        }
        let path = workspace_dir.join(&file_name);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("delete `{}`", path.display()))?;
            // Best effort: a stale lock sidecar is worthless afterwards.
            let _ = std::fs::remove_file(path.with_extension("lock"));
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
    let input_tx = {
        let sessions = state.sessions.lock().unwrap();
        sessions
            .get(&session_id)
            .map(|handle| handle.input_tx.clone())
    };
    let Some(input_tx) = input_tx else {
        let _ = responder.respond_with_error(
            AcError::invalid_params().data(format!("unknown session `{session_id}`")),
        );
        return Ok(());
    };

    // Exactly one prompt in flight per session: a concurrent second prompt
    // would interleave two conversations into one history.
    {
        let mut in_flight = state.prompts.in_flight.lock().unwrap();
        if in_flight.contains_key(&session_id) {
            drop(in_flight);
            let _ =
                responder.respond_with_error(AcError::invalid_request().data(
                    "a prompt is already running for this session; cancel it first".to_owned(),
                ));
            return Ok(());
        }
        in_flight.insert(
            session_id.clone(),
            InFlight {
                responder,
                cancelled: false,
            },
        );
    }

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
fn build_session_stack(
    cwd: &std::path::Path,
    session_root: &std::path::Path,
) -> Result<(SessionStore, ToolRegistry)> {
    let workspace_root = std::fs::canonicalize(cwd)
        .with_context(|| format!("resolve session cwd `{}`", cwd.display()))?;
    let store = SessionStore::new(session_root, &workspace_root)?;
    let tools = default_registry(ToolConfig::new(&workspace_root, false))?;
    Ok((store, tools))
}

/// Spawn the agent task plus its event forwarder and register the session's
/// input channel under `acp_session_id`. The forwarder owns everything
/// event-shaped: notification translation and prompt-turn resolution.
fn spawn_agent(
    state: &AcpState,
    store: SessionStore,
    tools: ToolRegistry,
    session: session::Session,
    connection: ConnectionTo<agent_client_protocol::Client>,
    acp_session_id: String,
) -> Result<()> {
    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    let project_context = project_context_for(tools.workspace_root(), state.no_context_files);

    // Subagents: same enablement rule as the other frontends. The ACP
    // frontend builds its registry with `rtk = false`; keep that choice
    // consistent for child registries.
    let mut tools = tools;
    if state.config.subagents.max_turns > 0 {
        let runner = crate::subagent::SubagentRunnerImpl::new(
            state.provider.clone(),
            state.config.model.clone(),
            tools.workspace_root().to_path_buf(),
            false,
            project_context.clone(),
            state.config.subagents,
            Some(store.clone()),
            Some(session.id()),
        );
        tools
            .register_subagent(std::sync::Arc::new(runner))
            .context("register subagent tool")?;
    }

    let mut agent = Agent::new(
        state.provider.clone(),
        tools,
        state.config.model.clone(),
        state.app_cancel.clone(),
    )
    .with_compaction(state.config.compaction.clone())
    .with_subagent_limits(crate::agent::SubagentLimits {
        max_concurrent: state.config.subagents.max_concurrent,
    })
    .with_project_context(project_context)
    .with_session(store, session);
    if let Some(auth) = &state.copilot_auth {
        agent = agent.with_copilot_auth(auth.clone());
    }
    tokio::spawn(agent.run(input_rx, event_tx));

    tokio::spawn(forward_events(
        event_rx,
        connection,
        state.prompts.clone(),
        acp_session_id.clone(),
    ));

    state
        .sessions
        .lock()
        .unwrap()
        .insert(acp_session_id, SessionHandle { input_tx });
    Ok(())
}

/// Consume one session's agent events until the agent task exits: translate
/// each into a `session/update` notification and resolve the parked prompt on
/// `TurnFinished`. Deliberately holds no input sender: the only live clones
/// belong to the sessions map, so `session/delete` dropping the handle closes
/// the channel and lets the agent run loop wind down.
async fn forward_events(
    mut event_rx: mpsc::UnboundedReceiver<AgentEvent>,
    connection: ConnectionTo<agent_client_protocol::Client>,
    prompts: Arc<PromptTracker>,
    acp_session_id: String,
) {
    let mut ids = ToolCallIds::default();
    // Context window starts unknown (0); the model catalog refines it.
    let mut context_window = 0u64;
    let mut turn_done = false;
    while let Some(event) = event_rx.recv().await {
        match &event {
            AgentEvent::TurnFinished => {
                let reason = if prompts.is_cancelled(&acp_session_id) {
                    StopReason::Cancelled
                } else {
                    StopReason::EndTurn
                };
                turn_done = true;
                prompts.resolve(&acp_session_id, Some(reason));
            }
            AgentEvent::ModelList { models, .. } => {
                // The provider's catalog carries authoritative context
                // lengths; prefer them over config overrides so usage updates
                // match what the model actually accepts.
                if let Some(window) = models.iter().find_map(|model| model.context_length) {
                    context_window = window;
                }
            }
            _ => {}
        }
        if let Some(update) = translate_event(&event, &mut ids, context_window) {
            let notification =
                SessionNotification::new(SessionId::from(acp_session_id.clone()), update);
            // A send error means the client connection is gone; stop feeding
            // it. The agent notices on its own at the next turn boundary.
            if connection.send_notification(notification).is_err() {
                break;
            }
        }
    }
    // Agent task gone: fail a still-parked prompt so the editor is not left
    // waiting on a turn that will never finish (`turn_done` distinguishes a
    // clean shutdown after a completed turn from a crashed one).
    if !turn_done {
        prompts.resolve(&acp_session_id, None);
    }
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
    fn started_and_finished_tool_calls_correlate_by_name_in_fifo_order() {
        let mut ids = ToolCallIds::default();
        let started_a = match translate_event(
            &AgentEvent::ToolCallStarted {
                name: "read".into(),
                summary: "read a.rs".into(),
            },
            &mut ids,
            0,
        ) {
            Some(SessionUpdate::ToolCall(call)) => call,
            other => panic!("expected tool call, got {other:?}"),
        };
        let started_b = match translate_event(
            &AgentEvent::ToolCallStarted {
                name: "grep".into(),
                summary: "grep foo".into(),
            },
            &mut ids,
            0,
        ) {
            Some(SessionUpdate::ToolCall(call)) => call,
            other => panic!("expected tool call, got {other:?}"),
        };
        assert_eq!(started_a.kind, ToolKind::Read);
        assert_eq!(started_b.kind, ToolKind::Search);

        // Finishing `grep` first must return grep's id (FIFO by name), not
        // read's.
        let finished = translate_event(
            &AgentEvent::ToolCallFinished {
                name: "grep".into(),
                summary: "grep foo".into(),
                ok: true,
                duration_ms: 3,
                output: "match".into(),
                error: None,
            },
            &mut ids,
            0,
        );
        let Some(SessionUpdate::ToolCallUpdate(update)) = finished else {
            panic!("expected tool call update, got {finished:?}");
        };
        assert_eq!(update.tool_call_id, started_b.tool_call_id);
        assert_eq!(update.fields.status, Some(ToolCallStatus::Completed));

        // The remaining in-flight call is read's.
        let finished = translate_event(
            &AgentEvent::ToolCallFinished {
                name: "read".into(),
                summary: "read a.rs".into(),
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
        assert_eq!(update.tool_call_id, started_a.tool_call_id);
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
    }

    #[test]
    fn usage_updates_are_suppressed_until_context_window_is_known() {
        let event = AgentEvent::UsageUpdated {
            input_tokens: 100,
            output_tokens: 20,
            cached_tokens: 10,
            reasoning_tokens: 5,
            cost: "0.5".into(),
        };
        let mut ids = ToolCallIds::default();
        assert!(
            translate_event(&event, &mut ids, 0).is_none(),
            "no window estimate yet: must not advertise a wrong size"
        );
        let update = match translate_event(&event, &mut ids, 128_000) {
            Some(SessionUpdate::UsageUpdate(update)) => update,
            other => panic!("expected usage update, got {other:?}"),
        };
        assert_eq!(update.used, 110); // input + output - cached
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
                reason: crate::agent::CompactionReason::Auto,
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

    #[test]
    fn prompt_tracker_resolves_once_then_ignores() {
        let tracker = Arc::new(PromptTracker {
            in_flight: Mutex::new(HashMap::new()),
        });
        // Nothing parked: resolve and cancel are both no-ops (the parked
        // responder is only constructible inside the SDK, so the tracker is
        // exercised through its bookkeeping API).
        tracker.resolve("s", Some(StopReason::EndTurn));
        assert!(!tracker.mark_cancelled("s"));
        assert!(!tracker.is_cancelled("s"));
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
            &FileConfig::default(),
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
                            session_root,
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

                // The turn was persisted under our session id; loading it again
                // through a second connection round-trips `session/load`.
                let _ = session_id;
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
