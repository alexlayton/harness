use crate::config::{build_provider_with_auth, save_settings};
use crate::prompt::system_prompt_with_workspace_context;
use crate::tools::{Concurrency, SkillEntry, ToolOutput, ToolRegistry, call_summary};
use auth::CopilotAuth;
use compact::{
    CompactionPolicy, SummaryOutcome, estimate_live_tokens, plan_compaction,
    summarize as compact_summarize,
};
use futures_util::stream::{FuturesUnordered, StreamExt};
use llm::{
    CompletionRequest, Content, LlmError, Message, Provider, RetryCallback, Role, StreamEvent,
    ToolCall, truncate_utf8,
};
use session::{
    ExportOptions, Session, SessionCreateOptions, SessionEvent, SessionStore, StoredMessage,
    StoredToolCall, export_jsonl, snapshot_entries, usage_summary,
};
use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

/// Maximum number of times a turn re-streams after a recoverable failure:
/// malformed tool-call arguments, a retryable mid-stream error, or an empty
/// response. A model that keeps failing should eventually give up instead of
/// looping forever.
const MAX_TURN_RECOVERIES: usize = 3;

/// Maximum emergency compactions performed per turn in response to a
/// context-overflow rejection. Each round is an extra summarizer call plus a
/// retried request, so it is bounded separately from `MAX_TURN_RECOVERIES`.
const MAX_OVERFLOW_RECOVERIES: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStarted {
        /// Provider-neutral tool call id from [`llm::ToolCall::id`]; the key
        /// frontends use to correlate start/finish of concurrent calls.
        call_id: String,
        name: String,
        summary: String,
    },
    ToolCallFinished {
        call_id: String,
        name: String,
        summary: String,
        ok: bool,
        duration_ms: u64,
        /// The complete tool result, retained for optional expansion.
        output: String,
        /// The complete error text when the tool failed. The compact renderer
        /// is responsible for showing only a one-line preview.
        error: Option<String>,
    },
    Retrying {
        attempt: u32,
        message: String,
    },
    TurnFinished,
    Error(String),
    Notice(String),
    ModelChanged {
        provider: String,
        model: String,
    },
    ModelList {
        provider: String,
        models: Vec<llm::ModelInfo>,
    },
    SessionChanged {
        id: String,
        title: Option<String>,
        loaded: bool,
    },
    SessionSnapshot {
        entries: Vec<SessionSnapshotEntry>,
    },
    SessionList {
        sessions: Vec<SessionListItem>,
    },
    SessionExported {
        path: String,
    },
    /// The discovered-skill view requested by the TUI's `/skills` command.
    SkillsLoaded {
        skills: Vec<SkillEntry>,
        diagnostics: Vec<String>,
        /// True when no skills were discovered at all.
        empty: bool,
    },
    UsageUpdated {
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        reasoning_tokens: u64,
        cost: String,
    },
    CompactionFinished {
        compacted_through: u64,
        summary_bytes: usize,
        auto: bool,
        reason: CompactionReason,
    },
}

/// Why a compaction ran. Drives the UI wording (auto vs manual vs overflow)
/// and is recorded on the `CompactionFinished` event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionReason {
    /// Pre-turn trigger fired past the threshold.
    Auto,
    /// User invoked `/compact`.
    Manual,
    /// Provider rejected a request for exceeding the context window.
    Overflow,
}

impl CompactionReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
            Self::Overflow => "overflow",
        }
    }
}

impl std::fmt::Display for CompactionReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionSnapshotEntry {
    User {
        text: String,
    },
    Assistant {
        markdown: String,
        reasoning: String,
    },
    Tool {
        name: String,
        summary: String,
        ok: bool,
        duration_ms: u64,
        output: String,
        error: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionListItem {
    pub id: String,
    pub short_id: String,
    pub title: Option<String>,
    pub updated_at: String,
    pub workspace: String,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Why a single turn did not complete normally.  `run` uses this to decide
/// whether to keep draining queued input.
#[derive(Debug, PartialEq, Eq)]
pub enum TurnError {
    /// The application cancellation token fired while the turn was in flight;
    /// the run loop should stop immediately.
    Shutdown,
    /// A durable session event could not be persisted; the turn was aborted.
    Persist(String),
}

/// Durable session state owned by the agent.  The TUI only receives status
/// events; it never reads or writes session files directly.
pub struct AgentSessionState {
    pub store: SessionStore,
    pub session: Session,
}

pub struct Agent {
    pub provider: Arc<dyn Provider>,
    pub tools: ToolRegistry,
    pub model: String,
    /// Kept public as a short-term compatibility path for existing callers.
    /// When a durable session is attached it is rebuilt from the session
    /// events whenever a session is loaded or created.
    pub history: Vec<Message>,
    pub cancel: CancellationToken,
    pub session: Option<AgentSessionState>,
    /// Shared with the Copilot provider so automatic refreshes survive model
    /// switches without rebuilding the agent.
    pub copilot_auth: Option<Arc<CopilotAuth>>,
    /// Input messages received while a turn is running.  They are drained by
    /// `run` before the next turn starts.
    queued: VecDeque<InputMessage>,
    /// Whether the input channel has not yet been observed closed.  Once
    /// closed, the run loop stops selecting on it (a closed `recv` would
    /// otherwise complete immediately and starve the stream polling).
    input_open: bool,
    /// Exact context occupation from the most recent completed request
    /// (`input_tokens + output_tokens` of its `Done` usage). Drives the
    /// pre-turn trigger; `None` before the first request or after a model
    /// switch, when the estimator takes over.
    last_context_tokens: Option<u64>,
    /// Resolved provider context window (config override → model-reported
    /// `context_length` → conservative default).
    context_window: u64,
    /// Token-aware compaction policy.
    compaction: CompactionPolicy,
    /// Rendered project-context block (AGENTS.md / CLAUDE.md), loaded once at
    /// construction and reused every turn via the system prompt. Empty when
    /// no context files apply or injection is disabled. Lives in the system
    /// prompt, so it is immune to compaction.
    project_context: String,
    /// Fan-out bounds for `Parallel` batches (subagents).
    subagent_limits: SubagentLimits,
    /// Shared subagent runner, when the host registered one. Kept so a
    /// successful `/model` switch can retarget future child runs to the new
    /// provider/model (`SubagentRunnerImpl::update_model`).
    subagent_runner: Option<Arc<crate::subagent::SubagentRunnerImpl>>,
}

/// Upper bound on read-only tool calls running at once.  Read tools are
/// cheap, but the shared `FileSearchIndex` already gates its own concurrency
/// and a runaway batch would thrash the disk; 8 keeps latency wins while
/// staying polite.
const MAX_CONCURRENT_READ_ONLY_TOOLS: usize = 8;

/// Upper bound on concurrent `Parallel` tool calls (subagents) and the
/// per-subagent turn budget.  Each parallel slot is an entire nested agent
/// loop, so this is deliberately separate from (and smaller than) the
/// read-only cap.
#[derive(Clone, Copy, Debug)]
pub struct SubagentLimits {
    pub max_concurrent: usize,
}

impl Default for SubagentLimits {
    fn default() -> Self {
        Self { max_concurrent: 4 }
    }
}

/// One scheduling unit of a turn's tool calls.  Read-only calls group into a
/// batch that runs concurrently; adjacent `Parallel` calls of the same
/// fan-out tool form their own concurrent batch; every exclusive call forms
/// a singleton batch, preserving program order around it.  A batch of one is
/// executed exactly like the historical serial path.
pub(crate) struct ToolBatch {
    pub(crate) calls: Vec<ToolCall>,
    /// Which concurrency class this batch runs under; only [`Concurrency::ReadOnly`]
    /// and [`Concurrency::Parallel`] batches ever hold more than one call.
    pub(crate) class: Concurrency,
}

impl ToolBatch {
    /// Whether this batch may launch more than one call at once.
    pub(crate) fn concurrent(&self) -> bool {
        matches!(self.class, Concurrency::ReadOnly | Concurrency::Parallel)
    }
}

/// An in-flight tool execution carrying its slot index so results can be
/// recorded in original call order rather than completion order.  Borrows
/// the registry only for the batch's lifetime.
type ToolRun<'a> = Pin<Box<dyn Future<Output = (usize, ToolOutput, Instant)> + Send + 'a>>;

/// Partition a turn's calls into batches without reordering anything: a
/// maximal run of read-only calls becomes one batch, a maximal run of
/// `Parallel` calls *of one tool* becomes its own batch, and every exclusive
/// call is a singleton.  A read is never hoisted above a write because the
/// model may intend the read to observe that write's effect; parallel
/// fan-out tools are likewise never merged across an intervening call.
pub(crate) fn plan_tool_batches(calls: Vec<ToolCall>, registry: &ToolRegistry) -> Vec<ToolBatch> {
    let mut batches: Vec<ToolBatch> = Vec::new();
    for call in calls {
        let class = registry.concurrency(&call.name, &call.arguments);
        match batches.last_mut() {
            // Read-only calls all share one class, so any maximal run merges.
            Some(batch)
                if batch.class == Concurrency::ReadOnly && class == Concurrency::ReadOnly =>
            {
                batch.calls.push(call);
            }
            // Parallel calls merge only with the same fan-out tool so two
            // different parallelizable tools cannot interleave their slots.
            Some(batch)
                if batch.class == Concurrency::Parallel
                    && class == Concurrency::Parallel
                    && batch.calls.iter().all(|prior| prior.name == call.name) =>
            {
                batch.calls.push(call);
            }
            _ => batches.push(ToolBatch {
                calls: vec![call],
                class,
            }),
        }
    }
    batches
}

impl Agent {
    /// Dispatch a turn's tool calls in program order, running provably
    /// read-only batches concurrently (bounded by
    /// [`MAX_CONCURRENT_READ_ONLY_TOOLS`]) and exclusive calls alone.
    ///
    /// Event contract: `ToolCallStarted` fires only when an execution future
    /// actually launches (never for a merely requested or queued call), and
    /// `ToolCallFinished` fires exactly once per started call as its future
    /// resolves — so live finish order may be completion order. History and
    /// the durable session remain in original call order.
    ///
    /// Cancellation is uniform and exactly-once: the in-flight batch shares
    /// a child token of the turn's `cancel`, so an interrupt kills the whole
    /// batch while leaving the turn token untouched for the next phase.
    /// Calls that completed keep their real events; launched-but-unfinished
    /// calls get one synthetic cancelled finish; calls never launched get a
    /// synthetic start + cancelled finish pair so every frontend sees a
    /// balanced lifecycle; all of them still receive failed tool results in
    /// parent history so provider history stays valid.
    async fn dispatch_tool_batches(
        &mut self,
        tool_calls: Vec<ToolCall>,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
        cancel: &CancellationToken,
    ) -> Result<(), TurnError> {
        for batch in plan_tool_batches(tool_calls, &self.tools) {
            let limit = if batch.concurrent() {
                match batch.class {
                    Concurrency::ReadOnly => MAX_CONCURRENT_READ_ONLY_TOOLS,
                    Concurrency::Parallel => self.subagent_limits.max_concurrent,
                    Concurrency::Exclusive => 1,
                }
            } else {
                1
            };
            if batch.calls.len() > 1 {
                tracing::debug!(
                    count = batch.calls.len(),
                    class = ?batch.class,
                    launch_limit = limit,
                    "running concurrent tool batch"
                );
            }
            // The batch shares a child token so an interrupt kills exactly
            // this phase; the turn token stays clean for the next batch.
            let batch_cancel = cancel.child_token();

            let mut futures: FuturesUnordered<ToolRun<'_>> = FuturesUnordered::new();
            let mut next_launch = 0usize;
            let mut starts: Vec<Option<Instant>> = (0..batch.calls.len()).map(|_| None).collect();
            let registry = &self.tools;
            while next_launch < batch.calls.len().min(limit) {
                let (future, started) = launch_call(
                    registry,
                    &batch.calls[next_launch],
                    next_launch,
                    batch_cancel.clone(),
                    events,
                );
                starts[next_launch] = Some(started);
                futures.push(future);
                next_launch += 1;
            }

            let mut slots: Vec<Option<(ToolOutput, Instant)>> =
                (0..batch.calls.len()).map(|_| None).collect();
            let mut finished = 0usize;
            loop {
                tokio::select! {
                    item = futures.next() => match item {
                        Some((index, result, started)) => {
                            // Live finish event: emitted in completion order,
                            // keyed by call id. Durable results below stay in
                            // original call order regardless.
                            send_finished(events, &batch.calls[index], &result, started);
                            slots[index] = Some((result, started));
                            finished += 1;
                            if finished == slots.len() {
                                break;
                            }
                            // Refill the freed slot from the remaining
                            // calls until the batch is exhausted.
                            if next_launch < batch.calls.len() {
                                let registry = &self.tools;
                                let (future, started) = launch_call(
                                    registry,
                                    &batch.calls[next_launch],
                                    next_launch,
                                    batch_cancel.clone(),
                                    events,
                                );
                                starts[next_launch] = Some(started);
                                futures.push(future);
                                next_launch += 1;
                            }
                        }
                        None => break,
                    },
                    message = input.recv(), if self.input_open => match message {
                        Some(InputMessage::Interrupt) => {
                            batch_cancel.cancel();
                            // Drop the in-flight futures (cancelling them)
                            // and let unfilled slots synthesize "cancelled"
                            // results below, mirroring the historical
                            // single-call interrupt behavior.
                            break;
                        }
                        Some(message) => self.queued.push_back(message),
                        None => self.input_open = false,
                    },
                    _ = cancel.cancelled() => {
                        batch_cancel.cancel();
                        break;
                    }
                }
            }
            drop(futures);

            // Drain whatever completed, then synthesize "cancelled" results
            // for the rest — mirroring the historical single-call behavior —
            // and end the turn. Finished slots already had their finish event
            // sent at resolution time, so they must not get a second one;
            // every call still gets its history entry and durable result.
            let mut cancelled = false;
            for (index, call) in batch.calls.iter().enumerate() {
                let mut was_cancelled = false;
                let (result, started) = match slots[index].take() {
                    Some((result, started)) => (result, started),
                    None => {
                        was_cancelled = true;
                        cancelled = true;
                        // A call never launched because the batch was
                        // interrupted still needs a balanced UI lifecycle:
                        // synthetic start now, cancelled finish below.
                        // Launched-but-unfinished calls already had their
                        // start; all of them get one cancelled finish here
                        // plus a failed durable result either way, so both
                        // frontends and provider history stay valid.
                        let started = match starts[index] {
                            Some(started) => started,
                            None => {
                                send_started(events, call);
                                Instant::now()
                            }
                        };
                        // Synthesize the same "cancelled" tool result the
                        // serial path produced, so provider history stays
                        // valid across the refactor.
                        (
                            ToolOutput {
                                content: "cancelled".to_owned(),
                                is_error: true,
                                summary: call_summary(&call.name, &call.arguments),
                            },
                            started,
                        )
                    }
                };
                if was_cancelled {
                    // Historical interrupt events carried an empty output
                    // field with the error carrying the reason.
                    send(
                        events,
                        AgentEvent::ToolCallFinished {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            summary: result.summary.clone(),
                            ok: false,
                            duration_ms: started.elapsed().as_millis() as u64,
                            output: String::new(),
                            error: Some("cancelled".to_owned()),
                        },
                    );
                }
                self.history.push(Message {
                    role: Role::Tool,
                    content: vec![Content::ToolResult {
                        tool_call_id: call.id.clone(),
                        content: result.content.clone(),
                        is_error: result.is_error,
                    }],
                });
                let _ = self.persist_tool_result(call, &result.content, result.is_error, events);
            }
            if cancelled {
                self.persist_cancelled("tool execution interrupted", events);
                send(events, AgentEvent::TurnFinished);
                if self.cancel.is_cancelled() {
                    return Err(TurnError::Shutdown);
                }
                return Ok(());
            }
        }
        Ok(())
    }
}

/// Launch one call's execution future: announce it (`ToolCallStarted` with
/// its original call id) immediately before pushing, stamping the true start
/// instant used later for the duration in `ToolCallFinished`. Each future
/// carries its own slot index; results land in that slot, never in
/// completion order.
fn launch_call<'a>(
    registry: &'a ToolRegistry,
    call: &'a ToolCall,
    index: usize,
    cancel: CancellationToken,
    events: &mpsc::UnboundedSender<AgentEvent>,
) -> (ToolRun<'a>, Instant) {
    send_started(events, call);
    let started = Instant::now();
    let name = call.name.clone();
    let arguments = call.arguments.clone();
    (
        Box::pin(async move {
            let result = registry.execute(&name, arguments, cancel).await;
            (index, result, started)
        }),
        started,
    )
}

fn send_started(events: &mpsc::UnboundedSender<AgentEvent>, call: &ToolCall) {
    send(
        events,
        AgentEvent::ToolCallStarted {
            call_id: call.id.clone(),
            name: call.name.clone(),
            summary: call_summary(&call.name, &call.arguments),
        },
    );
}

fn send_finished(
    events: &mpsc::UnboundedSender<AgentEvent>,
    call: &ToolCall,
    result: &ToolOutput,
    started: Instant,
) {
    send(
        events,
        AgentEvent::ToolCallFinished {
            call_id: call.id.clone(),
            name: call.name.clone(),
            summary: result.summary.clone(),
            ok: !result.is_error,
            duration_ms: started.elapsed().as_millis() as u64,
            output: result.content.clone(),
            error: result.is_error.then(|| result.content.clone()),
        },
    );
}

impl Agent {
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        model: impl Into<String>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            provider,
            tools,
            model: model.into(),
            history: Vec::new(),
            cancel,
            session: None,
            copilot_auth: None,
            queued: VecDeque::new(),
            input_open: true,
            last_context_tokens: None,
            context_window: 0,
            compaction: CompactionPolicy::default(),
            project_context: String::new(),
            subagent_limits: SubagentLimits::default(),
            subagent_runner: None,
        }
    }

    /// Configure fan-out bounds for `Parallel` batches (subagents).
    /// `max_concurrent` is clamped to at least 1: this type is public and a
    /// zero limit would otherwise launch no futures and synthesize
    /// cancellation for the whole batch.
    pub fn with_subagent_limits(mut self, limits: SubagentLimits) -> Self {
        self.subagent_limits.max_concurrent = limits.max_concurrent.max(1);
        self
    }

    /// Attach the shared subagent runner. The agent does not run children
    /// itself; it only forwards successful provider/model switches so future
    /// delegations use the active selection.
    pub fn with_subagent_runner(
        mut self,
        runner: Arc<crate::subagent::SubagentRunnerImpl>,
    ) -> Self {
        if let Some(state) = self.session.as_ref() {
            runner.update_parent_session(Some(state.session.id()));
        }
        self.subagent_runner = Some(runner);
        self
    }

    /// Attach a pre-rendered project-context block (AGENTS.md / CLAUDE.md).
    /// Pass an empty string to skip injection.
    pub fn with_project_context(mut self, project_context: impl Into<String>) -> Self {
        self.project_context = project_context.into();
        self
    }

    /// Attach a compaction policy (resolved from `config.toml`).
    pub fn with_compaction(mut self, policy: CompactionPolicy) -> Self {
        self.compaction = policy;
        self
    }

    pub fn with_copilot_auth(mut self, auth: Arc<CopilotAuth>) -> Self {
        self.copilot_auth = Some(auth);
        self
    }

    /// Attach a loaded/new durable session.  Active provider/model selection
    /// remains the caller's choice; saved metadata is informational only.
    pub fn with_session(mut self, store: SessionStore, mut session: Session) -> Self {
        if let Err(error) = store.repair_incomplete_tool_calls(&mut session) {
            tracing::warn!(error = %error, "could not repair incomplete session tool calls");
        }
        self.history = session.context_messages();
        if let Some(runner) = &self.subagent_runner {
            runner.update_parent_session(Some(session.id()));
        }
        self.session = Some(AgentSessionState { store, session });
        self
    }

    fn persist_event(
        &mut self,
        event: SessionEvent,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> bool {
        let Some(state) = self.session.as_mut() else {
            return true;
        };
        match state.store.append_event(&mut state.session, event) {
            Ok(_) => true,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("session persistence failed: {error}")),
                );
                false
            }
        }
    }

    /// Durable flush for deferred-sync stores (no-op otherwise). Called at
    /// turn boundaries; failures are logged, not surfaced — the data is in
    /// the OS page cache and the next boundary retries.
    fn flush_deferred_sync(&self) {
        let Some(state) = self.session.as_ref() else {
            return;
        };
        if !state.store.deferred_sync() {
            return;
        }
        if let Err(error) = state.store.sync_session(&state.session) {
            tracing::warn!(error = %error, "deferred session sync failed");
        }
    }

    fn persist_user_message(
        &mut self,
        message: &Message,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> bool {
        self.persist_event(
            SessionEvent::UserMessage {
                message: StoredMessage::from_llm(message),
            },
            events,
        )
    }

    fn persist_assistant(
        &mut self,
        reasoning: &str,
        text: &str,
        opaque: &[(String, serde_json::Value)],
        calls: &[ToolCall],
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> bool {
        if reasoning.is_empty() && text.is_empty() && opaque.is_empty() && calls.is_empty() {
            return true;
        }
        let mut content = Vec::new();
        if !reasoning.is_empty() {
            content.push(Content::Reasoning(reasoning.to_owned()));
        }
        if !text.is_empty() {
            content.push(Content::Text(text.to_owned()));
        }
        content.extend(opaque.iter().map(|(provider, data)| Content::Opaque {
            provider: provider.clone(),
            data: data.clone(),
        }));
        // Tool calls have their own durable events.  Keeping them out of this
        // message avoids duplicate calls while still making the event stream
        // explicit and easy to inspect/export.
        let message = Message::assistant(content);
        if !message.content.is_empty()
            && !self.persist_event(
                SessionEvent::AssistantMessage {
                    message: StoredMessage::from_llm(&message),
                },
                events,
            )
        {
            return false;
        }
        for call in calls {
            if !self.persist_event(
                SessionEvent::ToolCall {
                    call: StoredToolCall::from(call),
                },
                events,
            ) {
                return false;
            }
        }
        true
    }

    fn persist_tool_result(
        &mut self,
        call: &ToolCall,
        content: &str,
        is_error: bool,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> bool {
        self.persist_event(
            SessionEvent::ToolResult {
                tool_call_id: call.id.clone(),
                content: content.to_owned(),
                is_error,
                tool_name: Some(call.name.clone()),
            },
            events,
        )
    }

    fn persist_cancelled(
        &mut self,
        reason: impl Into<String>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let _ = self.persist_event(
            SessionEvent::TurnCancelled {
                reason: reason.into(),
            },
            events,
        );
    }

    /// Run until the input channel closes or the application cancellation
    /// token is cancelled.  Input submitted while a turn is running remains in
    /// the mpsc queue and is consumed after the current turn finishes.
    pub async fn run(
        mut self,
        mut input: mpsc::UnboundedReceiver<InputMessage>,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) {
        if let Some(session) = self.session.as_ref() {
            send(
                &events,
                AgentEvent::SessionChanged {
                    id: session.session.id().to_string(),
                    title: session.session.metadata.title.clone(),
                    loaded: false,
                },
            );
            send(
                &events,
                AgentEvent::SessionSnapshot {
                    entries: ui_snapshot_entries(snapshot_entries(&session.session)),
                },
            );
            send(&events, usage_event(&session.session.metadata.usage));
        }

        // Resolve the provider context window before the first turn so the
        // pre-turn trigger has a baseline. Kept after the initial session
        // events so the UI is not blocked on a model-list fetch; failed
        // fetches fall back to the config override / conservative default.
        self.refresh_context_window().await;

        loop {
            let next_message = if let Some(message) = self.queued.pop_front() {
                Some(message)
            } else if !self.input_open {
                None
            } else {
                tokio::select! {
                    message = input.recv() => {
                        if message.is_none() {
                            self.input_open = false;
                        }
                        message
                    }
                    _ = self.cancel.cancelled() => None,
                }
            };
            let Some(message) = next_message else {
                break;
            };
            match message {
                InputMessage::Message(text) if !text.trim().is_empty() => {
                    let turn_cancel = CancellationToken::new();
                    let result = self.run_turn(text, &events, &mut input, &turn_cancel).await;
                    // Turn-boundary durable flush: with deferred sync enabled
                    // the events of this turn were written+flushed but not
                    // fsynced; make them all durable once here instead of
                    // paying an fsync per streamed event.
                    self.flush_deferred_sync();
                    match result {
                        Err(TurnError::Shutdown) => break,
                        Err(TurnError::Persist(_)) | Ok(()) => {}
                    }
                }
                InputMessage::Message(_) | InputMessage::Interrupt => continue,
                InputMessage::NewConversation => {
                    self.handle_new_session(&events);
                    continue;
                }
                InputMessage::LoadSession { selector } => {
                    self.handle_load_session(selector, &events);
                    continue;
                }
                InputMessage::ListSessions => {
                    self.handle_list_sessions(&events);
                    continue;
                }
                InputMessage::ExportSession { destination } => {
                    self.handle_export_session(destination, &events);
                    continue;
                }
                InputMessage::CompactSession => {
                    let cancel = self.cancel.clone();
                    self.handle_compact_session(&events, &cancel).await;
                    continue;
                }
                InputMessage::SetModel { provider, model } => {
                    self.handle_set_model(provider, model, &events).await;
                    continue;
                }
                InputMessage::ListModels { provider } => {
                    self.handle_list_models(provider, &events);
                    continue;
                }
                InputMessage::ListSkills => {
                    self.handle_list_skills(&events);
                    continue;
                }
                InputMessage::InvokeSkill { name } => {
                    self.handle_invoke_skill(name, &events, &mut input).await;
                    continue;
                }
            };
        }
    }

    /// Run one user turn: persist the message, stream the provider response,
    /// execute any tool calls, and persist every durable event.  Returns
    /// [`TurnError::Shutdown`] only when the application cancellation token
    /// fired mid-turn, so `run` can stop immediately.
    #[tracing::instrument(
        name = "turn",
        skip(self, events, input, cancel),
        fields(user_text = %truncate_utf8(&user_text, 200))
    )]
    async fn run_turn(
        &mut self,
        user_text: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
        cancel: &CancellationToken,
    ) -> Result<(), TurnError> {
        // Pre-turn auto-compaction trigger: run *before* the request is built
        // (never mid-stream), so provider-history validity is trivial. Exact
        // context from the last request when available, plus the new message
        // this turn is about to add.
        if self.should_auto_compact(&user_text) {
            let context = self.context_tokens_estimate(user_text.len());
            let percent = if self.context_window > 0 {
                ((context as f64 / self.context_window as f64) * 100.0) as u32
            } else {
                0
            };
            if self
                .compact_and_reload(events, cancel, CompactionReason::Auto)
                .await
            {
                send(
                    events,
                    AgentEvent::Notice(format!("auto-compacted: context at {percent}% of window")),
                );
            }
        }

        let user_message = Message::user(user_text);
        self.history.push(user_message.clone());
        if !self.persist_user_message(&user_message, events) {
            send(events, AgentEvent::TurnFinished);
            return Err(TurnError::Persist("user message".into()));
        }
        let mut recoveries = 0;
        let mut overflow_recoveries = 0;
        loop {
            let request = CompletionRequest {
                model: self.model.clone(),
                system: Some(system_prompt_with_workspace_context(
                    &self.tools.workspace_root().display().to_string(),
                    &self.tools.prompt_context(),
                    self.tools.skills(),
                    &self.project_context,
                )),
                messages: self.history.clone(),
                tools: self.tools.definitions(),
                max_tokens: None,
                temperature: None,
                reasoning: true,
            };

            let retry_events = events.clone();
            let on_retry: RetryCallback = Arc::new(move |attempt, error| {
                let _ = retry_events.send(AgentEvent::Retrying {
                    attempt,
                    message: error.to_string(),
                });
            });
            // Clone the provider handle before constructing the future so
            // the future does not hold an immutable borrow of `self`
            // while durable events are appended below.
            let provider = self.provider.clone();
            let provider_future = provider.stream_with_retry(&request, on_retry);
            tokio::pin!(provider_future);
            let stream_result = loop {
                tokio::select! {
                    result = &mut provider_future => break Some(result),
                    message = input.recv(), if self.input_open => match message {
                        Some(InputMessage::Interrupt) => {
                            cancel.cancel();
                            break None;
                        }
                        Some(message) => self.queued.push_back(message),
                        None => self.input_open = false,
                    },
                    _ = cancel.cancelled() => break None,
                    _ = self.cancel.cancelled() => {
                        self.persist_cancelled("application shutdown", events);
                        send(events, AgentEvent::TurnFinished);
                        return Err(TurnError::Shutdown);
                    }
                }
            };
            let Some(stream_result) = stream_result else {
                self.persist_cancelled("turn interrupted before response", events);
                send(events, AgentEvent::TurnFinished);
                return Ok(());
            };
            let mut stream = match stream_result {
                Ok(stream) => stream,
                Err(error) => {
                    // A tool-heavy turn can grow past the window mid-turn: the
                    // *next* request is rejected with a context-exceeded 400.
                    // Compact the older material (keeping this turn's tail) and
                    // retry before surfacing the provider error.
                    if self
                        .try_overflow_recovery(&error, events, cancel, &mut overflow_recoveries)
                        .await
                    {
                        continue;
                    }
                    let message = error.to_string();
                    let _ = self.persist_event(
                        SessionEvent::Error {
                            message: message.clone(),
                        },
                        events,
                    );
                    send(events, AgentEvent::Error(message));
                    send(events, AgentEvent::TurnFinished);
                    return Ok(());
                }
            };

            let mut text = String::new();
            let mut reasoning = String::new();
            let mut tool_calls = Vec::<ToolCall>::new();
            let mut opaque = Vec::<(String, serde_json::Value)>::new();
            let mut cancelled = false;
            let mut stream_error = None;

            loop {
                tokio::select! {
                    next = stream.next() => {
                        let Some(next) = next else {
                            break;
                        };
                        match next {
                            Ok(StreamEvent::TextDelta(delta)) => {
                                text.push_str(&delta);
                                send(events, AgentEvent::TextDelta(delta));
                            }
                            Ok(StreamEvent::ReasoningDelta(delta)) => {
                                reasoning.push_str(&delta);
                                send(events, AgentEvent::ReasoningDelta(delta));
                            }
                            Ok(StreamEvent::OpaqueState { provider, data }) => opaque.push((provider, data)),
                            Ok(StreamEvent::ToolCallComplete(call)) => tool_calls.push(call),
                            Ok(StreamEvent::Done { usage: done_usage, .. }) => {
                                if let Some(done_usage) = done_usage {
                                    // Exact context occupancy of the request
                                    // just completed; the next request starts
                                    // from (approximately) this size, so it
                                    // drives the pre-turn trigger.
                                    self.last_context_tokens = Some(
                                        done_usage
                                            .input_tokens
                                            .saturating_add(done_usage.output_tokens),
                                    );
                                    let summary = usage_summary(&done_usage);
                                    let _ = self.persist_event(
                                        SessionEvent::Usage {
                                            usage: summary.clone(),
                                        },
                                        events,
                                    );
                                    if let Some(state) = self.session.as_ref() {
                                        send(events, usage_event(&state.session.metadata.usage));
                                    } else {
                                        send(events, usage_event(&summary));
                                    }
                                }
                            }
                            Err(error) => {
                                stream_error = Some(error);
                                break;
                            }
                        }
                    }
                    message = input.recv(), if self.input_open => match message {
                        Some(InputMessage::Interrupt) => {
                            cancel.cancel();
                            cancelled = true;
                            break;
                        }
                        Some(message) => self.queued.push_back(message),
                        None => self.input_open = false,
                    },
                    _ = cancel.cancelled() => {
                        cancelled = true;
                        break;
                    }
                    _ = self.cancel.cancelled() => {
                        cancelled = true;
                        break;
                    }
                }
            }

            if cancelled {
                append_assistant(
                    &mut self.history,
                    &reasoning,
                    &text,
                    &opaque,
                    tool_calls.clone(),
                );
                let _ = self.persist_assistant(&reasoning, &text, &opaque, &tool_calls, events);
                for call in &tool_calls {
                    let cancelled_result = "cancelled before tool execution";
                    self.history.push(Message::tool_result(
                        call.id.clone(),
                        cancelled_result,
                        true,
                    ));
                    let _ = self.persist_tool_result(call, cancelled_result, true, events);
                }
                self.persist_cancelled("turn interrupted", events);
                send(events, AgentEvent::TurnFinished);
                if self.cancel.is_cancelled() {
                    return Err(TurnError::Shutdown);
                }
                return Ok(());
            }

            append_assistant(
                &mut self.history,
                &reasoning,
                &text,
                &opaque,
                tool_calls.clone(),
            );
            let _ = self.persist_assistant(&reasoning, &text, &opaque, &tool_calls, events);

            if let Some(error) = stream_error {
                // A mid-stream context overflow (provider tears down an SSE
                // request that outgrew the window) can also be recovered by
                // compacting and re-streaming.
                if self
                    .try_overflow_recovery(&error, events, cancel, &mut overflow_recoveries)
                    .await
                {
                    // Mark this turn's calls as failed so the retry request
                    // sees consistent history (mirrors the error path below).
                    let message = error.to_string();
                    for call in &tool_calls {
                        let error_result = format!("provider stream interrupted: {message}");
                        self.history.push(Message::tool_result(
                            call.id.clone(),
                            error_result.clone(),
                            true,
                        ));
                        let _ = self.persist_tool_result(call, &error_result, true, events);
                    }
                    continue;
                }
                let message = error.to_string();
                let _ = self.persist_event(
                    SessionEvent::Error {
                        message: message.clone(),
                    },
                    events,
                );
                send(events, AgentEvent::Error(message.clone()));
                // Any calls already streamed before the failure must not
                // dangle; mark them failed so the next request (retry or next
                // turn) sees a consistent history.
                for call in &tool_calls {
                    let error_result = format!("provider stream interrupted: {message}");
                    self.history.push(Message::tool_result(
                        call.id.clone(),
                        error_result.clone(),
                        true,
                    ));
                    let _ = self.persist_tool_result(call, &error_result, true, events);
                }
                let mut retried = false;
                if let LlmError::Parse(parse_message) = &error {
                    // A tool call whose arguments failed to parse (usually
                    // truncated JSON) never became durable, so the turn can
                    // retry without leaving dangling state. Nudge the model
                    // and re-stream instead of dead-ending the turn.
                    if recoveries < MAX_TURN_RECOVERIES {
                        recoveries += 1;
                        push_recovery_note(
                            &mut self.history,
                            format!(
                                "[system note: your previous tool call had malformed JSON \
                                 arguments and was not executed: {parse_message}. Re-issue the \
                                 tool call with valid arguments.]"
                            ),
                        );
                        retried = true;
                    }
                } else if error.is_retryable() {
                    // Transient mid-stream failures (connection drops, decode
                    // errors) are worth an automatic re-stream: the partial
                    // content is already persisted, and the model can continue
                    // from where it left off.
                    if recoveries < MAX_TURN_RECOVERIES {
                        recoveries += 1;
                        push_recovery_note(
                            &mut self.history,
                            format!(
                                "[system note: your response stream was interrupted \
                                 ({message}); continue from where you left off.]"
                            ),
                        );
                        retried = true;
                    }
                }
                if retried {
                    continue;
                }
                send(events, AgentEvent::TurnFinished);
                return Ok(());
            }

            if tool_calls.is_empty() {
                // A turn that produced no text and no tool calls is almost
                // always a provider stall (reasoning emitted, then nothing).
                // Nudge once instead of silently ending the turn.
                if text.trim().is_empty() && recoveries < MAX_TURN_RECOVERIES {
                    recoveries += 1;
                    push_recovery_note(
                        &mut self.history,
                        "[system note: your previous response produced no output; continue \
                         and complete the task.]"
                            .into(),
                    );
                    continue;
                }
                send(events, AgentEvent::TurnFinished);
                return Ok(());
            }

            self.dispatch_tool_batches(tool_calls, events, input, cancel)
                .await?;
        }
    }

    fn handle_new_session(&mut self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(store) = self.session.as_ref().map(|state| state.store.clone()) else {
            self.history.clear();
            send(
                events,
                AgentEvent::Notice("Started a new conversation".into()),
            );
            return;
        };
        let session = match store.create(SessionCreateOptions {
            provider: Some(self.provider.name().to_owned()),
            model: Some(self.model.clone()),
            ..SessionCreateOptions::default()
        }) {
            Ok(session) => session,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not create session: {error}")),
                );
                return;
            }
        };
        let id = session.id().to_string();
        let parent_session_id = session.id();
        let title = session.metadata.title.clone();
        self.history.clear();
        self.last_context_tokens = None;
        self.session = Some(crate::agent::AgentSessionState { store, session });
        if let Some(runner) = &self.subagent_runner {
            runner.update_parent_session(Some(parent_session_id));
        }
        send(
            events,
            AgentEvent::SessionChanged {
                id,
                title,
                loaded: false,
            },
        );
        send(
            events,
            AgentEvent::SessionSnapshot {
                entries: Vec::new(),
            },
        );
        send(
            events,
            AgentEvent::Notice("Started a new conversation".into()),
        );
    }

    fn handle_load_session(
        &mut self,
        selector: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let Some(store) = self.session.as_ref().map(|state| state.store.clone()) else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        let mut session = match store.load(&selector) {
            Ok(session) => session,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not load session: {error}")),
                );
                return;
            }
        };
        if !session
            .file_path()
            .is_some_and(|path| store.is_path_in_store(path))
        {
            match store.adopt(&session) {
                Ok(adopted) => session = adopted,
                Err(error) => {
                    send(
                        events,
                        AgentEvent::Error(format!("could not adopt loaded session: {error}")),
                    );
                    return;
                }
            }
        }
        if let Err(error) = store.repair_incomplete_tool_calls(&mut session) {
            send(
                events,
                AgentEvent::Error(format!("could not repair loaded session: {error}")),
            );
            return;
        }
        let id = session.id().to_string();
        let parent_session_id = session.id();
        let title = session.metadata.title.clone();
        self.history = session.context_messages();
        self.last_context_tokens = None;
        let snapshot = ui_snapshot_entries(snapshot_entries(&session));
        self.session = Some(crate::agent::AgentSessionState { store, session });
        if let Some(runner) = &self.subagent_runner {
            runner.update_parent_session(Some(parent_session_id));
        }
        send(
            events,
            AgentEvent::SessionChanged {
                id,
                title,
                loaded: true,
            },
        );
        send(events, AgentEvent::SessionSnapshot { entries: snapshot });
        if let Some(state) = self.session.as_ref() {
            send(events, usage_event(&state.session.metadata.usage));
        }
        send(
            events,
            AgentEvent::Notice(format!(
                "Loaded session; active model remains {} · {}",
                self.provider.name(),
                self.model
            )),
        );
    }

    fn handle_list_sessions(&self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(store) = self.session.as_ref().map(|state| state.store.clone()) else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        match store.list() {
            Ok(entries) => send(
                events,
                AgentEvent::SessionList {
                    sessions: entries
                        .into_iter()
                        .map(|entry| SessionListItem {
                            id: entry.id.to_string(),
                            short_id: entry.short_id,
                            title: entry.title,
                            updated_at: entry.updated_at,
                            workspace: entry.workspace_root.display().to_string(),
                            provider: entry.provider,
                            model: entry.model,
                        })
                        .collect(),
                },
            ),
            Err(error) => send(
                events,
                AgentEvent::Error(format!("could not list sessions: {error}")),
            ),
        }
    }

    fn handle_export_session(
        &self,
        destination: Option<String>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let Some(session) = self.session.as_ref().map(|state| state.session.clone()) else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        let destination = destination.map(PathBuf::from);
        match export_jsonl(&session, destination.as_deref(), &ExportOptions::default()) {
            Ok(path) => {
                let path = path.display().to_string();
                send(events, AgentEvent::SessionExported { path: path.clone() });
                send(
                    events,
                    AgentEvent::Notice(format!("Exported session to {path}")),
                );
            }
            Err(error) => send(
                events,
                AgentEvent::Error(format!("could not export session: {error}")),
            ),
        }
    }

    async fn handle_compact_session(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) {
        if self.session.is_none() {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        }
        let _ = self
            .compact_and_reload(events, cancel, CompactionReason::Manual)
            .await;
    }

    async fn handle_set_model(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let needs_auth = provider
            .as_deref()
            .and_then(crate::config::ProviderArg::from_name)
            == Some(crate::config::ProviderArg::GithubCopilot)
            || self.provider.name() == "github-copilot";
        let mut auth = self.copilot_auth.clone();
        if needs_auth && auth.is_none() {
            match CopilotAuth::from_default() {
                Ok(value) => {
                    let value = Arc::new(value);
                    self.copilot_auth = Some(value.clone());
                    auth = Some(value);
                }
                Err(error) => {
                    send(events, AgentEvent::Error(error.to_string()));
                    return;
                }
            }
        }
        self.handle_set_model_with_factory(
            provider,
            model,
            events,
            Box::new(move |name| {
                let copilot_auth = if crate::config::ProviderArg::from_name(name)
                    == Some(crate::config::ProviderArg::GithubCopilot)
                {
                    auth.clone()
                } else {
                    None
                };
                build_provider_with_auth(name, copilot_auth)
            }),
        );
        // A different model may have a different context window and stale
        // token counts; reset both so the next trigger re-baselines.
        self.last_context_tokens = None;
        self.refresh_context_window().await;
    }

    fn handle_set_model_with_factory(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        factory: ProviderFactory,
    ) {
        let explicit_provider = provider.is_some();
        let requested = provider.unwrap_or_else(|| self.provider.name().to_owned());
        let known_provider = crate::config::ProviderArg::ALL
            .iter()
            .find(|known| known.to_string().eq_ignore_ascii_case(&requested));
        let canonical = known_provider
            .map(ToString::to_string)
            .unwrap_or_else(|| requested.clone());
        let current = self.provider.name().to_owned();
        let provider_changed =
            (explicit_provider && known_provider.is_none()) || current != canonical;
        let next_provider = if provider_changed {
            match factory(&canonical) {
                Ok(provider) => Some(provider),
                Err(error) => {
                    send(events, AgentEvent::Error(error.to_string()));
                    return;
                }
            }
        } else {
            None
        };

        if let Some(provider) = next_provider {
            self.provider = provider;
        }
        self.model = model.clone();
        // Future subagents must follow the parent's active selection; a
        // failed switch already returned above, so children never see a
        // half-applied state. Running children keep their own snapshot.
        if let Some(runner) = &self.subagent_runner {
            runner.update_model(self.provider.clone(), self.model.clone());
        }
        let _ = self.persist_event(
            SessionEvent::ModelChange {
                provider: canonical.clone(),
                model: model.clone(),
            },
            events,
        );
        if let Err(error) = save_settings(&canonical, &model) {
            tracing::warn!(error = %error, "could not persist model settings");
        }
        send(
            events,
            AgentEvent::ModelChanged {
                provider: canonical.clone(),
                model: model.clone(),
            },
        );
        send(
            events,
            AgentEvent::Notice(format!("Using {canonical} · {model}")),
        );
        spawn_model_list(canonical, self.provider.clone(), events.clone());
    }

    fn handle_list_models(&self, provider: String, events: &mpsc::UnboundedSender<AgentEvent>) {
        let provider_name = crate::config::ProviderArg::ALL
            .iter()
            .find(|known| known.to_string().eq_ignore_ascii_case(&provider))
            .map(ToString::to_string)
            .unwrap_or(provider.clone());
        let auth = if provider_name == "github-copilot" {
            self.copilot_auth.clone()
        } else {
            None
        };
        let provider = match build_provider_with_auth(&provider_name, auth) {
            Ok(provider) => provider,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Notice(format!("could not fetch model list: {error}")),
                );
                return;
            }
        };
        spawn_model_list(provider_name, provider, events.clone());
    }

    /// Reply to `/skills` with the discovered-skill view: invocable skills
    /// first, then discovery diagnostics so broken skills are visible to the
    /// user (they never reach the model prompt).
    fn handle_list_skills(&self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(catalog) = self.tools.skills() else {
            send(
                events,
                AgentEvent::SkillsLoaded {
                    skills: Vec::new(),
                    diagnostics: Vec::new(),
                    empty: true,
                },
            );
            return;
        };
        let empty = catalog.is_empty();
        send(
            events,
            AgentEvent::SkillsLoaded {
                skills: catalog
                    .skills
                    .iter()
                    .map(|skill| SkillEntry {
                        name: skill.name.clone(),
                        description: skill.description.clone(),
                    })
                    .collect(),
                diagnostics: catalog
                    .diagnostics
                    .iter()
                    .map(|diagnostic| match &diagnostic.path {
                        Some(path) => format!("{}: {}", path.display(), diagnostic.message),
                        None => diagnostic.message.clone(),
                    })
                    .collect(),
                empty,
            },
        );
    }

    /// Start a turn from a skill's instructions: the `SKILL.md` body without
    /// frontmatter, prefixed with a line naming the skill so both the model
    /// and the session transcript show what was invoked.
    async fn handle_invoke_skill(
        &mut self,
        name: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
    ) {
        let found = self.tools.skills().and_then(|catalog| {
            catalog
                .invocable()
                .into_iter()
                .find(|skill| skill.name.eq_ignore_ascii_case(&name))
                .map(|skill| (skill.file_path.clone(), skill.name.clone()))
        });
        let Some((file_path, name)) = found else {
            send(events, AgentEvent::Error(format!("unknown skill: {name}")));
            return;
        };
        let raw = match std::fs::read_to_string(&file_path) {
            Ok(raw) => raw,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not read {name}: {error}")),
                );
                return;
            }
        };
        let (_, body) = tools::parse_frontmatter(&raw);
        let body = body.trim();
        if body.is_empty() {
            send(events, AgentEvent::Error(format!("skill {name} is empty")));
            return;
        }
        let turn_cancel = CancellationToken::new();
        let result = self
            .run_turn(format!("/{name}\n\n{body}"), events, input, &turn_cancel)
            .await;
        match result {
            Err(TurnError::Shutdown) => {}
            Err(TurnError::Persist(_)) | Ok(()) => {}
        }
    }

    // ------------------------------------------------------------------ compaction

    /// Resolve the provider context window: config override → model-reported
    /// `context_length` → conservative default. Runs once at startup and again
    /// after a model switch; a failed fetch keeps the current value.
    async fn refresh_context_window(&mut self) {
        let resolved = self.compaction.resolved_window(0);
        if self.compaction.context_window > 0 {
            self.context_window = resolved;
            return;
        }
        if let Ok(models) = self.provider.list_models().await
            && let Some(model) = models.iter().find(|model| {
                model.id == self.model || model.name.as_deref() == Some(self.model.as_str())
            })
        {
            self.context_window = self
                .compaction
                .resolved_window(model.context_length.unwrap_or(0));
            return;
        }
        self.context_window = self.compaction.resolved_window(0);
    }

    /// Approximate current context occupation: exact from the last request's
    /// `Done` usage when available, else an estimate over the live session.
    /// `extra_bytes` covers material added since that request (the new user
    /// message); it is small relative to the reserved response slack.
    fn context_tokens_estimate(&self, extra_bytes: usize) -> u64 {
        let base = match self.last_context_tokens {
            Some(exact) => exact,
            None => match self.session.as_ref() {
                Some(state) => estimate_live_tokens(&state.session),
                None => self.estimate_history_tokens(),
            },
        };
        base.saturating_add(compact::estimate::estimate_tokens(extra_bytes))
    }

    /// Estimate context tokens directly from `self.history` (no durable
    /// session / no provider usage yet).
    fn estimate_history_tokens(&self) -> u64 {
        let mut bytes = 0usize;
        for message in &self.history {
            for content in &message.content {
                match content {
                    Content::Text(text) | Content::Reasoning(text) => {
                        bytes = bytes.saturating_add(text.len())
                    }
                    Content::ToolResult { content, .. } => {
                        bytes = bytes.saturating_add(content.len())
                    }
                    Content::Opaque { data, .. } => {
                        bytes = bytes.saturating_add(
                            serde_json::to_string(data)
                                .map(|value| value.len())
                                .unwrap_or(0),
                        );
                    }
                    Content::ToolCall(call) => {
                        bytes = bytes.saturating_add(call.name.len());
                        bytes = bytes.saturating_add(
                            serde_json::to_string(&call.arguments)
                                .map(|rendered| rendered.len())
                                .unwrap_or(0),
                        );
                    }
                }
            }
        }
        compact::estimate::estimate_tokens(bytes)
    }

    /// Whether the pre-turn auto-compaction trigger fires for a turn adding
    /// `user_text`.
    fn should_auto_compact(&self, user_text: &str) -> bool {
        if !self.compaction.auto {
            return false;
        }
        let context = self.context_tokens_estimate(user_text.len());
        self.compaction
            .should_auto_compact(context, self.context_window)
    }

    /// Shared compaction routine used by the pre-turn trigger, manual
    /// `/compact`, and overflow recovery. Plans, summarizes (LLM with a
    /// deterministic fallback), persists the summary + the summarizer's usage,
    /// and rebuilds `self.history` from the new boundary. Returns `false`
    /// (with a `Notice`) when there is nothing to compact or persistence
    /// failed.
    async fn compact_and_reload(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        reason: CompactionReason,
    ) -> bool {
        let Some(state) = self.session.as_ref() else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return false;
        };
        let session = state.session.clone();

        let estimated = self.context_tokens_estimate(0);
        let Some(plan) = plan_compaction(&session, &self.compaction, estimated) else {
            send(events, AgentEvent::Notice("nothing to compact yet".into()));
            return false;
        };

        let outcome = compact_summarize(
            self.provider.as_ref(),
            &self.model,
            &plan,
            &self.compaction,
            cancel,
        )
        .await;

        // Persist the summarizer's own usage so session cost totals stay
        // honest and the UI reflects it.
        if let SummaryOutcome::Model { usage, .. } = &outcome {
            let summary = usage_summary(usage);
            let _ = self.persist_event(SessionEvent::Usage { usage: summary }, events);
        }

        let compacted_through = plan.boundary;
        let summary = match &outcome {
            SummaryOutcome::Model { text, .. } | SummaryOutcome::Deterministic { text } => text,
        };
        let summary_bytes = summary.len();

        if !self.persist_event(
            SessionEvent::CompactionSummary {
                summary: summary.clone(),
                compacted_through,
            },
            events,
        ) {
            return false;
        }

        // This is also the fix for the manual `/compact` no-op: without this
        // rebuild the live conversation would keep stale (uncompacted) history
        // until next restart.
        if let Some(state) = self.session.as_ref() {
            self.history = state.session.context_messages();
            send(events, usage_event(&state.session.metadata.usage));
        }

        send(
            events,
            AgentEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
                auto: reason == CompactionReason::Auto,
                reason,
            },
        );
        true
    }

    /// Handle a context-overflow provider error (a 400 whose body matches
    /// context-exceeded patterns) by emergency-compacting and returning
    /// whether the caller should retry the request. Bounded to
    /// `MAX_OVERFLOW_RECOVERIES` per turn.
    async fn try_overflow_recovery(
        &mut self,
        error: &LlmError,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        attempts: &mut usize,
    ) -> bool {
        if !is_context_overflow(error) || *attempts >= MAX_OVERFLOW_RECOVERIES {
            return false;
        }
        *attempts += 1;
        if self
            .compact_and_reload(events, cancel, CompactionReason::Overflow)
            .await
        {
            send(
                events,
                AgentEvent::Notice(format!(
                    "overflow recovery: compacted ({attempts}/{MAX_OVERFLOW_RECOVERIES}); retrying"
                )),
            );
            true
        } else {
            false
        }
    }
}

/// Factory used to build providers when the model/provider selection changes.
/// A `Box<dyn Fn>` (rather than a generic) keeps the call site simple; the
/// dispatch cost is negligible because it only runs when `/model` is used.
type ProviderFactory = Box<dyn Fn(&str) -> anyhow::Result<Arc<dyn Provider>>>;

/// Fetch a provider's model list on a background task, reporting
/// `AgentEvent::ModelList` on success and a notice on failure.  Shared by the
/// startup fetch in `main` and the `/model` and `/models` handlers.
pub fn spawn_model_list(
    provider_name: String,
    provider: Arc<dyn Provider>,
    events: mpsc::UnboundedSender<AgentEvent>,
) {
    tokio::spawn(async move {
        match provider.list_models().await {
            Ok(models) => send(
                &events,
                AgentEvent::ModelList {
                    provider: provider_name,
                    models,
                },
            ),
            Err(error) => send(
                &events,
                AgentEvent::Notice(format!("could not fetch model list: {error}")),
            ),
        }
    });
}

/// Adapt the session-owned snapshot into the UI-facing entry type, adding
/// tool summaries. The session crate owns the event walk and tool-result
/// pairing; this conversion is purely presentational.
///
/// Summaries are display-only: raw JSON arguments stay in the session store
/// and in `context_messages`, so loaded and continued sessions keep full
/// fidelity. Do not persist these strings back into session events.
fn ui_snapshot_entries(entries: Vec<session::SessionSnapshotEntry>) -> Vec<SessionSnapshotEntry> {
    entries
        .into_iter()
        .map(|entry| match entry {
            session::SessionSnapshotEntry::User { text } => SessionSnapshotEntry::User { text },
            session::SessionSnapshotEntry::Assistant {
                markdown,
                reasoning,
            } => SessionSnapshotEntry::Assistant {
                markdown,
                reasoning,
            },
            session::SessionSnapshotEntry::Tool {
                name,
                arguments,
                ok,
                output,
                error,
            } => {
                let summary = call_summary(&name, &arguments);
                SessionSnapshotEntry::Tool {
                    name,
                    summary,
                    ok,
                    duration_ms: 0,
                    output,
                    error,
                }
            }
        })
        .collect()
}

/// Push a non-durable recovery note into history, preserving provider role
/// alternation: append into a trailing user message when present, otherwise
/// push a new user message. The note lives only in memory; durable events
/// stay clean.
fn push_recovery_note(history: &mut Vec<Message>, note: String) {
    let tail_is_user = matches!(
        history.last(),
        Some(Message {
            role: Role::User,
            ..
        })
    );
    if tail_is_user
        && let Some(Message {
            role: Role::User,
            content,
        }) = history.last_mut()
    {
        content.push(Content::Text(note));
        return;
    }
    history.push(Message::user(note));
}

fn append_assistant(
    history: &mut Vec<Message>,
    reasoning: &str,
    text: &str,
    opaque: &[(String, serde_json::Value)],
    calls: Vec<ToolCall>,
) {
    if reasoning.is_empty() && text.is_empty() && opaque.is_empty() && calls.is_empty() {
        return;
    }
    let mut content = Vec::new();
    if !reasoning.is_empty() {
        content.push(Content::Reasoning(reasoning.to_owned()));
    }
    if !text.is_empty() {
        content.push(Content::Text(text.to_owned()));
    }
    content.extend(opaque.iter().map(|(provider, data)| Content::Opaque {
        provider: provider.clone(),
        data: data.clone(),
    }));
    content.extend(calls.into_iter().map(Content::ToolCall));
    history.push(Message {
        role: Role::Assistant,
        content,
    });
}

fn usage_event(usage: &session::UsageSummary) -> AgentEvent {
    AgentEvent::UsageUpdated {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_tokens: usage.cached_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        cost: format_cost(usage.cost),
    }
}

fn format_cost(cost: f64) -> String {
    if cost == 0.0 {
        "0".into()
    } else {
        format!("{cost:.6}")
    }
}

fn send(events: &mpsc::UnboundedSender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
}

/// Whether a provider error is a context-window rejection (the request it
/// describes was too large to admit). Proxies vary in both status and
/// phrasing, so we match on the rendered error text; a 400 alone is not
/// enough (it could be a malformed request).
fn is_context_overflow(error: &LlmError) -> bool {
    let body = error.to_string().to_ascii_lowercase();
    const PATTERNS: &[&str] = &[
        "context length",
        "context_length",
        "context window",
        "too many tokens",
        "maximum context",
        "max context",
        "maximum prompt",
        "input is too long",
        "exceeds the maximum",
        "token limit",
    ];
    PATTERNS.iter().any(|pattern| body.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;
    use async_trait::async_trait;
    use futures_util::stream;
    use llm::{EventStream, LlmError, ModelInfo, Usage};
    use serde_json::{Value, json};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tempfile::tempdir;
    use tools::Tool;

    /// One step of a canned provider script.  Errors are carried as strings
    /// because `LlmError` embeds a non-cloneable `reqwest::Error`; `stream`
    /// maps them back into `LlmError::Stream`.
    type ScriptStep = Result<StreamEvent, String>;

    /// How mock stream errors are wrapped; lets tests exercise the
    /// parse-error and transient-error recovery paths specifically.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum MockErrorKind {
        Stream,
        Parse,
        Retryable,
    }

    /// Build a script that emits the given events successfully.
    fn script(events: Vec<StreamEvent>) -> Vec<ScriptStep> {
        events.into_iter().map(Ok).collect()
    }

    struct MockProvider {
        calls: AtomicUsize,
        scripts: Vec<Vec<ScriptStep>>,
        error_kind: MockErrorKind,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            let index = self.calls.fetch_add(1, Ordering::SeqCst);
            let script = self.scripts.get(index).cloned().unwrap_or_default();
            let error_kind = self.error_kind;
            Ok(Box::pin(stream::iter(script.into_iter().map(
                move |step| {
                    step.map_err(|message| match error_kind {
                        MockErrorKind::Stream => LlmError::Stream(message),
                        MockErrorKind::Parse => LlmError::Parse(message),
                        MockErrorKind::Retryable => LlmError::Http {
                            status: 500,
                            body: message,
                        },
                    })
                },
            ))))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    fn run_agent(provider: MockProvider) -> (Vec<AgentEvent>, Vec<Message>) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("hello".into()))
                .unwrap();
            drop(input_tx);
            let agent = Agent::new(Arc::new(provider), ToolRegistry::empty(), "demo", cancel);
            let history = agent.history.clone();
            agent.run(input_rx, event_tx).await;
            let mut events = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                events.push(event);
            }
            (events, history)
        })
    }

    /// Execution interval of one probe invocation, for overlap assertions.
    type ProbeLog = Arc<Mutex<Vec<(&'static str, Instant, Instant)>>>;

    #[test]
    fn plan_preserves_order_and_groups_read_only_runs() {
        let log: ProbeLog = Arc::default();
        let registry = ToolRegistry::try_new(vec![
            Box::new(ProbeTool::read_only("probe_a", &log)),
            Box::new(ProbeTool::read_only("probe_b", &log)),
            Box::new(ProbeTool::exclusive("probe_write", &log)),
            Box::new(ProbeTool::read_only("probe_c", &log)),
        ])
        .unwrap();
        let calls = vec![
            call("c1", "probe_a"),
            call("c2", "probe_b"),
            call("c3", "probe_write"),
            call("c4", "probe_c"),
        ];
        let batches = plan_tool_batches(calls, &registry);
        let described: Vec<(usize, bool)> = batches
            .iter()
            .map(|batch| (batch.calls.len(), batch.concurrent()))
            .collect();
        // Maximal read-only runs group; exclusive calls stay singletons; the
        // trailing read is NOT hoisted above the write.
        assert_eq!(described, vec![(2, true), (1, false), (1, true)]);
        let flattened: Vec<&str> = batches
            .iter()
            .flat_map(|batch| batch.calls.iter().map(|call| call.id.as_str()))
            .collect();
        assert_eq!(flattened, vec!["c1", "c2", "c3", "c4"]);
    }

    #[test]
    fn parallel_calls_group_per_tool_without_cross_tool_merging() {
        let log: ProbeLog = Arc::default();
        let registry = ToolRegistry::try_new(vec![
            Box::new(ProbeTool::parallel("fan_a", &log)),
            Box::new(ProbeTool::parallel("fan_b", &log)),
            Box::new(ProbeTool::read_only("probe_a", &log)),
        ])
        .unwrap();
        let calls = vec![
            call("c1", "fan_a"),
            call("c2", "fan_a"),
            call("c3", "fan_b"),
            call("c4", "probe_a"),
            call("c5", "fan_a"),
        ];
        let batches = plan_tool_batches(calls, &registry);
        // Adjacent same-tool fan-out merges into one concurrent run; a
        // different fan-out tool or an interleaved read splits the run;
        // nothing is reordered.
        let described: Vec<(usize, String)> = batches
            .iter()
            .map(|batch| (batch.calls.len(), format!("{:?}", batch.class)))
            .collect();
        assert_eq!(
            described,
            vec![
                (2, "Parallel".to_owned()),
                (1, "Parallel".to_owned()),
                (1, "ReadOnly".to_owned()),
                (1, "Parallel".to_owned()),
            ]
        );
        let flattened: Vec<&str> = batches
            .iter()
            .flat_map(|batch| batch.calls.iter().map(|call| call.id.as_str()))
            .collect();
        assert_eq!(flattened, vec!["c1", "c2", "c3", "c4", "c5"]);
    }

    #[test]
    fn parallel_calls_overlap_within_their_batch() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let log: ProbeLog = Arc::default();
            let registry =
                ToolRegistry::try_new(vec![Box::new(ProbeTool::parallel("fan", &log))]).unwrap();
            let provider = MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    script(vec![
                        StreamEvent::ToolCallComplete(call("c1", "fan")),
                        StreamEvent::ToolCallComplete(call("c2", "fan")),
                        StreamEvent::Done {
                            stop_reason: Some("tool_calls".into()),
                            usage: None,
                        },
                    ]),
                    script(vec![
                        StreamEvent::TextDelta("done".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                error_kind: MockErrorKind::Stream,
            };
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx.send(InputMessage::Message("go".into())).unwrap();
            drop(input_tx);
            Agent::new(Arc::new(provider), registry, "demo", cancel)
                .run(input_rx, event_tx)
                .await;
            while event_rx.try_recv().is_ok() {}

            // Both fan-out probes ran concurrently: their execution intervals
            // overlap even though neither is read-only.
            let entries = log.lock().unwrap().clone();
            assert_eq!(entries.len(), 2, "both probes ran");
            let (_, first_start, first_end) = entries[0];
            let (_, second_start, second_end) = entries[1];
            assert!(
                first_start < second_end && second_start < first_end,
                "expected overlapping intervals, got {entries:?}"
            );
        });
    }

    #[test]
    fn read_only_calls_overlap_and_exclusive_serializes() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let log: ProbeLog = Arc::default();
            let registry = ToolRegistry::try_new(vec![
                Box::new(ProbeTool::read_only("probe_a", &log)),
                Box::new(ProbeTool::read_only("probe_b", &log)),
                Box::new(ProbeTool::exclusive("probe_write", &log)),
            ])
            .unwrap();
            let provider = MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    script(vec![
                        StreamEvent::ToolCallComplete(call("c1", "probe_a")),
                        StreamEvent::ToolCallComplete(call("c2", "probe_b")),
                        StreamEvent::ToolCallComplete(call("c3", "probe_write")),
                        StreamEvent::Done {
                            stop_reason: Some("tool_calls".into()),
                            usage: None,
                        },
                    ]),
                    script(vec![
                        StreamEvent::TextDelta("done".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                error_kind: MockErrorKind::Stream,
            };
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx.send(InputMessage::Message("go".into())).unwrap();
            drop(input_tx);
            Agent::new(Arc::new(provider), registry, "demo", cancel)
                .run(input_rx, event_tx)
                .await;
            let mut events = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                events.push(event);
            }

            // Both read-only probes ran concurrently: their execution
            // intervals overlap.
            let entries = log.lock().unwrap().clone();
            let interval = |name: &str| {
                entries
                    .iter()
                    .find(|entry| entry.0 == name)
                    .copied()
                    .unwrap_or_else(|| panic!("{name} never executed"))
            };
            let (a_name, a_start, a_end) = interval("probe_a");
            let (b_name, b_start, b_end) = interval("probe_b");
            assert!(
                a_start < b_end && b_start < a_end,
                "{a_name} and {b_name} should have overlapped"
            );
            // The write started only after both reads finished.
            let (_, w_start, _) = interval("probe_write");
            assert!(w_start >= a_end && w_start >= b_end);

            // Finished events: live finishes may arrive in completion order
            // within the read-only batch, so compare as a set; the write in
            // its later singleton batch always finishes last.
            let finished: Vec<&str> = events
                .iter()
                .filter_map(|event| match event {
                    AgentEvent::ToolCallFinished { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(finished.len(), 3);
            assert_eq!(finished[2], "probe_write");
            let mut reads = finished[..2].to_vec();
            reads.sort_unstable();
            assert_eq!(reads, vec!["probe_a", "probe_b"]);
        });
    }

    /// A tool whose concurrency class and latency are configurable; records
    /// its execution interval so tests can assert real overlap.
    struct ProbeTool {
        name: &'static str,
        class: Concurrency,
        log: ProbeLog,
    }

    impl ProbeTool {
        fn read_only(name: &'static str, log: &ProbeLog) -> Self {
            Self {
                name,
                class: Concurrency::ReadOnly,
                log: Arc::clone(log),
            }
        }

        fn exclusive(name: &'static str, log: &ProbeLog) -> Self {
            Self {
                name,
                class: Concurrency::Exclusive,
                log: Arc::clone(log),
            }
        }

        /// A fan-out tool (the subagent stand-in): parallelizable with
        /// itself only.
        fn parallel(name: &'static str, log: &ProbeLog) -> Self {
            Self {
                name,
                class: Concurrency::Parallel,
                log: Arc::clone(log),
            }
        }
    }

    #[async_trait]
    impl Tool for ProbeTool {
        fn spec(&self) -> tools::ToolSpec {
            tools::ToolSpec {
                definition: llm::ToolDefinition {
                    name: self.name.into(),
                    description: "probe".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: tools::ToolPrompt::default(),
            }
        }

        fn concurrency(&self, _args: &Value) -> Concurrency {
            self.class
        }

        async fn execute(&self, _args: Value, cancel: CancellationToken) -> ToolOutput {
            let start = Instant::now();
            // Long enough that two probes reliably overlap on any CI box.
            tokio::time::sleep(Duration::from_millis(60)).await;
            let _ = cancel;
            let end = Instant::now();
            self.log.lock().unwrap().push((self.name, start, end));
            ToolOutput {
                content: format!("{} done", self.name),
                is_error: false,
                summary: self.name.into(),
            }
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({}),
        }
    }

    // ------------------------------------------------------------------
    // Concurrent lifecycle: stable ids, launch-accurate starts, exactly-once
    // finishes, and bounded fan-out (Task 1 + Task 10 coverage).
    // ------------------------------------------------------------------

    /// A parallel tool whose calls meet on a shared barrier: every call
    /// blocks until all parties arrive, so a batch can only complete if the
    /// calls genuinely overlapped.
    struct BarrierTool {
        barrier: Arc<tokio::sync::Barrier>,
    }

    #[async_trait]
    impl Tool for BarrierTool {
        fn spec(&self) -> tools::ToolSpec {
            tools::ToolSpec {
                definition: llm::ToolDefinition {
                    name: "gate".into(),
                    description: "probe".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: tools::ToolPrompt::default(),
            }
        }

        fn concurrency(&self, _args: &Value) -> Concurrency {
            Concurrency::Parallel
        }

        async fn execute(&self, _args: Value, cancel: CancellationToken) -> ToolOutput {
            let _ = cancel;
            self.barrier.wait().await;
            ToolOutput {
                content: "gated done".into(),
                is_error: false,
                summary: "gate".into(),
            }
        }
    }

    /// A tool that never resolves on its own; only being dropped by an
    /// interrupt ends it.
    struct HangingTool {
        name: &'static str,
        class: Concurrency,
    }

    #[async_trait]
    impl Tool for HangingTool {
        fn spec(&self) -> tools::ToolSpec {
            tools::ToolSpec {
                definition: llm::ToolDefinition {
                    name: self.name.into(),
                    description: "probe".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: tools::ToolPrompt::default(),
            }
        }

        fn concurrency(&self, _args: &Value) -> Concurrency {
            self.class
        }

        async fn execute(&self, _args: Value, cancel: CancellationToken) -> ToolOutput {
            cancel.cancelled().await;
            unreachable!("cancelled futures are dropped before returning")
        }
    }

    fn parallel_calls_script() -> Vec<Vec<ScriptStep>> {
        vec![
            script(vec![
                StreamEvent::ToolCallComplete(call("t1", "gate")),
                StreamEvent::ToolCallComplete(call("t2", "gate")),
                StreamEvent::Done {
                    stop_reason: Some("tool_calls".into()),
                    usage: None,
                },
            ]),
            script(vec![
                StreamEvent::TextDelta("done".into()),
                StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: None,
                },
            ]),
        ]
    }

    async fn run_with_registry_and_limits(
        registry: ToolRegistry,
        limits: SubagentLimits,
        scripts: Vec<Vec<ScriptStep>>,
        interrupt: bool,
    ) -> Vec<AgentEvent> {
        let cancel = CancellationToken::new();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        input_tx.send(InputMessage::Message("go".into())).unwrap();
        if interrupt {
            input_tx.send(InputMessage::Interrupt).unwrap();
        }
        drop(input_tx);
        let provider = MockProvider {
            calls: AtomicUsize::new(0),
            scripts,
            error_kind: MockErrorKind::Stream,
        };
        Agent::new(Arc::new(provider), registry, "demo", cancel)
            .with_subagent_limits(limits)
            .run(input_rx, event_tx)
            .await;
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        events
    }

    fn started_ids(events: &[AgentEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolCallStarted { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn finished_ids(events: &[AgentEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolCallFinished { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn fan_out_starts_both_before_either_finishes_with_distinct_ids() {
        let registry = ToolRegistry::try_new(vec![Box::new(BarrierTool {
            barrier: Arc::new(tokio::sync::Barrier::new(2)),
        })])
        .unwrap();
        let events = run_with_registry_and_limits(
            registry,
            SubagentLimits { max_concurrent: 2 },
            parallel_calls_script(),
            false,
        )
        .await;

        // Both calls start with their original llm::ToolCall ids, both
        // starts precede any finish, and each finishes exactly once.
        assert_eq!(started_ids(&events), vec!["t1", "t2"]);
        let last_start = events
            .iter()
            .rposition(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
            .unwrap();
        let first_finish = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ToolCallFinished { .. }))
            .unwrap();
        assert!(last_start < first_finish, "starts must precede finishes");
        let mut finishes = finished_ids(&events);
        finishes.sort_unstable();
        assert_eq!(finishes, vec!["t1", "t2"]);
        assert!(events.contains(&AgentEvent::TurnFinished));
    }

    #[tokio::test]
    async fn limit_one_delays_the_second_start_until_the_first_finishes() {
        let log: ProbeLog = Arc::default();
        // Registered under the same name the scripted calls use.
        let registry =
            ToolRegistry::try_new(vec![Box::new(ProbeTool::parallel("gate", &log))]).unwrap();
        let events = run_with_registry_and_limits(
            registry,
            SubagentLimits { max_concurrent: 1 },
            parallel_calls_script(),
            false,
        )
        .await;

        // With one slot, the second start cannot be announced before the
        // first call's finish event.
        let second_start = events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
            .count();
        assert_eq!(second_start, 2);
        let finish_of_first = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ToolCallFinished { .. }))
            .unwrap();
        let start_positions: Vec<usize> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| matches!(event, AgentEvent::ToolCallStarted { .. }))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(start_positions.len(), 2);
        assert_eq!(start_positions[0], 0); // relative to first tool event
        assert!(
            start_positions[1] > finish_of_first,
            "second start must wait for a freed slot"
        );

        // And the two probes really did not overlap.
        let entries = log.lock().unwrap().clone();
        assert_eq!(entries.len(), 2);
        let (_, a_start, a_end) = entries[0];
        let (_, b_start, b_end) = entries[1];
        assert!(a_end <= b_start || b_end <= a_start, "serialized");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interrupt_produces_exactly_one_finish_per_call_and_valid_history() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let session = store
            .create(SessionCreateOptions {
                provider: Some("mock".into()),
                model: Some("demo".into()),
                ..SessionCreateOptions::default()
            })
            .unwrap();

        let session_id = session.id();
        // Both calls target the same parallel tool so they merge into one
        // concurrent batch; each hangs until the interrupt drops the batch.
        let registry = ToolRegistry::try_new(vec![Box::new(HangingTool {
            name: "hang",
            class: Concurrency::Parallel,
        })])
        .unwrap();
        let cancel = CancellationToken::new();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        input_tx.send(InputMessage::Message("go".into())).unwrap();
        let agent = Agent::new(
            Arc::new(MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![script(vec![
                    StreamEvent::ToolCallComplete(call("t1", "hang")),
                    StreamEvent::ToolCallComplete(call("t2", "hang")),
                    StreamEvent::Done {
                        stop_reason: Some("tool_calls".into()),
                        usage: None,
                    },
                ])],
                error_kind: MockErrorKind::Stream,
            }),
            registry,
            "demo",
            cancel,
        )
        .with_session(store.clone(), session);
        let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

        // Wait until both calls are announced and running, then interrupt:
        // the dispatcher is provably parked in its select loop at that point,
        // so the interrupt lands on the tool batch deterministically.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut events = Vec::new();
        loop {
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for tool starts"
            );
            match event_rx.try_recv() {
                Ok(event) => {
                    events.push(event);
                    let starts = events
                        .iter()
                        .filter(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
                        .count();
                    if starts >= 2 {
                        break;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    panic!("agent exited before both tools started")
                }
            }
        }
        input_tx.send(InputMessage::Interrupt).unwrap();
        drop(input_tx);
        agent_task.await.unwrap();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }

        // Balanced lifecycle: one start per call, one cancelled finish per
        // call, no duplicates.
        assert_eq!(started_ids(&events), vec!["t1", "t2"]);
        assert_eq!(finished_ids(&events), vec!["t1", "t2"]);
        for event in &events {
            if let AgentEvent::ToolCallFinished { ok, output, .. } = event {
                assert!(!ok);
                assert!(output.is_empty());
            }
        }
        assert!(events.contains(&AgentEvent::TurnFinished));

        // Durable history: failed tool results for every model-issued call,
        // plus the TurnCancelled marker.
        let loaded = store.open(&session_id).unwrap();
        let results: Vec<(&String, &bool)> = loaded
            .events
            .iter()
            .filter_map(|record| match &record.event {
                SessionEvent::ToolResult {
                    tool_call_id,
                    is_error,
                    content,
                    ..
                } if content == "cancelled" => Some((tool_call_id, is_error)),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 2, "{results:?}");
        assert!(results.iter().all(|(_, is_error)| **is_error));
        assert!(
            loaded
                .events
                .iter()
                .any(|record| matches!(record.event, SessionEvent::TurnCancelled { .. }))
        );
    }

    #[tokio::test]
    async fn zero_max_concurrent_is_clamped_instead_of_deadlocking() {
        let registry = ToolRegistry::try_new(vec![Box::new(BarrierTool {
            barrier: Arc::new(tokio::sync::Barrier::new(1)),
        })])
        .unwrap();
        // A public caller constructing a zero limit must not wedge the batch:
        // the clamp makes it behave like 1 slot at minimum.
        let events = run_with_registry_and_limits(
            registry,
            SubagentLimits { max_concurrent: 0 },
            vec![script(vec![
                StreamEvent::ToolCallComplete(call("t1", "gate")),
                StreamEvent::Done {
                    stop_reason: Some("tool_calls".into()),
                    usage: None,
                },
            ])],
            false,
        )
        .await;
        let finished: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolCallFinished { call_id, ok, .. } if *ok => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(finished, vec!["t1"], "the single call ran to success");
    }

    // --------------------------------------------------------------- subagent
    // end-to-end fan-out through the real `subagent` tool (Task 10).

    /// Fake runner that records peak concurrency and optionally gates all
    /// in-flight runs behind a barrier.
    struct CountingRunner {
        current: AtomicUsize,
        max: AtomicUsize,
        gate: Option<Arc<tokio::sync::Barrier>>,
    }

    struct InFlightGuard<'a>(&'a AtomicUsize);

    impl Drop for InFlightGuard<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl tools::SubagentRunner for CountingRunner {
        async fn run(
            &self,
            description: &str,
            _prompt: &str,
            mode: tools::SubagentMode,
            _cancel: CancellationToken,
        ) -> Result<String, String> {
            let previous = self.current.fetch_add(1, Ordering::SeqCst);
            self.max.fetch_max(previous + 1, Ordering::SeqCst);
            let _guard = InFlightGuard(&self.current);
            if let Some(gate) = &self.gate {
                gate.wait().await;
            }
            Ok(format!("{}/{mode:?}", description))
        }
    }

    fn parent_script_for(calls: &[(&str, &str)]) -> Vec<Vec<ScriptStep>> {
        let mut first = Vec::new();
        for (id, args_mode) in calls {
            let mut arguments = serde_json::Map::new();
            arguments.insert("description".into(), json!(format!("task {id}")));
            arguments.insert("prompt".into(), json!("do it"));
            if !args_mode.is_empty() {
                arguments.insert("mode".into(), json!(args_mode));
            }
            first.push(StreamEvent::ToolCallComplete(ToolCall {
                id: (*id).into(),
                name: crate::tools::SUBAGENT_TOOL_NAME.into(),
                arguments: Value::Object(arguments),
            }));
        }
        first.push(StreamEvent::Done {
            stop_reason: Some("tool_calls".into()),
            usage: None,
        });
        vec![
            script(first),
            script(vec![
                StreamEvent::TextDelta("parent done".into()),
                StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: None,
                },
            ]),
        ]
    }

    async fn run_parent_fan_out(
        runner: Arc<CountingRunner>,
        limits: SubagentLimits,
        calls: &[(&str, &str)],
    ) -> Vec<AgentEvent> {
        let mut registry = ToolRegistry::empty();
        registry.register_subagent(runner).unwrap();
        let cancel = CancellationToken::new();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        input_tx.send(InputMessage::Message("go".into())).unwrap();
        drop(input_tx);
        let provider = MockProvider {
            calls: AtomicUsize::new(0),
            scripts: parent_script_for(calls),
            error_kind: MockErrorKind::Stream,
        };
        Agent::new(Arc::new(provider), registry, "demo", cancel)
            .with_subagent_limits(limits)
            .run(input_rx, event_tx)
            .await;
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        events
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_read_only_subagents_overlap_up_to_the_limit() {
        let runner = Arc::new(CountingRunner {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            gate: Some(Arc::new(tokio::sync::Barrier::new(2))),
        });
        let events = run_parent_fan_out(
            runner.clone(),
            SubagentLimits { max_concurrent: 2 },
            &[("t1", ""), ("t2", "")],
        )
        .await;

        // The barrier can only be passed by both runners entering together.
        assert_eq!(runner.max.load(Ordering::SeqCst), 2);
        assert_eq!(started_ids(&events), vec!["t1", "t2"]);
        // Results pair with their original call ids regardless of which
        // finished first.
        let reports: std::collections::HashMap<&str, &str> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolCallFinished {
                    call_id,
                    output,
                    ok: true,
                    ..
                } => Some((call_id.as_str(), output.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(reports.get("t1"), Some(&"task t1/ReadOnly"));
        assert_eq!(reports.get("t2"), Some(&"task t2/ReadOnly"));
        assert!(events.contains(&AgentEvent::TurnFinished));
    }

    #[tokio::test]
    async fn read_only_fan_out_respects_a_limit_of_one() {
        let runner = Arc::new(CountingRunner {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            gate: None,
        });
        let events = run_parent_fan_out(
            runner.clone(),
            SubagentLimits { max_concurrent: 1 },
            &[("t1", ""), ("t2", "")],
        )
        .await;

        assert_eq!(runner.max.load(Ordering::SeqCst), 1);
        let first_finish = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ToolCallFinished { .. }))
            .unwrap();
        let second_start = events
            .iter()
            .rposition(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
            .unwrap();
        assert!(
            second_start > first_finish,
            "the second delegation must wait for a free slot"
        );
    }

    #[tokio::test]
    async fn workspace_delegations_stay_serial_even_when_parallel_is_allowed() {
        let runner = Arc::new(CountingRunner {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            gate: None,
        });
        let events = run_parent_fan_out(
            runner.clone(),
            SubagentLimits { max_concurrent: 4 },
            &[("t1", "workspace"), ("t2", "workspace")],
        )
        .await;

        // Workspace mode classifies Exclusive: singleton batches, strict
        // serialization despite a limit of 4.
        assert_eq!(runner.max.load(Ordering::SeqCst), 1);
        let first_finish = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ToolCallFinished { .. }))
            .unwrap();
        let second_start = events
            .iter()
            .rposition(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
            .unwrap();
        assert!(second_start > first_finish);
    }

    #[test]
    fn simple_text_turn_forwards_deltas() {
        let (events, _) = run_agent(MockProvider {
            calls: AtomicUsize::new(0),
            scripts: vec![script(vec![
                StreamEvent::TextDelta("hello".into()),
                StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(Usage::default()),
                },
            ])],
            error_kind: MockErrorKind::Stream,
        });
        assert!(events.contains(&AgentEvent::TextDelta("hello".into())));
        assert!(events.contains(&AgentEvent::TurnFinished));
    }

    #[test]
    fn durable_session_can_be_loaded_after_a_turn() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = store
                .create(SessionCreateOptions {
                    provider: Some("mock".into()),
                    model: Some("demo".into()),
                    ..SessionCreateOptions::default()
                })
                .unwrap();
            let provider = MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![script(vec![
                    StreamEvent::ReasoningDelta("thinking".into()),
                    StreamEvent::TextDelta("answer".into()),
                    StreamEvent::Done {
                        stop_reason: Some("stop".into()),
                        usage: Some(Usage {
                            input_tokens: 3,
                            output_tokens: 2,
                            ..Usage::default()
                        }),
                    },
                ])],
                error_kind: MockErrorKind::Stream,
            };
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, _event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("hello".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(Arc::new(provider), ToolRegistry::empty(), "demo", cancel)
                .with_session(store.clone(), session.clone())
                .run(input_rx, event_tx)
                .await;
            let loaded = store.open(&session.id()).unwrap();
            assert_eq!(loaded.context_messages().len(), 2);
            assert_eq!(loaded.metadata.usage.input_tokens, 3);
            assert!(
                loaded
                    .events
                    .iter()
                    .any(|record| matches!(record.event, SessionEvent::AssistantMessage { .. }))
            );
        });
    }

    #[test]
    fn system_prompt_includes_project_context_block() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let provider = Arc::new(RecordingProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![script(vec![
                    StreamEvent::TextDelta("answer".into()),
                    StreamEvent::Done {
                        stop_reason: Some("stop".into()),
                        usage: None,
                    },
                ])],
                seen: Mutex::new(Vec::new()),
            });
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, _event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("hello".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(provider.clone(), ToolRegistry::empty(), "demo", cancel)
                .with_project_context("<project_context>\nrepo rule\n</project_context>")
                .run(input_rx, event_tx)
                .await;

            let seen = provider.seen.lock().unwrap();
            assert_eq!(seen.len(), 1);
            let system = seen[0].0.as_deref().unwrap_or("");
            assert!(
                system.contains("<project_context>"),
                "system prompt must include the project context block"
            );
            assert!(system.contains("repo rule"));
            // The skill catalog is not present (no skills), and the block is
            // appended after the base prompt.
            assert!(system.contains("You are harness"));
        });
    }

    #[test]
    fn tool_call_is_fed_back_before_second_stream() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let provider = MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    script(vec![
                        StreamEvent::ToolCallComplete(ToolCall {
                            id: "c".into(),
                            name: "missing".into(),
                            arguments: json!({}),
                        }),
                        StreamEvent::Done {
                            stop_reason: Some("tool_calls".into()),
                            usage: None,
                        },
                    ]),
                    script(vec![
                        StreamEvent::TextDelta("done".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                error_kind: MockErrorKind::Stream,
            };
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("use tool".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(Arc::new(provider), ToolRegistry::empty(), "demo", cancel)
                .run(input_rx, event_tx)
                .await;
            let mut got = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                got.push(event);
            }
            assert!(got.contains(&AgentEvent::TextDelta("done".into())));
            assert!(got.iter().any(|event| matches!(
                event,
                AgentEvent::ToolCallStarted { summary, .. } if summary == "missing"
            )));
            assert!(got.iter().any(|event| matches!(
                event,
                AgentEvent::ToolCallFinished {
                    output,
                    error: Some(error),
                    ..
                } if output == "unknown tool: missing" && error == "unknown tool: missing"
            )));
        });
    }

    /// Regression test for the cross-cutting review: when the provider emits
    /// tool calls and then the stream dies before `Done`, the turn must still
    /// finish (no busy wait) and the pending tool calls must be persisted as
    /// failed tool results so the next turn sees a consistent history.
    #[test]
    fn stream_error_after_tool_call_finishes_turn_and_persists_tool_error() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = store
                .create(SessionCreateOptions {
                    provider: Some("mock".into()),
                    model: Some("demo".into()),
                    ..SessionCreateOptions::default()
                })
                .unwrap();
            let provider = MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![vec![
                    Ok(StreamEvent::ToolCallComplete(ToolCall {
                        id: "c".into(),
                        name: "missing".into(),
                        arguments: json!({}),
                    })),
                    Err("connection dropped mid-stream".into()),
                ]],
                error_kind: MockErrorKind::Stream,
            };
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("use tool".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(Arc::new(provider), ToolRegistry::empty(), "demo", cancel)
                .with_session(store.clone(), session.clone())
                .run(input_rx, event_tx)
                .await;

            let mut got = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                got.push(event);
            }
            assert!(got.contains(&AgentEvent::TurnFinished));
            assert!(got.iter().any(|event| matches!(
                event,
                AgentEvent::Error(message) if message.contains("connection dropped mid-stream")
            )));

            // The interrupted tool call is persisted as a failed ToolResult
            // instead of being dropped or left dangling.
            let loaded = store.open(&session.id()).unwrap();
            let tool_results: Vec<(&bool, &String)> = loaded
                .events
                .iter()
                .filter_map(|record| match &record.event {
                    SessionEvent::ToolResult {
                        is_error, content, ..
                    } => Some((is_error, content)),
                    _ => None,
                })
                .collect();
            assert_eq!(tool_results.len(), 1, "expected one persisted tool result");
            let (is_error, content) = tool_results[0];
            assert!(
                is_error,
                "interrupted tool call must be persisted as an error"
            );
            assert!(
                content.contains("provider stream interrupted"),
                "unexpected tool result content: {content}"
            );
        });
    }

    /// A tool call whose arguments fail to parse (the "EOF while parsing a
    /// list" case from truncated output) must not dead-end the turn: the agent
    /// re-streams once with an in-memory recovery note, and the retried answer
    /// is persisted with nothing left dangling.
    #[test]
    fn parse_error_retries_turn_instead_of_dead_ending() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = store
                .create(SessionCreateOptions {
                    provider: Some("mock".into()),
                    model: Some("demo".into()),
                    ..SessionCreateOptions::default()
                })
                .unwrap();
            let provider = Arc::new(MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    vec![Err("EOF while parsing a list at line 1 column 253".into())],
                    script(vec![
                        StreamEvent::TextDelta("retried".into()),
                        StreamEvent::Done {
                            stop_reason: None,
                            usage: None,
                        },
                    ]),
                ],
                error_kind: MockErrorKind::Parse,
            });
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("use the tool".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(provider.clone(), ToolRegistry::empty(), "demo", cancel)
                .with_session(store.clone(), session.clone())
                .run(input_rx, event_tx)
                .await;

            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                2,
                "a parse error should re-stream once instead of ending the turn"
            );
            let mut got = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                got.push(event);
            }
            assert!(got.iter().any(|event| matches!(
                event,
                AgentEvent::Error(message) if message.contains("EOF while parsing")
            )));
            assert!(got.contains(&AgentEvent::TextDelta("retried".into())));
            assert!(got.contains(&AgentEvent::TurnFinished));

            // The retried answer is durable and no tool call is left dangling.
            let loaded = store.open(&session.id()).unwrap();
            let assistant_texts: Vec<&String> = loaded
                .events
                .iter()
                .filter_map(|record| match &record.event {
                    SessionEvent::AssistantMessage { message } => {
                        message.content.iter().find_map(|content| match content {
                            session::StoredContent::Text { text } => Some(text),
                            _ => None,
                        })
                    }
                    _ => None,
                })
                .collect();
            assert!(
                assistant_texts.iter().any(|text| text.contains("retried")),
                "the retried answer should be persisted"
            );
        });
    }

    /// A transient mid-stream failure (connection drop / decode error) must
    /// not dead-end the turn either: the agent re-streams once and the retried
    /// answer is delivered.
    #[test]
    fn retryable_stream_error_retries_turn_instead_of_dead_ending() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let provider = Arc::new(MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    vec![Err("error decoding response body".into())],
                    script(vec![
                        StreamEvent::TextDelta("recovered".into()),
                        StreamEvent::Done {
                            stop_reason: None,
                            usage: None,
                        },
                    ]),
                ],
                error_kind: MockErrorKind::Retryable,
            });
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("keep going".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(provider.clone(), ToolRegistry::empty(), "demo", cancel)
                .run(input_rx, event_tx)
                .await;

            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                2,
                "a retryable stream error should re-stream once"
            );
            let mut got = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                got.push(event);
            }
            assert!(got.iter().any(|event| matches!(
                event,
                AgentEvent::Error(message) if message.contains("http 500")
            )));
            assert!(got.contains(&AgentEvent::TextDelta("recovered".into())));
            assert!(got.contains(&AgentEvent::TurnFinished));
        });
    }

    /// A response with no text and no tool calls is a provider stall; the
    /// agent re-streams once with a nudge instead of ending the turn silently.
    #[test]
    fn empty_response_retries_turn_once() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let provider = Arc::new(MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    script(vec![StreamEvent::Done {
                        stop_reason: None,
                        usage: None,
                    }]),
                    script(vec![
                        StreamEvent::TextDelta("answer".into()),
                        StreamEvent::Done {
                            stop_reason: None,
                            usage: None,
                        },
                    ]),
                ],
                error_kind: MockErrorKind::Stream,
            });
            let cancel = CancellationToken::new();
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("do the work".into()))
                .unwrap();
            drop(input_tx);
            Agent::new(provider.clone(), ToolRegistry::empty(), "demo", cancel)
                .run(input_rx, event_tx)
                .await;

            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                2,
                "an empty response should re-stream once"
            );
            let mut got = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                got.push(event);
            }
            assert!(got.contains(&AgentEvent::TextDelta("answer".into())));
            assert!(got.contains(&AgentEvent::TurnFinished));
            assert!(
                !got.iter()
                    .any(|event| matches!(event, AgentEvent::Error(_))),
                "an empty response is a stall, not an error"
            );
        });
    }

    // ------------------------------------------------------------------
    // Token-aware auto-compaction tests
    // ------------------------------------------------------------------

    /// Provider that records every request (system + messages) so tests can
    /// assert on what was actually sent, and answers from a canned script.
    /// The summarizer request shares this same provider and is recognizable by
    /// its system prompt.
    struct RecordingProvider {
        calls: AtomicUsize,
        scripts: Vec<Vec<ScriptStep>>,
        seen: Mutex<Vec<(Option<String>, Vec<Message>)>>,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
            let index = self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen
                .lock()
                .unwrap()
                .push((request.system.clone(), request.messages.clone()));
            let script = self.scripts.get(index).cloned().unwrap_or_default();
            Ok(Box::pin(stream::iter(
                script
                    .into_iter()
                    .map(|step| step.map_err(LlmError::Stream)),
            )))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    fn summarizer_script() -> Vec<ScriptStep> {
        script(vec![
            StreamEvent::TextDelta("### Test Summary ###\nAll prior work is captured here.".into()),
            StreamEvent::Done {
                stop_reason: Some("stop".into()),
                usage: Some(Usage {
                    input_tokens: 100,
                    output_tokens: 5,
                    ..Usage::default()
                }),
            },
        ])
    }

    /// Create a durable session pre-populated with `turns` of user+assistant
    /// messages (assistant ~`assistant_bytes`), giving the planner material to
    /// summarize without the live estimate itself crossing the trigger.
    fn populate_session(store: &SessionStore, turns: usize, assistant_bytes: usize) -> Session {
        let mut session = store
            .create(SessionCreateOptions {
                provider: Some("mock".into()),
                model: Some("demo".into()),
                ..SessionCreateOptions::default()
            })
            .unwrap();
        for index in 0..turns {
            store
                .append_event(
                    &mut session,
                    SessionEvent::UserMessage {
                        message: StoredMessage::from_llm(&Message::user(format!(
                            "question {index}"
                        ))),
                    },
                )
                .unwrap();
            store
                .append_event(
                    &mut session,
                    SessionEvent::AssistantMessage {
                        message: StoredMessage::from_llm(&Message::assistant(vec![Content::Text(
                            "a".repeat(assistant_bytes),
                        )])),
                    },
                )
                .unwrap();
        }
        session
    }

    /// Run an agent with a durable session to completion over `inputs`,
    /// returning the emitted agent events and the (Arc) provider. Must be
    /// awaited inside the caller's runtime.
    async fn run_session_agent(
        store: &SessionStore,
        session: Session,
        provider: Arc<RecordingProvider>,
        inputs: Vec<InputMessage>,
    ) -> (Vec<AgentEvent>, Arc<RecordingProvider>) {
        let cancel = CancellationToken::new();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        for input in inputs {
            input_tx.send(input).unwrap();
        }
        drop(input_tx);
        Agent::new(provider.clone(), ToolRegistry::empty(), "demo", cancel)
            .with_session(store.clone(), session)
            .run(input_rx, event_tx)
            .await;
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        (events, provider)
    }

    fn is_summary_message(message: &Message) -> bool {
        message.content.iter().any(|content| match content {
            Content::Text(text) => text.contains("[Generated session summary"),
            _ => false,
        })
    }

    #[test]
    fn pre_turn_trigger_compacts_and_shrinks_next_request() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = populate_session(&store, 12, 12_000);
            let populated_session_id = session.id();

            let provider = Arc::new(RecordingProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    // Turn 1 reports a huge exact context (over the trigger).
                    script(vec![
                        StreamEvent::TextDelta("turn 1".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: Some(Usage {
                                input_tokens: 200_000,
                                output_tokens: 1_000,
                                ..Usage::default()
                            }),
                        },
                    ]),
                    summarizer_script(),
                    script(vec![
                        StreamEvent::TextDelta("turn 2".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                seen: Mutex::new(Vec::new()),
            });

            let (events, provider) = run_session_agent(
                &store,
                session,
                provider.clone(),
                vec![
                    InputMessage::Message("first".into()),
                    InputMessage::Message("second".into()),
                ],
            )
            .await;

            // The auto trigger fired between turns.
            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::CompactionFinished {
                    auto: true,
                    reason: CompactionReason::Auto,
                    ..
                }
            )));
            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::Notice(message) if message.contains("auto-compacted")
            )));

            // Three requests: turn 1, the summarizer, turn 2.
            let seen = provider.seen.lock().unwrap();
            assert_eq!(
                seen.len(),
                3,
                "expected turn1 + summarizer + turn2 requests"
            );

            // The summarizer request is distinguishable by its system prompt
            // and modeled as a standalone summarization, not a conversation.
            let (system, messages) = &seen[1];
            assert!(system.as_deref().unwrap_or("").contains("summarization"));
            assert_eq!(messages.len(), 1, "summarizer gets a single user prompt");

            // The turn-2 conversation request sees the summary and a *smaller*
            // history than the turn-1 request.
            let turn2 = &seen[2].1;
            assert!(
                turn2.iter().any(is_summary_message),
                "turn 2 request must include the generated summary"
            );
            assert!(
                turn2.len() < seen[0].1.len(),
                "history must shrink after compaction"
            );

            // The summarizer's usage was recorded so session cost stays honest.
            drop(seen);
            let reloaded = store.open(&populated_session_id).unwrap();
            assert!(
                reloaded.metadata.usage.input_tokens >= 200_100,
                "summarizer usage (input 100) must be folded into session usage"
            );
            assert!(reloaded.events.iter().any(|record| matches!(
                &record.event,
                SessionEvent::Usage { usage }
                    if usage.input_tokens == 100
            )));
        });
    }

    #[test]
    fn summarizer_failure_falls_back_to_deterministic_persisted() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = populate_session(&store, 12, 12_000);
            let session_id = session.id();

            let provider = Arc::new(RecordingProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    script(vec![
                        StreamEvent::TextDelta("turn 1".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: Some(Usage {
                                input_tokens: 200_000,
                                output_tokens: 1_000,
                                ..Usage::default()
                            }),
                        },
                    ]),
                    vec![Err("summarizer connection dropped".into())],
                    script(vec![
                        StreamEvent::TextDelta("turn 2".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                seen: Mutex::new(Vec::new()),
            });

            let (events, _provider) = run_session_agent(
                &store,
                session,
                provider.clone(),
                vec![
                    InputMessage::Message("first".into()),
                    InputMessage::Message("second".into()),
                ],
            )
            .await;

            // Even though the summarizer failed, compaction completed via the
            // deterministic fallback and the turn went on.
            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::CompactionFinished {
                    summary_bytes,
                    ..
                } if *summary_bytes > 0
            )));
            assert!(events.contains(&AgentEvent::TextDelta("turn 2".into())));

            // The persisted summary is the deterministic transcript.
            let reloaded = store.open(&session_id).unwrap();
            let deterministic = reloaded
                .events
                .iter()
                .find_map(|record| match &record.event {
                    SessionEvent::CompactionSummary { summary, .. } => Some(summary.clone()),
                    _ => None,
                });
            assert!(
                deterministic
                    .unwrap_or_default()
                    .contains("generated context"),
                "fallback summary must be the deterministic transcript"
            );
        });
    }

    #[test]
    fn overflow_recovery_compacts_and_retries_successfully() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            // Small enough that the pre-turn estimate does not trigger; the
            // overflow rejection is the only pressure source.
            let session = populate_session(&store, 12, 12_000);

            let provider = Arc::new(RecordingProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    // The first request is rejected for context overflow.
                    vec![Err("context length exceeded for this request".into())],
                    summarizer_script(),
                    script(vec![
                        StreamEvent::TextDelta("recovered answer".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                seen: Mutex::new(Vec::new()),
            });

            let (events, provider) = run_session_agent(
                &store,
                session,
                provider.clone(),
                vec![InputMessage::Message("do the work".into())],
            )
            .await;

            // Emergency compaction happened with Overflow reason and the retry
            // succeeded (turn finished normally).
            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::CompactionFinished {
                    reason: CompactionReason::Overflow,
                    auto: false,
                    ..
                }
            )));
            assert!(events.contains(&AgentEvent::TextDelta("recovered answer".into())));
            assert!(events.contains(&AgentEvent::TurnFinished));

            // Requests: overflow-rejected request, summarizer, retried request.
            let seen = provider.seen.lock().unwrap();
            assert_eq!(seen.len(), 3);
            // The retried request uses the compacted (summary + tail) history.
            let retried = &seen[2].1;
            assert!(
                retried.iter().any(is_summary_message),
                "retried request must include the generated summary"
            );
            assert!(retried.len() < seen[0].1.len(), "history must shrink");
        });
    }

    /// Regression: manual `/compact` used to be a no-op on the live
    /// conversation (the summary event was appended but `history` was never
    /// rebuilt). It must now rebuild history so the very next request sees the
    /// summary.
    #[test]
    fn manual_compact_rebuilds_history_for_next_request() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = populate_session(&store, 12, 12_000);
            let session_id = session.id();

            let provider = Arc::new(RecordingProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    summarizer_script(),
                    script(vec![
                        StreamEvent::TextDelta("after compact".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                seen: Mutex::new(Vec::new()),
            });

            let (events, provider) = run_session_agent(
                &store,
                session,
                provider.clone(),
                vec![
                    InputMessage::CompactSession,
                    InputMessage::Message("next".into()),
                ],
            )
            .await;

            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::CompactionFinished {
                    auto: false,
                    reason: CompactionReason::Manual,
                    ..
                }
            )));

            // The first request is the summarizer; the second is the turn after
            // compaction, and it must already carry the rebuilt summary.
            let seen = provider.seen.lock().unwrap();
            assert_eq!(seen.len(), 2);
            let turn_after = &seen[1].1;
            assert!(
                turn_after.iter().any(is_summary_message),
                "the request after manual /compact must include the summary"
            );

            // Reload round-trip: the compacted session loads with summary +
            // post-boundary events only (no pre-boundary user messages).
            let reloaded = store.open(&session_id).unwrap();
            let messages = reloaded.context_messages();
            assert!(
                messages.iter().any(is_summary_message),
                "reloaded session context must start from the summary"
            );
            let earliest_user = messages
                .iter()
                .position(|m| m.role == Role::User && !is_summary_message(m));
            let summary_at = messages.iter().position(is_summary_message).unwrap();
            match earliest_user {
                None => {}
                Some(user_index) => assert!(
                    user_index > summary_at,
                    "no user message may precede the generated summary"
                ),
            }
        });
    }

    /// Sum body: a long session that crosses the threshold compacts exactly
    /// once per crossing and the conversation remains coherent (the summary
    /// precedes every subsequent user message).
    #[test]
    fn long_session_auto_compaction_stays_coherent() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = populate_session(&store, 12, 12_000);
            let session_id = session.id();

            let provider = Arc::new(RecordingProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![
                    script(vec![
                        StreamEvent::TextDelta("turn 1".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: Some(Usage {
                                input_tokens: 200_000,
                                output_tokens: 1_000,
                                ..Usage::default()
                            }),
                        },
                    ]),
                    summarizer_script(),
                    script(vec![
                        StreamEvent::TextDelta("turn 2".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ]),
                ],
                seen: Mutex::new(Vec::new()),
            });

            let (events, _provider) = run_session_agent(
                &store,
                session,
                provider.clone(),
                vec![
                    InputMessage::Message("first".into()),
                    InputMessage::Message("second".into()),
                ],
            )
            .await;

            // Exactly one auto-compaction for this crossing.
            let auto_compactions = events
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        AgentEvent::CompactionFinished {
                            auto: true,
                            reason: CompactionReason::Auto,
                            ..
                        }
                    )
                })
                .count();
            assert_eq!(auto_compactions, 1, "expected exactly one auto compaction");

            // Coherence on disk: every real user message follows the summary.
            let reloaded = store.open(&session_id).unwrap();
            let messages = reloaded.context_messages();
            let mut seen_summary = false;
            for message in &messages {
                if is_summary_message(message) {
                    seen_summary = true;
                    continue;
                }
                if message.role == Role::User {
                    assert!(seen_summary, "user message must follow the summary");
                }
            }
        });
    }
}
