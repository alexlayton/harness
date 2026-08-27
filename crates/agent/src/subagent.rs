//! The subagent runner: a nested, bounded agent loop behind the
//! [`SubagentRunner`] trait that `crates/tools` injects into its `subagent`
//! tool.
//!
//! Design notes:
//!
//! - **Fresh context every run.** The child sees only its own system prompt
//!   plus the delegating `prompt`; parent history never crosses over. Token
//!   isolation is the point of delegation.
//! - **No recursion by construction.** The child registry is built by
//!   `default_registry`, which does not include the subagent tool; there is
//!   no code path that could register a runner inside a runner.
//! - **Provider/model snapshot per run.** Each child takes one snapshot of
//!   the parent's active provider/model and uses it consistently for every
//!   request, the child-session header, and retry logging; a parent `/model`
//!   switch retargets only future runs (`update_model`). Child usage is
//!   persisted to the child session only, so parent `/usage` totals are not
//!   double-counted.
//! - **Durable when possible.** With a session store available each run
//!   creates a child session linked via `parent_session` and titled with the
//!   task description; without one (`--no-session`) runs stay ephemeral.
//!
//! The loop itself is deliberately a lean copy of the parent turn shape
//! (stream → dispatch → repeat until a text-only reply) rather than a reuse
//! of `Agent`: it needs none of the frontends' input handling, compaction
//! triggers, or slash commands, and sharing those would couple the child to
//! UI concerns it must not have.

use crate::agent::plan_tool_batches;
use crate::assembly::SubagentPolicy;
use crate::prompt::subagent_system_prompt;
use async_trait::async_trait;
use futures_util::stream::{FuturesUnordered, StreamExt};
use llm::{
    CompletionRequest, Content, Message, Provider, ReasoningPolicy, RetryCallback, Role,
    StreamEvent, truncate_utf8,
};
use session::{SessionCreateOptions, SessionStore, usage_summary};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tools::{
    FileSearchIndex, SubagentMode, SubagentRunner, ToolConfig, ToolRegistry, call_summary,
    default_registry_with_index, read_only_registry_with_index,
};

/// Upper bound on the final report handed back to the parent conversation:
/// large enough for a thorough audit, small enough not to blow the parent's
/// context when several reports land in one turn.
const REPORT_MAX_BYTES: usize = 20_000;

/// Upper bound on read-only tool calls running at once inside one subagent.
/// Mirrors the parent's cap; children never see `Parallel` tools, so this is
/// the only in-flight limit they need.
const MAX_CONCURRENT_READ_ONLY_TOOLS: usize = 8;

/// One in-flight child tool execution carrying its slot index so results
/// land in original call order rather than completion order.
type ChildToolRun<'a> = Pin<Box<dyn Future<Output = (usize, tools::ToolOutput)> + Send + 'a>>;

/// One delegated subagent run.
pub(crate) struct SubagentRun {
    pub description: String,
    pub prompt: String,
}

/// The active provider/model/reasoning selection used by future child runs. Held behind a
/// short-held synchronous lock so a parent `/model` switch can retarget
/// subagents without rebuilding the runner; never held across `.await`.
struct SubagentModelState {
    provider: Arc<dyn Provider>,
    model: String,
    reasoning: ReasoningPolicy,
}

/// Everything a nested loop needs from the host process. One instance is
/// shared by every subagent invocation from an agent frontend.
pub struct SubagentRunnerImpl {
    model_state: RwLock<SubagentModelState>,
    workspace_root: PathBuf,
    search_index: Arc<FileSearchIndex>,
    rtk: bool,
    project_context: String,
    /// Resolved delegation bounds (`max_turns`; `0` disables subagents).
    config: SubagentPolicy,
    /// Store shared with the parent session (same workspace scope); `None`
    /// under `--no-session`, making runs ephemeral.
    store: Option<SessionStore>,
    /// Parent session id recorded on future child sessions for lineage. This
    /// changes when the interactive host starts or loads a conversation, so
    /// it cannot be a startup-only snapshot like the store itself. Shared
    /// through an `Arc` so a child run can snapshot it at launch.
    parent_session_id: Arc<RwLock<Option<session::SessionId>>>,
    /// Built lazily on first use, one cache per [`SubagentMode`]: the FFF
    /// index walk should not cost anything when the model never delegates.
    /// The fallible result is cached too — repeatedly re-building a registry
    /// that deterministically failed helps no one. `get_or_init` guarantees
    /// exactly one constructor run even when two first-use delegations race.
    child_registries: [OnceLock<Result<Arc<ToolRegistry>, String>>; 2],
    /// Test-only counter of registry constructor runs, for the single-init
    /// concurrency test.
    #[cfg(test)]
    test_registry_builds: std::sync::atomic::AtomicUsize,
}

/// Cache slot for a mode's child registry.
fn registry_slot(mode: SubagentMode) -> usize {
    match mode {
        SubagentMode::ReadOnly => 0,
        SubagentMode::Workspace => 1,
    }
}

impl SubagentRunnerImpl {
    /// Assemble the runner host state. Registration order matters: build this
    /// *before* registering the subagent tool on the parent registry, since
    /// the tool consumes an `Arc<dyn SubagentRunner>` pointing back here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        workspace_root: impl Into<PathBuf>,
        rtk: bool,
        project_context: impl Into<String>,
        config: SubagentPolicy,
        store: Option<SessionStore>,
        parent_session_id: Option<session::SessionId>,
    ) -> Self {
        let workspace_root = workspace_root.into();
        let search_index = Arc::new(
            FileSearchIndex::new(&workspace_root)
                .expect("subagent workspace was already validated by assembly"),
        );
        Self {
            model_state: RwLock::new(SubagentModelState {
                provider,
                model: model.into(),
                reasoning: ReasoningPolicy::Auto,
            }),
            workspace_root,
            search_index,
            rtk,
            project_context: project_context.into(),
            config,
            store,
            parent_session_id: Arc::new(RwLock::new(parent_session_id)),
            child_registries: [OnceLock::new(), OnceLock::new()],
            #[cfg(test)]
            test_registry_builds: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Set the initial reasoning policy before the runner is shared.
    pub fn with_reasoning(self, reasoning: ReasoningPolicy) -> Self {
        self.update_reasoning(reasoning);
        self
    }

    /// Replace the compatibility constructor's index with the assembly-owned
    /// parent index before the runner can be shared or used.
    pub fn with_file_search_index(mut self, index: Arc<FileSearchIndex>) -> Self {
        self.search_index = index;
        self
    }

    /// Retarget future child runs after the parent switched provider/model.
    /// Host-side hook (called by `Agent` after a successful `/model`);
    /// already-running children keep the snapshot they started with.
    pub fn update_model(&self, provider: Arc<dyn Provider>, model: impl Into<String>) {
        let mut state = self
            .model_state
            .write()
            .expect("subagent model state lock poisoned");
        state.provider = provider;
        state.model = model.into();
    }

    /// Retarget future child runs after `/reasoning` changes.
    pub fn update_reasoning(&self, reasoning: ReasoningPolicy) {
        self.model_state
            .write()
            .expect("subagent model state lock poisoned")
            .reasoning = reasoning;
    }

    /// Retarget child-session lineage after the host starts or loads another
    /// parent conversation. Already-running children have created their own
    /// headers before control can return to the host command loop.
    pub fn update_parent_session(&self, parent_session_id: Option<session::SessionId>) {
        *self
            .parent_session_id
            .write()
            .expect("subagent parent session lock poisoned") = parent_session_id;
    }

    /// Build each mode's child tool set exactly once. The read-only registry
    /// deliberately exposes no mutating tools and no shell (the scheduler
    /// class is not a sandbox — exclusion is the enforcement); the workspace
    /// registry is the normal built-in set. Neither contains a subagent tool,
    /// which is what makes recursion structurally impossible.
    fn registry(&self, mode: SubagentMode) -> Result<Arc<ToolRegistry>, String> {
        let cache = &self.child_registries[registry_slot(mode)];
        let workspace_root = self.workspace_root.clone();
        let rtk = self.rtk;
        let search_index = self.search_index.clone();
        #[cfg(test)]
        let builds = &self.test_registry_builds;
        let built = cache.get_or_init(move || {
            tracing::debug!(mode = mode.as_str(), "building child tool registry");
            #[cfg(test)]
            builds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let result = match mode {
                SubagentMode::Workspace => {
                    default_registry_with_index(ToolConfig::new(&workspace_root, rtk), search_index)
                }
                SubagentMode::ReadOnly => read_only_registry_with_index(
                    ToolConfig::new(&workspace_root, rtk),
                    search_index,
                ),
            };
            result
                .map(Arc::new)
                .map_err(|error| format!("could not build subagent tools: {error}"))
        });
        match built {
            Ok(registry) => Ok(registry.clone()),
            Err(error) => Err(error.clone()),
        }
    }

    /// Persist the child session header up front so even a hard crash leaves
    /// a resumable trace of the delegation. The header records the provider
    /// /model snapshot this run started with, not whatever the parent uses now.
    fn create_child_session(
        store: &Option<SessionStore>,
        provider: &Arc<dyn Provider>,
        model: &str,
        description: &str,
        parent_session: Option<session::SessionId>,
    ) -> Option<session::Session> {
        let store = store.as_ref()?;
        match store.create(SessionCreateOptions {
            title: Some(description.to_owned()),
            provider: Some(provider.name().to_owned()),
            model: Some(model.to_owned()),
            parent_session,
        }) {
            Ok(session) => Some(session),
            Err(error) => {
                // A failed child header degrades to an ephemeral run rather
                // than failing the delegation.
                tracing::warn!(error = %error, "could not create subagent session");
                None
            }
        }
    }

    /// Run the nested loop. Returns the final report or a human-readable
    /// failure. Cancellation aborts promptly with an error mentioning it, so
    /// the parent's interrupt path can synthesize its cancelled tool result.
    async fn execute(
        &self,
        run: &SubagentRun,
        mode: SubagentMode,
        cancel: CancellationToken,
    ) -> Result<String, String> {
        if self.config.max_turns == 0 {
            return Err("subagents are disabled".into());
        }
        let started = Instant::now();
        // One snapshot per child run: every request, the child-session
        // header/lineage, and retry logging use the selection that existed
        // when this child was launched, even if the parent changes while
        // registry initialization or provider work is in flight.
        let parent_session = *self
            .parent_session_id
            .read()
            .expect("subagent parent session lock poisoned");
        let (provider, model, reasoning) = {
            let state = self
                .model_state
                .read()
                .expect("subagent model state lock poisoned");
            (state.provider.clone(), state.model.clone(), state.reasoning)
        };
        let registry = self.registry(mode)?;
        let registry_snapshot = registry.snapshot();
        let mut history = vec![Message::user(run.prompt.clone())];
        let system = subagent_system_prompt(
            &self.workspace_root.display().to_string(),
            &registry_snapshot.prompt_context,
            registry.skills(),
            &self.project_context,
            mode,
        );
        let mut session = Self::create_child_session(
            &self.store,
            &provider,
            &model,
            &run.description,
            parent_session,
        );
        Self::persist(
            &self.store,
            &mut session,
            session::SessionEvent::UserMessage {
                message: session::StoredMessage::from_llm(&history[0]),
            },
        );
        let outcome = self
            .loop_turns(
                &provider,
                &model,
                &registry,
                &system,
                reasoning,
                &mut history,
                &mut session,
                cancel,
            )
            .await;
        tracing::info!(
            description = %run.description,
            mode = mode.as_str(),
            provider = provider.name(),
            model = %model,
            child_session = ?session.as_ref().map(|child| child.id().to_string()),
            duration_ms = started.elapsed().as_millis() as u64,
            ok = outcome.is_ok(),
            "subagent finished"
        );
        if let Some(child) = session.as_ref()
            && let Some(store) = self.store.as_ref()
            && let Err(error) = store.sync_session(child)
        {
            // Deferred-sync stores flush at the parent's turn boundary
            // anyway; a failed explicit sync here only loses durability of
            // the tail, which is not worth failing the delegation over.
            tracing::warn!(error = %error, "subagent session sync failed");
        }
        outcome
    }

    /// Persist one event to the child session when both a store and a child
    /// exist. Failures are logged, never fatal: an unpersisted delegation
    /// still completed its real work.
    fn persist(
        store: &Option<SessionStore>,
        session: &mut Option<session::Session>,
        event: session::SessionEvent,
    ) {
        if let (Some(store), Some(child)) = (store, session)
            && let Err(error) = store.append_event(child, event)
        {
            tracing::warn!(error = %error, "subagent session persist failed");
        }
    }

    /// Stream → dispatch → repeat until the model answers with text only.
    #[allow(clippy::too_many_arguments)]
    async fn loop_turns(
        &self,
        provider: &Arc<dyn Provider>,
        model: &str,
        registry: &ToolRegistry,
        system: &str,
        reasoning: ReasoningPolicy,
        history: &mut Vec<Message>,
        session: &mut Option<session::Session>,
        cancel: CancellationToken,
    ) -> Result<String, String> {
        let run_description = last_user_line(history);
        let model_for_log = model.to_owned();
        let mut turns = 0usize;
        loop {
            if turns >= self.config.max_turns {
                // Budget exhaustion is a degraded success: hand back whatever
                // the subagent produced instead of dead-ending the parent's
                // tool result with an error.
                tracing::warn!(turns, "subagent hit its turn budget");
                return Ok(format!(
                    "[subagent stopped after {} turns without a final report] Last output: {}",
                    self.config.max_turns,
                    last_assistant_text(history).unwrap_or_else(|| "(none)".into())
                ));
            }
            turns += 1;

            // Reserve the final logical turn for synthesis. In real use a
            // broad audit can otherwise spend every allowed turn issuing
            // another read and hit the budget with no assistant text at all.
            // The note is request-local (not durable child history), and an
            // empty tool list makes the expected terminal action unambiguous.
            let final_report_turn = turns == self.config.max_turns;
            let mut request_messages = history.clone();
            if final_report_turn {
                push_request_note(
                    &mut request_messages,
                    "[system note: this is your final turn. Do not call tools. Return the best \
                     final report you can from the evidence already gathered.]",
                );
            }
            let request = CompletionRequest {
                model: model.to_owned(),
                system: Some(system.to_owned()),
                messages: request_messages,
                tools: if final_report_turn {
                    Vec::new()
                } else {
                    registry.definitions()
                },
                max_tokens: None,
                temperature: None,
                reasoning,
            };
            // Standard initial-request retry policy, same as the parent: a
            // transient 429/5xx while *obtaining* the stream must not
            // immediately fail an otherwise valid delegation just because
            // several delegations fan out at once. Retries consume no extra
            // logical turn. The callback only logs — there is no child UI
            // event stream to notify.
            let on_retry: RetryCallback = Arc::new({
                let run_description = run_description.clone();
                let model_for_log = model_for_log.clone();
                move |attempt, error| {
                    tracing::warn!(
                        description = %run_description,
                        model = %model_for_log,
                        attempt,
                        error = %error,
                        "retrying subagent provider request"
                    );
                }
            });
            let mut stream = tokio::select! {
                result = provider.stream_with_retry(&request, on_retry) => result
                    .map_err(|error| format!("provider error: {error}"))?,
                _ = cancel.cancelled() => return Err("cancelled by user".into()),
            };

            let mut text = String::new();
            let mut opaque = Vec::<(String, serde_json::Value)>::new();
            let mut tool_calls = Vec::<llm::ToolCall>::new();
            let mut stream_error = None;
            loop {
                tokio::select! {
                    next = stream.next() => match next {
                        Some(Ok(StreamEvent::TextDelta(delta))) => text.push_str(&delta),
                        Some(Ok(StreamEvent::ReasoningDelta(_))) => {}
                        Some(Ok(StreamEvent::OpaqueState { provider, data })) => {
                            opaque.push((provider, data));
                        }
                        Some(Ok(StreamEvent::ToolCallComplete(call))) => tool_calls.push(call),
                        // Child usage lands in the child session only, so
                        // parent totals never double-count delegated work.
                        Some(Ok(StreamEvent::Done {
                            usage: Some(usage),..
                        })) => Self::persist(
                            &self.store,
                            session,
                            session::SessionEvent::Usage {
                                usage: usage_summary(&usage),
                            },
                        ),
                        Some(Ok(StreamEvent::Done { .. })) => {}
                        Some(Err(error)) => {
                            stream_error = Some(error);
                            break;
                        }
                        None => break,
                    },
                    _ = cancel.cancelled() => return Err("cancelled by user".into()),
                }
            }

            append_assistant(history, &text, &opaque, tool_calls.clone());
            for event in assistant_events(&text, &opaque, &tool_calls) {
                Self::persist(&self.store, session, event);
            }

            if let Some(error) = stream_error {
                mark_calls_failed(
                    history,
                    &self.store,
                    session,
                    &tool_calls,
                    &format!("provider stream interrupted: {error}"),
                );
                // No retry ladder here: the parent turn already has recovery
                // machinery, and a failed subagent surfaces as an errored
                // tool result the parent can choose to re-delegate.
                return Err(format!("provider stream interrupted: {error}"));
            }

            if tool_calls.is_empty() {
                if text.trim().is_empty() {
                    // Treat an empty reply like any other stall: nudge once
                    // by appending a user note and continuing within budget.
                    history.push(Message::user(
                        "[system note: your previous response produced no output; continue \
                         and complete the task.]",
                    ));
                    continue;
                }
                // Text-only reply: this *is* the report.
                return Ok(truncate_utf8(text.trim(), REPORT_MAX_BYTES).to_owned());
            }

            self.dispatch_tool_batches(registry, history, session, tool_calls, &cancel)
                .await?;
        }
    }

    /// Program-order dispatch mirroring the parent agent: maximal read-only
    /// runs batch concurrently, adjacent same-tool `Parallel` calls would fan
    /// out (the child has no such tool), everything else serializes. Results
    /// land in original call order; cancellation synthesizes "cancelled"
    /// results for unfilled slots before unwinding.
    async fn dispatch_tool_batches(
        &self,
        registry: &ToolRegistry,
        history: &mut Vec<Message>,
        session: &mut Option<session::Session>,
        tool_calls: Vec<llm::ToolCall>,
        cancel: &CancellationToken,
    ) -> Result<(), String> {
        for batch in plan_tool_batches(tool_calls, registry) {
            // The batch shares a child token so a cancel kills exactly this
            // phase while leaving the outer token untouched for cleanup.
            let batch_cancel = cancel.child_token();
            let mut futures: FuturesUnordered<ChildToolRun<'_>> = FuturesUnordered::new();
            let launch_limit = if batch.concurrent() {
                MAX_CONCURRENT_READ_ONLY_TOOLS
            } else {
                1
            };
            let mut next_launch = 0usize;
            while next_launch < batch.calls.len().min(launch_limit) {
                futures.push(Self::launch(
                    registry,
                    &batch.calls[next_launch],
                    next_launch,
                    batch_cancel.clone(),
                ));
                next_launch += 1;
            }
            let mut slots: Vec<Option<tools::ToolOutput>> =
                (0..batch.calls.len()).map(|_| None).collect();
            let mut finished = 0usize;
            loop {
                tokio::select! {
                    item = futures.next() => match item {
                        Some((index, output)) => {
                            slots[index] = Some(output);
                            finished += 1;
                            if finished == slots.len() {
                                break;
                            }
                            if next_launch < batch.calls.len() {
                                futures.push(Self::launch(registry, &batch.calls[next_launch], next_launch, batch_cancel.clone()));
                                next_launch += 1;
                            }
                        }
                        None => break,
                    },
                    _ = cancel.cancelled() => {
                        batch_cancel.cancel();
                        break;
                    }
                }
            }
            drop(futures);

            let mut cancelled = false;
            for (call, slot) in batch.calls.iter().zip(&mut slots) {
                let output = match slot.take() {
                    Some(output) => output,
                    None => {
                        cancelled = true;
                        tools::ToolOutput {
                            content: "cancelled".to_owned(),
                            is_error: true,
                            summary: call_summary(&call.name, &call.arguments),
                        }
                    }
                };
                history.push(Message {
                    role: Role::Tool,
                    content: vec![Content::ToolResult {
                        tool_call_id: call.id.clone(),
                        content: output.content.clone(),
                        is_error: output.is_error,
                    }],
                });
                Self::persist(
                    &self.store,
                    session,
                    session::SessionEvent::ToolResult {
                        tool_call_id: call.id.clone(),
                        content: output.content.clone(),
                        is_error: output.is_error,
                        tool_name: Some(call.name.clone()),
                    },
                );
            }
            if cancelled {
                return Err("cancelled by user".into());
            }
        }
        Ok(())
    }

    /// Start one registry execution as a slot-carrying future.
    fn launch<'a>(
        registry: &'a ToolRegistry,
        call: &'a llm::ToolCall,
        index: usize,
        cancel: CancellationToken,
    ) -> ChildToolRun<'a> {
        let name = call.name.clone();
        let arguments = call.arguments.clone();
        Box::pin(async move {
            let output = registry.execute(&name, arguments, cancel).await;
            (index, output)
        })
    }
}

#[async_trait]
impl SubagentRunner for SubagentRunnerImpl {
    async fn run(
        &self,
        description: &str,
        prompt: &str,
        mode: SubagentMode,
        cancel: CancellationToken,
    ) -> Result<String, String> {
        self.execute(
            &SubagentRun {
                description: description.to_owned(),
                prompt: prompt.to_owned(),
            },
            mode,
            cancel,
        )
        .await
    }
}

/// Last user message line, used as best-effort context in retry logs (never
/// the full prompt).
fn last_user_line(history: &[Message]) -> String {
    history
        .iter()
        .rev()
        .find_map(|message| {
            (message.role == Role::User).then(|| {
                message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
        })
        .map(|text| truncate_utf8(&text, 80).to_owned())
        .unwrap_or_default()
}

/// Add a request-local instruction while preserving provider role
/// alternation. In particular, a one-turn child already ends in its initial
/// user prompt, so the final-report note must fold into that message rather
/// than create two adjacent user messages.
fn push_request_note(history: &mut Vec<Message>, note: &str) {
    if let Some(Message {
        role: Role::User,
        content,
    }) = history.last_mut()
    {
        content.push(Content::Text(note.to_owned()));
    } else {
        history.push(Message::user(note));
    }
}

/// Append an assistant message and provider continuation state to history.
fn append_assistant(
    history: &mut Vec<Message>,
    text: &str,
    opaque: &[(String, serde_json::Value)],
    calls: Vec<llm::ToolCall>,
) {
    if text.is_empty() && opaque.is_empty() && calls.is_empty() {
        return;
    }
    let mut content = Vec::new();
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

/// Build the durable events for one assistant exchange, mirroring
/// `Agent::persist_assistant`: the `AssistantMessage` event carries text (and
/// reasoning, which children do not produce) ONLY — never tool calls — and
/// each call is emitted exactly once as a standalone `SessionEvent::ToolCall`.
/// Putting calls in both places used to make session validation reject the
/// duplicate, leaving exports with orphaned tool results. A response that
/// contains only tool calls emits no message event at all.
fn assistant_events(
    text: &str,
    opaque: &[(String, serde_json::Value)],
    calls: &[llm::ToolCall],
) -> Vec<session::SessionEvent> {
    let mut events = Vec::new();
    if !text.is_empty() || !opaque.is_empty() {
        let mut content = Vec::new();
        if !text.is_empty() {
            content.push(Content::Text(text.to_owned()));
        }
        content.extend(opaque.iter().map(|(provider, data)| Content::Opaque {
            provider: provider.clone(),
            data: data.clone(),
        }));
        events.push(session::SessionEvent::AssistantMessage {
            message: session::StoredMessage::from_llm(&Message::assistant(content)),
        });
    }
    events.extend(calls.iter().map(|call| session::SessionEvent::ToolCall {
        call: session::StoredToolCall::from(call),
    }));
    events
}

/// Give every streamed-but-unexecuted tool call a failed result so history
/// stays provider-valid, then record the same durably.
fn mark_calls_failed(
    history: &mut Vec<Message>,
    store: &Option<SessionStore>,
    session: &mut Option<session::Session>,
    calls: &[llm::ToolCall],
    reason: &str,
) {
    for call in calls {
        history.push(Message::tool_result(
            call.id.clone(),
            reason.to_owned(),
            true,
        ));
        SubagentRunnerImpl::persist(
            store,
            session,
            session::SessionEvent::ToolResult {
                tool_call_id: call.id.clone(),
                content: reason.to_owned(),
                is_error: true,
                tool_name: Some(call.name.clone()),
            },
        );
    }
}

/// Last assistant text in history, for budget-exhaustion hand-backs.
fn last_assistant_text(history: &[Message]) -> Option<String> {
    history
        .iter()
        .rev()
        .find_map(|message| {
            (message.role == Role::Assistant).then(|| {
                message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
        })
        .filter(|text| !text.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use llm::{EventStream, LlmError, ModelInfo};
    use serde_json::json;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    /// Provider that answers each request from a scripted list, recording
    /// every system prompt and tool set it was handed.
    struct ScriptProvider {
        scripts: Vec<Vec<Result<StreamEvent, String>>>,
        seen: Mutex<Vec<(Option<String>, Vec<String>)>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ScriptProvider {
        fn new(scripts: Vec<Vec<ScriptStep>>) -> Self {
            Self {
                scripts,
                seen: Mutex::new(Vec::new()),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn done(stop: &str) -> ScriptStep {
            Ok(StreamEvent::Done {
                stop_reason: Some(stop.to_owned()),
                usage: None,
            })
        }
    }

    type ScriptStep = Result<StreamEvent, String>;

    #[async_trait]
    impl Provider for ScriptProvider {
        fn name(&self) -> &str {
            "script"
        }

        async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
            let index = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.seen.lock().unwrap().push((
                request.system.clone(),
                request.tools.iter().map(|t| t.name.clone()).collect(),
            ));
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

    fn runner_with(provider: Arc<ScriptProvider>) -> SubagentRunnerImpl {
        let workspace = tempfile::tempdir().unwrap();
        // Leak the tempdir so the canonicalized workspace root outlives the
        // test; these tests never clean up /tmp otherwise.
        let root = std::mem::ManuallyDrop::new(workspace);
        SubagentRunnerImpl::new(
            provider,
            "demo",
            root.path().canonicalize().unwrap(),
            false,
            "",
            SubagentPolicy::default(),
            None,
            None,
        )
    }

    #[tokio::test]
    async fn text_only_reply_is_the_report() {
        let provider = Arc::new(ScriptProvider::new(vec![vec![
            Ok(StreamEvent::TextDelta("found 3 issues".into())),
            ScriptProvider::done("stop"),
        ]]));
        let runner = runner_with(provider.clone());
        let report = runner
            .run(
                "audit",
                "scan the crate",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(report, "found 3 issues");
        // The child saw a fresh one-message context with the subagent prompt.
        let (system, tools) = &provider.seen.lock().unwrap()[0];
        // The read-only preamble is mode-aware and states the enforced
        // restriction.
        assert!(system.as_deref().unwrap().contains("read-only subagent"));
        assert!(
            system
                .as_deref()
                .unwrap()
                .contains("mutation and command tools are unavailable")
        );
        assert!(!tools.iter().any(|name| name == "subagent"), "{tools:?}");
    }

    #[tokio::test]
    async fn workspace_mode_uses_the_workspace_preamble() {
        let provider = Arc::new(ScriptProvider::new(vec![vec![
            Ok(StreamEvent::TextDelta("done".into())),
            ScriptProvider::done("stop"),
        ]]));
        let runner = runner_with(provider.clone());
        runner
            .run(
                "fix it",
                "edit the file",
                SubagentMode::Workspace,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let (system, _) = &provider.seen.lock().unwrap()[0];
        assert!(
            system
                .as_deref()
                .unwrap()
                .contains("autonomous workspace subagent")
        );
        assert!(
            system
                .as_deref()
                .unwrap()
                .contains("Complete the task with the available tools")
        );
    }

    #[test]
    fn request_note_preserves_a_trailing_user_role() {
        let mut history = vec![Message::user("task")];
        push_request_note(&mut history, "final report now");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[0].content.len(), 2);
    }

    #[tokio::test]
    async fn final_budget_turn_disables_tools_and_requests_a_report() {
        // The model keeps asking for a tool until the final reserved turn,
        // where the request must expose no tools and explicitly demand the
        // best report from evidence already gathered.
        let max_turns = SubagentPolicy::default().max_turns;
        let mut scripts = Vec::new();
        for _ in 0..max_turns - 1 {
            scripts.push(vec![
                Ok(StreamEvent::ToolCallComplete(llm::ToolCall {
                    id: format!("c{}", rand_suffix()),
                    name: "missing".into(),
                    arguments: json!({}),
                })),
                ScriptProvider::done("tool_calls"),
            ]);
        }
        scripts.push(vec![
            Ok(StreamEvent::TextDelta("best available report".into())),
            ScriptProvider::done("stop"),
        ]);
        let provider = Arc::new(ScriptProvider::new(scripts));
        let runner = runner_with(provider.clone());
        let report = runner
            .run(
                "loop",
                "keep exploring",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(report, "best available report");
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), max_turns);
        assert!(seen.last().unwrap().1.is_empty(), "final request has tools");
    }

    fn rand_suffix() -> usize {
        use std::sync::atomic::Ordering;
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    /// A provider whose stream never yields; cancellation must break the
    /// hang.
    struct HangingProvider;

    #[async_trait]
    impl Provider for HangingProvider {
        fn name(&self) -> &str {
            "hang"
        }
        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            Ok(Box::pin(stream::unfold((), |state| async move {
                let _ = state;
                futures_util::future::pending::<()>().await;
                None::<(Result<StreamEvent, LlmError>, ())>
            })))
        }
        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn cancellation_surfaces_as_an_error() {
        let runner: SubagentRunnerImpl = runner_with_hanging();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = runner.run("x", "y", SubagentMode::ReadOnly, cancel).await;
        assert_eq!(outcome.unwrap_err(), "cancelled by user");
    }

    /// Like [`runner_with`] but for any provider double.
    fn runner_with_dyn(provider: Arc<dyn Provider>) -> SubagentRunnerImpl {
        let workspace = tempfile::tempdir().unwrap();
        let root = std::mem::ManuallyDrop::new(workspace);
        SubagentRunnerImpl::new(
            provider,
            "demo",
            root.path().canonicalize().unwrap(),
            false,
            "",
            SubagentPolicy::default(),
            None,
            None,
        )
    }

    fn runner_with_hanging() -> SubagentRunnerImpl {
        let workspace = tempfile::tempdir().unwrap();
        let root = std::mem::ManuallyDrop::new(workspace);
        SubagentRunnerImpl::new(
            Arc::new(HangingProvider),
            "demo",
            root.path().canonicalize().unwrap(),
            false,
            "",
            SubagentPolicy::default(),
            None,
            None,
        )
    }

    #[tokio::test]
    async fn disabled_runner_refuses_to_run() {
        let workspace = tempfile::tempdir().unwrap();
        let runner = SubagentRunnerImpl::new(
            Arc::new(ScriptProvider::new(Vec::new())),
            "demo",
            workspace.path(),
            false,
            "",
            SubagentPolicy {
                max_turns: 0,
                ..SubagentPolicy::default()
            },
            None,
            None,
        );
        let outcome = runner
            .run("x", "y", SubagentMode::ReadOnly, CancellationToken::new())
            .await;
        assert_eq!(outcome.unwrap_err(), "subagents are disabled");
    }

    #[tokio::test]
    async fn durable_child_session_links_to_parent_and_persists_transcript() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = session::SessionStore::new(
            root.path(),
            std::fs::canonicalize(workspace.path()).unwrap(),
        )
        .unwrap();
        let parent = store
            .create(session::SessionCreateOptions::default())
            .unwrap();

        let provider = Arc::new(ScriptProvider::new(vec![vec![
            Ok(StreamEvent::TextDelta("the audit found nothing".into())),
            ScriptProvider::done("stop"),
        ]]));
        let runner = SubagentRunnerImpl::new(
            provider,
            "demo",
            std::fs::canonicalize(workspace.path()).unwrap(),
            false,
            "",
            SubagentPolicy::default(),
            Some(store.clone()),
            Some(parent.id()),
        );
        let report = runner
            .run(
                "audit tui",
                "scan it",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(report, "the audit found nothing");

        // The child session exists in the same store, carries the task title
        // and parent link, and holds the transcript.
        let sessions = store.list().unwrap();
        assert_eq!(sessions.len(), 2, "parent plus one child");
        let child_entry = sessions
            .iter()
            .find(|entry| entry.id != parent.id())
            .expect("child session listed");
        assert_eq!(child_entry.title.as_deref(), Some("audit tui"));
        assert_eq!(child_entry.parent_session, Some(parent.id()));
        let child = store.open(&child_entry.id).unwrap();
        let texts: Vec<String> = child
            .events
            .iter()
            .filter_map(|record| match &record.event {
                session::SessionEvent::UserMessage { message } => {
                    message.content.first().and_then(|content| match content {
                        session::StoredContent::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                }
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|text| text.contains("scan it")));
    }

    #[tokio::test]
    async fn end_to_end_parent_agent_delegates_and_receives_report() {
        use crate::agent::InputMessage;
        use tools::{ToolConfig, default_registry};
        // Child script: one text-only reply (the report).
        let child_provider = Arc::new(ScriptProvider::new(vec![vec![
            Ok(StreamEvent::TextDelta("crate tui: 0 issues".into())),
            ScriptProvider::done("stop"),
        ]]));

        let workspace = tempfile::tempdir().unwrap();
        let root_canon = std::fs::canonicalize(workspace.path()).unwrap();
        let mut tools = default_registry(ToolConfig::new(&root_canon, false)).unwrap();
        tools
            .register_subagent(Arc::new(SubagentRunnerImpl::new(
                child_provider.clone(),
                "demo",
                root_canon.clone(),
                false,
                "",
                SubagentPolicy::default(),
                None,
                None,
            )))
            .unwrap();

        // Parent script: delegate, then finish after the tool result.
        let parent_provider = Arc::new(ScriptProvider::new(vec![
            vec![
                Ok(StreamEvent::ToolCallComplete(llm::ToolCall {
                    id: "t1".into(),
                    name: tools::SUBAGENT_TOOL_NAME.into(),
                    arguments: json!({
                        "description": "audit tui",
                        "prompt": "count issues in crates/tui"
                    }),
                })),
                ScriptProvider::done("tool_calls"),
            ],
            vec![
                Ok(StreamEvent::TextDelta("delegation done".into())),
                ScriptProvider::done("stop"),
            ],
        ]));

        let cancel = CancellationToken::new();
        let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        input_tx.send(InputMessage::Message("go".into())).unwrap();
        drop(input_tx);
        crate::agent::Agent::new(parent_provider.clone(), tools, "demo", cancel)
            .run(input_rx, event_tx)
            .await;

        // Drain events; the run must complete without hanging.
        let mut finished = false;
        while let Ok(event) = event_rx.try_recv() {
            if matches!(event, crate::agent::AgentEvent::TurnFinished) {
                finished = true;
            }
        }
        assert!(finished, "parent turn never finished");

        // The child provider served exactly one request (the delegation).
        assert_eq!(
            child_provider
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    // -------------------------------------------------------------- retries

    /// Provider double that records whether the child used the retrying
    /// entry point and serves one scripted result per *attempt*.
    struct RetryProbeProvider {
        /// Per-attempt outcomes: an `Err` simulates a transient failure that
        /// `stream_with_retry` should absorb.
        attempts: Mutex<Vec<Result<Vec<StreamEvent>, String>>>,
        attempt_count: std::sync::atomic::AtomicUsize,
        direct_stream_calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl Provider for RetryProbeProvider {
        fn name(&self) -> &str {
            "retry-probe"
        }

        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            self.direct_stream_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(LlmError::Stream("direct stream must not be used".into()))
        }

        async fn stream_with_retry(
            &self,
            _request: &CompletionRequest,
            _on_retry: RetryCallback,
        ) -> Result<EventStream, LlmError> {
            // Deterministic stand-in for the shared backoff: walk the
            // scripted attempts, absorbing transient failures exactly as the
            // real helper would, but with no wall-clock wait.
            loop {
                let index = self
                    .attempt_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let outcome = self
                    .attempts
                    .lock()
                    .unwrap()
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| Err("script exhausted".into()));
                match outcome {
                    Ok(events) => return Ok(Box::pin(stream::iter(events.into_iter().map(Ok)))),
                    Err(message) if index + 1 < self.attempts.lock().unwrap().len() => {
                        let _ = message;
                        continue;
                    }
                    Err(message) => {
                        return Err(LlmError::Http {
                            status: 500,
                            body: message,
                        });
                    }
                }
            }
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    fn report_done() -> StreamEvent {
        StreamEvent::Done {
            stop_reason: Some("stop".into()),
            usage: None,
        }
    }

    #[tokio::test]
    async fn child_uses_stream_with_retry_and_survives_a_transient_failure() {
        let provider = Arc::new(RetryProbeProvider {
            attempts: Mutex::new(vec![
                Err("transient 503".into()),
                Ok(vec![
                    StreamEvent::TextDelta("recovered report".into()),
                    report_done(),
                ]),
            ]),
            attempt_count: std::sync::atomic::AtomicUsize::new(0),
            direct_stream_calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let runner = runner_with_dyn(provider.clone());
        let report = runner
            .run("x", "y", SubagentMode::ReadOnly, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(report, "recovered report");
        assert_eq!(provider.attempt_count.load(Ordering::SeqCst), 2);
        assert_eq!(
            provider.direct_stream_calls.load(Ordering::SeqCst),
            0,
            "the child must go through stream_with_retry, never direct stream"
        );
    }

    /// A provider whose first `stream_with_retry` attempt never resolves,
    /// so cancellation must break the wait while the stream is being
    /// obtained (e.g. mid-backoff).
    struct HangOnFirstAttemptProvider {
        cancel_probe: CancellationToken,
    }

    #[async_trait]
    impl Provider for HangOnFirstAttemptProvider {
        fn name(&self) -> &str {
            "hang-retry"
        }

        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            unreachable!("direct stream must not be used")
        }

        async fn stream_with_retry(
            &self,
            _request: &CompletionRequest,
            _on_retry: RetryCallback,
        ) -> Result<EventStream, LlmError> {
            // Pend forever once cancelled so the *only* way this future can
            // resolve is the loop's cancel branch — no completion-order race.
            self.cancel_probe.cancelled().await;
            let never: std::convert::Infallible = std::future::pending().await;
            match never {}
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn cancellation_while_obtaining_the_stream_returns_promptly() {
        let cancel = CancellationToken::new();
        let provider = Arc::new(HangOnFirstAttemptProvider {
            cancel_probe: cancel.clone(),
        });
        let runner = runner_with_dyn(provider);
        // Cancel shortly after the run starts, while the provider future is
        // still pending.
        let cancel_task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                cancel.cancel();
            }
        });
        let outcome = runner.run("x", "y", SubagentMode::ReadOnly, cancel).await;
        cancel_task.await.unwrap();
        assert_eq!(outcome.unwrap_err(), "cancelled by user");
    }

    // --------------------------------------------- registry single-shot init

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_first_use_builds_each_registry_exactly_once() {
        let runner = Arc::new(runner_with(Arc::new(ScriptProvider::new(Vec::new()))));
        runner.test_registry_builds.store(0, Ordering::SeqCst);

        // Race eight first uses of each mode through get_or_init.
        let mut handles = Vec::new();
        for mode in [SubagentMode::ReadOnly, SubagentMode::Workspace] {
            for _ in 0..8 {
                let runner = runner.clone();
                handles.push(tokio::spawn(async move {
                    runner
                        .registry(mode)
                        .map(|registry| Arc::as_ptr(&registry) as usize)
                }));
            }
        }
        let mut pointers = Vec::new();
        for handle in handles {
            pointers.push(handle.await.unwrap().unwrap());
        }

        // All read-only callers got one Arc, all workspace callers got a
        // different single Arc, and only two constructions ever ran.
        let unique: std::collections::HashSet<usize> = pointers.into_iter().collect();
        assert_eq!(unique.len(), 2, "one distinct registry per mode");
        assert_eq!(
            runner.test_registry_builds.load(Ordering::SeqCst),
            2,
            "each mode's registry must be constructed exactly once"
        );
    }

    // ------------------------------------------------- model snapshot (T8)

    #[tokio::test]
    async fn update_model_retargests_future_children_without_touching_running_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = session::SessionStore::new(
            root.path(),
            std::fs::canonicalize(workspace.path()).unwrap(),
        )
        .unwrap();
        let parent = store
            .create(session::SessionCreateOptions::default())
            .unwrap();

        let report_a = || {
            vec![
                Ok(StreamEvent::TextDelta("report from a".into())),
                ScriptProvider::done("stop"),
            ]
        };
        let provider_a = Arc::new(ScriptProvider::new(vec![report_a()]));
        let provider_b = Arc::new(ScriptProvider::new(vec![vec![
            Ok(StreamEvent::TextDelta("report from b".into())),
            ScriptProvider::done("stop"),
        ]]));
        let runner = Arc::new(SubagentRunnerImpl::new(
            provider_a.clone(),
            "model-a",
            std::fs::canonicalize(workspace.path()).unwrap(),
            false,
            "",
            SubagentPolicy::default(),
            Some(store.clone()),
            Some(parent.id()),
        ));

        // Child one runs on provider/model A.
        let first = runner
            .run(
                "child one",
                "p",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(first, "report from a");

        // Parent switches to B; future children must follow.
        runner.update_model(provider_b.clone() as Arc<dyn Provider>, "model-b");
        let second = runner
            .run(
                "child two",
                "p",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(second, "report from b");

        assert_eq!(provider_a.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider_b.calls.load(Ordering::SeqCst), 1);

        // Each child session header recorded the snapshot it ran with.
        let entries = store.list().unwrap();
        let header_model = |title: &str| {
            entries
                .iter()
                .find(|entry| entry.title.as_deref() == Some(title))
                .and_then(|entry| entry.model.clone())
                .unwrap()
        };
        assert_eq!(header_model("child one"), "model-a");
        assert_eq!(header_model("child two"), "model-b");
    }

    #[tokio::test]
    async fn update_parent_session_retargets_future_child_lineage() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = session::SessionStore::new(
            root.path(),
            std::fs::canonicalize(workspace.path()).unwrap(),
        )
        .unwrap();
        let parent_a = store
            .create(session::SessionCreateOptions::default())
            .unwrap();
        let parent_b = store
            .create(session::SessionCreateOptions::default())
            .unwrap();
        let provider = Arc::new(ScriptProvider::new(vec![
            vec![
                Ok(StreamEvent::TextDelta("first".into())),
                ScriptProvider::done("stop"),
            ],
            vec![
                Ok(StreamEvent::TextDelta("second".into())),
                ScriptProvider::done("stop"),
            ],
        ]));
        let runner = SubagentRunnerImpl::new(
            provider,
            "demo",
            std::fs::canonicalize(workspace.path()).unwrap(),
            false,
            "",
            SubagentPolicy::default(),
            Some(store.clone()),
            Some(parent_a.id()),
        );

        runner
            .run(
                "child a",
                "p",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        runner.update_parent_session(Some(parent_b.id()));
        runner
            .run(
                "child b",
                "p",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let entries = store.list().unwrap();
        let parent_for = |title: &str| {
            entries
                .iter()
                .find(|entry| entry.title.as_deref() == Some(title))
                .and_then(|entry| entry.parent_session)
        };
        assert_eq!(parent_for("child a"), Some(parent_a.id()));
        assert_eq!(parent_for("child b"), Some(parent_b.id()));
    }

    // ------------------------------------- durable child transcript + usage

    /// Full durability round trip: real tool call in the child, usage-bearing
    /// `Done` events on both rounds, then reload/export assertions covering
    /// exact-once call persistence, call/result pairing, usage totals, and
    /// structural validity.
    #[tokio::test]
    async fn durable_child_persists_one_call_per_id_usage_and_valid_history() {
        use session::{ExportOptions, StoredContent, export_jsonl, snapshot_entries};

        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let ws_root = std::fs::canonicalize(workspace.path()).unwrap();
        std::fs::write(ws_root.join("lib.rs"), "fn main() {}\n").unwrap();
        let store = session::SessionStore::new(root.path(), ws_root.clone()).unwrap();
        let parent = store
            .create(session::SessionCreateOptions::default())
            .unwrap();

        let provider = Arc::new(ScriptProvider::new(vec![
            vec![
                Ok(StreamEvent::ToolCallComplete(llm::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: json!({"path": "lib.rs"}),
                })),
                Ok(StreamEvent::Done {
                    stop_reason: Some("tool_calls".into()),
                    usage: Some(llm::Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                        cached_tokens: None,
                        reasoning_tokens: None,
                        cost: Some(0.01),
                    }),
                }),
            ],
            vec![
                Ok(StreamEvent::TextDelta("final report".into())),
                Ok(StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(llm::Usage {
                        input_tokens: 20,
                        output_tokens: 8,
                        cached_tokens: None,
                        reasoning_tokens: None,
                        cost: Some(0.02),
                    }),
                }),
            ],
        ]));
        let runner = SubagentRunnerImpl::new(
            provider,
            "demo",
            ws_root,
            false,
            "",
            SubagentPolicy::default(),
            Some(store.clone()),
            Some(parent.id()),
        );
        let report = runner
            .run(
                "durable task",
                "read lib.rs and report",
                SubagentMode::ReadOnly,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(report, "final report");

        let sessions = store.list().unwrap();
        let child_entry = sessions
            .iter()
            .find(|entry| entry.title.as_deref() == Some("durable task"))
            .expect("child session listed");
        // Reload validates the stored history without warnings or errors.
        let child = store.open(&child_entry.id).unwrap();

        // Exactly one logical ToolCall exists for c1, retaining name/args.
        let calls: Vec<&session::StoredToolCall> = child
            .events
            .iter()
            .filter_map(|record| match &record.event {
                session::SessionEvent::ToolCall { call } => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "call must be persisted exactly once");
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments, json!({"path": "lib.rs"}));

        // No AssistantMessage carries embedded tool calls anymore.
        assert!(child.events.iter().all(|record| {
            match &record.event {
                session::SessionEvent::AssistantMessage { message } => message
                    .content
                    .iter()
                    .all(|content| !matches!(content, StoredContent::ToolCall { .. })),
                _ => true,
            }
        }));

        // Context reconstruction pairs one call with exactly one result.
        let messages = child.context_messages();
        let call_count = messages
            .iter()
            .flat_map(|message| &message.content)
            .filter(|content| matches!(content, Content::ToolCall(call) if call.id == "c1"))
            .count();
        let result_count = messages
            .iter()
            .flat_map(|message| &message.content)
            .filter(|content| {
                matches!(content, Content::ToolResult { tool_call_id, .. } if tool_call_id == "c1")
            })
            .count();
        assert_eq!(call_count, 1);
        assert_eq!(result_count, 1);

        // Snapshot pairs the same way and names the actual tool.
        let tools: Vec<String> = snapshot_entries(&child)
            .into_iter()
            .filter_map(|entry| match entry {
                session::SessionSnapshotEntry::Tool { name, .. } => Some(name),
                _ => None,
            })
            .collect::<Vec<String>>();
        assert_eq!(tools, vec!["read"]);

        // Usage from both rounds aggregates into the child metadata only.
        assert_eq!(child.metadata.usage.input_tokens, 30);
        assert_eq!(child.metadata.usage.output_tokens, 13);
        assert!((child.metadata.usage.cost - 0.03).abs() < 1e-9);

        // Export names the actual tool instead of an orphan generic `tool`.
        let destination = root.path().join("child-export.jsonl");
        let exported = export_jsonl(&child, Some(&destination), &ExportOptions::default()).unwrap();
        let text = std::fs::read_to_string(exported).unwrap();
        assert!(text.contains("\"read\""), "export lost the tool name");
        assert!(text.contains("\"c1\""));
    }
}
