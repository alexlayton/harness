use crate::tools::ToolRegistry;
use auth::CopilotAuth;
use compact::CompactionPolicy;
use llm::{Message, Provider};
use session::{Session, SessionStore, snapshot_entries};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

#[cfg(test)]
use crate::tools::{Concurrency, ToolOutput};
#[cfg(test)]
use llm::{CompletionRequest, Content, Role, StreamEvent, ToolCall};
#[cfg(test)]
use session::{SessionCreateOptions, SessionEvent, StoredMessage};
#[cfg(test)]
use std::time::Instant;

mod commands;
mod compaction;
mod events;
mod persistence;
mod tool_dispatch;
mod turn;

pub use commands::spawn_model_list;
pub use events::{AgentEvent, CompactionReason, SessionListItem, SessionSnapshotEntry, TurnError};
pub use persistence::AgentSessionState;
use persistence::{ui_snapshot_entries, usage_event};
pub use tool_dispatch::SubagentLimits;
pub(crate) use tool_dispatch::plan_tool_batches;

/// Maximum number of times a turn re-streams after a recoverable failure:
/// malformed tool-call arguments, a retryable mid-stream error, or an empty
/// response. A model that keeps failing should eventually give up instead of
/// looping forever.
const MAX_TURN_RECOVERIES: usize = 3;

/// Maximum emergency compactions performed per turn in response to a
/// context-overflow rejection. Each round is an extra summarizer call plus a
/// retried request, so it is bounded separately from `MAX_TURN_RECOVERIES`.
const MAX_OVERFLOW_RECOVERIES: usize = 2;

/// Stateful orchestrator for provider turns, tools, and optional sessions.
///
/// `history` remains public for compatibility. Mutating it while `session` is
/// attached can break the live-history/durable-session relationship and should
/// be avoided outside migration code.
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

impl Agent {
    /// Create an in-memory agent; attach optional policy and session builders
    /// before starting its input loop.
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

    /// Attach the shared Copilot authentication handle for provider switches.
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
}

impl Agent {
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
                        // Persistence errors are emitted at their source; the
                        // outer boundary owns the single terminal event.
                        Err(TurnError::Persist(_)) => send(&events, AgentEvent::TurnFinished),
                        Ok(()) => {}
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
                    if let Err(TurnError::Persist(_)) =
                        self.handle_compact_session(&events, &cancel).await
                    {
                        send(&events, AgentEvent::TurnFinished);
                    }
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
                InputMessage::SubscriptionUsage => {
                    self.handle_subscription_usage(&events);
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
}

fn send(events: &mpsc::UnboundedSender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
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
    fn persistence_failure_aborts_before_requesting_a_completion() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = store.create(SessionCreateOptions::default()).unwrap();
            let path = session.file_path().unwrap().to_path_buf();
            std::fs::remove_file(&path).unwrap();
            let provider = Arc::new(MockProvider {
                calls: AtomicUsize::new(0),
                scripts: vec![],
                error_kind: MockErrorKind::Stream,
            });
            let (input_tx, input_rx) = mpsc::unbounded_channel();
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            input_tx
                .send(InputMessage::Message("hello".into()))
                .unwrap();
            drop(input_tx);

            Agent::new(
                provider.clone(),
                ToolRegistry::empty(),
                "demo",
                CancellationToken::new(),
            )
            .with_session(store, session)
            .run(input_rx, event_tx)
            .await;

            let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, AgentEvent::Error(_)))
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, AgentEvent::TurnFinished))
                    .count(),
                1
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
