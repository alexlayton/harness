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
//! - **Same provider/model as the parent** (v1 decision, mirroring the
//!   compaction summarizer). Child usage is persisted to the child session
//!   only, so parent `/usage` totals are not double-counted.
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
use crate::config::SubagentPolicy;
use crate::prompt::subagent_system_prompt;
use crate::tools::{SubagentRunner, ToolConfig, ToolRegistry, call_summary, default_registry};
use async_trait::async_trait;
use futures_util::stream::{FuturesUnordered, StreamExt};
use llm::{CompletionRequest, Content, Message, Provider, Role, StreamEvent, truncate_utf8};
use session::{SessionCreateOptions, SessionStore};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

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
type ChildToolRun<'a> =
    Pin<Box<dyn Future<Output = (usize, crate::tools::ToolOutput)> + Send + 'a>>;

/// One delegated subagent run.
pub(crate) struct SubagentRun {
    pub description: String,
    pub prompt: String,
}

/// Everything a nested loop needs from the host process. One instance is
/// shared by every subagent invocation of a session.
pub struct SubagentRunnerImpl {
    provider: Arc<dyn Provider>,
    model: String,
    workspace_root: PathBuf,
    rtk: bool,
    project_context: String,
    /// Resolved delegation bounds (`max_turns`; `0` disables subagents).
    config: SubagentPolicy,
    /// Store shared with the parent session (same workspace scope); `None`
    /// under `--no-session`, making runs ephemeral.
    store: Option<SessionStore>,
    /// Parent session id recorded on child sessions for lineage.
    parent_session_id: Option<session::SessionId>,
    /// Built lazily on first use: the FFF index walk should not cost anything
    /// when the model never delegates.
    child_registry: OnceLock<Arc<ToolRegistry>>,
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
        Self {
            provider,
            model: model.into(),
            workspace_root: workspace_root.into(),
            rtk,
            project_context: project_context.into(),
            config,
            store,
            parent_session_id,
            child_registry: OnceLock::new(),
        }
    }

    /// Build the child tool set once. `default_registry` contains exactly the
    /// built-in tools — read/edit/write/bash/find/grep — and no subagent
    /// tool, which is what makes recursion structurally impossible.
    fn registry(&self) -> Result<&Arc<ToolRegistry>, String> {
        if let Some(registry) = self.child_registry.get() {
            return Ok(registry);
        }
        let registry = Arc::new(
            default_registry(ToolConfig::new(&self.workspace_root, self.rtk))
                .map_err(|error| format!("could not build subagent tools: {error}"))?,
        );
        let _ = self.child_registry.set(registry);
        Ok(self
            .child_registry
            .get()
            .expect("registry was just inserted"))
    }

    /// Persist the child session header up front so even a hard crash leaves
    /// a resumable trace of the delegation.
    fn create_child_session(&self, description: &str) -> Option<session::Session> {
        let store = self.store.as_ref()?;
        match store.create(SessionCreateOptions {
            title: Some(description.to_owned()),
            provider: Some(self.provider.name().to_owned()),
            model: Some(self.model.clone()),
            parent_session: self.parent_session_id,
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
        cancel: CancellationToken,
    ) -> Result<String, String> {
        if self.config.max_turns == 0 {
            return Err("subagents are disabled".into());
        }
        let registry = self.registry()?;
        let mut history = vec![Message::user(run.prompt.clone())];
        let system = subagent_system_prompt(
            &self.workspace_root.display().to_string(),
            &registry.prompt_context(),
            registry.skills(),
            &self.project_context,
        );
        let mut session = self.create_child_session(&run.description);
        Self::persist(
            &self.store,
            &mut session,
            session::SessionEvent::UserMessage {
                message: session::StoredMessage::from_llm(&history[0]),
            },
        );
        let outcome = self
            .loop_turns(registry, &system, &mut history, &mut session, cancel)
            .await;
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
    async fn loop_turns(
        &self,
        registry: &ToolRegistry,
        system: &str,
        history: &mut Vec<Message>,
        session: &mut Option<session::Session>,
        cancel: CancellationToken,
    ) -> Result<String, String> {
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

            let request = CompletionRequest {
                model: self.model.clone(),
                system: Some(system.to_owned()),
                messages: history.clone(),
                tools: registry.definitions(),
                max_tokens: None,
                temperature: None,
                reasoning: true,
            };
            let mut stream = tokio::select! {
                stream = self.provider.stream(&request) => stream
                    .map_err(|error| format!("provider error: {error}"))?,
                _ = cancel.cancelled() => return Err("cancelled by user".into()),
            };

            let mut text = String::new();
            let mut tool_calls = Vec::<llm::ToolCall>::new();
            let mut stream_error = None;
            loop {
                tokio::select! {
                    next = stream.next() => match next {
                        Some(Ok(StreamEvent::TextDelta(delta))) => text.push_str(&delta),
                        Some(Ok(StreamEvent::ReasoningDelta(_))) => {}
                        Some(Ok(StreamEvent::ToolCallComplete(call))) => tool_calls.push(call),
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

            append_assistant(history, &text, tool_calls.clone());
            for event in assistant_events(&text, &tool_calls) {
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
            let mut slots: Vec<Option<crate::tools::ToolOutput>> =
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
                        crate::tools::ToolOutput {
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
        cancel: CancellationToken,
    ) -> Result<String, String> {
        let started = Instant::now();
        let outcome = self
            .execute(
                &SubagentRun {
                    description: description.to_owned(),
                    prompt: prompt.to_owned(),
                },
                cancel,
            )
            .await;
        tracing::info!(
            duration_ms = started.elapsed().as_millis() as u64,
            ok = outcome.is_ok(),
            "subagent finished"
        );
        outcome
    }
}

/// Append an assistant message (text plus any tool calls) to history.
fn append_assistant(history: &mut Vec<Message>, text: &str, calls: Vec<llm::ToolCall>) {
    if text.is_empty() && calls.is_empty() {
        return;
    }
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(Content::Text(text.to_owned()));
    }
    content.extend(calls.into_iter().map(Content::ToolCall));
    history.push(Message {
        role: Role::Assistant,
        content,
    });
}

/// Build the durable events for one assistant exchange: the message event
/// followed by one event per tool call, mirroring the parent agent's split
/// so exports stay uniform.
fn assistant_events(text: &str, calls: &[llm::ToolCall]) -> Vec<session::SessionEvent> {
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(Content::Text(text.to_owned()));
    }
    content.extend(calls.iter().cloned().map(Content::ToolCall));
    let mut events = vec![session::SessionEvent::AssistantMessage {
        message: session::StoredMessage::from_llm(&Message::assistant(content)),
    }];
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
            .run("audit", "scan the crate", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(report, "found 3 issues");
        // The child saw a fresh one-message context with the subagent prompt.
        let (system, tools) = &provider.seen.lock().unwrap()[0];
        assert!(system.as_deref().unwrap().contains("autonomous subagent"));
        assert!(!tools.iter().any(|name| name == "subagent"), "{tools:?}");
    }

    #[tokio::test]
    async fn tool_round_trip_then_final_report() {
        let provider = Arc::new(ScriptProvider::new(vec![
            vec![
                Ok(StreamEvent::ToolCallComplete(llm::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: json!({"path": "lib.rs"}),
                })),
                ScriptProvider::done("tool_calls"),
            ],
            vec![
                Ok(StreamEvent::TextDelta("report after reading".into())),
                ScriptProvider::done("stop"),
            ],
        ]));
        let runner = runner_with(provider);
        let report = runner
            .run(
                "read task",
                "read lib.rs then report",
                CancellationToken::new(),
            )
            .await;
        // The read tool fails (file absent in the temp workspace) but that is
        // fine: what matters is history stayed valid for the second request.
        assert!(report.is_ok());
    }

    #[tokio::test]
    async fn turn_budget_exhaustion_returns_partial_findings() {
        // Every reply demands another tool call; the budget must end the loop
        // gracefully instead of spinning forever.
        let call_script = || {
            vec![
                Ok(StreamEvent::ToolCallComplete(llm::ToolCall {
                    id: format!("c{}", rand_suffix()),
                    name: "bash".into(),
                    arguments: json!({"command": "true"}),
                })),
                ScriptProvider::done("tool_calls"),
            ]
        };
        let mut scripts = Vec::new();
        for _ in 0..SubagentPolicy::default().max_turns + 1 {
            scripts.push(call_script());
        }
        let provider = Arc::new(ScriptProvider::new(scripts));
        let runner = runner_with(provider);
        let report = runner
            .run("loop", "never finish", CancellationToken::new())
            .await
            .unwrap();
        assert!(report.contains("without a final report"), "{report}");
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
        let outcome = runner.run("x", "y", cancel).await;
        assert_eq!(outcome.unwrap_err(), "cancelled by user");
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
        let outcome = runner.run("x", "y", CancellationToken::new()).await;
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
            Some(parent.id().clone()),
        );
        let report = runner
            .run("audit tui", "scan it", CancellationToken::new())
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
        use crate::agent::Agent;
        use crate::tools::{ToolConfig, default_registry};
        use tui::InputMessage;
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
                    name: crate::tools::SUBAGENT_TOOL_NAME.into(),
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
}
