use crate::config::{build_provider, save_settings};
use crate::prompt::system_prompt_with_tools;
use crate::tools::{ToolRegistry, call_recap, call_summary};
use futures_util::StreamExt;
use llm::{
    CompletionRequest, Content, LlmError, Message, Provider, RetryCallback, Role, StreamEvent,
    ToolCall, truncate_utf8,
};
use session::{
    CompactionPolicy, ExportOptions, Session, SessionCreateOptions, SessionEvent, SessionStore,
    StoredMessage, StoredToolCall, export_jsonl, snapshot_entries, usage_summary,
};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

/// Maximum number of times a turn re-streams after a tool call whose arguments
/// fail to parse. A model that keeps truncating its output should eventually
/// give up instead of looping forever.
const MAX_PARSE_RETRIES: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStarted {
        name: String,
        summary: String,
        /// Pretty-printed and bounded before it is sent to the serde-free TUI.
        arguments: String,
    },
    ToolCallFinished {
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
    },
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
        arguments: String,
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
    /// Input messages received while a turn is running.  They are drained by
    /// `run` before the next turn starts.
    queued: VecDeque<InputMessage>,
    /// Whether the input channel has not yet been observed closed.  Once
    /// closed, the run loop stops selecting on it (a closed `recv` would
    /// otherwise complete immediately and starve the stream polling).
    input_open: bool,
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
            queued: VecDeque::new(),
            input_open: true,
        }
    }

    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    /// Attach a loaded/new durable session.  Active provider/model selection
    /// remains the caller's choice; saved metadata is informational only.
    pub fn with_session(mut self, store: SessionStore, mut session: Session) -> Self {
        if let Err(error) = store.repair_incomplete_tool_calls(&mut session) {
            tracing::warn!(error = %error, "could not repair incomplete session tool calls");
        }
        self.history = session.context_messages();
        self.session = Some(AgentSessionState { store, session });
        self
    }

    pub fn session(&self) -> Option<&Session> {
        self.session.as_ref().map(|state| &state.session)
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
        calls: &[ToolCall],
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> bool {
        if reasoning.is_empty() && text.is_empty() && calls.is_empty() {
            return true;
        }
        let mut content = Vec::new();
        if !reasoning.is_empty() {
            content.push(Content::Reasoning(reasoning.to_owned()));
        }
        if !text.is_empty() {
            content.push(Content::Text(text.to_owned()));
        }
        // Tool calls have their own durable events.  Keeping them out of this
        // message avoids duplicate calls while still making the event stream
        // explicit and easy to inspect/export.
        let message = Message::assistant(content);
        if !content_is_empty(&message)
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
                    match self.run_turn(text, &events, &mut input, &turn_cancel).await {
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
                    self.handle_compact_session(&events);
                    continue;
                }
                InputMessage::SetModel { provider, model } => {
                    self.handle_set_model(provider, model, &events);
                    continue;
                }
                InputMessage::ListModels { provider } => {
                    self.handle_list_models(provider, &events);
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
        let user_message = Message::user(user_text);
        self.history.push(user_message.clone());
        if !self.persist_user_message(&user_message, events) {
            send(events, AgentEvent::TurnFinished);
            return Err(TurnError::Persist("user message".into()));
        }
        let mut parse_retries = 0;
        loop {
            let request = CompletionRequest {
                model: self.model.clone(),
                system: Some(system_prompt_with_tools(
                    &self.tools.workspace_root().display().to_string(),
                    &self.tools.prompt_context(),
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
                            Ok(StreamEvent::ToolCallComplete(call)) => tool_calls.push(call),
                            Ok(StreamEvent::Done { usage: done_usage, .. }) => {
                                if let Some(done_usage) = done_usage {
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
                append_assistant(&mut self.history, &reasoning, &text, tool_calls.clone());
                let _ = self.persist_assistant(&reasoning, &text, &tool_calls, events);
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

            append_assistant(&mut self.history, &reasoning, &text, tool_calls.clone());
            let _ = self.persist_assistant(&reasoning, &text, &tool_calls, events);

            if let Some(error) = stream_error {
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
                if let LlmError::Parse(parse_message) = error {
                    // A tool call whose arguments failed to parse (usually
                    // truncated JSON) never became durable, so the turn can
                    // retry without leaving dangling state. Nudge the model
                    // and re-stream instead of dead-ending the turn.
                    if parse_retries < MAX_PARSE_RETRIES {
                        parse_retries += 1;
                        let note = format!(
                            "[system note: your previous tool call had malformed JSON \
                             arguments and was not executed: {parse_message}. Re-issue the \
                             tool call with valid arguments.]"
                        );
                        // Appending into a trailing user message (instead of
                        // always pushing a new one) keeps provider role
                        // alternation valid across all dialects.
                        let tail_is_user = matches!(
                            self.history.last(),
                            Some(Message {
                                role: Role::User,
                                ..
                            })
                        );
                        if tail_is_user {
                            if let Some(Message {
                                role: Role::User,
                                content,
                            }) = self.history.last_mut()
                            {
                                content.push(Content::Text(note));
                            }
                        } else {
                            self.history.push(Message::user(note));
                        }
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
                send(events, AgentEvent::TurnFinished);
                return Ok(());
            }

            for call in tool_calls {
                let summary = call_summary(&call.name, &call.arguments);
                send(
                    events,
                    AgentEvent::ToolCallStarted {
                        name: call.name.clone(),
                        summary: summary.clone(),
                        arguments: call_recap(&call.name, &call.arguments),
                    },
                );
                let started = Instant::now();
                let result = {
                    let tool_future =
                        self.tools
                            .execute(&call.name, call.arguments.clone(), cancel.clone());
                    tokio::pin!(tool_future);
                    loop {
                        tokio::select! {
                            result = &mut tool_future => break Some(result),
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
                                cancel.cancel();
                                break None;
                            }
                        }
                    }
                };
                let Some(result) = result else {
                    let cancelled_output = "cancelled".to_owned();
                    self.history.push(Message::tool_result(
                        call.id.clone(),
                        cancelled_output.clone(),
                        true,
                    ));
                    let _ = self.persist_tool_result(&call, &cancelled_output, true, events);
                    send(
                        events,
                        AgentEvent::ToolCallFinished {
                            name: call.name.clone(),
                            summary,
                            ok: false,
                            duration_ms: started.elapsed().as_millis() as u64,
                            output: String::new(),
                            error: Some(cancelled_output),
                        },
                    );
                    self.persist_cancelled("tool execution interrupted", events);
                    send(events, AgentEvent::TurnFinished);
                    if self.cancel.is_cancelled() {
                        return Err(TurnError::Shutdown);
                    }
                    return Ok(());
                };
                let output = result.content.clone();
                let error = result.is_error.then(|| output.clone());
                send(
                    events,
                    AgentEvent::ToolCallFinished {
                        name: call.name.clone(),
                        summary: result.summary.clone(),
                        ok: !result.is_error,
                        duration_ms: started.elapsed().as_millis() as u64,
                        output,
                        error,
                    },
                );
                let result_content = result.content.clone();
                let result_is_error = result.is_error;
                self.history.push(Message {
                    role: Role::Tool,
                    content: vec![Content::ToolResult {
                        tool_call_id: call.id.clone(),
                        content: result_content.clone(),
                        is_error: result_is_error,
                    }],
                });
                let _ = self.persist_tool_result(&call, &result_content, result_is_error, events);
            }

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
        let title = session.metadata.title.clone();
        self.history.clear();
        self.session = Some(crate::agent::AgentSessionState { store, session });
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
        if let Err(error) = store.set_current(&session) {
            send(
                events,
                AgentEvent::Error(format!("could not update current session: {error}")),
            );
            return;
        }
        let id = session.id().to_string();
        let title = session.metadata.title.clone();
        self.history = session.context_messages();
        let snapshot = ui_snapshot_entries(snapshot_entries(&session));
        self.session = Some(crate::agent::AgentSessionState { store, session });
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

    fn handle_compact_session(&mut self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(state) = self.session.as_mut() else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        let policy = CompactionPolicy::default();
        let Some(result) = state.session.compact(&policy) else {
            send(
                events,
                AgentEvent::Notice("session does not need compaction yet".into()),
            );
            return;
        };
        let summary_bytes = result.summary.len();
        match state.store.append_event(
            &mut state.session,
            SessionEvent::CompactionSummary {
                summary: result.summary,
                compacted_through: result.compacted_through,
            },
        ) {
            Ok(_) => send(
                events,
                AgentEvent::CompactionFinished {
                    compacted_through: result.compacted_through,
                    summary_bytes,
                },
            ),
            Err(error) => send(
                events,
                AgentEvent::Error(format!("could not compact session: {error}")),
            ),
        }
    }

    fn handle_set_model(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        self.handle_set_model_with_factory(provider, model, events, Box::new(build_provider));
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
        let provider = match build_provider(&provider_name) {
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
/// tool summaries and per-tool recap strings.  The session crate owns the
/// event walk and tool-result pairing; this conversion is purely
/// presentational.
///
/// The recap is display-only: raw JSON arguments stay in the session store and
/// in `context_messages`, so loaded and continued sessions keep full fidelity.
/// Do not persist these strings back into session events.
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
                let recap = call_recap(&name, &arguments);
                SessionSnapshotEntry::Tool {
                    name,
                    summary,
                    arguments: recap,
                    ok,
                    duration_ms: 0,
                    output,
                    error,
                }
            }
        })
        .collect()
}

fn content_is_empty(message: &Message) -> bool {
    message.content.is_empty()
}

fn append_assistant(history: &mut Vec<Message>, reasoning: &str, text: &str, calls: Vec<ToolCall>) {
    if reasoning.is_empty() && text.is_empty() && calls.is_empty() {
        return;
    }
    let mut content = Vec::new();
    if !reasoning.is_empty() {
        content.push(Content::Reasoning(reasoning.to_owned()));
    }
    if !text.is_empty() {
        content.push(Content::Text(text.to_owned()));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;
    use async_trait::async_trait;
    use futures_util::stream;
    use llm::{EventStream, LlmError, ModelInfo, Usage};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    /// One step of a canned provider script.  Errors are carried as strings
    /// because `LlmError` embeds a non-cloneable `reqwest::Error`; `stream`
    /// maps them back into `LlmError::Stream`.
    type ScriptStep = Result<StreamEvent, String>;

    /// How mock stream errors are wrapped; lets tests exercise the
    /// parse-error recovery path specifically.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum MockErrorKind {
        Stream,
        Parse,
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
            Ok(Box::pin(stream::iter(script.into_iter().map(move |step| {
                step.map_err(|message| match error_kind {
                    MockErrorKind::Stream => LlmError::Stream(message),
                    MockErrorKind::Parse => LlmError::Parse(message),
                })
            }))))
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
                AgentEvent::ToolCallStarted {
                    arguments,
                    ..
                } if arguments == "missing"
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
                        is_error,
                        content,
                        ..
                    } => Some((is_error, content)),
                    _ => None,
                })
                .collect();
            assert_eq!(tool_results.len(), 1, "expected one persisted tool result");
            let (is_error, content) = tool_results[0];
            assert!(is_error, "interrupted tool call must be persisted as an error");
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
}
