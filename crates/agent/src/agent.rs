use crate::config::{build_provider, save_settings};
use crate::prompt::system_prompt_with_tools;
use crate::tools::{ToolRegistry, call_summary};
use futures_util::StreamExt;
use llm::{
    CompletionRequest, Content, Message, Provider, RetryCallback, Role, StreamEvent, ToolCall,
};
use session::{
    CompactionPolicy, ExportOptions, Session, SessionCreateOptions, SessionEvent, SessionStore,
    StoredContent, StoredMessage, StoredToolCall, export_jsonl, usage_summary,
};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

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
        let mut queued = VecDeque::new();
        let mut input_open = true;
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
                    entries: snapshot_entries(&session.session),
                },
            );
            send(&events, usage_event(&session.session.metadata.usage));
        }

        loop {
            let next_message = if let Some(message) = queued.pop_front() {
                Some(message)
            } else if !input_open {
                None
            } else {
                tokio::select! {
                    message = input.recv() => {
                        if message.is_none() {
                            input_open = false;
                        }
                        message
                    }
                    _ = self.cancel.cancelled() => None,
                }
            };
            let Some(message) = next_message else {
                break;
            };
            let user_text = match message {
                InputMessage::Message(text) if !text.trim().is_empty() => text,
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

            let user_message = Message::user(user_text);
            self.history.push(user_message.clone());
            if !self.persist_user_message(&user_message, &events) {
                send(&events, AgentEvent::TurnFinished);
                continue;
            }
            let turn_cancel = CancellationToken::new();
            let mut iteration = 0u32;
            let mut end_turn = false;

            while !end_turn {
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
                        message = input.recv(), if input_open => match message {
                            Some(InputMessage::Interrupt) => {
                                turn_cancel.cancel();
                                break None;
                            }
                            Some(message) => queued.push_back(message),
                            None => input_open = false,
                        },
                        _ = turn_cancel.cancelled() => break None,
                        _ = self.cancel.cancelled() => {
                            self.persist_cancelled("application shutdown", &events);
                            send(&events, AgentEvent::TurnFinished);
                            return;
                        }
                    }
                };
                let Some(stream_result) = stream_result else {
                    self.persist_cancelled("turn interrupted before response", &events);
                    send(&events, AgentEvent::TurnFinished);
                    end_turn = true;
                    continue;
                };
                let mut stream = match stream_result {
                    Ok(stream) => stream,
                    Err(error) => {
                        let message = error.to_string();
                        let _ = self.persist_event(
                            SessionEvent::Error {
                                message: message.clone(),
                            },
                            &events,
                        );
                        send(&events, AgentEvent::Error(message));
                        send(&events, AgentEvent::TurnFinished);
                        end_turn = true;
                        continue;
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
                                    send(&events, AgentEvent::TextDelta(delta));
                                }
                                Ok(StreamEvent::ReasoningDelta(delta)) => {
                                    reasoning.push_str(&delta);
                                    send(&events, AgentEvent::ReasoningDelta(delta));
                                }
                                Ok(StreamEvent::ToolCallComplete(call)) => tool_calls.push(call),
                                Ok(StreamEvent::Done { usage: done_usage, .. }) => {
                                    if let Some(done_usage) = done_usage {
                                        let summary = usage_summary(&done_usage);
                                        let _ = self.persist_event(
                                            SessionEvent::Usage {
                                                usage: summary.clone(),
                                            },
                                            &events,
                                        );
                                        if let Some(state) = self.session.as_ref() {
                                            send(&events, usage_event(&state.session.metadata.usage));
                                        } else {
                                            send(&events, usage_event(&summary));
                                        }
                                    }
                                }
                                Err(error) => {
                                    stream_error = Some(error.to_string());
                                    break;
                                }
                            }
                        }
                        message = input.recv(), if input_open => match message {
                            Some(InputMessage::Interrupt) => {
                                turn_cancel.cancel();
                                cancelled = true;
                                break;
                            }
                            Some(message) => queued.push_back(message),
                            None => input_open = false,
                        },
                        _ = turn_cancel.cancelled() => {
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
                    let _ = self.persist_assistant(&reasoning, &text, &tool_calls, &events);
                    for call in &tool_calls {
                        let cancelled_result = "cancelled before tool execution";
                        self.history.push(Message::tool_result(
                            call.id.clone(),
                            cancelled_result,
                            true,
                        ));
                        let _ = self.persist_tool_result(call, cancelled_result, true, &events);
                    }
                    self.persist_cancelled("turn interrupted", &events);
                    send(&events, AgentEvent::TurnFinished);
                    if self.cancel.is_cancelled() {
                        return;
                    }
                    end_turn = true;
                    continue;
                }

                append_assistant(&mut self.history, &reasoning, &text, tool_calls.clone());
                let _ = self.persist_assistant(&reasoning, &text, &tool_calls, &events);

                if let Some(error) = stream_error {
                    for call in &tool_calls {
                        let error_result = format!("provider stream interrupted: {error}");
                        self.history.push(Message::tool_result(
                            call.id.clone(),
                            error_result.clone(),
                            true,
                        ));
                        let _ = self.persist_tool_result(call, &error_result, true, &events);
                    }
                    let _ = self.persist_event(
                        SessionEvent::Error {
                            message: error.clone(),
                        },
                        &events,
                    );
                    send(&events, AgentEvent::Error(error));
                    send(&events, AgentEvent::TurnFinished);
                    end_turn = true;
                    continue;
                }

                if tool_calls.is_empty() {
                    send(&events, AgentEvent::TurnFinished);
                    end_turn = true;
                    continue;
                }

                iteration += 1;
                for call in tool_calls {
                    let summary = call_summary(&call.name, &call.arguments);
                    send(
                        &events,
                        AgentEvent::ToolCallStarted {
                            name: call.name.clone(),
                            summary: summary.clone(),
                            arguments: format_tool_arguments(&call.arguments),
                        },
                    );
                    let started = Instant::now();
                    let result = {
                        let tool_future = self.tools.execute(
                            &call.name,
                            call.arguments.clone(),
                            turn_cancel.clone(),
                        );
                        tokio::pin!(tool_future);
                        loop {
                            tokio::select! {
                                result = &mut tool_future => break Some(result),
                                message = input.recv(), if input_open => match message {
                                    Some(InputMessage::Interrupt) => {
                                        turn_cancel.cancel();
                                        break None;
                                    }
                                    Some(message) => queued.push_back(message),
                                    None => input_open = false,
                                },
                                _ = turn_cancel.cancelled() => break None,
                                _ = self.cancel.cancelled() => {
                                    turn_cancel.cancel();
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
                        let _ = self.persist_tool_result(&call, &cancelled_output, true, &events);
                        send(
                            &events,
                            AgentEvent::ToolCallFinished {
                                name: call.name.clone(),
                                summary,
                                ok: false,
                                duration_ms: started.elapsed().as_millis() as u64,
                                output: String::new(),
                                error: Some(cancelled_output),
                            },
                        );
                        self.persist_cancelled("tool execution interrupted", &events);
                        send(&events, AgentEvent::TurnFinished);
                        if self.cancel.is_cancelled() {
                            return;
                        }
                        end_turn = true;
                        break;
                    };
                    let output = result.content.clone();
                    let error = result.is_error.then(|| output.clone());
                    send(
                        &events,
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
                    let _ =
                        self.persist_tool_result(&call, &result_content, result_is_error, &events);
                }

                if !end_turn && iteration >= 100 {
                    let note = "max tool iterations reached";
                    self.history.push(Message::user(note));
                    let _ = self.persist_event(
                        SessionEvent::Error {
                            message: note.into(),
                        },
                        &events,
                    );
                    send(&events, AgentEvent::TextDelta(note.into()));
                    send(&events, AgentEvent::TurnFinished);
                    end_turn = true;
                }
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
        let snapshot = snapshot_entries(&session);
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
        self.handle_set_model_with_factory(provider, model, events, build_provider);
    }

    fn handle_set_model_with_factory<F>(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        factory: F,
    ) where
        F: Fn(&str) -> anyhow::Result<Arc<dyn Provider>>,
    {
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

fn spawn_model_list(
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

fn snapshot_entries(session: &Session) -> Vec<SessionSnapshotEntry> {
    let mut entries = Vec::new();
    let mut tool_indices = HashMap::<String, usize>::new();

    for record in &session.events {
        match &record.event {
            SessionEvent::UserMessage { message } => {
                let text = message.content.iter().find_map(|content| match content {
                    StoredContent::Text { text } => Some(text.clone()),
                    _ => None,
                });
                if let Some(text) = text.filter(|text| !text.is_empty()) {
                    entries.push(SessionSnapshotEntry::User { text });
                }
            }
            SessionEvent::AssistantMessage { message } => {
                let mut markdown = String::new();
                let mut reasoning = String::new();
                for content in &message.content {
                    match content {
                        StoredContent::Text { text } => {
                            if !markdown.is_empty() {
                                markdown.push('\n');
                            }
                            markdown.push_str(text);
                        }
                        StoredContent::Reasoning { text } => {
                            if !reasoning.is_empty() {
                                reasoning.push('\n');
                            }
                            reasoning.push_str(text);
                        }
                        StoredContent::ToolCall { .. } | StoredContent::ToolResult { .. } => {}
                    }
                }
                if !markdown.is_empty() || !reasoning.is_empty() {
                    entries.push(SessionSnapshotEntry::Assistant {
                        markdown,
                        reasoning,
                    });
                }
            }
            SessionEvent::ToolCall { call } => {
                let index = entries.len();
                tool_indices.insert(call.id.clone(), index);
                entries.push(SessionSnapshotEntry::Tool {
                    name: call.name.clone(),
                    summary: call_summary(&call.name, &call.arguments),
                    arguments: format_tool_arguments(&call.arguments),
                    ok: false,
                    duration_ms: 0,
                    output: String::new(),
                    error: None,
                });
            }
            SessionEvent::ToolResult {
                tool_call_id,
                content,
                is_error,
                tool_name,
            } => {
                if let Some(index) = tool_indices.get(tool_call_id).copied()
                    && let Some(SessionSnapshotEntry::Tool {
                        name,
                        summary,
                        ok,
                        output,
                        error,
                        ..
                    }) = entries.get_mut(index)
                {
                    if let Some(tool_name) = tool_name {
                        *name = tool_name.clone();
                    }
                    if summary.is_empty() {
                        *summary = name.clone();
                    }
                    *ok = !is_error;
                    *output = content.clone();
                    *error = is_error.then(|| content.clone());
                } else {
                    entries.push(SessionSnapshotEntry::Tool {
                        name: tool_name.clone().unwrap_or_else(|| "tool".into()),
                        summary: tool_name.clone().unwrap_or_else(|| "tool".into()),
                        arguments: "{}".into(),
                        ok: !is_error,
                        duration_ms: 0,
                        output: content.clone(),
                        error: is_error.then(|| content.clone()),
                    });
                }
            }
            SessionEvent::Reasoning { text } => {
                if let Some(SessionSnapshotEntry::Assistant { reasoning, .. }) = entries.last_mut()
                {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(text);
                } else {
                    entries.push(SessionSnapshotEntry::Assistant {
                        markdown: String::new(),
                        reasoning: text.clone(),
                    });
                }
            }
            SessionEvent::CompactionSummary { summary, .. } => {
                entries.push(SessionSnapshotEntry::Assistant {
                    markdown: format!("_Session summary:_\n\n{summary}"),
                    reasoning: String::new(),
                });
            }
            SessionEvent::ModelChange { .. }
            | SessionEvent::Usage { .. }
            | SessionEvent::MetadataChange { .. }
            | SessionEvent::TurnCancelled { .. }
            | SessionEvent::Error { .. }
            | SessionEvent::Unknown { .. } => {}
        }
    }

    entries
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

const MAX_TOOL_ARGUMENT_BYTES: usize = 2 * 1024;

fn format_tool_arguments(arguments: &serde_json::Value) -> String {
    let formatted =
        serde_json::to_string_pretty(arguments).unwrap_or_else(|_| arguments.to_string());
    cap_utf8(&formatted, MAX_TOOL_ARGUMENT_BYTES)
}

fn cap_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let suffix = "…";
    let mut end = max_bytes.saturating_sub(suffix.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{suffix}", &value[..end])
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

    struct MockProvider {
        calls: AtomicUsize,
        scripts: Vec<Vec<StreamEvent>>,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            let index = self.calls.fetch_add(1, Ordering::SeqCst);
            let script = self.scripts.get(index).cloned().unwrap_or_default();
            Ok(Box::pin(stream::iter(script.into_iter().map(Ok))))
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
            scripts: vec![vec![
                StreamEvent::TextDelta("hello".into()),
                StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(Usage::default()),
                },
            ]],
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
                scripts: vec![vec![
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
                ]],
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
                    vec![
                        StreamEvent::ToolCallComplete(ToolCall {
                            id: "c".into(),
                            name: "missing".into(),
                            arguments: json!({}),
                        }),
                        StreamEvent::Done {
                            stop_reason: Some("tool_calls".into()),
                            usage: None,
                        },
                    ],
                    vec![
                        StreamEvent::TextDelta("done".into()),
                        StreamEvent::Done {
                            stop_reason: Some("stop".into()),
                            usage: None,
                        },
                    ],
                ],
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
                AgentEvent::ToolCallStarted { arguments, .. } if arguments == "{}"
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
}
