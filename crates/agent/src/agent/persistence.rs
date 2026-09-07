use super::{Agent, AgentEvent, SessionSnapshotEntry, TurnError, send};
use llm::{Content, Message, ToolCall};
use session::{Session, SessionEvent, SessionStore, StoredMessage, StoredToolCall};
use tokio::sync::mpsc;
use tools::call_summary;

/// Durable session state owned by the agent. The TUI only receives status
/// events; it never reads or writes session files directly.
pub struct AgentSessionState {
    pub store: SessionStore,
    pub session: Session,
}

impl Agent {
    /// Stable id of the attached durable conversation, if any. Passed
    /// through on each request so providers with per-conversation
    /// accounting (OpenCode Go) can scope usage correctly. `None` leaves
    /// requests unscoped (ephemeral `--no-session` runs).
    pub(crate) fn session_id(&self) -> Option<String> {
        self.session
            .as_ref()
            .map(|state| state.session.id().to_string())
    }

    /// Append a history-bearing event, surfacing the failure to abort the turn.
    pub(crate) fn persist_event(
        &mut self,
        event: SessionEvent,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        let Some(state) = self.session.as_mut() else {
            return Ok(());
        };
        state
            .store
            .append_event(&mut state.session, event)
            .map(|_| ())
            .map_err(|error| {
                let message = format!("session persistence failed: {error}");
                send(events, AgentEvent::Error(message.clone()));
                TurnError::Persist(message)
            })
    }

    /// Persist usage opportunistically. Usage is telemetry rather than
    /// provider history, so a failure is reported but does not change turn
    /// control flow.
    pub(crate) fn persist_usage_best_effort(
        &mut self,
        usage: session::UsageSummary,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        if self
            .persist_event(SessionEvent::Usage { usage }, events)
            .is_err()
        {
            tracing::warn!("could not persist usage telemetry");
        }
    }

    /// Durable flush for deferred-sync stores (no-op otherwise). A sync
    /// failure is a persistence failure, not telemetry: quarantine before any
    /// queued operation can observe divergent durable history.
    pub(crate) fn flush_deferred_sync(
        &self,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        let Some(state) = self.session.as_ref() else {
            return Ok(());
        };
        if !state.store.deferred_sync() {
            return Ok(());
        }
        if let Err(error) = state.store.sync_session(&state.session) {
            let message = format!("session persistence failed during sync: {error}");
            send(events, AgentEvent::Error(message.clone()));
            return Err(TurnError::Persist(message));
        }
        Ok(())
    }

    pub(crate) fn persist_user_message(
        &mut self,
        message: &Message,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        self.persist_event(
            SessionEvent::UserMessage {
                message: StoredMessage::from_llm(message),
            },
            events,
        )
    }

    pub(crate) fn persist_assistant(
        &mut self,
        reasoning: &str,
        text: &str,
        items: &[Content],
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        if reasoning.is_empty() && text.is_empty() && items.is_empty() {
            return Ok(());
        }
        let mut content = Vec::new();
        if !reasoning.is_empty() {
            content.push(Content::Reasoning(reasoning.to_owned()));
        }
        if !text.is_empty() {
            content.push(Content::Text(text.to_owned()));
        }
        let has_opaque = items
            .iter()
            .any(|item| matches!(item, Content::Opaque { .. }));
        if has_opaque {
            // Codex continuation state must remain interleaved with calls in
            // one assistant message. Embedded calls are validated and replayed
            // by session using the same state machine as standalone calls.
            content.extend(items.iter().cloned());
        }
        let message = Message::assistant(content);
        if !message.content.is_empty() {
            self.persist_event(
                SessionEvent::AssistantMessage {
                    message: StoredMessage::from_llm(&message),
                },
                events,
            )?;
        }
        if !has_opaque {
            for call in items.iter().filter_map(|item| match item {
                Content::ToolCall(call) => Some(call),
                _ => None,
            }) {
                self.persist_event(
                    SessionEvent::ToolCall {
                        call: StoredToolCall::from(call),
                    },
                    events,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn persist_tool_result(
        &mut self,
        call: &ToolCall,
        content: &str,
        is_error: bool,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
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

    /// Record cancellation before the turn is allowed to finish.
    ///
    /// A missing cancellation marker leaves durable history ambiguous: the
    /// live agent may have stopped while the session still looks like it can
    /// continue.  Treat that append like every other history-bearing event so
    /// the turn boundary can quarantine instead of running queued work.
    pub(crate) fn persist_cancelled(
        &mut self,
        reason: impl Into<String>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        self.persist_event(
            SessionEvent::TurnCancelled {
                reason: reason.into(),
            },
            events,
        )
    }
}

/// Adapt the session-owned snapshot into the UI-facing entry type, adding
/// tool summaries. The session crate owns event replay and pairing; this
/// conversion is presentational only.
pub(crate) fn ui_snapshot_entries(
    entries: Vec<session::SessionSnapshotEntry>,
) -> Vec<SessionSnapshotEntry> {
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
            } => SessionSnapshotEntry::Tool {
                summary: call_summary(&name, &arguments),
                name,
                ok,
                duration_ms: 0,
                output,
                error,
            },
        })
        .collect()
}

pub(crate) fn usage_event(usage: &session::UsageSummary) -> AgentEvent {
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
