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

    /// Durable flush for deferred-sync stores (no-op otherwise). Called at
    /// turn boundaries; failures are logged, not surfaced — the data is in
    /// the OS page cache and the next boundary retries.
    pub(crate) fn flush_deferred_sync(&self) {
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
        opaque: &[(String, serde_json::Value)],
        calls: &[ToolCall],
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        if reasoning.is_empty() && text.is_empty() && opaque.is_empty() && calls.is_empty() {
            return Ok(());
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
        // Tool calls have their own durable events. Keeping them out of this
        // message avoids duplicates while retaining explicit call events.
        let message = Message::assistant(content);
        if !message.content.is_empty() {
            self.persist_event(
                SessionEvent::AssistantMessage {
                    message: StoredMessage::from_llm(&message),
                },
                events,
            )?;
        }
        for call in calls {
            self.persist_event(
                SessionEvent::ToolCall {
                    call: StoredToolCall::from(call),
                },
                events,
            )?;
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

    /// Try to record cancellation without obscuring the original interrupt.
    pub(crate) fn persist_cancelled(
        &mut self,
        reason: impl Into<String>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        if self
            .persist_event(
                SessionEvent::TurnCancelled {
                    reason: reason.into(),
                },
                events,
            )
            .is_err()
        {
            tracing::warn!("could not persist cancellation marker");
        }
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
