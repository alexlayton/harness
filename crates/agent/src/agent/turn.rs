use super::InputMessage;
use super::persistence::usage_event;
use super::{Agent, AgentEvent, CompactionReason, MAX_TURN_RECOVERIES, TurnError, send};
use crate::prompt::system_prompt_with_workspace_context;
use futures_util::stream::StreamExt;
use llm::{
    CompletionRequest, Content, LlmError, Message, RetryCallback, Role, StreamEvent, truncate_utf8,
};
use session::{SessionEvent, usage_summary};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

impl Agent {
    /// The single shared turn executor used by normal user messages and
    /// skill-invoked messages alike. It owns the whole operation boundary:
    ///
    /// - builds a turn-scoped child token of the application token;
    /// - propagates shutdown (`TurnControl::Shutdown`);
    /// - quarantines/stops after persistence failure
    ///   (`TurnControl::Quarantine`, mirroring the run-loop quarantine);
    /// - flushes deferred session writes exactly once at the boundary;
    /// - emits terminal events exactly once on early-exit paths (shutdown
    ///   before any body ran, persistence failure inside the body). A body
    ///   that already emitted `TurnFinished` and then fails its boundary
    ///   flush yields a second `TurnFinished` from the run-loop quarantine;
    ///   the quarantine (loop break, queued work never runs) is what the
    ///   boundary guarantees, not a literal single event in that corner.
    pub(crate) async fn execute_turn(
        &mut self,
        user_text: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
    ) -> TurnControl {
        let turn_cancel = self.cancel.child_token();
        let outcome = self
            .run_turn_body(user_text, events, input, &turn_cancel)
            .await;
        // Deferred-sync boundary: one durable flush per operation, for both
        // normal and skill-invoked turns. A failed fsync quarantines even if
        // the turn body otherwise completed successfully.
        if self.flush_deferred_sync(events).is_err() {
            return TurnControl::Quarantine;
        }
        match outcome {
            Ok(()) => TurnControl::Continue,
            Err(TurnError::Shutdown) => TurnControl::Shutdown,
            Err(TurnError::Persist(_)) => {
                // Mirror the run-loop quarantine: the terminal event was
                // already emitted at the failure source; stop here so no
                // queued work runs on divergent history.
                TurnControl::Quarantine
            }
        }
    }

    /// Turn body: everything `run_turn` historically did, minus token
    /// construction and the deferred-sync flush (both owned by
    /// [`Self::execute_turn`]).
    #[tracing::instrument(
        name = "turn",
        skip(self, events, input, cancel),
        fields(user_text = %truncate_utf8(&user_text, 200))
    )]
    async fn run_turn_body(
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
            let application = self.cancel.clone();
            let mut buffered = VecDeque::new();
            let mut input_open = self.input_open;
            let mut interrupted = false;
            let compacted = {
                let mut application_open = true;
                let compaction = self.compact_and_reload(
                    events,
                    cancel,
                    CompactionReason::Auto,
                    user_text.len(),
                );
                tokio::pin!(compaction);
                loop {
                    tokio::select! {
                        biased;
                        result = &mut compaction => break result,
                        _ = application.cancelled(), if application_open => {
                            application_open = false;
                            cancel.cancel();
                        }
                        message = input.recv(), if input_open => match message {
                            Some(InputMessage::Interrupt) => {
                                interrupted = true;
                                cancel.cancel();
                            }
                            Some(message) => buffered.push_back(message),
                            None => input_open = false,
                        },
                    }
                }
            };
            self.input_open = input_open;
            self.queued.extend(buffered);
            let compacted = match compacted {
                Ok(compacted) => compacted,
                Err(TurnError::Shutdown) => {
                    self.persist_cancelled("application shutdown", events);
                    send(events, AgentEvent::TurnFinished);
                    return Err(TurnError::Shutdown);
                }
                Err(error) => return Err(error),
            };
            if interrupted || cancel.is_cancelled() {
                self.persist_cancelled("turn interrupted during compaction", events);
                send(events, AgentEvent::TurnFinished);
                if application.is_cancelled() {
                    return Err(TurnError::Shutdown);
                }
                return Ok(());
            }
            if compacted {
                send(
                    events,
                    AgentEvent::Notice(format!("auto-compacted: context at {percent}% of window")),
                );
            }
        }

        let user_message = Message::user(user_text);
        self.persist_user_message(&user_message, events)?;
        self.history.push(user_message.clone());
        let mut recoveries = 0;
        let mut overflow_recoveries = 0;
        loop {
            let tool_snapshot = self.tools.snapshot();
            let request = CompletionRequest {
                model: self.model.clone(),
                system: Some(system_prompt_with_workspace_context(
                    &self.tools.workspace_root().display().to_string(),
                    &tool_snapshot.prompt_context,
                    self.tools.skills(),
                    &self.project_context,
                )),
                messages: self.history.clone(),
                tools: tool_snapshot.definitions,
                max_tokens: None,
                temperature: None,
                reasoning: self.reasoning,
                session_id: self.session_id(),
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
                    // Once an interrupt is queued it must win over a provider
                    // result that becomes ready at the same time. Otherwise a
                    // failed request can enter its retry path after Esc.
                    biased;
                    _ = self.cancel.cancelled() => {
                        self.persist_cancelled("application shutdown", events);
                        send(events, AgentEvent::TurnFinished);
                        return Err(TurnError::Shutdown);
                    }
                    _ = cancel.cancelled() => break None,
                    message = input.recv(), if self.input_open => match message {
                        Some(InputMessage::Interrupt) => {
                            cancel.cancel();
                            break None;
                        }
                        Some(message) => self.queued.push_back(message),
                        None => self.input_open = false,
                    },
                    result = &mut provider_future => break Some(result),
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
                        .await?
                    {
                        continue;
                    }
                    let message = error.to_string();
                    self.persist_event(
                        SessionEvent::Error {
                            message: message.clone(),
                        },
                        events,
                    )?;
                    send(events, AgentEvent::Error(message));
                    send(events, AgentEvent::TurnFinished);
                    return Ok(());
                }
            };

            let mut text = String::new();
            let mut reasoning = String::new();
            // Preserve the wire order of opaque provider state and function
            // calls. Codex requires encrypted reasoning to remain interleaved
            // with calls when the next request is rebuilt.
            let mut assistant_items = Vec::<Content>::new();
            let mut cancelled = false;
            let mut stream_error = None;

            loop {
                tokio::select! {
                    // Prefer cancellation/input when the stream completes or
                    // errors in the same scheduler tick as an Esc interrupt.
                    // Recovery decisions below must never outrun manual stop.
                    biased;
                    _ = self.cancel.cancelled() => {
                        cancelled = true;
                        break;
                    }
                    _ = cancel.cancelled() => {
                        cancelled = true;
                        break;
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
                            Ok(StreamEvent::OpaqueState { provider, data }) => assistant_items.push(Content::Opaque { provider, data }),
                            Ok(StreamEvent::ToolCallComplete(call)) => assistant_items.push(Content::ToolCall(call)),
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
                                    self.persist_usage_best_effort(summary.clone(), events);
                                    if let Some(state) = self.session.as_ref() {
                                        send(events, usage_event(&state.session.metadata.usage));
                                    } else {
                                        send(events, usage_event(&summary));
                                    }
                                    send(events, self.context_usage_event());
                                } else {
                                    // A provider may omit usage on a valid
                                    // terminal event. Do not keep using an
                                    // older exact count; fall back to the
                                    // current full-request estimator.
                                    self.last_context_tokens = None;
                                    send(events, self.context_usage_event());
                                }
                            }
                            Err(error) => {
                                stream_error = Some(error);
                                break;
                            }
                        }
                    }
                }
            }

            let tool_calls = assistant_items
                .iter()
                .filter_map(|item| match item {
                    Content::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            if cancelled {
                self.persist_assistant(&reasoning, &text, &assistant_items, events)?;
                append_assistant(&mut self.history, &reasoning, &text, &assistant_items);
                for call in &tool_calls {
                    let cancelled_result = "cancelled before tool execution";
                    self.persist_tool_result(call, cancelled_result, true, events)?;
                    self.history.push(Message::tool_result(
                        call.id.clone(),
                        cancelled_result,
                        true,
                    ));
                }
                self.persist_cancelled("turn interrupted", events);
                send(events, AgentEvent::TurnFinished);
                if self.cancel.is_cancelled() {
                    return Err(TurnError::Shutdown);
                }
                return Ok(());
            }

            self.persist_assistant(&reasoning, &text, &assistant_items, events)?;
            append_assistant(&mut self.history, &reasoning, &text, &assistant_items);
            if stream_error.is_some() || !tool_calls.is_empty() {
                // A stream may have emitted partial assistant content or tool
                // calls before its next request. The old Done count no longer
                // covers that newly appended history.
                self.last_context_tokens = None;
            }

            if let Some(error) = stream_error {
                let message = error.to_string();
                // Calls emitted before a broken stream must be closed before any
                // compaction/reload. Otherwise the compactor observes an invalid
                // assistant tail containing dangling provider tool calls.
                for call in &tool_calls {
                    let error_result = format!("provider stream interrupted: {message}");
                    self.persist_tool_result(call, &error_result, true, events)?;
                    self.history.push(Message::tool_result(
                        call.id.clone(),
                        error_result.clone(),
                        true,
                    ));
                }
                // A mid-stream context overflow (provider tears down an SSE
                // request that outgrew the window) can now compact valid history.
                if self
                    .try_overflow_recovery(&error, events, cancel, &mut overflow_recoveries)
                    .await?
                {
                    continue;
                }
                self.persist_event(
                    SessionEvent::Error {
                        message: message.clone(),
                    },
                    events,
                )?;
                send(events, AgentEvent::Error(message.clone()));
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
            // The dispatcher emits the terminal event and marks the
            // turn-scoped token when tool execution is interrupted. Do not
            // issue another provider request (or a second TurnFinished).
            if cancel.is_cancelled() {
                return Ok(());
            }
            // Tool results are appended after the provider's Done usage was
            // observed. That exact snapshot no longer describes the next
            // request, so use the complete-history estimator instead.
            self.last_context_tokens = None;
        }
    }
}

/// Explicit control flow for command handlers and turn execution.
/// Replaces the old pattern of swallowing `TurnError::Shutdown` and
/// `TurnError::Persist` at skill-invocation sites: every handler returns
/// how the run loop must proceed, and the loop acts on it in one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnControl {
    /// Operation completed (or reported its own terminal event); keep
    /// draining queued input.
    Continue,
    /// Application shutdown fired: stop the run loop immediately.
    Shutdown,
    /// Persistence failed mid-operation: stop like the run-loop
    /// quarantine so no queued work runs on divergent history.
    Quarantine,
}

/// Push a non-durable recovery note into history, preserving provider role
/// alternation: append into a trailing user message when present, otherwise
/// push a new user message. The note lives only in memory; durable events
/// stay clean.
pub(crate) fn push_recovery_note(history: &mut Vec<Message>, note: String) {
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

pub(crate) fn append_assistant(
    history: &mut Vec<Message>,
    reasoning: &str,
    text: &str,
    items: &[Content],
) {
    if reasoning.is_empty() && text.is_empty() && items.is_empty() {
        return;
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
        content.extend(items.iter().cloned());
    } else {
        content.extend(items.iter().filter_map(|item| match item {
            Content::Opaque { .. } => None,
            Content::ToolCall(call) => Some(Content::ToolCall(call.clone())),
            _ => None,
        }));
    }
    history.push(Message {
        role: Role::Assistant,
        content,
    });
}
