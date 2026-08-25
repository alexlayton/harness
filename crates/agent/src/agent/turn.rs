use super::persistence::usage_event;
use super::{Agent, AgentEvent, CompactionReason, MAX_TURN_RECOVERIES, TurnError, send};
use crate::prompt::system_prompt_with_workspace_context;
use futures_util::stream::StreamExt;
use llm::{
    CompletionRequest, Content, LlmError, Message, RetryCallback, Role, StreamEvent, ToolCall,
    truncate_utf8,
};
use session::{SessionEvent, usage_summary};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

impl Agent {
    /// Run one user turn: persist the message, stream the provider response,
    /// execute any tool calls, and persist every durable event.  Returns
    /// [`TurnError::Shutdown`] only when the application cancellation token
    /// fired mid-turn, so `run` can stop immediately.
    #[tracing::instrument(
        name = "turn",
        skip(self, events, input, cancel),
        fields(user_text = %truncate_utf8(&user_text, 200))
    )]
    pub(crate) async fn run_turn(
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
                .await?
            {
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
                                    self.persist_usage_best_effort(summary.clone(), events);
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
                self.persist_assistant(&reasoning, &text, &opaque, &tool_calls, events)?;
                append_assistant(
                    &mut self.history,
                    &reasoning,
                    &text,
                    &opaque,
                    tool_calls.clone(),
                );
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

            self.persist_assistant(&reasoning, &text, &opaque, &tool_calls, events)?;
            append_assistant(
                &mut self.history,
                &reasoning,
                &text,
                &opaque,
                tool_calls.clone(),
            );

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
        }
    }
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
