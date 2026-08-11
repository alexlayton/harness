use crate::config::{build_provider, save_settings};
use crate::prompt::current_system_prompt;
use crate::tools::{ToolRegistry, call_summary};
use futures_util::StreamExt;
use llm::{
    CompletionRequest, Content, Message, Provider, RetryCallback, Role, StreamEvent, ToolCall,
};
use std::collections::VecDeque;
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
}

pub struct Agent {
    pub provider: Arc<dyn Provider>,
    pub tools: ToolRegistry,
    pub model: String,
    pub history: Vec<Message>,
    pub cancel: CancellationToken,
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
        }
    }

    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
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
                    self.history.clear();
                    send(
                        &events,
                        AgentEvent::Notice("Started a new conversation".into()),
                    );
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

            self.history.push(Message::user(user_text));
            let turn_cancel = CancellationToken::new();
            let mut iteration = 0u32;
            let mut end_turn = false;

            while !end_turn {
                let request = CompletionRequest {
                    model: self.model.clone(),
                    system: Some(current_system_prompt()),
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
                let provider_future = self.provider.stream_with_retry(&request, on_retry);
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
                            send(&events, AgentEvent::TurnFinished);
                            return;
                        }
                    }
                };
                let Some(stream_result) = stream_result else {
                    send(&events, AgentEvent::TurnFinished);
                    end_turn = true;
                    continue;
                };
                let mut stream = match stream_result {
                    Ok(stream) => stream,
                    Err(error) => {
                        send(&events, AgentEvent::Error(error.to_string()));
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
                                Ok(StreamEvent::Done { .. }) => {}
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
                    append_assistant(&mut self.history, &reasoning, &text, tool_calls);
                    send(&events, AgentEvent::TurnFinished);
                    if self.cancel.is_cancelled() {
                        return;
                    }
                    end_turn = true;
                    continue;
                }

                append_assistant(&mut self.history, &reasoning, &text, tool_calls.clone());

                if let Some(error) = stream_error {
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
                    let tool_future =
                        self.tools
                            .execute(&call.name, call.arguments.clone(), turn_cancel.clone());
                    tokio::pin!(tool_future);
                    let result = loop {
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
                    };
                    let Some(result) = result else {
                        send(
                            &events,
                            AgentEvent::ToolCallFinished {
                                name: call.name.clone(),
                                summary,
                                ok: false,
                                duration_ms: started.elapsed().as_millis() as u64,
                                output: String::new(),
                                error: Some("cancelled".into()),
                            },
                        );
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
                    self.history.push(Message {
                        role: Role::Tool,
                        content: vec![Content::ToolResult {
                            tool_call_id: call.id,
                            content: result.content,
                            is_error: result.is_error,
                        }],
                    });
                }

                if !end_turn && iteration >= 100 {
                    let note = "max tool iterations reached";
                    self.history.push(Message::user(note));
                    send(&events, AgentEvent::TextDelta(note.into()));
                    send(&events, AgentEvent::TurnFinished);
                    end_turn = true;
                }
            }
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

    struct MockProvider {
        calls: AtomicUsize,
        scripts: Vec<Vec<StreamEvent>>,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &'static str {
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
