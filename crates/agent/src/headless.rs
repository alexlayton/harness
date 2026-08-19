//! Headless / non-interactive runtime: `harness -p "…"`.
//!
//! This is an alternative *frontend* that drives the exact same
//! [`Agent`](crate::agent::Agent) stack the TUI uses.  One prompt is ingested
//! (positional argument, or stdin when no positional is given), the agent runs
//! to completion, the model's answer is written to stdout, and the process
//! exits.  All progress chatter is optional and goes to stderr behind `-v`;
//! stdout carries only the answer.

use crate::agent::Agent;
use crate::agent::AgentEvent;
use crate::config::{Cli, Config};
use crate::tools::ToolRegistry;
use anyhow::{Context, Result, anyhow};
use llm::Provider;
use session::{SessionCreateOptions, SessionStore};
use std::io::{IsTerminal, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

/// Resolve the one-shot prompt for `--print` mode.
///
/// Precedence: a positional prompt wins over stdin; absent a positional, a
/// pipe (non-tty stdin) is read in full; a tty with no positional is an error.
/// Without `--print`, a positional is an error and the result is unused (the
/// interactive TUI path takes over).
pub fn resolve_prompt(cli: &Cli) -> Result<String> {
    resolve_prompt_with(
        cli,
        || std::io::stdin().is_terminal(),
        || {
            let mut buffer = String::new();
            std::io::stdin()
                .read_to_string(&mut buffer)
                .context("read prompt from stdin")?;
            Ok(buffer)
        },
    )
}

/// Testable core of [`resolve_prompt`]: the tty probe and the stdin read are
/// injected so unit tests need neither a real terminal nor a real pipe.
fn resolve_prompt_with(
    cli: &Cli,
    stdin_is_tty: impl FnOnce() -> bool,
    read_stdin: impl FnOnce() -> Result<String>,
) -> Result<String> {
    if !cli.print {
        if cli.prompt.is_empty() {
            return Ok(String::new());
        }
        return Err(anyhow!("prompt requires --print"));
    }
    if !cli.prompt.is_empty() {
        return Ok(cli.prompt.join(" ").trim().to_owned());
    }
    if stdin_is_tty() {
        return Err(anyhow!(
            "no prompt: pass a prompt argument or pipe one on stdin"
        ));
    }
    Ok(read_stdin()?.trim().to_owned())
}

/// Install a SIGINT handler that cancels the agent's application token.  The
/// agent already selects on that token throughout `run`/`run_turn`, persists
/// `TurnCancelled`, and returns cleanly, so no new cancellation logic lives
/// here.
fn cancel_on_sigint(cancel: CancellationToken) {
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel.cancel();
        }
    });
}

/// Drive the headless event stream until the agent task closes the channel.
/// Writes the answer to real stdout and maps the event stream to stderr per
/// the routing table in the design.  Returns the process exit code.
pub async fn drive_headless_events(
    event_rx: mpsc::UnboundedReceiver<AgentEvent>,
    verbose: bool,
) -> ExitCode {
    let mut stdout = std::io::BufWriter::new(std::io::stdout());
    let mut stderr = std::io::stderr();
    let code = drive_headless_events_into(event_rx, verbose, &mut stdout, &mut stderr).await;
    let _ = stdout.flush();
    code
}

/// Writer-injectable core of [`drive_headless_events`] so tests can assert the
/// stdout/stderr split without touching the process streams.
async fn drive_headless_events_into(
    mut event_rx: mpsc::UnboundedReceiver<AgentEvent>,
    verbose: bool,
    stdout: &mut (impl Write + Unpin),
    stderr: &mut (impl Write + Unpin),
) -> ExitCode {
    // An `AgentEvent::Error` marks the turn as failed unless a later event
    // proves the agent recovered (it re-streamed and produced output or more
    // tool activity).  `TurnFinished` does not clear it: it is the point where
    // a trailing unrecovered error is turned into exit code 1.
    let mut error_pending = false;
    // Only terminate the answer with a newline if something was actually
    // written; a tool-only turn whose final answer is the filesystem change
    // leaves stdout empty.
    let mut wrote_text = false;

    while let Some(event) = event_rx.recv().await {
        let is_error = matches!(event, AgentEvent::Error(_));
        let is_turn_finished = matches!(event, AgentEvent::TurnFinished);
        match &event {
            AgentEvent::AuthStarted => {}
            AgentEvent::AuthPrompt { message } | AgentEvent::AuthProgress { message } => {
                if verbose {
                    let _ = writeln!(stderr, "{message}");
                }
            }
            AgentEvent::AuthDeviceCode {
                verification_url,
                user_code,
                expires_in,
                interval,
            } => {
                let _ = writeln!(
                    stderr,
                    "GitHub Copilot login: open {verification_url} and enter code {user_code} \
                     (expires in {expires_in}s, polling every {interval}s)"
                );
            }
            AgentEvent::AuthFinished => {
                if verbose {
                    let _ = writeln!(stderr, "GitHub Copilot authentication complete");
                }
            }
            AgentEvent::AuthFailed { message } => {
                let _ = writeln!(stderr, "error: {message}");
            }
            AgentEvent::TextDelta(delta) => {
                wrote_text = true;
                let _ = write!(stdout, "{delta}");
                let _ = stdout.flush();
            }
            AgentEvent::ReasoningDelta(delta) => {
                if verbose {
                    let _ = writeln!(stderr, "{delta}");
                }
            }
            AgentEvent::ToolCallStarted { summary, .. } => {
                if verbose {
                    let _ = writeln!(stderr, "▸ {summary}");
                }
            }
            AgentEvent::ToolCallFinished {
                name: _,
                summary,
                ok,
                duration_ms,
                output,
                error,
            } => {
                if verbose {
                    let mark = if *ok { "✓" } else { "✗" };
                    let _ = writeln!(stderr, "{mark} {summary} ({duration_ms}ms)");
                    if let Some(error) = error {
                        let _ = writeln!(stderr, "{error}");
                    } else if !output.is_empty() {
                        let _ = writeln!(stderr, "{output}");
                    }
                }
            }
            AgentEvent::Retrying { attempt, message } => {
                if verbose {
                    let _ = writeln!(stderr, "retry #{attempt}: {message}");
                }
            }
            AgentEvent::Notice(message) => {
                if verbose {
                    let _ = writeln!(stderr, "{message}");
                }
            }
            AgentEvent::Error(message) => {
                let _ = writeln!(stderr, "error: {message}");
            }
            AgentEvent::UsageUpdated {
                input_tokens,
                output_tokens,
                cached_tokens,
                reasoning_tokens,
                cost,
            } => {
                if verbose {
                    let _ = writeln!(
                        stderr,
                        "tokens {input_tokens}/{output_tokens} (cached {cached_tokens}, \
                         reasoning {reasoning_tokens}) · ${cost}"
                    );
                }
            }
            AgentEvent::TurnFinished => {
                if wrote_text {
                    let _ = writeln!(stdout);
                }
                let _ = stdout.flush();
            }
            AgentEvent::SessionChanged { id, .. } => {
                if verbose {
                    let _ = writeln!(stderr, "session {id}");
                }
            }
            AgentEvent::ModelChanged { provider, model } => {
                if verbose {
                    let _ = writeln!(stderr, "model → {provider} · {model}");
                }
            }
            AgentEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
                auto,
                reason,
            } => {
                if verbose {
                    let _ = writeln!(
                        stderr,
                        "{}compacted through {compacted_through} ({summary_bytes}b) [{reason}]",
                        if *auto { "auto-" } else { "" }
                    );
                }
            }
            AgentEvent::SessionSnapshot { .. }
            | AgentEvent::SessionList { .. }
            | AgentEvent::ModelList { .. }
            | AgentEvent::SessionExported { .. } => {}
        }
        if is_error {
            error_pending = true;
        } else if !is_turn_finished {
            error_pending = false;
        }
    }

    if error_pending {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// Run one non-interactive prompt to completion and return the process code.
///
/// Fresh session by default (persisted, resumable later); `--resume <sel>`
/// loads an existing one through the store's existing selector.  The agent,
/// its channels, its persistence hooks, and the cancellation token are all the
/// same objects the interactive path uses.
pub async fn run_headless(
    config: &Config,
    cli: &Cli,
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    store: SessionStore,
) -> Result<ExitCode> {
    run_headless_with_cancel(config, cli, provider, tools, store, None).await
}

/// [`run_headless`] with an optional externally supplied cancellation token so
/// tests can interrupt a turn at a deterministic point.  Production always
/// passes `None`, which installs the real SIGINT handler.
async fn run_headless_with_cancel(
    config: &Config,
    cli: &Cli,
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    store: SessionStore,
    external_cancel: Option<CancellationToken>,
) -> Result<ExitCode> {
    let prompt = resolve_prompt(cli)?;

    let session = match &cli.resume {
        Some(selector) => store
            .load(selector)
            .with_context(|| format!("load session `{selector}`"))?,
        None => store.create(SessionCreateOptions {
            provider: Some(provider.name().to_owned()),
            model: Some(config.model.clone()),
            ..SessionCreateOptions::default()
        })?,
    };

    let install_sigint = external_cancel.is_none();
    let cancel = external_cancel.unwrap_or_default();
    let (input_tx, input_rx): (
        mpsc::UnboundedSender<InputMessage>,
        mpsc::UnboundedReceiver<InputMessage>,
    ) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    let agent = Agent::new(provider, tools, config.model.clone(), cancel.clone())
        .with_compaction(config.compaction.clone())
        .with_session(store, session);
    let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

    // Send the single user turn, then close the channel so the agent's run
    // loop exits after this turn has drained its tool-call rounds.
    input_tx
        .send(InputMessage::Message(prompt))
        .context("send prompt to agent")?;
    drop(input_tx);

    if install_sigint {
        cancel_on_sigint(cancel.clone());
    }

    let exit_code = drive_headless_events(event_rx, cli.verbose).await;
    let interrupted = cancel.is_cancelled();
    cancel.cancel();
    let _ = agent_task.await;
    if interrupted {
        // The agent has already persisted `TurnCancelled`; 130 mirrors the
        // conventional SIGINT status while the session flush is awaited above.
        return Ok(ExitCode::from(130));
    }
    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FileConfig;
    use async_trait::async_trait;
    use futures_util::stream;
    use llm::{
        CompletionRequest, Content, EventStream, LlmError, Message, ModelInfo, StreamEvent, Usage,
    };
    use session::SessionEvent;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    // ------------------------------------------------------------------ resolve_prompt

    #[test]
    fn positional_prompt_wins_over_stdin() {
        let cli = Cli {
            print: true,
            prompt: vec!["write".into(), "a".into(), "poem".into()],
            ..Cli::default()
        };
        let prompt = resolve_prompt_with(&cli, || true, || unreachable!()).unwrap();
        assert_eq!(prompt, "write a poem");
    }

    #[test]
    fn stdin_is_used_when_no_positional_and_stdin_is_not_a_tty() {
        let cli = Cli {
            print: true,
            ..Cli::default()
        };
        let prompt = resolve_prompt_with(&cli, || false, || Ok("piped\nprompt\n".into())).unwrap();
        assert_eq!(prompt, "piped\nprompt");
    }

    #[test]
    fn tty_without_positional_is_an_error() {
        let cli = Cli {
            print: true,
            ..Cli::default()
        };
        let error = resolve_prompt_with(&cli, || true, || unreachable!())
            .err()
            .unwrap();
        assert!(error.to_string().contains("no prompt"));
    }

    #[test]
    fn positional_without_print_is_an_error() {
        let cli = Cli {
            print: false,
            prompt: vec!["hello".into()],
            ..Cli::default()
        };
        let error = resolve_prompt_with(&cli, || true, || unreachable!())
            .err()
            .unwrap();
        assert!(error.to_string().contains("prompt requires --print"));
    }

    #[test]
    fn interactive_without_prompt_is_empty() {
        let cli = Cli::default();
        assert_eq!(
            resolve_prompt_with(&cli, || true, || unreachable!()).unwrap(),
            ""
        );
    }

    // ------------------------------------------------------------- event routing

    fn route(events: Vec<AgentEvent>, verbose: bool) -> (Vec<u8>, Vec<u8>, ExitCode) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (tx, rx) = mpsc::unbounded_channel();
            for event in events {
                tx.send(event).unwrap();
            }
            drop(tx);
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let code = drive_headless_events_into(rx, verbose, &mut stdout, &mut stderr).await;
            (stdout, stderr, code)
        })
    }

    fn clean_turn() -> Vec<AgentEvent> {
        vec![
            AgentEvent::SessionChanged {
                id: "s1".into(),
                title: None,
                loaded: false,
            },
            AgentEvent::TextDelta("answer ".into()),
            AgentEvent::TextDelta("text".into()),
            AgentEvent::TurnFinished,
        ]
    }

    #[test]
    fn default_mode_prints_text_and_stays_silent_on_stderr() {
        let (stdout, stderr, code) = route(clean_turn(), false);
        assert_eq!(String::from_utf8(stdout).unwrap(), "answer text\n");
        assert!(stderr.is_empty(), "default stderr must stay silent");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn verbose_mode_reports_tool_activity_on_stderr() {
        let events = vec![
            AgentEvent::ToolCallStarted {
                name: "bash".into(),
                summary: "echo hi".into(),
                arguments: "{}".into(),
            },
            AgentEvent::ToolCallFinished {
                name: "bash".into(),
                summary: "echo hi".into(),
                ok: true,
                duration_ms: 12,
                output: "hi\n".into(),
                error: None,
            },
            AgentEvent::TextDelta("done".into()),
            AgentEvent::TurnFinished,
        ];
        let (stdout, stderr, code) = route(events.clone(), false);
        assert_eq!(String::from_utf8(stdout).unwrap(), "done\n");
        assert!(stderr.is_empty(), "tool activity must be hidden by default");
        assert_eq!(code, ExitCode::SUCCESS);

        let (stdout, stderr, code) = route(events, true);
        assert_eq!(String::from_utf8(stdout).unwrap(), "done\n");
        let stderr = String::from_utf8(stderr).unwrap();
        assert!(
            stderr.contains("▸ echo hi"),
            "missing started marker: {stderr}"
        );
        assert!(
            stderr.contains("✓ echo hi (12ms)"),
            "missing finished line: {stderr}"
        );
        assert!(stderr.contains("hi"), "missing tool output: {stderr}");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn output_is_flushed_and_terminated_on_turn_finished() {
        let (stdout, _, _) = route(
            vec![
                AgentEvent::TextDelta("partial".into()),
                AgentEvent::TurnFinished,
            ],
            false,
        );
        assert_eq!(String::from_utf8(stdout).unwrap(), "partial\n");
    }

    #[test]
    fn tool_only_turn_leaves_stdout_empty() {
        let (stdout, _, code) = route(
            vec![
                AgentEvent::ToolCallStarted {
                    name: "bash".into(),
                    summary: "touch x".into(),
                    arguments: "{}".into(),
                },
                AgentEvent::ToolCallFinished {
                    name: "bash".into(),
                    summary: "touch x".into(),
                    ok: true,
                    duration_ms: 3,
                    output: String::new(),
                    error: None,
                },
                AgentEvent::TurnFinished,
            ],
            false,
        );
        assert!(stdout.is_empty(), "no TextDelta means empty stdout");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn unrecovered_error_exits_1() {
        let (stdout, stderr, code) = route(
            vec![
                AgentEvent::TextDelta("extra".into()),
                AgentEvent::Error("provider failed".into()),
                AgentEvent::TurnFinished,
            ],
            false,
        );
        assert_eq!(String::from_utf8(stdout).unwrap(), "extra\n");
        assert!(
            String::from_utf8(stderr)
                .unwrap()
                .contains("error: provider failed")
        );
        assert_eq!(code, ExitCode::from(1));
    }

    #[test]
    fn error_followed_by_recovery_exits_0() {
        // The agent emits Error, then re-streams and produces output: the turn
        // recovered and must not fail the run.
        let (_, stderr, code) = route(
            vec![
                AgentEvent::Error("http 500".into()),
                AgentEvent::Retrying {
                    attempt: 1,
                    message: "transient".into(),
                },
                AgentEvent::TextDelta("recovered".into()),
                AgentEvent::TurnFinished,
            ],
            false,
        );
        assert!(
            String::from_utf8(stderr)
                .unwrap()
                .contains("error: http 500")
        );
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn tool_error_does_not_fail_the_run() {
        let (_, _, code) = route(
            vec![
                AgentEvent::ToolCallStarted {
                    name: "bash".into(),
                    summary: "false".into(),
                    arguments: "{}".into(),
                },
                AgentEvent::ToolCallFinished {
                    name: "bash".into(),
                    summary: "false".into(),
                    ok: false,
                    duration_ms: 5,
                    output: "exit 1".into(),
                    error: Some("exit 1".into()),
                },
                AgentEvent::TextDelta("i'll fix it".into()),
                AgentEvent::TurnFinished,
            ],
            false,
        );
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn verbose_mode_reports_usage_model_and_session() {
        let events = vec![
            AgentEvent::SessionChanged {
                id: "abc123".into(),
                title: None,
                loaded: false,
            },
            AgentEvent::ModelChanged {
                provider: "opencode-go".into(),
                model: "gpt-x".into(),
            },
            AgentEvent::UsageUpdated {
                input_tokens: 10,
                output_tokens: 3,
                cached_tokens: 2,
                reasoning_tokens: 4,
                cost: "0.000123".into(),
            },
            AgentEvent::TurnFinished,
        ];
        let (_, silent, _) = route(events.clone(), false);
        assert!(silent.is_empty());

        let (_, stderr, _) = route(events, true);
        let stderr = String::from_utf8(stderr).unwrap();
        assert!(stderr.contains("session abc123"));
        assert!(stderr.contains("model → opencode-go · gpt-x"));
        assert!(stderr.contains("tokens 10/3 (cached 2, reasoning 4) · $0.000123"));
    }

    // ------------------------------------------------------------------ resume

    /// Records the requests it serves so tests can assert on the conversation
    /// history the agent actually sent.
    #[derive(Default)]
    struct RecordingProvider {
        requests: Mutex<Vec<CompletionRequest>>,
    }

    impl RecordingProvider {
        fn texts(&self) -> Vec<String> {
            self.requests
                .lock()
                .unwrap()
                .iter()
                .flat_map(|request| {
                    request
                        .messages
                        .iter()
                        .flat_map(|message| {
                            message.content.iter().filter_map(|content| match content {
                                Content::Text(text) => Some(text.clone()),
                                _ => None,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn name(&self) -> &str {
            "record"
        }

        async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
            self.requests.lock().unwrap().push(request.clone());
            Ok(Box::pin(stream::iter(vec![
                Ok(StreamEvent::TextDelta("hi back".into())),
                Ok(StreamEvent::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(Usage::default()),
                }),
            ])))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    fn headless_config(cli: &Cli) -> Config {
        Config::resolve_from_file(
            cli,
            &FileConfig::default(),
            PathBuf::from("/tmp/harness-headless-config.toml"),
            |_| Some("secret".into()),
        )
        .unwrap()
    }

    #[test]
    fn resume_loads_previous_context_into_the_prompt() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let mut session = store
                .create(SessionCreateOptions {
                    provider: Some("record".into()),
                    model: Some("demo".into()),
                    ..SessionCreateOptions::default()
                })
                .unwrap();
            store
                .append_event(
                    &mut session,
                    SessionEvent::UserMessage {
                        message: session::StoredMessage::from_llm(&Message::user(
                            "earlier question",
                        )),
                    },
                )
                .unwrap();
            store
                .append_event(
                    &mut session,
                    SessionEvent::AssistantMessage {
                        message: session::StoredMessage::from_llm(&Message::assistant(vec![
                            Content::Text("earlier answer".into()),
                        ])),
                    },
                )
                .unwrap();

            let provider = Arc::new(RecordingProvider::default());
            let cli = Cli {
                print: true,
                resume: Some(session.id().to_string()),
                prompt: vec!["follow up".into()],
                ..Cli::default()
            };
            let config = headless_config(&cli);
            let code = run_headless(
                &config,
                &cli,
                provider.clone(),
                ToolRegistry::empty(),
                store,
            )
            .await
            .unwrap();
            assert_eq!(code, ExitCode::SUCCESS);

            let texts = provider.texts();
            assert!(
                texts.iter().any(|text| text.contains("earlier answer")),
                "resumed context must be visible to the agent: {texts:?}"
            );
            assert!(
                texts.iter().any(|text| text.contains("follow up")),
                "the new prompt must be sent: {texts:?}"
            );
        });
    }

    #[test]
    fn fresh_run_has_no_previous_context() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let provider = Arc::new(RecordingProvider::default());
            let cli = Cli {
                print: true,
                prompt: vec!["hello".into()],
                ..Cli::default()
            };
            let config = headless_config(&cli);
            let code = run_headless(
                &config,
                &cli,
                provider.clone(),
                ToolRegistry::empty(),
                store,
            )
            .await
            .unwrap();
            assert_eq!(code, ExitCode::SUCCESS);
            let texts = provider.texts();
            assert_eq!(texts, vec!["hello".to_owned()]);
        });
    }

    // ------------------------------------------------------------ cancellation

    /// A provider whose stream never yields, so the turn stays in flight until
    /// the cancellation token fires.
    struct HangingProvider;

    #[async_trait]
    impl Provider for HangingProvider {
        fn name(&self) -> &str {
            "hang"
        }

        async fn stream(&self, _request: &CompletionRequest) -> Result<EventStream, LlmError> {
            Ok(Box::pin(stream::pending()))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    /// Mirror of §12's Ctrl-C test without delivering a real signal: external
    /// cancellation mid-turn must persist `TurnCancelled`, drain the agent
    /// task, and surface as exit 130.
    #[test]
    fn cancellation_persists_turn_cancelled_and_exits_130() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let root = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let store = SessionStore::new(root.path(), workspace.path()).unwrap();
            let session = store
                .create(SessionCreateOptions {
                    provider: Some("hang".into()),
                    model: Some("demo".into()),
                    ..SessionCreateOptions::default()
                })
                .unwrap();
            let id = session.id().to_string();

            let cli = Cli {
                print: true,
                resume: Some(id.clone()),
                prompt: vec!["hello".into()],
                ..Cli::default()
            };
            let config = headless_config(&cli);
            let cancel = CancellationToken::new();
            let store_for_run = store.clone();
            let cancel_for_run = cancel.clone();
            let task = tokio::spawn(async move {
                run_headless_with_cancel(
                    &config,
                    &cli,
                    Arc::new(HangingProvider),
                    ToolRegistry::empty(),
                    store_for_run,
                    Some(cancel_for_run),
                )
                .await
                .unwrap()
            });

            // Wait until the turn has actually started (user message persisted
            // before the provider hand-off) so cancellation lands mid-turn.
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let started = store
                    .open(&session::SessionId::parse(&id).unwrap())
                    .is_ok_and(|loaded| {
                        loaded
                            .events
                            .iter()
                            .any(|record| matches!(record.event, SessionEvent::UserMessage { .. }))
                    });
                if started || Instant::now() > deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            cancel.cancel();
            let code = task.await.unwrap();
            assert_eq!(code, ExitCode::from(130));

            let loaded = store
                .open(&session::SessionId::parse(&id).unwrap())
                .unwrap();
            assert!(
                loaded
                    .events
                    .iter()
                    .any(|record| matches!(record.event, SessionEvent::TurnCancelled { .. })),
                "cancellation must persist TurnCancelled"
            );
        });
    }
}
