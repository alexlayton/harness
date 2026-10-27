mod acp;
mod headless;
mod herdr;
mod login;
mod mcp_command;
mod tui_adapter;
mod update;
mod worktree;

use agent::assembly::AgentBuilder;
use agent::{AgentEvent, InputMessage};
use anyhow::{Context, Result};
use clap::Parser;
use config::{
    Cli, Command, Config, ProviderArg, WorktreeCommand, build_provider_with_auths, init_logging,
    provider_factory, save_prewalk_enabled, save_prewalk_model, save_prewalk_reasoning,
    save_reasoning, save_settings,
};
use harness::{config, context};
use headless::run_headless_with_prompt;
use llm::Provider;
use session::{SessionCreateOptions, SessionStore};
use std::process::ExitCode;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tools::{ToolConfig, default_registry};
use tui::{ContextFileEntry, CrossTerm, StartupEntries};

/// Turn a finished agent task into an error when it panicked (or was
/// aborted), keeping the panic message when it is a string.
fn agent_task_outcome(result: Result<(), tokio::task::JoinError>) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.is_panic() => {
            let payload = error.into_panic();
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_owned());
            anyhow::bail!("the agent stopped unexpectedly (panic: {message})")
        }
        Err(error) => anyhow::bail!("the agent stopped unexpectedly: {error}"),
    }
}

fn main() -> ExitCode {
    // `set_var` is only sound while no other thread can read the
    // environment, so path overrides are normalized here, before the Tokio
    // runtime starts its worker threads. Absolute overrides are unchanged,
    // so doing this for every command is harmless; worktree mode needs it
    // before it changes the process cwd.
    if let Err(error) = absolutize_process_path_overrides() {
        eprintln!("error: {error:#}");
        return ExitCode::FAILURE;
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: could not start the async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(main_inner()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn main_inner() -> Result<ExitCode> {
    let mut cli = Cli::parse();

    // Update and login are standalone commands and intentionally precede
    // config/provider resolution.
    if matches!(&cli.command, Some(Command::Update)) {
        return update::run().await.map(|()| ExitCode::SUCCESS);
    }
    // A stale configured API key cannot prevent signing in.
    if let Some(Command::Login(args)) = &cli.command {
        return login::run(args).await;
    }
    if let Some(Command::Mcp(args)) = &cli.command {
        return mcp_command::run(args);
    }

    // Worktree is a launch mode rather than a separate frontend. Enter it
    // before resolving any workspace-scoped tools, context, or session state,
    // then always restore the launch directory and attempt safe cleanup after
    // application startup or shutdown.
    if let Some(Command::Worktree(args)) = cli.command.clone() {
        let launch_directory = std::fs::canonicalize(
            std::env::current_dir().context("resolve the worktree launch directory")?,
        )
        .context("resolve the worktree launch directory")?;
        // Process-level path overrides (config/auth, logging, session roots)
        // were made absolute in `main`, before the runtime started.
        // Relative state/session overrides must retain launch-directory
        // semantics after the process cwd changes.
        let session_root = std::path::absolute(session::default_session_dir())
            .context("resolve the session state directory")?;
        let report_lifecycle = args
            .command
            .as_ref()
            .is_none_or(|WorktreeCommand::Prompt(prompt)| prompt.verbose);
        let lease = worktree::prepare(&args, &launch_directory)?;
        if report_lifecycle {
            let action = if lease.was_created() {
                "created"
            } else {
                "reusing"
            };
            eprintln!("worktree: {action} {}", lease.path().display());
        }
        if lease.source_was_dirty() {
            eprintln!(
                "worktree: warning: the launch checkout has uncommitted changes; the new worktree contains committed Git state only"
            );
        }
        cli.command = args
            .command
            .map(|WorktreeCommand::Prompt(prompt)| Command::Prompt(prompt));

        if let Err(error) = std::env::set_current_dir(lease.workspace_path()) {
            let enter_error = anyhow::Error::new(error).context(format!(
                "enter worktree workspace `{}`",
                lease.workspace_path().display()
            ));
            if let Err(cleanup_error) = lease.finish() {
                eprintln!("worktree: cleanup failed: {cleanup_error:#}");
            }
            return Err(enter_error);
        }
        let application_result = run_application(cli, Some(session_root)).await;
        if let Err(error) = std::env::set_current_dir(&launch_directory) {
            let retained_path = lease.release_without_cleanup();
            let restore_error = anyhow::Error::new(error).context(format!(
                "leave worktree and restore `{}`; retained `{}`",
                launch_directory.display(),
                retained_path.display()
            ));
            if application_result.is_ok() {
                return Err(restore_error);
            }
            eprintln!("worktree: cleanup failed: {restore_error:#}");
            return application_result;
        }
        match lease.finish() {
            Ok(worktree::CleanupOutcome::Removed(path)) if report_lifecycle => {
                eprintln!("worktree: removed {}", path.display());
            }
            Ok(worktree::CleanupOutcome::Kept(path)) if report_lifecycle => {
                eprintln!("worktree: kept {}", path.display());
            }
            Ok(worktree::CleanupOutcome::Removed(_) | worktree::CleanupOutcome::Kept(_)) => {}
            Ok(worktree::CleanupOutcome::Retained { path, reason }) => {
                eprintln!(
                    "worktree: retained {} because it could not be safely removed: {reason}",
                    path.display()
                );
            }
            Err(cleanup_error) if application_result.is_ok() => return Err(cleanup_error),
            Err(cleanup_error) => {
                eprintln!("worktree: cleanup failed: {cleanup_error:#}");
            }
        }
        return application_result;
    }

    run_application(cli, None).await
}

/// `--resume-session` is global, but prompt mode reads its own `--resume`,
/// so the global flag was silently ignored there. Treat it as the prompt's
/// `--resume`, and reject combinations that cannot both apply.
fn apply_global_resume_to_prompt(cli: &mut Cli) -> Result<()> {
    let Some(Command::Prompt(args)) = &mut cli.command else {
        return Ok(());
    };
    let Some(global) = cli.resume.take() else {
        return Ok(());
    };
    match &args.resume {
        Some(local) if *local != global => {
            anyhow::bail!("--resume-session `{global}` conflicts with prompt --resume `{local}`")
        }
        _ => {}
    }
    anyhow::ensure!(
        !args.no_session,
        "--resume-session cannot be used with --no-session"
    );
    args.resume = Some(global);
    Ok(())
}

/// Make relative path overrides absolute against the launch directory, so
/// they keep their meaning after worktree mode changes the cwd. Empty values
/// mean "unset" to every reader and are left alone (joining them used to
/// turn them into the cwd). `XDG_CONFIG_HOME` is included because config
/// and auth paths derive from it.
///
/// Must run before any other thread exists (see `main`).
fn absolutize_process_path_overrides() -> Result<()> {
    let mut current = None;
    for variable in [
        "HARNESS_CONFIG_DIR",
        "HARNESS_LOG",
        "HARNESS_SESSION_DIR",
        "HARNESS_STATE_DIR",
        "XDG_CONFIG_HOME",
    ] {
        let Some(value) = std::env::var_os(variable) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let path = std::path::PathBuf::from(value);
        if path.is_absolute() {
            continue;
        }
        let current = match &current {
            Some(current) => current,
            None => current.insert(std::env::current_dir().context("resolve launch directory")?),
        };
        let absolute = current.join(path);
        // SAFETY: called from `main` before the Tokio runtime (or any other
        // thread) starts, so no concurrent environment access is possible.
        unsafe { std::env::set_var(variable, absolute) };
    }
    Ok(())
}

async fn run_application(
    mut cli: Cli,
    session_root: Option<std::path::PathBuf>,
) -> Result<ExitCode> {
    apply_global_resume_to_prompt(&mut cli)?;
    let prompt_args = match &cli.command {
        Some(Command::Prompt(args)) => Some(args),
        _ => None,
    };
    // Resolve one-shot input before constructing the session store, registry,
    // or context bundle. A blank/invalid prompt must not leave startup state
    // behind, and stdin must only be consumed once.
    let resolved_prompt = prompt_args.map(headless::resolve_prompt).transpose()?;
    // Whether any part of this run touches the session store: interactive
    // mode always persists; prompt mode skips it under `--no-session`.
    let needs_session_store = prompt_args.is_none_or(|args| !args.no_session);

    init_logging()?;
    // Startup tracing: one log line per stage so "fast" stays measurable and
    // regressions show up in any HARNESS_LOG file. All stages share one
    // monotonic clock; each line reports time since process start.
    let started = std::time::Instant::now();
    let since_start = || started.elapsed().as_millis() as u64;
    // ACP sessions use only the MCP servers the editor declares. Replace the
    // local list with an empty one, so a broken or unexpandable local server
    // (for example an unset ${VAR}) cannot stop ACP from starting.
    let config: Config = if matches!(&cli.command, Some(Command::Acp)) {
        Config::resolve_with_mcp(&cli, Some(&[]))?
    } else {
        Config::resolve(&cli)?
    };
    tracing::info!(stage = "config", elapsed_ms = since_start());

    let workspace_root =
        std::fs::canonicalize(std::env::current_dir().with_context(|| "resolve workspace root")?)?;
    tracing::info!(stage = "workspace", elapsed_ms = since_start());

    // Reuse auth handles loaded during config resolution instead of re-reading
    // auth.json. OAuth providers remain constructible without credentials so
    // their local model catalogs work before login.
    let copilot_auth = config.copilot_auth.clone();
    let provider_factory = provider_factory(
        config.custom_providers.clone(),
        copilot_auth.clone(),
        config.codex_auth.clone(),
        config.anthropic_auth.clone(),
    );
    let provider: Arc<dyn Provider> = build_provider_with_auths(
        &config.provider.to_string(),
        &config.custom_providers,
        copilot_auth.clone(),
        config.codex_auth.clone(),
        config.anthropic_auth.clone(),
    )?;
    let provider_name = provider.name().to_owned();

    // ACP is the third frontend: same provider/config setup, but the process
    // becomes a stdio JSON-RPC server and never touches the terminal. The
    // workspace root stays the launch directory; per-session roots come from
    // each `session/new` request, so none of the process-cwd startup work
    // below (registry, store, context files) is built.
    if matches!(&cli.command, Some(Command::Acp)) {
        return acp::run(provider, config, copilot_auth, cli.no_context_files).await;
    }

    // These are synchronous filesystem operations, so `join!` would only add
    // ceremony: there are no await points to overlap. Discover context once
    // and reuse its rendered prompt and display paths.
    let context_bundle =
        context::load_context_bundle(&workspace_root, cli.no_context_files, prompt_args.is_none());
    tracing::debug!(
        bytes = context_bundle.rendered.len(),
        "project context rendered"
    );
    let tools = default_registry(ToolConfig::new(&workspace_root, config.rtk))?;
    // Under `--no-session` the store is never constructed: even its first-write
    // salt bootstrap would touch `~/.harness/sessions`.
    let session_store = if needs_session_store {
        Some(match session_root {
            Some(root) => SessionStore::new(root, &workspace_root),
            None => SessionStore::default_for_workspace(&workspace_root),
        })
    } else {
        None
    }
    .transpose()?
    .map(|store| store.with_deferred_sync(cli.defer_session_sync));
    tracing::info!(stage = "registry+store+context", elapsed_ms = since_start());

    if let (Some(args), Some(prompt)) = (prompt_args, resolved_prompt) {
        return run_headless_with_prompt(
            &config,
            args,
            provider,
            tools,
            session_store,
            prompt,
            context_bundle.rendered,
            cli.no_context_files,
            cli.defer_session_sync,
        )
        .await;
    }

    // Which skills were discovered; built from the registry-owned catalog
    // once, here, so the TUI only renders the names.
    let skill_entries = tools
        .skills()
        .map(|catalog| {
            catalog
                .entries()
                .into_iter()
                .map(|skill| tui::SkillEntry {
                    name: skill.name,
                    description: skill.description,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Interactive mode always has a store (enforced by needs_session_store
    // being true without the prompt subcommand); headless may run without one.
    let session_store = session_store.expect("interactive mode always builds the session store");

    let session = if let Some(selector) = &cli.resume {
        session_store
            .load(selector)
            .with_context(|| format!("load session `{selector}`"))?
    } else {
        session_store.create(SessionCreateOptions {
            provider: Some(provider_name.clone()),
            model: Some(config.model.clone()),
            ..SessionCreateOptions::default()
        })?
    };
    let session_id = session.id().to_string();
    let providers = ProviderArg::ALL
        .iter()
        .map(ToString::to_string)
        .chain(config.custom_providers.keys().cloned())
        .collect::<Vec<_>>();
    let cancel = CancellationToken::new();
    let (tui_input_tx, tui_input_rx) = mpsc::unbounded_channel();
    let (runtime_input_tx, runtime_input_rx): (
        mpsc::UnboundedSender<InputMessage>,
        mpsc::UnboundedReceiver<InputMessage>,
    ) = mpsc::unbounded_channel();
    let (runtime_event_tx, mut runtime_event_rx) = mpsc::unbounded_channel();
    // Bounded: a TUI that falls behind makes the forwarder wait (and merge
    // deltas) instead of letting the channel grow without limit.
    let (ui_event_tx, ui_event_rx) = mpsc::channel(tui_adapter::UI_EVENT_CAPACITY);

    // This catalog is local metadata, so a saved Claude login can populate
    // /model even when another provider is active. Do not refresh tokens or
    // make a network request just to display these suggestions.
    if let Ok(auth) = auth::AnthropicSubscriptionAuth::from_default()
        && auth.credential().ok().flatten().is_some()
    {
        let provider = llm::providers::AnthropicSubscriptionProvider::new(Arc::new(auth));
        if let Ok(models) = provider.list_models().await {
            let _ = ui_event_tx.try_send(tui::UiEvent::ModelList {
                provider: provider.name().into(),
                models: models
                    .into_iter()
                    .map(|model| tui::ModelEntry {
                        id: model.id,
                        name: model.name,
                        context_length: model.context_length,
                    })
                    .collect(),
            });
        }
    }

    // Construct the terminal before assembling the agent or claiming the
    // Herdr pane, so setup errors leave neither running MCPs nor a report.
    let update_notice = update::check_latest().await;
    let mut ui = CrossTerm::new(
        &config.model,
        &provider_name,
        providers,
        StartupEntries {
            skills: skill_entries,
            context_files: context_bundle
                .display_paths
                .into_iter()
                .map(|path| ContextFileEntry { path })
                .collect(),
            mcp_servers: config
                .mcp_servers
                .iter()
                .map(|server| server.name.clone())
                .collect(),
            update_notice,
        },
        config.reasoning.as_str(),
        config.tui_minimal,
        workspace_root,
    )?;
    ui.set_provider_aliases(ProviderArg::alias_names());
    let mut builder = AgentBuilder::new(provider, config.model.clone(), tools, cancel.clone())
        .with_secret_masker(config.secret_masker.clone())
        .with_reasoning(config.reasoning)
        .with_project_context(context_bundle.rendered)
        .with_compaction(config.compaction.clone())
        .with_subagents(config.subagents, config.rtk)
        .with_mcp_servers(config.mcp_servers.clone())
        .with_session(session_store, session)
        .with_prewalk(config.prewalk.settings(cli.resume.is_some())?)
        .with_provider_factory(provider_factory);
    if let Some(tools) = &config.code_mode {
        builder = builder.with_code_mode(tools.clone());
    }
    let agent = builder.build().await?;

    let herdr_options = herdr::ResumeOptions {
        provider: provider_name.clone(),
        model: config.model.clone(),
        reasoning: config.reasoning.as_str().into(),
        no_context_files: cli.no_context_files,
        defer_session_sync: cli.defer_session_sync,
    };
    let herdr = herdr::Herdr::from_env();
    let herdr_reporter = herdr.as_ref().map(|h| h.start());
    if let Some((reporter, _)) = &herdr_reporter {
        reporter.with_session(session_id, &herdr_options);
    }

    let input_reporter = herdr_reporter
        .as_ref()
        .map(|(reporter, _)| reporter.clone());
    let input_task = tokio::spawn(tui_adapter::forward_inputs(
        tui_input_rx,
        runtime_input_tx,
        move |input| {
            if matches!(
                input,
                tui::InputMessage::Message(_)
                    | tui::InputMessage::Ask { .. }
                    | tui::InputMessage::InvokeSkill { .. }
                    | tui::InputMessage::CompactSession
            ) && let Some(reporter) = &input_reporter
            {
                reporter.report(herdr::State::Working, None);
            }
        },
    ));
    let event_herdr_reporter = herdr_reporter
        .as_ref()
        .map(|(reporter, _)| reporter.clone());
    let event_task = tokio::spawn(async move {
        let mut herdr_options = herdr_options;
        let mut pending = tui_adapter::PendingUiEvents::default();
        loop {
            tokio::select! {
                biased;
                // Keep draining the agent so its side effects (config saves,
                // herdr reports) stay current; UI events wait in `pending`.
                event = runtime_event_rx.recv() => {
                    let Some(event) = event else { break };
                    // Config saves write and fsync a file. This task also
                    // forwards every UI event, so keep the disk wait off the
                    // async worker (main runs the multi-thread runtime).
                    tokio::task::block_in_place(|| {
                        if let AgentEvent::ModelChanged { provider, model } = &event
                            && let Err(error) = save_settings(provider, model)
                        {
                            tracing::warn!(error = %error, "could not persist model settings");
                        }
                        if let AgentEvent::ReasoningChanged { level } = &event
                            && let Ok(reasoning) = level.parse()
                            && let Err(error) = save_reasoning(reasoning)
                        {
                            tracing::warn!(error = %error, "could not persist reasoning setting");
                        }
                        // Only commands change preferences. `on` saves the target
                        // only when none is saved, so the config stays valid.
                        if let AgentEvent::PrewalkChanged { armed, target, reasoning, change } = &event {
                            let saved = match change {
                                agent::PrewalkChange::Enabled => {
                                    save_prewalk_enabled(*armed, target.as_deref())
                                }
                                agent::PrewalkChange::Target => target
                                    .as_deref()
                                    .map_or(Ok(()), save_prewalk_model),
                                agent::PrewalkChange::Reasoning => reasoning
                                    .as_deref()
                                    .map(str::parse)
                                    .transpose()
                                    .map_err(|error: String| anyhow::anyhow!(error))
                                    .and_then(save_prewalk_reasoning),
                                agent::PrewalkChange::Session => Ok(()),
                            };
                            if let Err(error) = saved {
                                tracing::warn!(error = %error, "could not persist prewalk setting");
                            }
                        }
                    });
                    if let Some(reporter) = &event_herdr_reporter {
                        reporter.observe(&event, &mut herdr_options);
                    }
                    pending.push(tui_adapter::into_ui_event(event));
                    if !pending.flush_ready(&ui_event_tx) {
                        break;
                    }
                }
                permit = ui_event_tx.reserve(), if !pending.is_empty() => {
                    let Ok(permit) = permit else { break };
                    pending.send_one(permit);
                }
                _ = ui_event_tx.closed() => break,
            }
        }
        pending.finish(&ui_event_tx).await;
    });
    let agent_task = tokio::spawn(agent.run(runtime_input_rx, runtime_event_tx));

    // The crossterm frontend drives the same agent as headless mode and
    // differs only in rendering and input handling. The first paint happens
    // inside `CrossTerm::run`, so this is the end of the startup path.
    tracing::info!(stage = "pre-first-frame", elapsed_ms = since_start());
    let ui_result = ui.run(ui_event_rx, tui_input_tx, cancel.clone()).await;
    cancel.cancel();
    let _ = input_task.await;
    let agent_result = agent_task.await;
    let _ = event_task.await;
    if let Some((reporter, task)) = herdr_reporter {
        drop(reporter);
        let _ = task.await;
        if let Some(herdr) = herdr {
            herdr.release().await;
        }
    }
    ui_result?;
    // A panicking agent closes its event channel, which ends the TUI like a
    // normal quit. Report it (after the terminal is restored) and exit
    // non-zero instead of claiming success.
    agent_task_outcome(agent_result)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::{Cli, agent_task_outcome, apply_global_resume_to_prompt};
    use clap::Parser;

    #[test]
    fn global_resume_session_applies_to_prompt_mode() {
        let mut cli = Cli::parse_from(["harness", "--resume-session", "latest", "prompt", "hi"]);
        apply_global_resume_to_prompt(&mut cli).unwrap();
        let Some(super::Command::Prompt(args)) = &cli.command else {
            panic!("prompt command expected");
        };
        assert_eq!(args.resume.as_deref(), Some("latest"));

        let mut cli = Cli::parse_from([
            "harness",
            "--resume-session",
            "latest",
            "prompt",
            "--no-session",
            "hi",
        ]);
        assert!(apply_global_resume_to_prompt(&mut cli).is_err());
    }

    #[tokio::test]
    async fn agent_panic_is_reported_instead_of_success() {
        // The panic message is printed by the default hook; the hook is
        // process-global, so it is not replaced in a parallel test run.
        let result = tokio::spawn(async { panic!("agent exploded") }).await;
        let error = agent_task_outcome(result).unwrap_err().to_string();
        assert!(error.contains("agent exploded"), "{error}");
        assert!(agent_task_outcome(Ok(())).is_ok());
    }
}
