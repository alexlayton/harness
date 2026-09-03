mod acp;
mod config;
mod context;
mod headless;
mod login;
mod tui_adapter;
mod worktree;

use agent::assembly::AgentBuilder;
use agent::{AgentEvent, InputMessage, spawn_model_list};
use anyhow::{Context, Result};
use clap::Parser;
use config::{
    Cli, Command, Config, ProviderArg, WorktreeCommand, build_provider_with_auths, init_logging,
    provider_factory, save_reasoning, save_settings,
};
use context::project_context_for;
use headless::run_headless;
use llm::Provider;
use session::{SessionCreateOptions, SessionStore};
use std::process::ExitCode;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tools::{ToolConfig, default_registry};
use tui::{ContextFileEntry, CrossTerm};

#[tokio::main]
async fn main() -> ExitCode {
    match main_inner().await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn main_inner() -> Result<ExitCode> {
    let mut cli = Cli::parse();

    // Login is credential-only and intentionally precedes config/provider
    // resolution: a stale configured API key cannot prevent signing in.
    if let Some(Command::Login(args)) = &cli.command {
        return login::run(args).await;
    }

    // Worktree is a launch mode rather than a separate frontend. Enter it
    // before resolving any workspace-scoped tools, context, or session state,
    // then always restore the launch directory and attempt safe cleanup after
    // application startup or shutdown.
    if let Some(Command::Worktree(args)) = cli.command.clone() {
        // Relative state/session overrides must retain launch-directory
        // semantics after prepare changes the process cwd.
        let session_root = std::path::absolute(session::default_session_dir())
            .context("resolve the session state directory")?;
        let report_lifecycle = args
            .command
            .as_ref()
            .is_none_or(|WorktreeCommand::Prompt(prompt)| prompt.verbose);
        let lease = worktree::prepare(&args)?;
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

        let application_result = run_application(cli, Some(session_root)).await;
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

async fn run_application(cli: Cli, session_root: Option<std::path::PathBuf>) -> Result<ExitCode> {
    let prompt_args = match &cli.command {
        Some(Command::Prompt(args)) => Some(args),
        _ => None,
    };
    // Whether any part of this run touches the session store: interactive
    // mode always persists; prompt mode skips it under `--no-session`.
    let needs_session_store = prompt_args.is_none_or(|args| !args.no_session);

    init_logging()?;
    // Startup tracing: one log line per stage so "fast" stays measurable and
    // regressions show up in any HARNESS_LOG file. All stages share one
    // monotonic clock; each line reports time since process start.
    let started = std::time::Instant::now();
    let since_start = || started.elapsed().as_millis() as u64;
    let config: Config = Config::resolve(&cli)?;
    tracing::info!(stage = "config", elapsed_ms = since_start());

    // Reuse auth handles loaded during config resolution instead of re-reading
    // auth.json. OAuth providers remain constructible without credentials so
    // their local model catalogs work before login.
    let copilot_auth = config.copilot_auth.clone();
    let provider_factory = provider_factory(copilot_auth.clone(), config.codex_auth.clone());
    let provider: Arc<dyn Provider> = build_provider_with_auths(
        &config.provider.to_string(),
        copilot_auth.clone(),
        config.codex_auth.clone(),
    )?;
    let provider_name = provider.name().to_owned();
    let workspace_root =
        std::fs::canonicalize(std::env::current_dir().with_context(|| "resolve workspace root")?)?;
    tracing::info!(stage = "workspace", elapsed_ms = since_start());

    // ACP is the third frontend: same provider/config setup, but the process
    // becomes a stdio JSON-RPC server and never touches the terminal. The
    // workspace root stays the launch directory; per-session roots come from
    // each `session/new` request, so none of the process-cwd startup work
    // below (registry, store, context files) is built.
    if matches!(&cli.command, Some(Command::Acp)) {
        return acp::run(provider, config, copilot_auth, cli.no_context_files).await;
    }

    // Independent startup work runs concurrently: skills discovery, context
    // files, and the session store are all filesystem walks/hashing that
    // don't depend on each other or the tool registry.  On a cold cache these
    // each cost real I/O, so serializing them delayed the first frame.  The
    // FFF index is no longer built here at all — it is created lazily on
    // first `find`/`grep` call.
    let (tools_result, session_store_result, project_context, context_files) = tokio::join!(
        async { default_registry(ToolConfig::new(&workspace_root, config.rtk)) },
        // Under `--no-session` the store is never constructed: even its
        // first-write salt bootstrap would touch `~/.harness/sessions`.
        async {
            match needs_session_store {
                true => Some(match session_root {
                    Some(root) => SessionStore::new(root, &workspace_root),
                    None => SessionStore::default_for_workspace(&workspace_root),
                }),
                false => None,
            }
        },
        async {
            let context = project_context_for(&workspace_root, cli.no_context_files);
            tracing::debug!(bytes = context.len(), "project context rendered");
            context
        },
        // UI-facing list of which AGENTS.md / CLAUDE.md files were injected;
        // the TUI only renders the names.
        async {
            if cli.no_context_files {
                Vec::new()
            } else {
                tools::load_context_files(&workspace_root)
                    .iter()
                    .map(|file| tools::display_path(&file.path))
                    .collect::<Vec<_>>()
            }
        },
    );
    let tools = tools_result?;

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
    let session_store = session_store_result
        .transpose()?
        .map(|store| store.with_deferred_sync(cli.defer_session_sync));
    tracing::info!(stage = "registry+store+context", elapsed_ms = since_start());

    if let Some(args) = prompt_args {
        return run_headless(
            &config,
            args,
            cli.no_context_files,
            provider,
            tools,
            session_store,
        )
        .await;
    }

    // Interactive mode always has a store (enforced by needs_session_store
    // being true without the prompt subcommand); headless may run without one.
    let session_store = session_store.expect("interactive mode always builds the session store");

    let session = session_store.create(SessionCreateOptions {
        provider: Some(provider_name.clone()),
        model: Some(config.model.clone()),
        ..SessionCreateOptions::default()
    })?;

    let providers = ProviderArg::ALL
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let cancel = CancellationToken::new();
    let (tui_input_tx, tui_input_rx) = mpsc::unbounded_channel();
    let (runtime_input_tx, runtime_input_rx): (
        mpsc::UnboundedSender<InputMessage>,
        mpsc::UnboundedReceiver<InputMessage>,
    ) = mpsc::unbounded_channel();
    let (runtime_event_tx, mut runtime_event_rx) = mpsc::unbounded_channel();
    let (ui_event_tx, ui_event_rx) = mpsc::unbounded_channel();

    // A model-list failure is informational and must not delay the first UI
    // frame or agent construction.
    spawn_model_list(
        provider_name.clone(),
        provider.clone(),
        runtime_event_tx.clone(),
    );

    let builder = AgentBuilder::new(provider, config.model.clone(), tools, cancel.clone())
        .with_reasoning(config.reasoning)
        .with_project_context(project_context)
        .with_compaction(config.compaction.clone())
        .with_subagents(config.subagents, config.rtk)
        .with_mcp_servers(config.mcp_servers.clone())
        .with_session(session_store, session)
        .with_provider_factory(provider_factory);
    let agent = builder.build().await?;

    let input_task = tokio::spawn(tui_adapter::forward_inputs(tui_input_rx, runtime_input_tx));
    let event_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                event = runtime_event_rx.recv() => {
                    let Some(event) = event else { break };
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
                    if ui_event_tx.send(tui_adapter::into_ui_event(event)).is_err() {
                        break;
                    }
                }
                _ = ui_event_tx.closed() => break,
            }
        }
    });
    let agent_task = tokio::spawn(agent.run(runtime_input_rx, runtime_event_tx));

    // The crossterm frontend drives the same agent as headless mode and
    // differs only in rendering and input handling. The first paint happens
    // inside `CrossTerm::run`, so this is the end of the startup path.
    tracing::info!(stage = "pre-first-frame", elapsed_ms = since_start());
    let ui = CrossTerm::new(
        &config.model,
        &provider_name,
        providers,
        skill_entries,
        context_files
            .into_iter()
            .map(|path| ContextFileEntry { path })
            .collect(),
        config.reasoning.as_str(),
        config.tui_minimal,
    )?;
    let ui_result = ui.run(ui_event_rx, tui_input_tx, cancel.clone()).await;
    cancel.cancel();
    let _ = input_task.await;
    let _ = agent_task.await;
    let _ = event_task.await;
    ui_result?;
    Ok(ExitCode::SUCCESS)
}
