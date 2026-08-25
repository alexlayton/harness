use agent::acp;
use agent::agent::spawn_model_list;
use agent::assembly::AgentBuilder;
use agent::config::{Cli, Command, Config, ProviderArg, build_provider_with_auths, init_logging};
use agent::headless::run_headless;
use agent::project_context_for;
use agent::tools::{ToolConfig, default_registry};
use anyhow::{Context, Result, bail};
use clap::Parser;
use llm::Provider;
use session::{SessionCreateOptions, SessionStore};
use std::process::ExitCode;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::{ContextFileEntry, CrossTerm, InputMessage};

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
    let cli = Cli::parse();

    // Login is credential-only and intentionally precedes config/provider
    // resolution: a stale configured API key cannot prevent signing in.
    if let Some(Command::Login(args)) = &cli.command {
        return agent::login::run(args).await;
    }

    // `--print`-only flags have no meaning in the interactive TUI, which
    // exposes the same capabilities through its own slash commands.
    if !cli.print && !cli.prompt.is_empty() {
        bail!("prompt requires --print");
    }
    if !cli.print && cli.resume.is_some() {
        bail!("--resume requires --print");
    }
    if !cli.print && cli.no_session {
        bail!("--no-session requires --print");
    }
    if cli.no_session && cli.resume.is_some() {
        bail!("--resume conflicts with --no-session");
    }
    // Whether any part of this run touches the session store: interactive
    // mode always persists; headless mode skips it under `--no-session`.
    let needs_session_store = !cli.print || !cli.no_session;

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
    if cli.acp {
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
                true => Some(SessionStore::default_for_workspace(&workspace_root)),
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

    if cli.print {
        return run_headless(&config, &cli, provider, tools, session_store).await;
    }

    // Interactive mode always has a store (enforced by needs_session_store
    // being true when !cli.print); headless may run without one.
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
    let (input_tx, input_rx): (
        mpsc::UnboundedSender<InputMessage>,
        mpsc::UnboundedReceiver<InputMessage>,
    ) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    // A model-list failure is informational and must not delay the first UI
    // frame or agent construction.
    spawn_model_list(provider_name.clone(), provider.clone(), event_tx.clone());

    let mut builder = AgentBuilder::new(provider, config.model.clone(), tools, cancel.clone())
        .with_project_context(project_context)
        .with_compaction(config.compaction.clone())
        .with_subagents(config.subagents, config.rtk)
        .with_mcp_servers(config.mcp_servers.clone())
        .with_session(session_store, session);
    if let Some(auth) = copilot_auth {
        builder = builder.with_copilot_auth(auth);
    }
    let agent = builder.build().await?;

    let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

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
    )?;
    ui.run(event_rx, input_tx, cancel.clone()).await?;
    cancel.cancel();
    let _ = agent_task.await;
    Ok(ExitCode::SUCCESS)
}
