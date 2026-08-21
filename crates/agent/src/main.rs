use agent::agent::{Agent, spawn_model_list};
use agent::config::{Cli, Config, ProviderArg, build_provider_with_auth, init_logging};
use agent::headless::run_headless;
use agent::project_context_for;
use agent::tools::{ToolConfig, default_registry};
use anyhow::{Context, Result, bail};
use auth::CopilotAuth;
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

    // `--print`-only flags have no meaning in the interactive TUI, which
    // exposes the same capabilities through its own slash commands.
    if !cli.print && !cli.prompt.is_empty() {
        bail!("prompt requires --print");
    }
    if !cli.print && cli.resume.is_some() {
        bail!("--resume requires --print");
    }

    init_logging()?;
    let config: Config = Config::resolve(&cli)?;

    // Keep one auth handle shared by the provider and agent.  Other providers
    // do not touch auth.json, while Copilot remains constructible without a
    // credential so the first `/auth` can run.
    let copilot_auth = if config.provider == ProviderArg::GithubCopilot {
        Some(Arc::new(CopilotAuth::from_default()?))
    } else {
        None
    };
    let provider: Arc<dyn Provider> =
        build_provider_with_auth(&config.provider.to_string(), copilot_auth.clone())?;
    let provider_name = provider.name().to_owned();
    let workspace_root =
        std::fs::canonicalize(std::env::current_dir().with_context(|| "resolve workspace root")?)?;
    let tools = default_registry(ToolConfig::new(&workspace_root, config.rtk))?;
    // UI-facing views of the auto-loaded context: which AGENTS.md / CLAUDE.md
    // files were injected and which skills were discovered. Built once here
    // because the registry owns the catalog; the TUI only renders the names.
    let context_file_paths = if cli.no_context_files {
        Vec::new()
    } else {
        tools::load_context_files(&workspace_root)
            .iter()
            .map(|file| tools::display_path(&file.path))
            .collect::<Vec<_>>()
    };
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
    let session_store = SessionStore::default_for_workspace(&workspace_root)?;

    if cli.print {
        return run_headless(&config, &cli, provider, tools, session_store).await;
    }

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

    // Prime completion as soon as the UI starts.  A failure is informational;
    // ordinary model entry and conversation use do not depend on this fetch.
    spawn_model_list(provider_name.clone(), provider.clone(), event_tx.clone());

    let agent = Agent::new(provider, tools, config.model.clone(), cancel.clone())
        .with_project_context(project_context_for(&workspace_root, cli.no_context_files));
    let agent = if let Some(auth) = copilot_auth {
        agent.with_copilot_auth(auth)
    } else {
        agent
    };
    let agent = agent
        .with_compaction(config.compaction.clone())
        .with_session(session_store, session);
    let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

    // The crossterm frontend drives the same agent as headless mode and
    // differs only in rendering and input handling.
    let ui = CrossTerm::new(
        &config.model,
        &provider_name,
        providers,
        skill_entries,
        context_file_paths
            .into_iter()
            .map(|path| ContextFileEntry { path })
            .collect(),
    )?;
    ui.run(event_rx, input_tx, cancel.clone()).await?;
    cancel.cancel();
    let _ = agent_task.await;
    Ok(ExitCode::SUCCESS)
}
