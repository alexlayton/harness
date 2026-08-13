use agent::agent::{Agent, AgentEvent};
use agent::config::{Cli, Config, ProviderArg, build_provider_with_auth, init_logging};
use agent::tools::{ToolConfig, default_registry};
use auth::CopilotAuth;
use clap::Parser;
use llm::Provider;
use session::{SessionCreateOptions, SessionStore};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::{InputMessage, Tui};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_logging()?;
    let config = Config::resolve(&cli)?;

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
    let workspace_root = std::fs::canonicalize(std::env::current_dir()?)?;
    let tools = default_registry(ToolConfig::new(&workspace_root, config.rtk))?;
    let session_store = SessionStore::default_for_workspace(&workspace_root)?;
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
    let startup_provider = provider.clone();
    let startup_events = event_tx.clone();
    let startup_name = provider_name.clone();
    tokio::spawn(async move {
        match startup_provider.list_models().await {
            Ok(models) => {
                let _ = startup_events.send(AgentEvent::ModelList {
                    provider: startup_name,
                    models,
                });
            }
            Err(error) => {
                let _ = startup_events.send(AgentEvent::Notice(format!(
                    "could not fetch model list: {error}"
                )));
            }
        }
    });

    let agent = Agent::new(provider, tools, config.model.clone(), cancel.clone());
    let agent = if let Some(auth) = copilot_auth {
        agent.with_copilot_auth(auth)
    } else {
        agent
    };
    let agent = agent.with_session(session_store, session);
    let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

    let tui = Tui::new(&config.model, &provider_name, providers)?;
    let result = tui.run(event_rx, input_tx, cancel.clone()).await;
    cancel.cancel();
    let _ = agent_task.await;
    result
}
