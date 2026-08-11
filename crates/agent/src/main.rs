use agent::agent::Agent;
use agent::config::{Cli, Config, ProviderArg, init_logging};
use agent::tools::default_registry;
use clap::Parser;
use llm::Provider;
use llm::providers::{OpenCodeGoProvider, OpenRouterProvider};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::Tui;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_logging()?;
    let config = Config::resolve(&cli)?;

    let provider: Arc<dyn Provider> = match config.provider {
        ProviderArg::OpencodeGo => Arc::new(OpenCodeGoProvider::new(config.api_key.clone())),
        ProviderArg::Openrouter => Arc::new(OpenRouterProvider::new(config.api_key.clone())),
    };
    let provider_name = provider.name().to_owned();
    let cancel = CancellationToken::new();
    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    let agent = Agent::new(
        provider,
        default_registry(),
        config.model.clone(),
        cancel.clone(),
    );
    let agent_task = tokio::spawn(agent.run(input_rx, event_tx));

    let tui = Tui::new(&config.model, &provider_name)?;
    let result = tui.run(event_rx, input_tx, cancel.clone()).await;
    cancel.cancel();
    let _ = agent_task.await;
    result
}
