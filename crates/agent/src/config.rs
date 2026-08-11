use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use std::fmt;
use std::fs::OpenOptions;

#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum ProviderArg {
    #[value(name = "opencode-go")]
    OpencodeGo,
    #[value(name = "openrouter")]
    Openrouter,
}

impl ProviderArg {
    /// The first name is the preferred one. The additional OpenCode name is
    /// supported for compatibility with existing OpenCode installations.
    pub fn env_vars(&self) -> &'static [&'static str] {
        match self {
            Self::OpencodeGo => &["OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"],
            Self::Openrouter => &["OPENROUTER_API_KEY"],
        }
    }

    pub fn env_var(&self) -> &'static str {
        self.env_vars()[0]
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Self::OpencodeGo => "gpt-5.6-luna",
            Self::Openrouter => "openai/gpt-5.6-luna",
        }
    }
}

impl fmt::Display for ProviderArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OpencodeGo => "opencode-go",
            Self::Openrouter => "openrouter",
        })
    }
}

#[derive(Clone, Debug, Parser)]
#[command(name = "harness", about = "A minimal coding-agent harness")]
pub struct Cli {
    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,
    #[arg(long)]
    pub model: Option<String>,
}

#[derive(Clone)]
pub struct Config {
    pub provider: ProviderArg,
    pub model: String,
    pub api_key: String,
}

impl Config {
    pub fn from_cli(cli: &Cli) -> Result<Self> {
        Self::resolve(cli)
    }

    pub fn resolve(cli: &Cli) -> Result<Self> {
        let provider = cli.provider.clone().unwrap_or(ProviderArg::OpencodeGo);
        let key = provider
            .env_vars()
            .iter()
            .find_map(|name| {
                std::env::var(name)
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .ok_or_else(|| anyhow!("missing API key: set {}", provider.env_vars().join(" or ")))?;
        Ok(Self {
            model: cli
                .model
                .clone()
                .unwrap_or_else(|| provider.default_model().to_owned()),
            provider,
            api_key: key,
        })
    }

    /// Testable form of resolution using the preferred key names.
    pub fn resolve_with_keys(
        cli: &Cli,
        opencode_key: Option<&str>,
        openrouter_key: Option<&str>,
    ) -> Result<Self> {
        Self::resolve_with_key_values(cli, opencode_key, None, openrouter_key)
    }

    /// Testable form that also covers the legacy `OPENCODE_API_KEY` fallback.
    pub fn resolve_with_key_values(
        cli: &Cli,
        opencode_go_key: Option<&str>,
        opencode_key: Option<&str>,
        openrouter_key: Option<&str>,
    ) -> Result<Self> {
        let provider = cli.provider.clone().unwrap_or(ProviderArg::OpencodeGo);
        let key = match provider {
            ProviderArg::OpencodeGo => opencode_go_key.or(opencode_key),
            ProviderArg::Openrouter => openrouter_key,
        }
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("missing API key: set {}", provider.env_vars().join(" or ")))?;
        Ok(Self {
            provider: provider.clone(),
            model: cli
                .model
                .clone()
                .unwrap_or_else(|| provider.default_model().to_owned()),
            api_key: key.to_owned(),
        })
    }
}

/// Install file-only tracing when requested.  stdout is intentionally left
/// untouched because the inline terminal owns it.
pub fn init_logging() -> Result<()> {
    let Some(path) = std::env::var_os("HARNESS_LOG") else {
        return Ok(());
    };
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    tracing_subscriber::fmt()
        .with_writer(file)
        .with_ansi(false)
        .try_init()
        .map_err(|error| anyhow!("could not initialize logging: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_defaults_and_overrides() {
        let cli = Cli {
            provider: None,
            model: None,
        };
        let config = Config::resolve_with_keys(&cli, Some("secret"), None).unwrap();
        assert_eq!(config.provider, ProviderArg::OpencodeGo);
        assert_eq!(config.model, "gpt-5.6-luna");

        let cli = Cli {
            provider: Some(ProviderArg::Openrouter),
            model: Some("anthropic/demo".into()),
        };
        let config = Config::resolve_with_keys(&cli, None, Some("router-secret")).unwrap();
        assert_eq!(config.model, "anthropic/demo");
    }

    #[test]
    fn supports_legacy_opencode_key_name() {
        let cli = Cli {
            provider: Some(ProviderArg::OpencodeGo),
            model: None,
        };
        let config =
            Config::resolve_with_key_values(&cli, None, Some("legacy-secret"), None).unwrap();
        assert_eq!(config.api_key, "legacy-secret");
    }

    #[test]
    fn missing_key_names_environment_variables() {
        let cli = Cli {
            provider: Some(ProviderArg::Openrouter),
            model: None,
        };
        let error = Config::resolve_with_keys(&cli, None, None).err().unwrap();
        assert!(error.to_string().contains("OPENROUTER_API_KEY"));

        let cli = Cli {
            provider: Some(ProviderArg::OpencodeGo),
            model: None,
        };
        let error = Config::resolve_with_key_values(&cli, None, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("OPENCODE_GO_API_KEY"));
        assert!(error.to_string().contains("OPENCODE_API_KEY"));
    }
}
