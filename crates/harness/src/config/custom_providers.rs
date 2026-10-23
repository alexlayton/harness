//! User-defined providers configured under `[providers.<name>]`.
//!
//! Kept apart from the rest of `config.toml` handling because these tables
//! own their own validation, construction, and secret handling.

use super::{ProviderArg, is_env_var_name};
use anyhow::{Result, anyhow};
use llm::Provider;
use llm::providers::OpenAiCompatibleProvider;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A user-defined provider under `[providers.<name>]`, selected with
/// `provider = "<name>"`.
///
/// Secrets stay out of `config.toml`, matching the built-in providers:
/// `api_key_env` and `env_headers` name environment variables, and only
/// `headers` holds literal (non-secret) values.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomProviderConfig {
    #[serde(rename = "type")]
    pub kind: CustomProviderKind,
    /// API root including any version segment, e.g.
    /// `http://localhost:1234/v1`; `/chat/completions` and `/models` are
    /// appended to it.
    pub base_url: String,
    /// Environment variable holding the bearer token. Unset means no
    /// `Authorization` header is sent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Model used when neither `--model` nor `model` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Context window for models whose `/models` entry reports none. Local
    /// servers rarely report one, and the global fallback is very large, so
    /// without this compaction could start too late.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Literal headers sent with every request.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Header name to the environment variable holding its value.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_headers: BTreeMap<String, String>,
}

/// Wire protocol of a user-defined provider.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CustomProviderKind {
    /// OpenAI Chat Completions (LM Studio, Ollama, vLLM, gateways, ...).
    #[serde(rename = "openai-compatible")]
    OpenAiCompatible,
}

/// User-defined providers keyed by name, shared with the provider factory so
/// model switches can rebuild them.
pub type CustomProviders = Arc<BTreeMap<String, CustomProviderConfig>>;

impl CustomProviderConfig {
    /// Check everything that does not need the environment, so mistakes fail
    /// at load time instead of on the first request.
    pub(super) fn validate(&self, name: &str) -> Result<()> {
        let valid_name = !name.is_empty()
            && name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character));
        if !valid_name {
            return Err(anyhow!(
                "[providers.{name}]: provider names may contain only ASCII letters, digits, `-`, `_` and `.`"
            ));
        }
        if ProviderArg::from_name(name).is_some() {
            return Err(anyhow!(
                "[providers.{name}]: `{name}` is a built-in provider name; choose another name"
            ));
        }
        let url = reqwest::Url::parse(&self.base_url)
            .map_err(|error| anyhow!("[providers.{name}].base_url is not a valid URL: {error}"))?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(anyhow!(
                "[providers.{name}].base_url must be an http:// or https:// URL with a host"
            ));
        }
        if let Some(variable) = &self.api_key_env
            && !is_env_var_name(variable)
        {
            return Err(anyhow!(
                "[providers.{name}].api_key_env is not a valid environment variable name: `{variable}`"
            ));
        }
        if self
            .default_model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(anyhow!(
                "[providers.{name}].default_model must not be blank"
            ));
        }
        if self.context_window == Some(0) {
            return Err(anyhow!(
                "[providers.{name}].context_window must be greater than zero"
            ));
        }
        let mut seen = std::collections::HashSet::new();
        let headers = self.headers.keys().chain(self.env_headers.keys());
        for header in headers {
            let parsed = HeaderName::from_bytes(header.as_bytes()).map_err(|_| {
                anyhow!("[providers.{name}]: `{header}` is not a valid HTTP header name")
            })?;
            if parsed == reqwest::header::CONTENT_TYPE {
                return Err(anyhow!(
                    "[providers.{name}]: the `{header}` header is set by harness"
                ));
            }
            if parsed == reqwest::header::AUTHORIZATION && self.api_key_env.is_some() {
                return Err(anyhow!(
                    "[providers.{name}]: set either `api_key_env` or an `{header}` header, not both"
                ));
            }
            if !seen.insert(parsed) {
                return Err(anyhow!(
                    "[providers.{name}]: header `{header}` is configured more than once"
                ));
            }
        }
        for (header, value) in &self.headers {
            HeaderValue::from_str(value).map_err(|_| {
                anyhow!("[providers.{name}].headers: `{header}` has an invalid header value")
            })?;
        }
        for (header, variable) in &self.env_headers {
            if !is_env_var_name(variable) {
                return Err(anyhow!(
                    "[providers.{name}].env_headers: `{header}` names an invalid environment variable `{variable}`"
                ));
            }
        }
        Ok(())
    }
}

/// Construct a user-defined provider, reading its secrets from the
/// environment. Errors name variables and headers but never values.
pub(super) fn build_custom_provider(
    name: &str,
    config: &CustomProviderConfig,
) -> Result<Arc<dyn Provider>> {
    let read = |variable: &str| {
        std::env::var(variable)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "provider `{name}` requires environment variable `{variable}`, which is not set"
                )
            })
    };
    let api_key = config
        .api_key_env
        .as_deref()
        .map(read)
        .transpose()?
        .unwrap_or_default();
    let mut headers = HeaderMap::new();
    for (header, value) in &config.headers {
        let header = HeaderName::from_bytes(header.as_bytes())
            .map_err(|_| anyhow!("provider `{name}`: invalid header name `{header}`"))?;
        let value = HeaderValue::from_str(value)
            .map_err(|_| anyhow!("provider `{name}`: invalid value for header `{header}`"))?;
        headers.insert(header, value);
    }
    for (header, variable) in &config.env_headers {
        let header = HeaderName::from_bytes(header.as_bytes())
            .map_err(|_| anyhow!("provider `{name}`: invalid header name `{header}`"))?;
        let mut value = HeaderValue::from_str(&read(variable)?).map_err(|_| {
            anyhow!("provider `{name}`: `{variable}` is not a valid value for header `{header}`")
        })?;
        value.set_sensitive(true);
        headers.insert(header, value);
    }
    Ok(match config.kind {
        CustomProviderKind::OpenAiCompatible => Arc::new(OpenAiCompatibleProvider::new(
            name,
            &config.base_url,
            api_key,
            headers,
            config.context_window,
        )),
    })
}

/// Add the credentials of user-defined providers to the secret masker
/// entries so tool output (e.g. `env`) cannot show them to the model.
///
/// Unset or short values are skipped: a missing key fails only when that
/// provider is built, and the masker rejects very short values. Names that
/// `[secrets].env` already lists are not added twice.
pub(super) fn add_secret_entries(
    providers: &BTreeMap<String, CustomProviderConfig>,
    entries: &mut Vec<(String, String)>,
) {
    for config in providers.values() {
        let variables = config.api_key_env.iter().chain(config.env_headers.values());
        for variable in variables {
            if entries.iter().any(|(name, _)| name == variable) {
                continue;
            }
            if let Ok(value) = std::env::var(variable)
                && value.len() >= agent::secrets::MIN_SECRET_BYTES
            {
                entries.push((variable.clone(), value));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Cli, Config, FileConfig, ProviderChoice};
    use std::path::PathBuf;

    #[test]
    fn custom_providers_resolve_by_name_and_reject_invalid_tables() {
        let file: FileConfig = toml::from_str(
            r#"
provider = "lmstudio"

[providers.lmstudio]
type = "openai-compatible"
base_url = "http://localhost:1234/v1"
default_model = "qwen3-coder"

[providers.lmstudio.headers]
X-Client = "harness"
"#,
        )
        .unwrap();
        let path = PathBuf::from("/tmp/harness-config.toml");
        // No key is needed, and the provider default fills in the model.
        let config =
            Config::resolve_from_file(&Cli::default(), &file, path.clone(), |_| None).unwrap();
        assert_eq!(config.provider, ProviderChoice::Custom("lmstudio".into()));
        assert_eq!(config.model, "qwen3-coder");
        assert!(config.custom_providers.contains_key("lmstudio"));

        // The CLI may select a custom name; unknown names still fail.
        let cli = Cli {
            provider: Some("lmstudio".into()),
            model: Some("other".into()),
            ..Cli::default()
        };
        let config = Config::resolve_from_file(&cli, &file, path.clone(), |_| None).unwrap();
        assert_eq!(config.model, "other");
        let cli = Cli {
            provider: Some("missing".into()),
            ..Cli::default()
        };
        assert!(Config::resolve_from_file(&cli, &file, path.clone(), |_| None).is_err());

        for (table, expected) in [
            (
                "[providers.openrouter]\ntype = \"openai-compatible\"\nbase_url = \"http://x\"",
                "built-in provider name",
            ),
            (
                "[providers.x]\ntype = \"openai-compatible\"\nbase_url = \"localhost:1234\"",
                "base_url",
            ),
            (
                "[providers.x]\ntype = \"openai-compatible\"\nbase_url = \"http://x\"\napi_key_env = \"K\"\nheaders = { authorization = \"y\" }",
                "not both",
            ),
            (
                "[providers.x]\ntype = \"openai-compatible\"\nbase_url = \"http://x\"\nenv_headers = { X-Token = \"bad-name\" }",
                "invalid environment variable",
            ),
        ] {
            let file: FileConfig = toml::from_str(table).unwrap();
            let error = file.validate().unwrap_err().to_string();
            assert!(error.contains(expected), "{table}: {error}");
        }
        assert!(
            toml::from_str::<FileConfig>(
                "[providers.x]\ntype = \"other\"\nbase_url = \"http://x\""
            )
            .is_err()
        );
    }
}
