use agent::assembly::SubagentPolicy;
use agent::secrets::SecretMasker;
use agent::{PrewalkSettings, PrewalkTarget};
use anyhow::{Context, Result, anyhow};
use auth::{AnthropicSubscriptionAuth, OpenAiCodexAuth};
use auth::{CopilotAuth, sku_from_proxy_token};
use clap::{Parser, ValueEnum};
use compact::policy::CompactionPolicy;
use fs2::FileExt;
use llm::providers::github_copilot::default_model_for;
use llm::providers::{
    AnthropicSubscriptionProvider, GithubCopilotProvider, OpenAiCodexProvider,
    OpenAiCompatibleProvider, OpenCodeGoProvider, OpenRouterProvider,
};
use llm::{Provider, ReasoningPolicy};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ProviderArg {
    #[value(name = "opencode-go")]
    OpencodeGo,
    #[value(name = "openrouter")]
    Openrouter,
    #[value(name = "github-copilot", alias = "copilot")]
    GithubCopilot,
    #[value(name = "openai-codex", alias = "codex")]
    OpenAiCodex,
    #[value(name = "anthropic-claude", aliases = ["anthropic-subscription", "claude-sub", "claude"])]
    AnthropicSubscription,
}

impl ProviderArg {
    /// All provider names understood by the command line, configuration file,
    /// and command completion UI.
    pub const ALL: &[ProviderArg] = &[
        Self::OpencodeGo,
        Self::Openrouter,
        Self::GithubCopilot,
        Self::OpenAiCodex,
        Self::AnthropicSubscription,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "opencode-go" => Some(Self::OpencodeGo),
            "openrouter" => Some(Self::Openrouter),
            "github-copilot" | "githubcopilot" | "copilot" => Some(Self::GithubCopilot),
            "anthropic-claude" | "anthropic-subscription" | "claude-sub" | "claude" => {
                Some(Self::AnthropicSubscription)
            }
            "openai-codex" | "codex" => Some(Self::OpenAiCodex),
            _ => None,
        }
    }

    /// The first name is the preferred one. The additional OpenCode name is
    /// supported for compatibility with existing OpenCode installations.
    /// The first name is the preferred one. The additional OpenCode name is
    /// supported for compatibility with existing OpenCode installations.
    pub fn env_vars(&self) -> &'static [&'static str] {
        match self {
            Self::OpencodeGo => &["OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"],
            Self::Openrouter => &["OPENROUTER_API_KEY"],
            // Copilot credentials are stored in auth.json rather than in the
            // environment or config.toml.  This value is retained as a
            // compatibility hint for embedders that display provider help;
            // resolution itself does not require it.
            Self::GithubCopilot => &["COPILOT_GITHUB_TOKEN"],
            Self::OpenAiCodex | Self::AnthropicSubscription => &[],
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Self::OpencodeGo => "gpt-5.6-luna",
            Self::Openrouter => "openai/gpt-5.6-luna",
            Self::GithubCopilot => "gpt-5.4",
            Self::OpenAiCodex => "gpt-5.5",
            Self::AnthropicSubscription => "claude-sonnet-4-6",
        }
    }
}

impl fmt::Display for ProviderArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OpencodeGo => "opencode-go",
            Self::Openrouter => "openrouter",
            Self::GithubCopilot => "github-copilot",
            Self::OpenAiCodex => "openai-codex",
            Self::AnthropicSubscription => "anthropic-claude",
        })
    }
}

/// The provider selected for this process: a built-in integration or a
/// user-defined `[providers.<name>]` endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderChoice {
    Builtin(ProviderArg),
    Custom(String),
}

impl ProviderChoice {
    /// The built-in integration, or `None` for a user-defined provider.
    pub fn builtin(&self) -> Option<ProviderArg> {
        match self {
            Self::Builtin(provider) => Some(*provider),
            Self::Custom(_) => None,
        }
    }
}

/// Lets callers ask "is this Copilot?" without unwrapping the choice.
impl PartialEq<ProviderArg> for ProviderChoice {
    fn eq(&self, other: &ProviderArg) -> bool {
        self.builtin() == Some(*other)
    }
}

impl fmt::Display for ProviderChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Builtin(provider) => provider.fmt(formatter),
            Self::Custom(name) => formatter.write_str(name),
        }
    }
}

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
    fn validate(&self, name: &str) -> Result<()> {
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
fn build_custom_provider(name: &str, config: &CustomProviderConfig) -> Result<Arc<dyn Provider>> {
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

/// Shell-style environment variable name: `[A-Za-z_][A-Za-z0-9_]*`.
fn is_env_var_name(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

/// Per-key compaction settings parsed from `[compaction]` in `config.toml`.
/// Every field is optional so a partial table overrides only what it sets;
/// the rest fall back to [`CompactionPolicy::default`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CompactConfig {
    pub auto: Option<bool>,
    /// Trigger fraction in the inclusive range `0.0..=1.0`.
    pub threshold: Option<f64>,
    pub reserve_tokens: Option<u64>,
    /// Zero is valid and means that no minimum number of recent turns is
    /// retained by policy; the token backstop still applies.
    pub keep_recent_turns: Option<usize>,
    /// Zero is valid and means that the token backstop retains the smallest
    /// valid provider-history tail.
    pub keep_recent_tokens: Option<u64>,
    pub max_summary_input_bytes: Option<usize>,
    pub max_summary_bytes: Option<usize>,
    pub context_window: Option<u64>,
}

/// Per-key subagent settings parsed from `[subagents]` in `config.toml`.
/// Every field is optional so a partial table overrides only what it sets;
/// the rest fall back to [`SubagentPolicy::default`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SubagentConfig {
    /// Turn budget for one nested run. `0` disables subagents entirely.
    pub max_turns: Option<usize>,
    /// How many subagents may run at once when several are delegated in one
    /// response.
    pub max_concurrent: Option<usize>,
}

/// Opt-in code mode settings parsed from `[code_mode]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeModeConfig {
    /// Expose selected tools through `run_code` only when explicitly enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Tool names to expose; an empty list selects all eligible tools.
    #[serde(default)]
    pub tools: Vec<String>,
}

/// One-shot planning-to-implementation handoff settings parsed from `[prewalk]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrewalkConfig {
    /// Arm prewalk for each new conversation. Resumed sessions stay unarmed.
    #[serde(default)]
    pub enabled: bool,
    /// Handoff target as `[<provider>:]<model>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Target reasoning effort; absent keeps the active policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningPolicy>,
}

impl PrewalkConfig {
    fn validate(&self) -> Result<()> {
        match &self.model {
            Some(model) => {
                parse_prewalk_target(model).context("[prewalk].model")?;
            }
            None if self.enabled => {
                return Err(anyhow!("[prewalk].enabled requires [prewalk].model"));
            }
            None => {}
        }
        Ok(())
    }
}

/// Parse `[<provider>:]<model>`. A prefix is a provider only when Harness
/// knows it, so model ids such as `vendor/model:free` keep their colon.
pub fn parse_prewalk_target(value: &str) -> Result<PrewalkTarget> {
    let value = value.trim();
    if value.is_empty() {
        return Err(anyhow!("prewalk model must not be blank"));
    }
    if let Some((provider, model)) = value.split_once(':')
        && let Some(provider) = ProviderArg::from_name(provider)
    {
        if model.is_empty() {
            return Err(anyhow!("prewalk model must not be blank"));
        }
        return Ok(PrewalkTarget {
            provider: Some(provider.to_string()),
            model: model.to_owned(),
        });
    }
    Ok(PrewalkTarget {
        provider: None,
        model: value.to_owned(),
    })
}

/// Resolved prewalk preference plus any explicit startup request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrewalkPolicy {
    /// Persisted preference for new conversations.
    pub enabled: bool,
    pub target: Option<PrewalkTarget>,
    /// Target reasoning; `None` keeps the active policy.
    pub reasoning: Option<ReasoningPolicy>,
    /// Explicit `--prewalk[=BOOL]` or `--prewalk-model` request.
    pub requested: Option<bool>,
}

impl PrewalkPolicy {
    /// Agent settings for the startup conversation. The preference arms new
    /// sessions only; an explicit request also arms a resumed one.
    pub fn settings(&self, resumed: bool) -> Result<PrewalkSettings> {
        let armed = self.requested.unwrap_or(self.enabled && !resumed);
        if armed && self.target.is_none() {
            return Err(anyhow!(
                "prewalk requires a target model: set [prewalk].model or pass --prewalk-model"
            ));
        }
        Ok(PrewalkSettings {
            enabled: self.enabled,
            target: self.target.clone(),
            armed,
            reasoning: self.reasoning,
        })
    }
}

/// Terminal UI presentation settings parsed from `[tui]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TuiConfig {
    /// Start directly at the input field without printing the welcome header.
    #[serde(default)]
    pub minimal: bool,
}

/// Explicit environment variables whose values are masked from model and
/// durable conversation content.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecretsConfig {
    #[serde(default)]
    pub env: Vec<String>,
}

impl SecretsConfig {
    fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for name in &self.env {
            if !is_env_var_name(name) {
                return Err(anyhow!(
                    "[secrets].env contains invalid environment variable name `{name}`"
                ));
            }
            if !seen.insert(name) {
                return Err(anyhow!(
                    "[secrets].env contains duplicate environment variable `{name}`"
                ));
            }
        }
        Ok(())
    }
}

fn resolve_subagents(config: &SubagentConfig) -> SubagentPolicy {
    let defaults = SubagentPolicy::default();
    SubagentPolicy {
        max_turns: config.max_turns.unwrap_or(defaults.max_turns),
        max_concurrent: config
            .max_concurrent
            .unwrap_or(defaults.max_concurrent)
            .max(1),
    }
}

const MAX_SUMMARY_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_SUMMARY_BYTES: usize = 1024 * 1024;

impl CompactConfig {
    /// Validate user-provided compaction values before they reach the planner.
    /// The key names in these errors intentionally match `config.toml` so a
    /// malformed setting can be fixed without guessing which field failed.
    pub fn validate(&self) -> Result<()> {
        if let Some(threshold) = self.threshold
            && (!threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
        {
            return Err(anyhow!(
                "[compaction].threshold must be finite and between 0.0 and 1.0"
            ));
        }
        if let Some(limit) = self.max_summary_input_bytes
            && (limit == 0 || limit > MAX_SUMMARY_INPUT_BYTES)
        {
            return Err(anyhow!(
                "[compaction].max_summary_input_bytes must be between 1 and {MAX_SUMMARY_INPUT_BYTES}"
            ));
        }
        if let Some(limit) = self.max_summary_bytes
            && (limit == 0 || limit > MAX_SUMMARY_BYTES)
        {
            return Err(anyhow!(
                "[compaction].max_summary_bytes must be between 1 and {MAX_SUMMARY_BYTES}"
            ));
        }
        let defaults = CompactionPolicy::default();
        let reserve = self.reserve_tokens.unwrap_or(defaults.reserve_tokens);
        let window = self.context_window.unwrap_or(defaults.context_window);
        if window > 0 && reserve >= window {
            return Err(anyhow!(
                "[compaction].reserve_tokens must be less than [compaction].context_window"
            ));
        }
        Ok(())
    }
}

impl From<&CompactConfig> for CompactionPolicy {
    fn from(config: &CompactConfig) -> Self {
        let defaults = CompactionPolicy::default();
        Self {
            auto: config.auto.unwrap_or(defaults.auto),
            threshold: config.threshold.unwrap_or(defaults.threshold),
            reserve_tokens: config.reserve_tokens.unwrap_or(defaults.reserve_tokens),
            keep_recent_turns: config
                .keep_recent_turns
                .unwrap_or(defaults.keep_recent_turns),
            keep_recent_tokens: config
                .keep_recent_tokens
                .unwrap_or(defaults.keep_recent_tokens),
            max_summary_input_bytes: config
                .max_summary_input_bytes
                .unwrap_or(defaults.max_summary_input_bytes),
            max_summary_bytes: config
                .max_summary_bytes
                .unwrap_or(defaults.max_summary_bytes),
            context_window: config.context_window.unwrap_or(defaults.context_window),
        }
    }
}

/// Settings persisted by harness.  API keys deliberately do not belong here;
/// they remain environment-only secrets.
///
/// The flattened map makes a config loaded by an older harness round-trip
/// fields added by a newer one instead of silently deleting them on save.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FileConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    /// Portable reasoning effort for normal agent and subagent requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningPolicy>,
    /// Opt-in RTK integration: rewrite supported bash commands to their
    /// token-optimized `rtk` equivalents before execution. Off unless the
    /// user explicitly sets `rtk = true`.
    #[serde(default)]
    pub rtk: bool,
    /// Compact settings. Anything the compiler cannot answer comes from
    /// [`CompactionPolicy::default`]. Absent from disk when unset, so older
    /// harness versions treat the config as unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactConfig>,
    /// Subagent bounds (`[subagents]`). Same layering contract as
    /// `compaction`: absent from disk when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagents: Option<SubagentConfig>,
    /// Opt-in code mode settings (`[code_mode]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_mode: Option<CodeModeConfig>,
    /// Exact-value secret masking configured under `[secrets]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secrets: Option<SecretsConfig>,
    /// External MCP servers configured under `[[mcp.servers]]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<mcp::McpConfig>,
    /// Terminal UI presentation settings (`[tui]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tui: Option<TuiConfig>,
    /// Prewalk handoff settings (`[prewalk]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prewalk: Option<PrewalkConfig>,
    /// User-defined providers (`[providers.<name>]`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, CustomProviderConfig>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

// `toml::Value` has value equality but does not promise `Eq` because TOML can
// contain floating point values.  Configuration equality is still useful in
// tests and the fields are compared using TOML's value equality semantics.
impl PartialEq for FileConfig {
    fn eq(&self, other: &Self) -> bool {
        self.provider == other.provider
            && self.model == other.model
            && self.reasoning_effort == other.reasoning_effort
            && self.rtk == other.rtk
            && self.compaction == other.compaction
            && self.subagents == other.subagents
            && self.code_mode == other.code_mode
            && self.secrets == other.secrets
            && self.mcp == other.mcp
            && self.tui == other.tui
            && self.prewalk == other.prewalk
            && self.providers == other.providers
            && self.extra == other.extra
    }
}

impl Eq for FileConfig {}

/// Return harness's configuration directory.
///
/// `HARNESS_CONFIG_DIR` is intentionally checked first so tests and isolated
/// launches do not need to touch the user's home directory.  XDG is honored
/// for Linux-style installations, while the fallback remains explicitly
/// `~/.config/harness` rather than the platform-specific `dirs::config_dir`.
pub fn config_dir() -> PathBuf {
    if let Some(path) = non_empty_env_path("HARNESS_CONFIG_DIR") {
        return path;
    }
    if let Some(path) = non_empty_env_path("XDG_CONFIG_HOME") {
        return path.join("harness");
    }
    dirs::home_dir()
        .map(|home| home.join(".config").join("harness"))
        .unwrap_or_else(|| PathBuf::from(".config").join("harness"))
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Load a TOML configuration.  A missing file is the same as an empty config.
pub fn load_file_config(path: &Path) -> Result<FileConfig> {
    load_file_config_with_mcp(path, None)
}

// Replace the entire persisted MCP section before deserialization, validation,
// and environment expansion. An explicit empty list must be able to disable
// stale servers, including entries whose credentials are no longer available.
fn load_file_config_with_mcp(
    path: &Path,
    servers: Option<&[mcp::McpServerConfig]>,
) -> Result<FileConfig> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("read config file {}", path.display()));
        }
    };
    let Some(servers) = servers else {
        return parse_file_config(&contents, path);
    };
    let mut document: toml::Table = toml::from_str(&contents)
        .with_context(|| format!("parse config file {}", path.display()))?;
    document.insert(
        "mcp".into(),
        toml::Value::try_from(mcp::McpConfig {
            servers: servers.to_vec(),
        })?,
    );
    let config: FileConfig = document
        .try_into()
        .with_context(|| format!("parse config file {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

fn parse_file_config(contents: &str, path: &Path) -> Result<FileConfig> {
    let config: FileConfig = toml::from_str(contents)
        .with_context(|| format!("parse config file {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

impl FileConfig {
    /// Validate nested settings before they are used or persisted.
    pub fn validate(&self) -> Result<()> {
        if let Some(compaction) = &self.compaction {
            compaction.validate()?;
        }
        if let Some(secrets) = &self.secrets {
            secrets.validate()?;
        }
        if let Some(mcp) = &self.mcp {
            mcp.validate().map_err(|error| anyhow!(error))?;
        }
        if let Some(prewalk) = &self.prewalk {
            prewalk.validate()?;
        }
        for (name, provider) in &self.providers {
            provider.validate(name)?;
        }
        Ok(())
    }

    /// Resolve a provider name against the built-ins, then `[providers]`.
    pub fn provider_choice(&self, name: &str) -> Option<ProviderChoice> {
        ProviderArg::from_name(name)
            .map(ProviderChoice::Builtin)
            .or_else(|| {
                self.providers
                    .contains_key(name)
                    .then(|| ProviderChoice::Custom(name.to_owned()))
            })
    }
}

/// Save a TOML configuration using a temporary file in the target directory,
/// followed by rename, so a partially-written config is never observed.
///
/// The advisory lock covers the complete write, including the rename. The
/// settings-specific helpers below use the same lock while editing the raw
/// document, so concurrent read-modify-write operations cannot lose a field.
#[cfg_attr(not(test), allow(dead_code))]
pub fn save_file_config(path: &Path, config: &FileConfig) -> Result<()> {
    let _lock = lock_config(path)?;
    config.validate()?;
    let contents = toml::to_string_pretty(config).context("serialize config")?;
    write_config_contents(path, &contents)
}

fn lock_config(path: &Path) -> Result<std::fs::File> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create config directory {}", parent.display()))?;
    let lock_path = path.with_file_name(format!(
        ".{}.lock",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("config.toml")
    ));
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("open config lock {}", lock_path.display()))?;
    lock.lock_exclusive()
        .with_context(|| format!("lock config {}", path.display()))?;
    Ok(lock)
}

fn write_config_contents(path: &Path, contents: &str) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create config directory {}", parent.display()))?;
    let temp_path = temporary_path(path);
    let write_result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options
            .open(&temp_path)
            .with_context(|| format!("create temporary config {}", temp_path.display()))?;
        file.write_all(contents.as_bytes())
            .with_context(|| format!("write temporary config {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("flush temporary config {}", temp_path.display()))?;
        drop(file);
        fs::rename(&temp_path, path)
            .with_context(|| format!("replace config file {}", path.display()))?;
        sync_config_parent(parent)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
}

#[cfg(unix)]
fn sync_config_parent(parent: &Path) -> Result<()> {
    std::fs::File::open(parent)
        .with_context(|| format!("open config directory {} for syncing", parent.display()))?
        .sync_all()
        .with_context(|| format!("sync config directory {}", parent.display()))
}

#[cfg(not(unix))]
fn sync_config_parent(_parent: &Path) -> Result<()> {
    Ok(())
}

fn update_config_document(
    path: &Path,
    update: impl FnOnce(&mut toml_edit::DocumentMut) -> Result<()>,
) -> Result<()> {
    let _lock = lock_config(path)?;
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("read config file {}", path.display()));
        }
    };
    let mut document: toml_edit::DocumentMut = contents
        .parse()
        .with_context(|| format!("parse config file {}", path.display()))?;
    update(&mut document)?;
    let rendered = document.to_string();
    parse_file_config(&rendered, path)?;
    write_config_contents(path, &rendered)
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One-time per-process entropy used to make temp-file names unique across
/// processes.  PIDs recycle quickly on some systems, so a recycled PID alone
/// could collide with a stale temp file left by an earlier process; the
/// random component eliminates that risk.
static TEMP_ENTROPY: OnceLock<u64> = OnceLock::new();

fn temporary_path(path: &Path) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config.toml");
    path.with_file_name(format!(
        ".{file_name}.tmp-{}-{sequence}-{:08x}",
        std::process::id(),
        temp_random(sequence)
    ))
}

/// 32 bits of process-lifetime randomness derived from a one-time seed
/// mixed with the per-call sequence.  Cheap, deterministic, and sufficient
/// for file-name entropy; not used for security.
fn temp_random(sequence: u64) -> u32 {
    let entropy = *TEMP_ENTROPY.get_or_init(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        // ASLR provides cheap per-process address-space entropy.
        let address = &now as *const u64 as u64;
        now ^ (std::process::id() as u64).rotate_left(32) ^ address
    });
    splitmix64(entropy.wrapping_add(sequence)) as u32
}

/// A tiny splitmix64 finalizer.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Update the two currently persisted settings while retaining any unknown
/// fields already present in the file.
pub fn save_settings(provider: &str, model: &str) -> Result<()> {
    save_settings_at(&config_path(), provider, model)
}

/// Path-injectable form used by callers that already resolved a config path
/// and by tests.
pub fn save_settings_at(path: &Path, provider: &str, model: &str) -> Result<()> {
    update_config_document(path, |document| {
        document["provider"] = toml_edit::value(provider);
        document["model"] = toml_edit::value(model);
        Ok(())
    })
}

/// Persist a reasoning policy while retaining provider, model, and unknown
/// settings already present in the file.
pub fn save_reasoning(reasoning: ReasoningPolicy) -> Result<()> {
    save_reasoning_at(&config_path(), reasoning)
}

/// Path-injectable reasoning save used by tests.
pub fn save_reasoning_at(path: &Path, reasoning: ReasoningPolicy) -> Result<()> {
    update_config_document(path, |document| {
        document["reasoning_effort"] = toml_edit::value(reasoning.as_str());
        Ok(())
    })
}

/// Persist `/prewalk on|off` while retaining other settings. `fallback_model`
/// is written only when no model is saved, so enabling keeps the config valid
/// without replacing a saved target with a process-only `--prewalk-model`.
pub fn save_prewalk_enabled(enabled: bool, fallback_model: Option<&str>) -> Result<()> {
    save_prewalk_enabled_at(&config_path(), enabled, fallback_model)
}

/// Path-injectable form of [`save_prewalk_enabled`].
pub fn save_prewalk_enabled_at(
    path: &Path,
    enabled: bool,
    fallback_model: Option<&str>,
) -> Result<()> {
    update_prewalk_table(path, |table| {
        if let Some(model) = fallback_model
            && !table.contains_key("model")
        {
            table.insert("model", toml_edit::value(model));
        }
        table.insert("enabled", toml_edit::value(enabled));
    })
}

/// Persist `/prewalk-model` while retaining other settings.
pub fn save_prewalk_model(model: &str) -> Result<()> {
    save_prewalk_model_at(&config_path(), model)
}

/// Path-injectable form of [`save_prewalk_model`].
pub fn save_prewalk_model_at(path: &Path, model: &str) -> Result<()> {
    update_prewalk_table(path, |table| {
        table.insert("model", toml_edit::value(model));
    })
}

fn update_prewalk_table(
    path: &Path,
    update: impl FnOnce(&mut dyn toml_edit::TableLike),
) -> Result<()> {
    update_config_document(path, |document| {
        let table = document
            .entry("prewalk")
            .or_insert_with(toml_edit::table)
            .as_table_like_mut()
            .ok_or_else(|| anyhow!("[prewalk] in config is not a table"))?;
        update(table);
        Ok(())
    })
}

/// Persist `/prewalk-reasoning`; `None` removes the key so the handoff keeps
/// the active reasoning policy.
pub fn save_prewalk_reasoning(reasoning: Option<ReasoningPolicy>) -> Result<()> {
    save_prewalk_reasoning_at(&config_path(), reasoning)
}

/// Path-injectable prewalk reasoning save used by tests.
pub fn save_prewalk_reasoning_at(path: &Path, reasoning: Option<ReasoningPolicy>) -> Result<()> {
    update_prewalk_table(path, |table| match reasoning {
        Some(reasoning) => {
            table.insert("reasoning_effort", toml_edit::value(reasoning.as_str()));
        }
        None => {
            table.remove("reasoning_effort");
        }
    })
}

/// Select an OAuth provider after login when the user has not already made a
/// choice. Login must never unexpectedly replace an existing provider.
pub fn select_provider_after_login(provider: ProviderArg) -> Result<bool> {
    select_provider_after_login_at(&config_path(), provider)
}

fn select_provider_after_login_at(path: &Path, provider: ProviderArg) -> Result<bool> {
    let selected = std::cell::Cell::new(false);
    update_config_document(path, |document| {
        if document.get("provider").is_none() {
            document["provider"] = toml_edit::value(provider.to_string());
            selected.set(true);
        }
        Ok(())
    })?;
    Ok(selected.get())
}

/// Add one stdio MCP server while retaining unrelated and unknown settings.
pub fn add_mcp_server(name: &str, command: &[String]) -> Result<()> {
    add_mcp_server_at(&config_path(), name, command)
}

fn add_mcp_server_at(path: &Path, name: &str, command: &[String]) -> Result<()> {
    let (executable, args) = command
        .split_first()
        .ok_or_else(|| anyhow!("an MCP server command is required"))?;
    let candidate = mcp::McpServerConfig {
        name: name.to_owned(),
        transport: mcp::McpTransportConfig::Stdio {
            command: executable.into(),
            args: args.to_vec(),
            env: BTreeMap::new(),
        },
    };
    mcp::McpConfig {
        servers: vec![candidate],
    }
    .validate()
    .map_err(|error| anyhow!(error))?;

    let mut server = toml_edit::Table::new();
    server["name"] = toml_edit::value(name);
    server["transport"] = toml_edit::value("stdio");
    server["command"] = toml_edit::value(executable);
    let mut arguments = toml_edit::Array::new();
    arguments.extend(args.iter().map(String::as_str));
    server["args"] = toml_edit::value(arguments);
    insert_mcp_server_at(path, name, server)
}

/// Add one Streamable HTTP MCP server while retaining unrelated settings.
pub fn add_http_mcp_server(
    name: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
) -> Result<()> {
    add_http_mcp_server_at(&config_path(), name, url, headers)
}

fn add_http_mcp_server_at(
    path: &Path,
    name: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
) -> Result<()> {
    let candidate = mcp::McpServerConfig {
        name: name.to_owned(),
        transport: mcp::McpTransportConfig::Http {
            url: url.to_owned(),
            headers: headers.clone(),
        },
    };
    mcp::McpConfig {
        servers: vec![candidate],
    }
    .validate()
    .map_err(|error| anyhow!(error))?;

    let mut server = toml_edit::Table::new();
    server["name"] = toml_edit::value(name);
    server["transport"] = toml_edit::value("http");
    server["url"] = toml_edit::value(url);
    if !headers.is_empty() {
        let mut header_table = toml_edit::Table::new();
        for (header, value) in headers {
            header_table[header] = toml_edit::value(value);
        }
        server["headers"] = toml_edit::Item::Table(header_table);
    }
    insert_mcp_server_at(path, name, server)
}

fn insert_mcp_server_at(path: &Path, name: &str, server: toml_edit::Table) -> Result<()> {
    update_config_document(path, |document| {
        if document.get("mcp").is_none() {
            document["mcp"] = toml_edit::Item::Table(toml_edit::Table::new());
        }
        let mcp = document["mcp"]
            .as_table_mut()
            .ok_or_else(|| anyhow!("`mcp` must be a table"))?;
        if mcp.get("servers").is_none() {
            mcp["servers"] = toml_edit::Item::ArrayOfTables(toml_edit::ArrayOfTables::new());
        }
        let servers = mcp["servers"]
            .as_array_of_tables_mut()
            .ok_or_else(|| anyhow!("`mcp.servers` must be an array of tables"))?;
        if servers
            .iter()
            .any(|server| server.get("name").and_then(|item| item.as_str()) == Some(name))
        {
            anyhow::bail!("MCP server `{name}` is already configured");
        }
        servers.push(server);
        Ok(())
    })
}

/// Delete one named MCP server while retaining unrelated and unknown settings.
pub fn delete_mcp_server(name: &str) -> Result<()> {
    delete_mcp_server_at(&config_path(), name)
}

fn delete_mcp_server_at(path: &Path, name: &str) -> Result<()> {
    update_config_document(path, |document| {
        let Some(servers) = document
            .get_mut("mcp")
            .and_then(toml_edit::Item::as_table_mut)
            .and_then(|mcp| mcp.get_mut("servers"))
            .and_then(toml_edit::Item::as_array_of_tables_mut)
        else {
            anyhow::bail!("MCP server `{name}` is not configured");
        };
        let Some(index) = servers
            .iter()
            .position(|server| server.get("name").and_then(|item| item.as_str()) == Some(name))
        else {
            anyhow::bail!("MCP server `{name}` is not configured");
        };
        servers.remove(index);
        if servers.is_empty() {
            let mcp = document["mcp"]
                .as_table_mut()
                .expect("MCP table was checked above");
            mcp.remove("servers");
            if mcp.is_empty() {
                document.remove("mcp");
            }
        }
        Ok(())
    })
}

#[derive(Clone, Debug, Default, Parser)]
#[command(name = "harness", version, about = "A minimal coding-agent harness")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Override the configured provider for this process: a built-in name
    /// or a `[providers.<name>]` entry from the config file.
    #[arg(long, value_name = "PROVIDER", global = true)]
    pub provider: Option<String>,

    /// Override the configured model for this process.
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Override model reasoning effort for this process.
    #[arg(long = "reasoning-effort", value_name = "LEVEL", global = true)]
    pub reasoning_effort: Option<ReasoningPolicy>,

    /// Arm prewalk for the startup conversation; `--prewalk=false` keeps it off.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        global = true
    )]
    pub prewalk: Option<bool>,

    /// Prewalk target as `[<provider>:]<model>`; also arms the startup conversation.
    #[arg(long = "prewalk-model", value_name = "MODEL", global = true)]
    pub prewalk_model: Option<String>,

    /// Override the reasoning effort that prewalk hands off with.
    #[arg(long = "prewalk-reasoning-effort", value_name = "LEVEL", global = true)]
    pub prewalk_reasoning_effort: Option<ReasoningPolicy>,

    /// Disable AGENTS.md / CLAUDE.md project-context injection.
    #[arg(long = "no-context-files", default_value_t = false, global = true)]
    pub no_context_files: bool,

    /// Defer session fsyncs to turn boundaries instead of syncing every
    /// persisted event. Faster chatty tool loops at the cost of losing the
    /// current turn's tail (not just the in-flight record) on power loss.
    #[arg(long = "defer-session-sync", default_value_t = false, global = true)]
    pub defer_session_sync: bool,

    /// Resume a persisted interactive session by id, unique prefix, `latest`, or path.
    #[arg(long = "resume-session", value_name = "ID|latest|PATH", global = true)]
    pub resume: Option<String>,
}

/// Frontend and authentication commands. With no command Harness starts the
/// interactive terminal UI.
#[derive(Clone, Debug, clap::Subcommand)]
pub enum Command {
    /// Download and install the latest compatible GitHub release.
    Update,
    /// Authenticate with an OAuth provider.
    Login(LoginArgs),
    /// Run one prompt and print only the final answer to stdout.
    Prompt(PromptArgs),
    /// Create or reuse a Git worktree and start a frontend there.
    Worktree(WorktreeArgs),
    /// Manage configured MCP servers.
    Mcp(McpArgs),
    /// Serve Agent Client Protocol over stdio for editor integrations.
    Acp,
}

#[derive(Clone, Debug, clap::Args)]
pub struct McpArgs {
    #[command(subcommand)]
    pub command: Option<McpCommand>,
}

#[derive(Clone, Debug, clap::Subcommand)]
pub enum McpCommand {
    /// Add a stdio or Streamable HTTP server.
    Add(McpAddArgs),
    /// Delete a server by name.
    #[command(alias = "remove", alias = "rm")]
    Delete { name: String },
    /// List configured servers.
    List,
}

#[derive(Clone, Debug, clap::Args)]
#[command(
    trailing_var_arg = true,
    after_help = "Examples:\n  harness mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem .\n  harness mcp add remote --url https://example.com/mcp --header 'Authorization=Bearer ${MCP_TOKEN}'"
)]
pub struct McpAddArgs {
    /// Stable name used to namespace this server's tools.
    pub name: String,
    /// Streamable HTTP endpoint instead of a stdio command.
    #[arg(long, value_name = "URL", conflicts_with = "command")]
    pub url: Option<String>,
    /// HTTP header as NAME=VALUE. Values may contain `${ENV_VAR}` placeholders.
    #[arg(long, value_name = "NAME=VALUE", requires = "url")]
    pub header: Vec<String>,
    /// Executable followed by its arguments. Use `--` before flags intended for the executable.
    #[arg(
        required_unless_present = "url",
        allow_hyphen_values = true,
        value_name = "COMMAND..."
    )]
    pub command: Vec<String>,
}

#[derive(Clone, Debug, clap::Args)]
pub struct WorktreeArgs {
    /// Local branch to create or check out.
    #[arg(value_name = "BRANCH")]
    pub branch: String,

    /// Commit-ish from which to create a missing branch. Defaults to HEAD.
    #[arg(long = "start-point", alias = "base", value_name = "REV")]
    pub start_point: Option<String>,

    /// Worktree directory. Relative paths are resolved from the launch directory.
    #[arg(long, value_name = "PATH")]
    pub dir: Option<PathBuf>,

    /// Retain this worktree across this and future Harness runs.
    #[arg(long, default_value_t = false, conflicts_with = "ephemeral")]
    pub keep: bool,

    /// Remove a clean worktree on exit even if an earlier run used --keep.
    #[arg(long, default_value_t = false)]
    pub ephemeral: bool,

    /// Optional non-interactive frontend to run in the worktree.
    #[command(subcommand)]
    pub command: Option<WorktreeCommand>,
}

/// Frontends that can sensibly inherit a prepared process worktree.
#[derive(Clone, Debug, clap::Subcommand)]
pub enum WorktreeCommand {
    /// Run one prompt in the worktree and print only the final answer to stdout.
    Prompt(PromptArgs),
}

#[derive(Clone, Debug, Default, clap::Args)]
pub struct PromptArgs {
    /// Stream reasoning and tool activity to stderr.
    #[arg(short = 'v', long = "verbose", default_value_t = false)]
    pub verbose: bool,

    /// Resume a session by id, unique prefix, `latest`, or file path.
    #[arg(long, value_name = "ID|latest|PATH")]
    pub resume: Option<String>,

    /// Run without persisting a session.
    #[arg(
        long = "no-session",
        default_value_t = false,
        conflicts_with = "resume"
    )]
    pub no_session: bool,

    /// Prompt text, joined with spaces. When absent, read it from stdin.
    #[arg(value_name = "PROMPT")]
    pub prompt: Vec<String>,
}

/// Credential-only login command. It is dispatched before configuration
/// resolution, so an unrelated configured API-key provider cannot block it.

#[derive(Clone, Debug, clap::Args)]
pub struct LoginArgs {
    // Keep this positional ID distinct from the global --provider option:
    // Clap stores them as different enum types and panics on a shared ID.
    #[arg(id = "login_provider", value_name = "PROVIDER", value_enum)]
    pub provider: LoginProvider,
    /// Use RFC 8628 device authorization instead of the local browser callback.
    #[arg(long)]
    pub device_code: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum LoginProvider {
    #[value(name = "github-copilot", alias = "copilot")]
    GithubCopilot,
    #[value(name = "openai-codex", alias = "codex")]
    OpenAiCodex,
    #[value(name = "anthropic-claude", aliases = ["anthropic-subscription", "claude-sub", "claude"])]
    AnthropicSubscription,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub provider: ProviderChoice,
    /// User-defined providers, needed again whenever the provider changes.
    pub custom_providers: CustomProviders,
    pub model: String,
    pub reasoning: ReasoningPolicy,
    pub rtk: bool,
    pub compaction: CompactionPolicy,
    /// Resolved subagent bounds (file `[subagents]` over defaults).
    pub subagents: SubagentPolicy,
    /// Selected code mode tools, present only when code mode is enabled.
    pub code_mode: Option<Vec<String>>,
    /// Resolved once at startup. Its `Debug` implementation omits values.
    pub secret_masker: Arc<SecretMasker>,
    /// MCP servers after deterministic validation and `${ENV_VAR}` expansion.
    pub mcp_servers: Vec<mcp::McpServerConfig>,
    /// Whether the interactive frontend skips its welcome header.
    pub tui_minimal: bool,
    /// Prewalk preference, target, and explicit startup request.
    pub prewalk: PrewalkPolicy,
    /// The Copilot credential handle loaded once during resolution. Startup
    /// used to construct `CopilotAuth::from_default()` twice — once for the
    /// entitled-model default inside `resolve`, again in `main` — reading and
    /// decrypting auth.json both times. `None` for non-Copilot providers.
    pub copilot_auth: Option<Arc<CopilotAuth>>,
    pub codex_auth: Option<Arc<OpenAiCodexAuth>>,
    /// Shared subscription credential lifecycle (loaded only for this provider).
    pub anthropic_auth: Option<Arc<AnthropicSubscriptionAuth>>,
}

impl Config {
    pub fn resolve(cli: &Cli) -> Result<Self> {
        Self::resolve_with_mcp(cli, None)
    }

    /// Resolve normal configuration with an optional replacement MCP list.
    ///
    /// `None` inherits the file; `Some(&[])` disables MCP. Overrides are applied
    /// before validating or expanding the persisted MCP section and never saved.
    pub fn resolve_with_mcp(cli: &Cli, servers: Option<&[mcp::McpServerConfig]>) -> Result<Self> {
        let path = config_path();
        let file = load_file_config_with_mcp(&path, servers)?;
        // One credential read serves both the entitled-model default below
        // and the provider/agent construction in `main`. Loaded only when the
        // *effective* provider (CLI override wins) is Copilot, matching the
        // historical behavior: a broken auth.json must not fail startup for
        // an unrelated `--provider` override.
        let effective_provider = cli
            .provider
            .as_deref()
            .or(file.provider.as_deref())
            .and_then(ProviderArg::from_name);
        let copilot_auth = (effective_provider == Some(ProviderArg::GithubCopilot))
            .then(|| CopilotAuth::from_default().map(Arc::new))
            .transpose()?;
        let codex_auth = (effective_provider == Some(ProviderArg::OpenAiCodex))
            .then(|| OpenAiCodexAuth::from_default().map(Arc::new))
            .transpose()?;
        let mut config = Self::resolve_from_file(cli, &file, path, env_api_key)?;
        config.copilot_auth = copilot_auth;
        config.codex_auth = codex_auth;
        config.anthropic_auth = (effective_provider == Some(ProviderArg::AnthropicSubscription))
            .then(|| AnthropicSubscriptionAuth::from_default().map(Arc::new))
            .transpose()?;
        // The static Copilot fallback model may not be entitled to this
        // account.  When no model was chosen explicitly, prefer the first
        // catalog model the signed-in account can actually use; the choice
        // is read from the local credential store, so this stays offline.
        if config.provider == ProviderArg::GithubCopilot
            && cli.model.is_none()
            && file.model.is_none()
            && let Some(model) = entitled_copilot_default(config.copilot_auth.as_ref())
        {
            config.model = model;
        }
        Ok(config)
    }

    /// Testable form that also covers the legacy `OPENCODE_API_KEY` fallback.
    #[cfg(test)]
    pub fn resolve_with_key_values(
        cli: &Cli,
        opencode_go_key: Option<&str>,
        opencode_key: Option<&str>,
        openrouter_key: Option<&str>,
    ) -> Result<Self> {
        let path = config_path();
        let file = load_file_config(&path)?;
        Self::resolve_from_file(cli, &file, path, |provider| match provider {
            ProviderArg::OpencodeGo => opencode_go_key.or(opencode_key).map(str::to_owned),
            ProviderArg::Openrouter => openrouter_key.map(str::to_owned),
            ProviderArg::GithubCopilot
            | ProviderArg::OpenAiCodex
            | ProviderArg::AnthropicSubscription => None,
        })
    }

    /// Resolve against a supplied file configuration.  Besides making the
    /// precedence explicit, this gives non-environment callers a convenient
    /// way to test or embed configuration resolution.
    pub fn resolve_from_file<F>(
        cli: &Cli,
        file: &FileConfig,
        path: PathBuf,
        mut key_for: F,
    ) -> Result<Self>
    where
        F: FnMut(ProviderArg) -> Option<String>,
    {
        file.validate()?;
        // Validate the persisted name even when a CLI override is present;
        // otherwise a typo could remain hidden until a later launch.
        let file_provider = match file.provider.as_deref() {
            Some(name) => Some(file.provider_choice(name).ok_or_else(|| {
                anyhow!(
                    "unknown provider `{name}` in config file {}",
                    path.display()
                )
            })?),
            None => None,
        };
        let cli_provider = match cli.provider.as_deref() {
            Some(name) => Some(file.provider_choice(name).ok_or_else(|| {
                anyhow!(
                    "unknown provider `{name}`: use a built-in provider or define `[providers.{name}]` in {}",
                    path.display()
                )
            })?),
            None => None,
        };
        let provider = cli_provider.or(file_provider).ok_or_else(|| {
            anyhow!(
                "no provider configured. Choose one with --provider <name> or set `provider` in {}:\n  opencode-go       OpenCode Go; set OPENCODE_GO_API_KEY or OPENCODE_API_KEY\n  openrouter        OpenRouter; set OPENROUTER_API_KEY\n  github-copilot    GitHub Copilot; run `harness login github-copilot`\n  openai-codex      OpenAI Codex; run `harness login openai-codex`\n  anthropic-claude  Unofficial Claude subscription; run `harness login anthropic-claude`\n  <name>            a `[providers.<name>]` table with `type = \"openai-compatible\"` and `base_url`",
                path.display()
            )
        })?;
        // GitHub Copilot authenticates through the private auth store and is
        // intentionally constructible before the first login.  Existing API
        // key providers retain their old environment-only requirement.
        // User-defined providers read their optional key when constructed.
        if let Some(builtin) = provider.builtin()
            && !matches!(
                builtin,
                ProviderArg::GithubCopilot
                    | ProviderArg::OpenAiCodex
                    | ProviderArg::AnthropicSubscription
            )
            && key_for(builtin).is_none_or(|value| value.trim().is_empty())
        {
            return Err(anyhow!(
                "missing API key: set {}",
                builtin.env_vars().join(" or ")
            ));
        }
        let default_model = match &provider {
            ProviderChoice::Builtin(builtin) => Some(builtin.default_model().to_owned()),
            ProviderChoice::Custom(name) => file.providers[name].default_model.clone(),
        };
        let model = cli
            .model
            .clone()
            .or_else(|| file.model.clone())
            .or(default_model)
            .ok_or_else(|| {
                anyhow!(
                    "no model configured for provider `{provider}`: pass --model, or set `model` or `[providers.{provider}].default_model` in {}",
                    path.display()
                )
            })?;
        let mut secret_entries = file
            .secrets
            .as_ref()
            .map(|secrets| {
                secrets
                    .env
                    .iter()
                    .map(|name| {
                        let value = std::env::var(name).map_err(|error| match error {
                            std::env::VarError::NotPresent => anyhow!(
                                "configured secret environment variable `{name}` is not set"
                            ),
                            std::env::VarError::NotUnicode(_) => anyhow!(
                                "configured secret environment variable `{name}` is not valid Unicode"
                            ),
                        })?;
                        Ok((name.clone(), value))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        // Credentials of user-defined providers are masked like `[secrets]`
        // so tool output (e.g. `env`) cannot show them to the model. Unset or
        // short values are skipped: a missing key fails only when that
        // provider is built, and the masker rejects very short values.
        for config in file.providers.values() {
            let variables = config.api_key_env.iter().chain(config.env_headers.values());
            for variable in variables {
                if secret_entries.iter().any(|(name, _)| name == variable) {
                    continue;
                }
                if let Ok(value) = std::env::var(variable)
                    && value.len() >= agent::secrets::MIN_SECRET_BYTES
                {
                    secret_entries.push((variable.clone(), value));
                }
            }
        }
        let secret_masker =
            Arc::new(SecretMasker::new(secret_entries).map_err(|error| anyhow!(error))?);
        if secret_masker.mask_text(&model) != model
            || secret_masker.mask_text(&provider.to_string()) != provider.to_string()
        {
            return Err(anyhow!(
                "provider and model names cannot contain a configured secret"
            ));
        }
        let prewalk = PrewalkPolicy {
            enabled: file.prewalk.as_ref().is_some_and(|prewalk| prewalk.enabled),
            target: cli
                .prewalk_model
                .as_deref()
                .or_else(|| file.prewalk.as_ref()?.model.as_deref())
                .map(parse_prewalk_target)
                .transpose()?,
            reasoning: cli
                .prewalk_reasoning_effort
                .or_else(|| file.prewalk.as_ref()?.reasoning_effort),
            requested: cli
                .prewalk
                .or_else(|| cli.prewalk_model.as_ref().map(|_| true)),
        };
        if let Some(target) = &prewalk.target
            && secret_masker.mask_text(&target.to_string()) != target.to_string()
        {
            return Err(anyhow!("prewalk model cannot contain a configured secret"));
        }
        let mcp_servers = file
            .mcp
            .as_ref()
            .map(|mcp| mcp.resolve_with(|name| std::env::var(name).ok()))
            .transpose()
            .map_err(|error| anyhow!(error))?;
        Ok(Self {
            provider,
            custom_providers: Arc::new(file.providers.clone()),
            model,
            reasoning: cli
                .reasoning_effort
                .or(file.reasoning_effort)
                .unwrap_or_default(),
            rtk: file.rtk,
            compaction: file
                .compaction
                .as_ref()
                .map(CompactionPolicy::from)
                .unwrap_or_default(),
            subagents: file
                .subagents
                .as_ref()
                .map(resolve_subagents)
                .unwrap_or_default(),
            code_mode: file
                .code_mode
                .as_ref()
                .filter(|config| config.enabled)
                .map(|config| config.tools.clone()),
            secret_masker,
            mcp_servers: mcp_servers.unwrap_or_default(),
            tui_minimal: file.tui.as_ref().is_some_and(|tui| tui.minimal),
            prewalk,
            // Only the process-wide [`Config::resolve`] loads credentials;
            // the testable forms never touch auth.json.
            copilot_auth: None,
            codex_auth: None,
            anthropic_auth: None,
        })
    }
}

fn env_api_key(provider: ProviderArg) -> Option<String> {
    provider.env_vars().iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

/// The best default model for a signed-in Copilot account: for free plans,
/// a model the plan can actually serve; otherwise the first entry of the
/// account's available list that this build knows how to route.  Reading
/// the credential is local-only; `None` keeps the static default.
fn entitled_copilot_default(auth: Option<&Arc<CopilotAuth>>) -> Option<String> {
    let credential = auth?.credential().ok()??;
    let sku = sku_from_proxy_token(&credential.access);
    default_model_for(sku, &credential.available_model_ids)
}

/// Construct a provider with shared OAuth handles. Keeping refresh state in
/// the provider-owned handles avoids reloading credentials on model switches.
/// Built-in names win; `custom_providers` supplies `[providers.<name>]`.
pub fn build_provider_with_auths(
    name: &str,
    custom_providers: &BTreeMap<String, CustomProviderConfig>,
    copilot_auth: Option<Arc<CopilotAuth>>,
    codex_auth: Option<Arc<OpenAiCodexAuth>>,
    anthropic_auth: Option<Arc<AnthropicSubscriptionAuth>>,
) -> Result<Arc<dyn Provider>> {
    let Some(provider) = ProviderArg::from_name(name) else {
        if let Some(config) = custom_providers.get(name) {
            return build_custom_provider(name, config);
        }
        let expected = ProviderArg::ALL
            .iter()
            .map(ToString::to_string)
            .chain(custom_providers.keys().cloned())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow!(
            "unknown provider: {name} (expected one of: {expected})"
        ));
    };
    Ok(match provider {
        ProviderArg::OpencodeGo => {
            let api_key = env_api_key(provider).ok_or_else(|| {
                anyhow!("missing API key: set {}", provider.env_vars().join(" or "))
            })?;
            Arc::new(OpenCodeGoProvider::new(api_key)) as Arc<dyn Provider>
        }
        ProviderArg::Openrouter => {
            let api_key = env_api_key(provider).ok_or_else(|| {
                anyhow!("missing API key: set {}", provider.env_vars().join(" or "))
            })?;
            Arc::new(OpenRouterProvider::new(api_key)) as Arc<dyn Provider>
        }
        ProviderArg::GithubCopilot => {
            let auth = match copilot_auth {
                Some(auth) => auth,
                None => Arc::new(CopilotAuth::from_default()?),
            };
            Arc::new(GithubCopilotProvider::new(auth)) as Arc<dyn Provider>
        }
        ProviderArg::AnthropicSubscription => {
            let auth = match anthropic_auth {
                Some(auth) => auth,
                None => Arc::new(AnthropicSubscriptionAuth::from_default()?),
            };
            Arc::new(AnthropicSubscriptionProvider::new(auth)) as Arc<dyn Provider>
        }
        ProviderArg::OpenAiCodex => {
            let auth = match codex_auth {
                Some(auth) => auth,
                None => Arc::new(OpenAiCodexAuth::from_default()?),
            };
            Arc::new(OpenAiCodexProvider::new(auth)) as Arc<dyn Provider>
        }
    })
}

/// Build the runtime's provider factory while retaining shared OAuth state.
pub fn provider_factory(
    custom_providers: CustomProviders,
    copilot_auth: Option<Arc<CopilotAuth>>,
    codex_auth: Option<Arc<OpenAiCodexAuth>>,
    anthropic_auth: Option<Arc<AnthropicSubscriptionAuth>>,
) -> agent::ProviderFactory {
    Arc::new(move |name| {
        build_provider_with_auths(
            name,
            &custom_providers,
            copilot_auth.clone(),
            codex_auth.clone(),
            anthropic_auth.clone(),
        )
    })
}

/// Install file-only tracing when requested. stdout is intentionally left
/// untouched because the fullscreen terminal owns it.
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
    use tempfile::tempdir;

    #[test]
    fn frontends_parse_as_subcommands() {
        let prompt =
            Cli::try_parse_from(["harness", "prompt", "--provider", "openai-codex", "hello"])
                .unwrap();
        assert_eq!(prompt.provider.as_deref(), Some("openai-codex"));
        assert!(matches!(
            prompt.command,
            Some(Command::Prompt(PromptArgs { prompt, .. })) if prompt == ["hello"]
        ));

        let worktree = Cli::try_parse_from([
            "harness",
            "worktree",
            "feat/new-feature",
            "--dir",
            "../feature",
            "--keep",
        ])
        .unwrap();
        assert!(matches!(
            worktree.command,
            Some(Command::Worktree(WorktreeArgs {
                branch,
                dir: Some(dir),
                keep,
                ..
            })) if branch == "feat/new-feature"
                    && dir == Path::new("../feature")
                    && keep
        ));
        assert!(Cli::try_parse_from(["harness", "worktree"]).is_err());

        let worktree_prompt = Cli::try_parse_from([
            "harness",
            "worktree",
            "feat/headless",
            "--start-point",
            "main~2",
            "prompt",
            "--no-session",
            "summarize",
        ])
        .unwrap();
        assert!(matches!(
            worktree_prompt.command,
            Some(Command::Worktree(WorktreeArgs {
                start_point: Some(start_point),
                command: Some(WorktreeCommand::Prompt(PromptArgs { no_session: true, prompt, .. })),
                ..
            })) if start_point == "main~2" && prompt == ["summarize"]
        ));

        let mcp = Cli::try_parse_from([
            "harness",
            "mcp",
            "add",
            "filesystem",
            "--",
            "npx",
            "-y",
            "server-filesystem",
        ])
        .unwrap();
        assert!(matches!(
            mcp.command,
            Some(Command::Mcp(McpArgs {
                command: Some(McpCommand::Add(McpAddArgs { name, command, .. }))
            })) if name == "filesystem" && command == ["npx", "-y", "server-filesystem"]
        ));
        let http_mcp = Cli::try_parse_from([
            "harness",
            "mcp",
            "add",
            "remote",
            "--url",
            "https://example.test/mcp",
            "--header",
            "Authorization=Bearer ${MCP_TOKEN}",
        ])
        .unwrap();
        assert!(matches!(
            http_mcp.command,
            Some(Command::Mcp(McpArgs {
                command: Some(McpCommand::Add(McpAddArgs { name, url: Some(url), header, command }))
            })) if name == "remote"
                && url == "https://example.test/mcp"
                && header == ["Authorization=Bearer ${MCP_TOKEN}"]
                && command.is_empty()
        ));
        assert!(matches!(
            Cli::try_parse_from(["harness", "mcp"]).unwrap().command,
            Some(Command::Mcp(McpArgs { command: None }))
        ));

        let resumed = Cli::try_parse_from(["harness", "--resume-session", "session-123"]).unwrap();
        assert_eq!(resumed.resume.as_deref(), Some("session-123"));

        let acp = Cli::try_parse_from(["harness", "acp", "--provider", "openai-codex"]).unwrap();
        assert!(matches!(acp.command, Some(Command::Acp)));
        assert!(Cli::try_parse_from(["harness", "--acp"]).is_err());
        assert!(Cli::try_parse_from(["harness", "-p", "hello"]).is_err());

        // A bare `--prewalk` must not consume the following subcommand.
        let prewalk = Cli::try_parse_from(["harness", "--prewalk", "prompt", "hi"]).unwrap();
        assert_eq!(prewalk.prewalk, Some(true));
        assert!(matches!(prewalk.command, Some(Command::Prompt(_))));
        let off = Cli::try_parse_from(["harness", "--prewalk=false"]).unwrap();
        assert_eq!(off.prewalk, Some(false));
    }

    #[test]
    fn login_provider_parses_separately_from_global_provider() {
        for (name, expected) in [
            (
                "anthropic-subscription",
                LoginProvider::AnthropicSubscription,
            ),
            ("anthropic-claude", LoginProvider::AnthropicSubscription),
            ("claude", LoginProvider::AnthropicSubscription),
            ("claude-sub", LoginProvider::AnthropicSubscription),
            ("codex", LoginProvider::OpenAiCodex),
            ("openai-codex", LoginProvider::OpenAiCodex),
            ("copilot", LoginProvider::GithubCopilot),
            ("github-copilot", LoginProvider::GithubCopilot),
        ] {
            let cli = Cli::try_parse_from(["harness", "login", name]).unwrap();
            assert!(matches!(
                cli.command,
                Some(Command::Login(LoginArgs { provider, .. })) if provider == expected
            ));
            assert_eq!(cli.provider, None);
        }

        let cli = Cli::try_parse_from([
            "harness",
            "--provider",
            "openrouter",
            "login",
            "codex",
            "--device-code",
        ])
        .unwrap();
        assert_eq!(cli.provider.as_deref(), Some("openrouter"));
        assert!(matches!(
            cli.command,
            Some(Command::Login(LoginArgs {
                provider: LoginProvider::OpenAiCodex,
                device_code: true,
            }))
        ));
    }

    #[test]
    fn resolves_defaults_and_overrides() {
        // Resolve against an explicit file config so the developer's real
        // ~/.config/harness/config.toml cannot leak into the assertions.
        let error = Config::resolve_from_file(
            &Cli::default(),
            &FileConfig::default(),
            PathBuf::from("/tmp/harness-config.toml"),
            |_| None,
        )
        .unwrap_err()
        .to_string();
        for provider in ProviderArg::ALL {
            assert!(error.contains(&provider.to_string()));
        }

        let cli = Cli {
            provider: Some("openrouter".into()),
            model: Some("anthropic/demo".into()),
            ..Cli::default()
        };
        let config = Config::resolve_from_file(
            &cli,
            &FileConfig::default(),
            PathBuf::from("/tmp/harness-config.toml"),
            |_| Some("router-secret".into()),
        )
        .unwrap();
        assert_eq!(config.provider, ProviderArg::Openrouter);
        assert_eq!(config.model, "anthropic/demo");
        assert_eq!(config.reasoning, ReasoningPolicy::Auto);
    }

    #[test]
    fn subscription_alias_defaults_and_login_selection() {
        for name in [
            "anthropic-claude",
            "anthropic-subscription",
            "claude-sub",
            "claude",
        ] {
            let cli = Cli::try_parse_from(["harness", "--provider", name]).unwrap();
            let config = Config::resolve_from_file(
                &cli,
                &FileConfig::default(),
                PathBuf::from("/tmp/config.toml"),
                |_| None,
            )
            .unwrap();
            assert_eq!(config.provider, ProviderArg::AnthropicSubscription);
            assert_eq!(config.model, "claude-sonnet-4-6");
            assert_eq!(ProviderArg::from_name(name), config.provider.builtin());
        }
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        assert!(select_provider_after_login_at(&path, ProviderArg::AnthropicSubscription).unwrap());
        assert_eq!(
            load_file_config(&path).unwrap().provider.as_deref(),
            Some("anthropic-claude")
        );
        assert!(!select_provider_after_login_at(&path, ProviderArg::OpenAiCodex).unwrap());
    }

    #[tokio::test]
    async fn claude_aliases_return_catalog_with_canonical_provider_name() {
        let directory = tempdir().unwrap();
        let store = auth::AuthStore::new(directory.path().join("auth.json"));
        store
            .save_anthropic_subscription(&auth::AnthropicSubscriptionCredential::new(
                "saved-access",
                "saved-refresh",
                u64::MAX,
                "account",
            ))
            .unwrap();
        let auth = Arc::new(AnthropicSubscriptionAuth::new(store).unwrap());
        let factory = provider_factory(Default::default(), None, None, Some(auth));
        for name in [
            "anthropic-claude",
            "anthropic-subscription",
            "claude-sub",
            "claude",
        ] {
            let provider = factory(name).unwrap();
            assert_eq!(
                provider.name(),
                ProviderArg::AnthropicSubscription.to_string()
            );
            let models = provider.list_models().await.unwrap();
            assert!(models.iter().any(|model| model.id == "claude-sonnet-4-6"));
        }
    }

    #[test]
    fn copilot_resolves_without_an_api_key() {
        let cli = Cli {
            provider: Some("github-copilot".into()),
            model: None,
            ..Cli::default()
        };
        let config = Config::resolve_from_file(
            &cli,
            &FileConfig::default(),
            PathBuf::from("/tmp/harness-config.toml"),
            |_| None,
        )
        .unwrap();
        assert_eq!(config.provider, ProviderArg::GithubCopilot);
        assert_eq!(config.model, "gpt-5.4");
    }

    #[test]
    fn supports_file_precedence_and_unknown_provider_errors() {
        let file = FileConfig {
            provider: Some("openrouter".into()),
            model: Some("file-model".into()),
            reasoning_effort: Some(ReasoningPolicy::Effort(llm::ReasoningEffort::Low)),
            ..FileConfig::default()
        };
        let cli = Cli::default();
        let config = Config::resolve_from_file(
            &cli,
            &file,
            PathBuf::from("/tmp/harness-config.toml"),
            |provider| match provider {
                ProviderArg::Openrouter => Some("secret".into()),
                ProviderArg::OpencodeGo
                | ProviderArg::GithubCopilot
                | ProviderArg::OpenAiCodex
                | ProviderArg::AnthropicSubscription => None,
            },
        )
        .unwrap();
        assert_eq!(config.provider, ProviderArg::Openrouter);
        assert_eq!(config.model, "file-model");
        assert_eq!(
            config.reasoning,
            ReasoningPolicy::Effort(llm::ReasoningEffort::Low)
        );

        let cli = Cli {
            provider: Some("opencode-go".into()),
            model: Some("cli-model".into()),
            reasoning_effort: Some(ReasoningPolicy::Off),
            ..Cli::default()
        };
        let config = Config::resolve_from_file(
            &cli,
            &file,
            PathBuf::from("/tmp/harness-config.toml"),
            |_| Some("secret".into()),
        )
        .unwrap();
        assert_eq!(config.provider, ProviderArg::OpencodeGo);
        assert_eq!(config.model, "cli-model");
        assert_eq!(config.reasoning, ReasoningPolicy::Off);

        let invalid = FileConfig {
            provider: Some("typo".into()),
            ..FileConfig::default()
        };
        let error = Config::resolve_from_file(
            &Cli::default(),
            &invalid,
            PathBuf::from("/tmp/bad-config.toml"),
            |_| Some("secret".into()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("/tmp/bad-config.toml"));
    }

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

    #[test]
    fn login_selects_only_when_provider_is_unset() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");

        assert!(select_provider_after_login_at(&path, ProviderArg::OpenAiCodex).unwrap());
        assert_eq!(
            load_file_config(&path).unwrap().provider.as_deref(),
            Some("openai-codex")
        );
        assert!(!select_provider_after_login_at(&path, ProviderArg::GithubCopilot).unwrap());
        assert_eq!(
            load_file_config(&path).unwrap().provider.as_deref(),
            Some("openai-codex")
        );
    }

    #[test]
    fn supports_legacy_opencode_key_name() {
        let cli = Cli {
            provider: Some("opencode-go".into()),
            ..Cli::default()
        };
        let config =
            Config::resolve_with_key_values(&cli, None, Some("legacy-secret"), None).unwrap();
        assert_eq!(config.provider, ProviderArg::OpencodeGo);
    }

    #[test]
    fn missing_key_names_environment_variables() {
        let cli = Cli {
            provider: Some("openrouter".into()),
            ..Cli::default()
        };
        let error = Config::resolve_with_key_values(&cli, None, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("OPENROUTER_API_KEY"));

        let cli = Cli {
            provider: Some("opencode-go".into()),
            ..Cli::default()
        };
        let error = Config::resolve_with_key_values(&cli, None, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("OPENCODE_GO_API_KEY"));
        assert!(error.to_string().contains("OPENCODE_API_KEY"));
    }

    #[test]
    fn file_round_trip_and_unknown_fields_survive_settings_save() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("nested").join("config.toml");
        let original = FileConfig {
            provider: Some("opencode-go".into()),
            model: Some("demo".into()),
            reasoning_effort: Some(ReasoningPolicy::Effort(llm::ReasoningEffort::High)),
            rtk: true,
            compaction: Some(CompactConfig {
                keep_recent_turns: Some(5),
                ..CompactConfig::default()
            }),
            subagents: Some(crate::config::SubagentConfig {
                max_turns: Some(10),
                max_concurrent: Some(2),
            }),
            code_mode: None,
            secrets: None,
            mcp: None,
            tui: Some(TuiConfig { minimal: true }),
            prewalk: None,
            providers: [(
                "lmstudio".to_owned(),
                CustomProviderConfig {
                    kind: CustomProviderKind::OpenAiCompatible,
                    base_url: "http://localhost:1234/v1".into(),
                    api_key_env: Some("LMSTUDIO_API_KEY".into()),
                    default_model: None,
                    context_window: Some(32_768),
                    headers: [("X-Client".to_owned(), "harness".to_owned())].into(),
                    env_headers: [("X-Token".to_owned(), "LMSTUDIO_TOKEN".to_owned())].into(),
                },
            )]
            .into(),
            extra: [("future".to_owned(), toml::Value::String("kept".into()))]
                .into_iter()
                .collect(),
        };
        save_file_config(&path, &original).unwrap();
        assert_eq!(load_file_config(&path).unwrap(), original);
        let loaded = load_file_config(&path).unwrap();
        assert_eq!(
            loaded.subagents.as_ref().and_then(|s| s.max_turns),
            Some(10)
        );

        save_settings_at(&path, "openrouter", "router/demo").unwrap();
        let saved = load_file_config(&path).unwrap();
        assert_eq!(saved.provider.as_deref(), Some("openrouter"));
        assert_eq!(saved.model.as_deref(), Some("router/demo"));
        assert_eq!(saved.reasoning_effort, original.reasoning_effort);
        assert!(saved.rtk);
        assert!(saved.tui.is_some_and(|tui| tui.minimal));
        assert_eq!(
            saved.extra.get("future"),
            Some(&toml::Value::String("kept".into()))
        );

        save_reasoning_at(&path, ReasoningPolicy::Off).unwrap();
        let saved = load_file_config(&path).unwrap();
        assert_eq!(saved.reasoning_effort, Some(ReasoningPolicy::Off));
        assert_eq!(saved.provider.as_deref(), Some("openrouter"));
        assert_eq!(
            saved.extra.get("future"),
            Some(&toml::Value::String("kept".into()))
        );

        // Enabling without a target would make the config invalid.
        assert!(save_prewalk_enabled_at(&path, true, None).is_err());
        save_prewalk_model_at(&path, "openrouter:vendor/model:free").unwrap();
        // A process-only target never replaces the saved one.
        save_prewalk_enabled_at(&path, true, Some("openrouter:other")).unwrap();
        save_prewalk_enabled_at(&path, false, None).unwrap();
        save_prewalk_reasoning_at(&path, Some(ReasoningPolicy::Off)).unwrap();
        let saved = load_file_config(&path).unwrap();
        assert_eq!(
            saved.prewalk,
            Some(PrewalkConfig {
                enabled: false,
                model: Some("openrouter:vendor/model:free".into()),
                reasoning_effort: Some(ReasoningPolicy::Off),
            })
        );
        save_prewalk_reasoning_at(&path, None).unwrap();
        let saved = load_file_config(&path).unwrap();
        assert_eq!(
            saved.prewalk.and_then(|prewalk| prewalk.reasoning_effort),
            None
        );
        assert_eq!(saved.model.as_deref(), Some("router/demo"));
    }

    #[test]
    fn prewalk_target_prefix_is_a_provider_only_when_known() {
        let target = parse_prewalk_target("openrouter:vendor/model:free").unwrap();
        assert_eq!(target.provider.as_deref(), Some("openrouter"));
        assert_eq!(target.model, "vendor/model:free");
        let target = parse_prewalk_target("vendor/model:free").unwrap();
        assert_eq!(target.provider, None);
        assert_eq!(target.model, "vendor/model:free");
        assert!(parse_prewalk_target("openrouter:").is_err());
        assert!(parse_prewalk_target("  ").is_err());
    }

    #[test]
    fn code_mode_requires_explicit_opt_in_and_preserves_tool_selection() {
        let path = PathBuf::from("config.toml");
        let absent: FileConfig = toml::from_str("").unwrap();
        assert_eq!(absent.code_mode, None);
        assert!(!toml::to_string(&absent).unwrap().contains("code_mode"));

        for (table, expected) in [
            ("", None),
            ("[code_mode]", None),
            ("[code_mode]\ntools = ['read']", None),
            ("[code_mode]\nenabled = false\ntools = ['read']", None),
            ("[code_mode]\nenabled = true", Some(Vec::new())),
            (
                "[code_mode]\nenabled = true\ntools = ['read', 'grep']",
                Some(vec!["read".to_owned(), "grep".to_owned()]),
            ),
        ] {
            let file =
                parse_file_config(&format!("provider = 'opencode-go'\n{table}"), &path).unwrap();
            if table == "[code_mode]" {
                assert_eq!(file.code_mode, Some(CodeModeConfig::default()));
            }
            let resolved = Config::resolve_from_file(&Cli::default(), &file, path.clone(), |_| {
                Some("test-key".into())
            })
            .unwrap();
            assert_eq!(resolved.code_mode, expected, "{table}");
            let saved = toml::to_string(&file).unwrap();
            assert_eq!(toml::from_str::<FileConfig>(&saved).unwrap(), file);
        }
    }

    #[test]
    fn rtk_defaults_off_and_requires_explicit_opt_in() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        assert!(!load_file_config(&path).unwrap().rtk);

        fs::write(&path, "rtk = true\n").unwrap();
        assert!(load_file_config(&path).unwrap().rtk);
    }

    #[test]
    fn targeted_settings_saves_preserve_unknown_nested_document_fields() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        fs::write(
            &path,
            r#"provider = "opencode-go"
model = "old"

[compaction]
threshold = 0.8
future_switch = "keep"

[compaction.future]
list = ["one", "two"]

[subagents]
max_turns = 4
future_limit = 17

[tui]
minimal = true
future_theme = "keep"

[mcp]
future_setting = true

[[mcp.servers]]
name = "demo"
transport = "stdio"
command = "/bin/echo"
future_server_key = "keep"
"#,
        )
        .unwrap();

        save_settings_at(&path, "openrouter", "new/model").unwrap();
        save_reasoning_at(&path, ReasoningPolicy::Effort(llm::ReasoningEffort::High)).unwrap();
        let saved = fs::read_to_string(&path).unwrap();
        for fragment in [
            "future_switch = \"keep\"",
            "list = [\"one\", \"two\"]",
            "future_limit = 17",
            "future_theme = \"keep\"",
            "future_setting = true",
            "future_server_key = \"keep\"",
        ] {
            assert!(saved.contains(fragment), "missing {fragment} in {saved}");
        }
        let config = load_file_config(&path).unwrap();
        assert_eq!(config.provider.as_deref(), Some("openrouter"));
        assert_eq!(config.model.as_deref(), Some("new/model"));
        assert_eq!(
            config.reasoning_effort,
            Some(ReasoningPolicy::Effort(llm::ReasoningEffort::High))
        );
    }

    #[test]
    fn mcp_mutators_preserve_unknown_fields_and_reject_missing_names() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        fs::write(
            &path,
            "future = true\n\n[mcp]\nfuture_setting = 7\n\n[[mcp.servers]]\nname = \"existing\"\ntransport = \"stdio\"\ncommand = \"old\"\nfuture_server_key = \"keep\"\n",
        )
        .unwrap();

        add_mcp_server_at(
            &path,
            "filesystem",
            &["npx".into(), "-y".into(), "server-filesystem".into()],
        )
        .unwrap();
        assert!(add_mcp_server_at(&path, "filesystem", &["other".into()]).is_err());
        let loaded = load_file_config(&path).unwrap();
        let servers = &loaded.mcp.unwrap().servers;
        assert_eq!(servers.len(), 2);
        let added = servers
            .iter()
            .find(|server| server.name == "filesystem")
            .unwrap();
        let mcp::McpTransportConfig::Stdio { command, args, .. } = &added.transport else {
            panic!("expected stdio server")
        };
        assert_eq!(command, Path::new("npx"));
        assert_eq!(args, &["-y", "server-filesystem"]);

        delete_mcp_server_at(&path, "filesystem").unwrap();
        assert!(delete_mcp_server_at(&path, "missing").is_err());
        let saved = fs::read_to_string(&path).unwrap();
        for fragment in [
            "future = true",
            "future_setting = 7",
            "future_server_key = \"keep\"",
        ] {
            assert!(saved.contains(fragment), "missing {fragment} in {saved}");
        }
        assert!(!saved.contains("filesystem"));
    }

    #[test]
    fn adds_http_mcp_server_with_environment_backed_headers() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        let headers = BTreeMap::from([
            ("Authorization".into(), "Bearer ${MCP_TOKEN}".into()),
            ("X-Tenant".into(), "acme".into()),
        ]);

        add_http_mcp_server_at(&path, "remote", "https://example.test/mcp", &headers).unwrap();
        let loaded = load_file_config(&path).unwrap();
        let server = &loaded.mcp.unwrap().servers[0];
        let mcp::McpTransportConfig::Http { url, headers } = &server.transport else {
            panic!("expected HTTP server")
        };
        assert_eq!(url, "https://example.test/mcp");
        assert_eq!(headers["Authorization"], "Bearer ${MCP_TOKEN}");
        assert_eq!(headers["X-Tenant"], "acme");
    }

    #[test]
    fn concurrent_config_mutators_do_not_lose_each_other() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        save_file_config(&path, &FileConfig::default()).unwrap();
        let settings_path = path.clone();
        let reasoning_path = path.clone();
        let settings = std::thread::spawn(move || {
            save_settings_at(&settings_path, "openrouter", "router/model")
        });
        let reasoning =
            std::thread::spawn(move || save_reasoning_at(&reasoning_path, ReasoningPolicy::Off));
        settings.join().unwrap().unwrap();
        reasoning.join().unwrap().unwrap();
        let saved = load_file_config(&path).unwrap();
        assert_eq!(saved.provider.as_deref(), Some("openrouter"));
        assert_eq!(saved.model.as_deref(), Some("router/model"));
        assert_eq!(saved.reasoning_effort, Some(ReasoningPolicy::Off));
    }

    #[test]
    fn invalid_compaction_values_name_their_toml_keys() {
        let cases = [
            (
                CompactConfig {
                    threshold: Some(-0.1),
                    ..Default::default()
                },
                "threshold",
            ),
            (
                CompactConfig {
                    threshold: Some(f64::NAN),
                    ..Default::default()
                },
                "threshold",
            ),
            (
                CompactConfig {
                    threshold: Some(1.1),
                    ..Default::default()
                },
                "threshold",
            ),
            (
                CompactConfig {
                    max_summary_input_bytes: Some(0),
                    ..Default::default()
                },
                "max_summary_input_bytes",
            ),
            (
                CompactConfig {
                    max_summary_bytes: Some(0),
                    ..Default::default()
                },
                "max_summary_bytes",
            ),
            (
                CompactConfig {
                    reserve_tokens: Some(10),
                    context_window: Some(10),
                    ..Default::default()
                },
                "reserve_tokens",
            ),
        ];
        for (compaction, key) in cases {
            let error = FileConfig {
                compaction: Some(compaction),
                ..Default::default()
            }
            .validate()
            .unwrap_err()
            .to_string();
            assert!(error.contains(key), "{error}");
        }
    }

    #[test]
    fn default_reserve_is_checked_against_explicit_window() {
        let error = FileConfig {
            compaction: Some(CompactConfig {
                context_window: Some(8_000),
                ..Default::default()
            }),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("reserve_tokens"), "{error}");
    }

    #[test]
    fn zero_keep_values_are_valid_and_have_documented_policy_meaning() {
        let config = CompactConfig {
            keep_recent_turns: Some(0),
            keep_recent_tokens: Some(0),
            ..Default::default()
        };
        config.validate().unwrap();
    }

    #[test]
    fn compaction_config_round_trips_and_overrides_defaults() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        let original = FileConfig {
            compaction: Some(CompactConfig {
                auto: Some(false),
                threshold: Some(0.9),
                reserve_tokens: Some(8_000),
                context_window: Some(12_000),
                ..CompactConfig::default()
            }),
            ..FileConfig::default()
        };
        save_file_config(&path, &original).unwrap();
        assert_eq!(load_file_config(&path).unwrap(), original);

        let policy = CompactionPolicy::from(original.compaction.as_ref().unwrap());
        assert!(!policy.auto);
        assert_eq!(policy.threshold, 0.9);
        assert_eq!(policy.reserve_tokens, 8_000);
        assert_eq!(policy.context_window, 12_000);
        // Unset fields fall back to defaults.
        assert_eq!(
            policy.keep_recent_turns,
            CompactionPolicy::default().keep_recent_turns
        );

        // The unknown-field survival path must not see the compaction table.
        let loaded = load_file_config(&path).unwrap();
        assert!(!loaded.extra.contains_key("compaction"));
    }

    #[test]
    fn secrets_config_round_trips_and_rejects_unsafe_names() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        let original = FileConfig {
            secrets: Some(SecretsConfig {
                env: vec!["DEPLOY_TOKEN".into(), "_PRIVATE_KEY".into()],
            }),
            ..FileConfig::default()
        };
        save_file_config(&path, &original).unwrap();
        assert_eq!(load_file_config(&path).unwrap(), original);

        for names in [
            vec![""],
            vec!["BAD-NAME"],
            vec!["1TOKEN"],
            vec!["TOKEN", "TOKEN"],
        ] {
            let error = FileConfig {
                secrets: Some(SecretsConfig {
                    env: names.into_iter().map(str::to_owned).collect(),
                }),
                ..FileConfig::default()
            }
            .validate()
            .unwrap_err()
            .to_string();
            assert!(error.contains("[secrets].env"), "{error}");
        }
    }

    #[test]
    fn mcp_call_overrides_replace_persisted_servers_before_validation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config.toml");
        let contents =
            "provider = 'opencode-go'\n[[mcp.servers]]\nname = 'stale'\ntransport = 'obsolete'\n";
        fs::write(&path, contents).unwrap();
        assert!(load_file_config(&path).is_err());

        let disabled = load_file_config_with_mcp(&path, Some(&[])).unwrap();
        let resolved = Config::resolve_from_file(&Cli::default(), &disabled, path.clone(), |_| {
            Some("unused-test-key".into())
        })
        .unwrap();
        assert!(resolved.mcp_servers.is_empty());

        let servers = vec![mcp::McpServerConfig {
            name: "replacement".into(),
            transport: mcp::McpTransportConfig::Http {
                url: "https://example.test/mcp".into(),
                headers: BTreeMap::new(),
            },
        }];
        let replaced = load_file_config_with_mcp(&path, Some(&servers)).unwrap();
        assert_eq!(replaced.mcp.unwrap().servers, servers);
        assert_eq!(fs::read_to_string(&path).unwrap(), contents);

        let missing = directory.path().join("missing.toml");
        let standalone = load_file_config_with_mcp(&missing, Some(&servers)).unwrap();
        assert_eq!(standalone.mcp.unwrap().servers, servers);
        assert!(!missing.exists());
    }

    #[test]
    fn missing_file_resolves_to_defaults() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config");
        assert_eq!(load_file_config(&path).unwrap(), FileConfig::default());
    }

    #[test]
    fn temporary_path_names_are_unique_and_embody_a_random_component() {
        let path = Path::new("/tmp/harness-config.toml");
        let first = temporary_path(path);
        let second = temporary_path(path);
        let first_name = first.file_name().unwrap().to_string_lossy().into_owned();
        let second_name = second.file_name().unwrap().to_string_lossy().into_owned();
        assert!(first_name.starts_with(".harness-config.toml.tmp-"));
        assert_ne!(first_name, second_name);
        // The tail is 8 lowercase hex digits.
        let tail = first_name.rsplit('-').next().unwrap();
        assert_eq!(tail.len(), 8);
        assert!(tail.chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_eq!(first.parent(), second.parent());
    }
}
