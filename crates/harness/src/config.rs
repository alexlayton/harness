use agent::assembly::SubagentPolicy;
use anyhow::{Context, Result, anyhow};
use auth::OpenAiCodexAuth;
use auth::{CopilotAuth, sku_from_proxy_token};
use clap::{Parser, ValueEnum};
use compact::policy::CompactionPolicy;
use llm::Provider;
use llm::providers::github_copilot::default_model_for;
use llm::providers::{
    GithubCopilotProvider, OpenAiCodexProvider, OpenCodeGoProvider, OpenRouterProvider,
};
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
}

impl ProviderArg {
    /// All provider names understood by the command line, configuration file,
    /// and command completion UI.
    pub const ALL: &[ProviderArg] = &[
        Self::OpencodeGo,
        Self::Openrouter,
        Self::GithubCopilot,
        Self::OpenAiCodex,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "opencode-go" => Some(Self::OpencodeGo),
            "openrouter" => Some(Self::Openrouter),
            "github-copilot" | "githubcopilot" | "copilot" => Some(Self::GithubCopilot),
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
            Self::OpenAiCodex => &[],
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Self::OpencodeGo => "gpt-5.6-luna",
            Self::Openrouter => "openai/gpt-5.6-luna",
            Self::GithubCopilot => "gpt-5.4",
            Self::OpenAiCodex => "gpt-5.5",
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
        })
    }
}

/// Per-key compaction settings parsed from `[compaction]` in `config.toml`.
/// Every field is optional so a partial table overrides only what it sets;
/// the rest fall back to [`CompactionPolicy::default`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CompactConfig {
    pub auto: Option<bool>,
    pub threshold: Option<f64>,
    pub reserve_tokens: Option<u64>,
    pub keep_recent_turns: Option<usize>,
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

/// Terminal UI presentation settings parsed from `[tui]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TuiConfig {
    /// Start directly at the input field without printing the welcome header.
    #[serde(default)]
    pub minimal: bool,
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
    /// Opt-in RTK integration: rewrite supported bash commands to their
    /// token-optimized `rtk` equivalents before execution.  Off unless the
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
    /// External MCP servers configured under `[[mcp.servers]]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<mcp::McpConfig>,
    /// Terminal UI presentation settings (`[tui]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tui: Option<TuiConfig>,
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
            && self.rtk == other.rtk
            && self.compaction == other.compaction
            && self.subagents == other.subagents
            && self.mcp == other.mcp
            && self.tui == other.tui
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
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(FileConfig::default());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read config file {}", path.display()));
        }
    };
    toml::from_str(&contents).with_context(|| format!("parse config file {}", path.display()))
}

/// Save a TOML configuration using a temporary file in the target directory,
/// followed by rename, so a partially-written config is never observed.
pub fn save_file_config(path: &Path, config: &FileConfig) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create config directory {}", parent.display()))?;

    let contents = toml::to_string_pretty(config).context("serialize config")?;
    let temp_path = temporary_path(path);
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .with_context(|| format!("create temporary config {}", temp_path.display()))?;
        file.write_all(contents.as_bytes())
            .with_context(|| format!("write temporary config {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("flush temporary config {}", temp_path.display()))?;
        drop(file);
        fs::rename(&temp_path, path)
            .with_context(|| format!("replace config file {}", path.display()))?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
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
    let mut config = load_file_config(path)?;
    config.provider = Some(provider.to_owned());
    config.model = Some(model.to_owned());
    save_file_config(path, &config)
}

/// Select an OAuth provider after login when the user has not already made a
/// choice. Login must never unexpectedly replace an existing provider.
pub fn select_provider_after_login(provider: ProviderArg) -> Result<bool> {
    select_provider_after_login_at(&config_path(), provider)
}

fn select_provider_after_login_at(path: &Path, provider: ProviderArg) -> Result<bool> {
    let mut config = load_file_config(path)?;
    if config.provider.is_some() {
        return Ok(false);
    }
    config.provider = Some(provider.to_string());
    save_file_config(path, &config)?;
    Ok(true)
}

#[derive(Clone, Debug, Default, Parser)]
#[command(name = "harness", version, about = "A minimal coding-agent harness")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Override the configured provider for this process.
    #[arg(long, value_enum, global = true)]
    pub provider: Option<ProviderArg>,

    /// Override the configured model for this process.
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Disable AGENTS.md / CLAUDE.md project-context injection.
    #[arg(long = "no-context-files", default_value_t = false, global = true)]
    pub no_context_files: bool,

    /// Defer session fsyncs to turn boundaries instead of syncing every
    /// persisted event. Faster chatty tool loops at the cost of losing the
    /// current turn's tail (not just the in-flight record) on power loss.
    #[arg(long = "defer-session-sync", default_value_t = false, global = true)]
    pub defer_session_sync: bool,
}

/// Frontend and authentication commands. With no command Harness starts the
/// interactive terminal UI.
#[derive(Clone, Debug, clap::Subcommand)]
pub enum Command {
    /// Authenticate with an OAuth provider.
    Login(LoginArgs),
    /// Run one prompt and print only the final answer to stdout.
    Prompt(PromptArgs),
    /// Serve Agent Client Protocol over stdio for editor integrations.
    Acp,
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
    #[arg(value_enum)]
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
}

#[derive(Clone, Debug)]
pub struct Config {
    pub provider: ProviderArg,
    pub model: String,
    /// Resolved key retained for config diagnostics and boundary tests;
    /// provider construction reads the environment directly.
    #[cfg_attr(not(test), allow(dead_code))]
    pub api_key: String,
    pub rtk: bool,
    pub compaction: CompactionPolicy,
    /// Resolved subagent bounds (file `[subagents]` over defaults).
    pub subagents: SubagentPolicy,
    /// MCP servers after deterministic validation and `${ENV_VAR}` expansion.
    pub mcp_servers: Vec<mcp::McpServerConfig>,
    /// Whether the interactive frontend skips its welcome header.
    pub tui_minimal: bool,
    /// The Copilot credential handle loaded once during resolution. Startup
    /// used to construct `CopilotAuth::from_default()` twice — once for the
    /// entitled-model default inside `resolve`, again in `main` — reading and
    /// decrypting auth.json both times. `None` for non-Copilot providers.
    pub copilot_auth: Option<Arc<CopilotAuth>>,
    pub codex_auth: Option<Arc<OpenAiCodexAuth>>,
}

impl Config {
    pub fn resolve(cli: &Cli) -> Result<Self> {
        let path = config_path();
        let file = load_file_config(&path)?;
        // One credential read serves both the entitled-model default below
        // and the provider/agent construction in `main`. Loaded only when the
        // *effective* provider (CLI override wins) is Copilot, matching the
        // historical behavior: a broken auth.json must not fail startup for
        // an unrelated `--provider` override.
        let effective_provider = cli
            .provider
            .or_else(|| file.provider.as_deref().and_then(ProviderArg::from_name));
        let copilot_auth = (effective_provider == Some(ProviderArg::GithubCopilot))
            .then(|| CopilotAuth::from_default().map(Arc::new))
            .transpose()?;
        let codex_auth = (effective_provider == Some(ProviderArg::OpenAiCodex))
            .then(|| OpenAiCodexAuth::from_default().map(Arc::new))
            .transpose()?;
        let mut config = Self::resolve_from_file(cli, &file, path, env_api_key)?;
        config.copilot_auth = copilot_auth;
        config.codex_auth = codex_auth;
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
            ProviderArg::GithubCopilot | ProviderArg::OpenAiCodex => None,
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
        // Validate the persisted name even when a CLI override is present;
        // otherwise a typo could remain hidden until a later launch.
        let file_provider = match file.provider.as_deref() {
            Some(name) => Some(ProviderArg::from_name(name).ok_or_else(|| {
                anyhow!(
                    "unknown provider `{name}` in config file {}",
                    path.display()
                )
            })?),
            None => None,
        };
        let provider = cli.provider.or(file_provider).ok_or_else(|| {
            anyhow!(
                "no provider configured. Choose one with --provider <name> or set `provider` in {}:\n  opencode-go      OpenCode Go; set OPENCODE_GO_API_KEY or OPENCODE_API_KEY\n  openrouter       OpenRouter; set OPENROUTER_API_KEY\n  github-copilot   GitHub Copilot; run `harness login github-copilot`\n  openai-codex     OpenAI Codex; run `harness login openai-codex`",
                path.display()
            )
        })?;
        // GitHub Copilot authenticates through the private auth store and is
        // intentionally constructible before the first login.  Existing API
        // key providers retain their old environment-only requirement.
        let api_key = key_for(provider)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_default();
        if !matches!(
            provider,
            ProviderArg::GithubCopilot | ProviderArg::OpenAiCodex
        ) && api_key.is_empty()
        {
            return Err(anyhow!(
                "missing API key: set {}",
                provider.env_vars().join(" or ")
            ));
        }
        let model = cli
            .model
            .clone()
            .or_else(|| file.model.clone())
            .unwrap_or_else(|| provider.default_model().to_owned());
        let mcp_servers = file
            .mcp
            .as_ref()
            .map(|mcp| mcp.resolve_with(|name| std::env::var(name).ok()))
            .transpose()
            .map_err(|error| anyhow!(error))?;
        Ok(Self {
            provider,
            model,
            api_key,
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
            mcp_servers: mcp_servers.unwrap_or_default(),
            tui_minimal: file.tui.as_ref().is_some_and(|tui| tui.minimal),
            // Only the process-wide [`Config::resolve`] loads credentials;
            // the testable forms never touch auth.json.
            copilot_auth: None,
            codex_auth: None,
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
pub fn build_provider_with_auths(
    name: &str,
    copilot_auth: Option<Arc<CopilotAuth>>,
    codex_auth: Option<Arc<OpenAiCodexAuth>>,
) -> Result<Arc<dyn Provider>> {
    let provider = ProviderArg::from_name(name).ok_or_else(|| {
        let expected = ProviderArg::ALL
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        anyhow!("unknown provider: {name} (expected one of: {expected})")
    })?;
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
    copilot_auth: Option<Arc<CopilotAuth>>,
    codex_auth: Option<Arc<OpenAiCodexAuth>>,
) -> agent::ProviderFactory {
    Arc::new(move |name| build_provider_with_auths(name, copilot_auth.clone(), codex_auth.clone()))
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
        assert_eq!(prompt.provider, Some(ProviderArg::OpenAiCodex));
        assert!(matches!(
            prompt.command,
            Some(Command::Prompt(PromptArgs { prompt, .. })) if prompt == ["hello"]
        ));

        let acp = Cli::try_parse_from(["harness", "acp", "--provider", "openai-codex"]).unwrap();
        assert!(matches!(acp.command, Some(Command::Acp)));
        assert!(Cli::try_parse_from(["harness", "--acp"]).is_err());
        assert!(Cli::try_parse_from(["harness", "-p", "hello"]).is_err());
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
            provider: Some(ProviderArg::Openrouter),
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
    }

    #[test]
    fn copilot_resolves_without_an_api_key() {
        let cli = Cli {
            provider: Some(ProviderArg::GithubCopilot),
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
        assert!(config.api_key.is_empty());
    }

    #[test]
    fn supports_file_precedence_and_unknown_provider_errors() {
        let file = FileConfig {
            provider: Some("openrouter".into()),
            model: Some("file-model".into()),
            ..FileConfig::default()
        };
        let cli = Cli::default();
        let config = Config::resolve_from_file(
            &cli,
            &file,
            PathBuf::from("/tmp/harness-config.toml"),
            |provider| match provider {
                ProviderArg::Openrouter => Some("secret".into()),
                ProviderArg::OpencodeGo | ProviderArg::GithubCopilot | ProviderArg::OpenAiCodex => {
                    None
                }
            },
        )
        .unwrap();
        assert_eq!(config.provider, ProviderArg::Openrouter);
        assert_eq!(config.model, "file-model");

        let cli = Cli {
            provider: Some(ProviderArg::OpencodeGo),
            model: Some("cli-model".into()),
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
            provider: Some(ProviderArg::OpencodeGo),
            ..Cli::default()
        };
        let config =
            Config::resolve_with_key_values(&cli, None, Some("legacy-secret"), None).unwrap();
        assert_eq!(config.api_key, "legacy-secret");
    }

    #[test]
    fn missing_key_names_environment_variables() {
        let cli = Cli {
            provider: Some(ProviderArg::Openrouter),
            ..Cli::default()
        };
        let error = Config::resolve_with_key_values(&cli, None, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("OPENROUTER_API_KEY"));

        let cli = Cli {
            provider: Some(ProviderArg::OpencodeGo),
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
            rtk: true,
            compaction: Some(CompactConfig {
                keep_recent_turns: Some(5),
                ..CompactConfig::default()
            }),
            subagents: Some(crate::config::SubagentConfig {
                max_turns: Some(10),
                max_concurrent: Some(2),
            }),
            mcp: None,
            tui: Some(TuiConfig { minimal: true }),
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
        assert!(saved.rtk);
        assert!(saved.tui.is_some_and(|tui| tui.minimal));
        assert_eq!(
            saved.extra.get("future"),
            Some(&toml::Value::String("kept".into()))
        );
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
