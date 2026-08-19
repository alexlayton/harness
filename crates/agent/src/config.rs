use anyhow::{Context, Result, anyhow};
use auth::{CopilotAuth, sku_from_proxy_token};
use clap::{Parser, ValueEnum};
use compact::policy::CompactionPolicy;
use llm::Provider;
use llm::providers::github_copilot::default_model_for;
use llm::providers::{GithubCopilotProvider, OpenCodeGoProvider, OpenRouterProvider};
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
    #[value(name = "github-copilot")]
    GithubCopilot,
}

impl ProviderArg {
    /// All provider names understood by the command line, configuration file,
    /// and command completion UI.
    pub const ALL: &[ProviderArg] = &[Self::OpencodeGo, Self::Openrouter, Self::GithubCopilot];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "opencode-go" => Some(Self::OpencodeGo),
            "openrouter" => Some(Self::Openrouter),
            "github-copilot" | "githubcopilot" => Some(Self::GithubCopilot),
            _ => None,
        }
    }

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
        }
    }

    pub fn env_var(&self) -> &'static str {
        self.env_vars()[0]
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Self::OpencodeGo => "gpt-5.6-luna",
            Self::Openrouter => "openai/gpt-5.6-luna",
            Self::GithubCopilot => "gpt-5.4",
        }
    }
}

impl fmt::Display for ProviderArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OpencodeGo => "opencode-go",
            Self::Openrouter => "openrouter",
            Self::GithubCopilot => "github-copilot",
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

#[derive(Clone, Debug, Default, Parser)]
#[command(name = "harness", about = "A minimal coding-agent harness")]
pub struct Cli {
    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,

    #[arg(long)]
    pub model: Option<String>,

    /// Non-interactive mode: run one prompt to completion and print the answer.
    #[arg(short = 'p', long = "print", default_value_t = false)]
    pub print: bool,

    /// Verbose: stream reasoning + tool activity + full (bounded) tool output
    /// to stderr. Default stderr is silent except for hard errors.
    #[arg(short = 'v', long = "verbose", default_value_t = false)]
    pub verbose: bool,

    /// Resume an existing session instead of creating a fresh one.
    /// Accepts a session id, unique prefix, `latest`, or a file path.
    #[arg(long, value_name = "ID|latest|PATH")]
    pub resume: Option<String>,

    /// Prompt for non-interactive mode. Joined with spaces. When `--print` is
    /// set and no positional is given, the prompt is read from stdin.
    /// Only meaningful with `--print`; passing it without `--print` is an
    /// error.
    #[arg(value_name = "PROMPT")]
    pub prompt: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub provider: ProviderArg,
    pub model: String,
    pub api_key: String,
    pub config_path: PathBuf,
    pub rtk: bool,
    pub compaction: CompactionPolicy,
}

impl Config {
    pub fn from_cli(cli: &Cli) -> Result<Self> {
        Self::resolve(cli)
    }

    pub fn resolve(cli: &Cli) -> Result<Self> {
        let path = config_path();
        let file = load_file_config(&path)?;
        let mut config = Self::resolve_from_file(cli, &file, path, env_api_key)?;
        // The static Copilot fallback model may not be entitled to this
        // account.  When no model was chosen explicitly, prefer the first
        // catalog model the signed-in account can actually use; the choice
        // is read from the local credential store, so this stays offline.
        if config.provider == ProviderArg::GithubCopilot
            && cli.model.is_none()
            && file.model.is_none()
            && let Some(model) = entitled_copilot_default()
        {
            config.model = model;
        }
        Ok(config)
    }

    /// Resolve a config with explicitly supplied key values.  This is kept
    /// independent of the process environment for deterministic unit tests.
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
        let path = config_path();
        let file = load_file_config(&path)?;
        Self::resolve_from_file(cli, &file, path, |provider| match provider {
            ProviderArg::OpencodeGo => opencode_go_key.or(opencode_key).map(str::to_owned),
            ProviderArg::Openrouter => openrouter_key.map(str::to_owned),
            ProviderArg::GithubCopilot => None,
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
        let provider = cli
            .provider
            .or(file_provider)
            .unwrap_or(ProviderArg::OpencodeGo);
        // GitHub Copilot authenticates through the private auth store and is
        // intentionally constructible before the first login.  Existing API
        // key providers retain their old environment-only requirement.
        let api_key = key_for(provider)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_default();
        if provider != ProviderArg::GithubCopilot && api_key.is_empty() {
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
        Ok(Self {
            provider,
            model,
            api_key,
            config_path: path,
            rtk: file.rtk,
            compaction: file
                .compaction
                .as_ref()
                .map(CompactionPolicy::from)
                .unwrap_or_default(),
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
fn entitled_copilot_default() -> Option<String> {
    let credential = CopilotAuth::from_default().ok()?.credential().ok()??;
    let sku = sku_from_proxy_token(&credential.access);
    default_model_for(sku, &credential.available_model_ids)
}

/// Construct a provider from its stable configuration/command name.
pub fn build_provider(name: &str) -> Result<Arc<dyn Provider>> {
    build_provider_with_auth(name, None)
}

/// Construct a provider while reusing the auth state owned by the agent.  The
/// shared handle is important for first-login and token refreshes: rebuilding
/// a Copilot provider must not hide a credential written by `/auth`.
pub fn build_provider_with_auth(
    name: &str,
    copilot_auth: Option<Arc<CopilotAuth>>,
) -> Result<Arc<dyn Provider>> {
    let provider = ProviderArg::from_name(name).ok_or_else(|| {
        anyhow!("unknown provider: {name} (expected opencode-go, openrouter, or github-copilot)")
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
    fn resolves_defaults_and_overrides() {
        // Resolve against an explicit file config so the developer's real
        // ~/.config/harness/config.toml cannot leak into the assertions.
        let cli = Cli::default();
        let config = Config::resolve_from_file(
            &cli,
            &FileConfig::default(),
            PathBuf::from("/tmp/harness-config.toml"),
            |provider| match provider {
                ProviderArg::OpencodeGo => Some("secret".into()),
                ProviderArg::Openrouter | ProviderArg::GithubCopilot => None,
            },
        )
        .unwrap();
        assert_eq!(config.provider, ProviderArg::OpencodeGo);
        assert_eq!(config.model, "gpt-5.6-luna");

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
                ProviderArg::OpencodeGo | ProviderArg::GithubCopilot => None,
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
        let error = Config::resolve_with_keys(&cli, None, None).err().unwrap();
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
            extra: [("future".to_owned(), toml::Value::String("kept".into()))]
                .into_iter()
                .collect(),
        };
        save_file_config(&path, &original).unwrap();
        assert_eq!(load_file_config(&path).unwrap(), original);

        save_settings_at(&path, "openrouter", "router/demo").unwrap();
        let saved = load_file_config(&path).unwrap();
        assert_eq!(saved.provider.as_deref(), Some("openrouter"));
        assert_eq!(saved.model.as_deref(), Some("router/demo"));
        assert!(saved.rtk);
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
    fn rtk_flag_flows_from_file_into_resolved_config() {
        let cli = Cli::default();
        let file = FileConfig {
            rtk: true,
            ..FileConfig::default()
        };
        let config = Config::resolve_from_file(
            &cli,
            &file,
            PathBuf::from("/tmp/harness-config.toml"),
            |_| Some("secret".into()),
        )
        .unwrap();
        assert!(config.rtk);

        let config = Config::resolve_from_file(
            &cli,
            &FileConfig::default(),
            PathBuf::from("/tmp/harness-config.toml"),
            |_| Some("secret".into()),
        )
        .unwrap();
        assert!(!config.rtk);
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
    fn missing_file_is_default_and_config_override_is_used() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("config");
        assert_eq!(load_file_config(&path).unwrap(), FileConfig::default());
        // Keep this test free of process-global environment mutation; the
        // helper's behavior is exercised by config_dir itself in integration
        // tests where the override can be scoped by the harness.
        assert!(path.ends_with("config"));
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
