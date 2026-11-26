//! Remote model catalogue sourced from [models.dev](https://models.dev).
//!
//! Subscription providers (Claude, ChatGPT Codex) validate model ids against a
//! local allowlist, so a newly released model used to require an app release.
//! This module supplements those built-in tables with models published on
//! models.dev. The built-in tables stay authoritative for ids they know (they
//! carry verified quirks); remote entries only *add* ids.
//!
//! Load time is unaffected: the filtered catalogue is cached as a small JSON
//! file under the Harness state directory and read lazily on first use. The
//! network is only touched by [`spawn_refresh`], a fire-and-forget background
//! task that runs when the cache is older than [`REFRESH_INTERVAL`]. Newly
//! fetched models therefore appear on the next lookup (for example the next
//! `/model`), never by blocking startup.
//!
//! Set `HARNESS_MODELS_DEV=off` to disable both the cache and the refresh.
use crate::{LlmError, ModelInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The public models.dev catalogue (all providers, a few MiB).
pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";
/// How long a cached catalogue is considered fresh.
pub const REFRESH_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// Catalogue key for the Claude subscription provider.
pub const ANTHROPIC_SUBSCRIPTION: &str = "anthropic-claude";
/// Catalogue key for the GitHub Copilot provider.
pub const GITHUB_COPILOT: &str = crate::providers::github_copilot::PROVIDER_NAME;
/// Catalogue key for the ChatGPT Codex subscription provider.
pub const OPENAI_CODEX: &str = crate::providers::openai_codex::PROVIDER_NAME;

const FETCH_TIMEOUT: Duration = Duration::from_secs(30);
/// The whole models.dev document is larger than typical metadata bodies.
const MAX_CATALOGUE_BYTES: usize = 64 * 1024 * 1024;
const CACHE_FILE_NAME: &str = "models.json";
const CACHE_VERSION: u32 = 2;
/// Claude models released before this predate the subscription catalogue
/// (and the built-in table already covers everything after it that matters).
const CLAUDE_MIN_RELEASE: &str = "2025-09-01";
/// models.dev lists OpenAI *API* models; Codex accepts only recent general
/// GPT models, so older or specialised variants are filtered out.
const CODEX_MIN_RELEASE: &str = "2026-03-01";
const CODEX_EXCLUDED: &[&str] = &[
    "-pro",
    "-nano",
    "-latest",
    "-chat",
    "-audio",
    "-search",
    "-image",
    "-transcribe",
    "-tts",
    "realtime",
    "daybreak",
];

/// One model published on models.dev, reduced to what providers need.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteModel {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    /// Supported reasoning effort values (`reasoning_options[type=effort]`).
    #[serde(default)]
    pub efforts: Vec<String>,
    /// Whether the model accepts an explicit thinking token budget.
    #[serde(default)]
    pub budget_tokens: bool,
    /// Maximum output tokens (`limit.output`).
    #[serde(default)]
    pub max_output: Option<u64>,
    /// Whether the model supports reasoning at all.
    #[serde(default)]
    pub reasoning: bool,
    /// Wire protocol hint from models.dev's `provider.npm`: `"anthropic"`
    /// (Messages), `"responses"` (OpenAI Responses), or `None` (chat
    /// completions, models.dev's default for OpenAI-compatible models).
    #[serde(default)]
    pub dialect: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    /// Unix seconds of the last successful fetch (or 304 revalidation).
    fetched_at: u64,
    #[serde(default)]
    etag: Option<String>,
    #[serde(default)]
    providers: BTreeMap<String, Vec<RemoteModel>>,
}

#[derive(Deserialize)]
struct WireModel {
    #[serde(default)]
    name: Option<String>,
    /// Present for non-chat models (`image`, `embedding`, `realtime`, ...).
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    release_date: Option<String>,
    #[serde(default)]
    reasoning_options: Vec<WireReasoningOption>,
    #[serde(default)]
    limit: Option<WireLimit>,
    #[serde(default)]
    provider: Option<WireProvider>,
}

#[derive(Deserialize)]
struct WireProvider {
    #[serde(default)]
    npm: Option<String>,
}

#[derive(Deserialize)]
struct WireReasoningOption {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    values: Vec<String>,
}

#[derive(Deserialize)]
struct WireLimit {
    #[serde(default)]
    context: Option<u64>,
    #[serde(default)]
    output: Option<u64>,
}

impl WireModel {
    fn into_remote(self, id: &str) -> RemoteModel {
        let efforts = self
            .reasoning_options
            .iter()
            .filter(|option| option.kind == "effort")
            .flat_map(|option| option.values.iter().cloned())
            .collect();
        let budget_tokens = self
            .reasoning_options
            .iter()
            .any(|option| option.kind == "budget_tokens");
        let dialect = match self.provider.and_then(|provider| provider.npm).as_deref() {
            Some("@ai-sdk/anthropic") => Some("anthropic".to_owned()),
            Some("@ai-sdk/openai") => Some("responses".to_owned()),
            _ => None,
        };
        RemoteModel {
            id: id.to_owned(),
            name: self.name,
            context_length: self.limit.as_ref().and_then(|limit| limit.context),
            efforts,
            budget_tokens,
            max_output: self.limit.and_then(|limit| limit.output),
            reasoning: self.reasoning,
            dialect,
        }
    }

    fn released_since(&self, floor: &str) -> bool {
        self.release_date
            .as_deref()
            .is_some_and(|date| date >= floor)
    }
}

/// `claude-<family>-<major>[-<minor>]`, excluding dated snapshot ids such as
/// `claude-opus-4-5-20251101` (the subscription uses the aliases).
fn is_claude_alias(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("claude-") else {
        return false;
    };
    let mut parts = rest.split('-');
    let family_ok = parts
        .next()
        .is_some_and(|family| !family.is_empty() && family.bytes().all(|b| b.is_ascii_lowercase()));
    let versions: Vec<&str> = parts.collect();
    family_ok
        && (1..=2).contains(&versions.len())
        && versions
            .iter()
            .all(|part| (1..=2).contains(&part.len()) && part.bytes().all(|b| b.is_ascii_digit()))
}

fn is_claude_subscription(id: &str, model: &WireModel) -> bool {
    model.kind.is_none()
        && model.tool_call
        && model.released_since(CLAUDE_MIN_RELEASE)
        && is_claude_alias(id)
}

fn is_codex(id: &str, model: &WireModel) -> bool {
    model.kind.is_none()
        && model.reasoning
        && model.tool_call
        && model.released_since(CODEX_MIN_RELEASE)
        && id.starts_with("gpt-")
        && !CODEX_EXCLUDED.iter().any(|excluded| id.contains(excluded))
}

/// Copilot serves chat models behind one token; keep anything that can call
/// tools and is not an image/embedding/realtime model. The account's own
/// model list still decides what is actually offered.
fn is_copilot(_id: &str, model: &WireModel) -> bool {
    model.kind.is_none() && model.tool_call
}

type Filter = fn(&str, &WireModel) -> bool;

/// Extract the subscription-relevant models from a models.dev `api.json`
/// document, newest first. Malformed individual entries are skipped; a
/// document without either source provider is an error so a broken response
/// never replaces a good cache.
pub(crate) fn parse_models_dev(body: &str) -> Result<BTreeMap<String, Vec<RemoteModel>>, LlmError> {
    let root: BTreeMap<String, Value> = serde_json::from_str(body)
        .map_err(|error| LlmError::Parse(format!("models.dev catalogue: {error}")))?;
    let sources: [(&str, &str, Filter); 3] = [
        ("anthropic", ANTHROPIC_SUBSCRIPTION, is_claude_subscription),
        ("openai", OPENAI_CODEX, is_codex),
        ("github-copilot", GITHUB_COPILOT, is_copilot),
    ];
    let mut out = BTreeMap::new();
    for (source, provider, filter) in sources {
        let Some(models) = root
            .get(source)
            .and_then(|value| value.get("models"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let mut list: Vec<(Option<String>, RemoteModel)> = models
            .iter()
            .filter_map(|(id, value)| {
                let model = WireModel::deserialize(value).ok()?;
                filter(id, &model).then(|| (model.release_date.clone(), model.into_remote(id)))
            })
            .collect();
        list.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
        out.insert(
            provider.to_owned(),
            list.into_iter().map(|(_, model)| model).collect(),
        );
    }
    if out.is_empty() {
        return Err(LlmError::Parse(
            "models.dev catalogue has no anthropic or openai provider".into(),
        ));
    }
    Ok(out)
}

/// Append remote-only models to a curated built-in list. Built-in entries
/// keep their order and metadata (they encode verified quirks); a missing
/// built-in context length is filled from the remote entry. Remote-only ids
/// are newer than the built-in table by construction, so they come first.
pub fn merge_models(builtin: Vec<ModelInfo>, remote: &[RemoteModel]) -> Vec<ModelInfo> {
    let mut builtin = builtin;
    for model in &mut builtin {
        if model.context_length.is_none()
            && let Some(remote) = remote.iter().find(|remote| remote.id == model.id)
        {
            model.context_length = remote.context_length;
        }
    }
    let mut merged: Vec<ModelInfo> = remote
        .iter()
        .filter(|remote| !builtin.iter().any(|model| model.id == remote.id))
        .map(|remote| ModelInfo {
            id: remote.id.clone(),
            name: remote.name.clone().or_else(|| Some(remote.id.clone())),
            context_length: remote.context_length,
        })
        .collect();
    merged.extend(builtin);
    merged
}

fn disabled() -> bool {
    std::env::var("HARNESS_MODELS_DEV").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    })
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// `$HARNESS_STATE_DIR/models.json`, else `~/.harness/models.json`.
pub fn default_cache_path() -> Option<PathBuf> {
    let root = non_empty_env_path("HARNESS_STATE_DIR").or_else(|| {
        non_empty_env_path("HOME")
            .or_else(|| non_empty_env_path("USERPROFILE"))
            .map(|home| home.join(".harness"))
    })?;
    Some(root.join(CACHE_FILE_NAME))
}

fn load_cache(path: &Path) -> Option<CacheFile> {
    let text = std::fs::read_to_string(path).ok()?;
    let cache: CacheFile = serde_json::from_str(&text).ok()?;
    (cache.version == CACHE_VERSION).then_some(cache)
}

/// Write via a sibling temp file and rename so a process exiting mid-write
/// (headless runs end while the refresh may still be in flight) never leaves
/// a truncated cache behind.
fn save_cache(path: &Path, cache: &CacheFile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec_pretty(cache).map_err(std::io::Error::other)?;
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

fn is_stale(cache: &CacheFile, now: u64) -> bool {
    cache.providers.is_empty() || now.saturating_sub(cache.fetched_at) >= REFRESH_INTERVAL.as_secs()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn state() -> &'static RwLock<CacheFile> {
    static STATE: OnceLock<RwLock<CacheFile>> = OnceLock::new();
    STATE.get_or_init(|| {
        // Unit tests must not observe the developer's real cache.
        let cache = if cfg!(test) || disabled() {
            None
        } else {
            default_cache_path().and_then(|path| load_cache(&path))
        };
        RwLock::new(cache.unwrap_or_default())
    })
}

/// Cached remote models for a provider key ([`ANTHROPIC_SUBSCRIPTION`] or
/// [`OPENAI_CODEX`]), newest first. Never touches the network.
pub fn remote_models(provider: &str) -> Vec<RemoteModel> {
    state()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .providers
        .get(provider)
        .cloned()
        .unwrap_or_default()
}

/// Look up one cached remote model.
pub fn remote_model(provider: &str, id: &str) -> Option<RemoteModel> {
    remote_models(provider)
        .into_iter()
        .find(|model| model.id == id)
}

/// Refresh the cache in the background when it is stale. Returns
/// immediately; must be called inside a Tokio runtime (otherwise it is a
/// no-op). Failures are logged and leave the existing cache in place.
pub fn spawn_refresh() {
    if disabled() {
        return;
    }
    let Some(path) = default_cache_path() else {
        return;
    };
    let stale = is_stale(
        &state()
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        unix_now(),
    );
    if !stale {
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn(async move {
        match refresh(&path).await {
            Ok(()) => tracing::debug!("models.dev catalogue refreshed"),
            Err(error) => tracing::debug!("models.dev catalogue refresh failed: {error}"),
        }
    });
}

async fn refresh(path: &Path) -> Result<(), LlmError> {
    use reqwest::header::{ETAG, IF_NONE_MATCH};
    let previous = state()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let mut request = crate::http::streaming_client()
        .get(MODELS_DEV_URL)
        .timeout(FETCH_TIMEOUT);
    if let Some(etag) = previous
        .etag
        .as_deref()
        .filter(|_| !previous.providers.is_empty())
    {
        request = request.header(IF_NONE_MATCH, etag);
    }
    let response = request.send().await.map_err(LlmError::Network)?;
    let status = response.status();
    let cache = if status == reqwest::StatusCode::NOT_MODIFIED {
        CacheFile {
            fetched_at: unix_now(),
            ..previous
        }
    } else if status.is_success() {
        let etag = response
            .headers()
            .get(ETAG)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = crate::http::read_bounded_text(response, MAX_CATALOGUE_BYTES).await?;
        CacheFile {
            version: CACHE_VERSION,
            fetched_at: unix_now(),
            etag,
            providers: parse_models_dev(&body)?,
        }
    } else {
        return Err(LlmError::Parse(format!(
            "models.dev returned HTTP {status}"
        )));
    };
    if let Err(error) = save_cache(path, &cache) {
        tracing::debug!("could not write {}: {error}", path.display());
    }
    *state()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = cache;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
      "anthropic": {"models": {
        "claude-fable-5-2": {"name": "Claude Fable 5.2", "tool_call": true, "reasoning": true,
          "release_date": "2026-11-09", "limit": {"context": 1000000},
          "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}]},
        "claude-opus-4-5": {"name": "Claude Opus 4.5", "tool_call": true, "reasoning": true,
          "release_date": "2025-11-24", "limit": {"context": 200000},
          "reasoning_options": [{"type": "effort", "values": ["low"]}, {"type": "budget_tokens", "min": 1024}]},
        "claude-opus-4-5-20251101": {"tool_call": true, "release_date": "2025-11-24"},
        "claude-3-5-sonnet": {"tool_call": true, "release_date": "2024-10-22"},
        "claude-opus-4": {"tool_call": true, "release_date": "2025-05-22"},
        "broken": 42
      }},
      "openai": {"models": {
        "gpt-6.1-sol": {"tool_call": true, "reasoning": true, "release_date": "2026-09-29",
          "limit": {"context": 1050000}},
        "gpt-5.5-pro": {"tool_call": true, "reasoning": true, "release_date": "2026-04-01"},
        "gpt-image-2": {"type": "image", "release_date": "2026-04-21"},
        "gpt-4o": {"tool_call": true, "release_date": "2024-05-13"},
        "o3": {"tool_call": true, "reasoning": true, "release_date": "2026-04-16"}
      }},
      "github-copilot": {"models": {
        "claude-fable-5.2": {"tool_call": true, "reasoning": true, "release_date": "2026-11-09",
          "limit": {"context": 1000000, "output": 128000},
          "provider": {"npm": "@ai-sdk/anthropic"}},
        "gpt-6.1-sol": {"tool_call": true, "reasoning": true, "release_date": "2026-09-29",
          "provider": {"npm": "@ai-sdk/openai"}},
        "kimi-k3": {"tool_call": true, "reasoning": true, "release_date": "2026-07-16"},
        "text-embedding": {"type": "embedding", "tool_call": true}
      }},
      "other": {"weird": true}
    }"#;

    #[test]
    fn parses_and_filters_subscription_models() {
        let parsed = parse_models_dev(FIXTURE).unwrap();
        let claude: Vec<_> = parsed[ANTHROPIC_SUBSCRIPTION]
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(claude, ["claude-fable-5-2", "claude-opus-4-5"]);
        let fable = &parsed[ANTHROPIC_SUBSCRIPTION][0];
        assert_eq!(fable.context_length, Some(1_000_000));
        assert_eq!(fable.efforts, ["low", "high", "max"]);
        assert!(!fable.budget_tokens);
        assert!(parsed[ANTHROPIC_SUBSCRIPTION][1].budget_tokens);
        let codex: Vec<_> = parsed[OPENAI_CODEX].iter().map(|m| m.id.as_str()).collect();
        assert_eq!(codex, ["gpt-6.1-sol"]);
    }

    #[test]
    fn copilot_models_carry_dialect_and_limits() {
        let parsed = parse_models_dev(FIXTURE).unwrap();
        let copilot = &parsed[GITHUB_COPILOT];
        let dialect = |id: &str| {
            copilot
                .iter()
                .find(|m| m.id == id)
                .map(|m| (m.dialect.clone(), m.max_output))
        };
        assert_eq!(copilot.len(), 3);
        assert_eq!(
            dialect("claude-fable-5.2"),
            Some((Some("anthropic".into()), Some(128_000)))
        );
        assert_eq!(
            dialect("gpt-6.1-sol"),
            Some((Some("responses".into()), None))
        );
        assert_eq!(dialect("kimi-k3"), Some((None, None)));
    }

    #[test]
    fn rejects_documents_without_source_providers() {
        assert!(parse_models_dev(r#"{"other": {}}"#).is_err());
        assert!(parse_models_dev("not json").is_err());
    }

    #[test]
    fn claude_alias_shape() {
        assert!(is_claude_alias("claude-fable-5"));
        assert!(is_claude_alias("claude-opus-4-8"));
        assert!(!is_claude_alias("claude-opus-4-5-20251101"));
        assert!(!is_claude_alias("claude-3-5-sonnet"));
        assert!(!is_claude_alias("gpt-5"));
    }

    #[test]
    fn merge_keeps_builtin_and_prepends_new_models() {
        let builtin = vec![
            ModelInfo {
                id: "a".into(),
                name: Some("A".into()),
                context_length: None,
            },
            ModelInfo {
                id: "b".into(),
                name: Some("B".into()),
                context_length: Some(1),
            },
        ];
        let remote = |id: &str, context| RemoteModel {
            id: id.into(),
            name: None,
            context_length: Some(context),
            efforts: vec![],
            budget_tokens: false,
            max_output: None,
            reasoning: false,
            dialect: None,
        };
        let merged = merge_models(builtin, &[remote("new", 9), remote("a", 5), remote("b", 7)]);
        let ids: Vec<_> = merged.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["new", "a", "b"]);
        assert_eq!(merged[0].name.as_deref(), Some("new"));
        assert_eq!(merged[1].context_length, Some(5));
        assert_eq!(merged[2].context_length, Some(1));
    }

    #[test]
    fn cache_round_trip_and_staleness() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join(CACHE_FILE_NAME);
        let cache = CacheFile {
            version: CACHE_VERSION,
            fetched_at: 100,
            etag: Some("\"x\"".into()),
            providers: parse_models_dev(FIXTURE).unwrap(),
        };
        save_cache(&path, &cache).unwrap();
        assert_eq!(load_cache(&path), Some(cache.clone()));
        assert!(!is_stale(&cache, 100 + REFRESH_INTERVAL.as_secs() - 1));
        assert!(is_stale(&cache, 100 + REFRESH_INTERVAL.as_secs()));
        assert!(is_stale(&CacheFile::default(), 0));
        std::fs::write(&path, r#"{"version": 999, "fetched_at": 1}"#).unwrap();
        assert_eq!(load_cache(&path), None);
    }
}
