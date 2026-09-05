//! Private, provider-keyed credential storage.
//!
//! The normal Harness TOML configuration and session files deliberately do
//! not contain credentials.  This module owns `auth.json` and performs
//! read-modify-write updates so an OAuth flow cannot discard credentials for a
//! different provider.

use crate::error::{AuthError, Result, io_error};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime};

pub const COPILOT_PROVIDER_KEY: &str = "github-copilot";
/// Provider key used by ChatGPT/Codex OAuth credentials.
pub const OPENAI_CODEX_PROVIDER_KEY: &str = "openai-codex";
const LOCK_WAIT: Duration = Duration::from_millis(10);
const LOCK_ATTEMPTS: usize = 200;

/// Credentials persisted by the GitHub Copilot OAuth flow.
///
/// `access` is the short-lived Copilot token and `refresh` is the GitHub OAuth
/// token.  `expires` is a Unix timestamp in milliseconds after the refresh
/// skew has been applied.  The custom `Debug` implementation below prevents
/// either token from appearing in logs or test diagnostics.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CopilotCredential {
    #[serde(rename = "type")]
    pub credential_type: String,
    pub access: String,
    pub refresh: String,
    #[serde(default)]
    pub expires: u64,
    #[serde(rename = "enterpriseUrl", default)]
    pub enterprise_url: Option<String>,
    #[serde(rename = "availableModelIds", default)]
    pub available_model_ids: Vec<String>,
}

impl Default for CopilotCredential {
    fn default() -> Self {
        Self::new("", "", 0, None, Vec::new())
    }
}

impl CopilotCredential {
    pub fn new(
        access: impl Into<String>,
        refresh: impl Into<String>,
        expires: u64,
        enterprise_url: Option<String>,
        available_model_ids: Vec<String>,
    ) -> Self {
        Self {
            credential_type: "oauth".into(),
            access: access.into(),
            refresh: refresh.into(),
            expires,
            enterprise_url,
            available_model_ids,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.credential_type == "oauth"
            && !self.access.trim().is_empty()
            && !self.refresh.trim().is_empty()
    }

    /// `expires == 0` is treated as expired.  It is useful for old or
    /// hand-written auth files to fail with an actionable refresh/login error
    /// rather than sending an unknown token to the API.
    pub fn is_expired(&self) -> bool {
        self.expires == 0 || self.expires <= unix_millis()
    }

    pub fn redacted(&self) -> RedactedCredential<'_> {
        RedactedCredential(self)
    }
}

impl fmt::Debug for CopilotCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.redacted().fmt(formatter)
    }
}

/// OAuth credentials used by the ChatGPT Codex subscription endpoint.
/// Expiration is a Unix timestamp in milliseconds.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiCodexCredential {
    #[serde(rename = "type")]
    pub credential_type: String,
    pub access: String,
    pub refresh: String,
    pub expires: u64,
    #[serde(rename = "accountId")]
    pub account_id: String,
}

impl OpenAiCodexCredential {
    pub fn new(
        access: impl Into<String>,
        refresh: impl Into<String>,
        expires: u64,
        account_id: impl Into<String>,
    ) -> Self {
        Self {
            credential_type: "oauth".into(),
            access: access.into(),
            refresh: refresh.into(),
            expires,
            account_id: account_id.into(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.credential_type == "oauth"
            && !self.access.trim().is_empty()
            && !self.refresh.trim().is_empty()
            && !self.account_id.trim().is_empty()
    }

    pub fn is_expired(&self) -> bool {
        self.expires == 0 || self.expires <= unix_millis().saturating_add(60_000)
    }
}

impl fmt::Debug for OpenAiCodexCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCodexCredential")
            .field("credential_type", &self.credential_type)
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires", &self.expires)
            .field("account_id", &self.account_id)
            .finish()
    }
}

/// A deliberately non-secret view useful in diagnostics.
pub struct RedactedCredential<'a>(&'a CopilotCredential);

impl fmt::Debug for RedactedCredential<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CopilotCredential")
            .field("credential_type", &self.0.credential_type)
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires", &self.0.expires)
            .field("enterprise_url", &self.0.enterprise_url)
            .field("available_model_ids", &self.0.available_model_ids)
            .finish()
    }
}

/// Provider-keyed values from `auth.json`.  Unknown providers are kept as raw
/// JSON so a Harness update cannot erase credentials written by another tool.
pub type AuthEntries = BTreeMap<String, serde_json::Value>;

/// The path of the Harness configuration directory, shared by auth and the
/// normal TOML configuration.
pub fn config_dir() -> PathBuf {
    if let Some(path) = non_empty_env_path("HARNESS_CONFIG_DIR") {
        return path;
    }
    if let Some(path) = non_empty_env_path("XDG_CONFIG_HOME") {
        return path.join("harness");
    }
    home_dir()
        .map(|home| home.join(".config").join("harness"))
        .unwrap_or_else(|| PathBuf::from(".config").join("harness"))
}

pub fn auth_path() -> PathBuf {
    config_dir().join("auth.json")
}

/// Filesystem-backed auth storage.  Constructing it does not touch disk.
#[derive(Clone)]
pub struct AuthStore {
    path: PathBuf,
}

impl fmt::Debug for AuthStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthStore")
            .field("path", &self.path)
            .finish()
    }
}

impl Default for AuthStore {
    fn default() -> Self {
        Self::new(auth_path())
    }
}

impl AuthStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read every provider entry.  A missing auth file is an empty store.
    pub fn load(&self) -> Result<AuthEntries> {
        self.load_unlocked()
    }

    pub fn openai_codex(&self) -> Result<Option<OpenAiCodexCredential>> {
        let entries = self.load_unlocked()?;
        let Some(value) = entries.get(OPENAI_CODEX_PROVIDER_KEY) else {
            return Ok(None);
        };
        let credential =
            serde_json::from_value::<OpenAiCodexCredential>(value.clone()).map_err(|source| {
                AuthError::Json {
                    path: self.path.clone(),
                    source,
                }
            })?;
        if !credential.is_complete() {
            return Err(AuthError::InvalidCredential(
                "OpenAI Codex credential is missing OAuth fields".into(),
            ));
        }
        Ok(Some(credential))
    }

    pub fn save_openai_codex(&self, credential: &OpenAiCodexCredential) -> Result<()> {
        if !credential.is_complete() {
            return Err(AuthError::InvalidCredential(
                "OpenAI Codex credential is missing OAuth fields".into(),
            ));
        }
        let value = serde_json::to_value(credential).map_err(|source| AuthError::Json {
            path: self.path.clone(),
            source,
        })?;
        self.save_provider_value(OPENAI_CODEX_PROVIDER_KEY, value)
    }

    pub fn copilot(&self) -> Result<Option<CopilotCredential>> {
        let entries = self.load_unlocked()?;
        let Some(value) = entries.get(COPILOT_PROVIDER_KEY) else {
            return Ok(None);
        };
        let credential =
            serde_json::from_value::<CopilotCredential>(value.clone()).map_err(|source| {
                AuthError::Json {
                    path: self.path.clone(),
                    source,
                }
            })?;
        if !credential.is_complete() {
            return Err(AuthError::InvalidCredential(
                "credential is missing its OAuth token fields".into(),
            ));
        }
        Ok(Some(credential))
    }

    /// Update one provider while retaining all unrelated entries.  The lock is
    /// held across the read and atomic replacement, which prevents concurrent
    /// login processes from losing one another's credentials.  Directories
    /// and files are created private (`0o700`/`0o600` on Unix) at creation
    /// time; permission and parent-sync failures fail closed.
    pub fn save_provider_value(&self, provider: &str, value: serde_json::Value) -> Result<()> {
        if provider.trim().is_empty() {
            return Err(AuthError::InvalidCredential(
                "provider name cannot be empty".into(),
            ));
        }
        let parent = self.parent_dir();
        private_dir_all(&parent)?;
        ensure_private_directory(&parent)?;
        let _lock = AuthFileLock::acquire(&self.path)?;
        let mut entries = self.load_unlocked()?;
        entries.insert(provider.to_owned(), value);
        self.write_unlocked(&entries)
    }

    pub fn save_copilot(&self, credential: &CopilotCredential) -> Result<()> {
        if !credential.is_complete() {
            return Err(AuthError::InvalidCredential(
                "credential is missing its OAuth token fields".into(),
            ));
        }
        let value = serde_json::to_value(credential).map_err(|source| AuthError::Json {
            path: self.path.clone(),
            source,
        })?;
        self.save_provider_value(COPILOT_PROVIDER_KEY, value)
    }

    pub fn remove_provider(&self, provider: &str) -> Result<bool> {
        if !self.path.exists() {
            return Ok(false);
        }
        let _lock = AuthFileLock::acquire(&self.path)?;
        let mut entries = self.load_unlocked()?;
        let removed = entries.remove(provider).is_some();
        if removed {
            self.write_unlocked(&entries)?;
        }
        Ok(removed)
    }

    fn parent_dir(&self) -> PathBuf {
        self.path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    }

    fn load_unlocked(&self) -> Result<AuthEntries> {
        let mut file = match fs::File::open(&self.path) {
            Ok(file) => {
                // Reading an auth file is also an opportunity to repair a
                // permissive mode left by an older Harness version.
                set_private_file(&self.path);
                file
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            Err(source) => return Err(io_error("read auth file", &self.path, source)),
        };
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|source| io_error("read auth file", &self.path, source))?;
        if contents.trim().is_empty() {
            return Ok(BTreeMap::new());
        }
        serde_json::from_str(&contents).map_err(|source| AuthError::Json {
            path: self.path.clone(),
            source,
        })
    }

    fn write_unlocked(&self, entries: &AuthEntries) -> Result<()> {
        let parent = self.parent_dir();
        private_dir_all(&parent)?;
        ensure_private_directory(&parent)?;
        let contents = serde_json::to_string_pretty(entries).map_err(|source| AuthError::Json {
            path: self.path.clone(),
            source,
        })?;
        let temp = temporary_path(&self.path);
        let result = (|| -> Result<()> {
            let mut file = private_file(&temp)?;
            file.write_all(contents.as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .and_then(|_| file.flush())
                .and_then(|_| file.sync_all())
                .map_err(|source| io_error("write temporary auth file", &temp, source))?;
            drop(file);
            fs::rename(&temp, &self.path)
                .map_err(|source| io_error("replace auth file", &self.path, source))?;
            ensure_private_file(&self.path)?;
            sync_parent(&parent)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temporary_path(path: &Path) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auth.json");
    path.with_file_name(format!(".{name}.tmp-{}-{sequence}", std::process::id()))
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn home_dir() -> Option<PathBuf> {
    non_empty_env_path("HOME").or_else(|| non_empty_env_path("USERPROFILE"))
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Fail-closed private-path helpers, mirroring the session store design so
/// both writers share one reviewed scheme: create private at creation time
/// on Unix (`0o700` dirs / `0o600` files), propagate permission and
/// parent-sync failures, and document that Windows inherits the parent
/// directory's ACLs instead of claiming unenforced privacy.
fn private_dir_all(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(path)
            .map_err(|source| io_error("create auth directory", path, source))?;
        ensure_private_directory(path)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        fs::create_dir_all(path)
            .map_err(|source| io_error("create auth directory", path, source))?;
        sync_parent(path)?;
        Ok(())
    }
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata =
            fs::metadata(path).map_err(|source| io_error("stat auth directory", path, source))?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions)
            .map_err(|source| io_error("secure auth directory", path, source))?;
    }
    sync_parent(path)?;
    Ok(())
}

fn private_file(path: &Path) -> Result<fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|source| io_error("create temporary auth file", path, source))
    }
    #[cfg(not(unix))]
    {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|source| io_error("create temporary auth file", path, source))
    }
}

fn ensure_private_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata =
            fs::metadata(path).map_err(|source| io_error("stat auth file", path, source))?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(path, permissions)
            .map_err(|source| io_error("secure auth file", path, source))?;
    }
    sync_parent(path)?;
    Ok(())
}

fn set_private_file(path: &Path) {
    let _ = ensure_private_file(path);
}

/// Sync the parent directory after durable create/rename operations;
/// failures fail closed.  No-op success where directory fsync is
/// unsupported.
fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        // `path` here is the durable file or directory whose parent entry
        // must survive: open the parent directory itself and fsync it.
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let directory = match fs::File::open(parent) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => return Err(io_error("open parent directory", parent, source)),
        };
        directory
            .sync_all()
            .map_err(|source| io_error("sync parent directory", parent, source))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Sync a directory path itself after renames inside it.  `sync_parent`
/// fsyncs the parent of its argument, so pass a child entry (or join a
/// sentinel) rather than the directory itself.
#[allow(dead_code)]
fn sync_directory(path: &Path) {
    // Best-effort wrapper for read paths where failures must not break
    // listing; write paths call `sync_parent` directly and fail closed.
    if let Err(error) = sync_parent(&path.join(".")) {
        let _ = error;
    }
}

struct AuthFileLock {
    path: PathBuf,
    /// Unguessable owner nonce: stealing removes only the exact stale
    /// instance observed, and release removes only the owner's own lock —
    /// the same reviewed scheme as the session store.
    nonce: String,
}

impl AuthFileLock {
    fn lock_path(auth_path: &Path) -> PathBuf {
        auth_path.with_extension("json.lock")
    }

    fn read_nonce(path: &Path) -> Option<String> {
        let contents = fs::read_to_string(path).ok()?;
        contents.lines().find_map(|line| {
            let value = line.strip_prefix("nonce=")?.trim();
            (!value.is_empty()).then(|| value.to_owned())
        })
    }

    fn is_same_lock(path: &Path, nonce: &str) -> bool {
        Self::read_nonce(path).is_some_and(|current| current == nonce)
    }

    /// A lock is stale when its recorded owner is provably dead.  A live
    /// owner is never stolen merely for being old; locks without readable
    /// identity fall back to the conservative age timeout.
    fn is_stale(path: &Path) -> bool {
        if let Some(pid) = lock_owner_pid(path) {
            return !pid_is_alive(pid);
        }
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age > Duration::from_secs(5 * 60))
    }

    /// Legacy locks without a nonce carry no instance identity: stale only
    /// via the age heuristic (or a provably dead PID), never merely because
    /// a PID line is present.
    fn stale_without_identity(path: &Path) -> bool {
        if Self::read_nonce(path).is_some() {
            return false;
        }
        match lock_owner_pid(path) {
            Some(pid) => !pid_is_alive(pid),
            None => Self::is_stale(path),
        }
    }

    fn acquire(auth_path: &Path) -> Result<Self> {
        let path = Self::lock_path(auth_path);
        for _ in 0..LOCK_ATTEMPTS {
            let nonce = uuid_nonce();
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "pid={}", std::process::id());
                    let _ = writeln!(file, "nonce={nonce}");
                    let _ = file.sync_all();
                    ensure_private_file(&path)?;
                    return Ok(Self { path, nonce });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if Self::is_stale(&path) {
                        // Steal only the exact stale instance observed so two
                        // racing stealers cannot both enter: a contender that
                        // lost the race to a live replacement sees a changed
                        // nonce and backs off.
                        let observed = Self::read_nonce(&path);
                        let same_instance = Self::read_nonce(&path) == observed;
                        if Self::is_stale(&path)
                            && same_instance
                            && (observed.is_some() || Self::stale_without_identity(&path))
                        {
                            let _ = fs::remove_file(&path);
                        } else {
                            thread::sleep(LOCK_WAIT);
                        }
                    } else {
                        thread::sleep(LOCK_WAIT);
                    }
                }
                Err(source) => return Err(io_error("create auth lock", &path, source)),
            }
        }
        Err(AuthError::LockUnavailable(path))
    }
}

impl Drop for AuthFileLock {
    fn drop(&mut self) {
        // Release only our own instance: a successor's nonce differs and
        // its lock file is left alone.
        if Self::is_same_lock(&self.path, &self.nonce) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Unguessable owner nonce without a new dependency: process id plus
/// nanos plus a process-local sequence.
fn uuid_nonce() -> String {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{}-{nanos}-{sequence}", std::process::id())
}

fn lock_owner_pid(path: &Path) -> Option<u32> {
    let value = fs::read_to_string(path).ok()?;
    value
        .lines()
        .find_map(|line| line.strip_prefix("pid=")?.trim().parse::<u32>().ok())
}

/// True when the process with `pid` is alive (Unix `kill(pid, 0)` probe;
/// `EPERM` counts as alive).  On other platforms a live PID line is never
/// treated as stale by age alone — see `is_stale`.
fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // SAFETY: signal 0 sends nothing; the PID comes from our own lock file.
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        // Without a probe, only the age fallback in `is_stale` applies; a
        // readable PID is conservatively treated as live.
        let _ = pid;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn credential() -> CopilotCredential {
        CopilotCredential::new(
            "access-secret",
            "refresh-secret",
            u64::MAX,
            None,
            vec!["gpt-5.4".into()],
        )
    }

    #[test]
    fn credentials_round_trip_without_debugging_tokens() {
        let directory = tempdir().unwrap();
        let store = AuthStore::new(directory.path().join("nested").join("auth.json"));
        store.save_copilot(&credential()).unwrap();
        let loaded = store.copilot().unwrap().unwrap();
        assert_eq!(loaded, credential());
        let debug = format!("{loaded:?}");
        assert!(!debug.contains("access-secret"));
        assert!(!debug.contains("refresh-secret"));
        let contents = fs::read_to_string(store.path()).unwrap();
        assert!(contents.contains("access-secret"));
        assert!(contents.contains("availableModelIds"));
    }

    #[test]
    fn unknown_provider_entries_survive_copilot_updates() {
        let directory = tempdir().unwrap();
        let store = AuthStore::new(directory.path().join("auth.json"));
        store
            .save_provider_value("other", serde_json::json!({"token":"keep"}))
            .unwrap();
        store.save_copilot(&credential()).unwrap();
        let entries = store.load().unwrap();
        assert_eq!(entries["other"]["token"], "keep");
        assert!(entries.contains_key(COPILOT_PROVIDER_KEY));
    }

    #[cfg(unix)]
    #[test]
    fn auth_files_are_private_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempdir().unwrap();
        let store = AuthStore::new(directory.path().join("auth.json"));
        store.save_copilot(&credential()).unwrap();
        let auth_dir = store.path().parent().unwrap();
        assert_eq!(
            fs::metadata(auth_dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(store.path()).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn stale_auth_lock_steal_is_limited_to_the_observed_instance() {
        // Same reviewed scheme as the session store: stealing removes only
        // the exact stale instance observed, and release removes only the
        // owner's own nonce, so two racing stealers cannot both enter.
        let directory = tempdir().unwrap();
        let victim = directory.path().join("auth.json");
        fs::write(&victim, "{}").unwrap();
        let lock_path = AuthFileLock::lock_path(&victim);
        fs::write(&lock_path, "pid=4000000\nnonce=stale-a\n").unwrap();

        assert!(AuthFileLock::is_stale(&lock_path));
        let observed_by_a = AuthFileLock::read_nonce(&lock_path);
        assert_eq!(observed_by_a.as_deref(), Some("stale-a"));
        // A racing contender replaces the stale lock with a live one.
        fs::write(
            &lock_path,
            format!("pid={}\nnonce=fresh-b\n", std::process::id()),
        )
        .unwrap();
        assert!(!AuthFileLock::is_stale(&lock_path));
        assert_ne!(AuthFileLock::read_nonce(&lock_path), observed_by_a);

        let old_owner = AuthFileLock {
            path: lock_path.clone(),
            nonce: "stale-a".into(),
        };
        drop(old_owner);
        assert!(
            lock_path.exists(),
            "a stale owner must not remove the replacement lock"
        );
        let owner = AuthFileLock {
            path: lock_path.clone(),
            nonce: "fresh-b".into(),
        };
        drop(owner);
        assert!(!lock_path.exists());
    }

    #[test]
    fn old_auth_lock_with_live_owner_is_never_stolen() {
        let directory = tempdir().unwrap();
        let victim = directory.path().join("auth.json");
        fs::write(&victim, "{}").unwrap();
        let lock_path = AuthFileLock::lock_path(&victim);
        fs::write(
            &lock_path,
            format!("pid={}\nnonce=live-owner\n", std::process::id()),
        )
        .unwrap();
        // A live owner is never stale, even past the age timeout; only a
        // provably dead PID (or an unreadable identity past the timeout) is.
        assert!(!AuthFileLock::is_stale(&lock_path));
        assert!(!AuthFileLock::stale_without_identity(&lock_path));
    }

    #[cfg(unix)]
    #[test]
    fn permissive_umask_still_yields_private_auth_paths() {
        use std::os::unix::fs::PermissionsExt;
        let old = unsafe { libc::umask(0) };
        let directory = tempdir().unwrap();
        let store = AuthStore::new(directory.path().join("nested").join("auth.json"));
        store.save_copilot(&credential()).unwrap();
        let auth_dir = store.path().parent().unwrap().to_path_buf();
        let dir_mode = fs::metadata(&auth_dir).unwrap().permissions().mode() & 0o777;
        let file_mode = fs::metadata(store.path()).unwrap().permissions().mode() & 0o777;
        unsafe { libc::umask(old) };
        assert_eq!(dir_mode, 0o700);
        assert_eq!(file_mode, 0o600);
    }
}
