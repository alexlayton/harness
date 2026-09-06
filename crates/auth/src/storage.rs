//! Private, provider-keyed credential storage.
//!
//! The normal Harness TOML configuration and session files deliberately do
//! not contain credentials.  This module owns `auth.json` and performs
//! read-modify-write updates so an OAuth flow cannot discard credentials for a
//! different provider.

use crate::error::{AuthError, Result, io_error};
use fs2::FileExt;
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
                ensure_private_file(&self.path)?;
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

struct AuthFileLock {
    file: fs::File,
}

impl AuthFileLock {
    fn lock_path(auth_path: &Path) -> PathBuf {
        auth_path.with_extension("json.lock")
    }

    fn acquire(auth_path: &Path) -> Result<Self> {
        let path = Self::lock_path(auth_path);
        let file = {
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .mode(0o600)
                    .open(&path)
            }
            #[cfg(not(unix))]
            {
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&path)
            }
        }
        .map_err(|source| io_error("open auth lock", &path, source))?;
        ensure_private_file(&path)?;

        for _ in 0..LOCK_ATTEMPTS {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { file }),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(LOCK_WAIT);
                }
                Err(source) => return Err(io_error("lock auth file", &path, source)),
            }
        }
        Err(AuthError::LockUnavailable(path))
    }
}

impl Drop for AuthFileLock {
    fn drop(&mut self) {
        // Retain the sidecar inode so a waiter cannot switch to a replacement
        // pathname between unlock and its next acquisition.
        let _ = self.file.unlock();
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
        let entries = store.load_unlocked().unwrap();
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
    fn advisory_auth_lock_serializes_contenders_and_retains_sidecar_inode() {
        let directory = tempdir().unwrap();
        let victim = directory.path().join("auth.json");
        fs::write(&victim, "{}").unwrap();
        let lock_path = AuthFileLock::lock_path(&victim);

        let owner = AuthFileLock::acquire(&victim).unwrap();
        assert!(lock_path.exists());
        let contender = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert_eq!(
            contender.try_lock_exclusive().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
        drop(owner);

        let next = AuthFileLock::acquire(&victim).unwrap();
        assert!(lock_path.exists());
        drop(next);
        assert!(lock_path.exists());
    }

    #[test]
    fn concurrent_auth_lock_contenders_never_overlap() {
        let directory = tempdir().unwrap();
        let victim = directory.path().join("auth.json");
        fs::write(&victim, "{}").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let active = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let victim = victim.clone();
            let barrier = barrier.clone();
            let active = active.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                let lock = AuthFileLock::acquire(&victim).unwrap();
                assert_eq!(active.fetch_add(1, std::sync::atomic::Ordering::SeqCst), 0);
                std::thread::sleep(Duration::from_millis(10));
                active.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                drop(lock);
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
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
