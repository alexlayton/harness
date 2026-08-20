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
    /// login processes from losing one another's credentials.
    pub fn save_provider_value(&self, provider: &str, value: serde_json::Value) -> Result<()> {
        if provider.trim().is_empty() {
            return Err(AuthError::InvalidCredential(
                "provider name cannot be empty".into(),
            ));
        }
        let parent = self.parent_dir();
        fs::create_dir_all(&parent)
            .map_err(|source| io_error("create auth directory", &parent, source))?;
        set_private_directory(&parent);
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
        fs::create_dir_all(&parent)
            .map_err(|source| io_error("create auth directory", &parent, source))?;
        set_private_directory(&parent);
        let contents = serde_json::to_string_pretty(entries).map_err(|source| AuthError::Json {
            path: self.path.clone(),
            source,
        })?;
        let temp = temporary_path(&self.path);
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
                .map_err(|source| io_error("create temporary auth file", &temp, source))?;
            set_private_file(&temp);
            file.write_all(contents.as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .and_then(|_| file.flush())
                .and_then(|_| file.sync_all())
                .map_err(|source| io_error("write temporary auth file", &temp, source))?;
            drop(file);
            fs::rename(&temp, &self.path)
                .map_err(|source| io_error("replace auth file", &self.path, source))?;
            set_private_file(&self.path);
            sync_directory(&parent);
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

fn set_private_directory(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::metadata(path) {
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o700);
            let _ = fs::set_permissions(path, permissions);
        }
    }
}

fn set_private_file(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::metadata(path) {
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o600);
            let _ = fs::set_permissions(path, permissions);
        }
    }
}

fn sync_directory(path: &Path) {
    #[cfg(unix)]
    {
        if let Ok(directory) = fs::File::open(path) {
            let _ = directory.sync_all();
        }
    }
}

struct AuthFileLock {
    path: PathBuf,
}

impl AuthFileLock {
    fn acquire(auth_path: &Path) -> Result<Self> {
        let path = auth_path.with_extension("json.lock");
        for _ in 0..LOCK_ATTEMPTS {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "pid={}", std::process::id());
                    set_private_file(&path);
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = fs::metadata(&path)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                        .is_some_and(|age| age > Duration::from_secs(5 * 60));
                    if stale {
                        let _ = fs::remove_file(&path);
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
        let _ = fs::remove_file(&self.path);
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
}
