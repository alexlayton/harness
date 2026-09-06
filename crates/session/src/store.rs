use crate::codec::{TailRecovery, decode_session_file, encode_header, encode_record};
use crate::error::{Result, SessionError, io_error};
use crate::model::{
    EventId, Session, SessionEvent, SessionEventRecord, SessionId, SessionMetadata, StoredContent,
    StoredToolCall, Timestamp, now_timestamp,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Shared deferred-sync flag so the store stays [`Clone`] (session handlers
/// clone it freely). `true` skips per-record `sync_all`.
#[derive(Debug, Default)]
struct DeferredSync(AtomicBool);

impl Clone for DeferredSync {
    fn clone(&self) -> Self {
        Self(AtomicBool::new(self.0.load(Ordering::Relaxed)))
    }
}

const LOCK_WAIT: Duration = Duration::from_millis(10);
const LOCK_ATTEMPTS: usize = 200;
/// Locks older than this are stolen even when the owning PID cannot be read.
const LOCK_STALE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Options used when creating a new durable session.
#[derive(Clone, Debug, Default)]
pub struct SessionCreateOptions {
    pub title: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub parent_session: Option<SessionId>,
}

/// A compact entry used by session pickers and `/sessions` output.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionIndexEntry {
    pub id: SessionId,
    pub short_id: String,
    pub title: Option<String>,
    pub workspace_root: PathBuf,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub parent_session: Option<SessionId>,
    pub event_count: usize,
    /// Whether replay produces any provider conversation context. Metadata,
    /// usage, diagnostics, and cancellation-only files remain false.
    pub has_conversation: bool,
    pub path: PathBuf,
    pub bytes: u64,
}

/// Filesystem-backed session storage.  Sessions are grouped by a stable key
/// derived from the workspace root, so a project never appears in another
/// project's normal listing.
#[derive(Clone, Debug)]
pub struct SessionStore {
    root: PathBuf,
    workspace_root: PathBuf,
    workspace_dir: PathBuf,
    /// When set, [`Self::append_event`] skips the per-record `sync_all`:
    /// records are still written and flushed to the OS, but the durable
    /// flush is owed to [`Self::sync_session`], which callers invoke at turn
    /// boundaries. Chatty tool loops pay several fsyncs per turn otherwise.
    /// Default is per-event durability: a crash loses at most the in-flight
    /// record. Deferred mode widens that window to the current turn's tail.
    deferred_sync: DeferredSync,
}

impl SessionStore {
    /// Construct a store with an explicit root.  For an existing store the
    /// root is only read here, keeping read-only listing useful for a missing
    /// state directory.  A brand-new store performs its first write eagerly:
    /// it persists a random `.salt` so workspace-key hashes cannot be
    /// pre-computed, while stores that already contain sessions keep the
    /// legacy unsalted layout and stay fully backward compatible.
    pub fn new(root: impl Into<PathBuf>, workspace_root: impl Into<PathBuf>) -> Result<Self> {
        let root = absolute_lexical(root.into())?;
        let workspace_root = normalize_workspace(workspace_root.into())?;
        let salt = resolve_salt(&root)?;
        let workspace_dir = root.join(workspace_key(&workspace_root, salt));
        Ok(Self {
            root,
            workspace_root,
            workspace_dir,
            deferred_sync: DeferredSync::default(),
        })
    }

    /// Toggle deferred durability (see the field docs). Returns the store for
    /// chaining.
    pub fn with_deferred_sync(self, deferred: bool) -> Self {
        self.deferred_sync.0.store(deferred, Ordering::Relaxed);
        self
    }

    /// Whether deferred durability is enabled.
    pub fn deferred_sync(&self) -> bool {
        self.deferred_sync.0.load(Ordering::Relaxed)
    }

    /// Durable flush for deferred-sync sessions: `fsync`s the session file so
    /// every record appended so far survives power loss. Cheap to call
    /// repeatedly; a no-op when the session has no file (memory-only).
    pub fn sync_session(&self, session: &Session) -> Result<()> {
        let Some(path) = session.path() else {
            return Ok(());
        };
        self.ensure_path_in_root(path)?;
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|source| io_error("open session for sync", path, source))?;
        file.sync_all()
            .map_err(|source| io_error("sync session", path, source))
    }

    /// Construct the default Harness store.  `HARNESS_SESSION_DIR` is an
    /// exact directory override; otherwise `HARNESS_STATE_DIR` is treated as
    /// the parent of `sessions`; the default is `~/.harness/sessions`.
    pub fn default_for_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::new(default_session_dir(), workspace_root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }

    pub fn is_path_in_store(&self, path: &Path) -> bool {
        self.ensure_path_in_root(path).is_ok()
    }

    /// Create and immediately persist a new session header.
    pub fn create(&self, options: SessionCreateOptions) -> Result<Session> {
        let mut metadata =
            SessionMetadata::new(self.workspace_root.clone(), options.provider, options.model);
        metadata.title = options.title;
        metadata.parent_session = options.parent_session;
        self.create_from_metadata(metadata)
    }

    /// Create a session from already prepared metadata.  The metadata ID is
    /// retained when it is not already used, which is useful for importers.
    pub fn create_with_metadata(&self, metadata: SessionMetadata) -> Result<Session> {
        if normalize_workspace(metadata.workspace_root.clone())? != self.workspace_root {
            return Err(SessionError::WorkspaceMismatch {
                stored: metadata.workspace_root,
                requested: self.workspace_root.clone(),
            });
        }
        self.create_from_metadata(metadata)
    }

    /// Persist a new session header on disk and return the in-memory session.
    /// Directories and the session file are created private (`0o700`/`0o600`
    /// on Unix) at creation time, and permission/parent-sync failures fail
    /// closed instead of leaving permissive or undiscoverable files.
    fn create_from_metadata(&self, metadata: SessionMetadata) -> Result<Session> {
        private_dir_all(&self.workspace_dir)?;
        self.ensure_path_in_root(&self.workspace_dir)?;
        // Repair pre-existing roots; failures fail closed.
        ensure_private_directory(&self.root)?;
        ensure_private_directory(&self.workspace_dir)?;
        let path = self.workspace_dir.join(format!("{}.jsonl", metadata.id));
        let mut file = private_file(&path)?;
        let header = encode_header(&metadata)?;
        let write_result = file
            .write_all(header.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush())
            .and_then(|_| file.sync_all())
            .map_err(|source| io_error("write session header", &path, source));
        if let Err(error) = write_result {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(error);
        }
        drop(file);
        ensure_private_file(&path)?;
        sync_parent(&path)?;
        Ok(Session {
            header_metadata: metadata.clone(),
            metadata,
            events: Vec::new(),
            path: Some(path),
        })
    }

    /// Append one event and make it durable before returning (or defer the
    /// `sync_all` when deferred sync is enabled — see the store field docs).
    /// A sidecar create-new lock prevents two Harness processes from
    /// interleaving JSON records.  Every record is followed by a newline.
    ///
    /// When the file ends with an incomplete crash tail (an unterminated
    /// malformed final line), the tail is truncated to the last valid record
    /// under the lock before appending, so the next write cannot cement the
    /// fragment into a terminated corrupt line.  Loading alone never repairs
    /// the file; only this locked append path truncates, and only the
    /// incomplete tail fragment.  Terminated malformed lines and mid-file
    /// corruption remain hard errors.
    #[tracing::instrument(
        name = "session_persist",
        skip_all,
        fields(session_id = %session.id())
    )]
    pub fn append_event(
        &self,
        session: &mut Session,
        event: SessionEvent,
    ) -> Result<SessionEventRecord> {
        let Some(path) = session.path().cloned() else {
            return Ok(session.append(event));
        };
        self.ensure_path_in_root(&path)?;
        let lock = SessionLock::acquire(&path)?;
        // Re-read under the lock.  Two processes may each hold an older
        // in-memory Session; deriving the sequence from disk prevents
        // duplicate sequence numbers and keeps append-only ordering valid.
        let (mut disk_session, recovery) = load_session_file_for_append(&path)?;
        if disk_session.id() != session.id() {
            return Err(SessionError::InvalidEvent(
                "session object does not match its file".into(),
            ));
        }
        if recovery.recovered {
            // Repair the incomplete crash tail before appending: truncate to
            // the last valid record so the fragment never becomes a
            // terminated corrupt line.  This keeps the session append-only
            // with respect to every previously valid record.
            truncate_to_valid_tail(&path, &recovery)?;
        }
        let record = SessionEventRecord {
            id: EventId::new(),
            sequence: disk_session
                .events
                .last()
                .map_or(1, |entry| entry.sequence + 1),
            timestamp: now_timestamp(),
            event,
        };
        let mut candidate_events = disk_session.events.clone();
        candidate_events.push(record.clone());
        crate::model::validate_events(&candidate_events)?;
        let line = encode_record(disk_session.id(), &record)?;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|source| io_error("open session for append", &path, source))?;
        let write_result = file
            .write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush());
        let write_result = if write_result.is_err() || !self.deferred_sync() {
            write_result.and_then(|_| file.sync_all())
        } else {
            write_result
        };
        write_result.map_err(|source| io_error("append session event", &path, source))?;
        drop(file);
        disk_session.append_record(record.clone());
        *session = disk_session;
        drop(lock);
        Ok(record)
    }

    pub fn open(&self, id: &SessionId) -> Result<Session> {
        let path = self.workspace_dir.join(format!("{id}.jsonl"));
        if !path.exists() {
            return Err(SessionError::NotFound(id.to_string()));
        }
        self.load_path(&path)
    }

    /// Load by exact ID, unique ID prefix, `latest`, or a path.  Paths are
    /// explicit user input; loading does not write outside this store, and a
    /// loaded external file cannot be appended through this store.
    pub fn load(&self, selector: &str) -> Result<Session> {
        let selector = selector.trim();
        if selector.is_empty() || selector.eq_ignore_ascii_case("latest") {
            let mut entries = self.list()?;
            entries.sort_by(|left, right| {
                right
                    .updated_at
                    .cmp(&left.updated_at)
                    .then_with(|| right.created_at.cmp(&left.created_at))
            });
            // Harness creates a persisted header immediately. Skip every
            // session without provider conversation context so repeated
            // `/new` commands cannot make `latest` select a placeholder.
            return entries
                .iter()
                .find(|entry| entry.has_conversation)
                .map(|entry| self.load_path(&entry.path))
                .unwrap_or(Err(SessionError::NoSession));
        }

        let path = PathBuf::from(selector);
        if path.exists() {
            return self.load_path(&path);
        }
        if let Ok(id) = SessionId::parse(selector) {
            return self.open(&id);
        }
        let mut matches = self
            .list()?
            .into_iter()
            .filter(|entry| {
                entry.id.to_string().starts_with(selector)
                    || entry.short_id.eq_ignore_ascii_case(selector)
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        match matches.as_slice() {
            [entry] => self.load_path(&entry.path),
            [] => Err(SessionError::NotFound(selector.to_owned())),
            _ => Err(SessionError::InvalidSessionId(format!(
                "session selector `{selector}` is ambiguous"
            ))),
        }
    }

    pub fn load_path(&self, path: &Path) -> Result<Session> {
        let resolved = path
            .canonicalize()
            .map_err(|source| io_error("resolve session path", path, source))?;
        let mut session = load_session_file(&resolved)?;
        let stored_workspace = normalize_workspace(session.metadata.workspace_root.clone())?;
        if stored_workspace != self.workspace_root {
            return Err(SessionError::WorkspaceMismatch {
                stored: session.metadata.workspace_root.clone(),
                requested: self.workspace_root.clone(),
            });
        }
        session.path = Some(resolved);
        Ok(session)
    }

    /// Load without applying this store's workspace filter.  This is used by
    /// export/import tooling and is intentionally read-only.
    pub fn load_any_path(path: &Path) -> Result<Session> {
        let resolved = path
            .canonicalize()
            .map_err(|source| io_error("resolve session path", path, source))?;
        let mut session = load_session_file(&resolved)?;
        session.path = Some(resolved);
        Ok(session)
    }

    pub fn list(&self) -> Result<Vec<SessionIndexEntry>> {
        list_directory(&self.workspace_dir, Some(&self.workspace_root))
    }

    /// Repair tool calls left at the end of a file by a process crash.  The
    /// synthetic error results are durable, so subsequent user messages pass
    /// strict ordering validation and providers receive a valid history.
    ///
    /// The repair marker is itself an ordinary `TurnCancelled` event: repair
    /// appends synthetic `ToolResult` events before the marker, so every
    /// repaired call is already complete when the marker lands.
    pub fn repair_incomplete_tool_calls(&self, session: &mut Session) -> Result<usize> {
        let mut pending = Vec::<StoredToolCall>::new();
        for record in &session.events {
            match &record.event {
                SessionEvent::AssistantMessage { message } => {
                    pending.extend(message.content.iter().filter_map(|content| {
                        let StoredContent::ToolCall {
                            id,
                            name,
                            arguments,
                        } = content
                        else {
                            return None;
                        };
                        Some(StoredToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        })
                    }));
                }
                SessionEvent::ToolCall { call } => pending.push(call.clone()),
                SessionEvent::ToolResult { tool_call_id, .. } => {
                    if let Some(index) = pending.iter().position(|call| call.id == *tool_call_id) {
                        pending.remove(index);
                    }
                }
                SessionEvent::TurnCancelled { .. } => pending.clear(),
                _ => {}
            }
        }
        if pending.is_empty() {
            return Ok(0);
        }
        let count = pending.len();
        for call in pending {
            self.append_event(
                session,
                SessionEvent::ToolResult {
                    tool_call_id: call.id,
                    content: "[session recovered: tool result was interrupted]".into(),
                    is_error: true,
                    tool_name: Some(call.name),
                },
            )?;
        }
        self.append_event(
            session,
            SessionEvent::TurnCancelled {
                reason: "recovered interrupted tool turn".into(),
            },
        )?;
        Ok(count)
    }

    /// Adopt an explicitly loaded external file into this workspace while
    /// retaining its session ID.  This is used when `/load <path>` points at a
    /// JSONL export in the current directory; future appends must remain under
    /// the configured store root.
    pub fn adopt(&self, source: &Session) -> Result<Session> {
        if source
            .file_path()
            .is_some_and(|path| self.is_path_in_store(path))
        {
            return Ok(source.clone());
        }
        let target = self.workspace_dir.join(format!("{}.jsonl", source.id()));
        if target.exists() {
            return self.open(&source.id());
        }
        let mut adopted = self.create_with_metadata(source.header_metadata.clone())?;
        for record in &source.events {
            self.append_event(&mut adopted, record.event.clone())?;
        }
        Ok(adopted)
    }

    fn ensure_path_in_root(&self, path: &Path) -> Result<()> {
        let canonical_root = self
            .root
            .canonicalize()
            .unwrap_or_else(|_| self.root.clone());
        let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if !canonical_path.starts_with(&canonical_root) {
            return Err(SessionError::PathOutsideStore {
                path: path.to_path_buf(),
                root: self.root.clone(),
            });
        }
        Ok(())
    }
}

fn load_session_file(path: &Path) -> Result<Session> {
    let contents = read_file(path)?;
    let (mut session, _) = decode_session_file(&contents, path)?;
    session.path = Some(path.to_path_buf());
    Ok(session)
}

/// Re-read a session file for appending: returns the decoded session plus
/// the crash-tail recovery metadata.  The file is never mutated here; the
/// locked `append_event` path truncates only after this reports an
/// incomplete tail.
fn load_session_file_for_append(path: &Path) -> Result<(Session, TailRecovery)> {
    let contents = read_file(path)?;
    let (mut session, recovery) = decode_session_file(&contents, path)?;
    session.path = Some(path.to_path_buf());
    Ok((session, recovery))
}

/// Truncate an incomplete crash tail to the last valid record, then flush
/// and sync before the next append.  Only the malformed trailing fragment
/// is removed; valid records are never rewritten.
fn truncate_to_valid_tail(path: &Path, recovery: &TailRecovery) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|source| io_error("open session for repair", path, source))?;
    file.set_len(recovery.valid_bytes as u64)
        .map_err(|source| io_error("truncate incomplete session tail", path, source))?;
    file.sync_all()
        .map_err(|source| io_error("sync repaired session", path, source))
}

fn read_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| io_error("open session", path, source))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|source| io_error("read session", path, source))?;
    Ok(contents)
}

#[derive(Deserialize)]
struct IndexEnvelope {
    version: Option<u32>,
    #[serde(rename = "type")]
    kind: Option<String>,
    session_id: Option<String>,
    event_id: Option<String>,
    sequence: Option<u64>,
    timestamp: Option<String>,
    #[serde(default)]
    data: Value,
}

/// Read only the metadata needed by the session picker. This deliberately
/// avoids rebuilding provider messages or cloning large tool results.
fn index_file(path: &Path, workspace: Option<&Path>) -> Result<Option<SessionIndexEntry>> {
    let file = File::open(path).map_err(|source| io_error("open session index", path, source))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .map_err(|source| io_error("read session header", path, source))?
        == 0
    {
        return Ok(None);
    }
    let header: IndexEnvelope = match serde_json::from_str(&line) {
        Ok(header) => header,
        Err(_) => return Ok(None),
    };
    if header.version.is_none()
        || header.version.unwrap_or_default() > crate::model::FORMAT_VERSION
        || header.kind.as_deref() != Some("session")
    {
        return Ok(None);
    }
    let id = match (
        header
            .session_id
            .as_deref()
            .and_then(|value| SessionId::parse(value).ok()),
        serde_json::from_value::<SessionMetadata>(header.data),
    ) {
        (Some(id), Ok(metadata)) if id == metadata.id => (id, metadata),
        _ => return Ok(None),
    };
    let (id, metadata) = id;
    if let Some(workspace) = workspace
        && normalize_workspace(metadata.workspace_root.clone())? != workspace
    {
        return Ok(None);
    }

    let mut title = metadata.title.clone();
    let mut provider = metadata.provider.clone();
    let mut model = metadata.model.clone();
    let mut updated_at = metadata.updated_at.clone();
    let mut event_count = 0usize;
    let mut has_conversation = false;
    let mut expected_sequence = 1u64;
    let mut event_ids = HashSet::new();
    line.clear();
    loop {
        let read = reader
            .read_line(&mut line)
            .map_err(|source| io_error("read session index", path, source))?;
        if read == 0 {
            break;
        }
        let raw: IndexEnvelope = match serde_json::from_str(&line) {
            Ok(raw) => raw,
            Err(_) => return Ok(None),
        };
        if raw.version.is_none()
            || raw.version.unwrap_or_default() > crate::model::FORMAT_VERSION
            || raw.kind.as_deref() == Some("session")
            || raw.kind.is_none()
            || raw
                .session_id
                .as_deref()
                .and_then(|value| SessionId::parse(value).ok())
                != Some(id)
            || raw.sequence != Some(expected_sequence)
            || raw.timestamp.as_deref().is_none_or(str::is_empty)
            || raw
                .event_id
                .as_deref()
                .and_then(|value| EventId::parse(value).ok())
                .is_none()
            || !event_ids.insert(raw.event_id.clone().unwrap_or_default())
        {
            return Ok(None);
        }
        let kind = raw.kind.as_deref().unwrap_or_default();
        let timestamp = raw.timestamp.unwrap_or_default();
        updated_at = timestamp;
        event_count = event_count.saturating_add(1);
        expected_sequence = expected_sequence.saturating_add(1);
        match kind {
            "user_message" | "assistant_message" => {
                has_conversation |= raw
                    .data
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|content| !content.is_empty());
            }
            "reasoning" | "tool_call" | "tool_result" | "compaction" => {
                has_conversation = true;
            }
            "model_change" => {
                if let (Some(next_provider), Some(next_model)) = (
                    raw.data.get("provider").and_then(Value::as_str),
                    raw.data.get("model").and_then(Value::as_str),
                ) {
                    provider = Some(next_provider.to_owned());
                    model = Some(next_model.to_owned());
                }
            }
            "metadata_change" => {
                title = raw
                    .data
                    .get("title")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            _ => {}
        }
        line.clear();
    }
    let bytes = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or_default();
    Ok(Some(SessionIndexEntry {
        id,
        short_id: id.short(),
        title,
        workspace_root: metadata.workspace_root,
        created_at: metadata.created_at,
        updated_at,
        provider,
        model,
        parent_session: metadata.parent_session,
        event_count,
        has_conversation,
        path: path.to_path_buf(),
        bytes,
    }))
}

fn list_directory(directory: &Path, workspace: Option<&Path>) -> Result<Vec<SessionIndexEntry>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    let entries =
        fs::read_dir(directory).map_err(|source| io_error("list sessions", directory, source))?;
    for entry in entries {
        let entry = entry.map_err(|source| io_error("read session entry", directory, source))?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
            continue;
        }
        if let Some(entry) = index_file(&path, workspace)? {
            result.push(entry);
        }
    }
    result.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(result)
}

/// Resolve the default session directory without creating it.
pub fn default_session_dir() -> PathBuf {
    if let Some(path) = non_empty_env_path("HARNESS_SESSION_DIR") {
        return path;
    }
    if let Some(path) = non_empty_env_path("HARNESS_STATE_DIR") {
        return path.join("sessions");
    }
    dirs_like_home()
        .map(|home| home.join(".harness").join("sessions"))
        .unwrap_or_else(|| PathBuf::from(".harness").join("sessions"))
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn dirs_like_home() -> Option<PathBuf> {
    non_empty_env_path("HOME")
        .or_else(|| non_empty_env_path("USERPROFILE"))
        .or_else(|| std::env::current_dir().ok())
}

fn normalize_workspace(path: PathBuf) -> Result<PathBuf> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|source| io_error("resolve workspace", path, source));
    }
    absolute_lexical(path)
}

fn absolute_lexical(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|source| io_error("resolve path", ".", source))
    }
}

/// File holding the per-store workspace-key salt.  Its presence distinguishes
/// a salted store from a legacy (pre-salt) store.
const SALT_FILE: &str = ".salt";

/// Resolve the store's workspace-key salt: the persisted value when present,
/// `0` for a legacy store that predates salting, or a freshly generated and
/// persisted random salt for a brand-new store.  `create_new` makes the first
/// writer win; a racing process re-reads the winner's value, so every process
/// resolves the same workspace directory.
fn resolve_salt(root: &Path) -> Result<u64> {
    if let Some(salt) = read_salt(root) {
        return Ok(salt);
    }
    if root.join(SALT_FILE).exists() {
        // The salt file exists but cannot be parsed.  Stay on the stable
        // legacy layout rather than churning keys on every process start.
        return Ok(0);
    }
    // A store that already contains session directories but no `.salt` was
    // created before salting.  Keep the unsalted keys so existing sessions
    // remain discoverable; this store never takes a salt.
    if root.exists() && store_has_legacy_directories(root) {
        return Ok(0);
    }
    // Brand-new store: create a random salt and persist it.  This is the
    // store's first write; the root itself is created here if needed.
    let salt = Uuid::new_v4().as_u128() as u64;
    private_dir_all(root)?;
    sync_parent(root)?;
    persist_salt(root, salt)
}

fn persist_salt(root: &Path, salt: u64) -> Result<u64> {
    let salt_path = root.join(SALT_FILE);
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&salt_path)
    {
        Ok(mut file) => {
            writeln!(file, "{salt}")
                .and_then(|_| file.sync_all())
                .map_err(|source| io_error("write store salt", &salt_path, source))?;
            drop(file);
            ensure_private_file(&salt_path)?;
            sync_parent(&salt_path)?;
            Ok(salt)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // Another process won the race.  Adopt its salt (retrying briefly
            // in case it has not finished writing) so both processes resolve
            // the same workspace directory.
            for _ in 0..20 {
                if let Some(winner) = read_salt(root) {
                    return Ok(winner);
                }
                thread::sleep(Duration::from_millis(1));
            }
            Ok(0)
        }
        Err(source) => Err(io_error("create store salt", &salt_path, source)),
    }
}

fn read_salt(root: &Path) -> Option<u64> {
    fs::read_to_string(root.join(SALT_FILE))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// True when `root` already holds session directories from a pre-salt store.
fn store_has_legacy_directories(root: &Path) -> bool {
    fs::read_dir(root).ok().is_some_and(|entries| {
        entries.filter_map(|entry| entry.ok()).any(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_dir())
                && !entry.file_name().to_string_lossy().starts_with('.')
        })
    })
}

fn workspace_key(path: &Path, salt: u64) -> String {
    // FNV-1a is small, deterministic across processes/platforms, and only
    // used as a directory disambiguator (the full workspace path is still
    // validated from session metadata).  The per-store salt (when present)
    // prevents an attacker from pre-computing collision keys for a shared
    // store; salt 0 reproduces the legacy unsalted layout exactly.
    let mut hash = 0xcbf29ce484222325u64 ^ salt;
    for byte in path.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let readable = path
        .to_string_lossy()
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
                byte as char
            } else {
                '_'
            }
        })
        .take(80)
        .collect::<String>();
    format!("{}-{hash:016x}", readable.trim_matches('_'))
}

/// Files and directories are created private at creation time (see
/// [`private_dir_all`] and [`private_file`]); existing paths are repaired by
/// [`ensure_private_directory`] and [`ensure_private_file`], whose failures
/// are propagated instead of being silently ignored.
///
/// Apply private (owner-only) permissions to a store directory.
///
/// This is a Unix-only hardening step.  On Windows, session files inherit
/// the permissions of the parent directory (typically the user's state
/// directory), which is consistent with the plan; the Windows ACL model
/// makes a portable equivalent out of scope here.
///
/// Fail-closed permission repair: permission or sync
/// failures are returned instead of silently continuing with permissive
/// files or unsynced parents.
fn ensure_private_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = fs::metadata(path)
            .map_err(|source| io_error("stat private directory", path, source))?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions)
            .map_err(|source| io_error("secure directory permissions", path, source))?;
    }
    sync_parent(path)?;
    Ok(())
    // #[cfg(not(unix))] — metadata repair is a no-op on Windows and other
    // platforms, but the parent sync above still runs (see above).
}

/// Fail-closed permission repair for files.
fn ensure_private_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata =
            fs::metadata(path).map_err(|source| io_error("stat private file", path, source))?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(path, permissions)
            .map_err(|source| io_error("secure file permissions", path, source))?;
    }
    sync_parent(path)?;
    Ok(())
}

/// Create a directory and every missing parent with `0o700` at creation time
/// (Unix), instead of writing with the process umask and securing later.
/// Parent-directory sync failures are propagated so a crash cannot leave a
/// durable file undiscoverable.
fn private_dir_all(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(path)
            .map_err(|source| io_error("create private directory", path, source))?;
        ensure_private_directory(path)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        fs::create_dir_all(path)
            .map_err(|source| io_error("create session directory", path, source))?;
        sync_parent(path)?;
        Ok(())
    }
}

/// Open a new file with `0o600` at creation time (Unix) so sensitive bytes
/// are never written first and secured later.
fn private_file(path: &Path) -> Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    SessionError::AlreadyExists(path.to_path_buf())
                } else {
                    io_error("create private file", path, source)
                }
            })
    }
    #[cfg(not(unix))]
    {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    SessionError::AlreadyExists(path.to_path_buf())
                } else {
                    io_error("create private file", path, source)
                }
            })
    }
}

/// Sync the parent directory after durable create/rename operations so the
/// new entry survives power loss; failures fail closed.  On platforms
/// without directory fsync support this is a no-op success.
fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        // A missing parent (relative paths in tests) has nothing to sync.
        let directory = match File::open(parent) {
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

struct SessionLock {
    path: PathBuf,
    /// Unguessable owner nonce written into the lock file.  Release and
    /// stealing compare against this exact instance so one owner can never
    /// remove another owner's replacement lock.
    nonce: String,
}

impl SessionLock {
    fn lock_path(session_path: &Path) -> PathBuf {
        session_path.with_extension("jsonl.lock")
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

    fn acquire(session_path: &Path) -> Result<Self> {
        let path = Self::lock_path(session_path);
        for _ in 0..LOCK_ATTEMPTS {
            // A fresh nonce per attempt keeps every contender's claim unique:
            // stealing removes only the exact stale instance observed, and
            // release removes only the owner's own instance.
            let nonce = Uuid::new_v4().to_string();
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "pid={}", std::process::id());
                    let _ = writeln!(file, "nonce={nonce}");
                    let _ = file.sync_all();
                    return Ok(Self { path, nonce });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_is_stale(&path) {
                        // Steal only the exact stale instance observed: if the
                        // contender's `create_new` lost the race to a live
                        // replacement owner, the nonce no longer matches and
                        // the replacement lock is left alone.  Legacy locks
                        // without a nonce carry no instance identity; they are
                        // stolen by age so dead owners (like the
                        // `pid=4000000` fixture) can still be recovered.
                        let observed = Self::read_nonce(&path);
                        let same_instance = Self::read_nonce(&path) == observed;
                        if lock_is_stale(&path)
                            && same_instance
                            && (observed.is_some() || stale_without_identity(&path))
                        {
                            let _ = fs::remove_file(&path);
                        } else {
                            thread::sleep(LOCK_WAIT);
                        }
                    } else {
                        thread::sleep(LOCK_WAIT);
                    }
                }
                Err(source) => return Err(io_error("create session lock", &path, source)),
            }
        }
        Err(SessionError::LockUnavailable(path))
    }
}

/// A lock is stale when its recorded owner is no longer alive (checked via
/// `kill(pid, 0)` on Unix and `OpenProcess` on Windows), or — for lock files
/// whose PID cannot be read (legacy files, unreadable, malformed) — when it is
/// older than the conservative timeout.
fn lock_is_stale(path: &Path) -> bool {
    // A live owner means the lock is never stale, even past the timeout:
    // a long append must not be interrupted by another process.
    if let Some(pid) = lock_owner_pid(path) {
        return !pid_is_alive(pid);
    }
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age > LOCK_STALE_TIMEOUT)
}

fn lock_owner_pid(path: &Path) -> Option<u32> {
    let value = fs::read_to_string(path).ok()?;
    value
        .lines()
        .find_map(|line| line.strip_prefix("pid=")?.trim().parse::<u32>().ok())
}

/// Legacy locks without a nonce carry no instance identity: they are stale
/// only via the conservative age heuristic (or a provably dead PID), never
/// merely because a PID line is present.  A lock with a live owner is never
/// stolen, even past the timeout.
fn stale_without_identity(path: &Path) -> bool {
    if SessionLock::read_nonce(path).is_some() {
        return false;
    }
    match lock_owner_pid(path) {
        Some(pid) => !pid_is_alive(pid),
        None => lock_is_stale(path),
    }
}

/// Returns true when the process with `pid` is alive.  On platforms without a
/// process-existence probe this reports false so the age heuristic applies.
fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // kill(pid, 0) delivers no signal; it only probes existence.  EPERM
        // means the process exists but is owned by another user.
        // SAFETY: the signal number is 0, so no signal is sent; the PID comes
        // from this store's own lock file.
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        // SAFETY: OpenProcess only queries; the returned handle is closed
        // immediately without touching any process state.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return false;
        }
        unsafe { CloseHandle(handle) };
        true
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        // Release only our own instance: if a successor already replaced
        // this lock, its nonce differs and the file is left alone.
        if Self::is_same_lock(&self.path, &self.nonce) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SessionEvent, StoredMessage};
    use llm::Message;
    use tempfile::tempdir;

    #[test]
    fn create_append_load_and_list_round_trip() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store
            .create(SessionCreateOptions {
                provider: Some("mock".into()),
                model: Some("demo".into()),
                ..SessionCreateOptions::default()
            })
            .unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("hello")),
                },
            )
            .unwrap();
        let loaded = store.open(&session.id()).unwrap();
        assert_eq!(loaded.context_messages(), session.context_messages());

        // Metadata-only sessions remain durable but do not count as
        // conversational choices or win the `latest` selector.
        let mut metadata_only = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut metadata_only,
                SessionEvent::ModelChange {
                    provider: "mock".into(),
                    model: "other".into(),
                },
            )
            .unwrap();
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(
            entries
                .iter()
                .find(|entry| entry.id == session.id())
                .unwrap()
                .has_conversation
        );
        assert!(
            !entries
                .iter()
                .find(|entry| entry.id == metadata_only.id())
                .unwrap()
                .has_conversation
        );
        assert_eq!(store.load("latest").unwrap().id(), session.id());
    }

    #[test]
    fn interrupted_tool_calls_are_repaired_before_continuation() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("read this")),
                },
            )
            .unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::ToolCall {
                    call: StoredToolCall {
                        id: "call-1".into(),
                        name: "read".into(),
                        arguments: serde_json::json!({"path": "file"}),
                    },
                },
            )
            .unwrap();
        let mut loaded = store.open(&session.id()).unwrap();
        assert_eq!(store.repair_incomplete_tool_calls(&mut loaded).unwrap(), 1);
        store
            .append_event(
                &mut loaded,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("continue")),
                },
            )
            .unwrap();
        assert!(
            loaded
                .events
                .iter()
                .any(|record| matches!(record.event, SessionEvent::TurnCancelled { .. }))
        );
    }

    #[test]
    fn append_repairs_an_unterminated_crash_tail() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("hello")),
                },
            )
            .unwrap();
        let valid_events = session.events.clone();
        // Simulate a crash mid-append: a partial JSON fragment with no
        // trailing newline.
        let path = session.path().unwrap().clone();
        use std::io::Write as _;
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"version\":1,\"type\":\"user_message\"")
            .unwrap();
        // Loading alone does not mutate the file.
        let loaded = store.open(&session.id()).unwrap();
        assert_eq!(loaded.events, valid_events);
        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.ends_with("user_message\""));
        // The next locked append truncates only the fragment, then writes.
        let mut repaired = loaded;
        store
            .append_event(
                &mut repaired,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("after crash")),
                },
            )
            .unwrap();
        let mut expected = valid_events;
        expected.push(repaired.events.last().unwrap().clone());
        assert_eq!(repaired.events, expected);
        let reopened = store.open(&session.id()).unwrap();
        assert_eq!(reopened.events, expected);
    }

    #[test]
    fn workspace_scoping_rejects_other_project() {
        let root = tempdir().unwrap();
        let first = tempdir().unwrap();
        let second = tempdir().unwrap();
        let first_store = SessionStore::new(root.path(), first.path()).unwrap();
        let session = first_store.create(SessionCreateOptions::default()).unwrap();
        let second_store = SessionStore::new(root.path(), second.path()).unwrap();
        let error = second_store.load_path(session.path().unwrap()).unwrap_err();
        assert!(matches!(error, SessionError::WorkspaceMismatch { .. }));
    }

    #[test]
    fn lock_file_owned_by_dead_pid_is_stale_immediately() {
        let directory = tempdir().unwrap();
        let lock = directory.path().join("session.jsonl.lock");
        fs::write(&lock, "pid=4000000\n").unwrap();
        assert!(lock_is_stale(&lock));
    }

    #[test]
    fn lock_file_owned_by_live_pid_is_not_stale() {
        let directory = tempdir().unwrap();
        let lock = directory.path().join("session.jsonl.lock");
        fs::write(&lock, format!("pid={}\n", std::process::id())).unwrap();
        assert!(!lock_is_stale(&lock));
    }

    #[test]
    fn lock_file_without_pid_uses_age_heuristic() {
        let directory = tempdir().unwrap();
        let lock = directory.path().join("session.jsonl.lock");
        // A legacy lock (no PID line) is fresh, so it must not be stolen yet.
        fs::write(&lock, "legacy lock without pid\n").unwrap();
        assert!(!lock_is_stale(&lock));
    }

    #[test]
    fn acquire_steals_lock_left_by_dead_process() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        let lock_path = session.path().unwrap().with_extension("jsonl.lock");
        // Simulate a harness process killed by SIGKILL mid-append: the lock
        // file persists with its owner's PID.
        fs::write(&lock_path, "pid=4000000\n").unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("hello")),
                },
            )
            .unwrap();
        assert!(
            !lock_path.exists(),
            "stale lock must be stolen and released"
        );
    }

    #[test]
    fn stale_lock_steal_is_limited_to_the_observed_instance() {
        // Two contenders racing to steal the same stale lock must not both
        // enter: stealing removes only the exact stale instance observed,
        // and release removes only the owner's own nonce.
        let directory = tempdir().unwrap();
        let victim = directory.path().join("victim.jsonl");
        fs::write(&victim, "").unwrap();
        let lock_path = SessionLock::lock_path(&victim);
        fs::write(&lock_path, "pid=4000000\nnonce=stale-a\n").unwrap();

        // Contender B observes the stale instance and replaces it first.
        assert!(lock_is_stale(&lock_path));
        let observed_by_a = SessionLock::read_nonce(&lock_path);
        assert_eq!(observed_by_a.as_deref(), Some("stale-a"));
        fs::write(
            &lock_path,
            format!("pid={}\nnonce=fresh-b\n", std::process::id()),
        )
        .unwrap();
        // Contender A re-checks before stealing: the instance changed, so
        // the fresh owner's lock is left alone.
        assert!(!lock_is_stale(&lock_path));
        assert_ne!(SessionLock::read_nonce(&lock_path), observed_by_a);

        // An old owner cannot remove a replacement owner's lock on release.
        let old_owner = SessionLock {
            path: lock_path.clone(),
            nonce: "stale-a".into(),
        };
        drop(old_owner);
        assert!(
            lock_path.exists(),
            "a stale owner must not remove the replacement lock"
        );
        let owner = SessionLock {
            path: lock_path.clone(),
            nonce: "fresh-b".into(),
        };
        drop(owner);
        assert!(!lock_path.exists());
    }

    #[test]
    fn old_lock_with_live_owner_is_never_stolen() {
        let directory = tempdir().unwrap();
        let victim = directory.path().join("victim.jsonl");
        fs::write(&victim, "").unwrap();
        let lock_path = SessionLock::lock_path(&victim);
        fs::write(
            &lock_path,
            format!("pid={}\nnonce=live-owner\n", std::process::id()),
        )
        .unwrap();
        // Even an ancient lock with a live owner is not stale.
        let aged = SystemTime::now() - Duration::from_secs(3600 * 24);
        OpenOptions::new()
            .write(true)
            .open(&lock_path)
            .unwrap()
            .set_modified(aged)
            .unwrap();
        assert!(!lock_is_stale(&lock_path));
        assert!(!stale_without_identity(&lock_path));
    }

    #[cfg(unix)]
    #[test]
    fn permissive_umask_still_yields_private_paths() {
        use std::os::unix::fs::PermissionsExt;
        // A permissive umask must not leak through: directories and files
        // are created private at creation time.
        let old = unsafe { libc::umask(0) };
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let session = store.create(SessionCreateOptions::default()).unwrap();
        let dir_mode = fs::metadata(store.workspace_dir())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let file_mode = fs::metadata(session.path().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        unsafe { libc::umask(old) };
        assert_eq!(dir_mode, 0o700);
        assert_eq!(file_mode, 0o600);
    }

    #[test]
    fn legacy_store_without_salt_keeps_unsalted_keys() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        // Simulate a pre-salt store: a workspace directory using the legacy
        // unsalted key and no `.salt` file.  The key is computed from the
        // canonical workspace path, matching `SessionStore::new`.
        let workspace = workspace.path().canonicalize().unwrap();
        let legacy_dir = root.path().join(workspace_key(&workspace, 0));
        fs::create_dir_all(&legacy_dir).unwrap();
        let store = SessionStore::new(root.path(), &workspace).unwrap();
        assert_eq!(store.workspace_dir(), legacy_dir.as_path());
        assert!(!root.path().join(SALT_FILE).exists());
    }

    #[test]
    fn fresh_store_persists_salt_and_is_stable_across_instances() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let salt_path = root.path().join(SALT_FILE);
        assert!(salt_path.exists());
        let salt = fs::read_to_string(&salt_path)
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap();
        assert_ne!(salt, 0);
        assert_ne!(
            store.workspace_dir().file_name().unwrap().to_string_lossy(),
            workspace_key(workspace.path(), 0)
        );
        // A second store instance resolves the same salted directory and can
        // see sessions created through the first.
        let second = SessionStore::new(root.path(), workspace.path()).unwrap();
        assert_eq!(second.workspace_dir(), store.workspace_dir());
        let session = store.create(SessionCreateOptions::default()).unwrap();
        assert!(
            second
                .list()
                .unwrap()
                .iter()
                .any(|entry| entry.id == session.id())
        );
    }
}
