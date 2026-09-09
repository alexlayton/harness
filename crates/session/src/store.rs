use crate::codec::{
    TailRecovery, decode_event_line, decode_session_file_bytes, encode_header, encode_record,
};
use crate::error::{Result, SessionError, io_error};
use crate::model::{
    EventId, Session, SessionEvent, SessionEventRecord, SessionId, SessionMetadata, StoredContent,
    StoredToolCall, Timestamp, now_timestamp, validate_event_suffix, validate_next_event,
};
use fs2::FileExt;
use serde::Deserialize;
use serde_json::value::RawValue;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
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

// Test-only fault injection for fail-closed hardening (SESSION-4).
//
// Running as root makes real `chmod`/`fsync` failures hard to trigger, so
// tests force them deterministically through these thread-local flags.
// Each flag makes one hardening step return the same `SessionError::Io`
// it would return for a real OS failure, proving callers propagate instead
// of silently continuing. Production builds contain no hook: every item
// here is `#[cfg(test)]` and the checks below compile out otherwise.
// Thread-local (not process-global) so parallel tests cannot interfere.
// (Plain `//` comments: `thread_local!` is a macro invocation, which
// rustdoc denies `///` docs on.)
#[cfg(test)]
thread_local! {
    static INJECT_SECURE_DIR_FAILURE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    static INJECT_SECURE_FILE_FAILURE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    static INJECT_SYNC_PARENT_FAILURE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Whether directory permission repair should fail. Test-only.
#[cfg(test)]
fn injected_secure_dir_failure() -> bool {
    INJECT_SECURE_DIR_FAILURE.with(|flag| flag.get())
}

/// Whether file permission repair should fail. Test-only.
#[cfg(test)]
fn injected_secure_file_failure() -> bool {
    INJECT_SECURE_FILE_FAILURE.with(|flag| flag.get())
}

/// Whether parent-directory sync should fail. Test-only.
#[cfg(test)]
fn injected_sync_parent_failure() -> bool {
    INJECT_SYNC_PARENT_FAILURE.with(|flag| flag.get())
}

// Whether the deferred-sync durable flush should fail. Test-only: forces
// [`SessionStore::sync_session`] to return the same `SessionError::Io` it
// would return for a real `fsync` failure, so agent turn-boundary tests
// can prove a failed flush quarantines (AGENT-1) without touching the OS.
//
// The flag itself is test-only, but the guard type is always compiled: the
// agent crate's `#[cfg(test)]` boundary tests need to name it, and
// `#[cfg(test)]` on an imported type does not propagate across crates.
// The `arm` body is test-only (production builds get a no-op guard).
// (Plain `//` comments: `thread_local!` is a macro invocation, which
// rustdoc denies `///` docs on.)
thread_local! {
    static INJECT_SYNC_SESSION_FAILURE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Whether the deferred-sync flush should fail. Test-only.
///
/// Always compiled (not `#[cfg(test)]`): the agent crate's `#[cfg(test)]`
/// boundary tests call through [`SyncSessionFaultGuard::arm`], and a
/// `#[cfg(test)]` gate here would not be active when compiling the session
/// dependency for the agent's test build. The body is test-only — in
/// production builds the flag is never armed, so this always returns false.
pub fn injected_sync_session_failure() -> bool {
    INJECT_SYNC_SESSION_FAILURE.with(|flag| flag.get())
}

/// Hold hardening-failure flags for a fail-closed test and clear them on
/// drop (including on panic) so no later test on this thread observes a
/// stale injection.
#[cfg(test)]
struct HardeningFaultGuard;

/// Hold the deferred-sync flush failure flag for an agent boundary test and
/// clear it on drop (including on panic). The agent crate cannot touch the
/// private thread-local directly, so this constructor is the seam.
pub struct SyncSessionFaultGuard;

impl SyncSessionFaultGuard {
    pub fn arm() -> Self {
        INJECT_SYNC_SESSION_FAILURE.with(|flag| flag.set(true));
        Self
    }
}

impl Drop for SyncSessionFaultGuard {
    fn drop(&mut self) {
        INJECT_SYNC_SESSION_FAILURE.with(|flag| flag.set(false));
    }
}

#[cfg(test)]
impl HardeningFaultGuard {
    fn new(secure_dir: bool, secure_file: bool, sync_parent: bool) -> Self {
        INJECT_SECURE_DIR_FAILURE.with(|flag| flag.set(secure_dir));
        INJECT_SECURE_FILE_FAILURE.with(|flag| flag.set(secure_file));
        INJECT_SYNC_PARENT_FAILURE.with(|flag| flag.set(sync_parent));
        Self
    }
}

#[cfg(test)]
impl Drop for HardeningFaultGuard {
    fn drop(&mut self) {
        INJECT_SECURE_DIR_FAILURE.with(|flag| flag.set(false));
        INJECT_SECURE_FILE_FAILURE.with(|flag| flag.set(false));
        INJECT_SYNC_PARENT_FAILURE.with(|flag| flag.set(false));
    }
}

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
        if injected_sync_session_failure() {
            return Err(io_error(
                "sync session",
                session
                    .path()
                    .map(|path| path.to_path_buf())
                    .unwrap_or_default(),
                std::io::Error::other("injected sync failure"),
            ));
        }
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
        let identity = file_identity(&path)?;
        Ok(Session {
            header_metadata: metadata.clone(),
            metadata,
            events: Vec::new(),
            path: Some(path),
            validated_bytes: header.len().saturating_add(1),
            file_identity: Some(identity),
            event_ids: std::collections::HashSet::new(),
            tracker: crate::model::ToolCallTracker::default(),
            compaction_boundary: None,
        })
    }

    /// Append one event and make it durable before returning (or defer the
    /// `sync_all` when deferred sync is enabled — see the store field docs).
    /// An advisory sidecar lock prevents two Harness processes from
    /// interleaving JSON records. The sidecar inode is retained so contenders
    /// always lock the same object. Every record is followed by a newline.
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
        let file_size = fs::metadata(&path)
            .map_err(|source| io_error("stat session for append", &path, source))?
            .len();
        let current_identity = file_identity(&path)?;
        if session.validated_bytes > 0
            && session.file_identity == Some(current_identity)
            && usize::try_from(file_size).ok() == Some(session.validated_bytes)
        {
            // The open Session still covers the complete file prefix and its
            // identity is unchanged. Validate only the new state transition.
            let record = SessionEventRecord {
                id: EventId::new(),
                sequence: session
                    .events
                    .last()
                    .map_or(1, |entry| entry.sequence.saturating_add(1)),
                timestamp: now_timestamp(),
                event,
            };
            validate_next_event(session, &record)?;
            let line = encode_record(session.id(), &record)?;
            let new_size = append_encoded_record(&path, &line, self.deferred_sync())?;
            session.append_record(record.clone());
            session.validated_bytes = usize::try_from(new_size).unwrap_or(usize::MAX);
            drop(lock);
            return Ok(record);
        }

        // A second store may have appended a suffix without replacing the
        // file. Reconcile just those bytes under the lock; full replay is
        // reserved for replacement, truncation, or an incomplete suffix.
        if session.validated_bytes > 0
            && session.file_identity == Some(current_identity)
            && usize::try_from(file_size)
                .ok()
                .is_some_and(|size| size > session.validated_bytes)
            && reconcile_external_tail(session, &path, file_size as usize)?
        {
            let record = SessionEventRecord {
                id: EventId::new(),
                sequence: session
                    .events
                    .last()
                    .map_or(1, |entry| entry.sequence.saturating_add(1)),
                timestamp: now_timestamp(),
                event,
            };
            validate_next_event(session, &record)?;
            let line = encode_record(session.id(), &record)?;
            let new_size = append_encoded_record(&path, &line, self.deferred_sync())?;
            session.append_record(record.clone());
            session.validated_bytes = usize::try_from(new_size).unwrap_or(usize::MAX);
            drop(lock);
            return Ok(record);
        }

        // Re-read under the lock. Two processes may each hold an older
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
        validate_next_event(&disk_session, &record)?;
        let line = encode_record(disk_session.id(), &record)?;
        let new_size = append_encoded_record(&path, &line, self.deferred_sync())?;
        disk_session.append_record(record.clone());
        disk_session.validated_bytes = usize::try_from(new_size).unwrap_or(usize::MAX);
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
    let (mut session, recovery) = decode_session_file_bytes(&contents, path)?;
    session.path = Some(path.to_path_buf());
    session.validated_bytes = recovery.valid_bytes;
    session.file_identity = Some(file_identity(path)?);
    Ok(session)
}

/// Re-read a session file for appending: returns the decoded session plus
/// the crash-tail recovery metadata.  The file is never mutated here; the
/// locked `append_event` path truncates only after this reports an
/// incomplete tail.
fn load_session_file_for_append(path: &Path) -> Result<(Session, TailRecovery)> {
    let contents = read_file(path)?;
    let (mut session, recovery) = decode_session_file_bytes(&contents, path)?;
    session.path = Some(path.to_path_buf());
    session.validated_bytes = recovery.valid_bytes;
    session.file_identity = Some(file_identity(path)?);
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

/// Reconcile records appended by another store while retaining the cached
/// session vector. Returns `false` when the suffix ends in an unterminated
/// malformed line; the caller then performs the recovery-aware full reload.
fn reconcile_external_tail(session: &mut Session, path: &Path, file_size: usize) -> Result<bool> {
    let mut file =
        File::open(path).map_err(|source| io_error("open session suffix", path, source))?;
    file.seek(SeekFrom::Start(session.validated_bytes as u64))
        .map_err(|source| io_error("seek session suffix", path, source))?;
    let mut suffix_bytes = Vec::new();
    file.read_to_end(&mut suffix_bytes)
        .map_err(|source| io_error("read session suffix", path, source))?;
    let mut suffix = match String::from_utf8(suffix_bytes) {
        Ok(suffix) => suffix,
        // Let the recovery-aware full decoder distinguish an invalid final
        // byte tail from corruption before the suffix.
        Err(_) => return Ok(false),
    };

    // A valid unterminated file is made canonical by the next writer. The
    // separator is not an event and belongs to the external append, not the
    // cached prefix. An extra separator after an already terminated prefix is
    // a real blank record and remains a hard error.
    if suffix.starts_with('\n') {
        if session.validated_bytes > 0 {
            let mut prefix =
                File::open(path).map_err(|source| io_error("open session prefix", path, source))?;
            prefix
                .seek(SeekFrom::Start(session.validated_bytes as u64 - 1))
                .map_err(|source| io_error("seek session prefix", path, source))?;
            let mut byte = [0u8; 1];
            prefix
                .read_exact(&mut byte)
                .map_err(|source| io_error("read session prefix", path, source))?;
            if byte[0] == b'\n' {
                return Err(SessionError::InvalidEvent(
                    "external session suffix starts with a blank line".into(),
                ));
            }
        }
        suffix.remove(0);
    }
    if suffix.is_empty() {
        session.validated_bytes = file_size;
        return Ok(true);
    }

    let mut records = Vec::new();
    let lines = suffix.split_inclusive('\n').collect::<Vec<_>>();
    for (index, raw_line) in lines.iter().enumerate() {
        let terminated = raw_line.ends_with('\n');
        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        if line.trim().is_empty() {
            return Err(SessionError::InvalidEvent(
                "external session suffix contains a blank line".into(),
            ));
        }
        match decode_event_line(line, session.id(), path, index + 1) {
            Ok(record) => records.push(record),
            Err(_error) if !terminated && index + 1 == lines.len() => return Ok(false),
            Err(error) => return Err(error),
        }
    }
    validate_event_suffix(session, &records)?;
    for record in records {
        session.append_record(record);
    }
    session.validated_bytes = file_size;
    Ok(true)
}

/// Append one canonical record, inserting a separator when the preceding
/// valid record ended without a newline. Returns the resulting file length.
fn append_encoded_record(path: &Path, line: &str, deferred_sync: bool) -> Result<u64> {
    let file_size = fs::metadata(path)
        .map_err(|source| io_error("stat session for append", path, source))?
        .len();
    let needs_separator = file_size > 0 && !file_ends_with_newline(path, file_size)?;
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|source| io_error("open session for append", path, source))?;
    let write_result = (|| -> std::io::Result<()> {
        if needs_separator {
            file.write_all(b"\n")?;
        }
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()
    })();
    let write_result = if write_result.is_err() || !deferred_sync {
        write_result.and_then(|_| file.sync_all())
    } else {
        write_result
    };
    write_result.map_err(|source| io_error("append session event", path, source))?;
    Ok(file_size
        .saturating_add(u64::from(needs_separator))
        .saturating_add(line.len() as u64)
        .saturating_add(1))
}

fn file_ends_with_newline(path: &Path, file_size: u64) -> Result<bool> {
    if file_size == 0 {
        return Ok(true);
    }
    let mut file =
        File::open(path).map_err(|source| io_error("open session tail", path, source))?;
    file.seek(SeekFrom::End(-1))
        .map_err(|source| io_error("seek session tail", path, source))?;
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte)
        .map_err(|source| io_error("read session tail", path, source))?;
    Ok(byte[0] == b'\n')
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|source| io_error("open session", path, source))?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .map_err(|source| io_error("read session", path, source))?;
    Ok(contents)
}

fn file_identity(path: &Path) -> Result<(u64, u64)> {
    let metadata = fs::metadata(path).map_err(|source| io_error("stat session", path, source))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok((metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos() as u64);
        Ok((metadata.len(), modified))
    }
}

#[derive(Deserialize)]
struct IndexHeaderEnvelope {
    version: Option<u32>,
    #[serde(rename = "type")]
    kind: Option<String>,
    session_id: Option<String>,
    data: Option<Box<RawValue>>,
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
}

#[derive(Deserialize, Default)]
struct IndexedDataEnvelope {
    #[serde(default)]
    data: IndexedData,
}

#[derive(Deserialize, Default)]
struct IndexedMessageEnvelope {
    #[serde(default)]
    data: IndexedMessage,
}

#[derive(Deserialize, Default)]
struct IndexedData {
    provider: Option<String>,
    model: Option<String>,
    title: Option<String>,
}

#[derive(Deserialize, Default)]
struct IndexedMessage {
    #[serde(default)]
    content: Vec<IndexedContent>,
}

#[derive(Deserialize)]
struct IndexedContent {
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
}

fn indexed_data(data: Option<&RawValue>) -> &str {
    data.map_or("{}", |data| data.get())
}

fn index_title(value: &str) -> String {
    let first_line = value
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(value)
        .trim();
    if first_line.len() <= 80 {
        return first_line.to_owned();
    }
    let mut end = 79;
    while end > 0 && !first_line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &first_line[..end])
}

#[derive(Clone, Copy)]
struct IndexSpan {
    start: u64,
    bytes: u64,
    end: u64,
}

enum IndexRecord<T> {
    Eof,
    Invalid,
    Value(T, IndexSpan),
}

/// Deserialize one physical JSONL line without first collecting it in a
/// line-sized buffer. Scanning uses `BufRead::fill_buf`; the reader is then
/// rewound and serde receives an exact-length `Take`, so concatenated objects,
/// blank lines, and trailing garbage remain invalid JSONL records.
fn next_index_record<T: serde::de::DeserializeOwned>(
    reader: &mut BufReader<File>,
) -> std::io::Result<IndexRecord<T>> {
    let start = reader.stream_position()?;
    let mut line_bytes = 0u64;
    let mut terminated = false;
    let mut has_non_whitespace = false;
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        if let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            has_non_whitespace |= buffer[..newline]
                .iter()
                .any(|byte| !byte.is_ascii_whitespace());
            line_bytes = line_bytes.saturating_add(newline as u64);
            reader.consume(newline + 1);
            terminated = true;
            break;
        }
        has_non_whitespace |= buffer.iter().any(|byte| !byte.is_ascii_whitespace());
        line_bytes = line_bytes.saturating_add(buffer.len() as u64);
        let consumed = buffer.len();
        reader.consume(consumed);
    }
    let end = reader.stream_position()?;
    if line_bytes == 0 && !terminated {
        return Ok(IndexRecord::Eof);
    }
    if !has_non_whitespace {
        return Ok(IndexRecord::Invalid);
    }

    reader.seek(SeekFrom::Start(start))?;
    let parsed = serde_json::from_reader(reader.by_ref().take(line_bytes));
    reader.seek(SeekFrom::Start(end))?;
    let span = IndexSpan {
        start,
        bytes: line_bytes,
        end,
    };
    Ok(match parsed {
        Ok(value) => IndexRecord::Value(value, span),
        Err(_) => IndexRecord::Invalid,
    })
}

fn parse_index_span<T: serde::de::DeserializeOwned>(
    reader: &mut BufReader<File>,
    span: IndexSpan,
) -> std::io::Result<Option<T>> {
    reader.seek(SeekFrom::Start(span.start))?;
    let parsed = serde_json::from_reader(reader.by_ref().take(span.bytes)).ok();
    reader.seek(SeekFrom::Start(span.end))?;
    Ok(parsed)
}

/// Read only the metadata needed by the session picker. This deliberately
/// avoids rebuilding provider messages or cloning large tool results.
fn index_file(path: &Path, workspace: Option<&Path>) -> Result<Option<SessionIndexEntry>> {
    let file = File::open(path).map_err(|source| io_error("open session index", path, source))?;
    let mut reader = BufReader::new(file);
    let header: IndexHeaderEnvelope = match next_index_record(&mut reader)
        .map_err(|source| io_error("read session header", path, source))?
    {
        IndexRecord::Value(header, _) => header,
        IndexRecord::Eof | IndexRecord::Invalid => return Ok(None),
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
        serde_json::from_str::<SessionMetadata>(indexed_data(header.data.as_deref())),
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
    // Parse one physical record at a time while serde skips irrelevant `data`
    // fields. Multi-megabyte tool results are neither retained as RawValue nor
    // accumulated into a line-sized String.
    loop {
        let raw = match next_index_record::<IndexEnvelope>(&mut reader)
            .map_err(|source| io_error("read session index", path, source))?
        {
            IndexRecord::Value(raw, span) => (raw, span),
            IndexRecord::Eof => break,
            IndexRecord::Invalid => return Ok(None),
        };
        let (raw, span) = raw;
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
            // Re-read only metadata-bearing records from the bounded physical
            // span. This second pass is independent of JSON object key order;
            // large tool-result payloads are never materialized.
            "user_message" | "assistant_message" => {
                let Some(message) =
                    parse_index_span::<IndexedMessageEnvelope>(&mut reader, span)
                        .map_err(|source| io_error("read session message index", path, source))?
                else {
                    return Ok(None);
                };
                has_conversation |= !message.data.content.is_empty();
                if title.is_none() && kind == "user_message" {
                    title = message
                        .data
                        .content
                        .iter()
                        .filter_map(|content| {
                            (content.kind.as_deref() == Some("text"))
                                .then_some(content.text.as_deref())
                                .flatten()
                        })
                        .find(|text| !text.trim().is_empty())
                        .map(index_title);
                }
            }
            "reasoning" | "tool_call" | "tool_result" | "compaction" => {
                has_conversation = true;
            }
            "model_change" => {
                let Some(change) = parse_index_span::<IndexedDataEnvelope>(&mut reader, span)
                    .map_err(|source| io_error("read session model index", path, source))?
                else {
                    return Ok(None);
                };
                provider = change.data.provider;
                model = change.data.model;
            }
            "metadata_change" => {
                let Some(change) = parse_index_span::<IndexedDataEnvelope>(&mut reader, span)
                    .map_err(|source| io_error("read session metadata index", path, source))?
                else {
                    return Ok(None);
                };
                title = change.data.title;
            }
            _ => {}
        }
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

fn index_from_session(session: &Session, path: &Path) -> SessionIndexEntry {
    let has_conversation = session.events.iter().any(|record| match &record.event {
        SessionEvent::UserMessage { message } | SessionEvent::AssistantMessage { message } => {
            !message.content.is_empty()
        }
        SessionEvent::Reasoning { text } => !text.trim().is_empty(),
        SessionEvent::ToolCall { .. }
        | SessionEvent::ToolResult { .. }
        | SessionEvent::CompactionSummary { .. } => true,
        _ => false,
    });
    SessionIndexEntry {
        id: session.id(),
        short_id: session.id().short(),
        title: session.title().map(str::to_owned),
        workspace_root: session.metadata.workspace_root.clone(),
        created_at: session.metadata.created_at.clone(),
        updated_at: session.metadata.updated_at.clone(),
        provider: session.metadata.provider.clone(),
        model: session.metadata.model.clone(),
        parent_session: session.metadata.parent_session,
        event_count: session.events.len(),
        has_conversation,
        path: path.to_path_buf(),
        bytes: fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or_default(),
    }
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
        match index_file(&path, workspace)? {
            Some(entry) => result.push(entry),
            None => {
                // A torn final JSONL line is recoverable for reads even
                // though the metadata-only scanner cannot parse that line.
                // Fall back to the recovery-aware loader; terminated or
                // middle-file corruption still remains excluded.
                if let Ok(session) = load_session_file(&path)
                    && workspace.is_none_or(|workspace| {
                        normalize_workspace(session.metadata.workspace_root.clone())
                            .ok()
                            .is_some_and(|root| root == workspace)
                    })
                {
                    result.push(index_from_session(&session, &path));
                }
            }
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
    // Fail-closed test hook: simulate an OS permission failure before any
    // repair so tests prove `create` propagates instead of continuing with
    // a permissive directory. Only active under `#[cfg(test)]` injection.
    #[cfg(test)]
    if injected_secure_dir_failure() {
        return Err(io_error(
            "secure directory permissions",
            path,
            std::io::Error::other("injected directory permission failure"),
        ));
    }
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
    // Fail-closed test hook: simulate an OS permission failure before any
    // repair so tests prove append/create propagates instead of continuing
    // with a permissive file. Only active under `#[cfg(test)]` injection.
    #[cfg(test)]
    if injected_secure_file_failure() {
        return Err(io_error(
            "secure file permissions",
            path,
            std::io::Error::other("injected file permission failure"),
        ));
    }
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
    // Fail-closed test hook: simulate a parent-fsync failure so tests prove
    // durable creates propagate instead of leaving an undiscoverable entry.
    // Only active under `#[cfg(test)]` injection.
    #[cfg(test)]
    if injected_sync_parent_failure() {
        return Err(io_error(
            "sync parent directory",
            path,
            std::io::Error::other("injected parent sync failure"),
        ));
    }
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
    file: File,
}

impl SessionLock {
    fn lock_path(session_path: &Path) -> PathBuf {
        session_path.with_extension("jsonl.lock")
    }

    fn acquire(session_path: &Path) -> Result<Self> {
        let path = Self::lock_path(session_path);
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
        .map_err(|source| io_error("open session lock", &path, source))?;
        // Existing sidecars from older versions are secured before any lock
        // metadata or session bytes are written. Failure is fail-closed.
        ensure_private_file(&path)?;

        for _ in 0..LOCK_ATTEMPTS {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { file }),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(LOCK_WAIT);
                }
                Err(source) => return Err(io_error("lock session", &path, source)),
            }
        }
        Err(SessionError::LockUnavailable(path))
    }
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        // The sidecar inode is intentionally retained. Removing it after
        // unlocking would let a waiting process create and lock a different
        // inode, bypassing contenders that still hold the old one. Advisory
        // locks are released by the OS when this file handle is dropped,
        // including after an unexpected process exit.
        let _ = self.file.unlock();
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
        assert_eq!(
            entries
                .iter()
                .find(|entry| entry.id == session.id())
                .and_then(|entry| entry.title.as_deref()),
            Some("hello")
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
    fn two_stores_append_alternately_without_duplicate_sequences() {
        // PERF-5 two-handle correctness: two `SessionStore` handles over
        // the same root/workspace (separate `validated_bytes` cursors,
        // same lock file) append alternately. Disk-derived sequences keep
        // every record unique and ordered — the cross-handle path the
        // single-store stale-view tests cannot exercise.
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let first_store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let second_store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let session = first_store.create(SessionCreateOptions::default()).unwrap();
        let id = session.id();
        let mut first = first_store.open(&id).unwrap();
        let mut second = second_store.open(&id).unwrap();
        for index in 0..20u32 {
            let (store, target) = if index % 2 == 0 {
                (&first_store, &mut first)
            } else {
                (&second_store, &mut second)
            };
            store
                .append_event(
                    target,
                    SessionEvent::UserMessage {
                        message: StoredMessage::from_llm(&Message::user(format!("user {index}"))),
                    },
                )
                .unwrap();
        }
        let loaded = first_store.open(&id).unwrap();
        assert_eq!(loaded.events.len(), 20);
        assert_eq!(
            loaded
                .events
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            (1..=20).collect::<Vec<_>>(),
            "alternating two-store appends must not duplicate sequences"
        );
        // The idle handle's cursor converges on its next append (it
        // reconciles the peer's suffix then); disk is authoritative.
        let (store, target) = (&first_store, &mut first);
        store
            .append_event(
                target,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("converge")),
                },
            )
            .unwrap();
        assert_eq!(target.events.len(), 21);
        assert_eq!(second.events.len(), 20);
    }

    #[test]
    fn listing_reports_conversation_without_materializing_payloads() {
        // PERF-5 metadata-only listing: `has_conversation` for a session
        // with a multi-megabyte tool result must come from the streaming
        // `index_file` path (kind/text envelope only), never from decoding
        // or cloning the full payload. A corrupt-payload session still
        // lists cheaply via the same path when its envelope is intact.
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::ToolCall {
                    call: StoredToolCall {
                        id: "big-1".into(),
                        name: "read".into(),
                        arguments: serde_json::json!({"path": "big.txt"}),
                    },
                },
            )
            .unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::ToolResult {
                    tool_call_id: "big-1".into(),
                    content: "y".repeat(4 * 1024 * 1024),
                    is_error: false,
                    tool_name: Some("read".into()),
                },
            )
            .unwrap();
        let started = std::time::Instant::now();
        let entries = store.list().unwrap();
        let elapsed = started.elapsed();
        let entry = entries
            .iter()
            .find(|entry| entry.id == session.id())
            .expect("session must be listed");
        assert!(entry.has_conversation);
        assert_eq!(entry.event_count, 2);
        // 4MB payload indexed in well under a second: the scanner never
        // built provider messages or cloned the result.
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "listing touched the payload: {elapsed:?}"
        );
    }

    #[test]
    fn listing_rejects_concatenated_records_on_one_line() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        for text in ["one", "two"] {
            store
                .append_event(
                    &mut session,
                    SessionEvent::UserMessage {
                        message: StoredMessage::from_llm(&Message::user(text)),
                    },
                )
                .unwrap();
        }
        let path = session.file_path().unwrap();
        let lines = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        fs::write(path, format!("{}\n{}{}\n", lines[0], lines[1], lines[2])).unwrap();
        assert!(
            index_file(path, Some(store.workspace_root()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn listing_rejects_blank_jsonl_records() {
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
        let path = session.file_path().unwrap();
        let contents = fs::read_to_string(path).unwrap();
        fs::write(path, contents.replacen('\n', "\n\n", 1)).unwrap();
        assert!(
            index_file(path, Some(store.workspace_root()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn listing_accepts_data_before_type_and_preserves_user_metadata() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("hello reordered")),
                },
            )
            .unwrap();
        let path = session.file_path().unwrap();
        let mut lines = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let value: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        let object = value.as_object().unwrap();
        let data = serde_json::to_string(&object["data"]).unwrap();
        let kind = serde_json::to_string(&object["type"]).unwrap();
        let mut reordered = format!("{{\"data\":{data},\"type\":{kind}");
        for key in ["version", "session_id", "event_id", "sequence", "timestamp"] {
            reordered.push(',');
            reordered.push_str(&serde_json::to_string(key).unwrap());
            reordered.push(':');
            reordered.push_str(&serde_json::to_string(&object[key]).unwrap());
        }
        reordered.push('}');
        lines[1] = reordered;
        fs::write(path, format!("{}\n{}\n", lines[0], lines[1])).unwrap();

        let entry = index_file(path, Some(store.workspace_root()))
            .unwrap()
            .unwrap();
        assert!(entry.has_conversation);
        assert_eq!(entry.title.as_deref(), Some("hello reordered"));
    }

    #[test]
    fn stale_store_appends_reconcile_external_tail() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let session = store.create(SessionCreateOptions::default()).unwrap();
        let id = session.id();
        let mut first = store.open(&id).unwrap();
        let mut second = store.open(&id).unwrap();

        store
            .append_event(
                &mut first,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("first")),
                },
            )
            .unwrap();
        store
            .append_event(
                &mut second,
                SessionEvent::AssistantMessage {
                    message: StoredMessage::from_llm(&Message::assistant(vec![
                        llm::Content::Text("second".into()),
                    ])),
                },
            )
            .unwrap();
        let loaded = store.open(&id).unwrap();
        assert_eq!(loaded.events.len(), 2);
        assert_eq!(loaded.events[1].sequence, 2);
    }

    #[test]
    fn stale_store_reconciles_repeated_external_tails() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let session = store.create(SessionCreateOptions::default()).unwrap();
        let id = session.id();
        let mut first = store.open(&id).unwrap();
        let mut second = store.open(&id).unwrap();
        for index in 0..40 {
            let target = if index % 2 == 0 {
                &mut first
            } else {
                &mut second
            };
            store
                .append_event(
                    target,
                    if index % 2 == 0 {
                        SessionEvent::UserMessage {
                            message: StoredMessage::from_llm(&Message::user(format!(
                                "user {index}"
                            ))),
                        }
                    } else {
                        SessionEvent::AssistantMessage {
                            message: StoredMessage::from_llm(&Message::assistant(vec![
                                llm::Content::Text(format!("assistant {index}")),
                            ])),
                        }
                    },
                )
                .unwrap();
        }
        let loaded = store.open(&id).unwrap();
        assert_eq!(loaded.events.len(), 40);
        assert_eq!(
            loaded
                .events
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            (1..=40).collect::<Vec<_>>()
        );
        assert_eq!(second.events.len(), 40);
    }

    #[test]
    fn append_after_valid_unterminated_record_inserts_separator() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("before")),
                },
            )
            .unwrap();
        let path = session.path().unwrap().clone();
        let mut raw = fs::read(&path).unwrap();
        assert_eq!(raw.pop(), Some(b'\n'));
        fs::write(&path, raw).unwrap();
        let mut loaded = store.open(&session.id()).unwrap();
        store
            .append_event(
                &mut loaded,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("after")),
                },
            )
            .unwrap();
        let reopened = store.open(&session.id()).unwrap();
        assert_eq!(reopened.events.len(), 2);
        assert_eq!(reopened.events[1].sequence, 2);
        assert!(
            fs::read(&path)
                .unwrap()
                .windows(2)
                .any(|pair| pair == b"\n{")
        );
    }

    #[test]
    fn corrupt_replacement_is_rejected_by_full_validation() {
        // A replacement whose bytes no longer decode must fail the append
        // loudly (full validation), never silently reconcile or truncate.
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("before")),
                },
            )
            .unwrap();
        let mut stale = store.open(&session.id()).unwrap();
        let path = session.path().unwrap().clone();
        let raw = fs::read(&path).unwrap();
        let header_end = raw.iter().position(|byte| *byte == b'\n').unwrap() + 1;
        let mut corrupt = raw[..header_end].to_vec();
        corrupt.extend_from_slice(b"{not valid json\n");
        fs::write(&path, &corrupt).unwrap();
        let error = store
            .append_event(
                &mut stale,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("after")),
                },
            )
            .unwrap_err();
        assert!(
            !matches!(error, SessionError::NotFound(_)),
            "corruption must fail validation, got {error:?}"
        );
        // The corrupt file is untouched by the failed append; the stale
        // view gained nothing.
        assert_eq!(fs::read(&path).unwrap(), corrupt);
        assert_eq!(stale.events.len(), 1);
    }

    #[test]
    fn replacement_with_same_length_is_reconciled_before_append() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("before")),
                },
            )
            .unwrap();
        let mut stale = store.open(&session.id()).unwrap();
        let path = session.path().unwrap().clone();
        let original = fs::read_to_string(&path).unwrap();
        fs::write(&path, original.replace("before", "altered")).unwrap();
        store
            .append_event(
                &mut stale,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("after")),
                },
            )
            .unwrap();
        let reopened = store.open(&session.id()).unwrap();
        assert_eq!(reopened.events.len(), 2);
        assert_eq!(reopened.context_messages()[0], Message::user("altered"));
    }

    #[test]
    fn truncation_to_valid_boundary_is_reconciled_before_append() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        for text in ["first", "second"] {
            store
                .append_event(
                    &mut session,
                    SessionEvent::UserMessage {
                        message: StoredMessage::from_llm(&Message::user(text)),
                    },
                )
                .unwrap();
        }
        let mut stale = store.open(&session.id()).unwrap();
        let path = session.path().unwrap().clone();
        let raw = fs::read(&path).unwrap();
        let header_end = raw.iter().position(|byte| *byte == b'\n').unwrap() + 1;
        let first_event_end = raw[header_end..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| header_end + offset + 1)
            .unwrap();
        fs::write(&path, &raw[..first_event_end]).unwrap();
        assert!(first_event_end > header_end);
        store
            .append_event(
                &mut stale,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("replacement")),
                },
            )
            .unwrap();
        let reopened = store.open(&session.id()).unwrap();
        assert_eq!(reopened.events.len(), 2);
        assert_eq!(reopened.events[1].sequence, 2);
        assert_eq!(reopened.context_messages()[1], Message::user("replacement"));
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
        // Listing and loading alone do not mutate the file; listing falls
        // back to the recovery-aware loader for the torn final line.
        let entries = store.list().unwrap();
        assert!(entries.iter().any(|entry| entry.id == session.id()));
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
    fn append_repairs_an_unterminated_utf8_crash_tail() {
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
        let path = session.path().unwrap().clone();
        // The first byte of the final `é` is valid UTF-8 only as part of the
        // next record; it must be treated as an incomplete crash tail.
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"message\":\"\xc3")
            .unwrap();

        let loaded = store.open(&session.id()).unwrap();
        assert_eq!(loaded.events, valid_events);
        let mut repaired = loaded;
        store
            .append_event(
                &mut repaired,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&Message::user("after crash")),
                },
            )
            .unwrap();
        assert_eq!(repaired.events.len(), valid_events.len() + 1);
        assert_eq!(store.open(&session.id()).unwrap().events, repaired.events);
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
    fn advisory_lock_serializes_contenders_and_retains_sidecar_inode() {
        let directory = tempdir().unwrap();
        let victim = directory.path().join("victim.jsonl");
        fs::write(&victim, "").unwrap();
        let lock_path = SessionLock::lock_path(&victim);

        let owner = SessionLock::acquire(&victim).unwrap();
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

        // The same inode is reused after release; a waiter cannot switch to a
        // replacement pathname between unlock and its next acquisition.
        let next = SessionLock::acquire(&victim).unwrap();
        assert!(lock_path.exists());
        drop(next);
        assert!(lock_path.exists());
    }

    #[test]
    fn concurrent_advisory_lock_contenders_never_overlap() {
        let directory = tempdir().unwrap();
        let victim = directory.path().join("victim.jsonl");
        fs::write(&victim, "").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let active = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let victim = victim.clone();
            let barrier = barrier.clone();
            let active = active.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                let lock = SessionLock::acquire(&victim).unwrap();
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

    /// SESSION-4: a directory-repair failure must fail the whole create
    /// instead of continuing with a permissive workspace directory. The
    /// injected error mirrors a real `chmod` failure, so `create` has no
    /// session file to fall back to and nothing durable is left behind.
    ///
    /// The flag is armed only around `create`: store construction also
    /// hardens the brand-new root (salt persistence), so arming earlier
    /// would fail `new` by design. The point is that hardening inside the
    /// write path propagates instead of being silently ignored.
    #[test]
    fn directory_permission_failure_fails_session_create() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let error = {
            let _guard = HardeningFaultGuard::new(true, false, false);
            store
                .create(SessionCreateOptions::default())
                .expect_err("create must fail closed on directory repair failure")
        };
        assert!(
            matches!(error, SessionError::Io { operation, .. } if operation == "secure directory permissions"),
            "unexpected error: {error:?}"
        );
        // `private_dir_all` created the workspace dir before the injected
        // repair failed; no session header may have been written into it.
        assert_eq!(fs::read_dir(store.workspace_dir()).unwrap().count(), 0);
    }

    /// SESSION-4: a file-repair failure inside lock acquisition must fail
    /// the append instead of writing through a permissively-mode sidecar.
    /// The store must surface the injected error and append nothing.
    ///
    /// Note the fault flag stays armed through the file-size assertion but
    /// is dropped before reopening: plain reads also repair permissions,
    /// so reading with the flag armed would fail by design. The point is
    /// that the failed append left prior records intact and readable.
    #[test]
    fn file_permission_failure_fails_session_append() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let mut session = store.create(SessionCreateOptions::default()).unwrap();
        let bytes_before = fs::metadata(session.path().unwrap()).unwrap().len();
        let error = {
            let _guard = HardeningFaultGuard::new(false, true, false);
            store
                .append_event(
                    &mut session,
                    SessionEvent::UserMessage {
                        message: StoredMessage::from_llm(&Message::user("blocked")),
                    },
                )
                .expect_err("append must fail closed on lock-sidecar repair failure")
        };
        assert!(
            matches!(error, SessionError::Io { operation, .. } if operation == "secure file permissions"),
            "unexpected error: {error:?}"
        );
        assert!(session.events.is_empty());
        assert_eq!(
            fs::metadata(session.path().unwrap()).unwrap().len(),
            bytes_before,
            "failed append must not grow the session file"
        );
        assert!(store.open(&session.id()).unwrap().events.is_empty());
    }

    /// SESSION-4: a parent-sync failure must fail the whole create instead
    /// of leaving a session header whose directory entry may not survive a
    /// crash. The injected error mirrors a real directory-`fsync` failure.
    ///
    /// As above, the flag is armed only around `create`: construction
    /// syncs the brand-new root, so arming earlier would fail `new` first.
    #[test]
    fn parent_sync_failure_fails_session_create() {
        let root = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let store = SessionStore::new(root.path(), workspace.path()).unwrap();
        let error = {
            let _guard = HardeningFaultGuard::new(false, false, true);
            store
                .create(SessionCreateOptions::default())
                .expect_err("create must fail closed on parent sync failure")
        };
        assert!(
            matches!(error, SessionError::Io { operation, .. } if operation == "sync parent directory"),
            "unexpected error: {error:?}"
        );
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
