use crate::codec::{decode_session_file, encode_header, encode_record};
use crate::error::{Result, SessionError, io_error};
use crate::model::{
    EventId, Session, SessionEvent, SessionEventRecord, SessionId, SessionMetadata, StoredContent,
    StoredToolCall, Timestamp, now_timestamp,
};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

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
    pub path: PathBuf,
    pub bytes: u64,
}

/// Result of loading a session.  A `true` recovery flag means the file ended
/// with a malformed unterminated JSON fragment, which was ignored safely.
#[derive(Clone, Debug)]
pub struct LoadReport {
    pub session: Session,
    pub recovered_trailing_line: bool,
    /// One-based line number when a trailing fragment was ignored.
    pub recovered_line: Option<usize>,
}

/// Result of truncating an incomplete trailing record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryReport {
    pub path: PathBuf,
    pub truncated: bool,
    pub removed_bytes: u64,
    /// One-based line number of the truncated fragment, if any.
    pub line: Option<usize>,
}

/// Filesystem-backed session storage.  Sessions are grouped by a stable key
/// derived from the workspace root, so a project never appears in another
/// project's normal listing.
#[derive(Clone, Debug)]
pub struct SessionStore {
    root: PathBuf,
    workspace_root: PathBuf,
    workspace_dir: PathBuf,
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
        })
    }

    /// Return the resolved default session directory without creating it.
    pub fn default_root() -> PathBuf {
        default_session_dir()
    }

    /// Construct the default Harness store.  `HARNESS_SESSION_DIR` is an
    /// exact directory override; otherwise `HARNESS_STATE_DIR` is treated as
    /// the parent of `sessions`; the default is `~/.harness/sessions`.
    pub fn default_for_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::new(default_session_dir(), workspace_root)
    }

    /// Alias useful to embedders that prefer the shorter name.
    pub fn for_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::default_for_workspace(workspace_root)
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
        fs::create_dir_all(&self.workspace_dir)
            .map_err(|source| io_error("create session directory", &self.workspace_dir, source))?;
        self.ensure_path_in_root(&self.workspace_dir)?;
        set_private_directory(&self.root);
        set_private_directory(&self.workspace_dir);

        let mut metadata =
            SessionMetadata::new(self.workspace_root.clone(), options.provider, options.model);
        metadata.title = options.title;
        metadata.parent_session = options.parent_session;
        let path = self.workspace_dir.join(format!("{}.jsonl", metadata.id));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    SessionError::AlreadyExists(path.clone())
                } else {
                    io_error("create session file", &path, source)
                }
            })?;
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
        set_private_file(&path);
        let session = Session {
            header_metadata: metadata.clone(),
            metadata,
            events: Vec::new(),
            path: Some(path.clone()),
        };
        self.write_current(&session.metadata.id)?;
        Ok(session)
    }

    /// Alias emphasizing that a new session is persisted before it is handed
    /// to the agent.
    pub fn create_session(&self, options: SessionCreateOptions) -> Result<Session> {
        self.create(options)
    }

    pub fn new_session(&self, options: SessionCreateOptions) -> Result<Session> {
        self.create(options)
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
        fs::create_dir_all(&self.workspace_dir)
            .map_err(|source| io_error("create session directory", &self.workspace_dir, source))?;
        self.ensure_path_in_root(&self.workspace_dir)?;
        set_private_directory(&self.root);
        set_private_directory(&self.workspace_dir);
        let path = self.workspace_dir.join(format!("{}.jsonl", metadata.id));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    SessionError::AlreadyExists(path.clone())
                } else {
                    io_error("create session file", &path, source)
                }
            })?;
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
        set_private_file(&path);
        let session = Session {
            header_metadata: metadata.clone(),
            metadata,
            events: Vec::new(),
            path: Some(path),
        };
        self.write_current(&session.id())?;
        Ok(session)
    }

    /// Append one event and make it durable before returning.  A sidecar
    /// create-new lock prevents two Harness processes from interleaving JSON
    /// records.  Every record is followed by a newline and `sync_all`.
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
        let mut disk_session = load_session_file(&path)?;
        if disk_session.id() != session.id() {
            return Err(SessionError::InvalidEvent(
                "session object does not match its file".into(),
            ));
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
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush())
            .and_then(|_| file.sync_all())
            .map_err(|source| io_error("append session event", &path, source))?;
        drop(file);
        disk_session.append_record(record.clone());
        *session = disk_session;
        drop(lock);
        Ok(record)
    }

    pub fn append(&self, session: &mut Session, event: SessionEvent) -> Result<SessionEventRecord> {
        self.append_event(session, event)
    }

    pub fn append_message(
        &self,
        session: &mut Session,
        message: &llm::Message,
    ) -> Result<SessionEventRecord> {
        let event = match message.role {
            llm::Role::User => SessionEvent::UserMessage {
                message: crate::model::StoredMessage::from_llm(message),
            },
            llm::Role::Assistant => SessionEvent::AssistantMessage {
                message: crate::model::StoredMessage::from_llm(message),
            },
            llm::Role::System | llm::Role::Tool => {
                return Err(SessionError::InvalidEvent(
                    "only user and assistant messages can be appended as messages".into(),
                ));
            }
        };
        self.append_event(session, event)
    }

    /// There is no long-lived buffered writer: append_event acknowledges only
    /// after flushing and syncing.  This method is provided for shutdown code
    /// and future buffered implementations.
    pub fn flush(&self, session: &Session) -> Result<()> {
        if let Some(path) = session.path() {
            let file = OpenOptions::new()
                .read(true)
                .open(path)
                .map_err(|source| io_error("open session for flush", path, source))?;
            file.sync_all()
                .map_err(|source| io_error("flush session", path, source))?;
        }
        Ok(())
    }

    pub fn open_session(&self, id: &SessionId) -> Result<Session> {
        self.open(id)
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
            // A freshly started Harness process creates an empty session
            // immediately.  Treating that placeholder as "latest" would
            // make `/load latest` unable to resume the previous conversation.
            if entries.first().is_some_and(|entry| entry.event_count == 0) && entries.len() > 1 {
                entries.remove(0);
            }
            return entries
                .first()
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

    pub fn load_by_id(&self, selector: &str) -> Result<Session> {
        self.load(selector)
    }

    pub fn open_path(&self, path: &Path) -> Result<Session> {
        self.load_path(path)
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

    pub fn load_with_report(&self, path: &Path) -> Result<LoadReport> {
        let resolved = path
            .canonicalize()
            .map_err(|source| io_error("resolve session path", path, source))?;
        let contents = read_file(&resolved)?;
        let (mut session, recovered) = decode_session_file(&contents, &resolved)?;
        let stored_workspace = normalize_workspace(session.metadata.workspace_root.clone())?;
        if stored_workspace != self.workspace_root {
            return Err(SessionError::WorkspaceMismatch {
                stored: session.metadata.workspace_root.clone(),
                requested: self.workspace_root.clone(),
            });
        }
        session.path = Some(resolved);
        Ok(LoadReport {
            session,
            recovered_trailing_line: recovered,
            recovered_line: recovered.then(|| contents.lines().count()),
        })
    }

    pub fn load_latest(&self) -> Result<Option<Session>> {
        match self.load("latest") {
            Ok(session) => Ok(Some(session)),
            Err(SessionError::NoSession) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub fn latest(&self) -> Result<Option<SessionIndexEntry>> {
        let mut entries = self.list()?;
        entries.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
        });
        Ok(entries.into_iter().next())
    }

    pub fn set_current(&self, session: &Session) -> Result<()> {
        if let Some(path) = session.path()
            && self.ensure_path_in_root(path).is_err()
        {
            // Explicitly loaded/exported files may be outside the store.  They
            // are valid read-only sessions but cannot be represented by the
            // workspace-local current pointer.
            return Ok(());
        }
        self.write_current(&session.id())
    }

    pub fn current(&self) -> Result<Option<Session>> {
        let pointer = self.workspace_dir.join(".current");
        let id = match fs::read_to_string(&pointer) {
            Ok(value) => value.trim().to_owned(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(io_error("read current session", &pointer, source)),
        };
        if id.is_empty() {
            return Ok(None);
        }
        self.load(&id).map(Some)
    }

    pub fn list(&self) -> Result<Vec<SessionIndexEntry>> {
        list_directory(&self.workspace_dir, Some(&self.workspace_root))
    }

    /// List sessions for every workspace below this store root.  Corrupt files
    /// are skipped by default; callers needing diagnostics can load each path
    /// directly and receive its first bad line.
    pub fn list_all(&self) -> Result<Vec<SessionIndexEntry>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        let directories = fs::read_dir(&self.root)
            .map_err(|source| io_error("list session store", &self.root, source))?;
        for directory in directories {
            let directory = directory
                .map_err(|source| io_error("read session store entry", &self.root, source))?;
            if !directory
                .file_type()
                .map_err(|source| {
                    io_error("inspect session store entry", directory.path(), source)
                })?
                .is_dir()
            {
                continue;
            }
            result.extend(list_directory(&directory.path(), None)?);
        }
        result.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        Ok(result)
    }

    pub fn export(
        &self,
        session: &Session,
        destination: Option<&Path>,
        options: &crate::export::ExportOptions,
    ) -> Result<PathBuf> {
        crate::export::export_jsonl(session, destination, options)
    }

    pub fn compact(
        &self,
        session: &mut Session,
        policy: &crate::compaction::CompactionPolicy,
    ) -> Result<Option<crate::compaction::CompactionResult>> {
        let Some(result) = crate::compaction::deterministic_compaction(session, policy) else {
            return Ok(None);
        };
        self.append_event(
            session,
            SessionEvent::CompactionSummary {
                summary: result.summary.clone(),
                compacted_through: result.compacted_through,
            },
        )?;
        Ok(Some(result))
    }

    pub fn rename(&self, session: &mut Session, title: Option<String>) -> Result<()> {
        self.append_event(session, SessionEvent::MetadataChange { title })?;
        Ok(())
    }

    /// Repair tool calls left at the end of a file by a process crash.  The
    /// synthetic error results are durable, so subsequent user messages pass
    /// strict ordering validation and providers receive a valid history.
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

    pub fn delete(&self, session: &Session) -> Result<()> {
        let Some(path) = session.path() else {
            return Err(SessionError::NotPersisted);
        };
        self.ensure_path_in_root(path)?;
        let current_id = fs::read_to_string(self.workspace_dir.join(".current"))
            .ok()
            .map(|value| value.trim().to_owned());
        fs::remove_file(path).map_err(|source| io_error("delete session", path, source))?;
        let lock = path.with_extension("jsonl.lock");
        let _ = fs::remove_file(lock);
        if current_id
            .as_deref()
            .is_some_and(|id| id == session.id().to_string())
        {
            let _ = fs::remove_file(self.workspace_dir.join(".current"));
        }
        Ok(())
    }

    /// Fork a session into a new append-only file.  Entries are replayed as
    /// fresh records, while `parent_session` records the origin identity.
    pub fn fork(&self, source: &Session, title: Option<String>) -> Result<Session> {
        let mut fork = self.create(SessionCreateOptions {
            title,
            provider: source.metadata.provider.clone(),
            model: source.metadata.model.clone(),
            parent_session: Some(source.id()),
        })?;
        for record in &source.events {
            self.append_event(&mut fork, record.event.clone())?;
        }
        Ok(fork)
    }

    /// Copy an external/exported session into this workspace with a new ID.
    pub fn import(&self, path: &Path, title: Option<String>) -> Result<Session> {
        let source = Self::load_any_path(path)?;
        let mut imported = self.create(SessionCreateOptions {
            title,
            provider: source.metadata.provider.clone(),
            model: source.metadata.model.clone(),
            parent_session: Some(source.id()),
        })?;
        for record in source.events {
            self.append_event(&mut imported, record.event)?;
        }
        Ok(imported)
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

    pub fn recover_trailing_line(&self, path: &Path) -> Result<RecoveryReport> {
        let resolved = path
            .canonicalize()
            .map_err(|source| io_error("resolve session path", path, source))?;
        self.ensure_path_in_root(&resolved)?;
        let contents = read_file(&resolved)?;
        let (_, recovered) = decode_session_file(&contents, &resolved)?;
        if !recovered {
            return Ok(RecoveryReport {
                path: resolved,
                truncated: false,
                removed_bytes: 0,
                line: None,
            });
        }
        let last_newline = contents
            .as_bytes()
            .iter()
            .rposition(|byte| *byte == b'\n')
            .map(|index| index + 1)
            .unwrap_or(0);
        let removed_bytes = (contents.len() - last_newline) as u64;
        let lock = SessionLock::acquire(&resolved)?;
        let temp = resolved.with_extension("jsonl.recover.tmp");
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp)
                .map_err(|source| io_error("create recovery file", &temp, source))?;
            file.write_all(&contents.as_bytes()[..last_newline])
                .and_then(|_| file.flush())
                .and_then(|_| file.sync_all())
                .map_err(|source| io_error("write recovery file", &temp, source))?;
            fs::rename(&temp, &resolved)
                .map_err(|source| io_error("replace recovered session", &resolved, source))?;
            set_private_file(&resolved);
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        drop(lock);
        result?;
        Ok(RecoveryReport {
            path: resolved,
            truncated: true,
            removed_bytes,
            line: Some(contents.lines().count()),
        })
    }

    pub fn cleanup(&self, policy: &RetentionPolicy) -> Result<usize> {
        let mut entries = self.list()?;
        entries.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        let now = SystemTime::now();
        let mut removed = 0;
        let mut total_bytes = entries.iter().map(|entry| entry.bytes).sum::<u64>();
        for (index, entry) in entries.into_iter().enumerate() {
            let over_count = policy.max_sessions.is_some_and(|max| index >= max);
            let over_bytes = policy.max_bytes.is_some_and(|max| total_bytes > max);
            let too_old = policy.max_age.is_some_and(|age| {
                fs::metadata(&entry.path)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|modified| now.duration_since(modified).ok())
                    .is_some_and(|elapsed| elapsed > age)
            });
            if !(over_count || over_bytes || too_old) {
                continue;
            }
            fs::remove_file(&entry.path)
                .map_err(|source| io_error("remove retained session", &entry.path, source))?;
            total_bytes = total_bytes.saturating_sub(entry.bytes);
            removed += 1;
        }
        Ok(removed)
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

    fn write_current(&self, id: &SessionId) -> Result<()> {
        fs::create_dir_all(&self.workspace_dir).map_err(|source| {
            io_error(
                "create current-session directory",
                &self.workspace_dir,
                source,
            )
        })?;
        let pointer = self.workspace_dir.join(".current");
        let temp = self
            .workspace_dir
            .join(format!(".current.tmp-{}", std::process::id()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp)
                .map_err(|source| io_error("create current-session file", &temp, source))?;
            file.write_all(id.to_string().as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .and_then(|_| file.flush())
                .and_then(|_| file.sync_all())
                .map_err(|source| io_error("write current-session file", &temp, source))?;
            fs::rename(&temp, &pointer)
                .map_err(|source| io_error("replace current-session file", &pointer, source))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    }
}

/// Age/count/size limits for optional retention cleanup.
#[derive(Clone, Debug, Default)]
pub struct RetentionPolicy {
    pub max_sessions: Option<usize>,
    pub max_age: Option<Duration>,
    pub max_bytes: Option<u64>,
}

fn load_session_file(path: &Path) -> Result<Session> {
    let contents = read_file(path)?;
    let (mut session, _) = decode_session_file(&contents, path)?;
    session.path = Some(path.to_path_buf());
    Ok(session)
}

fn read_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| io_error("open session", path, source))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|source| io_error("read session", path, source))?;
    Ok(contents)
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
        let Ok(session) = load_session_file(&path) else {
            continue;
        };
        if let Some(workspace) = workspace
            && normalize_workspace(session.metadata.workspace_root.clone())? != workspace
        {
            continue;
        }
        let bytes = fs::metadata(&path)
            .map(|metadata| metadata.len())
            .unwrap_or_default();
        result.push(SessionIndexEntry {
            id: session.id(),
            short_id: session.id().short(),
            title: session.metadata.title.clone(),
            workspace_root: session.metadata.workspace_root.clone(),
            created_at: session.metadata.created_at.clone(),
            updated_at: session.metadata.updated_at.clone(),
            provider: session.metadata.provider.clone(),
            model: session.metadata.model.clone(),
            parent_session: session.metadata.parent_session,
            event_count: session.events.len(),
            path,
            bytes,
        });
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
    fs::create_dir_all(root).map_err(|source| io_error("create session store", root, source))?;
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
            set_private_file(&salt_path);
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

/// Apply private (owner-only) permissions to a store directory.
///
/// This is a Unix-only hardening step.  On Windows, session files inherit
/// the permissions of the parent directory (typically the user's state
/// directory), which is consistent with the plan; the Windows ACL model
/// makes a portable equivalent out of scope here.
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
    // #[cfg(not(unix))] — no-op on Windows and other platforms (see above).
}

/// Apply private (owner-only) permissions to a session or salt file.
///
/// Unix-only for the same reason as [`set_private_directory`]; on Windows the
/// file inherits the parent directory's permissions.
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
    // #[cfg(not(unix))] — no-op on Windows and other platforms (see above).
}

struct SessionLock {
    path: PathBuf,
}

impl SessionLock {
    fn acquire(session_path: &Path) -> Result<Self> {
        let path = session_path.with_extension("jsonl.lock");
        for _ in 0..LOCK_ATTEMPTS {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "pid={}", std::process::id());
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_is_stale(&path) {
                        // The owner is gone (or the lock is ancient): steal it.
                        let _ = fs::remove_file(&path);
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
        let _ = fs::remove_file(&self.path);
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
        assert_eq!(store.list().unwrap().len(), 1);
        assert_eq!(store.current().unwrap().unwrap().id(), session.id());
    }

    #[test]
    fn corrupt_unterminated_tail_is_recoverable_but_middle_corruption_is_not() {
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
        let path = session.path().unwrap();
        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(br#"{"#).unwrap();
        file.flush().unwrap();
        let report = store.load_with_report(path).unwrap();
        assert!(report.recovered_trailing_line);
        let recovered = store.recover_trailing_line(path).unwrap();
        assert!(recovered.truncated);
        assert!(!store.recover_trailing_line(path).unwrap().truncated);
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
            .append_message(&mut loaded, &Message::user("continue"))
            .unwrap();
        assert!(
            loaded
                .events
                .iter()
                .any(|record| matches!(record.event, SessionEvent::TurnCancelled { .. }))
        );
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
