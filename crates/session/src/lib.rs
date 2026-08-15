//! Durable conversation sessions for Harness.
//!
//! # Format
//!
//! A session is one append-only UTF-8 JSONL file.  The first line is a
//! versioned `session` header containing identity and metadata.  Every later
//! line is a versioned event envelope with an event ID, monotonically
//! increasing sequence, timestamp, and typed `data` payload.  Unknown event
//! kinds are preserved and ignored for provider context, allowing newer
//! Harness versions to add events without silently rewriting old history.
//!
//! The default state directory is `~/.harness/sessions`.  Set
//! `HARNESS_SESSION_DIR` to replace it, or `HARNESS_STATE_DIR` to choose the
//! parent state directory.  Files are scoped below a workspace-derived
//! directory and are created with private directory permissions where the
//! platform supports them.

pub mod codec;
pub mod compaction;
pub mod error;
pub mod export;
pub mod model;
pub mod store;

pub use codec::{
    decode_session, decode_session_file, encode_header, encode_record, encode_session,
};
pub use compaction::{
    CompactionPolicy, CompactionResult, append_compaction, deterministic_compaction,
};
pub use error::{Result, SessionError};
pub use export::{ExportOptions, export_jsonl, export_transcript};
pub use model::{
    EventId, FORMAT_VERSION, Session, SessionEvent, SessionEventRecord, SessionId, SessionMetadata,
    SessionSnapshotEntry, SessionUsage, StoredContent, StoredMessage, StoredRole, StoredToolCall,
    Timestamp, UsageSummary, context_messages, latest_compaction_boundary, snapshot_entries,
    usage_summary, validate_events,
};
pub use store::{
    LoadReport, RecoveryReport, RetentionPolicy, SessionCreateOptions, SessionIndexEntry,
    SessionStore, default_session_dir,
};
