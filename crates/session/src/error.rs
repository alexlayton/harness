use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// The result type used by the session crate.
pub type Result<T> = std::result::Result<T, SessionError>;

/// Errors returned while creating, validating, or persisting a session.
#[derive(Debug, Error)]
pub enum SessionError {
    #[error("{operation} {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("invalid session JSON in {} at line {line}: {source}", path.display())]
    Json {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid session format in {} at line {line}: {message}", path.display())]
    InvalidLine {
        path: PathBuf,
        line: usize,
        message: String,
    },

    #[error(
        "unsupported session format version {found}; this Harness supports version {supported}"
    )]
    UnsupportedVersion { found: u32, supported: u32 },

    #[error("invalid session ID `{0}`")]
    InvalidSessionId(String),

    #[error("session `{0}` was not found")]
    NotFound(String),

    #[error("session belongs to workspace {}, not {}", stored.display(), requested.display())]
    WorkspaceMismatch { stored: PathBuf, requested: PathBuf },

    #[error("session file {} is outside the session store root {}", path.display(), root.display())]
    PathOutsideStore { path: PathBuf, root: PathBuf },

    #[error("session file {} already exists", .0.display())]
    AlreadyExists(PathBuf),

    #[error("could not acquire session lock {}", .0.display())]
    LockUnavailable(PathBuf),

    #[error("invalid session event: {0}")]
    InvalidEvent(String),

    #[error("session has no durable file")]
    NotPersisted,

    #[error("cannot export a session over its source file: {}", .0.display())]
    ExportWouldOverwrite(PathBuf),

    #[error("no session is available")]
    NoSession,
}

pub(crate) fn io_error(
    operation: &'static str,
    path: impl Into<PathBuf>,
    source: io::Error,
) -> SessionError {
    SessionError::Io {
        operation,
        path: path.into(),
        source,
    }
}
