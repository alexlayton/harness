//! A retained-mode terminal UI for Harness. The event adapter keeps this
//! crate independent from the agent crate, avoiding a library dependency cycle.

mod app;
pub mod attachments;
pub mod commands;
mod completion;
mod environment;
mod events;
mod input;
mod layout;
pub mod render;
mod state;

pub use app::Tui;
pub use render::TailTool;
pub use state::{ToolRecord, ToolStatus};

/// Messages sent from the terminal UI to the agent. Keeping this protocol in
/// the serde-free TUI crate avoids a dependency cycle while allowing commands
/// to travel through the same queue as ordinary user input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputMessage {
    /// Normal user text for the model.
    Message(String),
    /// Turn-local interrupt.
    Interrupt,
    /// Start and persist a new conversation without deleting the old one.
    NewConversation,
    /// Load a session by ID, unique prefix, `latest`, or path.
    LoadSession { selector: String },
    /// Ask the agent to list sessions for the current workspace.
    ListSessions,
    /// Export the current session to JSONL. `None` means the current directory
    /// and a generated filename.
    ExportSession { destination: Option<String> },
    /// Run the deterministic local compactor.
    CompactSession,
    /// Switch model, and provider when `provider` is present.
    SetModel {
        provider: Option<String>,
        model: String,
    },
    /// Ask the agent to fetch a provider's model list for completion.
    ListModels { provider: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    pub name: Option<String>,
    pub context_length: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionSnapshotEntry {
    User {
        text: String,
    },
    Assistant {
        markdown: String,
        reasoning: String,
    },
    Tool {
        name: String,
        summary: String,
        arguments: String,
        ok: bool,
        duration_ms: u64,
        output: String,
        error: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UiEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStarted {
        name: String,
        summary: String,
        /// Pretty-printed and bounded by the agent before crossing into the
        /// serde-free TUI crate.
        arguments: String,
    },
    ToolCallFinished {
        name: String,
        summary: String,
        ok: bool,
        duration_ms: u64,
        /// Tool output is retained for optional expansion.
        output: String,
        /// Full error text. The compact renderer displays only a small preview.
        error: Option<String>,
    },
    Retrying {
        attempt: u32,
        message: String,
    },
    TurnFinished,
    Error(String),
    /// Informational command feedback committed to the retained transcript.
    Notice(String),
    /// Confirmed provider/model labels for the metadata line.
    ModelChanged {
        provider: String,
        model: String,
    },
    /// Cached completion models for a provider.
    ModelList {
        provider: String,
        models: Vec<ModelEntry>,
    },
    SessionChanged {
        id: String,
        title: Option<String>,
        loaded: bool,
    },
    SessionSnapshot {
        entries: Vec<SessionSnapshotEntry>,
    },
    SessionList {
        sessions: Vec<SessionListEntry>,
    },
    SessionExported {
        path: String,
    },
    UsageUpdated {
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        reasoning_tokens: u64,
        cost: String,
    },
    CompactionFinished {
        compacted_through: u64,
        summary_bytes: usize,
        auto: bool,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionListEntry {
    pub id: String,
    pub short_id: String,
    pub title: Option<String>,
    pub updated_at: String,
    pub workspace: String,
    pub provider: Option<String>,
    pub model: Option<String>,
}

pub trait TuiEvent: Send {
    fn into_ui_event(self) -> UiEvent;
}

impl TuiEvent for UiEvent {
    fn into_ui_event(self) -> UiEvent {
        self
    }
}
