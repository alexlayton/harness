//! A direct-crossterm terminal UI for Harness: plain rows committed into the
//! terminal's native scrollback, a small live region at the bottom, and a
//! `›` input line. The host adapts runtime events into this crate's independent
//! protocol, avoiding a library dependency cycle.

mod app;
mod commands;
mod commit;
mod environment;
mod input;
mod paths;
mod render;
mod state;

pub use app::CrossTerm;
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
    /// Start a turn from a discovered skill's instructions.
    InvokeSkill { name: String },
    /// Ask the active provider for current subscription allowance usage.
    SubscriptionUsage,
    /// Ask the agent for the discovered-skill view (see [`UiEvent::SkillsLoaded`]).
    ListSkills,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    pub name: Option<String>,
    pub context_length: Option<u64>,
}

/// A skill as seen by the UI: name plus a short description. The TUI never
/// touches the filesystem or the frontmatter parser; main passes this list
/// straight from the discovered [`tools::skills`] catalog at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
}

/// A project-context file (AGENTS.md / CLAUDE.md) injected into the system
/// prompt. Only the display path is needed; the TUI shows which files are
/// loaded, not their contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFileEntry {
    /// Display path with `$HOME` abbreviated to `~` where applicable.
    pub path: String,
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
        ok: bool,
        duration_ms: u64,
        output: String,
        error: Option<String>,
    },
}

/// Provider-neutral subscription usage rendered by `/usage`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubscriptionUsage {
    /// Provider plan name, when available.
    pub plan: Option<String>,
    /// Independently resetting allowance windows.
    pub windows: Vec<SubscriptionUsageWindow>,
}

/// One display-ready subscription allowance window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubscriptionUsageWindow {
    /// Human-readable window name.
    pub label: String,
    /// Percentage of the allowance consumed.
    pub used_percent: u16,
    /// Provider status, when available.
    pub status: Option<String>,
    /// Provider-formatted absolute reset time.
    pub resets_at: Option<String>,
    /// Seconds until reset, when available.
    pub resets_after_seconds: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UiEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStarted {
        /// Stable harness call id; the key that correlates start/finish of
        /// concurrent calls in the keyed running-tool state.
        call_id: String,
        name: String,
        summary: String,
    },
    ToolCallFinished {
        call_id: String,
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
    /// The full discovered-skill view for `/skills`, delivered once per
    /// request. Includes diagnostics so broken skills surface to the user;
    /// they are never placed in the model prompt.
    SkillsLoaded {
        skills: Vec<SkillEntry>,
        diagnostics: Vec<String>,
        /// True when no skills were discovered at all.
        empty: bool,
    },
    UsageUpdated {
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        reasoning_tokens: u64,
        cost: String,
    },
    /// Current allowance returned by the active subscription provider.
    SubscriptionUsageLoaded {
        provider: String,
        usage: SubscriptionUsage,
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
