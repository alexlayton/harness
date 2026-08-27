use tools::SkillEntry;

/// Frontend-neutral input commands consumed by the agent runtime.
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
    /// List sessions for the current workspace.
    ListSessions,
    /// Export the current session to JSONL.
    ExportSession { destination: Option<String> },
    /// Run the deterministic local compactor.
    CompactSession,
    /// Switch model, and provider when present.
    SetModel {
        provider: Option<String>,
        model: String,
    },
    /// Set reasoning effort for subsequent parent and subagent requests.
    SetReasoning { level: String },
    /// Fetch a provider's model list.
    ListModels { provider: String },
    /// Start a turn from a discovered skill's instructions.
    InvokeSkill { name: String },
    /// Fetch current subscription allowance usage.
    SubscriptionUsage,
    /// Return the discovered-skill view.
    ListSkills,
}

/// Events emitted by the agent for frontend rendering and lifecycle tracking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStarted {
        /// Provider-neutral tool call id from [`llm::ToolCall::id`]; the key
        /// frontends use to correlate start/finish of concurrent calls.
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
        /// The complete tool result, retained for optional expansion.
        output: String,
        /// The complete error text when the tool failed. The compact renderer
        /// is responsible for showing only a one-line preview.
        error: Option<String>,
    },
    Retrying {
        attempt: u32,
        message: String,
    },
    TurnFinished,
    Error(String),
    Notice(String),
    ModelChanged {
        provider: String,
        model: String,
    },
    /// Confirmed portable reasoning policy.
    ReasoningChanged {
        level: String,
    },
    ModelList {
        provider: String,
        models: Vec<llm::ModelInfo>,
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
        sessions: Vec<SessionListItem>,
    },
    SessionExported {
        path: String,
    },
    /// The discovered-skill view requested by the TUI's `/skills` command.
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
    /// Current context occupation and the effective configured/model window.
    ContextUsageUpdated {
        used_tokens: u64,
        max_tokens: u64,
    },
    /// Current allowance returned by the active subscription provider.
    SubscriptionUsageLoaded {
        provider: String,
        usage: llm::SubscriptionUsage,
    },
    CompactionFinished {
        compacted_through: u64,
        summary_bytes: usize,
        auto: bool,
        reason: CompactionReason,
    },
}

/// Why a compaction ran. Drives the UI wording (auto vs manual vs overflow)
/// and is recorded on the `CompactionFinished` event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionReason {
    /// Pre-turn trigger fired past the threshold.
    Auto,
    /// User invoked `/compact`.
    Manual,
    /// Provider rejected a request for exceeding the context window.
    Overflow,
}

impl CompactionReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
            Self::Overflow => "overflow",
        }
    }
}

impl std::fmt::Display for CompactionReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

/// One display-ready entry in a loaded session snapshot.
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

/// Metadata used to render one selectable session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionListItem {
    pub id: String,
    pub short_id: String,
    pub title: Option<String>,
    pub updated_at: String,
    pub workspace: String,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Why a single turn did not complete normally.  `run` uses this to decide
/// whether to keep draining queued input.
#[derive(Debug, PartialEq, Eq)]
pub enum TurnError {
    /// The application cancellation token fired while the turn was in flight;
    /// the run loop should stop immediately.
    Shutdown,
    /// A durable session event could not be persisted; the turn was aborted.
    Persist(String),
}
