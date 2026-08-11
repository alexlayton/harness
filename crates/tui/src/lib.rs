//! A small inline-viewport terminal UI.  The event adapter keeps this crate
//! independent from the agent crate, avoiding a library dependency cycle.

mod app;
pub mod commands;
mod input;
pub mod render;

pub use app::Tui;
pub use render::{TailTool, ToolRecord};

/// Messages sent from the terminal UI to the agent. Keeping this protocol in
/// the serde-free TUI crate avoids a dependency cycle while allowing commands
/// to travel through the same queue as ordinary user input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputMessage {
    /// Normal user text for the model.
    Message(String),
    /// Turn-local Esc interrupt.
    Interrupt,
    /// Clear the current conversation history.
    NewConversation,
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
        /// Tool output is retained for the optional detail view.
        output: String,
        /// Full error text.  The compact renderer displays only its first
        /// line.
        error: Option<String>,
    },
    Retrying {
        attempt: u32,
        message: String,
    },
    TurnFinished,
    Error(String),
    /// Informational command feedback committed to scrollback.
    Notice(String),
    /// Confirmed provider/model labels for the status line.
    ModelChanged {
        provider: String,
        model: String,
    },
    /// Cached completion models for a provider.
    ModelList {
        provider: String,
        models: Vec<ModelEntry>,
    },
}

pub trait TuiEvent: Send {
    fn into_ui_event(self) -> UiEvent;
}

impl TuiEvent for UiEvent {
    fn into_ui_event(self) -> UiEvent {
        self
    }
}
