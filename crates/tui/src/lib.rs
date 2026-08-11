//! A small inline-viewport terminal UI.  The event adapter keeps this crate
//! independent from the agent crate, avoiding a library dependency cycle.

mod app;
mod input;
pub mod render;

pub use app::Tui;
pub use render::{TailTool, ToolRecord};

/// Internal input-channel control message used for a resettable Esc interrupt.
/// It is intentionally impossible to type through the normal text editor.
pub const INTERRUPT_MESSAGE: &str = "\0harness:interrupt";

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
}

pub trait TuiEvent: Send {
    fn into_ui_event(self) -> UiEvent;
}

impl TuiEvent for UiEvent {
    fn into_ui_event(self) -> UiEvent {
        self
    }
}
