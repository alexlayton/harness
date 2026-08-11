pub mod agent;
pub mod config;
pub mod prompt;
pub mod tools;

pub use agent::{Agent, AgentEvent};

impl tui::TuiEvent for AgentEvent {
    fn into_ui_event(self) -> tui::UiEvent {
        match self {
            AgentEvent::TextDelta(value) => tui::UiEvent::TextDelta(value),
            AgentEvent::ReasoningDelta(value) => tui::UiEvent::ReasoningDelta(value),
            AgentEvent::ToolCallStarted {
                name,
                summary,
                arguments,
            } => tui::UiEvent::ToolCallStarted {
                name,
                summary,
                arguments,
            },
            AgentEvent::ToolCallFinished {
                name,
                summary,
                ok,
                duration_ms,
                output,
                error,
            } => tui::UiEvent::ToolCallFinished {
                name,
                summary,
                ok,
                duration_ms,
                output,
                error,
            },
            AgentEvent::Retrying { attempt, message } => {
                tui::UiEvent::Retrying { attempt, message }
            }
            AgentEvent::TurnFinished => tui::UiEvent::TurnFinished,
            AgentEvent::Error(value) => tui::UiEvent::Error(value),
            AgentEvent::Notice(value) => tui::UiEvent::Notice(value),
            AgentEvent::ModelChanged { provider, model } => {
                tui::UiEvent::ModelChanged { provider, model }
            }
            AgentEvent::ModelList { provider, models } => tui::UiEvent::ModelList {
                provider,
                models: models
                    .into_iter()
                    .map(|model| tui::ModelEntry {
                        id: model.id,
                        name: model.name,
                        context_length: model.context_length,
                    })
                    .collect(),
            },
        }
    }
}
