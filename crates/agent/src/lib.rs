pub mod agent;
pub mod config;
pub mod headless;
pub mod prompt;
pub use tools;

use std::path::Path;

/// Render the project-context block (AGENTS.md / CLAUDE.md) for a workspace,
/// or an empty string when injection is disabled (`--no-context-files`). Used
/// by both frontends to attach context to the agent at construction.
pub fn project_context_for(workspace_root: &Path, disabled: bool) -> String {
    if disabled {
        return String::new();
    }
    let files = tools::context_files::load_context_files(workspace_root);
    tools::context_files::format_context_files(&files)
}

pub use agent::{Agent, AgentEvent};

impl tui::TuiEvent for AgentEvent {
    fn into_ui_event(self) -> tui::UiEvent {
        match self {
            AgentEvent::AuthStarted => tui::UiEvent::AuthStarted,
            AgentEvent::AuthPrompt { message } => tui::UiEvent::AuthPrompt { message },
            AgentEvent::AuthDeviceCode {
                verification_url,
                user_code,
                expires_in,
                interval,
            } => tui::UiEvent::AuthDeviceCode {
                verification_url,
                user_code,
                expires_in,
                interval,
            },
            AgentEvent::AuthProgress { message } => tui::UiEvent::AuthProgress { message },
            AgentEvent::AuthFinished => tui::UiEvent::AuthFinished,
            AgentEvent::AuthFailed { message } => tui::UiEvent::AuthFailed { message },
            AgentEvent::TextDelta(value) => tui::UiEvent::TextDelta(value),
            AgentEvent::ReasoningDelta(value) => tui::UiEvent::ReasoningDelta(value),
            AgentEvent::ToolCallStarted { name, summary } => {
                tui::UiEvent::ToolCallStarted { name, summary }
            }
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
            AgentEvent::SessionChanged { id, title, loaded } => {
                tui::UiEvent::SessionChanged { id, title, loaded }
            }
            AgentEvent::SessionSnapshot { entries } => tui::UiEvent::SessionSnapshot {
                entries: entries
                    .into_iter()
                    .map(|entry| match entry {
                        crate::agent::SessionSnapshotEntry::User { text } => {
                            tui::SessionSnapshotEntry::User { text }
                        }
                        crate::agent::SessionSnapshotEntry::Assistant {
                            markdown,
                            reasoning,
                        } => tui::SessionSnapshotEntry::Assistant {
                            markdown,
                            reasoning,
                        },
                        crate::agent::SessionSnapshotEntry::Tool {
                            name,
                            summary,
                            ok,
                            duration_ms,
                            output,
                            error,
                        } => tui::SessionSnapshotEntry::Tool {
                            name,
                            summary,
                            ok,
                            duration_ms,
                            output,
                            error,
                        },
                    })
                    .collect(),
            },
            AgentEvent::SessionList { sessions } => tui::UiEvent::SessionList {
                sessions: sessions
                    .into_iter()
                    .map(|session| tui::SessionListEntry {
                        id: session.id,
                        short_id: session.short_id,
                        title: session.title,
                        updated_at: session.updated_at,
                    })
                    .collect(),
            },
            AgentEvent::SessionExported { path } => tui::UiEvent::SessionExported { path },
            AgentEvent::UsageUpdated {
                input_tokens,
                output_tokens,
                cached_tokens,
                reasoning_tokens,
                cost,
            } => tui::UiEvent::UsageUpdated {
                input_tokens,
                output_tokens,
                cached_tokens,
                reasoning_tokens,
                cost,
            },
            AgentEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
                auto,
                reason,
            } => tui::UiEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
                auto,
                reason: reason.to_string(),
            },
        }
    }
}
