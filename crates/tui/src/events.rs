//! Event-to-state mapping: agent events become transcript mutations, and
//! input-level actions (command submission, history navigation, tool focus)
//! translate user intent into state changes and agent messages.

use crate::app::{new_textarea, textarea_with_text};
use crate::commands::{self, ParsedCommand};
use crate::input::{history_next, history_previous};
use crate::state::{EntryId, Focus, ToolRecord, ToolStatus, TranscriptEntry};
use crate::{InputMessage, SessionSnapshotEntry, UiEvent};
use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyModifiers};
use tokio::sync::mpsc;

impl crate::Tui {
    pub(crate) fn handle_tool_focus(&mut self, event: &Event) -> Result<bool> {
        let Event::Key(key) = event else {
            self.focused_tool = None;
            self.focus = Focus::Prompt;
            return Ok(false);
        };
        let indices = self.tool_indices();
        if indices.is_empty() {
            self.focused_tool = None;
            self.focus = Focus::Prompt;
            return Ok(false);
        }
        let current_position = self
            .focused_tool
            .and_then(|index| indices.iter().position(|candidate| *candidate == index))
            .unwrap_or(0);
        match key.code {
            KeyCode::Esc => {
                self.focused_tool = None;
                self.focus = Focus::Prompt;
                Ok(true)
            }
            KeyCode::Up => {
                let position = current_position.saturating_sub(1);
                self.focused_tool = Some(indices[position]);
                Ok(true)
            }
            KeyCode::Down => {
                let position = (current_position + 1).min(indices.len().saturating_sub(1));
                self.focused_tool = Some(indices[position]);
                Ok(true)
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.focused_tool = Some(indices[current_position]);
                self.toggle_tool_at(indices[current_position]);
                Ok(true)
            }
            KeyCode::Char('o')
                if key.modifiers.contains(KeyModifiers::CONTROL) || key.modifiers.is_empty() =>
            {
                self.toggle_all_tools();
                Ok(true)
            }
            KeyCode::Tab => {
                self.focused_tool = Some(indices[(current_position + 1) % indices.len()]);
                Ok(true)
            }
            _ => {
                self.focused_tool = None;
                self.focus = Focus::Prompt;
                Ok(false)
            }
        }
    }

    pub(crate) fn handle_navigation_edit(&mut self, event: &Event) -> bool {
        let Event::Key(key) = event else {
            return false;
        };
        match key.code {
            KeyCode::Up => {
                let (line, _) = self.textarea.cursor();
                if line > 0 {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event.clone());
                    return true;
                }
                if self.navigate_history_up() {
                    return true;
                }
                true
            }
            KeyCode::Down => {
                let (line, _) = self.textarea.cursor();
                if line + 1 < self.textarea.lines().len() {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event.clone());
                    return true;
                }
                if self.navigate_history_down() {
                    return true;
                }
                true
            }
            _ => false,
        }
    }

    fn navigate_history_up(&mut self) -> bool {
        if commands::is_command_input(&self.textarea.lines().join("\n")) {
            return false;
        }
        let (line, _) = self.textarea.cursor();
        if self.textarea.lines().len() > 1 || line != 0 || self.history_pos == Some(0) {
            return false;
        }
        let current = self.textarea.lines().join("\n");
        let Some(value) = history_previous(
            &self.history,
            &mut self.history_pos,
            &mut self.draft,
            &current,
        ) else {
            return false;
        };
        self.replace_textarea(&value);
        true
    }

    fn navigate_history_down(&mut self) -> bool {
        if commands::is_command_input(&self.textarea.lines().join("\n")) {
            return false;
        }
        let (line, _) = self.textarea.cursor();
        if self.textarea.lines().len() > 1 || line + 1 < self.textarea.lines().len() {
            return false;
        }
        let Some(value) = history_next(&self.history, &mut self.history_pos, &self.draft) else {
            return false;
        };
        self.replace_textarea(&value);
        true
    }

    fn replace_textarea(&mut self, value: &str) {
        self.textarea = textarea_with_text(value);
    }

    fn push_history_and_clear_input(&mut self, input: &str) {
        crate::input::push_history(&mut self.history, input, crate::app::MAX_HISTORY);
        self.history_pos = None;
        self.draft.clear();
        self.textarea = new_textarea();
        self.close_completion();
        self.focus = Focus::Prompt;
    }

    pub(crate) fn submit_command(
        &mut self,
        input: &str,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let command = match commands::parse_command(input) {
            Ok(command) => command,
            Err(error) => {
                self.add_error(error);
                return Ok(());
            }
        };
        if let ParsedCommand::SetModel {
            provider: None,
            model,
        } = &command
            && model.is_empty()
        {
            self.add_notice(format!(
                "usage: /model [<provider>:]<model> (current: {} · {})",
                self.provider, self.model
            ));
            return Ok(());
        }

        self.add_notice(format!("⌘ {input}"));
        self.push_history_and_clear_input(input);
        match command {
            ParsedCommand::New => input_tx
                .send(InputMessage::NewConversation)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Load { selector } => input_tx
                .send(InputMessage::LoadSession { selector })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Sessions => input_tx
                .send(InputMessage::ListSessions)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Export { destination } => input_tx
                .send(InputMessage::ExportSession { destination })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Compact => input_tx
                .send(InputMessage::CompactSession)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Auth => {
                self.add_notice(
                    "GitHub Copilot authentication is not wired up yet; run the agent once /auth is supported",
                );
            }
            ParsedCommand::SetModel { provider, model } => {
                let provider_for_fetch = provider.clone().and_then(|name| {
                    self.providers
                        .iter()
                        .find(|known| known.eq_ignore_ascii_case(&name))
                        .cloned()
                });
                input_tx
                    .send(InputMessage::SetModel { provider, model })
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                if let Some(provider) = provider_for_fetch
                    && !self.model_lists.contains_key(&provider)
                {
                    input_tx
                        .send(InputMessage::ListModels { provider })
                        .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn submit_message(
        &mut self,
        input: String,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let id = self.allocate_id();
        self.add_entry(TranscriptEntry::User {
            id,
            text: input.clone(),
        });
        self.push_history_and_clear_input(&input);
        input_tx
            .send(InputMessage::Message(input))
            .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        self.busy = true;
        self.activity = crate::app::Activity::Preparing;
        self.spinner = 0;
        self.retrying = None;
        Ok(())
    }

    pub(crate) fn apply_event(&mut self, event: UiEvent) -> Result<()> {
        match event {
            UiEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                self.busy = true;
                self.activity = crate::app::Activity::Working;
                self.retrying = None;
                if let TranscriptEntry::Assistant { markdown, .. } = self.ensure_assistant() {
                    markdown.push_str(&delta);
                }
                self.transcript_changed();
            }
            UiEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                self.busy = true;
                self.activity = crate::app::Activity::Reasoning;
                self.retrying = None;
                if let TranscriptEntry::Assistant { reasoning, .. } = self.ensure_assistant() {
                    reasoning.push_str(&delta);
                }
                self.transcript_changed();
            }
            UiEvent::ToolCallStarted {
                name,
                summary,
                arguments,
            } => {
                self.busy = true;
                self.activity = crate::app::Activity::Processing;
                self.retrying = None;
                self.streaming_assistant = None;
                let id = self.allocate_id();
                self.add_entry(TranscriptEntry::Tool {
                    id,
                    record: ToolRecord {
                        name,
                        args: arguments,
                        summary,
                        ok: false,
                        duration_ms: 0,
                        output: String::new(),
                        error: None,
                        status: ToolStatus::Running,
                    },
                    expanded: false,
                });
                self.running_tool = Some(id);
                self.spinner = 0;
            }
            UiEvent::ToolCallFinished {
                name,
                summary,
                ok,
                duration_ms,
                output,
                error,
            } => {
                self.busy = true;
                self.activity = crate::app::Activity::Working;
                let id = self
                    .running_tool
                    .take()
                    .unwrap_or_else(|| self.allocate_id());
                let mut updated = false;
                if let Some(entry) = self.transcript.iter_mut().find(|entry| entry.id() == id)
                    && let Some(record) = entry.tool_record_mut()
                {
                    record.name = name.clone();
                    record.summary = summary.clone();
                    record.ok = ok;
                    record.duration_ms = duration_ms;
                    record.output = output.clone();
                    record.error = error.clone();
                    record.status = if ok {
                        ToolStatus::Success
                    } else {
                        ToolStatus::Failure
                    };
                    updated = true;
                }
                if !updated {
                    self.add_entry(TranscriptEntry::Tool {
                        id,
                        record: ToolRecord {
                            name,
                            args: "{}".into(),
                            summary,
                            ok,
                            duration_ms,
                            output,
                            error,
                            status: if ok {
                                ToolStatus::Success
                            } else {
                                ToolStatus::Failure
                            },
                        },
                        expanded: false,
                    });
                } else {
                    self.transcript_changed();
                }
                self.focused_tool = None;
                self.retrying = None;
            }
            UiEvent::Retrying { attempt, .. } => {
                self.busy = true;
                self.activity = crate::app::Activity::Retrying;
                self.retrying = Some(attempt);
            }
            UiEvent::Error(error) => {
                self.session_completion_requested = false;
                self.add_error(error);
                self.running_tool = None;
                self.streaming_assistant = None;
                self.busy = false;
                self.activity = crate::app::Activity::Preparing;
                self.retrying = None;
            }
            UiEvent::TurnFinished => {
                if let Some(id) = self.streaming_assistant.take()
                    && let Some(entry) = self.transcript.iter_mut().find(|entry| entry.id() == id)
                    && let TranscriptEntry::Assistant { streaming, .. } = entry
                {
                    *streaming = false;
                }
                self.running_tool = None;
                self.retrying = None;
                self.busy = false;
                self.activity = crate::app::Activity::Preparing;
                self.focused_tool = None;
                self.focus = Focus::Prompt;
            }
            UiEvent::Notice(notice) => self.add_notice(notice),
            UiEvent::ModelChanged { provider, model } => {
                self.provider = provider;
                self.model = model;
                if commands::is_command_input(&self.textarea.lines().join("\n")) {
                    self.refresh_completion();
                }
            }
            UiEvent::ModelList { provider, models } => {
                self.model_lists.insert(provider, models);
                if commands::is_command_input(&self.textarea.lines().join("\n")) {
                    self.refresh_completion();
                }
            }
            UiEvent::SessionChanged { id, title, loaded } => {
                self.session_candidates.clear();
                self.session_completion_requested = false;
                self.session_id = Some(id.clone());
                self.session_title = title;
                self.transcript.clear();
                self.streaming_assistant = None;
                self.running_tool = None;
                self.focused_tool = None;
                if loaded {
                    self.add_notice(format!("loaded session {}", &id[..id.len().min(8)]));
                }
            }
            UiEvent::SessionSnapshot { entries } => {
                self.transcript.clear();
                self.streaming_assistant = None;
                self.running_tool = None;
                self.focused_tool = None;
                for snapshot in entries {
                    let id = self.allocate_id();
                    let entry = match snapshot {
                        SessionSnapshotEntry::User { text } => TranscriptEntry::User { id, text },
                        SessionSnapshotEntry::Assistant {
                            markdown,
                            reasoning,
                        } => TranscriptEntry::Assistant {
                            id,
                            markdown,
                            reasoning,
                            streaming: false,
                        },
                        SessionSnapshotEntry::Tool {
                            name,
                            summary,
                            arguments,
                            ok,
                            duration_ms,
                            output,
                            error,
                        } => TranscriptEntry::Tool {
                            id,
                            record: ToolRecord {
                                name,
                                args: arguments,
                                summary,
                                ok,
                                duration_ms,
                                output,
                                error,
                                status: if ok {
                                    ToolStatus::Success
                                } else {
                                    ToolStatus::Failure
                                },
                            },
                            expanded: false,
                        },
                    };
                    self.transcript.push(entry);
                }
                self.transcript_changed();
            }
            UiEvent::SessionList { sessions } => {
                self.session_candidates = sessions.clone();
                self.session_completion_requested = false;
                let notice = if sessions.is_empty() {
                    "No sessions for this workspace".to_owned()
                } else {
                    sessions
                        .into_iter()
                        .take(12)
                        .map(|session| {
                            let title = session.title.unwrap_or_else(|| "(untitled)".into());
                            let model = session.model.unwrap_or_else(|| "(model unknown)".into());
                            format!(
                                "{} · {} · {} · {}",
                                session.short_id, title, model, session.updated_at
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                self.add_notice(notice);
                if commands::is_command_input(&self.textarea.lines().join("\n")) {
                    self.refresh_completion();
                }
            }
            UiEvent::SessionExported { path } => {
                self.add_notice(format!("exported session to {path}"));
            }
            UiEvent::UsageUpdated { .. } => {}
            UiEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
                auto,
                reason,
            } => self.add_notice(format!(
                "{}compacted through event {compacted_through} ({summary_bytes} bytes) [{reason}]",
                if auto { "auto-" } else { "" }
            )),
        }
        Ok(())
    }

    fn ensure_assistant(&mut self) -> &mut TranscriptEntry {
        let id = if let Some(id) = self.streaming_assistant {
            id
        } else {
            let id = self.allocate_id();
            self.add_entry(TranscriptEntry::Assistant {
                id,
                markdown: String::new(),
                reasoning: String::new(),
                streaming: true,
            });
            self.streaming_assistant = Some(id);
            id
        };
        self.transcript
            .iter_mut()
            .find(|entry| entry.id() == id)
            .expect("streaming assistant entry exists")
    }

    fn add_notice(&mut self, text: impl Into<String>) {
        let id = self.allocate_id();
        self.add_entry(TranscriptEntry::Notice {
            id,
            text: text.into(),
        });
    }

    fn add_error(&mut self, text: impl Into<String>) {
        let id = self.allocate_id();
        self.add_entry(TranscriptEntry::Error {
            id,
            text: text.into(),
        });
    }

    fn add_entry(&mut self, entry: TranscriptEntry) {
        self.transcript.push(entry);
        self.transcript_changed();
    }

    fn allocate_id(&mut self) -> EntryId {
        let id = self.next_entry_id;
        self.next_entry_id = self.next_entry_id.wrapping_add(1).max(1);
        id
    }

    fn transcript_changed(&mut self) {
        // The alternate-screen scroll machinery is gone. With native
        // scrollback, transcript mutations need no bookkeeping here; the
        // Phase 2 commit pipeline decides what enters scrollback.
    }

    pub(crate) fn tool_indices(&self) -> Vec<usize> {
        self.transcript
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.tool_record().map(|_| index))
            .collect()
    }

    pub(crate) fn toggle_all_tools(&mut self) {
        // Expand every tool when at least one is collapsed, otherwise collapse
        // them all. This makes Ctrl+O a toggle across the whole transcript
        // regardless of focus.
        let expand = crate::state::toggle_all_direction(&self.transcript);
        for entry in &mut self.transcript {
            if let TranscriptEntry::Tool { expanded, .. } = entry {
                *expanded = expand;
            }
        }
        self.transcript_changed();
        self.focused_tool = None;
    }

    fn toggle_tool_at(&mut self, index: usize) {
        if let Some(TranscriptEntry::Tool { expanded, .. }) = self.transcript.get_mut(index) {
            *expanded = !*expanded;
            self.transcript_changed();
        }
    }
}
