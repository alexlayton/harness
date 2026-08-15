//! Completion state and scanning: slash-command candidates, `@` file
//! references, and command-argument paths. All completion is debounced and
//! cancellable so rapid typing cannot queue unbounded blocking scans.

use crate::InputMessage;
use crate::app::textarea_with_text_at_cursor;
use crate::attachments;
use crate::commands::{self, Candidate, CandidateKind, CompletionContext};
use crate::render;
use anyhow::Result;
use crossterm::event::{Event, KeyCode};
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Debounce before starting a filesystem scan. Typing continues at full
/// speed; only the most recent keystroke may trigger a scan, so a burst of
/// typing performs one scan instead of saturating the blocking pool.
const FILE_SCAN_DEBOUNCE: Duration = Duration::from_millis(200);

#[derive(Clone, Debug)]
pub(crate) struct Completion {
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) selected: usize,
    pub(crate) offset: usize,
    pub(crate) kind: CompletionKind,
    pub(crate) context: Option<CompletionContext>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompletionKind {
    Slash,
    Files,
    Paths,
}

#[derive(Debug)]
pub(crate) struct FileCompletionResult {
    pub(crate) generation: u64,
    pub(crate) query: String,
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) kind: CompletionKind,
    pub(crate) context: Option<CompletionContext>,
}

impl crate::Tui {
    pub(crate) fn handle_completion_input(
        &mut self,
        event: &Event,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<bool> {
        let Some(completion) = self.completion.as_ref() else {
            return Ok(false);
        };
        let Event::Key(key) = event else {
            return Ok(false);
        };
        match key.code {
            KeyCode::Up => {
                if !completion.candidates.is_empty() {
                    self.move_completion(-1);
                }
                Ok(true)
            }
            KeyCode::Down => {
                if !completion.candidates.is_empty() {
                    self.move_completion(1);
                }
                Ok(true)
            }
            KeyCode::Tab => {
                self.accept_completion(input_tx)?;
                Ok(true)
            }
            KeyCode::Esc => {
                self.close_completion();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn move_completion(&mut self, direction: isize) {
        let Some(completion) = self.completion.as_mut() else {
            return;
        };
        let length = completion.candidates.len();
        if length == 0 {
            return;
        }
        completion.selected = if direction < 0 {
            if completion.selected == 0 {
                length - 1
            } else {
                completion.selected - 1
            }
        } else {
            (completion.selected + 1) % length
        };
        self.keep_completion_visible();
    }

    fn keep_completion_visible(&mut self) {
        let Some(completion) = self.completion.as_mut() else {
            return;
        };
        let visible = render::MAX_COMPLETION_ROWS;
        if completion.selected < completion.offset {
            completion.offset = completion.selected;
        } else if completion.selected >= completion.offset + visible {
            completion.offset = completion.selected + 1 - visible;
        }
        completion.offset = completion
            .offset
            .min(completion.candidates.len().saturating_sub(visible));
    }

    fn accept_completion(&mut self, input_tx: &mpsc::UnboundedSender<InputMessage>) -> Result<()> {
        let Some((candidate, kind, context)) = self.completion.as_ref().and_then(|completion| {
            completion
                .candidates
                .get(completion.selected)
                .cloned()
                .map(|candidate| (candidate, completion.kind, completion.context.clone()))
        }) else {
            return Ok(());
        };

        if kind == CompletionKind::Files {
            self.accept_file_completion(&candidate);
            return Ok(());
        }
        let Some(context) = context else {
            return Ok(());
        };

        let (line_index, cursor_col) = self.textarea.cursor();
        let Some(line) = self.textarea.lines().get(line_index) else {
            return Ok(());
        };
        let Some((replacement, new_cursor_col)) =
            commands::apply_completion(line, cursor_col, &context, &candidate)
        else {
            return Ok(());
        };
        let mut lines = self.textarea.lines().to_vec();
        lines[line_index] = replacement;
        let text = lines.join("\n");
        self.textarea = textarea_with_text_at_cursor(&text, line_index, new_cursor_col);

        let no_argument_command = matches!(context.target, commands::CompletionTarget::Command)
            && commands::command_spec(&candidate.value)
                .is_some_and(|command| command.argument_kind == commands::ArgumentKind::None);
        let provider_to_fetch = candidate
            .value
            .strip_suffix(':')
            .filter(|provider| {
                self.providers
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(provider))
            })
            .map(str::to_owned);

        self.refresh_completion();
        self.request_session_completion(input_tx)?;
        if no_argument_command {
            self.close_completion();
            return Ok(());
        }

        if let Some(provider) = provider_to_fetch
            && !self.model_lists.contains_key(&provider)
        {
            input_tx
                .send(InputMessage::ListModels {
                    provider: provider.clone(),
                })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
            if self.completion.is_none() {
                self.completion = Some(Completion {
                    candidates: Vec::new(),
                    selected: 0,
                    offset: 0,
                    kind: CompletionKind::Slash,
                    context: self.current_command_context(),
                });
            }
        }
        Ok(())
    }

    fn accept_file_completion(&mut self, candidate: &Candidate) {
        let (line_index, cursor_col) = self.textarea.cursor();
        let Some(line) = self.textarea.lines().get(line_index) else {
            return;
        };
        let is_directory = candidate.kind == CandidateKind::Directory;
        let Some((replacement, new_cursor_col)) =
            attachments::replace_at_token(line, cursor_col, &candidate.value, is_directory)
        else {
            return;
        };
        let mut lines = self.textarea.lines().to_vec();
        lines[line_index] = replacement;
        let text = lines.join("\n");
        self.textarea = textarea_with_text_at_cursor(&text, line_index, new_cursor_col);
        self.close_completion();
        self.refresh_completion();
    }

    pub(crate) fn refresh_completion(&mut self) {
        let input = self.textarea.lines().join("\n");
        if commands::is_command_input(&input) {
            let old_value = self.completion.as_ref().and_then(|completion| {
                completion
                    .candidates
                    .get(completion.selected)
                    .map(|candidate| candidate.value.clone())
            });
            self.cancel_file_completion();
            let (_, cursor_col) = self.textarea.cursor();
            let Some(result) = commands::candidates_at_cursor(
                &input,
                cursor_col,
                &self.providers,
                &self.model_lists,
                &self.provider,
                &self.session_candidates,
            ) else {
                self.completion = None;
                return;
            };

            let path_completion = matches!(
                result.context.target,
                commands::CompletionTarget::Argument(
                    commands::ArgumentKind::Session | commands::ArgumentKind::Path
                )
            );
            if path_completion {
                self.request_command_path_completion(result.context, result.candidates, old_value);
                return;
            }

            if result.candidates.is_empty() {
                self.completion = None;
                return;
            }
            let selected = old_value
                .as_deref()
                .and_then(|value| {
                    result
                        .candidates
                        .iter()
                        .position(|candidate| candidate.value == value)
                })
                .unwrap_or(0)
                .min(result.candidates.len().saturating_sub(1));
            self.completion = Some(Completion {
                candidates: result.candidates,
                selected,
                offset: 0,
                kind: CompletionKind::Slash,
                context: Some(result.context),
            });
            self.keep_completion_visible();
            return;
        }

        self.session_completion_requested = false;
        let (line_index, cursor_col) = self.textarea.cursor();
        let Some(line) = self.textarea.lines().get(line_index) else {
            self.close_completion();
            return;
        };
        let Some(prefix) = attachments::extract_at_prefix(line, cursor_col) else {
            self.close_completion();
            return;
        };
        self.request_file_completion(prefix.query);
    }

    pub(crate) fn request_session_completion(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let Some(context) = self.current_command_context() else {
            self.session_completion_requested = false;
            return Ok(());
        };
        let is_session_context = matches!(
            context.target,
            commands::CompletionTarget::Argument(commands::ArgumentKind::Session)
        );
        if !is_session_context {
            self.session_completion_requested = false;
            return Ok(());
        }
        if self.session_candidates.is_empty() && !self.session_completion_requested {
            input_tx
                .send(InputMessage::ListSessions)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
            self.session_completion_requested = true;
        }
        Ok(())
    }

    fn current_command_context(&self) -> Option<CompletionContext> {
        let input = self.textarea.lines().join("\n");
        let (_, cursor_col) = self.textarea.cursor();
        commands::completion_context(&input, cursor_col)
    }

    fn request_file_completion(&mut self, query: String) {
        if self.file_completion_query.as_deref() == Some(query.as_str())
            && self
                .completion
                .as_ref()
                .is_some_and(|completion| completion.kind == CompletionKind::Files)
        {
            return;
        }

        self.cancel_file_completion();
        let generation = self.file_completion_generation;
        let cancel = CancellationToken::new();
        let scan_cancel = cancel.clone();
        let root = self.workspace_root.clone();
        let sender = self.file_completion_tx.clone();
        let scan_query = query.clone();
        self.file_completion_cancel = Some(cancel.clone());
        self.file_completion_query = Some(query.clone());
        self.completion = Some(Completion {
            candidates: Vec::new(),
            selected: 0,
            offset: 0,
            kind: CompletionKind::Files,
            context: None,
        });

        tokio::spawn(async move {
            tokio::time::sleep(FILE_SCAN_DEBOUNCE).await;
            if cancel.is_cancelled() {
                return;
            }
            let candidates = tokio::task::spawn_blocking(move || {
                attachments::find_candidates(&root, &scan_query, &scan_cancel)
            })
            .await
            .unwrap_or_default();
            if cancel.is_cancelled() {
                return;
            }
            let _ = sender.send(FileCompletionResult {
                generation,
                query,
                candidates,
                kind: CompletionKind::Files,
                context: None,
            });
        });
    }

    fn request_command_path_completion(
        &mut self,
        context: CompletionContext,
        initial_candidates: Vec<Candidate>,
        preferred_value: Option<String>,
    ) {
        let query = context.query.clone();
        if self.file_completion_query.as_deref() == Some(query.as_str())
            && self.completion.as_ref().is_some_and(|completion| {
                completion.kind == CompletionKind::Paths
                    && completion.context.as_ref() == Some(&context)
            })
        {
            return;
        }

        self.cancel_file_completion();
        let generation = self.file_completion_generation;
        let cancel = CancellationToken::new();
        let scan_cancel = cancel.clone();
        let root = self.workspace_root.clone();
        let sender = self.file_completion_tx.clone();
        let scan_query = query.clone();
        let selected = preferred_value
            .as_deref()
            .and_then(|value| {
                initial_candidates
                    .iter()
                    .position(|candidate| candidate.value == value)
            })
            .unwrap_or(0)
            .min(initial_candidates.len().saturating_sub(1));
        self.file_completion_cancel = Some(cancel.clone());
        self.file_completion_query = Some(query.clone());
        self.completion = Some(Completion {
            candidates: initial_candidates,
            selected,
            offset: 0,
            kind: CompletionKind::Paths,
            context: Some(context.clone()),
        });
        self.keep_completion_visible();

        tokio::spawn(async move {
            tokio::time::sleep(FILE_SCAN_DEBOUNCE).await;
            if cancel.is_cancelled() {
                return;
            }
            let candidates = tokio::task::spawn_blocking(move || {
                attachments::find_path_candidates(&root, &scan_query, &scan_cancel)
            })
            .await
            .unwrap_or_default();
            if cancel.is_cancelled() {
                return;
            }
            let _ = sender.send(FileCompletionResult {
                generation,
                query,
                candidates,
                kind: CompletionKind::Paths,
                context: Some(context),
            });
        });
    }

    pub(crate) fn apply_file_completion(&mut self, result: FileCompletionResult) {
        if result.generation != self.file_completion_generation {
            return;
        }
        let (line_index, cursor_col) = self.textarea.cursor();
        let Some(line) = self.textarea.lines().get(line_index) else {
            return;
        };

        if result.kind == CompletionKind::Files {
            let Some(prefix) = attachments::extract_at_prefix(line, cursor_col) else {
                return;
            };
            if prefix.query != result.query {
                return;
            }
        } else {
            let Some(context) = commands::completion_context(line, cursor_col) else {
                return;
            };
            if result.context.as_ref() != Some(&context) {
                return;
            }
        }

        self.file_completion_cancel = None;
        let old_value = self.completion.as_ref().and_then(|completion| {
            completion
                .candidates
                .get(completion.selected)
                .map(|candidate| candidate.value.clone())
        });
        let candidates = if result.kind == CompletionKind::Paths {
            let static_candidates = commands::candidates_at_cursor(
                line,
                cursor_col,
                &self.providers,
                &self.model_lists,
                &self.provider,
                &self.session_candidates,
            )
            .map(|result| result.candidates)
            .unwrap_or_default();
            merge_candidates(static_candidates, result.candidates)
        } else {
            result.candidates
        };
        if candidates.is_empty() {
            self.completion = None;
            return;
        }
        let selected = old_value
            .as_deref()
            .and_then(|value| {
                candidates
                    .iter()
                    .position(|candidate| candidate.value == value)
            })
            .unwrap_or(0)
            .min(candidates.len().saturating_sub(1));
        self.completion = Some(Completion {
            candidates,
            selected,
            offset: 0,
            kind: result.kind,
            context: result.context,
        });
        self.keep_completion_visible();
    }

    pub(crate) fn cancel_file_completion(&mut self) {
        if let Some(cancel) = self.file_completion_cancel.take() {
            cancel.cancel();
        }
        self.file_completion_query = None;
        self.file_completion_generation = self.file_completion_generation.wrapping_add(1);
        if self.completion.as_ref().is_some_and(|completion| {
            matches!(
                completion.kind,
                CompletionKind::Files | CompletionKind::Paths
            )
        }) {
            self.completion = None;
        }
    }

    pub(crate) fn close_completion(&mut self) {
        self.cancel_file_completion();
        self.completion = None;
    }
}

fn merge_candidates(primary: Vec<Candidate>, secondary: Vec<Candidate>) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    primary
        .into_iter()
        .chain(secondary)
        .filter(|candidate| seen.insert(candidate.value.clone()))
        .collect()
}
