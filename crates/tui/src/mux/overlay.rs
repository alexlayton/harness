use super::directory::{MAX_DIRECTORY_SUGGESTIONS, basename, expand_path};
use super::{MuxAction, MuxStatus, MuxUi, WorkspaceChoice};
use crate::render;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Overlay {
    NewMenu {
        selected: usize,
    },
    Worktree {
        branch: String,
        keep: bool,
    },
    Current {
        name: String,
    },
    Directory {
        input: String,
        suggestions: Vec<PathBuf>,
        selected: usize,
    },
    Close,
    ConfirmExit,
    ConfirmDuplicate {
        choice: WorkspaceChoice,
    },
    Help,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DirectoryRequest {
    pub(super) generation: u64,
    pub(super) input: String,
}

pub(super) struct DirectoryCompletion {
    pub(super) generation: u64,
    pub(super) input: String,
    pub(super) suggestions: Vec<PathBuf>,
}

impl MuxUi {
    pub(super) fn request_create(&mut self, choice: WorkspaceChoice, out: &mut Vec<MuxAction>) {
        let duplicate = match &choice {
            WorkspaceChoice::Directory { path, .. } => {
                std::fs::canonicalize(path).ok().is_some_and(|path| {
                    self.slots.iter().any(|slot| {
                        // A starting worktree temporarily displays the launch
                        // directory until Git returns its real path; do not
                        // treat that provisional label as a duplicate.
                        (!slot.worktree || slot.status != MuxStatus::Starting)
                            && std::fs::canonicalize(&slot.workspace).ok().as_ref() == Some(&path)
                    })
                })
            }
            WorkspaceChoice::Worktree { .. } => false,
        };
        if duplicate {
            self.overlay = Some(Overlay::ConfirmDuplicate { choice });
        } else {
            out.push(MuxAction::Create {
                choice,
                inherit_from: self.selected_id(),
            });
        }
    }

    pub(super) fn open_directory_overlay(&mut self) {
        self.directory_selection_explicit = false;
        self.overlay = Some(Overlay::Directory {
            input: String::new(),
            suggestions: vec![],
            selected: 0,
        });
        self.queue_directory_completion(String::new());
    }

    pub(super) fn queue_directory_completion(&mut self, input: String) {
        self.directory_generation = self.directory_generation.wrapping_add(1);
        self.directory_request = Some(DirectoryRequest {
            generation: self.directory_generation,
            input,
        });
    }

    pub(super) fn handle_overlay(
        &mut self,
        mut overlay: Overlay,
        key: KeyEvent,
        out: &mut Vec<MuxAction>,
    ) {
        if key.code == KeyCode::Esc {
            if matches!(overlay, Overlay::Directory { .. }) {
                self.invalidate_directory_completion();
            }
            if matches!(
                overlay,
                Overlay::Worktree { .. } | Overlay::Current { .. } | Overlay::Directory { .. }
            ) {
                self.overlay = Some(Overlay::NewMenu { selected: 0 })
            }
            return;
        }
        match &mut overlay {
            Overlay::Help => self.overlay = Some(overlay),
            Overlay::NewMenu { selected } => match key.code {
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    self.overlay = Some(overlay);
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(2);
                    self.overlay = Some(overlay);
                }
                KeyCode::Enter => match *selected {
                    0 => {
                        self.overlay = Some(Overlay::Worktree {
                            branch: String::new(),
                            keep: true,
                        })
                    }
                    1 => {
                        self.overlay = Some(Overlay::Current {
                            name: basename(&self.cwd),
                        })
                    }
                    _ => self.open_directory_overlay(),
                },
                _ => self.overlay = Some(overlay),
            },
            Overlay::Worktree { branch, keep } => match key.code {
                KeyCode::Enter if !branch.trim().is_empty() => self.request_create(
                    WorkspaceChoice::Worktree {
                        branch: branch.trim().into(),
                        keep: *keep,
                    },
                    out,
                ),
                KeyCode::Tab => {
                    *keep = !*keep;
                    self.overlay = Some(overlay);
                }
                _ => {
                    edit_string(branch, key);
                    self.overlay = Some(overlay)
                }
            },
            Overlay::Current { name } => match key.code {
                KeyCode::Enter if !name.trim().is_empty() => self.request_create(
                    WorkspaceChoice::Directory {
                        path: self.cwd.clone(),
                        name: name.trim().into(),
                    },
                    out,
                ),
                _ => {
                    edit_string(name, key);
                    self.overlay = Some(overlay)
                }
            },
            Overlay::Directory {
                input,
                suggestions,
                selected,
            } => match key.code {
                KeyCode::Tab => {
                    if let Some(p) = suggestions.get(*selected) {
                        *input = p.display().to_string();
                        suggestions.clear();
                        *selected = 0;
                        self.directory_selection_explicit = false;
                        self.queue_directory_completion(input.clone());
                    }
                    self.overlay = Some(overlay)
                }
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    self.directory_selection_explicit = true;
                    self.overlay = Some(overlay)
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(suggestions.len().saturating_sub(1));
                    self.directory_selection_explicit = true;
                    self.overlay = Some(overlay)
                }
                KeyCode::Enter => {
                    let typed = expand_path(input, &self.cwd);
                    let p = if self.directory_selection_explicit || !typed.is_dir() {
                        suggestions.get(*selected).cloned().unwrap_or(typed)
                    } else {
                        typed
                    };
                    if p.is_dir() {
                        self.request_create(
                            WorkspaceChoice::Directory {
                                name: basename(&p),
                                path: p,
                            },
                            out,
                        )
                    } else {
                        self.overlay = Some(overlay)
                    }
                }
                _ => {
                    let old_input = input.clone();
                    edit_string(input, key);
                    if *input != old_input {
                        suggestions.clear();
                        *selected = 0;
                        self.directory_selection_explicit = false;
                        self.queue_directory_completion(input.clone());
                    }
                    self.overlay = Some(overlay)
                }
            },
            Overlay::Close => {
                if key.code == KeyCode::Enter {
                    if let Some(id) = self.selected_id() {
                        out.push(MuxAction::Close { id })
                    }
                } else {
                    self.overlay = Some(overlay)
                }
            }
            Overlay::ConfirmExit => {
                if key.code == KeyCode::Enter {
                    out.push(MuxAction::Exit)
                } else {
                    self.overlay = Some(overlay)
                }
            }
            Overlay::ConfirmDuplicate { choice } => {
                if key.code == KeyCode::Enter {
                    out.push(MuxAction::Create {
                        choice: choice.clone(),
                        inherit_from: self.selected_id(),
                    });
                } else {
                    self.overlay = Some(overlay)
                }
            }
        }
    }

    pub(super) fn paste_into_overlay(&mut self, text: &str) {
        let text = render::sanitize_terminal_text(text).replace('\n', "");
        if text.is_empty() {
            return;
        }
        let mut directory_input = None;
        match self.overlay.as_mut() {
            Some(Overlay::Worktree { branch, .. }) => branch.push_str(&text),
            Some(Overlay::Current { name }) => name.push_str(&text),
            Some(Overlay::Directory {
                input,
                suggestions,
                selected,
            }) => {
                input.push_str(&text);
                suggestions.clear();
                *selected = 0;
                self.directory_selection_explicit = false;
                directory_input = Some(input.clone());
            }
            _ => {}
        }
        if let Some(input) = directory_input {
            self.queue_directory_completion(input);
        }
    }

    pub(super) fn invalidate_directory_completion(&mut self) {
        self.directory_generation = self.directory_generation.wrapping_add(1);
        self.directory_request = None;
    }

    pub(super) fn apply_directory_completion(&mut self, completion: DirectoryCompletion) -> bool {
        if completion.generation != self.directory_generation {
            return false;
        }
        let Some(Overlay::Directory {
            input,
            suggestions,
            selected,
        }) = &mut self.overlay
        else {
            return false;
        };
        if *input != completion.input {
            return false;
        }
        *suggestions = completion
            .suggestions
            .into_iter()
            .take(MAX_DIRECTORY_SUGGESTIONS)
            .collect();
        *selected = 0;
        self.directory_selection_explicit = false;
        true
    }
}

fn edit_string(s: &mut String, key: KeyEvent) {
    match key.code {
        KeyCode::Backspace => {
            s.pop();
        }
        KeyCode::Char(c)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) =>
        {
            s.push(c)
        }
        _ => {}
    }
}
