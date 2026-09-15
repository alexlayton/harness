//! Full-screen, host-neutral multi-agent terminal UI.
//!
//! The mux owns the terminal exactly once and keeps an [`AgentPane`] per slot;
//! it never creates agents, worktrees, or sessions itself. Those lifecycle
//! requests are returned to the host as [`MuxAction`] values.

use crate::app::line_to_ansi;
use crate::render;
use crate::{AgentPane, InputMessage, UiEvent};
use anyhow::{Context, Result};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    MouseButton, MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    self, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui_core::buffer::Buffer;
use ratatui_core::layout::Rect;
use ratatui_core::style::{Modifier, Style};
use ratatui_core::text::{Line, Span};
use std::fmt::Write as _;
use std::io::{self, Stdout, Write};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Stable host-assigned identifier. IDs are never derived from sidebar order.
pub type MuxId = u64;

/// Host-reported lifecycle state displayed beside a mux slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuxStatus {
    /// The host is creating the agent and its resources.
    Starting,
    /// The agent is actively processing a turn.
    Running,
    /// The agent is ready for input.
    Idle,
    /// The agent stopped because of an error.
    Error,
}

impl MuxStatus {
    fn glyph(self) -> &'static str {
        match self {
            Self::Starting => "◐",
            Self::Running => "●",
            Self::Idle => "✓",
            Self::Error => "!",
        }
    }
}

/// Workspace requested by the mux's new-agent flow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceChoice {
    /// Ask the host to create a worktree from HEAD. `keep` defaults true in UI.
    Worktree { branch: String, keep: bool },
    /// Run in an existing directory with a user-visible session name.
    Directory {
        /// Absolute or current-workspace-relative directory selected by the user.
        path: PathBuf,
        /// User-visible slot name.
        name: String,
    },
}

/// Lifecycle/editor output from mux to its host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MuxAction {
    AgentInput {
        id: MuxId,
        input: InputMessage,
    },
    /// Create a slot, inheriting mutable model settings from the slot that was
    /// selected when the dialog was confirmed.
    Create {
        choice: WorkspaceChoice,
        inherit_from: Option<MuxId>,
    },
    Close {
        id: MuxId,
    },
    Exit,
}

/// Input from the host. Agent wire formats deliberately do not appear here.
#[allow(clippy::large_enum_variant)]
pub enum MuxEvent {
    Add {
        id: MuxId,
        name: String,
        workspace: PathBuf,
        worktree: bool,
        status: MuxStatus,
        pane: AgentPane,
    },
    /// Replace an existing provisional pane without changing the active slot.
    Replace {
        id: MuxId,
        name: String,
        workspace: PathBuf,
        worktree: bool,
        status: MuxStatus,
        pane: AgentPane,
    },
    Ui {
        id: MuxId,
        event: UiEvent,
    },
    Status {
        id: MuxId,
        status: MuxStatus,
    },
    Remove {
        id: MuxId,
    },
}

struct Slot {
    id: MuxId,
    name: String,
    workspace: PathBuf,
    worktree: bool,
    status: MuxStatus,
    pane: AgentPane,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Overlay {
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

/// Terminal rectangles computed by [`MuxUi::layout`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MuxLayout {
    /// Sidebar origin and width, or `None` when collapsed.
    pub sidebar: Option<(u16, u16)>,
    /// First column of the selected agent pane.
    pub main_x: u16,
    /// Width available to the selected agent pane.
    pub main_width: u16,
    /// Full height available to both the sidebar and selected agent pane.
    pub body_height: u16,
}

/// Stateful mux controller. Constructing it does not touch the terminal.
pub struct MuxUi {
    slots: Vec<Slot>,
    selected: Option<usize>,
    sidebar_scroll: usize,
    prefix: bool,
    overlay: Option<Overlay>,
    cwd: PathBuf,
    width: u16,
    height: u16,
    directory_generation: u64,
    directory_request: Option<DirectoryRequest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectoryRequest {
    generation: u64,
    input: String,
}

struct DirectoryCompletion {
    generation: u64,
    input: String,
    suggestions: Vec<PathBuf>,
}

impl MuxUi {
    /// Create a mux rooted at `cwd` without accessing the terminal.
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            slots: vec![],
            selected: None,
            sidebar_scroll: 0,
            prefix: false,
            overlay: None,
            cwd,
            width: 80,
            height: 24,
            directory_generation: 0,
            directory_request: None,
        }
    }
    /// Return the stable ID of the selected slot.
    pub fn selected_id(&self) -> Option<MuxId> {
        self.selected.and_then(|i| self.slots.get(i)).map(|s| s.id)
    }
    /// Return the number of slots retained by the mux.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Return whether the mux has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Compute host-neutral pane geometry for terminal dimensions.
    pub fn layout(&self, width: u16, height: u16) -> MuxLayout {
        let body_height = height;
        // The command strip needs a little more room than the original roster.
        // Hide it only when that minimum would leave no useful agent pane.
        let side = if width >= 36 {
            Some((0, (width / 3).clamp(24, 30)))
        } else {
            None
        };
        // Keep the sidebar border visually separate from pane content.
        let main_x = side.map_or(0, |(_, w)| w.saturating_add(3));
        MuxLayout {
            sidebar: side,
            main_x,
            main_width: width.saturating_sub(main_x),
            body_height,
        }
    }
    /// Apply a host lifecycle or agent UI event to retained mux state.
    pub fn apply(&mut self, event: MuxEvent) {
        match event {
            MuxEvent::Add {
                id,
                name,
                workspace,
                worktree,
                status,
                pane,
            } => {
                if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
                    *slot = Slot {
                        id,
                        name,
                        workspace,
                        worktree,
                        status,
                        pane,
                    };
                } else {
                    self.slots.push(Slot {
                        id,
                        name,
                        workspace,
                        worktree,
                        status,
                        pane,
                    });
                }
                self.selected = self.slots.iter().position(|s| s.id == id);
                self.reveal_selected();
            }
            MuxEvent::Replace {
                id,
                name,
                workspace,
                worktree,
                status,
                mut pane,
            } => {
                if let Some(slot) = self.slots.iter_mut().find(|slot| slot.id == id) {
                    // Workspace setup may finish after the user has started a
                    // draft in the provisional pane. Carry that editor state
                    // into the fully configured pane rather than discarding it.
                    pane.set_draft(slot.pane.editor_text());
                    *slot = Slot {
                        id,
                        name,
                        workspace,
                        worktree,
                        status,
                        pane,
                    };
                }
            }
            MuxEvent::Ui { id, event } => {
                if let Some(s) = self.slots.iter_mut().find(|s| s.id == id) {
                    s.pane.apply_event(event);
                }
            }
            MuxEvent::Status { id, status } => {
                if let Some(s) = self.slots.iter_mut().find(|s| s.id == id) {
                    s.status = status;
                }
            }
            MuxEvent::Remove { id } => {
                if let Some(i) = self.slots.iter().position(|s| s.id == id) {
                    self.slots.remove(i);
                    self.selected = if self.slots.is_empty() {
                        None
                    } else {
                        Some(i.min(self.slots.len() - 1))
                    };
                    self.reveal_selected();
                }
            }
        }
    }
    fn sidebar_footer_rows(&self) -> u16 {
        // The separator and command rows grow upward from the bottom border.
        if self.prefix && self.height >= 7 {
            6
        } else {
            2
        }
    }

    fn visible_rows(&self) -> usize {
        // Top border, leading spacer, spacer before New, New, trailing spacer,
        // footer, and bottom border are fixed; every session occupies one row.
        self.layout(self.width, self.height)
            .body_height
            .saturating_sub(6 + self.sidebar_footer_rows()) as usize
    }
    fn clamp_sidebar_scroll(&mut self) {
        self.sidebar_scroll = self
            .sidebar_scroll
            .min(self.slots.len().saturating_sub(self.visible_rows()));
    }
    fn reveal_selected(&mut self) {
        self.clamp_sidebar_scroll();
        if let Some(i) = self.selected {
            let n = self.visible_rows();
            if i < self.sidebar_scroll {
                self.sidebar_scroll = i
            } else if n > 0 && i >= self.sidebar_scroll + n {
                self.sidebar_scroll = i + 1 - n
            }
        }
        self.clamp_sidebar_scroll();
    }
    fn switch(&mut self, delta: isize) {
        if self.slots.is_empty() {
            return;
        }
        let i = self.selected.unwrap_or(0) as isize;
        self.selected = Some((i + delta).rem_euclid(self.slots.len() as isize) as usize);
        self.reveal_selected();
    }
    fn is_prefix(key: &KeyEvent) -> bool {
        // Crossterm reports Ctrl+Space as either NUL or a controlled space,
        // depending on terminal keyboard protocol support.
        matches!(key.code, KeyCode::Null)
            || (key.code == KeyCode::Char(' ') && key.modifiers.contains(KeyModifiers::CONTROL))
            || (key.code == KeyCode::Char('@') && key.modifiers.contains(KeyModifiers::CONTROL))
    }

    /// Route an input event. Non-prefix/non-overlay keys are delegated without
    /// reinterpretation to the selected `AgentPane`.
    pub fn handle(&mut self, event: Event) -> Result<Vec<MuxAction>> {
        let mut out = Vec::new();
        if let Event::Resize(w, h) = event {
            self.width = w;
            self.height = h;
            self.reveal_selected();
            return Ok(out);
        }
        if let Event::Mouse(mouse) = event {
            if matches!(self.overlay, Some(Overlay::NewMenu { .. })) {
                if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                    let frame = overlay_frame(
                        self.layout(self.width, self.height),
                        self.overlay.as_ref().expect("overlay exists"),
                    );
                    if mouse.column >= frame.x && mouse.column < frame.x + frame.width {
                        let relative = mouse.row.saturating_sub(frame.y) as usize;
                        // Choices occupy content rows 2, 4, and 6. Include the
                        // blank row below each as a forgiving hit target.
                        if (2..=7).contains(&relative) {
                            if let Some(Overlay::NewMenu { selected }) = &mut self.overlay {
                                *selected = ((relative - 2) / 2).min(2);
                            }
                            let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
                            let overlay = self.overlay.take().expect("overlay exists");
                            self.handle_overlay(overlay, enter, &mut out);
                        }
                    }
                }
            } else if self.overlay.is_none() {
                self.handle_mouse(mouse.kind, mouse.column, mouse.row);
            }
            return Ok(out);
        }
        if matches!(event, Event::Paste(_)) && self.overlay.is_none() && !self.prefix {
            return self.delegate(event);
        }
        let Event::Key(key) = event else {
            return Ok(out);
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return Ok(out);
        }
        if let Some(overlay) = self.overlay.take() {
            self.handle_overlay(overlay, key, &mut out);
            return Ok(out);
        }
        if self.prefix {
            self.prefix = false;
            match key.code {
                KeyCode::Esc => {}
                KeyCode::Char('n') => self.overlay = Some(Overlay::NewMenu { selected: 0 }),
                KeyCode::Char('w') => {
                    self.overlay = Some(Overlay::Worktree {
                        branch: String::new(),
                        keep: true,
                    })
                }
                KeyCode::Char('c') => {
                    self.overlay = Some(Overlay::Current {
                        name: basename(&self.cwd),
                    })
                }
                KeyCode::Char('d') => self.open_directory_overlay(),
                KeyCode::Char('j') => self.switch(1),
                KeyCode::Char('k') => self.switch(-1),
                KeyCode::Char('?') => self.overlay = Some(Overlay::Help),
                KeyCode::Char('x') if self.selected.is_some() => {
                    self.overlay = Some(Overlay::Close)
                }
                KeyCode::Char(c @ '1'..='9') => {
                    let i = (c as u8 - b'1') as usize;
                    if i < self.slots.len() {
                        self.selected = Some(i);
                        self.reveal_selected()
                    }
                }
                _ => {}
            }
            return Ok(out);
        }
        if Self::is_prefix(&key) {
            self.prefix = true;
            self.reveal_selected();
            return Ok(out);
        }
        match key.code {
            KeyCode::PageUp => {
                self.scroll_selected(self.layout(self.width, self.height).body_height as i32);
                Ok(out)
            }
            KeyCode::PageDown => {
                self.scroll_selected(-(self.layout(self.width, self.height).body_height as i32));
                Ok(out)
            }
            _ => self.delegate(Event::Key(key)),
        }
    }

    fn delegate(&mut self, event: Event) -> Result<Vec<MuxAction>> {
        let Some(i) = self.selected else {
            return Ok(vec![]);
        };
        // Keep drafts editable during setup, but do not submit into a runtime
        // that is not ready (or has failed). Shift/Alt+Enter still edits a
        // multiline draft through the normal pane handler.
        if matches!(self.slots[i].status, MuxStatus::Starting | MuxStatus::Error)
            && matches!(
                &event,
                Event::Key(KeyEvent {
                    code: KeyCode::Enter,
                    modifiers: KeyModifiers::NONE,
                    ..
                })
            )
        {
            return Ok(vec![]);
        }
        let id = self.slots[i].id;
        let result = self.slots[i].pane.handle_input(&event)?;
        let out = result
            .messages
            .into_iter()
            .map(|input| MuxAction::AgentInput { id, input })
            .collect::<Vec<_>>();
        if result.exit_requested {
            // A mux roster is process-local, so an accidental Ctrl+C/Ctrl+D
            // would discard every running agent. Require explicit confirmation.
            self.overlay = Some(Overlay::ConfirmExit);
            return Ok(vec![]);
        }
        Ok(out)
    }

    fn scroll_selected(&mut self, rows: i32) {
        if let Some(index) = self.selected {
            self.slots[index].pane.scroll(rows);
        }
    }

    fn handle_mouse(&mut self, kind: MouseEventKind, x: u16, y: u16) {
        let layout = self.layout(self.width, self.height);
        let sidebar_width = layout.sidebar.map(|(_, width)| width);
        if x >= layout.main_x {
            match kind {
                MouseEventKind::ScrollUp => self.scroll_selected(3),
                MouseEventKind::ScrollDown => self.scroll_selected(-3),
                _ => {}
            }
            return;
        }
        // The two cells between the sidebar border and main pane are inert.
        if sidebar_width.is_some_and(|width| x > width) {
            return;
        }
        match kind {
            MouseEventKind::ScrollUp => self.sidebar_scroll = self.sidebar_scroll.saturating_sub(1),
            MouseEventKind::ScrollDown => {
                self.sidebar_scroll += 1;
                self.clamp_sidebar_scroll();
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let visible_count = self
                    .slots
                    .len()
                    .saturating_sub(self.sidebar_scroll)
                    .min(self.visible_rows());
                let new_y = 3 + visible_count;
                if y as usize == new_y && y < layout.body_height.saturating_sub(1) {
                    self.overlay = Some(Overlay::NewMenu { selected: 0 });
                } else if (2..2 + visible_count).contains(&(y as usize)) {
                    let i = self.sidebar_scroll + y as usize - 2;
                    if i < self.slots.len() {
                        self.selected = Some(i)
                    }
                }
            }
            _ => {}
        }
    }
    fn request_create(&mut self, choice: WorkspaceChoice, out: &mut Vec<MuxAction>) {
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

    fn open_directory_overlay(&mut self) {
        self.overlay = Some(Overlay::Directory {
            input: String::new(),
            suggestions: vec![],
            selected: 0,
        });
        self.queue_directory_completion(String::new());
    }

    fn queue_directory_completion(&mut self, input: String) {
        self.directory_generation = self.directory_generation.wrapping_add(1);
        self.directory_request = Some(DirectoryRequest {
            generation: self.directory_generation,
            input,
        });
    }

    fn handle_overlay(&mut self, mut overlay: Overlay, key: KeyEvent, out: &mut Vec<MuxAction>) {
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
                        self.queue_directory_completion(input.clone());
                    }
                    self.overlay = Some(overlay)
                }
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    self.overlay = Some(overlay)
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(suggestions.len().saturating_sub(1));
                    self.overlay = Some(overlay)
                }
                KeyCode::Enter => {
                    let p = suggestions
                        .get(*selected)
                        .cloned()
                        .unwrap_or_else(|| expand_path(input, &self.cwd));
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
                    out.push(MuxAction::Exit);
                } else {
                    self.overlay = Some(overlay);
                }
            }
            Overlay::ConfirmDuplicate { choice } => {
                if key.code == KeyCode::Enter {
                    out.push(MuxAction::Create {
                        choice: choice.clone(),
                        inherit_from: self.selected_id(),
                    });
                } else {
                    self.overlay = Some(overlay);
                }
            }
        }
    }

    fn invalidate_directory_completion(&mut self) {
        self.directory_generation = self.directory_generation.wrapping_add(1);
        self.directory_request = None;
    }

    fn apply_directory_completion(&mut self, completion: DirectoryCompletion) -> bool {
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
        true
    }

    /// Run as the sole terminal owner. Restoration is attempted on normal
    /// return, cancellation, input failure, and panic (including aborting
    /// release builds, via the process panic hook).
    pub async fn run(
        mut self,
        mut events: mpsc::UnboundedReceiver<MuxEvent>,
        actions: mpsc::UnboundedSender<MuxAction>,
        cancel: CancellationToken,
    ) -> Result<()> {
        install_mux_panic_hook();
        let mut terminal = TerminalGuard::enter()?;
        let mut input = EventStream::new();
        let mut tick = tokio::time::interval(Duration::from_millis(100));
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel();
        let mut completion_task: Option<JoinHandle<()>> = None;
        self.draw(terminal.out())?;

        let result: Result<()> = async {
            loop {
                tokio::select! {
                    event = events.recv() => {
                        let Some(event) = event else { break };
                        self.apply(event);
                    }
                    event = input.next() => {
                        let Some(event) = event else { break };
                        for action in self.handle(event.context("read terminal event")?)? {
                            if actions.send(action).is_err() {
                                return Ok(());
                            }
                        }
                        if let Some(request) = self.directory_request.take() {
                            if let Some(task) = completion_task.take() {
                                task.abort();
                            }
                            let tx = completion_tx.clone();
                            let cwd = self.cwd.clone();
                            completion_task = Some(tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_millis(120)).await;
                                let input = request.input;
                                let scan_input = input.clone();
                                let suggestions = tokio::task::spawn_blocking(move || {
                                    directory_suggestions(&scan_input, &cwd)
                                }).await;
                                if let Ok(suggestions) = suggestions {
                                    let _ = tx.send(DirectoryCompletion {
                                        generation: request.generation,
                                        input,
                                        suggestions,
                                    });
                                }
                            }));
                        } else if !matches!(self.overlay, Some(Overlay::Directory { .. }))
                            && let Some(task) = completion_task.take()
                        {
                            task.abort();
                        }
                    }
                    Some(completion) = completion_rx.recv() => {
                        self.apply_directory_completion(completion);
                    }
                    _ = tick.tick() => {
                        let changed = self.slots.iter_mut().fold(false, |changed, slot| {
                            slot.pane.tick_completion() || changed
                        });
                        if !changed {
                            continue;
                        }
                    }
                    _ = cancel.cancelled() => {
                        let _ = actions.send(MuxAction::Exit);
                        break;
                    }
                }
                self.draw(terminal.out())?;
            }
            Ok(())
        }
        .await;
        if let Some(task) = completion_task {
            task.abort();
        }
        drop(terminal);
        result
    }

    /// Render a complete, fixed-height sidebar. Each returned row has exactly
    /// `sidebar_width + 1` cells: the right border is therefore also the
    /// constant boundary immediately before `MuxLayout::main_x`.
    fn sidebar_lines(
        &self,
        sidebar_width: u16,
        height: u16,
        muted: Style,
        accent: Style,
    ) -> Vec<Line<'static>> {
        let total_width = sidebar_width.saturating_add(1) as usize;
        if height == 0 || total_width == 0 {
            return vec![];
        }
        let border = |text: &str| Span::styled(text.to_owned(), muted);
        if total_width == 1 {
            let mut rows = (0..height)
                .map(|_| Line::from(border("│")))
                .collect::<Vec<_>>();
            rows[0] = Line::from(border("╭"));
            if height > 1 {
                rows[height as usize - 1] = Line::from(border("╰"));
            }
            return rows;
        }
        let interior = total_width.saturating_sub(2);
        let framed = |text: &str, style: Style| {
            Line::from(vec![
                border("│"),
                Span::styled(fit(text, interior), style),
                border("│"),
            ])
        };
        let ruled = |left: &str, label: &str, right: &str, style: Style| {
            let label = if label.width() <= interior {
                label.to_owned()
            } else {
                fit(label, interior)
            };
            let fill = "─".repeat(interior.saturating_sub(label.width()));
            Line::from(vec![
                border(left),
                Span::styled(format!("{label}{fill}"), style),
                border(right),
            ])
        };
        let mut rows = (0..height).map(|_| framed("", muted)).collect::<Vec<_>>();
        rows[0] = ruled("╭", "─ SESSIONS ", "╮", muted.add_modifier(Modifier::BOLD));
        if height == 1 {
            return rows;
        }
        rows[height as usize - 1] = Line::from(border(&format!("╰{}╯", "─".repeat(interior))));
        if height == 2 {
            return rows;
        }

        for (n, slot) in self
            .slots
            .iter()
            .skip(self.sidebar_scroll)
            .take(self.visible_rows())
            .enumerate()
        {
            let y = 2 + n;
            let index = self.sidebar_scroll + n;
            let selected = Some(index) == self.selected;
            let text = fit(
                &format!(
                    " {} {} {} {}",
                    if selected { "›" } else { " " },
                    session_ordinal(index),
                    slot.name,
                    slot.status.glyph()
                ),
                interior,
            );
            rows[y] = framed(
                &text,
                if selected {
                    accent.add_modifier(Modifier::BOLD)
                } else {
                    muted
                },
            );
        }
        let visible = self
            .slots
            .len()
            .saturating_sub(self.sidebar_scroll)
            .min(self.visible_rows());
        let new_y = 3 + visible;
        let footer_start = height.saturating_sub(1 + self.sidebar_footer_rows()) as usize;
        if new_y < footer_start {
            rows[new_y] = framed("   + New agent", muted);
        }

        if self.prefix && self.sidebar_footer_rows() == 6 {
            rows[footer_start] = ruled(
                "├",
                "─ MUX › ^Space ",
                "┤",
                accent.add_modifier(Modifier::BOLD),
            );
            for (offset, command) in [
                " n  new      w  worktree",
                " c  current  d  directory",
                " x  close    j  next",
                " k  prev     1–9 jump",
                " ?  help",
            ]
            .iter()
            .enumerate()
            {
                rows[footer_start + 1 + offset] = framed(command, muted);
            }
        } else {
            rows[footer_start] = ruled("├", "", "┤", muted);
            rows[footer_start + 1] = framed(" ^Space  mux", muted);
        }
        rows
    }

    fn compose_frame(&mut self, width: u16, height: u16) -> (Buffer, Option<(u16, u16)>) {
        self.width = width;
        self.height = height;
        let layout = self.layout(width, height);
        let theme = render::Theme::default();
        let muted = Style::default()
            .fg(theme.muted_text)
            .add_modifier(Modifier::DIM);
        let accent = Style::default().fg(theme.accent);
        let mut buffer = Buffer::empty(Rect::new(0, 0, width, height));

        if let Some((_, sidebar_width)) = layout.sidebar {
            for (y, line) in self
                .sidebar_lines(sidebar_width, layout.body_height, muted, accent)
                .iter()
                .enumerate()
            {
                buffer.set_line(0, y as u16, line, sidebar_width.saturating_add(1));
            }
        }

        let pane_cursor = if let Some(i) = self.selected {
            let frame = self.slots[i]
                .pane
                .render(layout.main_width, layout.body_height);
            for y in 0..layout.body_height {
                let line = frame.lines.get(y as usize).cloned().unwrap_or_default();
                buffer.set_line(layout.main_x, y, &line, layout.main_width);
            }
            Some(pane_cursor_position(layout, &frame))
        } else {
            if layout.body_height > 4 {
                let y = layout.body_height / 2;
                buffer.set_string(layout.main_x, y, " No agents yet", muted);
                buffer.set_string(
                    layout.main_x,
                    y + 1,
                    " Press Ctrl+Space then n to create one.",
                    muted,
                );
            }
            None
        };

        let cursor = if let Some(overlay) = &self.overlay {
            draw_overlay(&mut buffer, layout, overlay, theme)
        } else {
            pane_cursor
        };
        (
            buffer,
            cursor.and_then(|cursor| bounded_cursor(cursor, width, height)),
        )
    }

    fn draw(&mut self, out: &mut Stdout) -> Result<()> {
        let (width, height) = terminal::size().unwrap_or((self.width, self.height));
        let (frame, cursor) = self.compose_frame(width, height);
        let mut output = Hide.to_string();
        for y in 0..height {
            let spans = (0..width)
                .map(|x| {
                    let cell = &frame[(x, y)];
                    Span::styled(cell.symbol().to_owned(), cell.style())
                })
                .collect::<Vec<_>>();
            let _ = write!(
                output,
                "{}{}",
                MoveTo(0, y),
                line_to_ansi(&Line::from(spans))
            );
        }
        if let Some((x, y)) = cursor {
            let _ = write!(output, "{}{}", MoveTo(x, y), Show);
        }
        out.write_all(output.as_bytes())?;
        out.flush()?;
        Ok(())
    }
}

fn pane_cursor_position(layout: MuxLayout, frame: &crate::PaneFrame) -> (u16, u16) {
    (
        layout.main_x.saturating_add(frame.cursor_col),
        frame.cursor_row,
    )
}

fn bounded_cursor(cursor: (u16, u16), width: u16, height: u16) -> Option<(u16, u16)> {
    (width > 0 && height > 0).then(|| (cursor.0.min(width - 1), cursor.1.min(height - 1)))
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
fn basename(path: &Path) -> String {
    path.file_name()
        .and_then(|v| v.to_str())
        .filter(|v| !v.is_empty())
        .unwrap_or("agent")
        .to_owned()
}
fn expand_path(value: &str, cwd: &Path) -> PathBuf {
    let p = if value == "~" || value.starts_with("~/") {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| cwd.to_owned())
            .join(value.trim_start_matches("~/"))
    } else {
        PathBuf::from(value)
    };
    if p.is_absolute() { p } else { cwd.join(p) }
}
const MAX_DIRECTORY_SUGGESTIONS: usize = 6;

fn directory_suggestions(value: &str, cwd: &Path) -> Vec<PathBuf> {
    let path = expand_path(value, cwd);
    let (parent, needle) = if path.is_dir() {
        (path, "".into())
    } else {
        (
            path.parent().unwrap_or(cwd).to_owned(),
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase(),
        )
    };
    let Ok(read) = std::fs::read_dir(parent) else {
        return vec![];
    };
    let mut found = read
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if !p.is_dir() || name.starts_with('.') {
                return None;
            }
            let lower = name.to_lowercase();
            is_subsequence(&needle, &lower).then_some((
                if lower.starts_with(&needle) { 0 } else { 1 },
                lower,
                p,
            ))
        })
        .collect::<Vec<_>>();
    found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    found.truncate(MAX_DIRECTORY_SUGGESTIONS);
    found.into_iter().map(|(_, _, path)| path).collect()
}
fn is_subsequence(q: &str, s: &str) -> bool {
    let mut chars = s.chars();
    q.chars().all(|c| chars.by_ref().any(|v| v == c))
}
fn session_ordinal(index: usize) -> String {
    (index + 1).to_string()
}

fn fit(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let cw = c.width().unwrap_or(0);
        if used + cw > width {
            break;
        }
        out.push(c);
        used += cw
    }
    out.push_str(&" ".repeat(width.saturating_sub(used)));
    out
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct OverlayFrame {
    x: u16,
    y: u16,
    width: u16,
    rows: Vec<String>,
}

fn overlay_frame(layout: MuxLayout, overlay: &Overlay) -> OverlayFrame {
    let title = match overlay {
        Overlay::NewMenu { .. } | Overlay::Current { .. } => "New agent",
        Overlay::Worktree { .. } => "New worktree agent",
        Overlay::Directory { .. } => "Choose directory",
        Overlay::Close => "Close agent?",
        Overlay::ConfirmExit => "Exit Harness?",
        Overlay::ConfirmDuplicate { .. } => "Duplicate workspace?",
        Overlay::Help => "Mux help",
    };
    let content_lines = match overlay {
        Overlay::NewMenu { selected } => ["New worktree", "Current directory", "Another directory"]
            .iter()
            .enumerate()
            .flat_map(|(index, label)| {
                [
                    format!("{} {label}", if index == *selected { '›' } else { ' ' }),
                    String::new(),
                ]
            })
            .collect::<Vec<_>>(),
        Overlay::Worktree { branch, keep } => vec![
            format!("Branch / name: {branch}"),
            format!("Keep worktree: {} (Tab)", if *keep { "yes" } else { "no" }),
        ],
        Overlay::Current { name } => vec![format!("Session name: {name}")],
        // Reserve a compact candidate list so asynchronous scans do not move
        // or resize the editor while results filter down as the user types.
        Overlay::Directory {
            input,
            suggestions,
            selected,
        } => std::iter::once(format!("Directory: {input}"))
            .chain((0..MAX_DIRECTORY_SUGGESTIONS).map(|index| {
                suggestions.get(index).map_or_else(String::new, |path| {
                    format!(
                        "{} {}",
                        if index == *selected { '›' } else { ' ' },
                        path.display()
                    )
                })
            }))
            .collect(),
        Overlay::Close => vec!["Enter to close; Esc to cancel".into()],
        Overlay::ConfirmExit => {
            vec!["All running agents will stop. Enter to exit; Esc to cancel".into()]
        }
        Overlay::ConfirmDuplicate { .. } => vec![
            "A direct agent already uses this directory. Enter to create anyway; Esc to cancel"
                .into(),
        ],
        Overlay::Help => vec![
            "Ctrl+Space: n menu · w worktree · c current".into(),
            "d directory · j/k switch · 1-9 jump".into(),
            "x close · ? help".into(),
            "Mouse: click sessions/new/menu choices; wheel scroll".into(),
        ],
    };
    let width = if layout.main_width < 10 {
        layout.main_width
    } else {
        layout.main_width.saturating_sub(4).clamp(10, 58)
    };
    let height = (content_lines.len() + 4).min(layout.body_height as usize) as u16;
    let x = layout.main_x + layout.main_width.saturating_sub(width) / 2;
    let y = layout.body_height.saturating_sub(height) / 2;
    let mut rows = Vec::with_capacity(height as usize);
    for row in 0..height {
        let text = match (width, height, row) {
            (0, _, _) => String::new(),
            (1, _, _) => "│".into(),
            (_, 1, _) => format!("╭{}╮", "─".repeat(width.saturating_sub(2) as usize)),
            (_, _, 0) => {
                let interior = width.saturating_sub(2) as usize;
                let label = format!("─ {title} ");
                let rule = if label.width() <= interior {
                    format!("{label}{}", "─".repeat(interior - label.width()))
                } else {
                    fit(&label, interior)
                };
                format!("╭{rule}╮")
            }
            (_, _, r) if r == height - 1 => {
                format!("╰{}╯", "─".repeat(width.saturating_sub(2) as usize))
            }
            (_, _, 1) => "│".into(),
            _ => format!(
                "│ {}",
                content_lines
                    .get((row - 2) as usize)
                    .map(String::as_str)
                    .unwrap_or("")
            ),
        };
        let mut fitted = fit(&text, width as usize);
        if width >= 2 {
            let right = if row == 0 {
                '╮'
            } else if row == height - 1 {
                '╯'
            } else {
                '│'
            };
            fitted.pop();
            fitted.push(right);
        }
        rows.push(fitted);
    }
    OverlayFrame { x, y, width, rows }
}

fn overlay_cursor(frame: &OverlayFrame, overlay: &Overlay) -> Option<(u16, u16)> {
    let (label, input) = match overlay {
        Overlay::Worktree { branch, .. } => ("Branch / name: ", branch.as_str()),
        Overlay::Current { name } => ("Session name: ", name.as_str()),
        Overlay::Directory { input, .. } => ("Directory: ", input.as_str()),
        _ => return None,
    };
    // Editable content is on the first content row, after the left border and
    // its padding. Clamp the caret to the final opaque interior cell when the
    // value is wider than the modal.
    if frame.width < 3 || frame.rows.len() <= 3 {
        return None;
    }
    let column = 2usize + label.width() + input.width();
    Some((
        frame.x + column.min(frame.width.saturating_sub(2) as usize) as u16,
        frame.y + 2,
    ))
}

fn draw_overlay(
    buffer: &mut Buffer,
    layout: MuxLayout,
    overlay: &Overlay,
    theme: render::Theme,
) -> Option<(u16, u16)> {
    let frame = overlay_frame(layout, overlay);
    let cursor = overlay_cursor(&frame, overlay);
    let primary = Style::default().fg(theme.primary_text);
    let muted = Style::default()
        .fg(theme.muted_text)
        .add_modifier(Modifier::DIM);
    let accent = Style::default().fg(theme.accent);
    for (row, text) in frame.rows.iter().enumerate() {
        let directory_selection = match overlay {
            Overlay::Directory { selected, .. } => row == selected + 3,
            _ => false,
        };
        let style = if row == 0 || row + 1 == frame.rows.len() {
            muted
        } else if directory_selection {
            accent.add_modifier(Modifier::BOLD)
        } else if matches!(overlay, Overlay::Directory { .. }) && row >= 3 {
            muted
        } else if matches!(overlay, Overlay::NewMenu { selected: 0 }) && row == 2
            || matches!(overlay, Overlay::NewMenu { selected: 1 }) && row == 4
            || matches!(overlay, Overlay::NewMenu { selected: 2 }) && row == 6
        {
            accent.add_modifier(Modifier::BOLD)
        } else {
            primary
        };
        let y = frame.y + row as u16;
        buffer.set_line(
            frame.x,
            y,
            &Line::from(Span::styled(text.clone(), style)),
            frame.width,
        );
        // Keep every modal outline the same muted gray as the sidebar,
        // independently of highlighted or primary-colored content.
        if row == 0 || row + 1 == frame.rows.len() {
            for x in frame.x..frame.x.saturating_add(frame.width) {
                buffer[(x, y)].set_style(muted);
            }
        } else if frame.width > 0 {
            buffer[(frame.x, y)].set_style(muted);
            if frame.width > 1 {
                buffer[(frame.x + frame.width - 1, y)].set_style(muted);
            }
        }
    }
    cursor
}

static MUX_RAW_MODE: AtomicBool = AtomicBool::new(false);
static MUX_ALT_SCREEN: AtomicBool = AtomicBool::new(false);
static MUX_BRACKETED_PASTE: AtomicBool = AtomicBool::new(false);
static MUX_MOUSE_CAPTURE: AtomicBool = AtomicBool::new(false);
static MUX_KEYBOARD_FLAGS: AtomicBool = AtomicBool::new(false);

fn restore_mux_terminal(out: &mut Stdout) {
    if MUX_KEYBOARD_FLAGS.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(out, Show);
    if MUX_MOUSE_CAPTURE.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, DisableMouseCapture);
    }
    if MUX_BRACKETED_PASTE.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, DisableBracketedPaste);
    }
    if MUX_ALT_SCREEN.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, LeaveAlternateScreen);
    }
    if MUX_RAW_MODE.swap(false, Ordering::SeqCst) {
        let _ = disable_raw_mode();
    }
    let _ = out.flush();
}

fn install_mux_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            // Do not rely on unwinding: release builds abort after this hook.
            // Only reverse modes this mux successfully enabled, so a partial
            // setup cannot pop terminal state owned by another component.
            restore_mux_terminal(&mut io::stdout());
            previous(panic);
        }));
    });
}

struct TerminalGuard {
    out: Stdout,
}
impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        MUX_RAW_MODE.store(true, Ordering::SeqCst);
        let mut this = Self { out: io::stdout() };
        let setup = (|| -> io::Result<()> {
            execute!(this.out, EnterAlternateScreen)?;
            MUX_ALT_SCREEN.store(true, Ordering::SeqCst);
            execute!(this.out, EnableBracketedPaste)?;
            MUX_BRACKETED_PASTE.store(true, Ordering::SeqCst);
            execute!(this.out, EnableMouseCapture)?;
            MUX_MOUSE_CAPTURE.store(true, Ordering::SeqCst);
            Ok(())
        })();
        if let Err(error) = setup {
            drop(this);
            return Err(error).context("configure mux terminal");
        }
        if execute!(
            this.out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok()
        {
            MUX_KEYBOARD_FLAGS.store(true, Ordering::SeqCst);
        }
        Ok(this)
    }
    fn out(&mut self) -> &mut Stdout {
        &mut self.out
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_mux_terminal(&mut self.out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContextFileEntry, SkillEntry};
    fn pane(path: &str) -> AgentPane {
        AgentPane::new(
            "m",
            "p",
            vec![],
            Vec::<SkillEntry>::new(),
            Vec::<ContextFileEntry>::new(),
            "auto",
            path.into(),
        )
    }
    fn key(c: char) -> Event {
        Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
    }
    fn strip_ansi(value: &str) -> String {
        let mut plain = String::new();
        let mut chars = value.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for code in chars.by_ref() {
                    if code.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                plain.push(c);
            }
        }
        plain
    }
    fn add_slots(mux: &mut MuxUi, count: u64) {
        for id in 1..=count {
            mux.apply(MuxEvent::Add {
                id,
                name: format!("agent-{id}"),
                workspace: PathBuf::from(format!("/agent-{id}")),
                worktree: false,
                status: MuxStatus::Idle,
                pane: pane("/"),
            });
        }
    }
    #[test]
    fn provisional_add_activates_but_replacement_preserves_newer_selection() {
        let mut mux = MuxUi::new("/x".into());
        mux.apply(MuxEvent::Add {
            id: 1,
            name: "first".into(),
            workspace: "/pending".into(),
            worktree: true,
            status: MuxStatus::Starting,
            pane: pane("/pending"),
        });
        assert_eq!(mux.selected_id(), Some(1));
        mux.apply(MuxEvent::Add {
            id: 2,
            name: "second".into(),
            workspace: "/second".into(),
            worktree: false,
            status: MuxStatus::Starting,
            pane: pane("/second"),
        });
        assert_eq!(mux.selected_id(), Some(2));

        mux.apply(MuxEvent::Replace {
            id: 1,
            name: "ready".into(),
            workspace: "/first".into(),
            worktree: true,
            status: MuxStatus::Starting,
            pane: pane("/first"),
        });

        assert_eq!(mux.selected_id(), Some(2));
        assert_eq!(mux.slots[0].name, "ready");
        assert_eq!(mux.slots[0].workspace, PathBuf::from("/first"));
    }

    #[test]
    fn routes_by_stable_id_and_retains_drafts() {
        let mut m = MuxUi::new("/x".into());
        m.apply(MuxEvent::Add {
            id: 9,
            name: "a".into(),
            workspace: "/a".into(),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane("/a"),
        });
        m.apply(MuxEvent::Add {
            id: 4,
            name: "b".into(),
            workspace: "/b".into(),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane("/b"),
        });
        assert!(m.handle(key('z')).unwrap().is_empty());
        assert_eq!(m.slots[1].pane.editor_text(), "z");
        m.handle(Event::Key(KeyEvent::new(
            KeyCode::Null,
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        m.handle(key('k')).unwrap();
        assert_eq!(m.selected_id(), Some(9));
        assert_eq!(m.slots[1].pane.editor_text(), "z");
        m.handle(Event::Key(KeyEvent::new(
            KeyCode::Null,
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        m.handle(key('j')).unwrap();
        let actions = m
            .handle(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .unwrap();
        assert!(matches!(
            &actions[0],
            MuxAction::AgentInput { id: 4, input: InputMessage::Message(text) } if text == "z"
        ));
    }
    #[test]
    fn prefix_and_overlays() {
        let mut m = MuxUi::new("/x".into());
        m.handle(Event::Key(KeyEvent::new(
            KeyCode::Char(' '),
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        m.handle(key('n')).unwrap();
        assert!(matches!(m.overlay, Some(Overlay::NewMenu { .. })));
        m.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)))
            .unwrap();
        assert!(m.overlay.is_none())
    }
    #[test]
    fn duplicate_direct_workspace_requires_confirmation_and_keeps_inheritance() {
        let cwd = std::fs::canonicalize(".").unwrap();
        let mut mux = MuxUi::new(cwd.clone());
        mux.apply(MuxEvent::Add {
            id: 12,
            name: "one".into(),
            workspace: cwd.clone(),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane(cwd.to_str().unwrap()),
        });
        let mut actions = vec![];
        mux.request_create(
            WorkspaceChoice::Directory {
                path: cwd,
                name: "two".into(),
            },
            &mut actions,
        );
        assert!(actions.is_empty());
        assert!(matches!(
            mux.overlay,
            Some(Overlay::ConfirmDuplicate { .. })
        ));
        let actions = mux
            .handle(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .unwrap();
        assert!(matches!(
            actions.as_slice(),
            [MuxAction::Create {
                inherit_from: Some(12),
                ..
            }]
        ));
    }

    #[test]
    fn worktree_tab_toggles_keep_without_closing_editor() {
        let mut mux = MuxUi::new("/x".into());
        mux.overlay = Some(Overlay::Worktree {
            branch: "feature".into(),
            keep: true,
        });

        mux.handle(Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)))
            .unwrap();

        assert!(matches!(
            mux.overlay,
            Some(Overlay::Worktree { keep: false, .. })
        ));
    }

    #[test]
    fn directory_completion_is_debounced_and_stale_results_are_ignored() {
        let mut mux = MuxUi::new("/x".into());
        mux.overlay = Some(Overlay::Directory {
            input: String::new(),
            suggestions: vec![],
            selected: 0,
        });
        mux.handle(key('a')).unwrap();
        let first = mux.directory_request.take().unwrap();
        mux.handle(key('b')).unwrap();
        let second = mux.directory_request.as_ref().unwrap();
        assert!(second.generation > first.generation);
        assert_eq!(second.input, "ab");
        let generation = second.generation;

        assert!(!mux.apply_directory_completion(DirectoryCompletion {
            generation: first.generation,
            input: first.input,
            suggestions: vec!["/stale".into()],
        }));
        let Overlay::Directory { suggestions, .. } = mux.overlay.as_ref().unwrap() else {
            panic!("directory overlay closed unexpectedly");
        };
        assert!(suggestions.is_empty());

        assert!(mux.apply_directory_completion(DirectoryCompletion {
            generation,
            input: "ab".into(),
            suggestions: vec!["/fresh".into()],
        }));
        mux.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)))
            .unwrap();
        assert!(!mux.apply_directory_completion(DirectoryCompletion {
            generation,
            input: "ab".into(),
            suggestions: vec!["/late".into()],
        }));
    }

    #[test]
    fn layout_gives_both_panes_the_full_terminal_height() {
        let m = MuxUi::new("/x".into());
        assert!(m.layout(29, 10).sidebar.is_none());
        assert!(m.layout(41, 10).sidebar.is_some());
        let layout = m.layout(80, 20);
        assert!(layout.sidebar.is_some());
        assert_eq!(layout.body_height, 20);
        assert_eq!(layout.main_x + layout.main_width, 80);
    }

    #[test]
    fn rendered_sidebar_boundary_spans_every_screen_row_for_any_session_count() {
        let theme = render::Theme::default();
        let muted = Style::default().fg(theme.muted_text);
        let accent = Style::default().fg(theme.accent);

        for count in [1, 6] {
            let mut mux = MuxUi::new("/".into());
            mux.handle(Event::Resize(80, 12)).unwrap();
            add_slots(&mut mux, count);
            let layout = mux.layout(80, 12);
            let (_, sidebar_width) = layout.sidebar.unwrap();
            let sidebar = mux.sidebar_lines(sidebar_width, 12, muted, accent);
            assert_eq!(sidebar.len(), 12);

            for (y, row) in sidebar.iter().enumerate() {
                let plain = strip_ansi(&line_to_ansi(row));
                assert_eq!(plain.width(), sidebar_width as usize + 1, "sidebar row {y}");
                let boundary = plain.chars().last().unwrap();
                assert!(matches!(boundary, '╮' | '│' | '┤' | '╯'), "sidebar row {y}");

                // Model final composition with a visible main-pane first cell.
                let screen = format!(
                    "{plain}  A{}",
                    " ".repeat(layout.main_width.saturating_sub(1) as usize)
                );
                assert_eq!(screen.width(), 80, "screen row {y}");
                assert_eq!(screen.chars().nth(layout.main_x as usize), Some('A'));
            }
            assert!(strip_ansi(&line_to_ansi(&sidebar[0])).starts_with("╭─ SESSIONS "));
            assert!(strip_ansi(&line_to_ansi(sidebar.last().unwrap())).starts_with('╰'));
        }
    }

    #[test]
    fn normal_sidebar_uses_compact_ordinals_spacers_and_anchored_footer() {
        let mut mux = MuxUi::new("/".into());
        mux.handle(Event::Resize(80, 14)).unwrap();
        add_slots(&mut mux, 10);
        mux.selected = Some(0);
        mux.sidebar_scroll = 0;
        let lines = mux.sidebar_lines(26, 14, Style::default(), Style::default());
        let plain = lines
            .iter()
            .map(|line| strip_ansi(&line_to_ansi(line)))
            .collect::<Vec<_>>();

        assert!(plain[0].starts_with("╭─ SESSIONS "));
        assert!(plain[1].trim_matches(['│', ' ']).is_empty());
        assert!(plain[2].contains("› 1 agent-1 ✓"));
        assert!(plain[3].contains("  2 agent-2 ✓"));
        assert!(plain[7].contains("  6 agent-6 ✓"));
        assert!(plain[8].trim_matches(['│', ' ']).is_empty());
        assert!(plain[9].contains("+ New agent"));
        assert!(plain[10].trim_matches(['│', ' ']).is_empty());
        assert_eq!(plain[11], format!("├{}┤", "─".repeat(25)));
        assert!(plain[12].contains("^Space  mux"));
        assert_eq!(plain[13], format!("╰{}╯", "─".repeat(25)));
        assert_eq!(session_ordinal(8), "9");
        assert_eq!(session_ordinal(9), "10");
    }

    #[test]
    fn sidebar_rows_handle_tiny_heights_without_overflow() {
        let mux = MuxUi::new("/".into());
        let style = Style::default();
        for height in 0..=2 {
            for sidebar_width in 0..=20 {
                let rows = mux.sidebar_lines(sidebar_width, height, style, style);
                assert_eq!(rows.len(), height as usize);
                assert!(
                    rows.iter()
                        .all(|row| row.width() == sidebar_width as usize + 1)
                );
            }
        }
    }

    #[test]
    fn tiny_layout_overlay_and_cursor_are_bounded() {
        let m = MuxUi::new("/x".into());
        for height in 0..=2 {
            for width in 0..=1 {
                let layout = m.layout(width, height);
                assert!(layout.main_x <= width);
                assert!(layout.main_width <= width);
                assert!(layout.body_height <= height);
                let frame = overlay_frame(layout, &Overlay::Help);
                assert!(frame.x.saturating_add(frame.width) <= width);
                assert!(frame.y.saturating_add(frame.rows.len() as u16) <= layout.body_height);
            }
        }
        assert_eq!(bounded_cursor((9, 9), 0, 2), None);
        assert_eq!(bounded_cursor((9, 9), 1, 1), Some((0, 0)));
        assert_eq!(bounded_cursor((9, 9), 2, 2), Some((1, 1)));
    }

    fn visible_buffer(buffer: &Buffer) -> String {
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn composed_new_agent_frame_is_unique_selectable_and_activates_selection() {
        let mut mux = MuxUi::new("/workspace".into());
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Char(' '),
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        mux.handle(key('n')).unwrap();

        let (initial, _) = mux.compose_frame(100, 30);
        let initial_text = visible_buffer(&initial);
        for option in ["New worktree", "Current directory", "Another directory"] {
            assert_eq!(
                initial_text.matches(option).count(),
                1,
                "{option}: {initial_text}"
            );
        }
        assert!(initial_text.contains("› New worktree"));
        assert!(!initial_text.contains("› Current directory"));

        mux.handle(Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)))
            .unwrap();
        let (selected, _) = mux.compose_frame(100, 30);
        let selected_text = visible_buffer(&selected);
        assert!(!selected_text.contains("› New worktree"));
        assert!(
            selected_text.contains("› Current directory"),
            "{selected_text}"
        );

        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();
        assert!(matches!(mux.overlay, Some(Overlay::Current { .. })));
        let (current, cursor) = mux.compose_frame(100, 30);
        let current_text = visible_buffer(&current);
        assert!(current_text.contains("Session name:"));
        assert!(!current_text.contains("New worktree"));
        assert!(cursor.is_some());

        mux.overlay = Some(Overlay::NewMenu { selected: 2 });
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();
        assert!(matches!(mux.overlay, Some(Overlay::Directory { .. })));
        let (directory, cursor) = mux.compose_frame(100, 30);
        let directory_text = visible_buffer(&directory);
        assert!(directory_text.contains("Directory:"));
        assert!(!directory_text.contains("Current directory"));
        assert!(cursor.is_some());
    }

    #[test]
    fn overlays_are_centered_in_main_pane_closed_and_opaque() {
        let mux = MuxUi::new("/x".into());
        let layout = mux.layout(100, 30);
        let first = overlay_frame(layout, &Overlay::NewMenu { selected: 0 });
        assert!(first.x >= layout.main_x);
        assert_eq!(
            first.x - layout.main_x,
            (layout.main_width - first.width) / 2
        );
        assert!(first.rows[0].starts_with("╭─ New agent "));
        assert!(first.rows[0].ends_with('╮'));
        assert!(
            first.rows[0]
                .trim_matches(['╭', '╮', '─'])
                .contains("New agent")
        );
        assert!(
            first.rows.last().unwrap().starts_with('╰')
                && first.rows.last().unwrap().ends_with('╯')
                && first
                    .rows
                    .last()
                    .unwrap()
                    .chars()
                    .skip(1)
                    .take(first.width.saturating_sub(2) as usize)
                    .all(|character| character == '─')
        );
        assert!(
            first
                .rows
                .iter()
                .all(|row| row.width() == first.width as usize)
        );
        assert!(
            first.rows[1..first.rows.len() - 1]
                .iter()
                .all(|row| row.starts_with('│') && row.ends_with('│'))
        );

        // A subsequent, shorter overlay still paints every cell in its rectangle;
        // no text or divider from the New Agent menu can survive underneath it.
        let second = overlay_frame(layout, &Overlay::Close);
        assert!(
            second
                .rows
                .iter()
                .all(|row| row.width() == second.width as usize)
        );
        assert!(!second.rows.join("\n").contains("New worktree"));
    }

    #[test]
    fn editable_overlay_cursor_tracks_unicode_and_edits() {
        let mut mux = MuxUi::new("/x".into());
        mux.handle(Event::Resize(100, 30)).unwrap();
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Null,
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        mux.handle(key('n')).unwrap();
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();

        let layout = mux.layout(100, 30);
        let Overlay::Worktree { .. } = mux.overlay.as_ref().unwrap() else {
            panic!("new menu did not open worktree editor");
        };
        let empty = overlay_frame(layout, mux.overlay.as_ref().unwrap());
        let empty_cursor = overlay_cursor(&empty, mux.overlay.as_ref().unwrap()).unwrap();
        assert_eq!(
            empty_cursor,
            (empty.x + 2 + "Branch / name: ".width() as u16, empty.y + 2)
        );

        mux.handle(key('界')).unwrap();
        let typed = overlay_frame(layout, mux.overlay.as_ref().unwrap());
        assert_eq!(
            overlay_cursor(&typed, mux.overlay.as_ref().unwrap())
                .unwrap()
                .0,
            empty_cursor.0 + 2
        );
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Backspace,
            KeyModifiers::NONE,
        )))
        .unwrap();
        let erased = overlay_frame(layout, mux.overlay.as_ref().unwrap());
        assert_eq!(
            overlay_cursor(&erased, mux.overlay.as_ref().unwrap()).unwrap(),
            empty_cursor
        );

        assert!(overlay_cursor(&empty, &Overlay::Help).is_none());
    }

    #[test]
    fn sidebar_scroll_and_new_hit_use_rendered_capacity() {
        let mut mux = MuxUi::new("/".into());
        mux.handle(Event::Resize(80, 10)).unwrap();
        for id in 1..=5 {
            mux.apply(MuxEvent::Add {
                id,
                name: id.to_string(),
                workspace: "/".into(),
                worktree: false,
                status: MuxStatus::Idle,
                pane: pane("/"),
            });
        }
        for _ in 0..10 {
            mux.handle_mouse(MouseEventKind::ScrollDown, 0, 2);
        }
        assert_eq!(mux.visible_rows(), 2);
        assert_eq!(mux.sidebar_scroll, 3);

        // The session rows are 2..4, row 4 is the spacer, and New is row 5.
        mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 0, 4);
        assert!(mux.overlay.is_none());
        mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 0, 5);
        assert!(matches!(mux.overlay, Some(Overlay::NewMenu { .. })));
    }

    #[test]
    fn prefix_command_strip_is_titled_complete_and_bottom_anchored() {
        let mut mux = MuxUi::new("/".into());
        mux.handle(Event::Resize(80, 16)).unwrap();
        assert_eq!(mux.sidebar_footer_rows(), 2);
        assert_eq!(mux.visible_rows(), 8);

        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Null,
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        assert_eq!(mux.sidebar_footer_rows(), 6);
        assert_eq!(mux.visible_rows(), 4);

        let theme = render::Theme::default();
        let muted = Style::default().fg(theme.muted_text);
        let accent = Style::default().fg(theme.accent);
        let lines = mux.sidebar_lines(26, 16, muted, accent);
        let plain = lines
            .iter()
            .map(|line| strip_ansi(&line_to_ansi(line)))
            .collect::<Vec<_>>();
        assert!(plain[9].starts_with("├─ MUX › ^Space "));
        for (row, label) in [
            (10, "n  new      w  worktree"),
            (11, "c  current  d  directory"),
            (12, "x  close    j  next"),
            (13, "k  prev     1–9 jump"),
            (14, "?  help"),
        ] {
            assert!(
                plain[row].contains(label),
                "missing {label}: {}",
                plain[row]
            );
        }
        assert!(plain[15].starts_with('╰'));
    }

    #[test]
    fn direct_workspace_prefix_shortcuts_open_editors() {
        let mut mux = MuxUi::new("/some/project".into());
        for (shortcut, expected) in [('w', "worktree"), ('c', "current"), ('d', "directory")] {
            mux.overlay = None;
            mux.handle(Event::Key(KeyEvent::new(
                KeyCode::Null,
                KeyModifiers::CONTROL,
            )))
            .unwrap();
            mux.handle(key(shortcut)).unwrap();
            match (expected, mux.overlay.as_ref()) {
                ("worktree", Some(Overlay::Worktree { branch, keep })) => {
                    assert!(branch.is_empty());
                    assert!(*keep);
                }
                ("current", Some(Overlay::Current { name })) => assert_eq!(name, "project"),
                (
                    "directory",
                    Some(Overlay::Directory {
                        input, suggestions, ..
                    }),
                ) => {
                    assert!(input.is_empty());
                    assert!(suggestions.is_empty());
                }
                _ => panic!("{shortcut} opened the wrong overlay"),
            }
        }
    }

    #[test]
    fn directory_overlay_has_fixed_filterable_suggestion_list() {
        let layout = MuxUi::new("/x".into()).layout(100, 30);
        let empty = Overlay::Directory {
            input: "a".into(),
            suggestions: vec![],
            selected: 0,
        };
        let many = Overlay::Directory {
            input: "a".into(),
            suggestions: vec!["/first".into(), "/second".into(), "/third".into()],
            selected: 0,
        };
        let empty_frame = overlay_frame(layout, &empty);
        let many_frame = overlay_frame(layout, &many);
        assert_eq!(empty_frame.rows.len(), many_frame.rows.len());
        assert_eq!(empty_frame.y, many_frame.y);
        assert!(many_frame.rows.join("\n").contains("› /first"));
        assert!(many_frame.rows.join("\n").contains("/second"));
        assert!(many_frame.rows.join("\n").contains("/third"));

        let theme = render::Theme::default();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 100, 30));
        draw_overlay(&mut buffer, layout, &many, theme);
        let selected_style = buffer[(many_frame.x + 2, many_frame.y + 3)].style();
        assert_eq!(selected_style.fg, Some(theme.accent));
        assert!(selected_style.add_modifier.contains(Modifier::BOLD));
        let other_style = buffer[(many_frame.x + 2, many_frame.y + 4)].style();
        assert_eq!(other_style.fg, Some(theme.muted_text));
        assert!(other_style.add_modifier.contains(Modifier::DIM));
        let input_style = buffer[(many_frame.x + 2, many_frame.y + 2)].style();
        assert_eq!(input_style.fg, Some(theme.primary_text));
        assert!(!input_style.add_modifier.contains(Modifier::DIM));
    }

    #[test]
    fn help_lists_all_prefix_shortcuts() {
        let text = overlay_frame(MuxUi::new("/".into()).layout(100, 30), &Overlay::Help)
            .rows
            .join("\n");
        for shortcut in [
            "n menu",
            "w worktree",
            "c current",
            "d directory",
            "j/k",
            "1-9",
            "x close",
            "? help",
        ] {
            assert!(text.contains(shortcut), "missing {shortcut}: {text}");
        }
    }

    #[test]
    fn remove_selects_neighbor() {
        let mut m = MuxUi::new("/".into());
        for id in 1..=2 {
            m.apply(MuxEvent::Add {
                id,
                name: id.to_string(),
                workspace: "/".into(),
                worktree: false,
                status: MuxStatus::Starting,
                pane: pane("/"),
            })
        }
        m.apply(MuxEvent::Remove { id: 2 });
        assert_eq!(m.selected_id(), Some(1))
    }

    #[test]
    fn cursor_starts_at_the_top_of_the_main_pane() {
        let layout = MuxLayout {
            main_x: 23,
            ..MuxLayout::default()
        };
        let frame = crate::PaneFrame {
            lines: vec![],
            cursor_row: 4,
            cursor_col: 7,
        };
        assert_eq!(pane_cursor_position(layout, &frame), (30, 4));
    }

    #[test]
    fn page_and_mouse_wheel_route_to_main_pane() {
        let mut mux = MuxUi::new("/".into());
        mux.apply(MuxEvent::Add {
            id: 1,
            name: "one".into(),
            workspace: "/".into(),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane("/"),
        });
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )))
        .unwrap();
        assert_eq!(mux.slots[0].pane.scroll_offset(), 24);

        mux.handle(Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 60,
            row: 5,
            modifiers: KeyModifiers::NONE,
        }))
        .unwrap();
        assert_eq!(mux.slots[0].pane.scroll_offset(), 21);

        mux.handle(Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 2,
            row: 5,
            modifiers: KeyModifiers::NONE,
        }))
        .unwrap();
        assert_eq!(mux.slots[0].pane.scroll_offset(), 21);
        assert_eq!(mux.sidebar_scroll, 0);
    }

    #[test]
    fn delegated_exit_requires_confirmation_after_draft_is_cleared() {
        let mut mux = MuxUi::new("/".into());
        mux.apply(MuxEvent::Add {
            id: 7,
            name: "one".into(),
            workspace: "/".into(),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane("/"),
        });
        mux.slots[0].pane.set_draft("discard me");
        let ctrl_c = Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(mux.handle(ctrl_c.clone()).unwrap().is_empty());
        assert!(mux.overlay.is_none());
        assert!(mux.handle(ctrl_c.clone()).unwrap().is_empty());
        assert!(matches!(mux.overlay, Some(Overlay::ConfirmExit)));

        assert!(
            mux.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)))
                .unwrap()
                .is_empty()
        );
        assert!(mux.overlay.is_none());

        assert!(mux.handle(ctrl_c).unwrap().is_empty());
        assert!(matches!(mux.overlay, Some(Overlay::ConfirmExit)));
        assert!(matches!(
            mux.handle(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE
            )))
            .unwrap()
            .as_slice(),
            [MuxAction::Exit]
        ));
    }
}
