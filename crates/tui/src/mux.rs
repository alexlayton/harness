//! Full-screen, host-neutral multi-agent terminal UI.
//!
//! The mux owns the terminal exactly once and keeps an [`AgentPane`] per slot;
//! it never creates agents, worktrees, or sessions itself. Those lifecycle
//! requests are returned to the host as [`MuxAction`] values.

mod directory;

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
use directory::{MAX_DIRECTORY_SUGGESTIONS, basename, directory_suggestions, expand_path};
use futures_util::StreamExt;
use ratatui_core::buffer::Buffer;
use ratatui_core::layout::Rect;
use ratatui_core::style::{Modifier, Style};
use ratatui_core::text::{Line, Span};
use std::fmt::Write as _;
use std::io::{self, Stdout, Write};
use std::path::PathBuf;
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
    /// Ask the host to create a worktree from HEAD. `keep` controls whether
    /// retention remains sticky for later runs; mux always retains on close.
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
    directory_selection_explicit: bool,
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
            directory_selection_explicit: false,
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
                    pane.restore_draft(slot.pane.draft_snapshot());
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
        if let Event::Paste(text) = event {
            if self.overlay.is_none() {
                self.prefix = false;
                self.reveal_selected();
                return self.delegate(Event::Paste(text));
            }
            self.paste_into_overlay(&text);
            return Ok(out);
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
        // Sidebar borders and the gap before the main pane are inert.
        if sidebar_width.is_some_and(|width| x == 0 || x >= width) {
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
                let footer_start = layout
                    .body_height
                    .saturating_sub(1 + self.sidebar_footer_rows())
                    as usize;
                if y as usize == new_y && new_y < footer_start {
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
        self.directory_selection_explicit = false;
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

    fn paste_into_overlay(&mut self, text: &str) {
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
        self.directory_selection_explicit = false;
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
                        let selected = self.selected;
                        let changed = self.slots.iter_mut().enumerate().fold(
                            false,
                            |changed, (index, slot)| slot.pane.tick(selected == Some(index)) || changed,
                        );
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
            let line = buffer_row(&frame, y, width);
            let _ = write!(output, "{}{}", MoveTo(0, y), line_to_ansi(&line));
        }
        if let Some((x, y)) = cursor {
            let _ = write!(output, "{}{}", MoveTo(x, y), Show);
        }
        out.write_all(output.as_bytes())?;
        out.flush()?;
        Ok(())
    }
}

/// Convert one Ratatui buffer row back to styled text without emitting the
/// reset continuation cells that follow a wide grapheme. Writing those cells
/// would make the physical terminal row wider than its buffer geometry.
fn buffer_row(buffer: &Buffer, y: u16, width: u16) -> Line<'static> {
    let mut spans = Vec::new();
    let mut x = 0;
    while x < width {
        let cell = &buffer[(x, y)];
        let symbol = cell.symbol();
        spans.push(Span::styled(symbol.to_owned(), cell.style()));
        let cells = symbol.width().max(1).min(u16::MAX as usize) as u16;
        x = x.saturating_add(cells);
    }
    Line::from(spans)
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
            format!(
                "Retain for future runs: {} (Tab)",
                if *keep { "yes" } else { "no" }
            ),
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
            // Abort builds cannot run guards, so restoration must happen in
            // the hook. Unwind builds may catch a background-task panic; in
            // that case restoring here would dismantle a still-running mux.
            // The terminal-owning future restores through its guard instead.
            #[cfg(panic = "abort")]
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
mod tests;
