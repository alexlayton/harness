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
    fn label(self) -> &'static str {
        match self {
            Self::Starting => "◐ Starting",
            Self::Running => "● Running",
            Self::Idle => "✓ Idle",
            Self::Error => "! Error",
        }
    }
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
    Rename {
        id: MuxId,
        name: String,
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
    Rename {
        id: MuxId,
        name: String,
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
    Rename {
        input: String,
    },
    Close,
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
    /// Height above the mux footer.
    pub body_height: u16,
    /// Row occupied by the mux footer.
    pub footer_y: u16,
}

/// Stateful mux controller. Constructing it does not touch the terminal.
pub struct MuxUi {
    slots: Vec<Slot>,
    selected: Option<usize>,
    sidebar_visible: bool,
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
            sidebar_visible: true,
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
    /// Return whether the sidebar is enabled (it may still collapse when narrow).
    pub fn sidebar_visible(&self) -> bool {
        self.sidebar_visible
    }
    /// Compute host-neutral pane geometry for terminal dimensions.
    pub fn layout(&self, width: u16, height: u16) -> MuxLayout {
        let body_height = height.saturating_sub(1);
        let side = if self.sidebar_visible && width >= 42 {
            Some((0, (width / 4).clamp(20, 32)))
        } else {
            None
        };
        let main_x = side.map_or(0, |(_, w)| w.saturating_add(1));
        MuxLayout {
            sidebar: side,
            main_x,
            main_width: width.saturating_sub(main_x),
            body_height,
            footer_y: height.saturating_sub(1),
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
            MuxEvent::Rename { id, name } => {
                if let Some(s) = self.slots.iter_mut().find(|s| s.id == id) {
                    s.name = name;
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
    fn visible_rows(&self) -> usize {
        let step = if self.width >= 70 { 2 } else { 1 };
        // Two headers and one always-visible `+ New agent` row occupy the rest.
        self.layout(self.width, self.height)
            .body_height
            .saturating_sub(3) as usize
            / step
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
            if self.overlay.is_none() {
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
                KeyCode::Char('j') => self.switch(1),
                KeyCode::Char('k') => self.switch(-1),
                KeyCode::Char('b') => self.sidebar_visible = !self.sidebar_visible,
                KeyCode::Char('?') => self.overlay = Some(Overlay::Help),
                KeyCode::Char('x') if self.selected.is_some() => {
                    self.overlay = Some(Overlay::Close)
                }
                KeyCode::Char('r') if self.selected.is_some() => {
                    self.overlay = Some(Overlay::Rename {
                        input: self.slots[self.selected.unwrap()].name.clone(),
                    })
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
            // Preserve standalone semantics: after Ctrl+C first clears a
            // draft, a second Ctrl+C exits the frontend. Closing just one slot
            // remains the explicit, confirmed mux-prefix `x` action.
            return Ok(vec![MuxAction::Exit]);
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
        if y >= layout.body_height {
            return;
        }
        let sidebar_width = layout.sidebar.map(|(_, width)| width);
        if sidebar_width.is_none_or(|width| x > width) {
            match kind {
                MouseEventKind::ScrollUp => self.scroll_selected(3),
                MouseEventKind::ScrollDown => self.scroll_selected(-3),
                _ => {}
            }
            return;
        }
        match kind {
            MouseEventKind::ScrollUp => self.sidebar_scroll = self.sidebar_scroll.saturating_sub(1),
            MouseEventKind::ScrollDown => {
                self.sidebar_scroll += 1;
                self.clamp_sidebar_scroll();
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let step = if self.width < 70 { 1 } else { 2 };
                let visible_count = self
                    .slots
                    .len()
                    .saturating_sub(self.sidebar_scroll)
                    .min(self.visible_rows());
                let new_y = 2 + visible_count * step;
                if y as usize == new_y {
                    self.overlay = Some(Overlay::NewMenu { selected: 0 });
                } else if y as usize >= 2 && (y as usize) < new_y {
                    let row = (y as usize - 2) / step;
                    let i = self.sidebar_scroll + row;
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
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Down => *selected = (*selected + 1).min(2),
                KeyCode::Enter => {
                    self.overlay = Some(match *selected {
                        0 => Overlay::Worktree {
                            branch: String::new(),
                            keep: true,
                        },
                        1 => Overlay::Current {
                            name: basename(&self.cwd),
                        },
                        _ => Overlay::Directory {
                            input: String::new(),
                            suggestions: vec![],
                            selected: 0,
                        },
                    })
                }
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
                KeyCode::Tab => *keep = !*keep,
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
                        *input = p.display().to_string()
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
                        self.directory_generation = self.directory_generation.wrapping_add(1);
                        self.directory_request = Some(DirectoryRequest {
                            generation: self.directory_generation,
                            input: input.clone(),
                        });
                    }
                    self.overlay = Some(overlay)
                }
            },
            Overlay::Rename { input } => match key.code {
                KeyCode::Enter if !input.trim().is_empty() => {
                    if let Some(id) = self.selected_id() {
                        out.push(MuxAction::Rename {
                            id,
                            name: input.trim().into(),
                        })
                    }
                }
                _ => {
                    edit_string(input, key);
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
        *suggestions = completion.suggestions;
        *selected = (*selected).min(suggestions.len().saturating_sub(1));
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
                        for slot in &mut self.slots {
                            slot.pane.tick_completion();
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

    fn draw(&mut self, out: &mut Stdout) -> Result<()> {
        let (w, h) = terminal::size().unwrap_or((self.width, self.height));
        self.width = w;
        self.height = h;
        let layout = self.layout(w, h);
        let theme = render::Theme::default();
        let muted = Style::default()
            .fg(theme.muted_text)
            .add_modifier(Modifier::DIM);
        let accent = Style::default().fg(theme.accent);
        let mut rows = vec![String::new(); h as usize];

        if let Some((_, sidebar_width)) = layout.sidebar {
            if let Some(row) = rows.get_mut(0) {
                *row = styled_fit(
                    " MUX",
                    sidebar_width as usize,
                    accent.add_modifier(Modifier::BOLD),
                );
            }
            if layout.body_height > 1 {
                rows[1] = styled_fit(" SESSIONS", sidebar_width as usize, muted);
            }
            let compact = w < 70;
            let step = if compact { 1 } else { 2 };
            for (n, slot) in self
                .slots
                .iter()
                .skip(self.sidebar_scroll)
                .take(self.visible_rows())
                .enumerate()
            {
                let y = 2 + n * step;
                let selected = Some(self.sidebar_scroll + n) == self.selected;
                let text = format!(
                    "{} {}  {}  {}",
                    if selected { ">" } else { " " },
                    self.sidebar_scroll + n + 1,
                    slot.name,
                    slot.status.glyph()
                );
                rows[y] = styled_fit(
                    &text,
                    sidebar_width as usize,
                    if selected {
                        accent.add_modifier(Modifier::BOLD)
                    } else {
                        muted
                    },
                );
                if !compact && y + 1 < layout.body_height as usize {
                    let detail = format!(
                        "    {}{}",
                        slot.workspace.display(),
                        if slot.worktree { " · worktree" } else { "" }
                    );
                    rows[y + 1] = styled_fit(&detail, sidebar_width as usize, muted);
                }
            }
            let visible = self
                .slots
                .len()
                .saturating_sub(self.sidebar_scroll)
                .min(self.visible_rows());
            let new_y = 2 + visible * step;
            if new_y < layout.body_height as usize {
                rows[new_y] = styled_fit(" + New agent", sidebar_width as usize, muted);
            }
            for row in rows.iter_mut().take(layout.body_height as usize) {
                row.push_str(&line_to_ansi(&Line::from(Span::styled("│", muted))));
            }
        }

        let mut pane_cursor = None;
        if let Some(i) = self.selected {
            let status_style = status_style(self.slots[i].status, theme);
            let title = styled_fit(
                &format!(" {}", self.slots[i].name),
                layout.main_width as usize,
                muted,
            );
            // Paint status over the right side without measuring ANSI bytes.
            if let Some(row) = rows.get_mut(0) {
                row.push_str(&title);
            }
            let label = self.slots[i].status.label();
            if layout.main_width as usize > label.width() + 1 {
                let x = layout.main_x + layout.main_width - label.width() as u16 - 1;
                let _ = write!(
                    rows[0],
                    "{}{}",
                    MoveTo(x, 0),
                    line_to_ansi(&Line::from(Span::styled(label, status_style)))
                );
            }
            let frame = self.slots[i]
                .pane
                .render(layout.main_width, layout.body_height.saturating_sub(1));
            for n in 0..layout.body_height.saturating_sub(1) as usize {
                let line = frame.lines.get(n).cloned().unwrap_or_default();
                rows[n + 1].push_str(&padded_line(&line, layout.main_width as usize));
            }
            pane_cursor = Some(pane_cursor_position(layout, &frame));
        } else {
            for row in rows.iter_mut().take(layout.body_height as usize) {
                row.push_str(&" ".repeat(layout.main_width as usize));
            }
            if layout.body_height > 4 {
                let y = layout.body_height as usize / 2;
                rows[y].push_str(&format!(
                    "{}{}",
                    MoveTo(layout.main_x, y as u16),
                    styled_fit(" No agents yet", layout.main_width as usize, muted)
                ));
                rows[y + 1].push_str(&format!(
                    "{}{}",
                    MoveTo(layout.main_x, y as u16 + 1),
                    styled_fit(
                        " Press Ctrl+Space then n to create one.",
                        layout.main_width as usize,
                        muted
                    )
                ));
            }
        }
        if let Some(row) = rows.get_mut(layout.footer_y as usize) {
            *row = styled_fit(
                if self.prefix {
                    " MUX > n:new  j/k:switch  1-9:jump  x:close  r:rename  b:sidebar  ?:help"
                } else {
                    " ^Space prefix   n new   j/k switch   1-9 jump   x close   ? help"
                },
                w as usize,
                muted,
            );
        }

        let mut buffer = format!("{}{}", Hide, MoveTo(0, 0));
        for (index, row) in rows.iter().enumerate() {
            if index > 0 {
                buffer.push_str("\r\n");
            }
            buffer.push_str(row);
        }
        if let Some(overlay) = &self.overlay {
            draw_overlay(&mut buffer, layout, overlay, theme);
        } else if let Some((x, y)) = pane_cursor.and_then(|cursor| bounded_cursor(cursor, w, h)) {
            let _ = write!(buffer, "{}{}", MoveTo(x, y), Show);
        }
        out.write_all(buffer.as_bytes())?;
        out.flush()?;
        Ok(())
    }
}

fn pane_cursor_position(layout: MuxLayout, frame: &crate::PaneFrame) -> (u16, u16) {
    (
        layout.main_x.saturating_add(frame.cursor_col),
        1_u16.saturating_add(frame.cursor_row),
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
            if !p.is_dir() {
                return None;
            }
            let name = e.file_name().to_string_lossy().to_lowercase();
            is_subsequence(&needle, &name).then_some(p)
        })
        .take(20)
        .collect::<Vec<_>>();
    found.sort();
    found
}
fn is_subsequence(q: &str, s: &str) -> bool {
    let mut chars = s.chars();
    q.chars().all(|c| chars.by_ref().any(|v| v == c))
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
fn padded_line(line: &Line<'_>, width: usize) -> String {
    let fitted = render::fit_line_to_width(line, width);
    let used = fitted
        .spans
        .iter()
        .map(|span| span.content.width())
        .sum::<usize>();
    format!(
        "{}{}",
        line_to_ansi(&fitted),
        " ".repeat(width.saturating_sub(used))
    )
}

fn styled_fit(text: &str, width: usize, style: Style) -> String {
    padded_line(&Line::from(Span::styled(text.to_owned(), style)), width)
}

fn status_style(status: MuxStatus, theme: render::Theme) -> Style {
    Style::default().fg(match status {
        MuxStatus::Starting => theme.muted_text,
        MuxStatus::Running => theme.accent,
        MuxStatus::Idle => theme.success,
        MuxStatus::Error => theme.error,
    })
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
        Overlay::Rename { .. } => "Rename agent",
        Overlay::Close => "Close agent?",
        Overlay::ConfirmDuplicate { .. } => "Duplicate workspace?",
        Overlay::Help => "Mux help",
    };
    let content = match overlay {
        Overlay::NewMenu { selected } => ["New worktree", "Current directory", "Another directory"]
            .iter()
            .enumerate()
            .map(|(index, label)| format!("{} {label}", if index == *selected { '>' } else { ' ' }))
            .collect::<Vec<_>>()
            .join("\n"),
        Overlay::Worktree { branch, keep } => format!(
            "Branch / name: {branch}\nKeep worktree: {} (Tab)",
            if *keep { "yes" } else { "no" }
        ),
        Overlay::Current { name } => format!("Session name: {name}"),
        Overlay::Directory {
            input, suggestions, ..
        } => format!(
            "Directory: {input}\n{}",
            suggestions
                .iter()
                .take(4)
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join("\n")
        ),
        Overlay::Rename { input } => format!("Name: {input}"),
        Overlay::Close => "Enter to close; Esc to cancel".into(),
        Overlay::ConfirmDuplicate { .. } => {
            "A direct agent already uses this directory. Enter to create anyway; Esc to cancel"
                .into()
        }
        Overlay::Help => concat!(
            "Ctrl+Space n new · j/k switch · 1-9 jump\n",
            "x close · r rename · b sidebar · ? help\n",
            "Mouse: click sessions/new; wheel scroll"
        )
        .into(),
    };
    let width = if layout.main_width < 10 {
        layout.main_width
    } else {
        layout.main_width.saturating_sub(4).clamp(10, 58)
    };
    let content_lines = content.lines().collect::<Vec<_>>();
    let height = (content_lines.len() + 4).min(layout.body_height as usize) as u16;
    let x = layout.main_x + layout.main_width.saturating_sub(width) / 2;
    let y = layout.body_height.saturating_sub(height) / 2;
    let mut rows = Vec::with_capacity(height as usize);
    for row in 0..height {
        let text = match (width, height, row) {
            (0, _, _) => String::new(),
            (1, _, _) => "│".into(),
            (_, 1, _) => format!("┌{}┐", "─".repeat(width.saturating_sub(2) as usize)),
            (_, _, 0) => format!("┌─ {title} "),
            (_, _, r) if r == height - 1 => "└".into(),
            _ => format!(
                "│ {}",
                content_lines
                    .get(row.saturating_sub(2) as usize)
                    .unwrap_or(&"")
            ),
        };
        let mut fitted = fit(&text, width as usize);
        if width >= 2 {
            let right = if row == 0 {
                '┐'
            } else if row == height - 1 {
                '┘'
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

fn draw_overlay(buffer: &mut String, layout: MuxLayout, overlay: &Overlay, theme: render::Theme) {
    use std::fmt::Write as _;
    let frame = overlay_frame(layout, overlay);
    let primary = Style::default().fg(theme.primary_text);
    let muted = Style::default()
        .fg(theme.muted_text)
        .add_modifier(Modifier::DIM);
    let accent = Style::default().fg(theme.accent);
    for (row, text) in frame.rows.iter().enumerate() {
        let style = if row == 0 || row + 1 == frame.rows.len() {
            muted
        } else if text.starts_with("│ >") {
            accent
        } else {
            primary
        };
        let _ = write!(
            buffer,
            "{}{}",
            MoveTo(frame.x, frame.y + row as u16),
            styled_fit(text, frame.width as usize, style)
        );
    }
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
    fn narrow_layout_collapses_sidebar() {
        let m = MuxUi::new("/x".into());
        assert!(m.layout(41, 10).sidebar.is_none());
        assert!(m.layout(80, 20).sidebar.is_some())
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
        assert!(first.rows[0].starts_with('┌') && first.rows[0].ends_with('┐'));
        assert!(
            first.rows.last().unwrap().starts_with('└')
                && first.rows.last().unwrap().ends_with('┘')
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
        assert_eq!(mux.visible_rows(), 3);
        assert_eq!(mux.sidebar_scroll, 2);

        mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 0, 8);
        assert!(matches!(mux.overlay, Some(Overlay::NewMenu { .. })));
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
    fn cursor_is_offset_by_pane_rectangle_and_header() {
        let layout = MuxLayout {
            main_x: 23,
            ..MuxLayout::default()
        };
        let frame = crate::PaneFrame {
            lines: vec![],
            cursor_row: 4,
            cursor_col: 7,
        };
        assert_eq!(pane_cursor_position(layout, &frame), (30, 5));
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
        assert_eq!(mux.slots[0].pane.scroll_offset(), 23);

        mux.handle(Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 60,
            row: 5,
            modifiers: KeyModifiers::NONE,
        }))
        .unwrap();
        assert_eq!(mux.slots[0].pane.scroll_offset(), 20);

        mux.handle(Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 2,
            row: 5,
            modifiers: KeyModifiers::NONE,
        }))
        .unwrap();
        assert_eq!(mux.slots[0].pane.scroll_offset(), 20);
        assert_eq!(mux.sidebar_scroll, 0);
    }

    #[test]
    fn delegated_exit_preserves_standalone_ctrl_c_semantics() {
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
        assert!(matches!(
            mux.handle(ctrl_c).unwrap().as_slice(),
            [MuxAction::Exit]
        ));
    }
}
