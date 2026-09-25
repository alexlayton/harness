//! Full-screen, host-neutral multi-agent terminal UI.
//!
//! The mux owns the terminal exactly once and keeps an [`AgentPane`] per slot;
//! it never creates agents, worktrees, or sessions itself. Those lifecycle
//! requests are returned to the host as [`MuxAction`] values.

mod directory;
mod overlay;
mod terminal;
mod view;

#[cfg(test)]
use crate::render;
use crate::{AgentPane, InputMessage, UiEvent};
use anyhow::Result;
use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use directory::basename;
use overlay::{DirectoryCompletion, DirectoryRequest, Overlay};
use std::path::PathBuf;
use view::overlay_frame;
#[cfg(test)]
use view::{
    bounded_cursor, buffer_row, draw_overlay, overlay_cursor, pane_cursor_position, session_ordinal,
};

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
        // The pane submits every Enter that is not Shift/Alt+Enter,
        // including Ctrl+Enter. Keep drafts editable during setup or error,
        // but never send one to a runtime that is not ready.
        if matches!(self.slots[i].status, MuxStatus::Starting | MuxStatus::Error)
            && matches!(
                &event,
                Event::Key(KeyEvent { code: KeyCode::Enter, modifiers, .. })
                    if !modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
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
}
#[cfg(test)]
mod tests;
