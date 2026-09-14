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
use std::io::{self, Stdout, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use unicode_width::UnicodeWidthChar;

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
    AgentInput { id: MuxId, input: InputMessage },
    Create(WorkspaceChoice),
    Rename { id: MuxId, name: String },
    Close { id: MuxId },
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
        self.height.saturating_sub(5) as usize / if self.width >= 70 { 2 } else { 1 }
    }
    fn reveal_selected(&mut self) {
        if let Some(i) = self.selected {
            let n = self.visible_rows().max(1);
            if i < self.sidebar_scroll {
                self.sidebar_scroll = i
            } else if i >= self.sidebar_scroll + n {
                self.sidebar_scroll = i + 1 - n
            }
        }
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
        let id = self.slots[i].id;
        let result = self.slots[i].pane.handle_input(&event)?;
        let out = result
            .messages
            .into_iter()
            .map(|input| MuxAction::AgentInput { id, input })
            .collect::<Vec<_>>();
        if result.exit_requested {
            // Standalone's second Ctrl+C means "request exit". In a mux the
            // terminal owner must retain the documented destructive-action
            // confirmation rather than translating that request immediately.
            self.overlay = Some(Overlay::Close);
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
                self.sidebar_scroll =
                    (self.sidebar_scroll + 1).min(self.slots.len().saturating_sub(1))
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let compact = self.width < 70;
                let row = y.saturating_sub(2) as usize / (if compact { 1 } else { 2 });
                let visible = self.visible_rows();
                if row < visible {
                    let i = self.sidebar_scroll + row;
                    if i < self.slots.len() {
                        self.selected = Some(i)
                    } else if i == self.slots.len() {
                        self.overlay = Some(Overlay::NewMenu { selected: 0 })
                    }
                }
            }
            _ => {}
        }
    }
    fn handle_overlay(&mut self, mut overlay: Overlay, key: KeyEvent, out: &mut Vec<MuxAction>) {
        if key.code == KeyCode::Esc {
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
                KeyCode::Enter if !branch.trim().is_empty() => {
                    out.push(MuxAction::Create(WorkspaceChoice::Worktree {
                        branch: branch.trim().into(),
                        keep: *keep,
                    }))
                }
                KeyCode::Tab => *keep = !*keep,
                _ => {
                    edit_string(branch, key);
                    self.overlay = Some(overlay)
                }
            },
            Overlay::Current { name } => match key.code {
                KeyCode::Enter if !name.trim().is_empty() => {
                    out.push(MuxAction::Create(WorkspaceChoice::Directory {
                        path: self.cwd.clone(),
                        name: name.trim().into(),
                    }))
                }
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
                        out.push(MuxAction::Create(WorkspaceChoice::Directory {
                            name: basename(&p),
                            path: p,
                        }))
                    } else {
                        self.overlay = Some(overlay)
                    }
                }
                _ => {
                    edit_string(input, key);
                    *suggestions = directory_suggestions(input, &self.cwd);
                    *selected = 0;
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
        }
    }

    /// Run as the sole terminal owner. Restoration is attempted on normal
    /// return, cancellation, input failure, and panic (via RAII unwinding).
    pub async fn run(
        mut self,
        mut events: mpsc::UnboundedReceiver<MuxEvent>,
        actions: mpsc::UnboundedSender<MuxAction>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let mut terminal = TerminalGuard::enter()?;
        let mut input = EventStream::new();
        let mut tick = tokio::time::interval(Duration::from_millis(100));
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
        drop(terminal);
        result
    }

    fn draw(&mut self, out: &mut Stdout) -> Result<()> {
        let (w, h) = terminal::size().unwrap_or((self.width, self.height));
        self.width = w;
        self.height = h;
        let l = self.layout(w, h);
        let mut rows = vec![String::new(); h as usize];
        if let Some((_, sw)) = l.sidebar {
            rows[0] = " MUX".into();
            rows[1] = " SESSIONS".into();
            let compact = w < 70;
            let step = if compact { 1 } else { 2 };
            for (n, s) in self
                .slots
                .iter()
                .skip(self.sidebar_scroll)
                .take(self.visible_rows())
                .enumerate()
            {
                let y = 2 + n * step;
                let mark = if Some(self.sidebar_scroll + n) == self.selected {
                    ">"
                } else {
                    " "
                };
                rows[y] = fit(
                    &format!(
                        "{mark} {}  {}  {}",
                        self.sidebar_scroll + n + 1,
                        s.name,
                        s.status.glyph()
                    ),
                    sw as usize,
                );
                if !compact && y + 1 < l.body_height as usize {
                    rows[y + 1] = fit(
                        &format!(
                            "    {}{}",
                            s.workspace.display(),
                            if s.worktree { " · worktree" } else { "" }
                        ),
                        sw as usize,
                    )
                }
            }
            let y = 2 + self.slots.len().saturating_sub(self.sidebar_scroll) * step;
            if y < l.body_height as usize {
                rows[y] = " + New agent".into()
            }
            for row in rows.iter_mut().take(l.body_height as usize) {
                *row = format!("{}│", fit(row, sw as usize))
            }
        }
        let mut pane_cursor = None;
        if let Some(i) = self.selected {
            rows[0].push_str(&fit(
                &format!(" {}   {}", self.slots[i].name, self.slots[i].status.label()),
                l.main_width as usize,
            ));
            let frame = self.slots[i]
                .pane
                .render(l.main_width, l.body_height.saturating_sub(1));
            for n in 0..l.body_height.saturating_sub(1) as usize {
                if let Some(line) = frame.lines.get(n) {
                    let fitted = render::fit_line_to_width(line, l.main_width as usize);
                    let rendered_width = fitted
                        .spans
                        .iter()
                        .map(|span| unicode_width::UnicodeWidthStr::width(span.content.as_ref()))
                        .sum::<usize>();
                    rows[n + 1].push_str(&line_to_ansi(&fitted));
                    rows[n + 1].push_str(
                        &" ".repeat((l.main_width as usize).saturating_sub(rendered_width)),
                    );
                } else {
                    rows[n + 1].push_str(&" ".repeat(l.main_width as usize));
                }
            }
            pane_cursor = Some(pane_cursor_position(l, &frame));
        } else if l.body_height > 4 {
            let y = l.body_height as usize / 2;
            rows[y].push_str(" No agents yet");
            rows[y + 1].push_str(" Press Ctrl+Space then n to create one.");
        }
        rows[l.footer_y as usize] = fit(
            if self.prefix {
                " MUX > n:new  j/k:switch  1-9:jump  x:close  r:rename  b:sidebar  ?:help"
            } else {
                " ^Space prefix   n new   j/k switch   1-9 jump   x close   ? help"
            },
            w as usize,
        );

        let mut buffer = format!("{}{}", Hide, MoveTo(0, 0));
        for (index, row) in rows.iter().enumerate() {
            if index > 0 {
                buffer.push_str("\r\n");
            }
            if self.selected.is_some() && index < l.body_height as usize {
                buffer.push_str(row);
            } else {
                buffer.push_str(&fit(row, w as usize));
            }
        }
        if let Some(overlay) = &self.overlay {
            draw_overlay(&mut buffer, w, h, overlay);
        } else if let Some((x, y)) = pane_cursor {
            use std::fmt::Write as _;
            let _ = write!(buffer, "{}{}", MoveTo(x.min(w.saturating_sub(1)), y), Show);
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
fn draw_overlay(buffer: &mut String, width: u16, height: u16, overlay: &Overlay) {
    use std::fmt::Write as _;

    let title = match overlay {
        Overlay::NewMenu { .. } | Overlay::Current { .. } => "New agent",
        Overlay::Worktree { .. } => "New worktree agent",
        Overlay::Directory { .. } => "Choose directory",
        Overlay::Rename { .. } => "Rename agent",
        Overlay::Close => "Close agent?",
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
        Overlay::Help => concat!(
            "Ctrl+Space n new · j/k switch · 1-9 jump\n",
            "x close · r rename · b sidebar · ? help\n",
            "Mouse: click sessions/new; wheel scroll"
        )
        .into(),
    };
    let box_width = if width < 10 {
        width
    } else {
        width.saturating_sub(4).clamp(10, 58)
    };
    let lines = content.lines().collect::<Vec<_>>();
    let box_height = (lines.len() + 4).min(height as usize) as u16;
    let x = (width - box_width) / 2;
    let y = (height - box_height) / 2;
    for row in 0..box_height {
        let text = if row == 0 {
            format!("┌─ {title} ")
        } else if row == box_height - 1 {
            "└".into()
        } else {
            format!(
                "│ {}",
                lines.get(row.saturating_sub(2) as usize).unwrap_or(&"")
            )
        };
        let _ = write!(
            buffer,
            "{}{}",
            MoveTo(x, y + row),
            fit(&text, box_width as usize)
        );
    }
}

struct TerminalGuard {
    out: Stdout,
    keyboard: bool,
}
impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        let mut this = Self {
            out: io::stdout(),
            keyboard: false,
        };
        if let Err(e) = execute!(
            this.out,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        ) {
            drop(this);
            return Err(e).context("configure mux terminal");
        }
        if execute!(
            this.out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok()
        {
            this.keyboard = true
        }
        Ok(this)
    }
    fn out(&mut self) -> &mut Stdout {
        &mut self.out
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.keyboard {
            let _ = execute!(self.out, PopKeyboardEnhancementFlags);
        }
        let _ = execute!(
            self.out,
            Show,
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
        let _ = self.out.flush();
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
    fn narrow_layout_collapses_sidebar() {
        let m = MuxUi::new("/x".into());
        assert!(m.layout(41, 10).sidebar.is_none());
        assert!(m.layout(80, 20).sidebar.is_some())
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
    fn delegated_exit_opens_confirmation() {
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
        assert!(mux.handle(ctrl_c).unwrap().is_empty());
        assert!(matches!(mux.overlay, Some(Overlay::Close)));
    }
}
