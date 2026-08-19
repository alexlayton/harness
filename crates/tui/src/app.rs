//! Terminal lifecycle, the retained `Tui` state, and the main event loop.
//!
//! Input dispatch lives here; event-to-state mapping is in `events`,
//! completion state/scanning in `completion`, and layout arithmetic in
//! `layout`.

use crate::commands;
use crate::completion::Completion;
use crate::environment::EnvironmentInfo;
use crate::input::{InputAction, classify};
use crate::render::Theme;
use crate::state::{EntryId, Focus, TranscriptEntry};
use crate::{InputMessage, ModelEntry, TuiEvent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
    KeyModifiers,
};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::{Terminal, TerminalOptions, Viewport};
use std::collections::HashMap;
use std::io::{self, Stdout, Write};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui_textarea::{CursorMove, TextArea};

pub(crate) const MAX_INPUT_FRACTION: usize = 30;
pub(crate) const MAX_HISTORY: usize = 1_000;
/// Bounds for the fixed inline viewport height computed once at startup.
const MIN_VIEWPORT_ROWS: usize = 5;
const MAX_VIEWPORT_ROWS: usize = 16;
const PLACEHOLDER: &str = "Type your message...";

/// The fixed inline viewport height for the process lifetime (ratatui's
/// inline height is immutable). Mirrors pi's `max(5, floor(rows * 0.3))` via
/// the existing `MAX_INPUT_FRACTION = 30` constant.
pub(crate) fn viewport_height(rows: u16) -> u16 {
    ((rows as usize * MAX_INPUT_FRACTION) / 100).clamp(MIN_VIEWPORT_ROWS, MAX_VIEWPORT_ROWS) as u16
}

/// Busy-state label used by the activity line. The variant drives both the
/// label and which transcript updates are expected next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Activity {
    Preparing,
    Working,
    Reasoning,
    Processing,
    Retrying,
}

impl Activity {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Preparing => "Preparing...",
            Self::Working => "Working...",
            Self::Reasoning => "Reasoning...",
            Self::Processing => "Processing...",
            Self::Retrying => "Retrying...",
        }
    }
}

/// The complete retained TUI state. Fields are `pub(crate)` so the
/// event/completion/layout modules can implement methods on `Tui`; the type is
/// consumed externally only through `Tui::new` and `Tui::run`.
pub struct Tui {
    pub(crate) terminal: Terminal<CrosstermBackend<Stdout>>,
    /// The fixed inline viewport height H, computed once at startup (immutable
    /// for the process lifetime). Used by the streaming commit to bound how
    /// much content stays in the live tail.
    pub(crate) viewport_height: u16,
    pub(crate) theme: Theme,
    pub(crate) model: String,
    pub(crate) provider: String,
    pub(crate) providers: Vec<String>,
    pub(crate) model_lists: HashMap<String, Vec<ModelEntry>>,
    pub(crate) session_candidates: Vec<crate::SessionListEntry>,
    pub(crate) session_completion_requested: bool,
    pub(crate) workspace_root: PathBuf,
    pub(crate) environment: EnvironmentInfo,

    pub(crate) completion: Option<Completion>,
    pub(crate) file_completion_tx: mpsc::UnboundedSender<crate::completion::FileCompletionResult>,
    pub(crate) file_completion_rx: mpsc::UnboundedReceiver<crate::completion::FileCompletionResult>,
    pub(crate) file_completion_generation: u64,
    pub(crate) file_completion_query: Option<String>,
    pub(crate) file_completion_cancel: Option<CancellationToken>,

    pub(crate) textarea: TextArea<'static>,
    pub(crate) prompt_scroll: usize,
    pub(crate) history: Vec<String>,
    pub(crate) history_pos: Option<usize>,
    pub(crate) draft: String,

    pub(crate) transcript: Vec<TranscriptEntry>,
    /// Number of entries at the front of `transcript` already committed into
    /// scrollback. Everything after this index is the live (uncommitted) tail.
    pub(crate) committed: usize,
    /// Whether the startup welcome banner has been committed into scrollback.
    pub(crate) welcome_shown: bool,
    pub(crate) next_entry_id: EntryId,
    pub(crate) streaming_assistant: Option<EntryId>,
    pub(crate) running_tool: Option<EntryId>,
    pub(crate) focused_tool: Option<usize>,
    pub(crate) focus: Focus,

    pub(crate) spinner: usize,
    pub(crate) activity: Activity,
    pub(crate) busy: bool,
    pub(crate) retrying: Option<u32>,
    pub(crate) restored: bool,
    pub(crate) session_id: Option<String>,
    pub(crate) session_title: Option<String>,
}

impl Tui {
    pub fn new(model: &str, provider: &str, providers: Vec<String>) -> Result<Self> {
        install_panic_hook();
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let environment = EnvironmentInfo::discover(workspace_root.clone());

        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnableBracketedPaste, cursor::Hide) {
            let _ = terminal::disable_raw_mode();
            let _ = execute!(stdout, DisableBracketedPaste);
            return Err(error).context("configure terminal input");
        }

        // The viewport is anchored inline at the current cursor row and stays
        // a single fixed height H for the whole process. No alternate screen,
        // no mouse capture: the terminal's own scrollback is the transcript.
        let rows = terminal::size().map(|(_, rows)| rows).unwrap_or(24);
        let height = viewport_height(rows);
        let backend = CrosstermBackend::new(stdout);
        let terminal = match Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(height),
            },
        ) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = terminal::disable_raw_mode();
                let _ = execute!(io::stdout(), DisableBracketedPaste);
                return Err(error).context("create inline terminal");
            }
        };
        // No post-construction `clear()`: the first `draw()` paints the whole
        // viewport, and in inline mode `clear()` would not touch scrollback.

        let (file_completion_tx, file_completion_rx) = mpsc::unbounded_channel();
        Ok(Self {
            terminal,
            viewport_height: height,
            theme: Theme::default(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            providers,
            model_lists: HashMap::new(),
            session_candidates: Vec::new(),
            session_completion_requested: false,
            workspace_root,
            environment,
            completion: None,
            file_completion_tx,
            file_completion_rx,
            file_completion_generation: 0,
            file_completion_query: None,
            file_completion_cancel: None,
            textarea: new_textarea(),
            prompt_scroll: 0,
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            transcript: Vec::new(),
            committed: 0,
            welcome_shown: false,
            next_entry_id: 1,
            streaming_assistant: None,
            running_tool: None,
            focused_tool: None,
            focus: Focus::Prompt,
            spinner: 0,
            activity: Activity::Preparing,
            busy: false,
            retrying: None,
            restored: false,
            session_id: None,
            session_title: None,
        })
    }

    pub async fn run<E>(
        mut self,
        mut events: mpsc::UnboundedReceiver<E>,
        input_tx: mpsc::UnboundedSender<InputMessage>,
        cancel: CancellationToken,
    ) -> Result<()>
    where
        E: TuiEvent + 'static,
    {
        let result = self.run_inner(&mut events, input_tx, cancel).await;
        let restore_result = self.restore();
        result.and(restore_result)
    }

    async fn run_inner<E>(
        &mut self,
        events: &mut mpsc::UnboundedReceiver<E>,
        input_tx: mpsc::UnboundedSender<InputMessage>,
        cancel: CancellationToken,
    ) -> Result<()>
    where
        E: TuiEvent + 'static,
    {
        // The startup welcome banner (title + tagline + version) is written
        // into scrollback once before anything else; `insert_before` clears the
        // viewport, so the first `draw()` right after repaints it.
        self.commit_welcome_banner()?;
        self.draw()?;
        let mut input_events = EventStream::new();
        let mut spinner_tick = tokio::time::interval(Duration::from_millis(200));
        spinner_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                maybe_event = events.recv() => {
                    if let Some(event) = maybe_event {
                        self.apply_event(event.into_ui_event())?;
                        self.draw()?;
                    }
                }
                maybe_input = input_events.next() => {
                    let event = match maybe_input {
                        Some(Ok(event)) => event,
                        Some(Err(error)) => return Err(error).context("read terminal event"),
                        None => return Ok(()),
                    };
                    if let Event::Resize(_, _) = event {
                        self.handle_resize();
                        self.draw()?;
                        continue;
                    }
                    if self.handle_input(event, &input_tx, &cancel)? {
                        return Ok(());
                    }
                    self.draw()?;
                }
                maybe_file_completion = self.file_completion_rx.recv() => {
                    if let Some(result) = maybe_file_completion {
                        self.apply_file_completion(result);
                        self.draw()?;
                    }
                }
                _ = spinner_tick.tick() => {
                    if self.busy {
                        self.spinner = self.spinner.wrapping_add(1);
                        self.draw()?;
                    }
                }
                _ = cancel.cancelled(), if !self.busy => {
                    return Ok(());
                }
            }
        }
    }

    fn handle_input(
        &mut self,
        event: Event,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        cancel: &CancellationToken,
    ) -> Result<bool> {
        if let Event::Key(key) = &event
            && !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
        {
            return Ok(false);
        }

        // Mouse capture is intentionally disabled: the wheel scrolls the
        // terminal's native scrollback, so there is nothing to handle here.

        if self.handle_completion_input(&event, input_tx)? {
            return Ok(false);
        }

        // Tool focus is reachable while a turn is running: the only focusable
        // tool is the running one, and it exists only while busy.
        if self.focused_tool.is_some() && self.handle_tool_focus(&event)? {
            return Ok(false);
        }

        match classify(&event) {
            InputAction::Quit => {
                cancel.cancel();
                return Ok(true);
            }
            InputAction::Interrupt => {
                let ctrl_c = matches!(
                    &event,
                    Event::Key(key)
                        if key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                );
                if ctrl_c {
                    if self.busy {
                        let _ = input_tx.send(InputMessage::Interrupt);
                    } else {
                        cancel.cancel();
                        return Ok(true);
                    }
                } else if self.completion.is_some() {
                    self.close_completion();
                } else if self.focused_tool.take().is_some() {
                    self.focus = Focus::Prompt;
                } else if self.busy {
                    let _ = input_tx.send(InputMessage::Interrupt);
                } else {
                    self.focus = Focus::Prompt;
                }
            }
            InputAction::Newline => {
                self.history_pos = None;
                self.draft.clear();
                self.textarea.input(event);
                self.refresh_completion();
                self.request_session_completion(input_tx)?;
            }
            InputAction::Submit => {
                let input = self.textarea.lines().join("\n");
                if input.trim().is_empty() {
                    return Ok(false);
                }
                if commands::is_command_input(&input) {
                    self.submit_command(&input, input_tx)?;
                    return Ok(false);
                }
                self.submit_message(input, input_tx)?;
            }
            InputAction::ToggleAllTools => {
                self.toggle_live_tool();
            }
            InputAction::FocusTools => {
                // The only focusable tool is the running one, so Tab
                // focuses mid-turn and falls through to the textarea
                // whenever nothing is running.
                if let Some(index) = self.live_tool_index() {
                    self.focused_tool = Some(index);
                    self.focus = Focus::Tool;
                } else {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event);
                    self.refresh_completion();
                    self.request_session_completion(input_tx)?;
                }
            }
            InputAction::Edit => {
                if !self.handle_navigation_edit(&event) {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event);
                    self.focus = Focus::Prompt;
                }
                self.refresh_completion();
                self.request_session_completion(input_tx)?;
            }
            InputAction::Ignore => {}
        }
        Ok(false)
    }

    fn handle_resize(&mut self) {
        // Resizing needs no extra work beyond drawing: ratatui's `autoresize`
        // (inside every `draw()`) detects the size change, re-anchors the
        // inline viewport at the new width, clamps its height to the terminal
        // rows, and clears the live region — so the next draw is a full
        // repaint of the live region at the new size. Committed scrollback is
        // immutable pixels and keeps its original wrap width; the terminal
        // soft-reflows it itself. Only the prompt scroll needs resetting.
        self.prompt_scroll = 0;
    }

    pub(crate) fn restore(&mut self) -> Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        self.cancel_file_completion();
        terminal::disable_raw_mode().context("restore terminal raw mode")?;
        execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            cursor::Show
        )
        .context("restore terminal input")?;
        // Leave one blank line below the fixed-height viewport so the shell
        // prompt lands below the UI rather than on its last row.
        writeln!(self.terminal.backend_mut()).context("leave terminal")?;
        Ok(())
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

pub(crate) fn new_textarea() -> TextArea<'static> {
    textarea_with_text("")
}

pub(crate) fn textarea_with_text(value: &str) -> TextArea<'static> {
    textarea_with_text_at_cursor(value, usize::MAX, usize::MAX)
}

pub(crate) fn textarea_with_text_at_cursor(
    value: &str,
    line: usize,
    column: usize,
) -> TextArea<'static> {
    let mut textarea = TextArea::new(value.split('\n').map(str::to_owned).collect());
    if line != usize::MAX {
        textarea.move_cursor(CursorMove::Jump(
            line.min(u16::MAX as usize) as u16,
            column.min(u16::MAX as usize) as u16,
        ));
    }
    textarea.set_placeholder_text(PLACEHOLDER);
    textarea.set_placeholder_style(
        Style::default()
            .fg(Theme::default().dim_text)
            .add_modifier(Modifier::DIM),
    );
    textarea.set_style(Style::default().fg(Theme::default().primary_text));
    textarea.set_cursor_line_style(Style::default());
    textarea
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), DisableBracketedPaste, cursor::Show);
        let _ = writeln!(io::stdout());
        previous(panic);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullscreen_ui_keeps_terminal_lifecycle_outside_pure_render_tests() {
        // The real terminal lifecycle is intentionally exercised manually;
        // render.rs, input.rs, state.rs, and environment.rs contain the pure
        // behavior tests that do not require a TTY.
    }

    #[test]
    fn viewport_height_is_clamped_to_the_5_16_band() {
        assert_eq!(viewport_height(5), 5);
        assert_eq!(viewport_height(16), 5);
        assert_eq!(viewport_height(17), 5);
        assert_eq!(viewport_height(20), 6);
        assert_eq!(viewport_height(24), 7);
        assert_eq!(viewport_height(50), 15);
        assert_eq!(viewport_height(53), 15);
        assert_eq!(viewport_height(54), 16);
        assert_eq!(viewport_height(200), 16);
        // Exactly 30% of the terminal rows, floor-divide then clamp.
        assert_eq!(viewport_height(100), 16);
    }
}
