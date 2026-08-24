//! The direct-crossterm UI: no retained buffer, no viewport, no
//! `insert_before`. Everything final is written once as plain rows into the
//! terminal's native scrollback; only a small live region at the bottom
//! (streaming tail, running tool line, activity marker, and the `›` input
//! line) is rewritten in place with cursor-relative ANSI moves.
//!
//! Screen model and invariants:
//!
//! ```text
//! [ native scrollback: banner, metadata, user echoes, finished messages, ]
//! [ tool lines, notices — plain rows, never touched again               ]
//! [ live region: streaming tail · tool line · activity · input          ]
//!                                       ^ the real terminal cursor here
//! ```
//!
//! - `region` holds the rows we believe are painted and `cursor_row` /
//!   `cursor_col` locate the terminal cursor inside it. Both are updated only
//!   by [`CrossTerm::write_frame`], which makes them true by construction.
//! - The region never exceeds the terminal height: the streaming tail is
//!   clipped to a budget, the input is clipped around its cursor, and a final
//!   clamp guards degenerate sizes. This keeps every cursor move on-screen.
//! - Rows become scrollback by *commitment* rather than by being moved:
//!   pending entries are printed above the region within the same frame and
//!   then forgotten — their pixels are already final, so nothing needs to be
//!   redrawn or scrolled by us. Growth at the bottom scrolls the terminal
//!   itself, carrying committed rows into scrollback.
//! - A width change reflows every wrapped row, which invalidates the region
//!   bookkeeping; the screen is then cleared and repainted from `transcript`,
//!   which stores source entries (rather than rendered rows) for exactly this
//!   reason.

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

pub(crate) const MAX_HISTORY: usize = 1_000;
pub(crate) const PLACEHOLDER: &str = "Type your message...";

use crate::commands::{
    self, ArgumentKind, Candidate, CandidateKind, CompletionContext, CompletionResult,
    CompletionTarget, ParsedCommand,
};
use crate::commit::stable_block_split_offset;
use crate::environment::EnvironmentInfo;
use crate::input::{history_next, history_previous, push_history};
use crate::paths::{self, AtPrefix};
use crate::render::{self, Theme};
use crate::state::{ToolRecord, ToolStatus};
use crate::{
    ContextFileEntry, InputMessage, ModelEntry, SessionListEntry, SessionSnapshotEntry, SkillEntry,
    TuiEvent, UiEvent,
};
use anyhow::{Context, Result};
use crossterm::cursor::{MoveDown, MoveRight, MoveTo, MoveUp};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::style::{
    Attribute, Color as AnsiColor, SetAttribute, SetBackgroundColor, SetForegroundColor,
};
use crossterm::terminal::{self, Clear, ClearType};
use futures_util::StreamExt;
use ratatui_core::style::{Modifier, Style};
use ratatui_core::text::{Line, Span, Text};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{self, Stdout, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// The user-input prefix. Same glyph as the committed user echo so a
/// submitted message's pixels are exactly the prompt line it was typed on.
const INPUT_PREFIX: &str = "› ";
const INPUT_CONTINUATION: &str = "  ";
const INPUT_PREFIX_WIDTH: usize = 2;

/// One final or in-flight transcript block, stored at the source level so a
/// resize can re-render it at a new width.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Entry {
    /// The startup wordmark; `tagline` is picked once per launch so
    /// re-rendering after a resize does not re-roll it.
    Banner {
        tagline: String,
    },
    /// The header metadata line: `cwd  (branch)` left, `provider · model`
    /// right — the same metadata a classic status bar would show above
    /// its input box.
    Metadata {
        cwd: String,
        branch: Option<String>,
        provider: String,
        model: String,
        /// Auto-loaded AGENTS.md / CLAUDE.md paths (display form).
        context_files: Vec<String>,
        /// Discovered skill names.
        skills: Vec<String>,
    },
    User {
        text: String,
    },
    Assistant {
        markdown: String,
        reasoning: String,
    },
    Tool {
        record: ToolRecord,
    },
    Notice {
        text: String,
    },
    Error {
        text: String,
    },
    /// A centered `── label ──` rule marking a conversation boundary.
    Separator {
        label: String,
    },
}

/// The in-flight assistant message. The stable markdown prefix is drained
/// into `pending` (as [`Entry::Assistant`]) once it outgrows the live-tail
/// budget, keeping long responses flowing into scrollback incrementally.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StreamState {
    reasoning: String,
    markdown: String,
}

/// One active tool call in the keyed running-tool state.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RunningTool {
    call_id: String,
    record: ToolRecord,
}

/// How many active tool rows render before the rest fold into the compact
/// overflow line. Keeps a large `[subagents] max_concurrent` from pushing
/// the input line off-screen; the overflow row makes the hidden calls
/// visible as a count rather than silently clipped.
const MAX_VISIBLE_RUNNING_TOOLS: usize = 4;

/// The visual rows of the input line plus the cursor's position within them.
struct InputLayout {
    rows: Vec<Line<'static>>,
    cursor_row: usize,
    cursor_col: usize,
}

/// Token usage reported by [`UiEvent::UsageUpdated`], kept for the inline
/// counter trailer on the input placeholder row (`↑ in ↓ out · cost`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
    cost: String,
}

/// An open completion list driving the fish-style ghost preview, the Tab
/// accept, and the one-row hint above the input.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Completion {
    context: CompletionContext,
    candidates: Vec<Candidate>,
    selected: usize,
    kind: CompletionKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionKind {
    Slash,
    Session,
    Model,
    Path,
}

/// A debounced path-completion scan result delivered back over the run-loop
/// channel so a large directory tree never blocks the UI thread. `at_prefix`
/// is set when the scan was for an `@` file reference rather than a command
/// argument.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PathCompletionResult {
    generation: u64,
    context: CompletionContext,
    candidates: Vec<Candidate>,
    at_prefix: Option<AtPrefix>,
}

/// One frame's live region: the rows to paint and where the terminal cursor
/// belongs inside them.
struct RegionBuild {
    rows: Vec<Line<'static>>,
    cursor_row: usize,
    cursor_col: usize,
    /// Index of the activity row inside `rows` when this frame paints one;
    /// `None` otherwise. Lets spinner ticks rewrite only that row.
    activity_row: Option<usize>,
}

/// The active-tool section of a frame: its rendered lines and total row
/// count (including any overflow row) for the sizing budgets.
struct RunningRegion {
    lines: Vec<Line<'static>>,
    rows: usize,
}

/// The direct-crossterm UI. See the module docs for the screen model; the
/// struct is consumed externally only through `CrossTerm::new` and
/// `CrossTerm::run`.
pub struct CrossTerm {
    /// Raw stdout handle; every frame is one formatted ANSI write + flush.
    out: Stdout,
    theme: Theme,
    model: String,
    provider: String,
    environment: EnvironmentInfo,

    /// Cached terminal size, refreshed on resize events.
    width: u16,
    height: u16,

    /// Final entries of the current conversation, in order. Unlike the inline
    /// UI (which tracks a commit index into one transcript) these are kept as
    /// source entries because a width change must re-render the visible
    /// window of history at the new width.
    transcript: Vec<Entry>,
    /// Entries queued to print above the live region at the next paint; they
    /// become immutable scrollback rows and move into `transcript`.
    pending: Vec<Entry>,

    stream: Option<StreamState>,
    /// Currently running tool calls, keyed by harness call id and kept in
    /// launch order. Concurrent fan-out (e.g. several `subagent` calls in one
    /// response) means more than one record can be live at once; a finish
    /// removes exactly the matching id and never disturbs its neighbors.
    running_tools: Vec<RunningTool>,

    input: String,
    /// Byte offset of the editing cursor (always on a char boundary).
    cursor: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
    draft: String,

    /// Most recent turn usage for the placeholder counter trailer.
    usage: Option<Usage>,

    /// Global tool expansion state: Ctrl+O toggles every tool call's output
    /// expanded or collapsed and repaints the whole visible session. Rows
    /// already scrolled into the terminal's native scrollback are immutable
    /// pixels; after the repaint older rows show as the existing
    /// `… N rows above` ellipsis. Expanded outputs of entries visible in the
    /// window are correctly re-rendered.
    tools_expanded: bool,

    /// Cached completion models for a provider.
    providers: Vec<String>,
    model_lists: HashMap<String, Vec<ModelEntry>>,
    session_candidates: Vec<SessionListEntry>,
    /// Discovered skills, for `/<skill>` completion and `/skill <name>`
    /// argument completion.
    skills: Vec<SkillEntry>,
    /// Auto-loaded AGENTS.md / CLAUDE.md paths for the header context row.
    context_files: Vec<ContextFileEntry>,
    session_completion_requested: bool,
    /// Providers we have already asked the agent to fetch, to avoid duplicate
    /// `ListModels` requests while typing through a model token.
    model_list_requested: HashSet<String>,
    completion: Option<Completion>,
    path_completion_tx: mpsc::UnboundedSender<PathCompletionResult>,
    path_completion_rx: mpsc::UnboundedReceiver<PathCompletionResult>,
    path_completion_generation: u64,
    path_completion_query: Option<String>,
    path_completion_cancel: Option<CancellationToken>,

    busy: bool,
    activity: Activity,
    spinner: usize,

    /// Rows we believe are painted on screen and may rewrite in place.
    region: Vec<Line<'static>>,
    /// Cursor position within `region` (row is 0-based from the region top).
    cursor_row: usize,
    cursor_col: usize,

    /// Index of the activity row inside [`Self::region`] when the current
    /// frame paints one; `None` otherwise. Lets spinner ticks rewrite only
    /// that row instead of the whole region.
    activity_region_row: Option<usize>,

    restored: bool,
}

impl CrossTerm {
    /// Assemble the UI state without touching the terminal. `new` layers the
    /// terminal setup on top, and tests use this directly.
    fn base(
        model: &str,
        provider: &str,
        providers: Vec<String>,
        skills: Vec<SkillEntry>,
        context_files: Vec<ContextFileEntry>,
        width: u16,
        height: u16,
    ) -> Self {
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let environment = EnvironmentInfo::discover(workspace_root);
        let (path_completion_tx, path_completion_rx) = mpsc::unbounded_channel();
        Self {
            out: io::stdout(),
            theme: Theme::default(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            environment,
            width,
            height,
            transcript: Vec::new(),
            pending: Vec::new(),
            stream: None,
            running_tools: Vec::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            usage: None,
            tools_expanded: false,
            providers,
            model_lists: HashMap::new(),
            session_candidates: Vec::new(),
            skills,
            context_files,
            session_completion_requested: false,
            model_list_requested: HashSet::new(),
            completion: None,
            path_completion_tx,
            path_completion_rx,
            path_completion_generation: 0,
            path_completion_query: None,
            path_completion_cancel: None,
            busy: false,
            activity: Activity::Preparing,
            spinner: 0,
            region: Vec::new(),
            cursor_row: 0,
            cursor_col: 0,
            activity_region_row: None,
            restored: false,
        }
    }

    /// `skills` and `context_files` come from the startup discovery in main
    /// (the TUI never touches the filesystem for them).
    pub fn new(
        model: &str,
        provider: &str,
        providers: Vec<String>,
        skills: Vec<SkillEntry>,
        context_files: Vec<ContextFileEntry>,
    ) -> Result<Self> {
        install_panic_hook();
        let (width, height) = terminal::size().unwrap_or((80, 24));
        let mut ui = Self::base(
            model,
            provider,
            providers,
            skills,
            context_files,
            width,
            height,
        );
        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        if let Err(error) = execute!(ui.out, EnableBracketedPaste) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("configure terminal input");
        }
        // The kitty keyboard protocol makes Shift+Enter report as `Enter` with
        // the SHIFT modifier, which the input handler already maps to a newline
        // (works on Ghostty; iTerm/Terminal.app do not support it). Only
        // `DISAMBIGUATE_ESCAPE_CODES` is pushed — `REPORT_ALL_KEYS_AS_ESCAPE_CODES`
        // would swallow plain characters. Popped in `restore` and the panic hook.
        let _ = execute!(
            ui.out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
        // The cursor stays visible: it *is* the input caret, sitting right
        // after the `› ` prefix. No hide, no fake cell.
        Ok(ui)
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
        let restore = self.restore();
        result.and(restore)
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
        // The startup header: the wordmark banner augmented with the
        // cwd/branch and provider/model metadata line.
        self.pending.push(Entry::Banner {
            tagline: render::pick_tagline().to_owned(),
        });
        self.pending.push(self.metadata_entry());
        self.paint()?;

        let mut input_events = EventStream::new();
        let mut spinner_tick = tokio::time::interval(Duration::from_millis(200));
        spinner_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                maybe_event = events.recv() => {
                    let Some(event) = maybe_event else { return Ok(()) };
                    self.apply_event(event.into_ui_event());
                    // Coalesced painting: an agent burst (streaming deltas,
                    // tool start/finish pairs, notices) arrives as many
                    // channel events within a few milliseconds. Drain
                    // everything already queued and paint once per wake so
                    // the per-delta repaint tax stays off the streaming
                    // path. `try_recv` never blocks, so this only ever
                    // consumes what has already been sent — no added latency.
                    while let Ok(event) = events.try_recv() {
                        self.apply_event(event.into_ui_event());
                    }
                    self.paint()?;
                }
                maybe_input = input_events.next() => {
                    let event = match maybe_input {
                        Some(Ok(event)) => event,
                        Some(Err(error)) => return Err(error).context("read terminal event"),
                        None => return Ok(()),
                    };
                    if let Event::Resize(width, height) = event {
                        self.handle_resize(width, height)?;
                        continue;
                    }
                    if self.handle_input(&event, &input_tx, &cancel)? {
                        return Ok(());
                    }
                    self.paint()?;
                }
                maybe_path = self.path_completion_rx.recv() => {
                    if let Some(result) = maybe_path {
                        self.apply_path_completion(result);
                        self.paint()?;
                    }
                }
                _ = spinner_tick.tick() => {
                    // A busy tick only animates the activity glyph. Rewrite
                    // just that row in place instead of paying a full
                    // live-region serialization five times per second; fall
                    // back to the normal frame when the row cannot be
                    // located.
                    if !self.busy {
                        continue;
                    }
                    self.spinner = self.spinner.wrapping_add(1);
                    if !self.repaint_activity_only()? {
                        self.paint()?;
                    }
                }
                _ = cancel.cancelled(), if !self.busy => {
                    return Ok(());
                }
            }
        }
    }

    /// Rewrite only the activity row in place, leaving the rest of the live
    /// region untouched. Used by spinner ticks: the glyph animation alone
    /// must not re-serialize the streaming tail and input rows five times a
    /// second. Returns `false` when the activity row's position is unknown
    /// (caller falls back to a full [`Self::paint`]).
    fn repaint_activity_only(&mut self) -> Result<bool> {
        let Some(row) = self.activity_region_row else {
            return Ok(false);
        };
        let theme = self.theme;
        let gutter = render::horizontal_pad(self.width) as usize;
        let line = activity_line(self.activity, self.spinner, theme);
        let mut buffer = String::new();
        let up = self.cursor_row.saturating_sub(row);
        if up > 0 {
            let _ = write!(buffer, "{}", MoveUp(up as u16));
        }
        let _ = write!(buffer, "\r{}", Clear(ClearType::UntilNewLine));
        write_row(&mut buffer, &line, gutter);
        if up > 0 {
            let _ = write!(buffer, "\r{}", MoveDown(up as u16));
        }
        buffer.push('\r');
        let col = self.cursor_col.min(u16::MAX as usize) as u16;
        if col > 0 {
            let _ = write!(buffer, "{}", MoveRight(col));
        }
        self.out
            .write_all(buffer.as_bytes())
            .context("write activity")?;
        self.out.flush().context("flush activity")?;
        Ok(true)
    }

    fn handle_resize(&mut self, width: u16, height: u16) -> Result<()> {
        let width_changed = width != self.width;
        self.width = width;
        self.height = height;
        // Rows are pre-wrapped to the terminal width, so a width change
        // reflows them and invalidates the region bookkeeping — repaint the
        // whole visible window from the source transcript. A height-only
        // change reflows nothing; a plain repaint re-clips the budgets. Only
        // when the height shrank below the painted region is the bookkeeping
        // stale (the region top may have scrolled off-screen).
        if width_changed || self.region.len() > height as usize {
            self.repaint_all()
        } else {
            self.paint()
        }
    }

    fn handle_input(
        &mut self,
        event: &Event,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        cancel: &CancellationToken,
    ) -> Result<bool> {
        let Event::Key(key) = event else {
            if let Event::Paste(text) = event {
                insert_text(&mut self.input, &mut self.cursor, text);
                self.refresh_completion();
            }
            return Ok(false);
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return Ok(false);
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if control => {
                if self.busy {
                    let _ = input_tx.send(InputMessage::Interrupt);
                } else {
                    cancel.cancel();
                    return Ok(true);
                }
            }
            KeyCode::Char('d') if control => {
                cancel.cancel();
                return Ok(true);
            }
            // Ctrl+O toggles all tool outputs expanded/collapsed and repaints
            // the whole visible session from source entries. The Char arm's
            // CONTROL guard prevents this from falling through to insert.
            KeyCode::Char('o') if control => {
                self.tools_expanded = !self.tools_expanded;
                self.repaint_all()?;
                return Ok(false);
            }
            // Esc interrupts a running turn (the same intent as Ctrl+C); when
            // a completion list is open, Esc closes the list instead.
            KeyCode::Esc => {
                if self.busy {
                    let _ = input_tx.send(InputMessage::Interrupt);
                } else if self.completion.is_some() {
                    self.close_completion();
                    // Skip the tail refresh: it would reopen the same list.
                    return Ok(false);
                }
            }
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
            {
                insert_text(&mut self.input, &mut self.cursor, "\n");
            }
            KeyCode::Enter => self.submit(input_tx)?,
            KeyCode::Backspace => delete_backward(&mut self.input, &mut self.cursor),
            KeyCode::Delete => delete_forward(&mut self.input, &mut self.cursor),
            KeyCode::Left => move_left(&self.input, &mut self.cursor),
            KeyCode::Right => move_right(&self.input, &mut self.cursor),
            KeyCode::Home | KeyCode::Char('a') if control => {
                self.cursor = line_bounds(&self.input, self.cursor).0;
            }
            KeyCode::End | KeyCode::Char('e') if control => {
                self.cursor = line_bounds(&self.input, self.cursor).1;
            }
            // Up/Down move within a multi-line draft; at the top/bottom edge
            // they recall input history (readline-style).
            KeyCode::Up => {
                if !self.move_input_line(-1) {
                    self.history_up();
                }
            }
            KeyCode::Down => {
                if !self.move_input_line(1) {
                    self.history_down();
                }
            }
            // PageUp/PageDown scroll the terminal's native scrollback; they
            // must never insert control characters into the prompt.
            KeyCode::PageUp | KeyCode::PageDown => {}
            KeyCode::Char(character)
                if !key.modifiers.intersects(
                    KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                ) =>
            {
                insert_text(&mut self.input, &mut self.cursor, &character.to_string());
            }
            KeyCode::Tab => self.handle_tab(input_tx)?,
            _ => {}
        }
        // Recompute completion after any input mutation; refreshing at the
        // end of every handled key keeps the draft's ghost/hint in sync.
        self.refresh_completion();
        Ok(false)
    }

    fn history_up(&mut self) {
        let current = self.input.clone();
        if let Some(value) = history_previous(
            &self.history,
            &mut self.history_pos,
            &mut self.draft,
            &current,
        ) {
            self.input = value;
            self.cursor = self.input.len();
        }
    }

    fn history_down(&mut self) {
        if let Some(value) = history_next(&self.history, &mut self.history_pos, &self.draft) {
            self.input = value;
            self.cursor = self.input.len();
        }
    }

    /// Move the editing cursor one logical line up (`delta < 0`) or down.
    /// Returns `false` at the outer edges, where the caller falls through to
    /// history recall.
    fn move_input_line(&mut self, delta: i32) -> bool {
        match vertical_move(&self.input, self.cursor, delta) {
            Some(cursor) => {
                self.cursor = cursor;
                true
            }
            None => false,
        }
    }

    // ------------------------------------------------------------------
    // Inline completion
    // ------------------------------------------------------------------

    /// Rebuild the completion list for the current draft and cursor. Cheap
    /// except for path completion, which spawns a debounced blocking scan and
    /// delivers results over `path_completion_rx`. Pure callers (tests) drive
    /// this directly; the run loop calls it after every input mutation.
    fn refresh_completion(&mut self) {
        if commands::is_command_input(&self.input) {
            self.refresh_command_completion();
            return;
        }
        // Outside slash commands, an active `@` token completes file
        // references against a debounced workspace scan.
        let Some(prefix) = paths::extract_at_prefix(&self.input, self.cursor_char_col()) else {
            self.close_completion();
            return;
        };
        // Seed the request with the open list so the hint never blanks out
        // while the debounced scan for the refined query is in flight; the
        // fresh results replace them on arrival.
        let (initial, preferred) = self
            .completion
            .as_ref()
            .filter(|completion| completion.kind == CompletionKind::Path)
            .map(|completion| {
                let preferred = completion.candidates.get(completion.selected).cloned();
                (completion.candidates.clone(), preferred)
            })
            .unwrap_or_default();
        self.request_path_completion(
            CompletionContext {
                target: CompletionTarget::Argument(ArgumentKind::Path),
                token_start: prefix.token_start,
                token_end: prefix.token_end,
                query: prefix.query.clone(),
            },
            initial,
            preferred.map(|candidate| candidate.value),
            Some(prefix),
        );
    }

    /// [`Self::refresh_completion`] half that handles `/command` drafts:
    /// static candidates plus, for path arguments, a debounced scan.
    fn refresh_command_completion(&mut self) {
        let old_value = self.completion.as_ref().and_then(|completion| {
            completion
                .candidates
                .get(completion.selected)
                .map(|candidate| candidate.value.clone())
        });
        self.cancel_path_completion();
        let cursor_col = self.cursor_char_col();
        let Some(result) = commands::candidates_at_cursor(
            &self.input,
            cursor_col,
            &self.providers,
            &self.model_lists,
            &self.provider,
            &self.session_candidates,
            &self.skills,
        ) else {
            self.completion = None;
            self.session_completion_requested = false;
            return;
        };

        // Path arguments scan the filesystem on a debounced task.
        if matches!(
            result.context.target,
            CompletionTarget::Argument(ArgumentKind::Path),
        ) {
            self.request_path_completion(result.context, result.candidates, old_value, None);
            return;
        }
        let kind = match result.context.target {
            CompletionTarget::Argument(ArgumentKind::Session) => CompletionKind::Session,
            CompletionTarget::Argument(ArgumentKind::Model) => CompletionKind::Model,
            _ => CompletionKind::Slash,
        };
        self.set_completion(kind, result, old_value);
    }

    fn set_completion(
        &mut self,
        kind: CompletionKind,
        result: CompletionResult,
        old_value: Option<String>,
    ) {
        if result.candidates.is_empty() {
            // Keep a session list open (empty) so Tab can request it; close
            // everything else.
            if kind == CompletionKind::Session {
                self.completion = Some(Completion {
                    context: result.context,
                    candidates: Vec::new(),
                    selected: 0,
                    kind,
                });
            } else {
                self.completion = None;
            }
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
            context: result.context,
            candidates: result.candidates,
            selected,
            kind,
        });
    }

    fn cursor_char_col(&self) -> usize {
        self.input[..self.cursor].chars().count()
    }

    fn close_completion(&mut self) {
        self.cancel_path_completion();
        self.completion = None;
        self.session_completion_requested = false;
    }

    /// Apply a candidate to the draft, move the cursor, then recompute the
    /// list and fire any on-demand backend fetch (sessions / model lists).
    fn accept_candidate(
        &mut self,
        candidate: &Candidate,
        context: &CompletionContext,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        self.apply_replacement(candidate, context);
        self.completion = None;
        // A None-argument command (e.g. `/new`) has no arguments to complete;
        // the refresh below would otherwise keep its own name listed.
        let argument_less = matches!(context.target, CompletionTarget::Command,)
            && commands::command_spec(&candidate.value)
                .is_some_and(|spec| spec.argument_kind == ArgumentKind::None);
        if argument_less {
            return Ok(());
        }
        self.request_backend(input_tx, context, candidate)?;
        self.refresh_completion();
        Ok(())
    }

    fn apply_replacement(&mut self, candidate: &Candidate, context: &CompletionContext) {
        let cursor_col = self.cursor_char_col();
        let Some((replacement, new_cursor_col)) =
            commands::apply_completion(&self.input, cursor_col, context, candidate, &self.skills)
        else {
            return;
        };
        self.input = replacement;
        self.cursor = byte_index_at_char(&self.input, new_cursor_col);
    }

    /// Send anything the new draft now needs from the agent: a `ListSessions`
    /// for `/load` arguments or a `ListModels` when a `provider:` token was
    /// just completed and its model list is unknown. The dedupe sets keep
    /// repeated Tab presses from spamming the agent.
    fn request_backend(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        context: &CompletionContext,
        candidate: &Candidate,
    ) -> Result<()> {
        let is_session = matches!(
            context.target,
            CompletionTarget::Argument(ArgumentKind::Session),
        );
        if is_session && self.session_candidates.is_empty() && !self.session_completion_requested {
            self.session_completion_requested = true;
            input_tx
                .send(InputMessage::ListSessions)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        }
        let provider = candidate
            .value
            .strip_suffix(':')
            .filter(|name| {
                self.providers
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(name))
            })
            .map(str::to_owned);
        if let Some(provider) = provider
            && !self.model_lists.contains_key(&provider)
            && !self.model_list_requested.contains(provider.as_str())
        {
            self.model_list_requested.insert(provider.clone());
            input_tx
                .send(InputMessage::ListModels {
                    provider: provider.clone(),
                })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        }
        Ok(())
    }

    /// The Tab handler for completion.
    fn handle_tab(&mut self, input_tx: &mpsc::UnboundedSender<InputMessage>) -> Result<()> {
        self.refresh_completion();
        let Some(completion) = self.completion.as_ref().cloned() else {
            return Ok(());
        };
        if completion.candidates.is_empty() {
            // Session/model lists often arrive a beat late; repeat Tab asks
            // the backend again.
            self.request_backend(
                input_tx,
                &completion.context,
                &Candidate {
                    value: String::new(),
                    description: String::new(),
                    kind: CandidateKind::Slash,
                },
            )?;
            return Ok(());
        }
        if completion.candidates.len() == 1 {
            let candidate = completion.candidates[0].clone();
            self.accept_candidate(&candidate, &completion.context, input_tx)?;
            return Ok(());
        }
        // File and model references rank fuzzily, so a shared prefix is often
        // meaningless (`luna` may rank `openrouter:openai/gpt-5.6-luna`
        // first). Tab accepts the highlighted candidate — the top-ranked
        // match on the first press — rather than becoming a no-op merely
        // because the typed query is not a literal prefix.
        if matches!(
            completion.kind,
            CompletionKind::Path | CompletionKind::Model
        ) {
            let mut selected = completion.selected;
            let token = &self.input[completion.context.token_start
                ..completion.context.token_end.min(self.input.len())];
            if completion.candidates[selected].value == token {
                selected = (selected + 1) % completion.candidates.len();
            }
            let candidate = completion.candidates[selected].clone();
            self.accept_candidate(&candidate, &completion.context, input_tx)?;
            return Ok(());
        }
        // Multiple candidates: extend the shared prefix, or cycle among them
        // once the token already matches the common prefix.
        let prefix = commands::common_prefix(&completion.candidates);
        if prefix.is_empty() {
            return Ok(());
        }
        let token = &self.input
            [completion.context.token_start..completion.context.token_end.min(self.input.len())];
        if token == prefix {
            let selected = (completion.selected + 1) % completion.candidates.len();
            let candidate = completion.candidates[selected].clone();
            self.accept_candidate(&candidate, &completion.context, input_tx)?;
        } else {
            let prefix_candidate = Candidate {
                value: prefix,
                description: String::new(),
                kind: CandidateKind::Slash,
            };
            // Extend to the common prefix and keep the list open.
            self.apply_replacement(&prefix_candidate, &completion.context);
            self.refresh_completion();
        }
        Ok(())
    }

    // Path completion helpers -------------------------------------------------

    fn cancel_path_completion(&mut self) {
        if let Some(cancel) = self.path_completion_cancel.take() {
            cancel.cancel();
        }
        self.path_completion_query = None;
        self.path_completion_generation = self.path_completion_generation.wrapping_add(1);
        if self
            .completion
            .as_ref()
            .is_some_and(|completion| completion.kind == CompletionKind::Path)
        {
            self.completion = None;
        }
    }

    fn request_path_completion(
        &mut self,
        context: CompletionContext,
        initial: Vec<Candidate>,
        preferred: Option<String>,
        at_prefix: Option<AtPrefix>,
    ) {
        let query = context.query.clone();
        if self.path_completion_query.as_deref() == Some(query.as_str())
            && self.completion.as_ref().is_some_and(|completion| {
                completion.kind == CompletionKind::Path && completion.context == context
            })
        {
            return;
        }
        self.cancel_path_completion();
        let generation = self.path_completion_generation;
        let cancel = CancellationToken::new();
        let scan_cancel = cancel.clone();
        let root = self.environment.cwd.clone();
        let sender = self.path_completion_tx.clone();
        let scan_query = query.clone();
        let selected = preferred
            .as_deref()
            .and_then(|value| initial.iter().position(|c| c.value == value))
            .unwrap_or(0)
            .min(initial.len().saturating_sub(1));
        self.path_completion_cancel = Some(cancel.clone());
        self.path_completion_query = Some(query.clone());
        self.completion = Some(Completion {
            context: context.clone(),
            candidates: initial,
            selected,
            kind: CompletionKind::Path,
        });
        tokio::spawn(async move {
            // Debounce so a burst of typing triggers one scan, not many.
            tokio::time::sleep(Duration::from_millis(200)).await;
            if cancel.is_cancelled() {
                return;
            }
            // `@` references walk the whole workspace; command arguments list
            // one directory.
            let at = at_prefix.is_some();
            let scan = move |root: &Path, query: &str, cancel: &CancellationToken| {
                if at {
                    paths::find_candidates(root, query, cancel)
                } else {
                    paths::find_path_candidates(root, query, cancel)
                }
            };
            let candidates =
                tokio::task::spawn_blocking(move || scan(&root, &scan_query, &scan_cancel))
                    .await
                    .unwrap_or_default();
            if cancel.is_cancelled() {
                return;
            }
            let _ = sender.send(PathCompletionResult {
                generation,
                context,
                candidates,
                at_prefix,
            });
        });
    }

    fn apply_path_completion(&mut self, result: PathCompletionResult) {
        if result.generation != self.path_completion_generation {
            return;
        }
        let cursor_col = self.cursor_char_col();
        // The draft must still be on the same token the scan was started for.
        let context_now = match &result.at_prefix {
            Some(prefix) => paths::extract_at_prefix(&self.input, cursor_col)
                .filter(|now| now.token_start == prefix.token_start)
                .map(|now| now.into_context()),
            None => commands::completion_context(&self.input, cursor_col),
        };
        let Some(context_now) = context_now else {
            return;
        };
        if result.context != context_now {
            return;
        }
        self.path_completion_cancel = None;
        let old_value = self
            .completion
            .as_ref()
            .and_then(|completion| completion.candidates.get(completion.selected))
            .map(|candidate| candidate.value.as_str());
        // Merge static candidates (sessions for `/load`, providers/models for
        // `/model`) with the scanned filesystem results.
        let static_candidates = commands::candidates_at_cursor(
            &self.input,
            cursor_col,
            &self.providers,
            &self.model_lists,
            &self.provider,
            &self.session_candidates,
            &self.skills,
        )
        .map(|result| result.candidates)
        .unwrap_or_default();
        let merged = merge_candidates(static_candidates, result.candidates);
        if merged.is_empty() {
            self.completion = None;
            return;
        }
        let selected = old_value
            .and_then(|value| merged.iter().position(|c| c.value == value))
            .unwrap_or(0)
            .min(merged.len().saturating_sub(1));
        self.completion = Some(Completion {
            context: result.context,
            candidates: merged,
            selected,
            kind: CompletionKind::Path,
        });
    }

    /// The dim ghost suffix drawn after the cursor when the input matches a
    /// single untyped completion (or the shared prefix of several).
    fn ghost_text(&self) -> String {
        let Some(completion) = self.completion.as_ref() else {
            return String::new();
        };
        // Ghost only previews at the very end of the draft.
        if self.cursor != self.input.len() {
            return String::new();
        }
        let cursor_col = self.cursor_char_col();
        if completion.candidates.is_empty() {
            return String::new();
        }
        // Path completions preview the highlighted candidate — exactly what
        // the next Tab inserts — as the untyped remainder of the token. A
        // fuzzy match (`agent.rs` → `crates/agent/src/agent.rs`) shares no
        // literal prefix, so there is nothing honest to dim inline; the hint
        // row above the input carries those suggestions instead.
        if completion.kind == CompletionKind::Path {
            let candidate = &completion.candidates[completion.selected];
            let token_end = completion.context.token_end.min(self.input.len());
            let typed = &self.input[completion.context.token_start..token_end];
            return candidate
                .value
                .strip_prefix(typed)
                .map(str::to_owned)
                .unwrap_or_default();
        }
        if completion.candidates.len() == 1 {
            return commands::candidate_suffix(
                &self.input,
                cursor_col,
                &completion.context,
                &completion.candidates[0],
            );
        }
        let prefix = commands::common_prefix(&completion.candidates);
        if prefix.is_empty() {
            return String::new();
        }
        // Already at the shared prefix: the list hint stands in for the ghost.
        if self.input
            [completion.context.token_start..completion.context.token_end.min(self.input.len())]
            == prefix
        {
            return String::new();
        }
        let prefix_candidate = Candidate {
            value: prefix,
            description: String::new(),
            kind: CandidateKind::Slash,
        };
        commands::candidate_suffix(
            &self.input,
            cursor_col,
            &completion.context,
            &prefix_candidate,
        )
    }

    /// One dim hint row above the input. It is fitted to the actual content
    /// width so long provider-qualified model IDs can never soft-wrap behind
    /// the region bookkeeping and leave apparent duplicate suggestion rows.
    fn completion_hint(&self, width: usize) -> String {
        let Some(completion) = self.completion.as_ref() else {
            return String::new();
        };
        fit_completion_hint(&completion.candidates, completion.selected, width)
    }

    fn submit(&mut self, input_tx: &mpsc::UnboundedSender<InputMessage>) -> Result<()> {
        if self.input.trim().is_empty() {
            return Ok(());
        }
        let text = std::mem::take(&mut self.input);
        self.cursor = 0;
        push_history(&mut self.history, &text, MAX_HISTORY);
        self.history_pos = None;
        self.draft.clear();
        if commands::is_command_input(&text) {
            self.submit_command(&text, input_tx)?;
        } else {
            // The typed prompt line is already on screen with the `› ` prefix;
            // committing it as a user entry below re-prints the same pixels.
            self.pending.push(Entry::User { text: text.clone() });
            input_tx
                .send(InputMessage::Message(text))
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
            self.busy = true;
            self.activity = Activity::Preparing;
            self.spinner = 0;
        }
        Ok(())
    }

    fn submit_command(
        &mut self,
        input: &str,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let command = match commands::parse_command_with_skills(input, &self.skills) {
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
        // The typed command is echoed without the ⌘ glyph: the input line the
        // user already sees carries the `› ` prefix, so the notice is the raw
        // command text.
        self.add_notice(input.to_owned());
        let message = match command {
            ParsedCommand::Help => {
                self.add_notice(
                    commands::COMMANDS
                        .iter()
                        .map(|spec| {
                            format!("{:<10} {}  ({})", spec.name, spec.description, spec.usage)
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
                return Ok(());
            }
            ParsedCommand::New => InputMessage::NewConversation,
            ParsedCommand::Load { selector } => InputMessage::LoadSession { selector },
            ParsedCommand::Sessions => InputMessage::ListSessions,
            ParsedCommand::Export { destination } => InputMessage::ExportSession { destination },
            ParsedCommand::Compact => InputMessage::CompactSession,
            ParsedCommand::SetModel { provider, model } => {
                InputMessage::SetModel { provider, model }
            }
            ParsedCommand::InvokeSkill { name, alias } => {
                // The echo already shows what was typed; label the alias form
                // so it is clear which skill will run.
                if alias {
                    self.add_notice(format!("{input} · skill: {name}"));
                }
                InputMessage::InvokeSkill { name }
            }
            ParsedCommand::Skills => {
                // Filled asynchronously by the agent's SkillsLoaded event.
                input_tx
                    .send(InputMessage::ListSkills)
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                return Ok(());
            }
        };
        input_tx
            .send(message)
            .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Event application
    // ------------------------------------------------------------------

    fn apply_event(&mut self, event: UiEvent) {
        match event {
            UiEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                self.busy = true;
                self.activity = Activity::Working;
                self.stream().markdown.push_str(&delta);
            }
            UiEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                self.busy = true;
                self.activity = Activity::Reasoning;
                self.stream().reasoning.push_str(&delta);
            }
            UiEvent::ToolCallStarted {
                call_id,
                name,
                summary,
            } => {
                self.busy = true;
                self.activity = Activity::Processing;
                // Text streamed before the call is a complete message.
                self.finalize_stream();
                // Insert the new active record without disturbing existing
                // ones; a duplicate id updates in place (defensive against
                // a malformed duplicate start).
                let record = ToolRecord {
                    name,
                    summary,
                    ok: false,
                    duration_ms: 0,
                    output: String::new(),
                    error: None,
                    status: ToolStatus::Running,
                };
                match self
                    .running_tools
                    .iter_mut()
                    .find(|running| running.call_id == call_id)
                {
                    Some(running) => running.record = record,
                    None => self.running_tools.push(RunningTool { call_id, record }),
                }
            }
            UiEvent::ToolCallFinished {
                call_id,
                name,
                summary,
                ok,
                duration_ms,
                output,
                error,
            } => {
                self.busy = true;
                self.activity = Activity::Working;
                // Remove exactly the matching start; unrelated running tools
                // stay live and keep rendering in the region below.
                if let Some(position) = self
                    .running_tools
                    .iter()
                    .position(|running| running.call_id == call_id)
                {
                    self.running_tools.remove(position);
                }
                let record = ToolRecord {
                    name,
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
                };
                self.pending.push(Entry::Tool { record });
            }
            UiEvent::Retrying { .. } => {
                self.busy = true;
                self.activity = Activity::Retrying;
            }
            UiEvent::Error(error) => {
                self.finalize_stream();
                // Tools still marked running when the turn aborted finalize
                // as failed so their lines do not silently vanish.
                for mut running in self.running_tools.drain(..) {
                    running.record.status = ToolStatus::Failure;
                    running.record.error = Some(error.clone());
                    self.pending.push(Entry::Tool {
                        record: running.record,
                    });
                }
                self.busy = false;
                self.activity = Activity::Preparing;
                self.add_error(error);
            }
            UiEvent::TurnFinished => {
                self.finalize_stream();
                self.running_tools.clear();
                self.busy = false;
                self.activity = Activity::Preparing;
            }
            UiEvent::Notice(notice) => self.add_notice(notice),
            UiEvent::ModelChanged { provider, model } => {
                self.provider = provider;
                self.model = model;
                // The startup header is immutable scrollback; surface the new
                // model as a fresh metadata line instead.
                self.pending.push(self.metadata_entry());
            }
            UiEvent::ModelList { provider, models } => {
                self.model_lists.insert(provider, models);
                if commands::is_command_input(&self.input) {
                    self.refresh_completion();
                }
            }
            UiEvent::UsageUpdated {
                input_tokens,
                output_tokens,
                cost,
                ..
            } => {
                self.usage = Some(Usage {
                    input_tokens,
                    output_tokens,
                    cost,
                });
            }
            UiEvent::SessionChanged { id, loaded, .. } => {
                self.finalize_stream();
                self.running_tools.clear();
                // The terminal keeps everything physically, but the resize
                // repaint redraws from this store, so the session-global
                // chrome (banner, metadata, and the boundary separator that
                // follows) must survive the conversation switch.
                retain_chrome(&mut self.transcript);
                let label = if loaded {
                    format!("loaded session {}", &id[..id.len().min(8)])
                } else {
                    "new conversation".to_owned()
                };
                self.pending.push(Entry::Separator { label });
            }
            UiEvent::SessionSnapshot { entries } => {
                self.finalize_stream();
                self.running_tools.clear();
                // Same chrome retention as `SessionChanged`: the snapshot
                // replaces the conversation, not the header.
                retain_chrome(&mut self.transcript);
                for snapshot in entries {
                    self.pending.push(match snapshot {
                        SessionSnapshotEntry::User { text } => Entry::User { text },
                        SessionSnapshotEntry::Assistant {
                            markdown,
                            reasoning,
                        } => Entry::Assistant {
                            markdown,
                            reasoning,
                        },
                        SessionSnapshotEntry::Tool {
                            name,
                            summary,
                            ok,
                            duration_ms,
                            output,
                            error,
                        } => Entry::Tool {
                            record: ToolRecord {
                                name,
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
                        },
                    });
                }
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
                            format!("{} · {} · {}", session.short_id, title, session.updated_at)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                self.add_notice(notice);
                if commands::is_command_input(&self.input) {
                    self.refresh_completion();
                }
            }
            UiEvent::SessionExported { path } => {
                self.add_notice(format!("exported session to {path}"));
            }
            UiEvent::SkillsLoaded {
                skills,
                diagnostics,
                empty,
            } => {
                if empty {
                    self.add_notice(
                        "no skills discovered\nlooked in .harness/skills and .agents/skills (project and ~)",
                    );
                    return;
                }
                let mut text = skills
                    .iter()
                    .map(|skill| format!("{:<16} {}", skill.name, skill.description))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !diagnostics.is_empty() {
                    text.push_str("\n\ndiagnostics:\n");
                    text.push_str(&diagnostics.join("\n"));
                }
                self.add_notice(text);
            }
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
    }

    fn stream(&mut self) -> &mut StreamState {
        self.stream.get_or_insert_with(StreamState::default)
    }

    /// Commit the in-flight assistant message as a final entry.
    fn finalize_stream(&mut self) {
        let Some(stream) = self.stream.take() else {
            return;
        };
        if stream.reasoning.is_empty() && stream.markdown.is_empty() {
            return;
        }
        self.pending.push(Entry::Assistant {
            markdown: stream.markdown,
            reasoning: stream.reasoning,
        });
    }

    fn add_notice(&mut self, text: impl Into<String>) {
        self.pending.push(Entry::Notice { text: text.into() });
    }

    fn add_error(&mut self, text: impl Into<String>) {
        self.pending.push(Entry::Error { text: text.into() });
    }

    fn metadata_entry(&self) -> Entry {
        Entry::Metadata {
            cwd: self.environment.cwd_display.clone(),
            branch: self.environment.branch.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            context_files: self
                .context_files
                .iter()
                .map(|file| file.path.clone())
                .collect(),
            skills: self.skills.iter().map(|skill| skill.name.clone()).collect(),
        }
    }

    // ------------------------------------------------------------------
    // Frame assembly
    // ------------------------------------------------------------------

    fn input_layout(&self) -> InputLayout {
        let content = render::content_width(self.width);
        let usage_trailer = self.usage.as_ref().map_or_else(String::new, |usage| {
            format!(
                "   ↑ {} ↓ {} · {}",
                format_tokens(usage.input_tokens),
                format_tokens(usage.output_tokens),
                usage.cost
            )
        });
        let input_width = content.saturating_sub(INPUT_PREFIX_WIDTH).max(1);
        let ghost = self.ghost_text();
        let hint = self.completion_hint(input_width);
        input_layout(
            &self.input,
            self.cursor,
            input_width,
            self.theme,
            &usage_trailer,
            &ghost,
            &hint,
        )
    }

    /// Rows of the live region: [streaming tail] · [tool lines] · [activity]
    /// · [input]. Every section except the input is optional; the tail and
    /// the input are each clipped so the whole region fits the screen.
    fn build_region(&self, input: &InputLayout) -> RegionBuild {
        let theme = self.theme;
        let content = render::content_width(self.width);
        let running = self.running_region();
        let (input_rows, input_cursor_row) =
            clip_input(input, self.height as usize, self.busy, running.rows);

        let mut rows: Vec<Line<'static>> = Vec::new();

        // The streaming tail shows only its newest rows; rows that scrolled
        let mut activity_row_index: Option<usize> = None;
        // out of the budget are printed in full when the message finalizes.
        let tail = self.stream_tail_lines(content);
        let budget = self.tail_budget(input_rows.len());
        let start = tail.len().saturating_sub(budget);
        rows.extend(tail[start..].iter().cloned());

        // All active tool records render in deterministic launch order
        // (collapsed to one line each unless expanded globally); anything
        // beyond the visible cap folds into one explicit overflow row so a
        // large fan-out stays discoverable without breaking screen height.
        if !running.lines.is_empty() && !rows.is_empty() {
            render::push_blank(&mut rows, render::SECTION_GAP);
        }
        rows.extend(running.lines);

        if self.busy {
            if !rows.is_empty() {
                render::push_blank(&mut rows, render::SECTION_GAP);
            }
            let activity_row = rows.len();
            rows.push(activity_line(self.activity, self.spinner, theme));
            activity_row_index = Some(activity_row);
        }

        if !rows.is_empty() {
            render::push_blank(&mut rows, render::SECTION_GAP);
        }
        let mut cursor_row = rows.len() + input_cursor_row;
        rows.extend(input_rows);

        // Degenerate-terminal guard: the region must never exceed the screen
        // or cursor-relative moves would clamp at the top and corrupt the
        // frame. Keep the bottom rows (the input lives there).
        if self.height > 0 && rows.len() > self.height as usize {
            let dropped = rows.len() - self.height as usize;
            rows.drain(..dropped);
            cursor_row = cursor_row.saturating_sub(dropped).min(rows.len() - 1);
        }

        let gutter = render::horizontal_pad(self.width) as usize;
        RegionBuild {
            rows,
            cursor_row,
            cursor_col: gutter + INPUT_PREFIX_WIDTH + input.cursor_col,
            activity_row: activity_row_index,
        }
    }

    /// The active-tool section of the live region: rendered rows for the
    /// visible subset of running tools in launch order plus an overflow row
    /// (`… N more running`) when more exist than fit, and the total row count
    /// for the sizing budgets.
    fn running_region(&self) -> RunningRegion {
        let content = render::content_width(self.width);
        let theme = self.theme;
        let hidden = self
            .running_tools
            .len()
            .saturating_sub(MAX_VISIBLE_RUNNING_TOOLS);
        let visible = self.running_tools.len() - hidden;
        let mut lines = Vec::new();
        for running in &self.running_tools[..visible] {
            lines.extend(tool_lines(
                &running.record,
                self.tools_expanded,
                content,
                theme,
            ));
        }
        if hidden > 0 {
            lines.push(Line::from(Span::styled(
                format!("… {hidden} more running"),
                Style::default()
                    .fg(theme.dim_text)
                    .add_modifier(Modifier::DIM),
            )));
        }
        let rows = lines.len();
        RunningRegion { lines, rows }
    }

    /// Rendered rows of the in-flight assistant message (reasoning block,
    /// then markdown). Re-rendered from source on every frame.
    fn stream_tail_lines(&self, width: usize) -> Vec<Line<'static>> {
        let Some(stream) = &self.stream else {
            return Vec::new();
        };
        let mut lines = Vec::new();
        if !stream.reasoning.is_empty() {
            lines.extend(render::reasoning_lines(
                &stream.reasoning,
                self.theme,
                width,
            ));
        }
        if !stream.markdown.is_empty() {
            if !lines.is_empty() {
                render::push_blank(&mut lines, render::BLOCK_GAP);
            }
            lines.extend(render::markdown_lines(&stream.markdown, self.theme, width));
        }
        lines
    }

    /// Row budget for the streaming tail: whatever is left of the screen once
    /// the input, the active-tool rows, the activity row, separators, and a
    /// safety row are reserved.
    fn tail_budget(&self, input_rows: usize) -> usize {
        (self.height as usize)
            .saturating_sub(input_rows + self.running_region().rows + usize::from(self.busy) + 3)
            .max(1)
    }

    /// Move the stable prefix of the streaming markdown into `pending` once
    /// it outgrows the tail budget, so long responses flow into scrollback
    /// incrementally instead of appearing all at once at finalize.
    /// Reasoning-only streams stay fully live; the display clip handles them.
    fn commit_stream_prefix(&mut self, input_rows: usize) {
        // Snapshot the immutable state first so `self.stream` can be borrowed
        // mutably for the rest of the function.
        let budget = self.tail_budget(input_rows);
        let width = render::content_width(self.width);
        let theme = self.theme;
        let Some(stream) = self.stream.as_mut() else {
            return;
        };
        if stream.markdown.is_empty() {
            return;
        }
        let Some(offset) = stable_block_split_offset(&stream.markdown) else {
            return;
        };
        if offset == 0 {
            return;
        }
        let prefix = Entry::Assistant {
            markdown: stream.markdown[..offset].to_owned(),
            reasoning: stream.reasoning.clone(),
        };
        let prefix_height = entry_lines(&prefix, width, theme, self.tools_expanded).len();
        if prefix_height <= budget {
            return;
        }
        self.pending.push(prefix);
        stream.reasoning.clear();
        stream.markdown.drain(..offset);
    }

    // ------------------------------------------------------------------
    // Painting
    // ------------------------------------------------------------------

    /// Repaint the live region in place (plus any pending entries above it).
    fn paint(&mut self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        let input = self.input_layout();
        self.commit_stream_prefix(input.rows.len());
        let build = self.build_region(&input);
        self.write_frame(build, false)
    }

    /// Clear the screen and repaint the visible window of history plus the
    /// live region from scratch. Used after a width change (rows reflow) and
    /// after a height shrink below the painted region. Rows already in
    /// scrollback keep their old wrap; the on-screen window is re-rendered at
    /// the new width from source entries.
    fn repaint_all(&mut self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        self.transcript.append(&mut self.pending);
        let input = self.input_layout();
        self.commit_stream_prefix(input.rows.len());
        let build = self.build_region(&input);
        self.write_frame(build, true)
    }

    /// Emit one frame. In incremental mode the frame is [pending entries] +
    /// [region], rewritten in place from the previous region top; pending
    /// rows are then committed by forgetting them. In `clear_all` mode the
    /// screen is wiped and the visible window of `transcript` is reprinted
    /// above the region.
    fn write_frame(&mut self, build: RegionBuild, clear_all: bool) -> Result<()> {
        let theme = self.theme;
        let content = render::content_width(self.width);
        let gutter = render::horizontal_pad(self.width) as usize;

        let mut above: Vec<Line<'static>> = Vec::new();
        if clear_all {
            for (index, entry) in self.transcript.iter().enumerate() {
                if index > 0 {
                    render::push_blank(&mut above, render::SECTION_GAP);
                }
                above.extend(entry_lines(entry, content, theme, self.tools_expanded));
            }
            // Keep as much history as fits above the region, plus one
            // ellipsis row when older rows fall outside the window.
            let keep = (self.height as usize).saturating_sub(build.rows.len());
            if keep == 0 {
                above.clear();
            } else if above.len() > keep {
                let hidden = above.len() - (keep - 1);
                let mut window = Vec::with_capacity(keep);
                window.push(Line::from(Span::styled(
                    format!("… {hidden} rows above"),
                    Style::default()
                        .fg(theme.dim_text)
                        .add_modifier(Modifier::DIM),
                )));
                window.extend(above.split_off(hidden));
                above = window;
            }
        } else {
            for (index, entry) in self.pending.iter().enumerate() {
                if index > 0 {
                    render::push_blank(&mut above, render::SECTION_GAP);
                }
                above.extend(entry_lines(entry, content, theme, self.tools_expanded));
            }
        }

        let mut rows = above;
        if !rows.is_empty() && !build.rows.is_empty() {
            render::push_blank(&mut rows, render::SECTION_GAP);
        }
        rows.extend(build.rows.iter().cloned());
        let total = rows.len();
        // The cursor target, counted from the first printed row.
        let cursor_abs = total - build.rows.len() + build.cursor_row;

        let mut buffer = String::new();
        if clear_all {
            let _ = write!(buffer, "{}{}", Clear(ClearType::All), MoveTo(0, 0));
        } else if !self.region.is_empty() {
            // Walk the cursor up to the top of the previously painted region.
            // The cursor sits at `cursor_row` inside the region, so exactly
            // that many rows separate it from the top.
            if self.cursor_row > 0 {
                let _ = write!(buffer, "{}", MoveUp(self.cursor_row as u16));
            }
        }

        for (index, line) in rows.iter().enumerate() {
            let _ = write!(buffer, "\r{}", Clear(ClearType::UntilNewLine));
            write_row(&mut buffer, line, gutter);
            if index + 1 < total {
                buffer.push('\n');
            }
        }

        // When the frame shrank, erase the leftover rows below it so stale
        // pixels do not linger. (Pending rows only ever grow the frame, so
        // this fires when the live region itself got smaller.)
        if !clear_all && total < self.region.len() {
            for _ in total..self.region.len() {
                let _ = write!(buffer, "{}\r{}", MoveDown(1), Clear(ClearType::CurrentLine));
            }
            let up = (self.region.len() - 1).saturating_sub(cursor_abs);
            if up > 0 {
                let _ = write!(buffer, "{}", MoveUp(up as u16));
            }
        } else {
            let up = (total - 1).saturating_sub(cursor_abs);
            if up > 0 {
                let _ = write!(buffer, "{}", MoveUp(up as u16));
            }
        }

        // Horizontal placement: column 0 plus the input cursor offset.
        buffer.push('\r');
        if build.cursor_col > 0 {
            let col = build.cursor_col.min(u16::MAX as usize) as u16;
            let _ = write!(buffer, "{}", MoveRight(col));
        }

        self.out
            .write_all(buffer.as_bytes())
            .context("write frame")?;
        self.out.flush().context("flush frame")?;

        // Bookkeeping: the printed `above` rows are now immutable scrollback.
        self.transcript.append(&mut self.pending);
        self.region = build.rows;
        self.cursor_row = build.cursor_row;
        self.cursor_col = build.cursor_col;
        // The frame just painted decides whether an activity row exists for
        // spinner ticks to target; a frame without one invalidates the
        // previous row's position.
        self.activity_region_row = build.activity_row;
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        terminal::disable_raw_mode().context("restore terminal raw mode")?;
        execute!(self.out, DisableBracketedPaste).context("restore terminal input")?;
        let _ = execute!(self.out, PopKeyboardEnhancementFlags);
        // Leave one blank line so the shell prompt lands below the UI.
        writeln!(self.out).context("leave terminal")?;
        Ok(())
    }
}

impl Drop for CrossTerm {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Merge two candidate lists, deduplicating by value while preserving the
/// first list's order. Used to combine static session candidates with the
/// scanned path results for command arguments.
fn merge_candidates(primary: Vec<Candidate>, secondary: Vec<Candidate>) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    primary
        .into_iter()
        .chain(secondary)
        .filter(|candidate| seen.insert(candidate.value.clone()))
        .collect()
}

/// Keep the session-global chrome when a conversation switch replaces the
/// transcript: the startup banner, metadata lines, and the most recent
/// boundary separator. Everything else was already printed into scrollback;
/// it only loses its place in the resize-repaint source, not on screen.
fn retain_chrome(transcript: &mut Vec<Entry>) {
    let last_separator = transcript
        .iter()
        .rposition(|entry| matches!(entry, Entry::Separator { .. }));
    let mut index = 0usize;
    transcript.retain(|entry| {
        let keep = match entry {
            Entry::Banner { .. } | Entry::Metadata { .. } => true,
            Entry::Separator { .. } => Some(index) == last_separator,
            _ => false,
        };
        index += 1;
        keep
    });
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
        let _ = writeln!(io::stdout());
        previous(panic);
    }));
}

// ---------------------------------------------------------------------------
// Entry rendering
// ---------------------------------------------------------------------------

/// Build the wrapped rows for one final or in-flight entry. Shared by the
/// pending path (print above the region) and the resize repaint (window of
/// history), which is why entries are stored at the source level.
/// `tools_expanded` is the global Ctrl+O state; committed entries render
/// collapsed unless the toggle is on, and the progress snapshot used the
/// same global for its in-flight tool line so both agree after a repaint.
fn entry_lines(
    entry: &Entry,
    width: usize,
    theme: Theme,
    tools_expanded: bool,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    match entry {
        Entry::Banner { tagline } => render::welcome_lines_with_tagline(width, theme, tagline),
        Entry::Metadata {
            cwd,
            branch,
            provider,
            model,
            context_files,
            skills,
        } => metadata_lines(
            cwd,
            branch.as_deref(),
            provider,
            model,
            context_files,
            skills,
            theme,
        ),
        Entry::User { text } => render::user_lines(text, theme, width),
        Entry::Assistant {
            markdown,
            reasoning,
        } => {
            let mut lines = Vec::new();
            if !reasoning.is_empty() {
                lines.extend(render::reasoning_lines(reasoning, theme, width));
            }
            if !markdown.is_empty() {
                if !lines.is_empty() {
                    render::push_blank(&mut lines, render::BLOCK_GAP);
                }
                lines.extend(render::markdown_lines(markdown, theme, width));
            }
            lines
        }
        Entry::Tool { record } => tool_lines(record, tools_expanded, width, theme),
        Entry::Notice { text } => render::notice_lines(text, theme, width),
        Entry::Error { text } => render::error_lines(text, theme, width),
        Entry::Separator { label } => vec![separator_line(label, width, theme)],
    }
}

/// The simplified tool call: one line with the tool type and its primary
/// parameter — `$ git status`, `read file.txt` — plus a duration on success
/// or a `✗` error preview on failure. Collapsed this is the compact line;
/// expanded it adds the bounded output tail, the first error line, and a
/// `running…` marker, each indented two spaces.
fn tool_lines(
    record: &ToolRecord,
    expanded: bool,
    width: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    // Agent summaries read `bash: <command>` / `read <path>` / …; bash reads
    // as a shell line, every other tool as `<verb> <param>` with the verb in
    // the same accent as the `$` so all tool lines share one signature.
    let marker_style = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    let command = record
        .summary
        .strip_prefix("bash:")
        .map(|rest| rest.trim_start())
        .filter(|rest| !rest.is_empty());
    match command {
        Some(command) => {
            spans.push(Span::styled("$ ", marker_style));
            spans.push(Span::styled(
                command.to_owned(),
                Style::default().fg(theme.primary_text),
            ));
        }
        None => {
            let summary = record.summary.trim();
            let (verb, rest) = summary.split_once(' ').unwrap_or((summary, ""));
            spans.push(Span::styled(verb.to_owned(), marker_style));
            if !rest.is_empty() {
                spans.push(Span::styled(
                    format!(" {rest}"),
                    Style::default().fg(theme.primary_text),
                ));
            }
        }
    }
    match record.status {
        ToolStatus::Running => {}
        ToolStatus::Success => spans.push(Span::styled(
            format!(" · {}", render::duration_text(record.duration_ms)),
            Style::default().fg(theme.dim_text),
        )),
        ToolStatus::Failure => {
            let preview = record
                .error
                .as_deref()
                .and_then(|error| error.lines().next())
                .unwrap_or("failed");
            spans.push(Span::styled("  ✗ ", Style::default().fg(theme.error)));
            spans.push(Span::styled(
                preview.to_owned(),
                Style::default().fg(theme.error),
            ));
        }
    }
    // The marker/verb is part of the spans, so no extra message prefix is
    // applied; the wrapper only reflows the line to the available width.
    let summary_lines = render::wrap_text(
        &Text::from(Line::from(spans)),
        width.saturating_sub(INPUT_PREFIX_WIDTH).max(1),
        Style::default(),
    );
    if !expanded {
        return summary_lines;
    }

    // Expanded: the compact line, then the bounded output tail, the first
    // error line, and a trailing `running…` marker, each indented two spaces
    // and dimmed so they read as details under the summary.
    let dim_style = Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM);
    let mut lines = summary_lines;
    if !record.output.is_empty() {
        for tail in render::output_tail(&record.output) {
            push_detail_line(&mut lines, &tail, dim_style, width);
        }
    }
    if let Some(error) = record.error.as_deref()
        && let Some(first) = error.lines().next()
    {
        push_detail_line(&mut lines, first, Style::default().fg(theme.error), width);
    }
    if matches!(record.status, ToolStatus::Running) {
        push_detail_line(&mut lines, "running…", dim_style, width);
    }
    lines
}

/// One indented detail row under an expanded tool line, wrapped to the
/// content width with the given style applied to every span.
fn push_detail_line(lines: &mut Vec<Line<'static>>, text: &str, style: Style, width: usize) {
    for wrapped in render::wrap_text(
        &Text::from(Line::from(Span::styled(text.to_owned(), style))),
        width.saturating_sub(INPUT_PREFIX_WIDTH).max(1),
        Style::default(),
    ) {
        lines.push(Line::from(
            std::iter::once(Span::raw("  "))
                .chain(wrapped.spans)
                .collect::<Vec<_>>(),
        ));
    }
}

/// How many context/skill entries each header row shows before folding into
/// `+N more`. Skills are cheap to accumulate (one per directory), so the cap
/// keeps a skill-heavy workspace from pushing the transcript out of view.
const METADATA_MAX_ENTRIES: usize = 4;

fn fold_entries(entries: &[String]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let shown = entries.len().min(METADATA_MAX_ENTRIES);
    let mut text = entries[..shown].join(", ");
    let hidden = entries.len() - shown;
    if hidden > 0 {
        let _ = write!(text, " · +{hidden} more");
    }
    Some(text)
}

/// The header metadata: two dim, left-aligned rows — `cwd  (branch)` on the
/// first, `provider · model` on the second — replacing the old right-aligned
/// split so both lines read as plain left-aligned chrome above the transcript.
/// Two optional rows follow when project context or skills were auto-loaded:
/// `context: …` and `skills: …`, capped at [`METADATA_MAX_ENTRIES`] names.
fn metadata_lines(
    cwd: &str,
    branch: Option<&str>,
    provider: &str,
    model: &str,
    context_files: &[String],
    skills: &[String],
    theme: Theme,
) -> Vec<Line<'static>> {
    let style = Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM);
    let left = match branch {
        Some(branch) => format!("{cwd}  ({branch})"),
        None => cwd.to_owned(),
    };
    let mut lines = vec![
        Line::from(Span::styled(left, style)),
        Line::from(Span::styled(format!("{provider} \u{b7} {model}"), style)),
    ];
    for (label, entries) in [("context", context_files), ("skills", skills)] {
        if let Some(text) = fold_entries(entries) {
            lines.push(Line::from(Span::styled(format!("{label}: {text}"), style)));
        }
    }
    lines
}

/// A centered `── label ──` rule marking a conversation boundary.
fn separator_line(label: &str, width: usize, theme: Theme) -> Line<'static> {
    let text = format!("── {label} ──");
    let padding = width.saturating_sub(UnicodeWidthStr::width(text.as_str()));
    let left = padding / 2;
    let right = padding - left;
    Line::from(Span::styled(
        format!("{}{}{}", "─".repeat(left), text, "─".repeat(right)),
        Style::default().fg(theme.dim_text),
    ))
}

/// The busy indicator row: an animated marker plus the activity label.
fn activity_line(activity: Activity, spinner: usize, theme: Theme) -> Line<'static> {
    let marker = render::ACTIVITY_FRAMES[spinner % render::ACTIVITY_FRAMES.len()];
    Line::from(vec![
        Span::styled(format!("{marker} "), Style::default().fg(theme.accent)),
        Span::styled(
            activity.label(),
            Style::default()
                .fg(theme.dim_text)
                .add_modifier(Modifier::DIM),
        ),
    ])
}

// ---------------------------------------------------------------------------
// Input line layout
// ---------------------------------------------------------------------------

/// Wrap the input draft into visual rows (first row prefixed with `› `,
/// continuations indented to match) and locate the cursor inside them. The
/// real terminal cursor is placed at `cursor_col` columns into `cursor_row`.
///
/// `usage_trailer` is appended, dim, to the empty-input placeholder row;
/// nothing is shown while typing (the counter reappears after a submit
/// because the input empties again). `ghost` is the fish-style dim suffix
/// preview painted after the cursor (only at end-of-input, clamped to the row
/// width), and `completion_hint` is one dim candidate row above the input.
fn input_layout(
    input: &str,
    cursor: usize,
    width: usize,
    theme: Theme,
    usage_trailer: &str,
    ghost: &str,
    completion_hint: &str,
) -> InputLayout {
    let width = width.max(1);
    let cursor = cursor.min(input.len());
    let prefix_style = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    let text_style = Style::default().fg(theme.primary_text);
    let dim_style = Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM);

    if input.is_empty() {
        // Placeholder row; the terminal cursor sits right after the prefix.
        // When usage is available the counter trailer trails it, dim.
        let mut spans = vec![
            Span::styled(INPUT_PREFIX, prefix_style),
            Span::styled(PLACEHOLDER, dim_style),
        ];
        if !usage_trailer.is_empty() {
            spans.push(Span::styled(usage_trailer.to_owned(), dim_style));
        }
        let mut rows = vec![Line::from(spans)];
        if !completion_hint.is_empty() {
            rows.insert(
                0,
                Line::from(Span::styled(completion_hint.to_owned(), dim_style)),
            );
        }
        return InputLayout {
            rows,
            cursor_row: usize::from(!completion_hint.is_empty()),
            cursor_col: 0,
        };
    }

    let mut rows: Vec<Line<'static>> = Vec::new();
    let mut cursor_row = 0usize;
    let mut cursor_col = 0usize;

    let mut byte_base = 0usize;
    for logical in input.split('\n') {
        let line_end = byte_base + logical.len();
        let cursor_here = cursor >= byte_base && cursor <= line_end;
        let chars: Vec<(char, usize)> = logical
            .chars()
            .map(|c| (c, UnicodeWidthChar::width(c).unwrap_or(1).max(1)))
            .collect();
        let cursor_char = if cursor_here {
            Some(input[byte_base..cursor].chars().count())
        } else {
            None
        };

        // Greedy wrap; rows are append-stable, so the cursor can be located
        // afterwards by walking the row boundaries.
        let mut visual: Vec<Vec<(char, usize)>> = Vec::new();
        let mut current: Vec<(char, usize)> = Vec::new();
        let mut current_width = 0usize;
        for &(character, character_width) in &chars {
            if current_width > 0 && current_width + character_width > width {
                visual.push(std::mem::take(&mut current));
                current_width = 0;
            }
            current.push((character, character_width));
            current_width += character_width;
        }
        visual.push(current);

        if let Some(cursor_char) = cursor_char {
            if cursor_char >= chars.len() {
                // End of this logical line. A row filled to exactly the wrap
                // width gets a fresh row for the cursor, mirroring
                // `render::prompt_layout`.
                let last = visual.len() - 1;
                let last_width: usize = visual[last].iter().map(|(_, w)| *w).sum();
                if last_width >= width && !visual[last].is_empty() {
                    cursor_row = rows.len() + visual.len();
                    cursor_col = 0;
                    visual.push(Vec::new());
                } else {
                    cursor_row = rows.len() + last;
                    cursor_col = last_width;
                }
            } else {
                let mut index = 0usize;
                for (row_index, row) in visual.iter().enumerate() {
                    if index + row.len() > cursor_char {
                        cursor_row = rows.len() + row_index;
                        cursor_col = row.iter().take(cursor_char - index).map(|(_, w)| *w).sum();
                        break;
                    }
                    index += row.len();
                }
            }
        }

        for (row_index, row) in visual.iter().enumerate() {
            let text: String = row.iter().map(|(c, _)| *c).collect();
            let lead = if rows.is_empty() && row_index == 0 {
                Span::styled(INPUT_PREFIX, prefix_style)
            } else {
                Span::raw(INPUT_CONTINUATION)
            };
            rows.push(Line::from(vec![lead, Span::styled(text, text_style)]));
        }

        byte_base = line_end + 1;
    }

    if rows.is_empty() {
        rows.push(Line::from(Span::styled(INPUT_PREFIX, prefix_style)));
    }

    // The list hint is one dim row above the input.
    if !completion_hint.is_empty() {
        rows.insert(
            0,
            Line::from(Span::styled(completion_hint.to_owned(), dim_style)),
        );
        cursor_row += 1;
    }
    // Ghost preview: append the dim suffix after the typed text on the
    // cursor's row. Only at end-of-input, clamped to the remaining width so
    // the row never overflows; the cursor position is untouched.
    if !ghost.is_empty()
        && cursor >= input.len()
        && let Some(row) = rows.get_mut(cursor_row)
    {
        let used = row
            .spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum();
        let available = width.saturating_sub(used);
        let fits = ghost_to_width(ghost, available);
        if !fits.is_empty() {
            row.spans.push(Span::styled(fits, dim_style));
        }
    }

    InputLayout {
        rows,
        cursor_row,
        cursor_col,
    }
}

/// Fit completion values onto exactly one visual row. Candidates begin at the
/// highlighted item and wrap around, so the value Tab will accept is never
/// hidden behind the `… +N` counter.
fn fit_completion_hint(candidates: &[Candidate], selected: usize, width: usize) -> String {
    if candidates.is_empty() || width == 0 {
        return String::new();
    }
    let selected = selected.min(candidates.len() - 1);
    let values = candidates[selected..]
        .iter()
        .chain(&candidates[..selected])
        .map(|candidate| candidate.value.as_str())
        .collect::<Vec<_>>();

    let mut best = String::new();
    for shown in 1..=values.len() {
        let mut hint = values[..shown].join(" · ");
        let hidden = values.len() - shown;
        if hidden > 0 {
            let _ = write!(hint, " … +{hidden}");
        }
        if UnicodeWidthStr::width(hint.as_str()) > width {
            break;
        }
        best = hint;
    }
    if !best.is_empty() {
        return best;
    }

    // Even one provider-qualified ID can be wider than a small terminal.
    // Reserve the last cell for an ellipsis rather than letting the terminal
    // soft-wrap a row the live-region model intentionally counts as one.
    if width == 1 {
        return "…".into();
    }
    let mut hint = ghost_to_width(values[0], width - 1);
    hint.push('…');
    hint
}

/// Truncate `text` (in display columns) to fit `width`, only ever cutting at
/// whole characters. Returns empty when nothing fits.
fn ghost_to_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut result = String::new();
    let mut used = 0usize;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(1).max(1);
        if used + character_width > width {
            break;
        }
        result.push(character);
        used += character_width;
    }
    result
}

/// The byte offset of the `column`-th character (0-based) into `input`.
fn byte_index_at_char(input: &str, column: usize) -> usize {
    input
        .char_indices()
        .nth(column)
        .map(|(index, _)| index)
        .unwrap_or(input.len())
}

/// Clip the input rows to a window containing the cursor when the draft is
/// taller than the screen can show. Returns the visible rows and the cursor
/// row within them.
fn clip_input(
    input: &InputLayout,
    height: usize,
    busy: bool,
    tool_rows: usize,
) -> (Vec<Line<'static>>, usize) {
    let cap = height
        .saturating_sub(usize::from(busy) + tool_rows + 2)
        .max(1);
    if input.rows.len() <= cap {
        return (input.rows.clone(), input.cursor_row);
    }
    // Window pinned so the cursor row stays visible, biased toward showing
    // as much above it as fits.
    let start = input
        .cursor_row
        .saturating_sub(cap - 1)
        .min(input.rows.len() - cap);
    (input.rows[start..].to_vec(), input.cursor_row - start)
}

// ---------------------------------------------------------------------------
// Input editing (pure helpers over `String` + byte cursor)
// ---------------------------------------------------------------------------

/// Insert text at the cursor, normalizing CRLF/CR paste payloads to `\n`.
fn insert_text(input: &mut String, cursor: &mut usize, text: &str) {
    if text.is_empty() {
        return;
    }
    let text = if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_owned()
    };
    input.insert_str(*cursor, &text);
    *cursor += text.len();
}

fn delete_backward(input: &mut String, cursor: &mut usize) {
    if let Some(len) = input[..*cursor].chars().next_back().map(char::len_utf8) {
        *cursor -= len;
        input.drain(*cursor..*cursor + len);
    }
}

fn delete_forward(input: &mut String, cursor: &mut usize) {
    if let Some(len) = input[*cursor..].chars().next().map(char::len_utf8) {
        input.drain(*cursor..*cursor + len);
    }
}

fn move_left(input: &str, cursor: &mut usize) {
    if let Some(len) = input[..*cursor].chars().next_back().map(char::len_utf8) {
        *cursor -= len;
    }
}

fn move_right(input: &str, cursor: &mut usize) {
    if let Some(len) = input[*cursor..].chars().next().map(char::len_utf8) {
        *cursor += len;
    }
}

/// Byte offsets of the start and end of the logical line containing `cursor`.
fn line_bounds(input: &str, cursor: usize) -> (usize, usize) {
    let start = input[..cursor].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = input[cursor..]
        .find('\n')
        .map(|i| cursor + i)
        .unwrap_or(input.len());
    (start, end)
}

/// Move the cursor one logical line up (`delta < 0`) or down, preserving the
/// character column. Returns `None` at the outer edges.
fn vertical_move(input: &str, cursor: usize, delta: i32) -> Option<usize> {
    let before = &input[..cursor];
    let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let column = before[line_start..].chars().count();
    let line_index = before.matches('\n').count() as i32;
    let total_lines = input.matches('\n').count() as i32 + 1;
    let target = line_index + delta;
    if target < 0 || target >= total_lines {
        return None;
    }
    let mut position = 0usize;
    for _ in 0..target {
        position += input[position..].find('\n').expect("target line exists") + 1;
    }
    let line_end = input[position..]
        .find('\n')
        .map(|i| position + i)
        .unwrap_or(input.len());
    let byte_column: usize = input[position..line_end]
        .chars()
        .take(column)
        .map(char::len_utf8)
        .sum();
    Some(position + byte_column)
}

/// Compact token counts the way the context-length formatter does: `1_234`
/// renders as `1.2k`, `1_000_000` as `1M`.
fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        let value = n as f64 / 1_000_000.0;
        if value.fract() == 0.0 {
            format!("{}M", value as u64)
        } else {
            format!("{value:.1}M")
        }
    } else if n >= 1_000 {
        let value = n as f64 / 1_000.0;
        if value.fract() == 0.0 {
            format!("{}k", value as u64)
        } else {
            format!("{value:.1}k")
        }
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------------------
// ANSI output
// ---------------------------------------------------------------------------

/// Serialize one styled ratatui line to ANSI. Ratatui styles describe each
/// span independently, while terminal attributes persist until reset, so a
/// reset is required when adjacent effective styles differ. Without it the
/// input prefix's bold attribute leaks into the first text row even though
/// continuation rows correctly use normal weight.
fn line_to_ansi(line: &Line<'_>) -> String {
    let mut out = String::new();
    let mut previous = None;
    let mut styled = false;
    for span in &line.spans {
        if span.content.is_empty() {
            continue;
        }
        let style = line.style.patch(span.style);
        if previous.is_some_and(|previous| previous != style) {
            out.push_str("\u{1b}[0m");
            styled = true;
        }
        if previous != Some(style) {
            let prefix = style_prefix(&style);
            styled |= !prefix.is_empty();
            out.push_str(&prefix);
        }
        out.push_str(span.content.as_ref());
        previous = Some(style);
    }
    if styled {
        out.push_str("\u{1b}[0m");
    }
    out
}

fn style_prefix(style: &Style) -> String {
    let mut out = String::new();
    if let Some(color) = style.fg {
        let _ = write!(out, "{}", SetForegroundColor(ansi_color(color)));
    }
    if let Some(color) = style.bg {
        let _ = write!(out, "{}", SetBackgroundColor(ansi_color(color)));
    }
    for (modifier, attribute) in MODIFIER_ATTRIBUTES {
        if style.add_modifier.contains(*modifier) {
            let _ = write!(out, "{}", SetAttribute(*attribute));
        }
    }
    out
}

/// ratatui → crossterm modifier pairs for [`style_prefix`].
const MODIFIER_ATTRIBUTES: &[(Modifier, Attribute)] = &[
    (Modifier::BOLD, Attribute::Bold),
    (Modifier::DIM, Attribute::Dim),
    (Modifier::ITALIC, Attribute::Italic),
    (Modifier::UNDERLINED, Attribute::Underlined),
    (Modifier::SLOW_BLINK, Attribute::SlowBlink),
    (Modifier::RAPID_BLINK, Attribute::RapidBlink),
    (Modifier::REVERSED, Attribute::Reverse),
    (Modifier::HIDDEN, Attribute::Hidden),
    (Modifier::CROSSED_OUT, Attribute::CrossedOut),
];

/// ratatui-core → crossterm color mapping, mirroring ratatui's own
/// crossterm-backend mapping so the theme renders identically.
fn ansi_color(color: ratatui_core::style::Color) -> AnsiColor {
    use ratatui_core::style::Color as CoreColor;
    match color {
        CoreColor::Reset => AnsiColor::Reset,
        CoreColor::Black => AnsiColor::Black,
        CoreColor::Red => AnsiColor::DarkRed,
        CoreColor::Green => AnsiColor::DarkGreen,
        CoreColor::Yellow => AnsiColor::DarkYellow,
        CoreColor::Blue => AnsiColor::DarkBlue,
        CoreColor::Magenta => AnsiColor::DarkMagenta,
        CoreColor::Cyan => AnsiColor::DarkCyan,
        CoreColor::Gray => AnsiColor::Grey,
        CoreColor::DarkGray => AnsiColor::DarkGrey,
        CoreColor::LightRed => AnsiColor::Red,
        CoreColor::LightGreen => AnsiColor::Green,
        CoreColor::LightYellow => AnsiColor::Yellow,
        CoreColor::LightBlue => AnsiColor::Blue,
        CoreColor::LightMagenta => AnsiColor::Magenta,
        CoreColor::LightCyan => AnsiColor::Cyan,
        CoreColor::White => AnsiColor::White,
        CoreColor::Rgb(r, g, b) => AnsiColor::Rgb { r, g, b },
        CoreColor::Indexed(value) => AnsiColor::AnsiValue(value),
    }
}

/// Write one row: the shared left gutter plus the line's ANSI serialization.
/// Blank rows become gutter-width space runs, which render identically.
fn write_row(buffer: &mut String, line: &Line<'_>, gutter: usize) {
    if gutter > 0 {
        buffer.push_str(&" ".repeat(gutter));
    }
    buffer.push_str(&line_to_ansi(line));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;

    fn ui(width: u16, height: u16) -> CrossTerm {
        CrossTerm::base(
            "test-model",
            "test-provider",
            vec!["opencode-go".into(), "openrouter".into()],
            Vec::new(),
            Vec::new(),
            width,
            height,
        )
    }

    fn row_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    fn record(status: ToolStatus) -> ToolRecord {
        ToolRecord {
            name: "bash".into(),
            summary: "bash: cargo test".into(),
            ok: !matches!(status, ToolStatus::Failure),
            duration_ms: 1_200,
            output: String::new(),
            error: matches!(status, ToolStatus::Failure).then(|| "boom\ntrace".into()),
            status,
        }
    }

    fn start(id: &str, summary: &str) -> UiEvent {
        UiEvent::ToolCallStarted {
            call_id: id.into(),
            name: "subagent".into(),
            summary: summary.into(),
        }
    }

    fn finish(id: &str, summary: &str) -> UiEvent {
        UiEvent::ToolCallFinished {
            call_id: id.into(),
            name: "subagent".into(),
            summary: summary.into(),
            ok: true,
            duration_ms: 10,
            output: "report".into(),
            error: None,
        }
    }

    fn active_summaries(ui: &CrossTerm) -> Vec<String> {
        ui.running_tools
            .iter()
            .map(|running| running.record.summary.clone())
            .collect()
    }

    #[test]
    fn concurrent_starts_accumulate_and_finishes_remove_by_call_id() {
        let mut ui = ui(80, 24);
        // Two same-name subagents with similar summaries — correlation must
        // be by call id, not by name or FIFO position.
        ui.apply_event(start("a", "subagent: audit one"));
        ui.apply_event(start("b", "subagent: audit two"));
        assert_eq!(
            active_summaries(&ui),
            vec!["subagent: audit one", "subagent: audit two"]
        );

        // Finish out of order: B first, A second.
        ui.apply_event(finish("b", "subagent: audit two"));
        assert_eq!(active_summaries(&ui), vec!["subagent: audit one"]);
        ui.apply_event(finish("a", "subagent: audit one"));
        assert!(ui.running_tools.is_empty());

        // Both completions were committed as tool entries.
        let committed = ui
            .pending
            .iter()
            .filter(|entry| matches!(entry, Entry::Tool { .. }))
            .count();
        assert_eq!(committed, 2);
    }

    #[test]
    fn duplicate_start_updates_in_place_instead_of_duplicating() {
        let mut ui = ui(80, 24);
        ui.apply_event(start("a", "first"));
        ui.apply_event(start("a", "second"));
        assert_eq!(active_summaries(&ui), vec!["second"]);
    }

    #[test]
    fn error_drains_all_active_tools_as_failed_and_turn_finished_clears() {
        let mut failing = ui(80, 24);
        failing.apply_event(start("a", "one"));
        failing.apply_event(start("b", "two"));
        failing.apply_event(UiEvent::Error("turn blew up".into()));
        assert!(failing.running_tools.is_empty());
        let failed = failing
            .pending
            .iter()
            .filter_map(|entry| match entry {
                Entry::Tool { record } => Some(record),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(failed.len(), 2);
        assert!(
            failed
                .iter()
                .all(|record| record.status == ToolStatus::Failure)
        );

        // TurnFinished clears any remaining actives without committing them.
        let mut fresh = ui(80, 24);
        fresh.apply_event(start("a", "one"));
        fresh.apply_event(UiEvent::TurnFinished);
        assert!(fresh.running_tools.is_empty());
    }

    #[test]
    fn region_renders_every_active_tool_plus_overflow_line() {
        let mut ui = ui(80, 24);
        for index in 0..6 {
            ui.apply_event(start(
                &format!("call-{index}"),
                &format!("subagent: task {index}"),
            ));
        }
        let input = ui.input_layout();
        let build = ui.build_region(&input);
        let text = build
            .rows
            .iter()
            .map(row_text)
            .collect::<Vec<_>>()
            .join("\n");
        // Four visible in launch order, the rest folded into the overflow.
        assert!(text.contains("subagent: task 0"));
        assert!(text.contains("subagent: task 3"));
        assert!(text.contains("… 2 more running"));
        // The whole region still fits the terminal height.
        assert!(build.rows.len() <= 24);
    }

    #[test]
    fn format_tokens_uses_k_and_m_units() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_000), "1k");
        assert_eq!(format_tokens(1_234), "1.2k");
        assert_eq!(format_tokens(12_345), "12.3k");
        assert_eq!(format_tokens(1_000_000), "1M");
        assert_eq!(format_tokens(1_500_000), "1.5M");
    }

    #[test]
    fn placeholder_row_shows_the_usage_trailer_only_when_presented() {
        let layout = input_layout(
            "",
            0,
            60,
            Theme::default(),
            "   ↑ 1.2k ↓ 3.4k · $0.01",
            "",
            "",
        );
        assert_eq!(
            row_text(&layout.rows[0]),
            "› Type your message...   ↑ 1.2k ↓ 3.4k · $0.01"
        );

        // Without usage the placeholder stays bare.
        let layout = input_layout("", 0, 60, Theme::default(), "", "", "");
        assert_eq!(row_text(&layout.rows[0]), "› Type your message...");
    }

    #[test]
    fn usage_updated_is_kept_for_the_placeholder_counter() {
        let mut ui = ui(80, 24);
        ui.apply_event(UiEvent::UsageUpdated {
            input_tokens: 1_234,
            output_tokens: 3_456,
            cached_tokens: 5,
            reasoning_tokens: 2,
            cost: "$0.01".into(),
        });
        let layout = ui.input_layout();
        assert_eq!(
            row_text(&layout.rows[0]),
            "› Type your message...   ↑ 1.2k ↓ 3.5k · $0.01"
        );
    }

    #[test]
    fn input_layout_wraps_and_places_the_cursor() {
        // "abcdefgh" at width 4 wraps into two full rows; a cursor at the
        // very end lands on a fresh third row (mirrors prompt_layout).
        let layout = input_layout("abcdefgh", 8, 4, Theme::default(), "", "", "");
        let values: Vec<String> = layout.rows.iter().map(row_text).collect();
        assert_eq!(values, vec!["› abcd", "  efgh", "  "]);
        assert_eq!(layout.cursor_row, 2);
        assert_eq!(layout.cursor_col, 0);

        // Cursor mid-word stays on the first row at the right column.
        let layout = input_layout("abcdef", 2, 4, Theme::default(), "", "", "");
        assert_eq!(layout.cursor_row, 0);
        assert_eq!(layout.cursor_col, 2);
    }

    #[test]
    fn input_layout_handles_multiline_drafts() {
        // Two logical lines, cursor at the end of the second: continuation
        // rows are indented to line up under the prefix.
        let layout = input_layout("ab\ncd", 5, 10, Theme::default(), "", "", "");
        let values: Vec<String> = layout.rows.iter().map(row_text).collect();
        assert_eq!(values, vec!["› ab", "  cd"]);
        assert_eq!(layout.cursor_row, 1);
        assert_eq!(layout.cursor_col, 2);

        // Empty input shows the placeholder with the cursor after `› `.
        let layout = input_layout("", 0, 10, Theme::default(), "", "", "");
        assert_eq!(row_text(&layout.rows[0]), format!("› {PLACEHOLDER}"));
        assert_eq!(layout.cursor_row, 0);
        assert_eq!(layout.cursor_col, 0);
    }

    #[test]
    fn input_layout_counts_wide_characters_by_display_width() {
        // Two CJK characters fill a width-4 row; the cursor after the first
        // sits at display column 2, not byte offset 3.
        let layout = input_layout("你好", 3, 4, Theme::default(), "", "", "");
        assert_eq!(layout.cursor_row, 0);
        assert_eq!(layout.cursor_col, 2);
    }

    #[test]
    fn tool_lines_are_one_shell_style_line() {
        // Running: just the command with the `$` marker.
        let lines = tool_lines(&record(ToolStatus::Running), false, 60, Theme::default());
        assert_eq!(lines.len(), 1);
        assert_eq!(row_text(&lines[0]), "$ cargo test");

        // Success: duration suffix.
        let lines = tool_lines(&record(ToolStatus::Success), false, 60, Theme::default());
        assert_eq!(row_text(&lines[0]), "$ cargo test · 1.2s");

        // Failure: cross and the first error line only.
        let lines = tool_lines(&record(ToolStatus::Failure), false, 60, Theme::default());
        assert_eq!(row_text(&lines[0]), "$ cargo test  ✗ boom");

        // Non-bash tools keep their `tool param` summary shape, with the
        // verb accented like the `$` marker.
        let mut read = record(ToolStatus::Success);
        read.summary = "read src/main.rs".into();
        let lines = tool_lines(&read, false, 60, Theme::default());
        assert_eq!(row_text(&lines[0]), "read src/main.rs · 1.2s");
        assert!(matches!(
            lines[0].spans[0],
            Span { .. }
            if lines[0].spans[0].style.fg == Some(Theme::default().accent)
        ));
        assert_eq!(lines[0].spans[0].content, "read");
        assert_eq!(lines[0].spans[1].content, " src/main.rs");

        // A single-word summary (no parameter) is all accent, no crash.
        let mut bare = record(ToolStatus::Running);
        bare.summary = "compact".into();
        let lines = tool_lines(&bare, false, 60, Theme::default());
        assert_eq!(row_text(&lines[0]), "compact");
        assert_eq!(lines[0].spans.len(), 1);
    }

    #[test]
    fn tool_lines_expanded_show_output_tail_and_error() {
        // A finished tool with output and an error: expanded shows the
        // bounded tail plus the first error line, indented.
        let mut failed = record(ToolStatus::Failure);
        failed.output = (0..10)
            .map(|i| format!("out line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let lines = tool_lines(&failed, true, 60, Theme::default());
        // 1 summary + 4 tail lines (… N above + last 4) + 1 error line.
        assert_eq!(lines.len(), 1 + 5 + 1);
        assert_eq!(row_text(&lines[0]), "$ cargo test  ✗ boom");
        assert_eq!(row_text(&lines[1]), "  … 6 lines above");
        assert_eq!(row_text(&lines[5]), "  out line 9");
        assert_eq!(row_text(&lines[6]), "  boom");

        // Collapsed stays compact regardless of retained output.
        let collapsed = tool_lines(&failed, false, 60, Theme::default());
        assert_eq!(collapsed.len(), 1);

        // A running tool gains a trailing `running…` marker when expanded.
        let lines = tool_lines(&record(ToolStatus::Running), true, 60, Theme::default());
        assert_eq!(row_text(lines.last().unwrap()), "  running…");
    }

    #[test]
    fn metadata_header_is_two_dim_left_aligned_lines() {
        let lines = metadata_lines(
            "~/proj",
            Some("main"),
            "opencode-go",
            "gpt-5",
            &[],
            &[],
            Theme::default(),
        );
        assert_eq!(lines.len(), 2);
        assert_eq!(row_text(&lines[0]), "~/proj  (main)");
        assert_eq!(row_text(&lines[1]), "opencode-go · gpt-5");
        // Both rows are dimmed.
        for line in &lines {
            assert!(
                line.spans
                    .iter()
                    .all(|span| span.style.add_modifier.contains(Modifier::DIM)),
                "metadata row should be dim: {}",
                row_text(line)
            );
        }

        // No branch: the first row is just the cwd.
        let lines = metadata_lines("~/proj", None, "p", "m", &[], &[], Theme::default());
        assert_eq!(row_text(&lines[0]), "~/proj");
    }

    #[test]
    fn metadata_header_shows_context_and_skill_rows_capped() {
        let lines = metadata_lines(
            "~/proj",
            None,
            "p",
            "m",
            &["AGENTS.md".into(), "~/.harness/AGENTS.md".into()],
            &["alpha".into(), "beta".into()],
            Theme::default(),
        );
        assert_eq!(lines.len(), 4);
        assert_eq!(
            row_text(&lines[2]),
            "context: AGENTS.md, ~/.harness/AGENTS.md"
        );
        assert_eq!(row_text(&lines[3]), "skills: alpha, beta");

        // More than the cap folds into `+N more`.
        let many = (0..6).map(|index| format!("s{index}")).collect::<Vec<_>>();
        let lines = metadata_lines(
            "~/proj",
            None,
            "p",
            "m",
            &[],
            &many
                .iter()
                .map(String::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            Theme::default(),
        );
        assert_eq!(lines.len(), 3);
        assert_eq!(row_text(&lines[2]), "skills: s0, s1, s2, s3 · +2 more");

        // Nothing loaded: no extra rows.
        let lines = metadata_lines("~/proj", None, "p", "m", &[], &[], Theme::default());
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn separator_line_is_centered() {
        let line = separator_line("new conversation", 30, Theme::default());
        let text = row_text(&line);
        assert_eq!(UnicodeWidthStr::width(text.as_str()), 30);
        assert!(text.contains("── new conversation ──"));
    }

    #[test]
    fn line_to_ansi_emits_style_codes_only_for_styled_spans() {
        let styled = Line::from(Span::styled(
            "hi",
            Style::default()
                .fg(Theme::default().accent)
                .add_modifier(Modifier::BOLD),
        ));
        let ansi = line_to_ansi(&styled);
        assert!(ansi.contains("\u{1b}["));
        assert!(ansi.ends_with("\u{1b}[0m"));

        let plain = Line::from("plain");
        assert_eq!(line_to_ansi(&plain), "plain");
    }

    #[test]
    fn line_to_ansi_resets_bold_between_the_input_prefix_and_text() {
        let layout = input_layout("first\nsecond", 12, 40, Theme::default(), "", "", "");
        let first = line_to_ansi(&layout.rows[0]);
        let second = line_to_ansi(&layout.rows[1]);

        // The bold prefix is followed by a reset before normal input text;
        // both logical input rows therefore render at the same weight.
        assert!(first.contains("› \u{1b}[0m"));
        assert!(!second.contains(&format!("{}", SetAttribute(Attribute::Bold))));
    }

    #[test]
    fn ansi_color_matches_ratatui_backend_mapping() {
        assert_eq!(
            ansi_color(ratatui_core::style::Color::DarkGray),
            AnsiColor::DarkGrey
        );
        assert_eq!(
            ansi_color(ratatui_core::style::Color::Cyan),
            AnsiColor::DarkCyan
        );
        assert_eq!(
            ansi_color(ratatui_core::style::Color::LightRed),
            AnsiColor::Red
        );
    }

    #[test]
    fn editor_helpers_respect_char_boundaries() {
        let mut input = String::new();
        let mut cursor = 0usize;
        insert_text(&mut input, &mut cursor, "aé你");
        assert_eq!(cursor, 1 + 2 + 3);

        delete_backward(&mut input, &mut cursor);
        assert_eq!(input, "aé");
        assert_eq!(cursor, 3);

        move_left(&input, &mut cursor);
        assert_eq!(cursor, 1);
        move_right(&input, &mut cursor);
        assert_eq!(cursor, 3);
        delete_forward(&mut input, &mut cursor);
        assert_eq!(input, "aé");

        // CRLF paste payloads normalize to a single newline.
        let mut input = String::new();
        let mut cursor = 0usize;
        insert_text(&mut input, &mut cursor, "a\r\nb");
        assert_eq!(input, "a\nb");
    }

    #[test]
    fn vertical_move_crosses_logical_lines_and_preserves_columns() {
        let input = "abc\ndefghi\nx";
        // Cursor 7 sits after "def" (column 3): moving up lands after "abc".
        assert_eq!(vertical_move(input, 7, -1), Some(3));
        assert_eq!(vertical_move(input, 1, 1), Some(5)); // after "a" → "d|efghi"
        assert_eq!(vertical_move(input, 1, -1), None); // already on the first line
        assert_eq!(vertical_move(input, input.len(), 1), None); // last line
    }

    #[test]
    fn build_region_clips_the_streaming_tail_to_the_screen() {
        let mut ui = ui(80, 10);
        ui.stream = Some(StreamState {
            reasoning: String::new(),
            // One giant single block: no stable split, so the display clip is
            // the only bound. The full text prints when the message
            // finalizes.
            markdown: (0..60).map(|i| format!("line {i}\n")).collect(),
        });
        let input = ui.input_layout();
        let build = ui.build_region(&input);
        assert!(build.rows.len() <= 10, "region must fit the screen");
        // The newest rows are visible and the input line is at the bottom.
        let last = row_text(build.rows.last().expect("input row"));
        assert_eq!(last, format!("› {PLACEHOLDER}"));
        let text: String = build
            .rows
            .iter()
            .map(row_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("line 59"));
        assert!(!text.contains("line 0"));
    }

    #[test]
    fn build_region_clips_huge_input_around_the_cursor() {
        let mut ui = ui(80, 10);
        ui.input = (0..50)
            .map(|i| format!("row{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        ui.cursor = ui.input.len();
        let input = ui.input_layout();
        assert!(input.rows.len() > 10);
        let build = ui.build_region(&input);
        assert!(build.rows.len() <= 10);
        // The cursor (at the end of the draft) stays inside the window.
        assert!(build.cursor_row < build.rows.len());
        let last = row_text(build.rows.last().expect("clipped input"));
        assert_eq!(last, "  row49");
    }

    #[test]
    fn commit_stream_prefix_moves_stable_blocks_into_pending() {
        let mut ui = ui(80, 8);
        // A first paragraph tall enough to outgrow the height-8 tail budget
        // (~4 rows), followed by an unfinished second paragraph.
        let tall = format!("{}\n\n", "word ".repeat(100));
        ui.stream = Some(StreamState {
            reasoning: String::new(),
            markdown: tall.clone() + &"second ".repeat(30),
        });
        let input = ui.input_layout();
        ui.commit_stream_prefix(input.rows.len());
        // The stable first paragraph was committed and drained from the live
        // stream.
        assert_eq!(ui.pending.len(), 1);
        assert_eq!(
            ui.stream.as_ref().expect("tail remains").markdown,
            "second ".repeat(30)
        );
        match &ui.pending[0] {
            Entry::Assistant { markdown, .. } => assert_eq!(markdown, &tall),
            other => panic!("expected assistant prefix entry, got {other:?}"),
        }
    }

    #[test]
    fn commit_stream_prefix_keeps_short_blocks_live() {
        let mut ui = ui(80, 24);
        ui.stream = Some(StreamState {
            reasoning: String::new(),
            markdown: "short\n\nanswer".into(),
        });
        let input = ui.input_layout();
        ui.commit_stream_prefix(input.rows.len());
        assert!(ui.pending.is_empty());
        assert_eq!(
            ui.stream.as_ref().expect("stream remains").markdown,
            "short\n\nanswer"
        );
    }

    #[test]
    fn finalize_stream_commits_the_live_message() {
        let mut live = ui(80, 24);
        live.stream = Some(StreamState {
            reasoning: "thinking".into(),
            markdown: "answer".into(),
        });
        live.finalize_stream();
        assert!(live.stream.is_none());
        match &live.pending[0] {
            Entry::Assistant {
                markdown,
                reasoning,
            } => {
                assert_eq!(markdown, "answer");
                assert_eq!(reasoning, "thinking");
            }
            other => panic!("expected assistant entry, got {other:?}"),
        }

        // An empty stream finalizes to nothing.
        let mut empty = ui(80, 24);
        empty.stream = Some(StreamState::default());
        empty.finalize_stream();
        assert!(empty.pending.is_empty());
    }

    #[test]
    fn session_events_reset_the_transcript_and_queue_a_separator() {
        let mut ui = ui(80, 24);
        ui.transcript.push(Entry::User { text: "old".into() });
        ui.stream = Some(StreamState {
            reasoning: String::new(),
            markdown: "live".into(),
        });
        ui.apply_event(UiEvent::SessionChanged {
            id: "abcd1234efgh".into(),
            title: None,
            loaded: true,
        });
        assert!(ui.transcript.is_empty());
        assert!(ui.stream.is_none());
        // The live message finalizes first, then the separator follows it.
        assert!(matches!(&ui.pending[0], Entry::Assistant { .. }));
        assert!(matches!(&ui.pending[1], Entry::Separator { label } if label.contains("abcd1234")));
    }

    #[test]
    fn completion_accepts_a_single_command_candidate_on_tab() {
        let mut ui = ui(80, 24);
        ui.input = "/model".to_owned();
        ui.cursor = ui.input.len();
        let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
        ui.handle_tab(&tx).unwrap();
        // `/model` is a command that takes an argument; accepting it appends
        // the argument space and leaves the caret right after it.
        assert_eq!(ui.input, "/model ");
    }

    #[test]
    fn completion_tab_accepts_the_best_ranked_provider() {
        let mut ui = ui(80, 24);
        ui.input = "/model open".to_owned();
        ui.cursor = ui.input.len();
        let (tx, mut rx) = mpsc::unbounded_channel::<InputMessage>();
        // Provider completion is fuzzy-ranked; Tab accepts the best match.
        ui.handle_tab(&tx).unwrap();
        assert_eq!(ui.input, "/model openrouter:");
        // Accepting the provider requested its model list on demand.
        assert_eq!(
            rx.try_recv().ok(),
            Some(InputMessage::ListModels {
                provider: "openrouter".into(),
            })
        );
    }

    #[test]
    fn fuzzy_model_tab_accepts_a_provider_qualified_match() {
        let mut ui = ui(80, 24);
        ui.model_lists.insert(
            "openrouter".into(),
            vec![ModelEntry {
                id: "anthropic/claude-sonnet-4".into(),
                name: Some("Claude Sonnet 4".into()),
                context_length: Some(200_000),
            }],
        );
        ui.input = "/model cs4".to_owned();
        ui.cursor = ui.input.len();
        let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
        ui.handle_tab(&tx).unwrap();
        assert_eq!(ui.input, "/model openrouter:anthropic/claude-sonnet-4");
    }

    #[test]
    fn ghost_previews_a_single_candidate_suffix() {
        let mut ui = ui(80, 24);
        ui.input = "/model openc".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        // Only `opencode-go:` matches "openc", so the ghost shows its untyped
        // remainder.
        assert_eq!(ui.ghost_text(), "ode-go:");
    }

    #[test]
    fn ghost_clears_when_the_cursor_is_mid_token() {
        let mut ui = ui(80, 24);
        ui.input = "/model open".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        assert_eq!(ui.ghost_text(), "");
        // Even with a ghost, a cursor not at the end disables it.
        ui.cursor = "/model".len();
        assert_eq!(ui.ghost_text(), "");
    }

    #[tokio::test]
    async fn at_reference_opens_a_file_completion_context() {
        let mut ui = ui(80, 24);
        ui.input = "see @src/ma".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        // The `@` token opens a path completion even though the draft is not
        // a slash command; the scan starts from the token's query.
        let completion = ui.completion.as_ref().expect("@ opens completion");
        assert_eq!(completion.kind, CompletionKind::Path);
        assert_eq!(completion.context.token_start, 4);
        assert_eq!(completion.context.query, "src/ma");
        assert_eq!(ui.path_completion_query.as_deref(), Some("src/ma"));

        // Moving the cursor off the token (e.g. after a space) closes it.
        ui.input.push(' ');
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        assert!(ui.completion.is_none());
    }

    #[tokio::test]
    async fn at_reference_scan_results_replace_the_candidate_list() {
        let mut ui = ui(80, 24);
        ui.input = "look at @Re".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![Candidate {
                value: "@src/README.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            }],
            at_prefix: paths::extract_at_prefix("look at @Re", 11),
        });
        let completion = ui.completion.as_ref().expect("candidates applied");
        assert_eq!(completion.candidates[0].value, "@src/README.md");
        // A fuzzy match (`Re` → README) shares no literal prefix with the
        // typed token, so there is no ghost to preview.
        assert_eq!(ui.ghost_text(), "");
        // Tab accepts the candidate and appends a trailing space.
        let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
        ui.handle_tab(&tx).unwrap();
        assert_eq!(ui.input, "look at @src/README.md ");
    }

    #[tokio::test]
    async fn tab_accepts_the_top_ranked_at_candidate_then_cycles() {
        let mut ui = ui(80, 24);
        ui.input = "look at @REA".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        // Ranking put the root README first (see the paths tests).
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![
                Candidate {
                    value: "@README.md".into(),
                    description: "file".into(),
                    kind: CandidateKind::File,
                },
                Candidate {
                    value: "@crates/agent/README.md".into(),
                    description: "file".into(),
                    kind: CandidateKind::File,
                },
            ],
            at_prefix: paths::extract_at_prefix("look at @REA", 11),
        });

        // The ghost previews exactly what Tab will insert.
        assert_eq!(ui.ghost_text(), "DME.md");

        // First Tab accepts the top-ranked candidate; the trailing space ends
        // the token so typing continues on a fresh one.
        let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
        ui.handle_tab(&tx).unwrap();
        assert_eq!(ui.input, "look at @README.md ");
    }

    #[tokio::test]
    async fn fuzzy_at_tab_accepts_the_ranked_match_and_a_repeat_cycles() {
        let mut ui = ui(80, 24);
        ui.input = "@agent.rs".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![
                Candidate {
                    value: "@crates/agent/src/agent.rs".into(),
                    description: "file".into(),
                    kind: CandidateKind::File,
                },
                Candidate {
                    value: "@crates/tui/src/app.rs".into(),
                    description: "file".into(),
                    kind: CandidateKind::File,
                },
            ],
            at_prefix: paths::extract_at_prefix("@agent.rs", 9),
        });

        // No literal extension of `@agent.rs` exists, so nothing is dimmed
        // inline — but Tab still completes to the ranked match.
        assert_eq!(ui.ghost_text(), "");
        let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
        ui.handle_tab(&tx).unwrap();
        assert_eq!(ui.input, "@crates/agent/src/agent.rs ");

        // The accepted file closes the list; re-opening on the same token and
        // pressing Tab again cycles to the next candidate.
        ui.cursor = "@crates/agent/src/agent.rs".len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![
                Candidate {
                    value: "@crates/agent/src/agent.rs".into(),
                    description: "file".into(),
                    kind: CandidateKind::File,
                },
                Candidate {
                    value: "@crates/tui/src/app.rs".into(),
                    description: "file".into(),
                    kind: CandidateKind::File,
                },
            ],
            at_prefix: paths::extract_at_prefix("@crates/agent/src/agent.rs", 26),
        });
        ui.handle_tab(&tx).unwrap();
        assert_eq!(ui.input, "@crates/tui/src/app.rs ");
    }

    #[tokio::test]
    async fn enter_still_submits_while_an_at_list_is_open() {
        let mut ui = ui(80, 24);
        ui.input = "look at @Re".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![Candidate {
                value: "@README.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            }],
            at_prefix: paths::extract_at_prefix("look at @Re", 11),
        });

        // Enter is submit-only: completion stays a Tab affair.
        let (tx, mut rx) = mpsc::unbounded_channel::<InputMessage>();
        let cancel = CancellationToken::new();
        let enter = Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()));
        ui.handle_input(&enter, &tx, &cancel).unwrap();
        assert_eq!(
            rx.try_recv().ok(),
            Some(InputMessage::Message("look at @Re".into()))
        );
    }

    #[tokio::test]
    async fn refining_an_at_query_keeps_the_previous_candidates_visible() {
        let mut ui = ui(80, 24);
        ui.input = "look at @Re".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![Candidate {
                value: "@README.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            }],
            at_prefix: paths::extract_at_prefix("look at @Re", 11),
        });

        // Keep typing: the refined scan is still in flight, but the open
        // list must not blank out while waiting for it.
        ui.input.push('a');
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let completion = ui.completion.as_ref().expect("list stays open");
        assert_eq!(completion.candidates.len(), 1);
        assert_eq!(completion.candidates[0].value, "@README.md");
        assert_eq!(ui.path_completion_query.as_deref(), Some("Rea"));
    }

    #[test]
    fn path_ghost_previews_the_highlighted_remainder() {
        let mut ui = ui(80, 24);
        ui.input = "see @REA".to_owned();
        ui.cursor = ui.input.len();
        ui.completion = Some(Completion {
            context: CompletionContext {
                target: CompletionTarget::Argument(ArgumentKind::Path),
                token_start: 4,
                token_end: 8,
                query: "REA".into(),
            },
            candidates: vec![Candidate {
                value: "@README.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            }],
            selected: 0,
            kind: CompletionKind::Path,
        });
        // The dimmed suffix is exactly what Tab inserts.
        assert_eq!(ui.ghost_text(), "DME.md");
    }

    #[tokio::test]
    async fn stale_at_scan_results_are_ignored() {
        let mut ui = ui(80, 24);
        ui.input = "@a".to_owned();
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui.completion.as_ref().unwrap().context.clone();
        // The user kept typing before the scan landed: the result is stale.
        ui.input = "@ab".to_owned();
        ui.cursor = ui.input.len();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![Candidate {
                value: "@a.txt".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            }],
            at_prefix: paths::extract_at_prefix("@a", 2),
        });
        let completion = ui.completion.as_ref().expect("still open for @ab");
        assert!(completion.candidates.is_empty());
    }

    #[test]
    fn completion_hint_is_always_one_bounded_visual_row() {
        let candidates = vec![
            Candidate {
                value: "openrouter:anthropic/claude-sonnet-4-very-long-name".into(),
                description: String::new(),
                kind: CandidateKind::Slash,
            },
            Candidate {
                value: "github-copilot:claude-sonnet-4".into(),
                description: String::new(),
                kind: CandidateKind::Slash,
            },
        ];
        let hint = fit_completion_hint(&candidates, 0, 24);
        assert!(UnicodeWidthStr::width(hint.as_str()) <= 24);
        assert!(hint.ends_with('…'));
        assert!(!hint.contains(['\n', '\r']));

        // The selected candidate leads even when it is not index zero.
        let hint = fit_completion_hint(&candidates, 1, 80);
        assert!(hint.starts_with("github-copilot:"));
    }

    #[test]
    fn input_layout_renders_ghost_and_hint_rows() {
        // Ghost suffix follows the typed text on the input row.
        let layout = input_layout("/model openc", 12, 60, Theme::default(), "", "ode-go:", "");
        assert_eq!(row_text(&layout.rows[0]), "› /model opencode-go:");
        assert_eq!(layout.cursor_row, 0);

        // The completion hint row sits above the input.
        let layout = input_layout(
            "/model",
            6,
            60,
            Theme::default(),
            "",
            "",
            "opencode-go: · openrouter:",
        );
        assert_eq!(row_text(&layout.rows[0]), "opencode-go: · openrouter:");
        assert_eq!(row_text(&layout.rows[1]), "› /model");
        assert_eq!(layout.cursor_row, 1);
    }
}
