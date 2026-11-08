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
    Asking,
    Compacting,
    Working,
    Reasoning,
    Processing,
    Retrying,
}

impl Activity {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Preparing => "Preparing...",
            Self::Asking => "Answering /ask...",
            Self::Compacting => "Compacting session...",
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
use crate::commit::{SplitScanner, last_fenced_code_block};
use crate::environment::EnvironmentInfo;
use crate::input::{history_next, history_previous, push_history};
use crate::paths::{self, AtPrefix};
use crate::program_status;
use crate::render::{self, Theme};
use crate::state::{ToolRecord, ToolStatus};
use crate::{
    ContextFileEntry, InputMessage, ModelEntry, PrewalkChange, SessionListEntry,
    SessionSnapshotEntry, SkillEntry, StartupEntries, SubscriptionUsage, SubscriptionUsageWindow,
    UiEvent,
};
use anyhow::{Context, Result};
use crossterm::cursor::{MoveDown, MoveRight, MoveTo, MoveUp};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, MouseButton, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
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
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};
use std::time::Duration;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// The user-input prefix. Same glyph as the committed user echo so a
/// submitted message's pixels are exactly the prompt line it was typed on.
const INPUT_PREFIX: &str = "› ";
const INPUT_CONTINUATION: &str = "  ";
const INPUT_PREFIX_WIDTH: usize = 2;

/// One final or in-flight transcript block, stored at the source level so a
/// resize can re-render it at a new width.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Entry {
    /// The startup wordmark and discoverability footer.
    Banner {
        /// Random launch order retained so resize repaints stay consistent.
        title_order: render::WelcomeTitleOrder,
    },
    /// Startup details in aligned, accented rows above the transcript.
    Metadata {
        cwd: String,
        branch: Option<String>,
        provider: String,
        model: String,
        /// Auto-loaded AGENTS.md / CLAUDE.md paths (display form).
        context_files: Vec<String>,
        /// Configured MCP server names.
        mcp_servers: Vec<String>,
        /// Discovered skill names.
        skills: Vec<String>,
    },
    User {
        text: String,
    },
    Assistant {
        markdown: String,
        reasoning: String,
        /// True when this entry continues a message whose earlier chunk was
        /// already committed, so it gets no `‹` prefix of its own.
        continued: bool,
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
    /// A left-oriented rule marking a session boundary.
    Separator {
        label: String,
    },
}

/// Lifecycle of an on-demand backend fetch. `Failed` is distinct from `Idle`
/// so typed refreshes do not re-fire a failing request on every keystroke;
/// only an explicit Tab retries it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Fetch {
    #[default]
    Idle,
    InFlight,
    Loaded,
    Failed,
}

/// The in-flight assistant message. The stable markdown prefix is drained
/// into `pending` (as [`Entry::Assistant`]) once it outgrows the live-tail
/// budget, keeping long responses flowing into scrollback incrementally.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StreamState {
    reasoning: String,
    markdown: String,
    /// Markdown of this message was already committed to scrollback, so the
    /// remaining text continues it instead of opening a new `‹` block.
    markdown_continued: bool,
    /// Incremental stable-split scan over `markdown`; reset when it is
    /// drained.
    split: SplitScanner,
    /// Wrapped reasoning rows for the last paint. Reasoning only grows
    /// during a stream, so its length (with width and theme) identifies the
    /// rendering; re-wrapping it on every frame was O(n) per paint.
    reasoning_render: std::cell::RefCell<Option<ReasoningRender>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReasoningRender {
    width: usize,
    theme: Theme,
    len: usize,
    lines: Vec<Line<'static>>,
}

/// Cached rendering for completed markdown blocks in the live stream. The
/// current trailing block remains reparsed, while completed blocks are copied
/// into this cache once when a blank-line boundary arrives.
struct StreamMarkdownCache {
    width: usize,
    theme: Theme,
    source_offset: usize,
    lines: Vec<Line<'static>>,
    /// Number of completed source chunks parsed into the cache. Besides being
    /// useful when profiling, this guards the incremental-rendering invariant
    /// in colocated tests.
    rendered_blocks: usize,
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

/// Upper bounds for one coalesced event drain. A producer that keeps the
/// channel non-empty (a fast stream, a large snapshot replay) must not keep
/// the loop away from terminal input, so Esc and Ctrl+C get a turn after at
/// most this much work.
const EVENT_DRAIN_MAX_EVENTS: usize = 256;
const EVENT_DRAIN_BUDGET: Duration = Duration::from_millis(8);

/// Quiet time after the last resize event before the window is repainted.
const RESIZE_SETTLE: Duration = Duration::from_millis(40);

/// Largest payload `/copy` sends; terminals commonly cap OSC 52 near 1 MB.
const MAX_COPY_BYTES: usize = 512 * 1024;

/// OSC 52 clipboard-set sequence for `text`, wrapped in tmux's passthrough
/// envelope when running inside tmux.
fn clipboard_sequence(text: &str, tmux: bool) -> String {
    use base64::Engine as _;
    let payload = base64::engine::general_purpose::STANDARD.encode(text);
    let osc = format!("\u{1b}]52;c;{payload}\u{07}");
    if tmux {
        format!("\u{1b}Ptmux;\u{1b}{osc}\u{1b}\\")
    } else {
        osc
    }
}

/// OSC 133 prompt start asking for click reports (kitty `click_events`).
const PROMPT_START: &str = "\u{1b}]133;A;click_events=1\u{07}";
/// OSC 133 output start, closing the prompt after the input rows.
const PROMPT_END: &str = "\u{1b}]133;C\u{07}";

/// Opens every frame: begin synchronized update (DEC 2026) and disable
/// autowrap (DECAWM). Terminals without 2026 ignore it.
const FRAME_BEGIN: &str = "\u{1b}[?2026h\u{1b}[?7l";
/// Closes every frame in the same write, so autowrap is never left off.
const FRAME_END: &str = "\u{1b}[?7h\u{1b}[?2026l";

/// Token usage reported by [`UiEvent::UsageUpdated`], kept for the inline
/// counter trailer on the input placeholder row.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

/// Current context occupation and effective maximum reported by the agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContextUsage {
    used_tokens: u64,
    max_tokens: u64,
}

/// An open completion list driving the fish-style ghost preview, the Tab
/// accept, and the one-row hint below the input.
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
    /// Where the input text sits, for mapping mouse clicks back to the draft.
    input: Option<InputSpan>,
}

/// Position of the visible input rows inside a frame's live region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InputSpan {
    /// Region row of the first visible input row.
    first_row: usize,
    /// Layout row (before clipping) shown on that region row.
    layout_start: usize,
    /// Visible text rows, excluding the completion hint row.
    rows: usize,
}

/// The active-tool section of a frame: its rendered lines and total row
/// count (including any overflow row) for the sizing budgets.
struct RunningRegion {
    lines: Vec<Line<'static>>,
    rows: usize,
}

/// Operations needed to change terminal state. Keeping this small seam here
/// makes setup and cleanup testable without making the rendering or input
/// protocol depend on a different terminal implementation.
trait TerminalBackend: Send {
    fn size(&self) -> io::Result<(u16, u16)>;
    /// Cursor `(column, row)`, when the terminal answers the query. Used once
    /// at startup to locate the live region for click-to-position.
    fn cursor_position(&self) -> Option<(u16, u16)> {
        None
    }
    fn enable_raw_mode(&mut self) -> io::Result<()>;
    fn disable_raw_mode(&mut self) -> io::Result<()>;
    fn enable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn disable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn push_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn pop_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn write_newline(&mut self, out: &mut Stdout) -> io::Result<()>;
}

/// The production terminal operation layer. Crossterm is deliberately kept
/// behind [`TerminalBackend`], so mode bookkeeping can be tested independently
/// of the process terminal and its platform-specific implementation.
struct CrosstermBackend;

impl TerminalBackend for CrosstermBackend {
    fn size(&self) -> io::Result<(u16, u16)> {
        terminal::size()
    }

    fn cursor_position(&self) -> Option<(u16, u16)> {
        crossterm::cursor::position().ok()
    }

    fn enable_raw_mode(&mut self) -> io::Result<()> {
        terminal::enable_raw_mode()
    }

    fn disable_raw_mode(&mut self) -> io::Result<()> {
        terminal::disable_raw_mode()
    }

    fn enable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(out, EnableBracketedPaste)
    }

    fn disable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(out, DisableBracketedPaste)
    }

    fn push_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(
            out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
    }

    fn pop_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(out, PopKeyboardEnhancementFlags)
    }

    fn write_newline(&mut self, out: &mut Stdout) -> io::Result<()> {
        writeln!(out)
    }
}

/// Terminal modes successfully enabled by one UI instance. Cleanup only
/// reverses modes recorded here, so a partial setup cannot pop another
/// component's keyboard state. Failed cleanup remains recorded until a later
/// retry succeeds.
#[derive(Default)]
struct TerminalModes {
    raw_mode: bool,
    bracketed_paste: bool,
    keyboard_flags: bool,
    newline_written: bool,
}

struct TerminalState {
    backend: Box<dyn TerminalBackend>,
    modes: TerminalModes,
}

impl TerminalState {
    fn is_restored(&self) -> bool {
        !self.modes.raw_mode
            && !self.modes.bracketed_paste
            && !self.modes.keyboard_flags
            && self.modes.newline_written
    }
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
    reasoning: String,
    /// Whether this conversation has a pending prewalk handoff.
    prewalk_armed: bool,
    /// Prewalk target as `[<provider>:]<model>`, once the agent reports one.
    prewalk_target: Option<String>,
    /// Prewalk target reasoning; `None` keeps the active level.
    prewalk_reasoning: Option<String>,
    environment: EnvironmentInfo,
    /// Skip the welcome banner and metadata while retaining normal turn UI.
    minimal: bool,

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
    stream_markdown_cache: Option<StreamMarkdownCache>,
    /// Latest isolated answer, shown inline in the live region at its full
    /// height and never committed to history. Esc dismisses it.
    ask_answer: Option<String>,
    /// Currently running tool calls, keyed by harness call id and kept in
    /// launch order. Concurrent fan-out (e.g. several `subagent` calls in one
    /// response) means more than one record can be live at once; a finish
    /// removes exactly the matching id and never disturbs its neighbors.
    running_tools: Vec<RunningTool>,
    /// Background lookups do not own or clear the active turn's busy state.
    background_operations: Vec<(u64, String)>,

    input: String,
    /// Byte offset of the editing cursor (always on a char boundary).
    cursor: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
    draft: String,

    /// Active session usage for the placeholder counter trailer.
    usage: Option<Usage>,
    /// Current versus maximum context for the same trailer.
    context_usage: Option<ContextUsage>,
    /// Whether the runtime's initial session announcement has been handled.
    initial_session_seen: bool,

    /// Global tool expansion state: Ctrl+O toggles every tool call's output
    /// expanded or collapsed and repaints the whole visible session. Rows
    /// already scrolled into the terminal's native scrollback are immutable
    /// pixels; after the repaint older rows show as the existing
    /// `… N rows above` ellipsis. Expanded outputs of entries visible in the
    /// window are correctly re-rendered.
    tools_expanded: bool,

    /// Cached completion models for a provider.
    providers: Vec<String>,
    /// Alternative names of `providers` that `/model <alias>:<model>` accepts.
    /// Kept apart so completion lists only canonical names.
    provider_aliases: Vec<String>,
    model_lists: HashMap<String, Vec<ModelEntry>>,
    session_candidates: Vec<SessionListEntry>,
    /// Discovered skills, for `/<skill>` completion and `/skill <name>`
    /// argument completion.
    skills: Vec<SkillEntry>,
    /// Auto-loaded AGENTS.md / CLAUDE.md paths for the header context row.
    context_files: Vec<ContextFileEntry>,
    /// Configured MCP server names for the header MCP row.
    mcp_servers: Vec<String>,
    update_notice: Option<String>,
    /// Completion-driven `ListSessions` lifecycle.
    session_fetch: Fetch,
    session_list_display_requested: bool,
    /// Per-provider `ListModels` lifecycle, so typing through a model token
    /// neither duplicates a request nor hammers a provider that failed.
    model_fetch: HashMap<String, Fetch>,
    completion: Option<Completion>,
    path_completion_tx: mpsc::UnboundedSender<PathCompletionResult>,
    path_completion_rx: mpsc::UnboundedReceiver<PathCompletionResult>,
    path_completion_generation: u64,
    path_completion_query: Option<String>,
    path_completion_cancel: Option<CancellationToken>,

    busy: bool,
    /// True when busyness belongs to a model turn rather than a foreground
    /// operation, so `OperationFinished` cannot clear a live turn.
    turn_active: bool,
    activity: Activity,
    operation_label: Option<String>,
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

    /// Screen row where the next incremental frame begins (the first row of
    /// `region`), when known. `None` after a height-only resize, until the
    /// next full repaint re-establishes it.
    region_top: Option<usize>,
    /// Cursor row when the terminal was set up, seeding `region_top`.
    start_row: Option<usize>,
    /// Input location in the painted region, for click handling.
    input_span: Option<InputSpan>,

    terminal: Arc<Mutex<TerminalState>>,

    /// OSC 7501 program-status tracker; `None` when reporting is off.
    program_status: Option<program_status::Tracker>,
    /// A clipboard (OSC 52) sequence queued by `/copy`, written at the start
    /// of the next frame so it never splits one.
    clipboard_osc: Option<String>,
}

impl CrossTerm {
    /// Assemble the UI state without touching the terminal. `new` layers the
    /// terminal setup on top, and tests use this directly.
    #[allow(clippy::too_many_arguments)]
    fn base(
        model: &str,
        provider: &str,
        providers: Vec<String>,
        startup: StartupEntries,
        workspace_root: PathBuf,
        width: u16,
        height: u16,
    ) -> Self {
        let environment = EnvironmentInfo::discover(workspace_root);
        let StartupEntries {
            skills,
            context_files,
            mcp_servers,
            update_notice,
        } = startup;
        let (path_completion_tx, path_completion_rx) = mpsc::unbounded_channel();
        Self {
            out: io::stdout(),
            theme: Theme::default(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            reasoning: "auto".to_owned(),
            prewalk_armed: false,
            prewalk_target: None,
            prewalk_reasoning: None,
            environment,
            minimal: false,
            width,
            height,
            transcript: Vec::new(),
            pending: Vec::new(),
            stream: None,
            stream_markdown_cache: None,
            ask_answer: None,
            running_tools: Vec::new(),
            background_operations: Vec::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            usage: None,
            context_usage: None,
            initial_session_seen: false,
            tools_expanded: false,
            providers,
            provider_aliases: Vec::new(),
            model_lists: HashMap::new(),
            session_candidates: Vec::new(),
            skills,
            context_files,
            mcp_servers,
            update_notice,
            session_fetch: Fetch::Idle,
            session_list_display_requested: false,
            model_fetch: HashMap::new(),
            completion: None,
            path_completion_tx,
            path_completion_rx,
            path_completion_generation: 0,
            path_completion_query: None,
            path_completion_cancel: None,
            busy: false,
            turn_active: false,
            activity: Activity::Preparing,
            operation_label: None,
            spinner: 0,
            region: Vec::new(),
            cursor_row: 0,
            cursor_col: 0,
            activity_region_row: None,
            region_top: None,
            start_row: None,
            input_span: None,
            program_status: None,
            clipboard_osc: None,
            terminal: Arc::new(Mutex::new(TerminalState {
                backend: Box::new(CrosstermBackend),
                modes: TerminalModes::default(),
            })),
        }
    }

    fn with_backend(mut ui: Self, backend: Box<dyn TerminalBackend>) -> Self {
        ui.terminal = Arc::new(Mutex::new(TerminalState {
            backend,
            modes: TerminalModes::default(),
        }));
        ui
    }

    /// Add the canonical startup entries shared by standalone and retained panes.
    fn enqueue_welcome(&mut self) {
        if !self.minimal {
            self.pending.push(Entry::Banner {
                title_order: render::WelcomeTitleOrder::random(),
            });
            self.pending.push(self.metadata_entry());
            if let Some(text) = self.update_notice.clone() {
                self.pending.push(Entry::Notice { text });
            }
        }
    }

    /// `skills`, `context_files`, MCP names, and the initial reasoning label come from
    /// startup configuration (the TUI never touches their backing stores).
    /// `minimal` suppresses only the initial banner and metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &str,
        provider: &str,
        providers: Vec<String>,
        startup: StartupEntries,
        reasoning: &str,
        minimal: bool,
        workspace_root: PathBuf,
    ) -> Result<Self> {
        let backend = CrosstermBackend;
        let (width, height) = backend.size().unwrap_or((80, 24));
        let mut ui = Self::with_backend(
            Self::base(
                model,
                provider,
                providers,
                startup,
                workspace_root,
                width,
                height,
            ),
            Box::new(backend),
        );
        ui.reasoning = reasoning.to_owned();
        ui.minimal = minimal;
        ui.enqueue_welcome();
        install_panic_hook(&ui.terminal);
        ui.setup_terminal()?;
        Ok(ui)
    }

    /// Report status to the terminal with OSC 7501 (the Program Status
    /// Protocol). Terminals that do not know it ignore the sequence. Reports
    /// are written at the start of a frame so they never split one, and the
    /// record is cleared on exit so no stale `done` outlives the process.
    pub fn enable_program_status(&mut self) {
        self.program_status = Some(program_status::Tracker::new());
    }

    /// Update the status tracker, if reporting is on.
    fn status(&mut self, update: impl FnOnce(&mut program_status::Tracker)) {
        if let Some(tracker) = &mut self.program_status {
            update(tracker);
        }
    }

    /// Register provider aliases so `/model <alias>:<model>` still switches
    /// provider instead of being read as a model id.
    pub fn set_provider_aliases(&mut self, aliases: Vec<String>) {
        self.provider_aliases = aliases;
    }

    /// Enable terminal modes in dependency order. Only successful operations
    /// are recorded; a setup failure restores the successful prefix before it
    /// is returned to the caller. Keyboard enhancement is optional because
    /// unsupported terminals report that operation as an error.
    fn setup_terminal(&mut self) -> Result<()> {
        let raw_result = {
            let mut terminal = lock_terminal(&self.terminal);
            terminal
                .backend
                .enable_raw_mode()
                .context("enable terminal raw mode")
        };
        if let Err(error) = raw_result {
            return self.setup_failed(error);
        }
        lock_terminal(&self.terminal).modes.raw_mode = true;
        self.start_row = lock_terminal(&self.terminal)
            .backend
            .cursor_position()
            .map(|(_, row)| usize::from(row));
        self.region_top = self.start_row;

        let bracketed_result = {
            let mut terminal = lock_terminal(&self.terminal);
            terminal
                .backend
                .enable_bracketed_paste(&mut self.out)
                .context("configure terminal input")
        };
        if let Err(error) = bracketed_result {
            return self.setup_failed(error);
        }
        lock_terminal(&self.terminal).modes.bracketed_paste = true;

        // The kitty keyboard protocol makes Shift+Enter report as `Enter` with
        // the SHIFT modifier, which the input handler already maps to a newline
        // (works on Ghostty; iTerm/Terminal.app do not support it). Only
        // `DISAMBIGUATE_ESCAPE_CODES` is pushed — `REPORT_ALL_KEYS_AS_ESCAPE_CODES`
        // would swallow plain characters. Popped in `restore` and the panic hook.
        if lock_terminal(&self.terminal)
            .backend
            .push_keyboard_flags(&mut self.out)
            .is_ok()
        {
            lock_terminal(&self.terminal).modes.keyboard_flags = true;
        }
        // The cursor stays visible: it *is* the input caret, sitting right
        // after the `› ` prefix. No hide, no fake cell.
        Ok(())
    }

    fn setup_failed(&mut self, error: anyhow::Error) -> Result<()> {
        match self.restore() {
            Ok(()) => Err(error),
            Err(restore_error) => Err(error.context(format!(
                "terminal restoration also failed: {restore_error:#}"
            ))),
        }
    }

    pub async fn run(
        mut self,
        mut events: mpsc::Receiver<UiEvent>,
        input_tx: mpsc::UnboundedSender<InputMessage>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let result = self.run_inner(&mut events, input_tx, cancel).await;
        if let Some(report) = self.program_status.as_mut().and_then(|s| s.clear_report()) {
            let _ = self.out.write_all(report.as_bytes());
            let _ = self.out.flush();
        }
        let restore = self.restore();
        match (result, restore) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(restore_error)) => Err(error.context(format!(
                "terminal restoration also failed: {restore_error:#}"
            ))),
        }
    }

    async fn run_inner(
        &mut self,
        events: &mut mpsc::Receiver<UiEvent>,
        input_tx: mpsc::UnboundedSender<InputMessage>,
        cancel: CancellationToken,
    ) -> Result<()> {
        self.paint()?;

        let mut input_events = EventStream::new();
        let mut spinner_tick = tokio::time::interval(Duration::from_millis(200));
        spinner_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Dragging a window edge emits many resize events; each full repaint
        // reflows the visible history. Keep only the latest size and apply it
        // once the events pause (or before anything else needs the layout).
        let mut pending_resize: Option<(u16, u16)> = None;
        let resize_settle = tokio::time::sleep(Duration::ZERO);
        tokio::pin!(resize_settle);

        loop {
            tokio::select! {
                // Terminal input is polled first: after a bounded event
                // drain, a pending Esc or Ctrl+C always wins the next wake.
                biased;
                maybe_input = input_events.next() => {
                    let event = match maybe_input {
                        Some(Ok(event)) => event,
                        Some(Err(error)) => return Err(error).context("read terminal event"),
                        None => return Ok(()),
                    };
                    if let Event::Resize(width, height) = event {
                        pending_resize = Some((width, height));
                        resize_settle
                            .as_mut()
                            .reset(tokio::time::Instant::now() + RESIZE_SETTLE);
                        continue;
                    }
                    if let Some((width, height)) = pending_resize.take() {
                        self.handle_resize(width, height)?;
                    }
                    if self.handle_input(&event, &input_tx, &cancel)? {
                        return Ok(());
                    }
                    self.paint()?;
                }
                () = &mut resize_settle, if pending_resize.is_some() => {
                    if let Some((width, height)) = pending_resize.take() {
                        self.handle_resize(width, height)?;
                    }
                }
                maybe_event = events.recv() => {
                    let Some(event) = maybe_event else { return Ok(()) };
                    if let Some((width, height)) = pending_resize.take() {
                        self.handle_resize(width, height)?;
                    }
                    self.drain_queued_events(event, events);
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
                    if !self.busy && self.background_operations.is_empty() {
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

    /// Coalesced painting: an agent burst (streaming deltas, tool start/finish
    /// pairs, notices) arrives as many channel events within a few
    /// milliseconds. Apply `first` and what is already queued so one paint
    /// covers the burst, but stop after [`EVENT_DRAIN_MAX_EVENTS`] events or
    /// [`EVENT_DRAIN_BUDGET`], so a producer that never lets the queue empty
    /// cannot starve input. `try_recv` never blocks, so this adds no latency.
    ///
    /// Adjacent text (or reasoning) deltas are merged before they are
    /// applied: each applied text delta refreshes the streaming markdown
    /// cache, so one merged delta per drain avoids that work per token.
    /// Returns the number of events received.
    fn drain_queued_events(
        &mut self,
        first: UiEvent,
        events: &mut mpsc::Receiver<UiEvent>,
    ) -> usize {
        let started = std::time::Instant::now();
        let mut pending = first;
        let mut received = 1;
        while received < EVENT_DRAIN_MAX_EVENTS && started.elapsed() < EVENT_DRAIN_BUDGET {
            let Ok(next) = events.try_recv() else {
                break;
            };
            received += 1;
            match (&mut pending, next) {
                (UiEvent::TextDelta(text), UiEvent::TextDelta(more))
                | (UiEvent::ReasoningDelta(text), UiEvent::ReasoningDelta(more)) => {
                    text.push_str(&more);
                }
                (_, next) => {
                    let ready = std::mem::replace(&mut pending, next);
                    self.apply_event(ready);
                }
            }
        }
        self.apply_event(pending);
        received
    }

    /// Rewrite only the activity row in place, leaving the rest of the live
    /// region untouched. Used by spinner ticks: the glyph animation alone
    /// must not re-serialize the streaming tail and input rows five times a
    /// second. Returns `false` when the activity row's position is unknown
    /// (caller falls back to a full [`Self::paint`]).
    fn repaint_activity_only(&mut self) -> Result<bool> {
        // Background rows also animate, and can coexist with the turn row.
        if !self.background_operations.is_empty() {
            return Ok(false);
        }
        let Some(row) = self.activity_region_row else {
            return Ok(false);
        };
        if row >= self.region.len() {
            return Ok(false);
        }
        let theme = self.theme;
        let gutter = render::horizontal_pad(self.width) as usize;
        let line = activity_line_with_label(
            self.operation_label
                .as_deref()
                .unwrap_or(self.activity.label()),
            self.spinner,
            theme,
        );
        let mut buffer = String::from(FRAME_BEGIN);
        let up = self.cursor_row.saturating_sub(row);
        if up > 0 {
            let _ = write!(buffer, "{}", MoveUp(up as u16));
        }
        let _ = write!(buffer, "\r{}", Clear(ClearType::UntilNewLine));
        write_row(&mut buffer, &line, gutter, self.width as usize);
        if up > 0 {
            let _ = write!(buffer, "\r{}", MoveDown(up as u16));
        }
        buffer.push('\r');
        let col = self.cursor_col.min(u16::MAX as usize) as u16;
        if col > 0 {
            let _ = write!(buffer, "{}", MoveRight(col));
        }
        buffer.push_str(FRAME_END);
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
            // The terminal may have moved the region on a height change.
            self.region_top = None;
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
            if let Event::Mouse(mouse) = event {
                if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                    && let Some(cursor) = self.click_to_cursor(mouse.column, mouse.row)
                {
                    self.cursor = cursor;
                    self.history_pos = self.history_pos.filter(|_| false);
                    self.refresh_completion();
                }
                return Ok(false);
            }
            if let Event::Paste(text) = event {
                let text = render::sanitize_terminal_text(&normalize_paste_newlines(text));
                insert_text(&mut self.input, &mut self.cursor, &text);
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
                if !self.input.is_empty() {
                    // Match readline-style two-step Ctrl+C behavior: discard
                    // the current draft first, then let a subsequent press
                    // close Harness. Esc remains the turn-local interrupt.
                    self.input.clear();
                    self.cursor = 0;
                    self.history_pos = None;
                    self.draft.clear();
                    self.close_completion();
                    return Ok(false);
                }
                cancel.cancel();
                return Ok(true);
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
            // Esc interrupts a running turn; when a completion list is open,
            // Esc closes the list instead.
            KeyCode::Esc => {
                if self.busy {
                    self.status(|status| status.interrupt());
                    let _ = input_tx.send(InputMessage::Interrupt);
                } else if self.completion.is_some() {
                    self.close_completion();
                    // Skip the tail refresh: it would reopen the same list.
                    return Ok(false);
                } else {
                    self.dismiss_ask();
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
            KeyCode::Home => {
                self.cursor = line_bounds(&self.input, self.cursor).0;
            }
            KeyCode::End => {
                self.cursor = line_bounds(&self.input, self.cursor).1;
            }
            KeyCode::Char('a') if control => {
                self.cursor = line_bounds(&self.input, self.cursor).0;
            }
            KeyCode::Char('e') if control => {
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
        self.request_typed_backend(input_tx, false)?;
        Ok(false)
    }

    /// Map a left click to a byte offset in the draft. Terminals that honor
    /// the OSC 133 `click_events` mark report clicks on the prompt as SGR
    /// mouse events without capturing the mouse, so native selection and
    /// scrollback keep working.
    fn click_to_cursor(&self, column: u16, row: u16) -> Option<usize> {
        let span = self.input_span?;
        let top = self.region_top?;
        let region_row = usize::from(row).checked_sub(top)?;
        let visible = region_row.checked_sub(span.first_row)?;
        if visible >= span.rows || self.input.is_empty() {
            return None;
        }
        let gutter = render::horizontal_pad(self.width) as usize;
        let col = usize::from(column).saturating_sub(gutter + INPUT_PREFIX_WIDTH);
        let width = render::content_width(self.width)
            .saturating_sub(INPUT_PREFIX_WIDTH)
            .max(1);
        input_offset_at(&self.input, width, span.layout_start + visible, col)
    }

    /// If the input no longer matches the recalled entry, the user edited it:
    /// adopt the edit as a fresh draft so the next Up/Down keeps it instead of
    /// overwriting it with a history entry.
    fn detach_edited_history(&mut self) {
        if let Some(pos) = self.history_pos
            && self.history.get(pos) != Some(&self.input)
        {
            self.history_pos = None;
            self.draft.clear();
        }
    }

    fn history_up(&mut self) {
        self.detach_edited_history();
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
        self.detach_edited_history();
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
            self.close_completion();
            return;
        };

        // Path arguments scan the filesystem on a debounced task. Do not
        // cancel first: `request_path_completion` keeps the current scan and
        // its results when the token context is unchanged (as `@` paths do),
        // so Tab after the results arrive accepts them instead of clearing
        // them and starting again.
        if matches!(
            result.context.target,
            CompletionTarget::Argument(ArgumentKind::Path | ArgumentKind::Session),
        ) {
            self.request_path_completion(result.context, result.candidates, old_value, None);
            return;
        }
        self.cancel_path_completion();
        // Session arguments use the path-completion scan above.
        let kind = match result.context.target {
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
            // Keep a model list open (empty) so Tab can request it; close
            // everything else.
            if kind == CompletionKind::Model {
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
        self.session_fetch = Fetch::Idle;
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
        // Also cancels any in-flight path scan so a stale result cannot
        // reopen the popup over the accepted text.
        self.close_completion();
        // A None-argument command (e.g. `/new`) has no arguments to complete;
        // the refresh below would otherwise keep its own name listed.
        let argument_less = matches!(context.target, CompletionTarget::Command,)
            && commands::command_spec(&candidate.value)
                .is_some_and(|spec| spec.argument_kind == ArgumentKind::None);
        if argument_less {
            return Ok(());
        }
        self.request_backend(input_tx, context, candidate, true)?;
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

    /// Request backend data for a directly typed model provider prefix or a
    /// session/path command. Candidate acceptance already calls
    /// [`Self::request_backend`]; this covers the equivalent typed form.
    fn request_typed_backend(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        explicit: bool,
    ) -> Result<()> {
        let Some(context) = self
            .completion
            .as_ref()
            .map(|completion| completion.context.clone())
        else {
            return Ok(());
        };
        let CompletionTarget::Argument(kind) = context.target else {
            return Ok(());
        };
        if !matches!(kind, ArgumentKind::Model | ArgumentKind::Session) {
            return Ok(());
        }
        let start = context.token_start;
        let end = context.token_end.min(self.input.len());
        let value = self.input[start..end].to_owned();
        self.request_backend(
            input_tx,
            &context,
            &Candidate {
                value,
                description: String::new(),
                kind: CandidateKind::Slash,
            },
            explicit,
        )
    }

    /// Send anything the new draft now needs from the agent: a `ListSessions`
    /// for `/load` arguments or a `ListModels` when a `provider:` token was
    /// just completed and its model list is unknown. The dedupe sets keep
    /// repeated Tab presses from spamming the agent. Typing (`explicit` false)
    /// only fetches what was never requested; an explicit Tab also retries a
    /// failed fetch.
    fn request_backend(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        context: &CompletionContext,
        candidate: &Candidate,
        explicit: bool,
    ) -> Result<()> {
        let is_session = matches!(
            context.target,
            CompletionTarget::Argument(ArgumentKind::Session),
        );
        let session_may_fetch = match self.session_fetch {
            Fetch::Idle => true,
            // A list that arrived empty is only re-asked on an explicit Tab,
            // not on every keystroke.
            Fetch::Loaded | Fetch::Failed => explicit,
            Fetch::InFlight => false,
        };
        if is_session && self.session_candidates.is_empty() && session_may_fetch {
            self.session_fetch = Fetch::InFlight;
            input_tx
                .send(InputMessage::ListSessions)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        }
        // A bare `provider:` token requests the catalogue; so does a
        // partially typed `provider:model` token whose provider part is
        // known but uncached — the user is clearly addressing that
        // provider's catalogue even before Tab-accepting the prefix.
        let typed = candidate.value.as_str();
        let provider = typed
            .strip_suffix(':')
            .or_else(|| typed.split_once(':').map(|(provider, _)| provider))
            .filter(|name| {
                self.providers
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(name))
            })
            .map(str::to_owned);
        if let Some(provider) = provider
            && !self.model_lists.contains_key(&provider)
            && match self
                .model_fetch
                .get(provider.as_str())
                .copied()
                .unwrap_or_default()
            {
                Fetch::Idle => true,
                Fetch::Failed => explicit,
                Fetch::InFlight | Fetch::Loaded => false,
            }
        {
            self.model_fetch.insert(provider.clone(), Fetch::InFlight);
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
            // the backend again. Use the typed token (not an empty one) so a
            // `provider:` model catalogue can be re-requested too.
            self.request_typed_backend(input_tx, true)?;
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
        // row below the input carries those suggestions instead.
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

    /// One dim hint row below the input. It is fitted to the actual content
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
            self.dismiss_ask();
            // The typed prompt line is already on screen with the `› ` prefix;
            // committing it as a user entry below re-prints the same pixels.
            self.pending.push(Entry::User { text: text.clone() });
            input_tx
                .send(InputMessage::Message(text))
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
            self.busy = true;
            self.turn_active = true;
            self.activity = Activity::Preparing;
            self.spinner = 0;
            self.status(|status| status.start_work());
        }
        Ok(())
    }

    /// Show a prewalk model switch; it is not a user selection.
    fn apply_prewalk_switch(
        &mut self,
        label: &str,
        provider: String,
        model: String,
        reasoning: Option<String>,
    ) {
        self.provider = provider;
        self.model = model;
        if let Some(reasoning) = reasoning {
            self.reasoning = reasoning;
        }
        self.add_notice(format!(
            "{label} → {} · {} · reasoning {}",
            self.provider, self.model, self.reasoning
        ));
    }

    fn submit_command(
        &mut self,
        input: &str,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let command = match commands::parse_command_with_skills(input, &self.skills) {
            Ok(command) => commands::fold_unknown_provider_prefix(
                command,
                &self.providers,
                &self.provider_aliases,
            ),
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
        if matches!(&command, ParsedCommand::SetPrewalk { enabled: None }) {
            let reasoning = self
                .prewalk_reasoning
                .as_deref()
                .unwrap_or(commands::PREWALK_INHERIT);
            let state = match (self.prewalk_armed, &self.prewalk_target) {
                (true, Some(target)) => format!("on → {target} · reasoning {reasoning}"),
                (_, Some(target)) => format!("off, target {target} · reasoning {reasoning}"),
                (_, None) => "off, no target".to_owned(),
            };
            self.add_notice(format!("usage: /prewalk [on|off] (current: {state})"));
            return Ok(());
        }
        if matches!(&command, ParsedCommand::SetPrewalkReasoning { level: None }) {
            self.add_notice(format!(
                "usage: /prewalk-reasoning [inherit|auto|off|minimal|low|medium|high|maximum] (current: {})",
                self.prewalk_reasoning.as_deref().unwrap_or(commands::PREWALK_INHERIT)
            ));
            return Ok(());
        }
        if let ParsedCommand::SetPrewalkModel {
            provider: None,
            model,
        } = &command
            && model.is_empty()
        {
            self.add_notice(format!(
                "usage: /prewalk-model [<provider>:]<model> (current: {})",
                self.prewalk_target.as_deref().unwrap_or("none")
            ));
            return Ok(());
        }
        if matches!(&command, ParsedCommand::SetReasoning { level: None }) {
            self.add_notice(format!(
                "usage: /reasoning [auto|off|minimal|low|medium|high|maximum] (current: {})",
                self.reasoning
            ));
            return Ok(());
        }
        // A new command dismisses the previous isolated result, including
        // when the next ask fails before an answer arrives.
        self.dismiss_ask();
        // Ask is intentionally absent from retained UI transcript as well as
        // session history; only its answer is shown when it arrives.
        if !matches!(&command, ParsedCommand::Ask { .. }) {
            // The typed command is echoed without the ⌘ glyph: the input line
            // already carries the `› ` prefix, so the notice is the raw text.
            let echo = match &command {
                ParsedCommand::InvokeSkill { name, alias: true } => {
                    format!("{input} · skill: {name}")
                }
                _ => input.to_owned(),
            };
            self.add_notice(echo);
        }
        let message = match command {
            ParsedCommand::Help => {
                self.add_notice(
                    commands::COMMANDS
                        .iter()
                        .map(|spec| {
                            format!("{:<18} {}  ({})", spec.name, spec.description, spec.usage)
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
                return Ok(());
            }
            ParsedCommand::Ask { question } => InputMessage::Ask { question },
            ParsedCommand::New => InputMessage::NewConversation,
            ParsedCommand::Load { selector } => InputMessage::LoadSession { selector },
            ParsedCommand::Sessions => {
                self.session_list_display_requested = true;
                InputMessage::ListSessions
            }
            ParsedCommand::Export { destination } => InputMessage::ExportSession { destination },
            ParsedCommand::Copy { code } => {
                self.copy_last_reply(code);
                return Ok(());
            }
            ParsedCommand::Compact => InputMessage::CompactSession,
            ParsedCommand::SetModel { provider, model } => {
                InputMessage::SetModel { provider, model }
            }
            ParsedCommand::SetReasoning { level: Some(level) } => {
                InputMessage::SetReasoning { level }
            }
            ParsedCommand::SetReasoning { level: None } => unreachable!("handled above"),
            ParsedCommand::SetPrewalk {
                enabled: Some(enabled),
            } => InputMessage::SetPrewalk { enabled },
            ParsedCommand::SetPrewalk { enabled: None } => unreachable!("handled above"),
            ParsedCommand::SetPrewalkModel { provider, model } => {
                // Match the config rule: a prefix is a provider only when it
                // is known, so `vendor/model:free` stays one model id.
                match provider {
                    Some(provider)
                        if !self
                            .providers
                            .iter()
                            .any(|known| known.eq_ignore_ascii_case(&provider)) =>
                    {
                        InputMessage::SetPrewalkModel {
                            provider: None,
                            model: format!("{provider}:{model}"),
                        }
                    }
                    provider => InputMessage::SetPrewalkModel { provider, model },
                }
            }
            ParsedCommand::SetPrewalkReasoning { level: Some(level) } => {
                InputMessage::SetPrewalkReasoning {
                    level: (level != commands::PREWALK_INHERIT).then_some(level),
                }
            }
            ParsedCommand::SetPrewalkReasoning { level: None } => unreachable!("handled above"),
            ParsedCommand::Usage => InputMessage::SubscriptionUsage,
            ParsedCommand::InvokeSkill { name, .. } => InputMessage::InvokeSkill { name },
            ParsedCommand::Skills => {
                // Filled asynchronously by the agent's SkillsLoaded event.
                input_tx
                    .send(InputMessage::ListSkills)
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                return Ok(());
            }
        };
        let activity = match &message {
            InputMessage::Ask { .. } => Some(Activity::Asking),
            InputMessage::CompactSession => Some(Activity::Compacting),
            InputMessage::InvokeSkill { .. } => Some(Activity::Preparing),
            _ => None,
        };
        if !self.busy
            && let Some(activity) = activity
        {
            self.busy = true;
            self.turn_active = false;
            self.activity = activity;
            self.spinner = 0;
        }
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
                self.turn_active = true;
                self.activity = Activity::Working;
                self.operation_label = None;
                self.status(|status| status.progress());
                self.dismiss_ask();
                self.stream().markdown.push_str(&delta);
                self.refresh_stream_markdown_cache();
            }
            UiEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return;
                }
                self.busy = true;
                self.turn_active = true;
                self.activity = Activity::Reasoning;
                self.operation_label = None;
                self.status(|status| status.start_work());
                self.dismiss_ask();
                self.stream().reasoning.push_str(&delta);
            }
            UiEvent::ToolCallStarted {
                call_id,
                name,
                summary,
            } => {
                self.busy = true;
                self.turn_active = true;
                self.activity = Activity::Processing;
                self.operation_label = None;
                self.status(|status| status.start_work());
                self.dismiss_ask();
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
                    omitted_lines: 0,
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
                // A finish only reports on a call from a running turn; if the
                // turn already ended (late event) it must not re-arm the spinner.
                if self.busy {
                    self.activity = Activity::Working;
                    self.operation_label = None;
                }
                // Remove exactly the matching start; unrelated running tools
                // stay live and keep rendering in the region below.
                if let Some(position) = self
                    .running_tools
                    .iter()
                    .position(|running| running.call_id == call_id)
                {
                    self.running_tools.remove(position);
                }
                let record = ToolRecord::finished(name, summary, ok, duration_ms, output, error);
                self.pending.push(Entry::Tool { record });
            }
            UiEvent::Retrying { .. } => {
                // Retries also occur inside `/ask`, which ends with
                // `OperationFinished`, so a retry must not claim turn
                // ownership; `turn_active` stays whatever started the work.
                if self.busy {
                    self.activity = Activity::Retrying;
                    self.operation_label = None;
                    self.status(|status| status.start_work());
                }
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
                self.turn_active = false;
                self.activity = Activity::Preparing;
                self.operation_label = None;
                // Failures carry no request id. `ListSessions` is answered
                // synchronously, so an error while one is in flight is its
                // reply; drop the display request so a later completion-driven
                // list is not printed unasked.
                self.session_list_display_requested = false;
                if self.session_fetch == Fetch::InFlight {
                    self.session_fetch = Fetch::Failed;
                }
                self.status(|status| status.fail(&error));
                self.add_error(error);
            }
            UiEvent::AskStarted => {
                self.dismiss_ask();
                self.busy = true;
                self.turn_active = false;
                self.activity = Activity::Asking;
                self.operation_label = None;
                self.spinner = 0;
                self.status(|status| status.start_work());
            }
            UiEvent::AskAnswer(answer) => {
                self.ask_answer = Some(answer);
            }
            UiEvent::TurnFinished => {
                self.finalize_stream();
                self.running_tools.clear();
                self.busy = false;
                self.turn_active = false;
                self.activity = Activity::Preparing;
                self.operation_label = None;
                self.status(|status| status.finish_turn());
            }
            UiEvent::OperationStarted { label } => {
                self.busy = true;
                self.turn_active = false;
                self.activity = Activity::Preparing;
                self.operation_label = Some(label);
                self.spinner = 0;
                self.status(|status| status.start_work());
            }
            UiEvent::OperationFinished => {
                // A turn still streaming owns the spinner; its own
                // TurnFinished/Error will clear it.
                if self.turn_active {
                    return;
                }
                self.busy = false;
                self.activity = Activity::Preparing;
                self.operation_label = None;
                self.status(|status| status.finish_operation());
            }
            UiEvent::BackgroundOperationStarted { id, label } => {
                if let Some((_, current)) = self
                    .background_operations
                    .iter_mut()
                    .find(|(key, _)| *key == id)
                {
                    *current = label;
                } else {
                    self.background_operations.push((id, label));
                }
            }
            UiEvent::BackgroundOperationFinished { id, error } => {
                self.background_operations.retain(|(key, _)| *key != id);
                if let Some(error) = error {
                    // A lookup failure must not finalize streamed text or tools.
                    // Model fetches are the only in-flight background lookups
                    // keyed here; allow an explicit Tab to retry them.
                    for state in self.model_fetch.values_mut() {
                        if *state == Fetch::InFlight {
                            *state = Fetch::Failed;
                        }
                    }
                    self.add_error(error);
                }
            }
            UiEvent::Notice(notice) => self.add_notice(notice),
            UiEvent::ModelChanged { provider, model } => {
                self.provider = provider;
                self.model = model;
                // A model change needs one short confirmation, not a second
                // copy of all startup metadata.
                self.add_notice(format!("Using {} · {}", self.provider, self.model));
            }
            UiEvent::ReasoningChanged { level } => {
                self.reasoning = level;
                self.add_notice(format!("Reasoning effort: {}", self.reasoning));
            }
            UiEvent::PrewalkChanged {
                armed,
                target,
                reasoning,
                change,
            } => {
                // Armed without a target cannot hand off; never show it armed.
                let armed = armed && target.is_some();
                let was_armed = std::mem::replace(&mut self.prewalk_armed, armed);
                self.prewalk_target = target;
                self.prewalk_reasoning = reasoning;
                let target = self.prewalk_target.as_deref().unwrap_or("none");
                let notice = match change {
                    PrewalkChange::Enabled if armed => Some(format!("Prewalk on → {target}")),
                    PrewalkChange::Enabled => Some("Prewalk off".to_owned()),
                    PrewalkChange::Target if armed => {
                        Some(format!("Prewalk target: {target} (armed)"))
                    }
                    PrewalkChange::Target => Some(format!("Prewalk target: {target}")),
                    PrewalkChange::Reasoning => Some(format!(
                        "Prewalk reasoning effort: {}",
                        self.prewalk_reasoning
                            .as_deref()
                            .unwrap_or(commands::PREWALK_INHERIT)
                    )),
                    // Startup reports a configured target without a notice.
                    PrewalkChange::Session if armed == was_armed => None,
                    PrewalkChange::Session if armed => Some(format!("Prewalk armed → {target}")),
                    PrewalkChange::Session => Some("Prewalk disarmed".to_owned()),
                };
                if let Some(notice) = notice {
                    self.add_notice(notice);
                }
            }
            UiEvent::PrewalkHandoff {
                provider,
                model,
                reasoning,
            } => {
                self.prewalk_armed = false;
                self.apply_prewalk_switch("Prewalk handoff", provider, model, reasoning);
            }
            UiEvent::PrewalkRestored {
                provider,
                model,
                reasoning,
            } => {
                self.apply_prewalk_switch("Prewalk planning model", provider, model, reasoning);
            }
            UiEvent::ModelList { provider, models } => {
                self.model_fetch.insert(provider.clone(), Fetch::Loaded);
                self.model_lists.insert(provider, models);
                if commands::is_command_input(&self.input) {
                    self.refresh_completion();
                }
            }
            UiEvent::UsageUpdated {
                input_tokens,
                output_tokens,
                ..
            } => {
                self.usage = Some(Usage {
                    input_tokens,
                    output_tokens,
                });
            }
            UiEvent::ContextUsageUpdated {
                used_tokens,
                max_tokens,
            } => {
                self.context_usage = Some(ContextUsage {
                    used_tokens,
                    max_tokens,
                });
            }
            UiEvent::SubscriptionUsageLoaded { provider, usage } => {
                self.add_notice(format_subscription_usage(&provider, &usage));
            }
            UiEvent::SessionChanged { id, loaded, title } => {
                self.dismiss_ask();
                self.finalize_stream();
                self.running_tools.clear();
                self.status(|status| {
                    status.reset();
                    status.set_title(title.as_deref());
                });
                // The terminal keeps everything physically, but the resize
                // repaint redraws from this store, so the session-global
                // chrome (banner, metadata, and the boundary separator that
                // follows) must survive the conversation switch.
                retain_chrome(&mut self.transcript);
                if !self.initial_session_seen && !loaded {
                    self.initial_session_seen = true;
                    if !self.minimal {
                        self.pending.push(Entry::Separator {
                            label: String::new(),
                        });
                    }
                } else {
                    self.initial_session_seen = true;
                    let label = if loaded {
                        format!("Loaded Session {}", &id[..id.len().min(8)])
                    } else {
                        "New Session".to_owned()
                    };
                    self.pending.push(Entry::Separator { label });
                }
            }
            UiEvent::SessionSnapshot { entries } => {
                self.dismiss_ask();
                self.finalize_stream();
                self.running_tools.clear();
                // Same chrome retention as `SessionChanged`: the snapshot
                // replaces the conversation, not the header.
                retain_chrome(&mut self.transcript);
                // A long session must not load (and print) every entry at
                // once: keep the newest entries within the transcript budget.
                let entries = newest_snapshot_entries(entries, TRANSCRIPT_BUDGET_BYTES);
                if entries.dropped > 0 {
                    self.pending.push(Entry::Notice {
                        text: format!(
                            "{} earlier entries are not shown; export the session to see them",
                            entries.dropped
                        ),
                    });
                }
                for snapshot in entries.kept {
                    self.pending.push(match snapshot {
                        SessionSnapshotEntry::User { text } => Entry::User { text },
                        SessionSnapshotEntry::Assistant {
                            markdown,
                            reasoning,
                        } => Entry::Assistant {
                            markdown,
                            reasoning,
                            continued: false,
                        },
                        SessionSnapshotEntry::Tool {
                            name,
                            summary,
                            ok,
                            duration_ms,
                            output,
                            error,
                        } => Entry::Tool {
                            record: ToolRecord::finished(
                                name,
                                summary,
                                ok,
                                duration_ms,
                                output,
                                error,
                            ),
                        },
                    });
                }
            }
            UiEvent::SessionList { sessions } => {
                self.session_candidates = sessions.clone();
                self.session_fetch = Fetch::Loaded;
                let total = sessions.len();
                let mut notice = if sessions.is_empty() {
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
                if total > 12 {
                    notice.push_str(&format!(
                        "\nShowing 12 of {total} sessions. Use /load and Tab to find a session."
                    ));
                }
                // Completion fetches fill the cache without printing a list.
                if std::mem::take(&mut self.session_list_display_requested) {
                    self.add_notice(notice);
                }
                if commands::is_command_input(&self.input) {
                    self.refresh_completion();
                }
            }
            UiEvent::SessionExported { path } => {
                self.add_notice(format!("exported session to {path}"));
            }
            UiEvent::SessionTitleChanged { title } => {
                self.status(|status| status.set_title(Some(&title)));
            }
            UiEvent::SkillsLoaded {
                skills,
                diagnostics,
                empty,
            } => {
                let mut text = if empty {
                    "no skills discovered\nlooked in .harness/skills and .agents/skills (project and ~)".to_owned()
                } else {
                    skills
                        .iter()
                        .map(|skill| format!("{:<16} {}", skill.name, skill.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
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

    /// Parse newly completed markdown blocks once as deltas arrive. The
    /// unfinished suffix is intentionally left out of the cache because its
    /// Markdown meaning can still change with the next delta.
    fn refresh_stream_markdown_cache(&mut self) {
        let Some(stream) = self.stream.as_mut() else {
            return;
        };
        let stable_offset = stream.split.split_offset(&stream.markdown).unwrap_or(0);
        let stream = &*stream;
        let width = render::content_width(self.width);
        let theme = self.theme;
        let reusable = self.stream_markdown_cache.as_ref().is_some_and(|cache| {
            cache.width == width && cache.theme == theme && cache.source_offset <= stable_offset
        });
        if !reusable {
            self.stream_markdown_cache = Some(StreamMarkdownCache {
                width,
                theme,
                source_offset: 0,
                lines: Vec::new(),
                rendered_blocks: 0,
            });
        }
        let source_offset = self
            .stream_markdown_cache
            .as_ref()
            .map_or(0, |cache| cache.source_offset);
        if stable_offset <= source_offset {
            return;
        }
        let chunk = stream.markdown[source_offset..stable_offset].to_owned();
        let continued = stream.markdown_continued || source_offset > 0;
        let rendered = render::markdown_lines_continued(&chunk, theme, width, continued);
        if let Some(cache) = self.stream_markdown_cache.as_mut() {
            cache.lines.extend(rendered);
            cache.source_offset = stable_offset;
            cache.rendered_blocks = cache.rendered_blocks.saturating_add(1);
        }
    }

    /// Commit the in-flight assistant message as a final entry.
    fn finalize_stream(&mut self) {
        let Some(stream) = self.stream.take() else {
            self.stream_markdown_cache = None;
            return;
        };
        self.stream_markdown_cache = None;
        if stream.reasoning.is_empty() && stream.markdown.is_empty() {
            return;
        }
        self.pending.push(Entry::Assistant {
            markdown: stream.markdown,
            reasoning: stream.reasoning,
            continued: stream.markdown_continued,
        });
    }

    /// Source markdown of the newest completed assistant reply. A long reply
    /// committed in chunks is rejoined, so the copy is the model's text rather
    /// than the wrapped, indented rows shown on screen.
    fn last_reply_markdown(&self) -> Option<String> {
        let entries: Vec<&Entry> = self.transcript.iter().chain(&self.pending).collect();
        let mut index = entries.iter().rposition(
            |entry| matches!(entry, Entry::Assistant { markdown, .. } if !markdown.is_empty()),
        )?;
        let mut chunks = Vec::new();
        while let Entry::Assistant {
            markdown,
            continued,
            ..
        } = entries[index]
        {
            if !markdown.is_empty() {
                chunks.push(markdown.as_str());
                if !continued {
                    break;
                }
            }
            if index == 0 {
                break;
            }
            index -= 1;
        }
        chunks.reverse();
        Some(chunks.concat().trim_end().to_owned())
    }

    /// Queue the last reply (or its last fenced code block) for the system
    /// clipboard through OSC 52 and report what happened.
    fn copy_last_reply(&mut self, code: bool) {
        let Some(markdown) = self.last_reply_markdown() else {
            self.add_error("nothing to copy yet");
            return;
        };
        let text = if code {
            match last_fenced_code_block(&markdown) {
                Some(code) => code,
                None => {
                    self.add_error("the last reply has no code block");
                    return;
                }
            }
        } else {
            markdown
        };
        if text.len() > MAX_COPY_BYTES {
            self.add_error("reply is too large to copy through the terminal");
            return;
        }
        self.clipboard_osc = Some(clipboard_sequence(
            &text,
            std::env::var_os("TMUX").is_some(),
        ));
        let lines = text.lines().count();
        self.add_notice(format!(
            "Copied {} {}{}",
            lines,
            if lines == 1 { "line" } else { "lines" },
            if code { " of code" } else { "" }
        ));
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
            mcp_servers: self.mcp_servers.clone(),
            skills: self.skills.iter().map(|skill| skill.name.clone()).collect(),
        }
    }

    // ------------------------------------------------------------------
    // Frame assembly
    // ------------------------------------------------------------------

    fn input_layout(&self) -> InputLayout {
        let content = render::content_width(self.width);
        let usage_trailer = self.usage.as_ref().map_or_else(String::new, |usage| {
            let context = self.context_usage.map_or_else(String::new, |context| {
                format!(
                    " · {}/{} ctx",
                    format_tokens(context.used_tokens),
                    format_tokens(context.max_tokens)
                )
            });
            format!(
                "   ↑ {} ↓ {}{context}",
                format_tokens(usage.input_tokens),
                format_tokens(usage.output_tokens),
            )
        });
        let input_width = content.saturating_sub(INPUT_PREFIX_WIDTH).max(1);
        let ghost = self.ghost_text();
        let hint = self.completion_hint(input_width);
        input_layout_with_row_width(
            &self.input,
            self.cursor,
            input_width,
            content,
            self.theme,
            &usage_trailer,
            &ghost,
            &hint,
        )
    }

    /// Rows of the live region: [streaming tail] · [tool lines] · [activity]
    /// · [input]. Every section except the input is optional; the tail and
    /// the input are each clipped so the whole region fits the screen.
    #[cfg(test)]
    fn build_region(&self, input: &InputLayout) -> RegionBuild {
        self.build_region_with_running(input, self.running_region())
    }

    fn build_region_with_running(
        &self,
        input: &InputLayout,
        running: RunningRegion,
    ) -> RegionBuild {
        let theme = self.theme;
        let content = render::content_width(self.width);
        // Reserve at least one answer row even for a large
        // draft. The final height guard still covers degenerate terminals.
        let answer_reserve = usize::from(self.ask_answer.is_some());
        let (input_rows, input_cursor_row, input_layout_start) = clip_input(
            input,
            self.height as usize,
            self.busy,
            running.rows + answer_reserve,
        );

        let mut rows: Vec<Line<'static>> = Vec::new();
        let mut activity_row_index: Option<usize> = None;
        // Small screens prioritize answer text over gaps.
        let section_gap = if self.ask_answer.is_some() && self.height < 6 {
            0
        } else {
            render::SECTION_GAP
        };
        if self.ask_answer.is_some() {
            let answer = self.ask_lines(content);
            // Oversized answers were already moved to scrollback by
            // `commit_oversized_ask`, so the whole answer fits here.
            rows.extend(answer.iter().cloned());
        } else {
            // The streaming tail shows only its newest rows; rows that
            // scrolled out are printed in full when the message finalizes.
            let tail = self.stream_tail_lines(content);
            let budget = self.tail_budget(input_rows.len(), running.rows);
            let start = tail.len().saturating_sub(budget);
            rows.extend(tail[start..].iter().cloned());
        }

        // All active tool records render in deterministic launch order
        // (collapsed to one line each unless expanded globally); anything
        // beyond the visible cap folds into one explicit overflow row so a
        // large fan-out stays discoverable without breaking screen height.
        if !running.lines.is_empty() && !rows.is_empty() {
            render::push_blank(&mut rows, section_gap);
        }
        rows.extend(running.lines);

        if self.busy {
            if !rows.is_empty() {
                render::push_blank(&mut rows, section_gap);
            }
            let activity_row = rows.len();
            rows.push(activity_line_with_label(
                self.operation_label
                    .as_deref()
                    .unwrap_or(self.activity.label()),
                self.spinner,
                theme,
            ));
            activity_row_index = Some(activity_row);
        }

        if !rows.is_empty() {
            render::push_blank(&mut rows, section_gap);
        }
        let mut cursor_row = rows.len() + input_cursor_row;
        let mut input_span = Some(InputSpan {
            first_row: rows.len(),
            layout_start: input_layout_start,
            rows: input_rows.len(),
        });
        rows.extend(input_rows);
        // Every producer has a local budget, but keep one final guard here so
        // newly added metadata/tool rows cannot invalidate terminal geometry.
        rows = rows
            .into_iter()
            .map(|line| render::fit_line_to_width(&line, content))
            .collect();

        // Degenerate-terminal guard: the region must never exceed the screen
        // or cursor-relative moves would clamp at the top and corrupt the
        // frame. Keep the bottom rows (the input lives there).
        if self.height > 0 && rows.len() > self.height as usize {
            let dropped = rows.len() - self.height as usize;
            rows.drain(..dropped);
            cursor_row = cursor_row.saturating_sub(dropped).min(rows.len() - 1);
            activity_row_index = activity_row_index.and_then(|row| row.checked_sub(dropped));
            input_span = input_span.and_then(|span| match span.first_row.checked_sub(dropped) {
                Some(first_row) => Some(InputSpan { first_row, ..span }),
                None => {
                    // Top input rows were cut off-screen; skip them.
                    let cut = dropped - span.first_row;
                    (span.rows > cut).then(|| InputSpan {
                        first_row: 0,
                        layout_start: span.layout_start + cut,
                        rows: span.rows - cut,
                    })
                }
            });
        }

        let gutter = render::horizontal_pad(self.width) as usize;
        let prefix_width = INPUT_PREFIX_WIDTH.min(content);
        let cursor_col = gutter
            .saturating_add(prefix_width)
            .saturating_add(input.cursor_col)
            .min(self.width.saturating_sub(1) as usize);
        RegionBuild {
            rows,
            cursor_row,
            cursor_col,
            activity_row: activity_row_index,
            input: input_span,
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
        for (_, label) in self
            .background_operations
            .iter()
            .take(MAX_VISIBLE_RUNNING_TOOLS)
        {
            lines.push(activity_line_with_label(label, self.spinner, theme));
        }
        let hidden_lookups = self
            .background_operations
            .len()
            .saturating_sub(MAX_VISIBLE_RUNNING_TOOLS);
        if hidden_lookups > 0 {
            lines.push(Line::from(format!(
                "… {hidden_lookups} more lookups running"
            )));
        }
        let rows = lines.len();
        RunningRegion { lines, rows }
    }

    fn dismiss_ask(&mut self) {
        self.ask_answer = None;
    }

    /// Split and wrap sanitized source before producing terminal rows. The
    /// complete result stays in this transient inline view, never in the transcript.
    fn ask_lines(&self, width: usize) -> Vec<Line<'static>> {
        let answer = render::sanitize_terminal_text(self.ask_answer.as_deref().unwrap_or(""));
        render::wrap_text(
            &Text::from(answer),
            width,
            Style::default().fg(self.theme.assistant_text),
        )
    }

    /// Rows available to the inline answer beside the input and activity.
    fn ask_budget(&self, input_rows: usize, running_rows: usize) -> usize {
        let gaps = if self.height < 6 {
            0
        } else {
            render::SECTION_GAP * (1 + usize::from(self.busy) + usize::from(running_rows > 0))
        };
        (self.height as usize)
            .saturating_sub(input_rows + running_rows + usize::from(self.busy) + gaps)
            .max(1)
    }

    /// An answer taller than the screen cannot live in the region. Print it
    /// into scrollback instead, where the terminal scrolls it, rather than
    /// clipping or paging it.
    fn commit_oversized_ask(&mut self, input_rows: usize, running_rows: usize) {
        if let Some(text) = self.take_oversized_ask(input_rows, running_rows) {
            self.pending.push(Entry::Notice { text });
        }
    }

    /// Remove and return the answer (sanitized) when it no longer fits.
    fn take_oversized_ask(&mut self, input_rows: usize, running_rows: usize) -> Option<String> {
        self.ask_answer.as_ref()?;
        let budget = self.ask_budget(input_rows, running_rows);
        let rows = self.ask_lines(render::content_width(self.width)).len();
        if rows <= budget {
            return None;
        }
        self.ask_answer
            .take()
            .map(|text| render::sanitize_terminal_text(&text))
    }

    /// Rendered rows of the in-flight assistant message (reasoning block,
    /// then markdown). Re-rendered from source on every frame.
    fn stream_tail_lines(&self, width: usize) -> Vec<Line<'static>> {
        let Some(stream) = &self.stream else {
            return Vec::new();
        };
        let mut lines = Vec::new();
        if !stream.reasoning.is_empty() {
            let mut cache = stream.reasoning_render.borrow_mut();
            let fresh = cache.as_ref().is_some_and(|cached| {
                cached.width == width
                    && cached.theme == self.theme
                    && cached.len == stream.reasoning.len()
            });
            if !fresh {
                *cache = Some(ReasoningRender {
                    width,
                    theme: self.theme,
                    len: stream.reasoning.len(),
                    lines: render::reasoning_lines(&stream.reasoning, self.theme, width),
                });
            }
            if let Some(cached) = cache.as_ref() {
                lines.extend(cached.lines.iter().cloned());
            }
        }
        if !stream.markdown.is_empty() {
            if !lines.is_empty() {
                render::push_blank(&mut lines, render::BLOCK_GAP);
            }
            let markdown = if let Some(cache) = &self.stream_markdown_cache
                && cache.width == width
                && cache.theme == self.theme
                && cache.source_offset <= stream.markdown.len()
            {
                let mut rendered = cache.lines.clone();
                if cache.source_offset < stream.markdown.len() {
                    rendered.extend(render::markdown_lines_continued(
                        &stream.markdown[cache.source_offset..],
                        self.theme,
                        width,
                        stream.markdown_continued || cache.source_offset > 0,
                    ));
                }
                rendered
            } else {
                render::markdown_lines_continued(
                    &stream.markdown,
                    self.theme,
                    width,
                    stream.markdown_continued,
                )
            };
            lines.extend(markdown);
        }
        lines
    }

    /// Row budget for the streaming tail: whatever is left of the screen once
    /// the input, the active-tool rows, the activity row, separators, and a
    /// safety row are reserved.
    fn tail_budget(&self, input_rows: usize, running_rows: usize) -> usize {
        (self.height as usize)
            .saturating_sub(input_rows + running_rows + usize::from(self.busy) + 3)
            .max(1)
    }

    /// Move the stable prefix of the streaming markdown into `pending` once
    /// it outgrows the tail budget, so long responses flow into scrollback
    /// incrementally instead of appearing all at once at finalize.
    /// Reasoning-only streams stay fully live; the display clip handles them.
    #[cfg(test)]
    fn commit_stream_prefix(&mut self, input_rows: usize) {
        let running_rows = self.running_region().rows;
        self.commit_stream_prefix_with_running(input_rows, running_rows);
    }

    fn commit_stream_prefix_with_running(&mut self, input_rows: usize, running_rows: usize) {
        // Snapshot the immutable state first so `self.stream` can be borrowed
        // mutably for the rest of the function.
        let budget = self.tail_budget(input_rows, running_rows);
        let width = render::content_width(self.width);
        let theme = self.theme;
        let Some(stream) = self.stream.as_mut() else {
            return;
        };
        if stream.markdown.is_empty() {
            // A reasoning-only stream would otherwise show just its newest
            // rows until the turn ends, leaving the rest absent from
            // scrollback. Commit whole paragraphs once they outgrow the tail.
            let trimmed = stream.reasoning.trim_end_matches(['\r', '\n']);
            let Some(boundary) = trimmed.rfind("\n\n") else {
                return;
            };
            let head = trimmed[..boundary].trim_end_matches(['\r', '\n']);
            if head.is_empty() || render::reasoning_lines(head, theme, width).len() <= budget {
                return;
            }
            let head = head.to_owned();
            stream.reasoning.drain(..boundary + 2);
            *stream.reasoning_render.borrow_mut() = None;
            self.pending.push(Entry::Assistant {
                markdown: String::new(),
                reasoning: head,
                continued: false,
            });
            return;
        }
        let Some(offset) = stream.split.split_offset(&stream.markdown) else {
            return;
        };
        if offset == 0 {
            return;
        }
        let prefix = Entry::Assistant {
            markdown: stream.markdown[..offset].to_owned(),
            reasoning: stream.reasoning.clone(),
            continued: stream.markdown_continued,
        };
        let prefix_height = entry_lines(&prefix, width, theme, self.tools_expanded).len();
        if prefix_height <= budget {
            return;
        }
        self.pending.push(prefix);
        stream.reasoning.clear();
        *stream.reasoning_render.borrow_mut() = None;
        stream.markdown.drain(..offset);
        stream.markdown_continued = true;
        stream.split = SplitScanner::default();
        // The source origin moved; cached offsets are no longer meaningful.
        self.stream_markdown_cache = None;
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
        let running = self.running_region();
        self.commit_stream_prefix_with_running(input.rows.len(), running.rows);
        self.commit_oversized_ask(input.rows.len(), running.rows);
        let build = self.build_region_with_running(&input, running);
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
        let (build, oversized) = self.stage_full_repaint();
        self.write_frame(build, true)?;
        // The clipped history repaint above cannot show a newly oversized
        // answer, so print it the normal incremental way afterwards: it then
        // reaches scrollback in full, above the live region.
        if let Some(text) = oversized {
            self.pending.push(Entry::Notice { text });
            self.paint()?;
        }
        Ok(())
    }

    /// Prepare a full repaint. A stream prefix committed here must reach the
    /// transcript before the frame is built: the full repaint prints only
    /// `transcript`, and the frame bookkeeping then moves `pending` into it
    /// without printing, so a prefix committed after the move was lost from
    /// the screen on resize or Ctrl+O.
    ///
    /// An answer that no longer fits is returned (and removed from the live
    /// region) rather than queued: queuing it here would move it into the
    /// transcript unprinted, and the clipped repaint would drop its head.
    fn stage_full_repaint(&mut self) -> (RegionBuild, Option<String>) {
        let input = self.input_layout();
        let running = self.running_region();
        self.commit_stream_prefix_with_running(input.rows.len(), running.rows);
        commit_to_transcript(&mut self.transcript, &mut self.pending);
        let oversized = self.take_oversized_ask(input.rows.len(), running.rows);
        (self.build_region_with_running(&input, running), oversized)
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
            // The separator row between history and the region counts too:
            // without it the frame is one row taller than the screen and its
            // top row scrolls into native scrollback on every full repaint.
            let gap = if build.rows.is_empty() {
                0
            } else {
                render::SECTION_GAP
            };
            let keep = (self.height as usize)
                .saturating_sub(build.rows.len())
                .saturating_sub(gap);
            above = history_window(&self.transcript, keep, content, theme, self.tools_expanded);
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
        // OSC 7501 moves no cursor and draws nothing; leading the frame keeps
        // it from splitting any of the frame's own escape sequences.
        if let Some(report) = self.program_status.as_mut().and_then(|s| s.take_report()) {
            buffer.push_str(&report);
        }
        // Synchronized output keeps a tall frame from tearing, and autowrap is
        // off while it is written: a row the terminal measures wider than we
        // do is clipped instead of wrapping into an untracked row, which would
        // desynchronize the region bookkeeping and strand stale rows.
        if let Some(copy) = self.clipboard_osc.take() {
            buffer.push_str(&copy);
        }
        buffer.push_str(FRAME_BEGIN);
        if clear_all {
            // ED 0 from home rather than ED 2: several terminals push the
            // whole screen into scrollback on ED 2, duplicating the visible
            // window above the repaint on every resize or Ctrl+O.
            let _ = write!(
                buffer,
                "{}{}",
                MoveTo(0, 0),
                Clear(ClearType::FromCursorDown)
            );
        } else if !self.region.is_empty() {
            // Walk the cursor up to the top of the previously painted region.
            // The cursor sits at `cursor_row` inside the region, so exactly
            // that many rows separate it from the top.
            if self.cursor_row > 0 {
                let _ = write!(buffer, "{}", MoveUp(self.cursor_row as u16));
            }
        }

        // OSC 133 marks tell the terminal which rows are the prompt and ask it
        // to report clicks there (`click_events`), which `click_to_cursor`
        // maps back to the draft. Terminals without it ignore the marks.
        let region_start = total - build.rows.len();
        let prompt = build.input.map(|span| {
            (
                region_start + span.first_row,
                region_start + span.first_row + span.rows,
            )
        });
        for (index, line) in rows.iter().enumerate() {
            let _ = write!(buffer, "\r{}", Clear(ClearType::UntilNewLine));
            match prompt {
                Some((first, _)) if index == first => buffer.push_str(PROMPT_START),
                Some((_, end)) if index == end => buffer.push_str(PROMPT_END),
                _ => {}
            }
            write_row(&mut buffer, line, gutter, self.width as usize);
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
        buffer.push_str(FRAME_END);

        self.out
            .write_all(buffer.as_bytes())
            .context("write frame")?;
        self.out.flush().context("flush frame")?;

        // Where the region begins on screen: the frame prints from the old
        // region top (or home after a clear), and growth past the bottom
        // scrolls the terminal so the frame ends on the last row.
        let top = if clear_all { Some(0) } else { self.region_top };
        self.region_top = top.map(|top| {
            let last = (top + total - 1).min(self.height.saturating_sub(1) as usize);
            (last + 1).saturating_sub(build.rows.len())
        });
        self.input_span = build.input;

        // Bookkeeping: the printed `above` rows are now immutable scrollback.
        commit_to_transcript(&mut self.transcript, &mut self.pending);
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
        let mut terminal = lock_terminal(&self.terminal);
        restore_terminal(&mut terminal, &mut self.out)
    }
}

/// Restore every state component that still needs work. An operation that
/// fails remains recorded, allowing a later call from `run`, `Drop`, or a
/// panic path to retry it. All operations are attempted before the first
/// error is returned.
fn restore_terminal(state: &mut TerminalState, out: &mut Stdout) -> Result<()> {
    if state.is_restored() {
        return Ok(());
    }

    let mut first_error = None;
    if state.modes.raw_mode {
        if let Err(error) = state
            .backend
            .disable_raw_mode()
            .context("restore terminal raw mode")
        {
            first_error.get_or_insert(error);
        } else {
            state.modes.raw_mode = false;
        }
    }
    if state.modes.bracketed_paste {
        if let Err(error) = state
            .backend
            .disable_bracketed_paste(out)
            .context("restore bracketed paste")
        {
            first_error.get_or_insert(error);
        } else {
            state.modes.bracketed_paste = false;
        }
    }
    if state.modes.keyboard_flags {
        if let Err(error) = state
            .backend
            .pop_keyboard_flags(out)
            .context("restore keyboard enhancement flags")
        {
            first_error.get_or_insert(error);
        } else {
            state.modes.keyboard_flags = false;
        }
    }
    if !state.modes.newline_written {
        if let Err(error) = state.backend.write_newline(out).context("leave terminal") {
            first_error.get_or_insert(error);
        } else {
            state.modes.newline_written = true;
        }
    }

    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
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

/// Lock the shared terminal state. A panic while the lock was held must not
/// stop later restoration, so a poisoned lock is recovered: the state only
/// records which modes still need to be undone.
fn lock_terminal(terminal: &Mutex<TerminalState>) -> MutexGuard<'_, TerminalState> {
    terminal.lock().unwrap_or_else(PoisonError::into_inner)
}

fn install_panic_hook(terminal: &Arc<Mutex<TerminalState>>) {
    let terminal = Arc::downgrade(terminal);
    let ui_thread = std::thread::current().id();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        if std::thread::current().id() == ui_thread
            && let Some(terminal) = terminal.upgrade()
        {
            let guard = match terminal.try_lock() {
                Ok(guard) => Some(guard),
                Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            };
            if let Some(mut guard) = guard {
                let mut stdout = io::stdout();
                // Panic hooks cannot report a cleanup error without obscuring
                // the original panic, but they still use the same best-effort
                // operation layer. A caught panic or a later Drop can retry
                // anything that failed.
                let _ = restore_terminal(&mut guard, &mut stdout);
            }
        }
        previous(panic);
    }));
}

// ---------------------------------------------------------------------------
// Entry rendering
// ---------------------------------------------------------------------------

/// An accent bar and aligned, muted labels make startup details easy to scan.
/// Values use the terminal's normal text colour; optional context, skill, and
/// MCP rows remain capped at [`METADATA_MAX_ENTRIES`] names.
#[expect(
    clippy::too_many_arguments,
    reason = "the renderer keeps each metadata row explicit for focused tests"
)]
// ---------------------------------------------------------------------------
// Input line layout
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Input editing (pure helpers over `String` + byte cursor)
// ---------------------------------------------------------------------------

// Editing moves and deletes whole grapheme clusters (a family emoji, a
// flag, an accented letter with a combining mark), never part of one.

// ---------------------------------------------------------------------------
// ANSI output
// ---------------------------------------------------------------------------
mod usage;
use usage::*;

mod ansi;
use ansi::*;

mod editor;
use editor::*;

mod transcript;
use transcript::*;

#[cfg(test)]
mod tests;
