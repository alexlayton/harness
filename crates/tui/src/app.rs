use crate::attachments;
use crate::commands::{self, Candidate, CandidateKind, ParsedCommand};
use crate::input::{InputAction, classify, history_next, history_previous, push_history};
use crate::render;
use crate::render::{TailTool, ToolRecord};
use crate::{InputMessage, ModelEntry, TuiEvent, UiEvent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
    KeyModifiers,
};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Paragraph;
use ratatui::{Terminal, TerminalOptions, Viewport};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui_textarea::{CursorMove, TextArea};

const MAX_LIVE_ROWS: u16 = 10;
const MAX_INPUT_ROWS: u16 = 10;
const VIEWPORT_ROWS: u16 = 22;
const MAX_TAIL_TOOLS: usize = 5;
const MAX_RECENT_TOOLS: usize = 20;
const MAX_HISTORY: usize = 1_000;
const PLACEHOLDER: &str = "Type your message...";

#[derive(Clone, Debug)]
struct RunningTool {
    name: String,
    summary: String,
    arguments: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionKind {
    Slash,
    Files,
}

#[derive(Clone, Debug)]
struct Completion {
    candidates: Vec<Candidate>,
    selected: usize,
    offset: usize,
    kind: CompletionKind,
}

#[derive(Debug)]
struct FileCompletionResult {
    generation: u64,
    query: String,
    candidates: Vec<Candidate>,
}

pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    model: String,
    provider: String,
    providers: Vec<String>,
    model_lists: HashMap<String, Vec<ModelEntry>>,
    workspace_root: PathBuf,
    completion: Option<Completion>,
    file_completion_tx: mpsc::UnboundedSender<FileCompletionResult>,
    file_completion_rx: mpsc::UnboundedReceiver<FileCompletionResult>,
    file_completion_generation: u64,
    file_completion_query: Option<String>,
    file_completion_cancel: Option<CancellationToken>,
    textarea: TextArea<'static>,
    pending_text: String,
    pending_reasoning: String,
    running_tool: Option<RunningTool>,
    tail: Vec<TailTool>,
    recent_tools: VecDeque<ToolRecord>,
    focused_tool: Option<usize>,
    spinner: usize,
    busy: bool,
    retrying: Option<u32>,
    status_flash: Option<String>,
    needs_section_break: bool,
    header_printed: bool,
    history: Vec<String>,
    history_pos: Option<usize>,
    draft: String,
    restored: bool,
    session_id: Option<String>,
    session_title: Option<String>,
    usage: Option<UsageState>,
}

#[derive(Clone, Debug, Default)]
struct UsageState {
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    reasoning_tokens: u64,
    cost: String,
}

impl Tui {
    pub fn new(model: &str, provider: &str, providers: Vec<String>) -> Result<Self> {
        install_panic_hook();
        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnableBracketedPaste, cursor::Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("configure terminal input");
        }
        let (_, rows) = terminal::size().unwrap_or((80, 24));
        let viewport_rows = rows.clamp(4, VIEWPORT_ROWS);
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let (file_completion_tx, file_completion_rx) = mpsc::unbounded_channel();
        let backend = CrosstermBackend::new(stdout);
        let terminal = match Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(viewport_rows),
            },
        ) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = terminal::disable_raw_mode();
                return Err(error).context("create inline terminal viewport");
            }
        };
        Ok(Self {
            terminal,
            model: model.to_owned(),
            provider: provider.to_owned(),
            providers,
            model_lists: HashMap::new(),
            workspace_root,
            completion: None,
            file_completion_tx,
            file_completion_rx,
            file_completion_generation: 0,
            file_completion_query: None,
            file_completion_cancel: None,
            textarea: new_textarea(),
            pending_text: String::new(),
            pending_reasoning: String::new(),
            running_tool: None,
            tail: Vec::new(),
            recent_tools: VecDeque::new(),
            focused_tool: None,
            spinner: 0,
            busy: false,
            retrying: None,
            status_flash: None,
            needs_section_break: false,
            header_printed: false,
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            restored: false,
            session_id: None,
            session_title: None,
            usage: None,
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
        self.draw()?;
        let mut input_events = EventStream::new();
        let mut spinner_tick = tokio::time::interval(Duration::from_millis(80));
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
                    if self.running_tool.is_some() {
                        self.spinner = self.spinner.wrapping_add(1);
                        self.draw()?;
                    }
                }
                _ = cancel.cancelled(), if !self.busy => {
                    // Cancellation while idle is the application's shutdown
                    // path. During a turn the agent sends TurnFinished and the
                    // input handler remains available for a final redraw.
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

        // Application quit always wins over a tool-focus binding.
        if matches!(classify(&event), InputAction::Quit) {
            cancel.cancel();
            return Ok(true);
        }

        // Completion navigation has precedence over history and tool focus.
        if self.handle_completion_input(&event, input_tx)? {
            return Ok(false);
        }

        if !self.busy && self.focused_tool.is_some() && self.handle_tool_focus(&event)? {
            return Ok(false);
        }

        match classify(&event) {
            InputAction::Quit => {
                cancel.cancel();
                return Ok(true);
            }
            InputAction::Interrupt => {
                if self.focused_tool.take().is_some() && !self.busy {
                    // Esc exits tool focus while idle rather than interrupting
                    // a future turn.
                    return Ok(false);
                }
                if self.busy {
                    // Esc is a turn-local interrupt. The agent receives this
                    // control message and creates a fresh token for the next
                    // queued turn; Ctrl+C/D remain application quit.
                    let _ = input_tx.send(InputMessage::Interrupt);
                }
            }
            InputAction::Newline => {
                self.history_pos = None;
                self.textarea.input(event);
                self.refresh_completion();
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
                // A queued user message must come after any model content that
                // is currently in the redrawable area, and after finished tail
                // tools. The running tool itself remains live at the bottom.
                self.flush_everything()?;
                self.commit_tail()?;
                let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
                render::insert_user(&mut self.terminal, &input, width)
                    .context("echo user message")?;
                push_history(&mut self.history, &input, MAX_HISTORY);
                self.history_pos = None;
                self.draft.clear();
                input_tx
                    .send(InputMessage::Message(input))
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                self.textarea = new_textarea();
                self.close_completion();
                self.needs_section_break = true;
                self.busy = true;
                self.retrying = None;
                self.status_flash = None;
            }
            InputAction::ExpandDetails => {
                // Pi-style toggle: expand/collapse the focused tool call, or
                // the most recent one still in the live tail.  Once every
                // tool has been committed to scrollback, fall back to dumping
                // the latest call's full detail view.
                if self.focused_tool.is_some() {
                    self.toggle_focused_tool();
                } else if let Some(entry) = self.tail.last_mut() {
                    entry.expanded = !entry.expanded;
                } else {
                    self.expand_latest_tool()?;
                }
            }
            InputAction::FocusTools => {
                if !self.busy && !self.tail.is_empty() {
                    self.focused_tool = Some(0);
                } else {
                    // Tab remains a normal textarea insertion while a turn is
                    // running (or before any tool exists).
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event);
                    self.refresh_completion();
                }
            }
            InputAction::Edit => {
                let history_navigation = match &event {
                    Event::Key(key) if key.code == KeyCode::Up => self.navigate_history_up(),
                    Event::Key(key) if key.code == KeyCode::Down => self.navigate_history_down(),
                    _ => false,
                };
                if !history_navigation {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event);
                }
                self.refresh_completion();
            }
            InputAction::Ignore => {}
        }
        Ok(false)
    }

    fn handle_completion_input(
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
            KeyCode::Esc if !self.busy => {
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
        let Some((candidate, kind)) = self.completion.as_ref().and_then(|completion| {
            completion
                .candidates
                .get(completion.selected)
                .cloned()
                .map(|candidate| (candidate, completion.kind))
        }) else {
            return Ok(());
        };

        if kind == CompletionKind::Files {
            self.accept_file_completion(&candidate);
            return Ok(());
        }

        let input = self.textarea.lines().join("\n");
        let replacement = replace_current_command_token(&input, &candidate.value);
        self.replace_textarea(&replacement);
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
        if let Some(provider) = provider_to_fetch
            && !self.model_lists.contains_key(&provider)
        {
            input_tx
                .send(InputMessage::ListModels {
                    provider: provider.clone(),
                })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
            self.status_flash = Some("model list still loading…".into());
            // Keep the completion state alive while the asynchronous list is
            // on its way, even though there are no model rows to render yet.
            if self.completion.is_none() {
                self.completion = Some(Completion {
                    candidates: Vec::new(),
                    selected: 0,
                    offset: 0,
                    kind: CompletionKind::Slash,
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

    fn refresh_completion(&mut self) {
        let input = self.textarea.lines().join("\n");
        if commands::is_command_input(&input) {
            self.cancel_file_completion();
            let candidates =
                commands::candidates(&input, &self.providers, &self.model_lists, &self.provider);
            if candidates.is_empty() {
                self.completion = None;
                return;
            }
            let old_value = self.completion.as_ref().and_then(|completion| {
                (completion.kind == CompletionKind::Slash)
                    .then(|| {
                        completion
                            .candidates
                            .get(completion.selected)
                            .map(|candidate| candidate.value.as_str())
                    })
                    .flatten()
            });
            let selected = old_value
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
                kind: CompletionKind::Slash,
            });
            self.keep_completion_visible();
            return;
        }

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
        });

        tokio::spawn(async move {
            // Avoid starting a filesystem walk for every individual character
            // when the user is typing quickly.
            tokio::time::sleep(Duration::from_millis(20)).await;
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
            });
        });
    }

    fn apply_file_completion(&mut self, result: FileCompletionResult) {
        if result.generation != self.file_completion_generation {
            return;
        }
        let (line_index, cursor_col) = self.textarea.cursor();
        let Some(line) = self.textarea.lines().get(line_index) else {
            return;
        };
        let Some(prefix) = attachments::extract_at_prefix(line, cursor_col) else {
            return;
        };
        if prefix.query != result.query {
            return;
        }

        self.file_completion_cancel = None;
        if result.candidates.is_empty() {
            self.completion = None;
            return;
        }
        self.completion = Some(Completion {
            candidates: result.candidates,
            selected: 0,
            offset: 0,
            kind: CompletionKind::Files,
        });
        self.keep_completion_visible();
    }

    fn cancel_file_completion(&mut self) {
        if let Some(cancel) = self.file_completion_cancel.take() {
            cancel.cancel();
        }
        self.file_completion_query = None;
        self.file_completion_generation = self.file_completion_generation.wrapping_add(1);
        if self
            .completion
            .as_ref()
            .is_some_and(|completion| completion.kind == CompletionKind::Files)
        {
            self.completion = None;
        }
    }

    fn close_completion(&mut self) {
        self.cancel_file_completion();
        self.completion = None;
    }

    fn submit_command(
        &mut self,
        input: &str,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let command = match commands::parse_command(input) {
            Ok(command) => command,
            Err(error) => {
                self.status_flash = Some(error);
                return Ok(());
            }
        };
        if let ParsedCommand::SetModel {
            provider: None,
            model,
        } = &command
            && model.is_empty()
        {
            self.status_flash = Some(format!(
                "usage: /model [<provider>:]<model> (current: {} · {})",
                self.provider, self.model
            ));
            return Ok(());
        }

        self.flush_everything()?;
        self.commit_tail()?;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_command_echo(&mut self.terminal, input, width).context("echo command")?;
        push_history(&mut self.history, input, MAX_HISTORY);
        self.history_pos = None;
        self.draft.clear();
        self.textarea = new_textarea();
        self.close_completion();
        self.status_flash = None;

        match command {
            ParsedCommand::New => input_tx
                .send(InputMessage::NewConversation)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Load { selector } => input_tx
                .send(InputMessage::LoadSession { selector })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Sessions => input_tx
                .send(InputMessage::ListSessions)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Export { destination } => input_tx
                .send(InputMessage::ExportSession { destination })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::Compact => input_tx
                .send(InputMessage::CompactSession)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?,
            ParsedCommand::SetModel { provider, model } => {
                let provider_for_fetch = provider.clone().and_then(|name| {
                    self.providers
                        .iter()
                        .find(|known| known.eq_ignore_ascii_case(&name))
                        .cloned()
                });
                input_tx
                    .send(InputMessage::SetModel { provider, model })
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                if let Some(provider) = provider_for_fetch
                    && !self.model_lists.contains_key(&provider)
                {
                    input_tx
                        .send(InputMessage::ListModels {
                            provider: provider.clone(),
                        })
                        .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                }
            }
        }
        Ok(())
    }

    fn handle_tool_focus(&mut self, event: &Event) -> Result<bool> {
        let Event::Key(key) = event else {
            self.focused_tool = None;
            return Ok(false);
        };
        match key.code {
            KeyCode::Esc => {
                self.focused_tool = None;
                Ok(true)
            }
            KeyCode::Up => {
                if let Some(index) = self.focused_tool.as_mut() {
                    *index = index.saturating_sub(1);
                }
                Ok(true)
            }
            KeyCode::Down => {
                if let Some(index) = self.focused_tool.as_mut() {
                    *index = (*index + 1).min(self.tail.len().saturating_sub(1));
                }
                Ok(true)
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.toggle_focused_tool();
                Ok(true)
            }
            KeyCode::Char('d') if key.modifiers.is_empty() => {
                // Dump the focused call's full detail into scrollback, then
                // hand focus back to the editor.
                self.dump_focused_tool()?;
                self.focused_tool = None;
                Ok(true)
            }
            KeyCode::Char(value)
                if value.eq_ignore_ascii_case(&'o')
                    && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                // Ctrl+O expands/collapses in place, matching the global
                // binding; Enter or Space work too.
                self.toggle_focused_tool();
                Ok(true)
            }
            KeyCode::Tab => {
                if !self.tail.is_empty() {
                    let index = self.focused_tool.unwrap_or(0);
                    self.focused_tool = Some((index + 1) % self.tail.len());
                }
                Ok(true)
            }
            _ => {
                // Leave focus and let ordinary input handling process the key.
                self.focused_tool = None;
                Ok(false)
            }
        }
    }

    fn toggle_focused_tool(&mut self) {
        if let Some(index) = self.focused_tool
            && let Some(entry) = self.tail.get_mut(index)
        {
            entry.expanded = !entry.expanded;
        }
    }

    fn navigate_history_up(&mut self) -> bool {
        if commands::is_command_input(&self.textarea.lines().join("\n")) {
            return false;
        }
        let (line, _) = self.textarea.cursor();
        if line != 0 {
            return false;
        }
        let current = self.textarea.lines().join("\n");
        let Some(value) = history_previous(
            &self.history,
            &mut self.history_pos,
            &mut self.draft,
            &current,
        ) else {
            return false;
        };
        self.replace_textarea(&value);
        true
    }

    fn navigate_history_down(&mut self) -> bool {
        if commands::is_command_input(&self.textarea.lines().join("\n")) {
            return false;
        }
        let (line, _) = self.textarea.cursor();
        if line + 1 < self.textarea.lines().len() {
            return false;
        }
        let Some(value) = history_next(&self.history, &mut self.history_pos, &self.draft) else {
            return false;
        };
        self.replace_textarea(&value);
        true
    }

    fn replace_textarea(&mut self, value: &str) {
        self.textarea = textarea_with_text(value);
    }

    fn apply_event(&mut self, event: UiEvent) -> Result<()> {
        self.status_flash = None;
        match event {
            UiEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                self.busy = true;
                self.retrying = None;
                // Reasoning is committed immediately before the first text
                // delta so the two streams remain interleaved.
                self.flush_pending_reasoning()?;
                self.ensure_section_break()?;
                self.pending_text.push_str(&delta);
                self.flush_stable_text()?;
                self.flush_overflow_text()?;
            }
            UiEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                self.busy = true;
                self.retrying = None;
                // A provider normally sends reasoning before text. If a
                // provider interleaves the other way, commit the text already
                // received before placing reasoning above it.
                if !self.pending_text.is_empty() {
                    self.flush_pending_text_all()?;
                }
                self.ensure_section_break()?;
                self.pending_reasoning.push_str(&delta);
            }
            UiEvent::ToolCallStarted {
                name,
                summary,
                arguments,
            } => {
                self.busy = true;
                self.retrying = None;
                // Everything already streamed belongs before the running tool.
                self.flush_everything()?;
                self.commit_tail()?;
                self.running_tool = Some(RunningTool {
                    name,
                    summary,
                    arguments,
                });
                self.spinner = 0;
            }
            UiEvent::ToolCallFinished {
                name,
                summary,
                ok,
                duration_ms,
                output,
                error,
            } => {
                self.busy = true;
                let arguments = self
                    .running_tool
                    .take()
                    .map(|tool| tool.arguments)
                    .unwrap_or_else(|| "{}".into());
                let record = ToolRecord {
                    name,
                    args: arguments,
                    summary,
                    ok,
                    duration_ms,
                    output,
                    error,
                };
                self.recent_tools.push_back(record.clone());
                while self.recent_tools.len() > MAX_RECENT_TOOLS {
                    self.recent_tools.pop_front();
                }

                // Keep only a small interactive tail. Overflow is committed in
                // chronological order before the new entry is made live.
                if self.tail.len() >= MAX_TAIL_TOOLS {
                    let oldest = self.tail.remove(0);
                    self.commit_tool_record(&oldest.record)?;
                }
                self.tail.push(TailTool {
                    record,
                    expanded: false,
                });
                self.focused_tool = None;
                self.retrying = None;
                // The next assistant text after a tool begins a new segment.
                self.needs_section_break = true;
            }
            UiEvent::Retrying { attempt, .. } => {
                self.busy = true;
                self.retrying = Some(attempt);
            }
            UiEvent::Error(error) => {
                self.flush_everything()?;
                self.insert_error(&error)?;
                self.running_tool = None;
                self.busy = false;
            }
            UiEvent::TurnFinished => {
                // Do not commit an otherwise idle tail here: it remains
                // interactively expandable until newer content needs the
                // scrollback position.
                self.flush_everything()?;
                self.running_tool = None;
                self.retrying = None;
                self.busy = false;
                self.focused_tool = None;
            }
            UiEvent::Notice(notice) => {
                self.flush_everything()?;
                self.commit_tail()?;
                let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
                render::insert_notice(&mut self.terminal, &notice, width)
                    .context("insert notice")?;
            }
            UiEvent::ModelChanged { provider, model } => {
                self.provider = provider;
                self.model = model;
                if commands::is_command_input(&self.textarea.lines().join("\n")) {
                    self.refresh_completion();
                }
            }
            UiEvent::ModelList { provider, models } => {
                self.model_lists.insert(provider, models);
                if commands::is_command_input(&self.textarea.lines().join("\n")) {
                    self.refresh_completion();
                }
            }
            UiEvent::SessionChanged { id, title, loaded } => {
                self.session_id = Some(id.clone());
                self.session_title = title;
                self.usage = None;
                self.flush_everything()?;
                self.commit_tail()?;
                self.status_flash = Some(if loaded {
                    format!("loaded session {}", &id[..id.len().min(8)])
                } else {
                    format!("session {}", &id[..id.len().min(8)])
                });
            }
            UiEvent::SessionList { sessions } => {
                self.flush_everything()?;
                self.commit_tail()?;
                let notice = if sessions.is_empty() {
                    "No sessions for this workspace".to_owned()
                } else {
                    sessions
                        .into_iter()
                        .take(12)
                        .map(|session| {
                            let title = session.title.unwrap_or_else(|| "(untitled)".into());
                            let model = session.model.unwrap_or_else(|| "(model unknown)".into());
                            format!(
                                "{} · {} · {} · {}",
                                session.short_id, title, model, session.updated_at
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
                render::insert_notice(&mut self.terminal, &notice, width)
                    .context("insert session list")?;
            }
            UiEvent::SessionExported { path } => {
                self.status_flash = Some(format!("exported to {path}"));
            }
            UiEvent::UsageUpdated {
                input_tokens,
                output_tokens,
                cached_tokens,
                reasoning_tokens,
                cost,
            } => {
                self.usage = Some(UsageState {
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                    reasoning_tokens,
                    cost,
                });
            }
            UiEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
            } => {
                self.status_flash = Some(format!(
                    "compacted through event {compacted_through} ({summary_bytes} bytes)"
                ));
            }
        }
        Ok(())
    }

    fn ensure_section_break(&mut self) -> Result<()> {
        if !self.needs_section_break {
            return Ok(());
        }
        self.commit_tail()?;
        render::insert_section_gap(&mut self.terminal).context("insert section gap")?;
        self.needs_section_break = false;
        Ok(())
    }

    fn flush_stable_text(&mut self) -> Result<()> {
        let (stable, pending) = render::split_stable_prefix(&self.pending_text);
        if stable.is_empty() {
            return Ok(());
        }
        self.pending_text = pending;
        self.insert_markdown(&stable)
    }

    fn flush_overflow_text(&mut self) -> Result<()> {
        if render::wrap_count(
            &self.pending_text,
            self.terminal.size().map(|size| size.width).unwrap_or(80) as usize,
        ) <= MAX_LIVE_ROWS as usize
        {
            return Ok(());
        }
        let Some(index) = self.pending_text.rfind('\n') else {
            return Ok(());
        };
        let prefix = self.pending_text[..=index].to_owned();
        self.pending_text = self.pending_text[index + 1..].to_owned();
        self.insert_markdown(&prefix)
    }

    fn insert_markdown(&mut self, markdown: &str) -> Result<()> {
        if markdown.is_empty() {
            return Ok(());
        }
        self.ensure_section_break()?;
        self.commit_tail()?;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_markdown(&mut self.terminal, markdown, width).context("insert markdown")?;
        Ok(())
    }

    fn flush_pending_reasoning(&mut self) -> Result<()> {
        if self.pending_reasoning.is_empty() {
            return Ok(());
        }
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        self.ensure_section_break()?;
        self.commit_tail()?;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_reasoning(&mut self.terminal, &reasoning, width)
            .context("insert reasoning")?;
        Ok(())
    }

    fn flush_pending_text_all(&mut self) -> Result<()> {
        if self.pending_text.is_empty() {
            return Ok(());
        }
        let text = std::mem::take(&mut self.pending_text);
        self.insert_markdown(&text)
    }

    fn flush_everything(&mut self) -> Result<()> {
        self.flush_pending_reasoning()?;
        self.flush_pending_text_all()?;
        Ok(())
    }

    /// Commit all finished live-tail tools in their original order.  Every
    /// path that writes a new scrollback block calls this first.
    fn commit_tail(&mut self) -> Result<()> {
        while !self.tail.is_empty() {
            let entry = self.tail.remove(0);
            self.commit_tool_record(&entry.record)?;
        }
        self.focused_tool = None;
        Ok(())
    }

    fn commit_tool_record(&mut self, record: &ToolRecord) -> Result<()> {
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_tool_finished(
            &mut self.terminal,
            &record.name,
            &record.summary,
            record.ok,
            record.duration_ms,
            record.error.as_deref(),
            width,
        )
        .context("insert completed tool")?;
        self.needs_section_break = true;
        Ok(())
    }

    fn insert_error(&mut self, error: &str) -> Result<()> {
        self.commit_tail()?;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_error(&mut self.terminal, error, width).context("insert error")?;
        Ok(())
    }

    fn expand_latest_tool(&mut self) -> Result<()> {
        let Some(record) = self.recent_tools.back().cloned() else {
            self.status_flash = Some("no tool calls yet".into());
            return Ok(());
        };
        self.append_tool_detail(record)
    }

    fn dump_focused_tool(&mut self) -> Result<()> {
        let record = self
            .focused_tool
            .and_then(|index| self.tail.get(index))
            .map(|entry| entry.record.clone())
            .or_else(|| self.recent_tools.back().cloned());
        let Some(record) = record else {
            self.status_flash = Some("no tool calls yet".into());
            return Ok(());
        };
        self.append_tool_detail(record)
    }

    fn append_tool_detail(&mut self, record: ToolRecord) -> Result<()> {
        self.flush_everything()?;
        self.commit_tail()?;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_tool_detail(&mut self.terminal, &record, width)
            .context("insert tool details")?;
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        if !self.header_printed {
            let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
            render::insert_header(&mut self.terminal, width).context("insert welcome header")?;
            self.header_printed = true;
        }

        self.textarea.set_block(render::input_block());
        let model = self.model.clone();
        let provider = self.provider.clone();
        let pending_text = self.pending_text.clone();
        let pending_reasoning = self.pending_reasoning.clone();
        let running_tool = self.running_tool.clone();
        let tail = self.tail.clone();
        let focused_tool = self.focused_tool;
        let spinner = self.spinner;
        let busy = self.busy;
        let retrying = self.retrying;
        let status_flash = self.status_flash.clone();
        let session_id = self.session_id.clone();
        let session_title = self.session_title.clone();
        let usage = self.usage.clone();
        let completion = self.completion.clone();
        let textarea = &self.textarea;
        let clear_flash = self.status_flash.is_some();

        self.terminal.draw(|frame| {
            let area = frame.area();
            let requested_completion_rows = completion
                .as_ref()
                .map(|completion| {
                    completion.candidates.len().min(render::MAX_COMPLETION_ROWS) as u16
                })
                .unwrap_or(0);
            // Reserve the status line and the input box (text + 2 border rows).
            let completion_rows = requested_completion_rows.min(area.height.saturating_sub(4));
            let inner_rows = (textarea.lines().len() as u16)
                .clamp(1, MAX_INPUT_ROWS)
                .min(area.height.saturating_sub(3 + completion_rows).max(1));
            let input_rows = inner_rows + 2;
            let mut constraints = vec![
                Constraint::Min(1),
                Constraint::Length(1),
                Constraint::Length(input_rows),
            ];
            if completion.is_some() {
                constraints.push(Constraint::Length(completion_rows));
            }
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(area);

            let running = running_tool
                .as_ref()
                .map(|tool| (tool.name.as_str(), tool.summary.as_str(), spinner));
            render::render_live_with_tail(
                chunks[0],
                &tail,
                focused_tool,
                &pending_reasoning,
                &pending_text,
                running,
                frame,
            );

            let dim_style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM);
            let session_label = session_id
                .as_deref()
                .map(|id| format!(" · session {}", &id[..id.len().min(8)]))
                .unwrap_or_default();
            let title_label = session_title
                .as_deref()
                .filter(|title| !title.is_empty())
                .map(|title| format!(" · {title}"))
                .unwrap_or_default();
            let left = format!("{provider} · {model}{session_label}{title_label}");
            let (right, right_style) = if let Some(flash) = status_flash.as_deref() {
                (flash.to_owned(), Style::default().fg(Color::Cyan))
            } else if let Some(attempt) = retrying {
                (
                    format!("↻ retrying (attempt {attempt})"),
                    Style::default().fg(Color::Yellow),
                )
            } else if busy {
                ("… generating".to_owned(), dim_style)
            } else if let Some(usage) = usage {
                let cost = if usage.cost != "0" {
                    format!(" · ${}", usage.cost)
                } else {
                    String::new()
                };
                (
                    format!(
                        "{} in · {} out · {} cached · {} reasoning{cost}",
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.cached_tokens,
                        usage.reasoning_tokens,
                    ),
                    dim_style,
                )
            } else if focused_tool.is_some() {
                (
                    "↑/↓ select · enter expand · d dump · esc close".to_owned(),
                    dim_style,
                )
            } else {
                (String::new(), dim_style)
            };
            let status = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(left.chars().count().min(u16::MAX as usize) as u16),
                    Constraint::Min(1),
                    Constraint::Length(right.chars().count().min(u16::MAX as usize) as u16),
                ])
                .split(chunks[1]);
            frame.render_widget(Paragraph::new(left).style(dim_style), status[0]);
            if !right.is_empty() {
                frame.render_widget(
                    Paragraph::new(right)
                        .alignment(ratatui::layout::Alignment::Right)
                        .style(right_style),
                    status[2],
                );
            }

            // The bordered input box renders the textarea (block included).
            frame.render_widget(textarea, chunks[2]);

            if let Some(completion) = completion.as_ref() {
                render::render_completion(
                    chunks[3],
                    &completion.candidates,
                    completion.selected,
                    completion.offset,
                    frame,
                );
            }
        })?;
        if clear_flash {
            self.status_flash = None;
        }
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
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
        Ok(())
    }
}

fn replace_current_command_token(input: &str, value: &str) -> String {
    let Some(command_end) = input.find(char::is_whitespace) else {
        return value.to_owned();
    };
    let token_start = input[command_end..]
        .find(|character: char| !character.is_whitespace())
        .map(|offset| command_end + offset)
        .unwrap_or(input.len());
    let token_end = input[token_start..]
        .find(char::is_whitespace)
        .map(|offset| token_start + offset)
        .unwrap_or(input.len());
    format!("{}{}{}", &input[..token_start], value, &input[token_end..])
}

fn new_textarea() -> TextArea<'static> {
    textarea_with_text("")
}

fn textarea_with_text(value: &str) -> TextArea<'static> {
    textarea_with_text_at_cursor(value, usize::MAX, usize::MAX)
}

fn textarea_with_text_at_cursor(value: &str, line: usize, column: usize) -> TextArea<'static> {
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
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    );
    // Main input text is white; the bordered box already marks the active
    // line, so the default underline cursor-line styling is dropped.
    textarea.set_style(Style::default().fg(Color::White));
    textarea.set_cursor_line_style(Style::default());
    textarea
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), DisableBracketedPaste, cursor::Show);
        previous(panic);
    }));
}

#[cfg(test)]
mod tests {
    // The terminal lifecycle needs a real tty; pure key/render behavior is
    // tested in input.rs and render.rs. Keeping this module documents that no
    // alternate-screen teardown is used.
}
