use crate::attachments;
use crate::commands::{self, Candidate, CandidateKind, ParsedCommand};
use crate::environment::EnvironmentInfo;
use crate::input::{InputAction, classify, history_next, history_previous, push_history};
use crate::render::{self, Theme};
use crate::state::{EntryId, Focus, ScrollState, ToolRecord, ToolStatus, TranscriptEntry};
use crate::{InputMessage, ModelEntry, SessionSnapshotEntry, TuiEvent, UiEvent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
    KeyModifiers,
};
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, Paragraph};
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui_textarea::{CursorMove, TextArea};
use unicode_width::UnicodeWidthStr;

const MAX_INPUT_FRACTION: usize = 30;
const MAX_HISTORY: usize = 1_000;
const PLACEHOLDER: &str = "Type your message...";

#[derive(Clone, Debug)]
struct Completion {
    candidates: Vec<Candidate>,
    selected: usize,
    offset: usize,
    kind: CompletionKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionKind {
    Slash,
    Files,
}

#[derive(Debug)]
struct FileCompletionResult {
    generation: u64,
    query: String,
    candidates: Vec<Candidate>,
}

pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    theme: Theme,
    model: String,
    provider: String,
    providers: Vec<String>,
    model_lists: HashMap<String, Vec<ModelEntry>>,
    workspace_root: PathBuf,
    environment: EnvironmentInfo,

    completion: Option<Completion>,
    file_completion_tx: mpsc::UnboundedSender<FileCompletionResult>,
    file_completion_rx: mpsc::UnboundedReceiver<FileCompletionResult>,
    file_completion_generation: u64,
    file_completion_query: Option<String>,
    file_completion_cancel: Option<CancellationToken>,

    textarea: TextArea<'static>,
    prompt_scroll: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
    draft: String,

    transcript: Vec<TranscriptEntry>,
    next_entry_id: EntryId,
    streaming_assistant: Option<EntryId>,
    running_tool: Option<EntryId>,
    focused_tool: Option<usize>,
    scroll: ScrollState,
    transcript_dirty: bool,
    focus: Focus,

    spinner: usize,
    busy: bool,
    retrying: Option<u32>,
    restored: bool,
    session_id: Option<String>,
    session_title: Option<String>,
}

impl Tui {
    pub fn new(model: &str, provider: &str, providers: Vec<String>) -> Result<Self> {
        install_panic_hook();
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let environment = EnvironmentInfo::discover(workspace_root.clone());

        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            cursor::Hide
        ) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("configure terminal input");
        }

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = terminal::disable_raw_mode();
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
                return Err(error).context("create fullscreen terminal");
            }
        };
        if let Err(error) = terminal.clear() {
            let _ = terminal::disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
            return Err(error).context("clear fullscreen terminal");
        }

        let (file_completion_tx, file_completion_rx) = mpsc::unbounded_channel();
        Ok(Self {
            terminal,
            theme: Theme::default(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            providers,
            model_lists: HashMap::new(),
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
            next_entry_id: 1,
            streaming_assistant: None,
            running_tool: None,
            focused_tool: None,
            scroll: ScrollState::default(),
            transcript_dirty: true,
            focus: Focus::Prompt,
            spinner: 0,
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
                    if self.running_tool.is_some() {
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
            InputAction::PageUp => self.scroll_transcript(-(self.scroll.page_size() as isize)),
            InputAction::PageDown => self.scroll_transcript(self.scroll.page_size() as isize),
            InputAction::Bottom => self.scroll_to_bottom(),
            InputAction::Newline => {
                self.history_pos = None;
                self.draft.clear();
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
                self.submit_message(input, input_tx)?;
            }
            InputAction::ExpandDetails => {
                self.toggle_selected_or_latest_tool();
            }
            InputAction::FocusTools => {
                if !self.busy && !self.tool_indices().is_empty() {
                    let indices = self.tool_indices();
                    self.focused_tool = Some(indices[0]);
                    self.focus = Focus::Tool;
                } else {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event);
                    self.refresh_completion();
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
            }
            InputAction::Ignore => {}
        }
        Ok(false)
    }

    fn handle_navigation_edit(&mut self, event: &Event) -> bool {
        let Event::Key(key) = event else {
            return false;
        };
        match key.code {
            KeyCode::Up => {
                let (line, _) = self.textarea.cursor();
                if line > 0 {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event.clone());
                    return true;
                }
                if self.navigate_history_up() {
                    return true;
                }
                self.scroll_transcript(-1);
                true
            }
            KeyCode::Down => {
                let (line, _) = self.textarea.cursor();
                if line + 1 < self.textarea.lines().len() {
                    self.history_pos = None;
                    self.draft.clear();
                    self.textarea.input(event.clone());
                    return true;
                }
                if self.navigate_history_down() {
                    return true;
                }
                self.scroll_transcript(1);
                true
            }
            KeyCode::Char('k') if key.modifiers.is_empty() && self.textarea.is_empty() => {
                self.scroll_transcript(-1);
                true
            }
            KeyCode::Char('j') if key.modifiers.is_empty() && self.textarea.is_empty() => {
                self.scroll_transcript(1);
                true
            }
            _ => false,
        }
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
            KeyCode::Esc => {
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
                (completion.kind == CompletionKind::Slash).then(|| {
                    completion
                        .candidates
                        .get(completion.selected)
                        .map(|candidate| candidate.value.as_str())
                })
            });
            let selected = old_value
                .flatten()
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

        self.add_notice(format!("⌘ {input}"));
        self.push_history_and_clear_input(input);
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
                        .send(InputMessage::ListModels { provider })
                        .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                }
            }
        }
        Ok(())
    }

    fn handle_tool_focus(&mut self, event: &Event) -> Result<bool> {
        let Event::Key(key) = event else {
            self.focused_tool = None;
            self.focus = Focus::Prompt;
            return Ok(false);
        };
        let indices = self.tool_indices();
        if indices.is_empty() {
            self.focused_tool = None;
            self.focus = Focus::Prompt;
            return Ok(false);
        }
        let current_position = self
            .focused_tool
            .and_then(|index| indices.iter().position(|candidate| *candidate == index))
            .unwrap_or(0);
        match key.code {
            KeyCode::Esc => {
                self.focused_tool = None;
                self.focus = Focus::Prompt;
                Ok(true)
            }
            KeyCode::Up => {
                let position = current_position.saturating_sub(1);
                self.focused_tool = Some(indices[position]);
                Ok(true)
            }
            KeyCode::Down => {
                let position = (current_position + 1).min(indices.len().saturating_sub(1));
                self.focused_tool = Some(indices[position]);
                Ok(true)
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.focused_tool = Some(indices[current_position]);
                self.toggle_tool_at(indices[current_position]);
                Ok(true)
            }
            KeyCode::Char('o')
                if key.modifiers.contains(KeyModifiers::CONTROL) || key.modifiers.is_empty() =>
            {
                self.focused_tool = Some(indices[current_position]);
                self.toggle_tool_at(indices[current_position]);
                Ok(true)
            }
            KeyCode::Tab => {
                self.focused_tool = Some(indices[(current_position + 1) % indices.len()]);
                Ok(true)
            }
            _ => {
                self.focused_tool = None;
                self.focus = Focus::Prompt;
                Ok(false)
            }
        }
    }

    fn navigate_history_up(&mut self) -> bool {
        if commands::is_command_input(&self.textarea.lines().join("\n")) {
            return false;
        }
        let (line, _) = self.textarea.cursor();
        if self.textarea.lines().len() > 1 || line != 0 || self.history_pos == Some(0) {
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
        if self.textarea.lines().len() > 1 || line + 1 < self.textarea.lines().len() {
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

    fn push_history_and_clear_input(&mut self, input: &str) {
        push_history(&mut self.history, input, MAX_HISTORY);
        self.history_pos = None;
        self.draft.clear();
        self.textarea = new_textarea();
        self.close_completion();
        self.focus = Focus::Prompt;
    }

    fn submit_message(
        &mut self,
        input: String,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        let id = self.allocate_id();
        self.add_entry(TranscriptEntry::User {
            id,
            text: input.clone(),
        });
        self.push_history_and_clear_input(&input);
        input_tx
            .send(InputMessage::Message(input))
            .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        self.busy = true;
        self.retrying = None;
        Ok(())
    }

    fn apply_event(&mut self, event: UiEvent) -> Result<()> {
        match event {
            UiEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                self.busy = true;
                self.retrying = None;
                if let TranscriptEntry::Assistant { markdown, .. } = self.ensure_assistant() {
                    markdown.push_str(&delta);
                }
                self.transcript_changed();
            }
            UiEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                self.busy = true;
                self.retrying = None;
                if let TranscriptEntry::Assistant { reasoning, .. } = self.ensure_assistant() {
                    reasoning.push_str(&delta);
                }
                self.transcript_changed();
            }
            UiEvent::ToolCallStarted {
                name,
                summary,
                arguments,
            } => {
                self.busy = true;
                self.retrying = None;
                self.streaming_assistant = None;
                let id = self.allocate_id();
                self.add_entry(TranscriptEntry::Tool {
                    id,
                    record: ToolRecord {
                        name,
                        args: arguments,
                        summary,
                        ok: false,
                        duration_ms: 0,
                        output: String::new(),
                        error: None,
                        status: ToolStatus::Running,
                    },
                    expanded: false,
                });
                self.running_tool = Some(id);
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
                let id = self
                    .running_tool
                    .take()
                    .unwrap_or_else(|| self.allocate_id());
                let mut updated = false;
                if let Some(entry) = self.transcript.iter_mut().find(|entry| entry.id() == id)
                    && let Some(record) = entry.tool_record_mut()
                {
                    record.name = name.clone();
                    record.summary = summary.clone();
                    record.ok = ok;
                    record.duration_ms = duration_ms;
                    record.output = output.clone();
                    record.error = error.clone();
                    record.status = if ok {
                        ToolStatus::Success
                    } else {
                        ToolStatus::Failure
                    };
                    updated = true;
                }
                if !updated {
                    self.add_entry(TranscriptEntry::Tool {
                        id,
                        record: ToolRecord {
                            name,
                            args: "{}".into(),
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
                        expanded: false,
                    });
                } else {
                    self.transcript_changed();
                }
                self.focused_tool = None;
                self.retrying = None;
            }
            UiEvent::Retrying { attempt, .. } => {
                self.busy = true;
                self.retrying = Some(attempt);
            }
            UiEvent::Error(error) => {
                self.add_error(error);
                self.running_tool = None;
                self.streaming_assistant = None;
                self.busy = false;
                self.retrying = None;
            }
            UiEvent::TurnFinished => {
                if let Some(id) = self.streaming_assistant.take()
                    && let Some(entry) = self.transcript.iter_mut().find(|entry| entry.id() == id)
                    && let TranscriptEntry::Assistant { streaming, .. } = entry
                {
                    *streaming = false;
                }
                self.running_tool = None;
                self.retrying = None;
                self.busy = false;
                self.focused_tool = None;
                self.focus = Focus::Prompt;
            }
            UiEvent::Notice(notice) => self.add_notice(notice),
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
                self.transcript.clear();
                self.streaming_assistant = None;
                self.running_tool = None;
                self.focused_tool = None;
                self.scroll = ScrollState::default();
                self.transcript_dirty = true;
                if loaded {
                    self.add_notice(format!("loaded session {}", &id[..id.len().min(8)]));
                }
            }
            UiEvent::SessionSnapshot { entries } => {
                self.transcript.clear();
                self.streaming_assistant = None;
                self.running_tool = None;
                self.focused_tool = None;
                for snapshot in entries {
                    let id = self.allocate_id();
                    let entry = match snapshot {
                        SessionSnapshotEntry::User { text } => TranscriptEntry::User { id, text },
                        SessionSnapshotEntry::Assistant {
                            markdown,
                            reasoning,
                        } => TranscriptEntry::Assistant {
                            id,
                            markdown,
                            reasoning,
                            streaming: false,
                        },
                        SessionSnapshotEntry::Tool {
                            name,
                            summary,
                            arguments,
                            ok,
                            duration_ms,
                            output,
                            error,
                        } => TranscriptEntry::Tool {
                            id,
                            record: ToolRecord {
                                name,
                                args: arguments,
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
                            expanded: false,
                        },
                    };
                    self.transcript.push(entry);
                }
                self.transcript_changed();
            }
            UiEvent::SessionList { sessions } => {
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
                self.add_notice(notice);
            }
            UiEvent::SessionExported { path } => {
                self.add_notice(format!("exported session to {path}"));
            }
            UiEvent::UsageUpdated { .. } => {}
            UiEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
            } => self.add_notice(format!(
                "compacted through event {compacted_through} ({summary_bytes} bytes)"
            )),
        }
        Ok(())
    }

    fn ensure_assistant(&mut self) -> &mut TranscriptEntry {
        let id = if let Some(id) = self.streaming_assistant {
            id
        } else {
            let id = self.allocate_id();
            self.add_entry(TranscriptEntry::Assistant {
                id,
                markdown: String::new(),
                reasoning: String::new(),
                streaming: true,
            });
            self.streaming_assistant = Some(id);
            id
        };
        self.transcript
            .iter_mut()
            .find(|entry| entry.id() == id)
            .expect("streaming assistant entry exists")
    }

    fn add_notice(&mut self, text: impl Into<String>) {
        let id = self.allocate_id();
        self.add_entry(TranscriptEntry::Notice {
            id,
            text: text.into(),
        });
    }

    fn add_error(&mut self, text: impl Into<String>) {
        let id = self.allocate_id();
        self.add_entry(TranscriptEntry::Error {
            id,
            text: text.into(),
        });
    }

    fn add_entry(&mut self, entry: TranscriptEntry) {
        self.transcript.push(entry);
        self.transcript_changed();
    }

    fn allocate_id(&mut self) -> EntryId {
        let id = self.next_entry_id;
        self.next_entry_id = self.next_entry_id.wrapping_add(1).max(1);
        id
    }

    fn transcript_changed(&mut self) {
        if !self.scroll.follow_latest && !self.scroll.at_bottom() {
            self.scroll.new_content_below = true;
        }
        self.transcript_dirty = true;
    }

    fn tool_indices(&self) -> Vec<usize> {
        self.transcript
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.tool_record().map(|_| index))
            .collect()
    }

    fn toggle_selected_or_latest_tool(&mut self) {
        let index = self
            .focused_tool
            .or_else(|| self.tool_indices().last().copied());
        if let Some(index) = index {
            self.toggle_tool_at(index);
        }
    }

    fn toggle_tool_at(&mut self, index: usize) {
        if let Some(TranscriptEntry::Tool { expanded, .. }) = self.transcript.get_mut(index) {
            *expanded = !*expanded;
            self.transcript_changed();
        }
    }

    fn scroll_transcript(&mut self, delta: isize) {
        self.scroll.scroll_by(delta);
        self.focus = Focus::Transcript;
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll.go_bottom();
        self.focus = Focus::Transcript;
    }

    fn handle_resize(&mut self) {
        self.prompt_scroll = 0;
        self.transcript_dirty = true;
    }

    fn draw(&mut self) -> Result<()> {
        let size = self.terminal.size().context("read terminal size")?;
        let area = Rect::new(0, 0, size.width, size.height);
        let horizontal = if area.width >= 80 {
            2
        } else if area.width >= 40 {
            1
        } else {
            0
        };
        let vertical = if area.height >= 8 { 1 } else { 0 };
        let outer = area.inner(Margin {
            horizontal,
            vertical,
        });
        if outer.width == 0 || outer.height == 0 {
            return Ok(());
        }

        let show_welcome = !self.transcript.iter().any(TranscriptEntry::is_meaningful);
        let transcript_lines = render::transcript_lines(
            &self.transcript,
            show_welcome,
            outer.width as usize,
            self.theme,
        );
        let content_height = transcript_lines.len();
        let prompt_content_width = outer.width.saturating_sub(4).max(1) as usize;
        let prompt_layout = render::prompt_layout(&self.textarea, prompt_content_width, self.theme);
        let desired_prompt_rows = prompt_layout.lines.len().saturating_add(2) as u16;
        let requested_completion_rows = self
            .completion
            .as_ref()
            .map(|completion| render::completion_rows(&completion.candidates))
            .unwrap_or(0);
        let requested_indicator_rows = u16::from(self.scroll.new_content_below);
        let minimum_layout_rows = 1u16.saturating_add(3).saturating_add(1);
        let indicator_rows = if outer.height >= minimum_layout_rows.saturating_add(1) {
            requested_indicator_rows
        } else {
            0
        };
        let completion_capacity = outer
            .height
            .saturating_sub(indicator_rows)
            .saturating_sub(minimum_layout_rows);
        let completion_rows = requested_completion_rows.min(completion_capacity);
        let fixed_rows = indicator_rows
            .saturating_add(completion_rows)
            .saturating_add(1);
        let available = outer.height.saturating_sub(fixed_rows);
        let max_prompt = ((outer.height as usize * MAX_INPUT_FRACTION) / 100)
            .max(3)
            .min(u16::MAX as usize) as u16;
        let prompt_rows = desired_prompt_rows
            .max(3)
            .min(max_prompt)
            .min(available.saturating_sub(1).max(1));
        let transcript_rows = available.saturating_sub(prompt_rows).max(1);

        let was_following = self.scroll.follow_latest || self.scroll.at_bottom();
        self.scroll.content_height = content_height;
        self.scroll.viewport_height = transcript_rows as usize;
        if self.transcript_dirty {
            self.scroll.on_content_changed(was_following);
            self.transcript_dirty = false;
        } else {
            self.scroll.clamp();
        }
        if show_welcome {
            self.scroll.offset = 0;
        }

        let prompt_inner_height = prompt_rows.saturating_sub(2).max(1) as usize;
        let prompt_max_scroll = prompt_layout
            .lines
            .len()
            .saturating_sub(prompt_inner_height);
        if prompt_layout.cursor_row < self.prompt_scroll {
            self.prompt_scroll = prompt_layout.cursor_row;
        } else if prompt_layout.cursor_row >= self.prompt_scroll + prompt_inner_height {
            self.prompt_scroll = prompt_layout
                .cursor_row
                .saturating_add(1)
                .saturating_sub(prompt_inner_height);
        }
        self.prompt_scroll = self.prompt_scroll.min(prompt_max_scroll);

        let transcript = &transcript_lines;
        let offset = self.scroll.offset;
        let completion = self.completion.clone();
        let provider = self.provider.clone();
        let model = self.model.clone();
        let cwd = self.environment.cwd_display.clone();
        let branch = self.environment.branch.clone();
        let textarea = &self.textarea;
        let prompt_scroll = self.prompt_scroll;
        let theme = self.theme;
        let new_content = self.scroll.new_content_below;

        self.terminal.draw(|frame| {
            let full = frame.area();
            frame.render_widget(
                Block::default().style(Style::default().bg(theme.background)),
                full,
            );
            let constraints = vec![
                Constraint::Length(transcript_rows),
                Constraint::Length(indicator_rows),
                Constraint::Length(completion_rows),
                Constraint::Length(1),
                Constraint::Length(prompt_rows),
            ];
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(outer);

            render::render_transcript_lines(chunks[0], transcript, offset, theme, frame);
            if new_content {
                render::render_new_content_indicator(chunks[1], theme, frame);
            }
            if let Some(completion) = completion.as_ref() {
                render::render_completion(
                    chunks[2],
                    &completion.candidates,
                    completion.selected,
                    completion.offset,
                    theme,
                    frame,
                );
            }
            render_metadata(
                chunks[3],
                &cwd,
                branch.as_deref(),
                &provider,
                &model,
                theme,
                frame,
            );
            render::render_prompt(chunks[4], textarea, prompt_scroll, theme, frame);
        })?;
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
            cursor::Show,
            LeaveAlternateScreen
        )
        .context("restore terminal input")?;
        Ok(())
    }
}

fn render_metadata(
    area: Rect,
    cwd: &str,
    branch: Option<&str>,
    provider: &str,
    model: &str,
    theme: Theme,
    frame: &mut ratatui::Frame<'_>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let left = match branch {
        Some(branch) => format!("{cwd}  ({branch})"),
        None => cwd.to_owned(),
    };
    let right = format!("{provider} · {model}");
    let left_width = UnicodeWidthStr::width(left.as_str()).min(area.width as usize / 2);
    let right_width = UnicodeWidthStr::width(right.as_str()).min(area.width as usize / 2);
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(left_width as u16),
            Constraint::Min(1),
            Constraint::Length(right_width as u16),
        ])
        .split(area);
    let style = Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM);
    frame.render_widget(
        Paragraph::new(truncate_display(&left, left_width)).style(style),
        layout[0],
    );
    frame.render_widget(
        Paragraph::new(truncate_display(&right, right_width))
            .alignment(ratatui::layout::Alignment::Right)
            .style(style),
        layout[2],
    );
}

fn truncate_display(value: &str, width: usize) -> String {
    if UnicodeWidthStr::width(value) <= width {
        return value.to_owned();
    }
    if width <= 1 {
        return "…".to_owned();
    }
    let mut result = String::new();
    let mut used = 0usize;
    for character in value.chars() {
        let character_width = unicode_width::UnicodeWidthChar::width(character)
            .unwrap_or(1)
            .max(1);
        if used + character_width + 1 > width {
            break;
        }
        result.push(character);
        used += character_width;
    }
    result.push('…');
    result
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
            .fg(Theme::default().dim_text)
            .add_modifier(Modifier::DIM),
    );
    textarea.set_style(Style::default().fg(Theme::default().primary_text));
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
        let _ = execute!(
            io::stdout(),
            DisableBracketedPaste,
            cursor::Show,
            LeaveAlternateScreen
        );
        previous(panic);
    }));
}

#[cfg(test)]
mod tests {
    #[test]
    fn fullscreen_ui_keeps_terminal_lifecycle_outside_pure_render_tests() {
        // The real terminal lifecycle is intentionally exercised manually;
        // render.rs, input.rs, state.rs, and environment.rs contain the pure
        // behavior tests that do not require a TTY.
    }
}
