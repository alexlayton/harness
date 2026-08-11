use crate::input::{InputAction, classify, history_next, history_previous, push_history};
use crate::render;
use crate::render::{TailTool, ToolRecord};
use crate::{TuiEvent, UiEvent};
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
use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui_textarea::TextArea;

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

pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    model: String,
    provider: String,
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
    needs_model_label: bool,
    header_printed: bool,
    history: Vec<String>,
    history_pos: Option<usize>,
    draft: String,
    restored: bool,
}

impl Tui {
    pub fn new(model: &str, provider: &str) -> Result<Self> {
        install_panic_hook();
        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnableBracketedPaste, cursor::Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("configure terminal input");
        }
        let (_, rows) = terminal::size().unwrap_or((80, 24));
        let viewport_rows = rows.clamp(4, VIEWPORT_ROWS);
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
            needs_model_label: false,
            header_printed: false,
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            restored: false,
        })
    }

    pub async fn run<E>(
        mut self,
        mut events: mpsc::UnboundedReceiver<E>,
        input_tx: mpsc::UnboundedSender<String>,
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
        input_tx: mpsc::UnboundedSender<String>,
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
        input_tx: &mpsc::UnboundedSender<String>,
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
                    let _ = input_tx.send(crate::INTERRUPT_MESSAGE.to_owned());
                }
            }
            InputAction::Newline => {
                self.history_pos = None;
                self.textarea.input(event);
            }
            InputAction::Submit => {
                let input = self.textarea.lines().join("\n");
                if input.trim().is_empty() {
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
                    .send(input)
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                self.textarea = new_textarea();
                self.needs_model_label = true;
                self.busy = true;
                self.retrying = None;
                self.status_flash = None;
            }
            InputAction::ExpandDetails => {
                if !self.busy && !self.tail.is_empty() {
                    self.focused_tool = Some(
                        self.focused_tool
                            .unwrap_or(0)
                            .min(self.tail.len().saturating_sub(1)),
                    );
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
            }
            InputAction::Ignore => {}
        }
        Ok(false)
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
                if let Some(index) = self.focused_tool
                    && let Some(entry) = self.tail.get_mut(index)
                {
                    entry.expanded = !entry.expanded;
                }
                Ok(true)
            }
            KeyCode::Char(value)
                if value.eq_ignore_ascii_case(&'o')
                    && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                // Ctrl+O while focused is the fallback dump action. Enter or
                // Space remains the in-place expand/collapse action.
                self.dump_focused_tool()?;
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

    fn navigate_history_up(&mut self) -> bool {
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
                self.ensure_model_label()?;
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
                self.ensure_model_label()?;
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
                self.needs_model_label = true;
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
        }
        Ok(())
    }

    fn ensure_model_label(&mut self) -> Result<()> {
        if !self.needs_model_label {
            return Ok(());
        }
        self.commit_tail()?;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_model_label(&mut self.terminal, width).context("insert model label")?;
        self.needs_model_label = false;
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
        self.ensure_model_label()?;
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
        self.ensure_model_label()?;
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
        self.needs_model_label = true;
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
        let textarea = &self.textarea;
        let clear_flash = self.status_flash.is_some();

        self.terminal.draw(|frame| {
            let area = frame.area();
            let input_rows = (textarea.lines().len() as u16 + 1)
                .clamp(1, MAX_INPUT_ROWS)
                .min(area.height.saturating_sub(2).max(1));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),
                    Constraint::Length(1),
                    Constraint::Length(input_rows),
                ])
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

            let left = format!("{provider} · {model}");
            let right = if let Some(flash) = status_flash.as_deref() {
                flash.to_owned()
            } else if let Some(attempt) = retrying {
                format!("↻ retrying (attempt {attempt})")
            } else if busy {
                "… generating".to_owned()
            } else {
                String::new()
            };
            let status_style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM);
            let status = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(left.chars().count().min(u16::MAX as usize) as u16),
                    Constraint::Min(1),
                    Constraint::Length(right.chars().count().min(u16::MAX as usize) as u16),
                ])
                .split(chunks[1]);
            frame.render_widget(Paragraph::new(left).style(status_style), status[0]);
            if !right.is_empty() {
                frame.render_widget(
                    Paragraph::new(right)
                        .alignment(ratatui::layout::Alignment::Right)
                        .style(status_style),
                    status[2],
                );
            }

            let input = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(2), Constraint::Min(1)])
                .split(chunks[2]);
            frame.render_widget(
                Paragraph::new("> ").style(Style::default().add_modifier(Modifier::BOLD)),
                input[0],
            );
            frame.render_widget(textarea, input[1]);
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

fn new_textarea() -> TextArea<'static> {
    textarea_with_text("")
}

fn textarea_with_text(value: &str) -> TextArea<'static> {
    let mut textarea = TextArea::new(value.split('\n').map(str::to_owned).collect());
    textarea.set_placeholder_text(PLACEHOLDER);
    textarea.set_placeholder_style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    );
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
