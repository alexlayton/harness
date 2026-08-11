use crate::input::{InputAction, classify};
use crate::render;
use crate::{TuiEvent, UiEvent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyEventKind,
};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};
use std::io::{self, Stdout};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui_textarea::TextArea;

const MAX_LIVE_ROWS: u16 = 10;
const MAX_INPUT_ROWS: u16 = 10;
const VIEWPORT_ROWS: u16 = 22;

pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    model: String,
    provider: String,
    textarea: TextArea<'static>,
    pending_text: String,
    pending_reasoning: String,
    running_tool: Option<String>,
    spinner: usize,
    busy: bool,
    retrying: Option<u32>,
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
            textarea: TextArea::default(),
            pending_text: String::new(),
            pending_reasoning: String::new(),
            running_tool: None,
            spinner: 0,
            busy: false,
            retrying: None,
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
                    // path.  During a turn the agent sends TurnFinished and
                    // the input handler remains available for a final redraw.
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
        match classify(&event) {
            InputAction::Quit => {
                cancel.cancel();
                return Ok(true);
            }
            InputAction::Interrupt => {
                if self.busy {
                    // Esc is a turn-local interrupt. The agent receives this
                    // control message and creates a fresh token for the next
                    // queued turn; Ctrl+C/D below remain application quit.
                    let _ = input_tx.send(crate::INTERRUPT_MESSAGE.to_owned());
                }
            }
            InputAction::Newline => {
                self.textarea.insert_newline();
            }
            InputAction::Submit => {
                let input = self.textarea.lines().join("\n");
                if input.trim().is_empty() {
                    return Ok(false);
                }
                let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
                render::insert_user(&mut self.terminal, &input, width)
                    .context("echo user message")?;
                input_tx
                    .send(input)
                    .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
                self.textarea = TextArea::default();
                self.busy = true;
                self.retrying = None;
            }
            InputAction::Edit => {
                self.textarea.input(event);
            }
            InputAction::Ignore => {}
        }
        Ok(false)
    }

    fn apply_event(&mut self, event: UiEvent) -> Result<()> {
        match event {
            UiEvent::TextDelta(delta) => {
                self.busy = true;
                self.retrying = None;
                self.pending_text.push_str(&delta);
                self.flush_stable_text()?;
                self.flush_overflow_text()?;
            }
            UiEvent::ReasoningDelta(delta) => {
                self.busy = true;
                self.retrying = None;
                self.pending_reasoning.push_str(&delta);
            }
            UiEvent::ToolCallStarted { summary, .. } => {
                self.busy = true;
                self.retrying = None;
                self.running_tool = Some(summary);
                self.spinner = 0;
            }
            UiEvent::ToolCallFinished {
                summary,
                ok,
                duration_ms,
                error,
                ..
            } => {
                let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
                render::insert_tool_finished(
                    &mut self.terminal,
                    &summary,
                    ok,
                    duration_ms,
                    error.as_deref(),
                    width,
                )?;
                self.running_tool = None;
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
                self.flush_everything()?;
                self.running_tool = None;
                self.retrying = None;
                self.busy = false;
            }
        }
        Ok(())
    }

    fn flush_stable_text(&mut self) -> Result<()> {
        let (stable, pending) = render::split_stable_prefix(&self.pending_text);
        if stable.is_empty() {
            return Ok(());
        }
        self.pending_text = pending;
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_markdown(&mut self.terminal, &stable, width).context("insert markdown")?;
        Ok(())
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
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        render::insert_markdown(&mut self.terminal, &prefix, width)
            .context("insert streaming markdown")?;
        Ok(())
    }

    fn flush_everything(&mut self) -> Result<()> {
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        if !self.pending_reasoning.is_empty() {
            let reasoning = std::mem::take(&mut self.pending_reasoning);
            render::insert_reasoning(&mut self.terminal, &reasoning, width)
                .context("insert reasoning")?;
        }
        if !self.pending_text.is_empty() {
            let text = std::mem::take(&mut self.pending_text);
            render::insert_markdown(&mut self.terminal, &text, width).context("insert markdown")?;
        }
        Ok(())
    }

    fn insert_error(&mut self, error: &str) -> Result<()> {
        let width = self.terminal.size().map(|size| size.width).unwrap_or(80);
        let text = format!("✗ {error}");
        let height = render::wrap_count(&text, width as usize) as u16;
        self.terminal.insert_before(height.max(1), |buffer| {
            Paragraph::new(Line::from(Span::styled(
                text,
                Style::default().fg(Color::Red),
            )))
            .wrap(ratatui::widgets::Wrap { trim: false })
            .render(buffer.area, buffer);
        })?;
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        let model = self.model.clone();
        let provider = self.provider.clone();
        let pending_text = self.pending_text.clone();
        let pending_reasoning = self.pending_reasoning.clone();
        let running_tool = self.running_tool.clone();
        let spinner = self.spinner;
        let busy = self.busy;
        let retrying = self.retrying;
        let textarea = &self.textarea;
        self.terminal.draw(|frame| {
            let area = frame.area();
            let input_rows = (textarea.lines().len() as u16 + 1)
                .clamp(1, MAX_INPUT_ROWS)
                .min(area.height.saturating_sub(2).max(1));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Min(1),
                    Constraint::Length(input_rows),
                ])
                .split(area);

            let status_style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM);
            let status = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(model.len() as u16 + 1),
                    Constraint::Min(1),
                    Constraint::Length(provider.len() as u16 + 1),
                ])
                .split(chunks[0]);
            frame.render_widget(
                Paragraph::new(model.as_str()).style(status_style),
                status[0],
            );
            if let Some(attempt) = retrying {
                frame.render_widget(
                    Paragraph::new(format!("↻ retrying (attempt {attempt})"))
                        .alignment(Alignment::Center)
                        .style(status_style),
                    status[1],
                );
            } else if busy {
                frame.render_widget(
                    Paragraph::new("… generating")
                        .alignment(Alignment::Center)
                        .style(status_style),
                    status[1],
                );
            }
            frame.render_widget(
                Paragraph::new(provider.as_str())
                    .alignment(Alignment::Right)
                    .style(status_style),
                status[2],
            );

            render::render_live(
                chunks[1],
                &pending_reasoning,
                &pending_text,
                running_tool.as_deref().map(|summary| (summary, spinner)),
                frame,
            );

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
    // tested in input.rs and render.rs.  Keeping this module documents that
    // no alternate-screen teardown is used.
}
