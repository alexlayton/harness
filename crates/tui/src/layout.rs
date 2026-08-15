//! Layout arithmetic and terminal painting. `draw` sizes the transcript,
//! indicator, completion, activity, metadata, and prompt rows, then delegates
//! painting to `render`.

use crate::render::{self, Theme};
use anyhow::{Context, Result};
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

impl crate::Tui {
    pub(crate) fn draw(&mut self) -> Result<()> {
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

        let show_welcome = !self
            .transcript
            .iter()
            .any(crate::state::TranscriptEntry::is_meaningful);
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
        let activity_rows = u16::from(self.busy);
        let minimum_layout_rows = 1u16
            .saturating_add(3)
            .saturating_add(1)
            .saturating_add(activity_rows);
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
            .saturating_add(1)
            .saturating_add(activity_rows);
        let available = outer.height.saturating_sub(fixed_rows);
        let max_prompt = ((outer.height as usize * crate::app::MAX_INPUT_FRACTION) / 100)
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
        self.prompt_scroll = render::prompt_scroll_for_cursor(
            prompt_layout.cursor_row,
            prompt_layout.lines.len(),
            prompt_inner_height,
        );

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
        let activity = self.activity;
        let spinner = self.spinner;
        let new_content = self.scroll.new_content_below;

        self.terminal.draw(|frame| {
            let constraints = vec![
                Constraint::Length(transcript_rows),
                Constraint::Length(indicator_rows),
                Constraint::Length(completion_rows),
                Constraint::Length(activity_rows),
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
            render::render_activity(chunks[3], activity.label(), spinner, theme, frame);
            render_metadata(
                chunks[4],
                &cwd,
                branch.as_deref(),
                &provider,
                &model,
                theme,
                frame,
            );
            render::render_prompt(chunks[5], textarea, prompt_scroll, theme, frame);
        })?;
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
    // Fill any remaining middle space with the default background so the
    // metadata line does not inherit leftover characters from the terminal.
    let middle_width = layout[1].width as usize;
    if middle_width > 0 {
        frame.render_widget(
            Paragraph::new(" ".repeat(middle_width)).style(style),
            layout[1],
        );
    }
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
