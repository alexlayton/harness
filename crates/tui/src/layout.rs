//! Layout arithmetic and terminal painting. `draw` sizes the live tail,
//! completion, activity, metadata, and prompt rows, then delegates painting
//! to `render`.
//!
//! The layout arithmetic is still the temporary pre-migration version (the
//! Phase 3 rewrite lands later), but the live region now holds only the
//! uncommitted tail: `transcript[committed..]`, the streaming assistant and/or
//! running tool that the commit pipeline has not yet written into scrollback.

use crate::render::{self, Theme};
use crate::state::TranscriptEntry;
use anyhow::Result;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

impl crate::Tui {
    pub(crate) fn draw(&mut self) -> Result<()> {
        self.terminal.draw(|frame| {
            // `frame.area()` is the H-row inline viewport; ratatui already
            // auto-resized it, so layout can only ever address rows within it.
            let area = frame.area();
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
                return;
            }

            // Build the live tail from the uncommitted entries only. Each
            // entry keeps its own expansion state (Ctrl+O expands the running
            // tool while it streams). The welcome banner is no longer painted
            // here; Phase 3 commits it into scrollback at startup.
            let committed = self.committed.min(self.transcript.len());
            let live = &self.transcript[committed..];
            let mut lines = Vec::new();
            for entry in live {
                if !lines.is_empty() {
                    render::push_blank(&mut lines, render::SECTION_GAP);
                }
                let expanded = matches!(entry, TranscriptEntry::Tool { expanded: true, .. });
                lines.extend(render::entry_lines(
                    entry,
                    expanded,
                    outer.width as usize,
                    self.theme,
                ));
            }
            let prompt_content_width = outer.width.saturating_sub(4).max(1) as usize;
            let prompt_layout =
                render::prompt_layout(&self.textarea, prompt_content_width, self.theme);
            let desired_prompt_rows = prompt_layout.lines.len().saturating_add(2) as u16;
            let requested_completion_rows = self
                .completion
                .as_ref()
                .map(|completion| render::completion_rows(&completion.candidates))
                .unwrap_or(0);
            let activity_rows = u16::from(self.busy);
            let minimum_layout_rows = 1u16
                .saturating_add(3)
                .saturating_add(1)
                .saturating_add(activity_rows);
            let completion_capacity = outer.height.saturating_sub(minimum_layout_rows);
            let completion_rows = requested_completion_rows.min(completion_capacity);
            let fixed_rows = completion_rows
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

            let prompt_inner_height = prompt_rows.saturating_sub(2).max(1) as usize;
            self.prompt_scroll = render::prompt_scroll_for_cursor(
                prompt_layout.cursor_row,
                prompt_layout.lines.len(),
                prompt_inner_height,
            );

            // render_tail always shows the newest rows of the live tail.
            let constraints = vec![
                Constraint::Length(transcript_rows),
                Constraint::Length(completion_rows),
                Constraint::Length(activity_rows),
                Constraint::Length(1),
                Constraint::Length(prompt_rows),
            ];
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(outer);

            render::render_tail(chunks[0], &lines, self.theme, frame);
            if let Some(completion) = self.completion.as_ref() {
                render::render_completion(
                    chunks[1],
                    &completion.candidates,
                    completion.selected,
                    completion.offset,
                    self.theme,
                    frame,
                );
            }
            render::render_activity(
                chunks[2],
                self.activity.label(),
                self.spinner,
                self.theme,
                frame,
            );
            render_metadata(
                chunks[3],
                &self.environment.cwd_display,
                self.environment.branch.as_deref(),
                &self.provider,
                &self.model,
                self.theme,
                frame,
            );
            render::render_prompt(
                chunks[4],
                &self.textarea,
                self.prompt_scroll,
                self.theme,
                frame,
            );
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
