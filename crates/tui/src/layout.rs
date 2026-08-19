//! Layout arithmetic and terminal painting. `draw` sizes the live tail,
//! activity, completion, metadata, and input rows inside the fixed H-row
//! inline viewport, then delegates painting to `render`.
//!
//! The canvas is a fixed budget: the input box is anchored at the bottom and
//! everything above it (metadata, activity, completion, then the live tail)
//! takes whatever rows are left, degrading gracefully on tiny terminals. The
//! live tail is always the newest rows of `transcript[committed..]` — it grows
//! downward while streaming and the bottom stays visible with no user
//! scrolling; committed history lives in the terminal's native scrollback.

use crate::render::{self, Theme};
use crate::state::TranscriptEntry;
use anyhow::Result;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

/// The row budget for one live-viewport frame: one rect per section, top to
/// bottom. Pure and exhaustive so the fixed-canvas arithmetic can be unit
/// tested without a TTY (see the layout tests below).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveLayout {
    pub(crate) prompt_rows: u16,
    pub(crate) completion_rows: u16,
    pub(crate) activity_rows: u16,
    pub(crate) metadata_rows: u16,
    /// Blank row between the live tail and the first section below it, so
    /// streaming text never touches the activity line or the input box.
    pub(crate) gap_rows: u16,
    pub(crate) tail_rows: u16,
}

/// Compute the section heights for a `height`-row canvas.
///
/// Budgeting rules:
/// - The input box keeps a 3-row minimum (when the canvas allows it) and caps
///   at `max(3, height - 2)` (the `−2` reserves the metadata and activity
///   rows); while busy the tail needs at least one row, so the cap is trimmed
///   to `min(cap, height - 4)`. On a degenerate 1–2 row canvas the input box
///   degrades to the whole canvas rather than panicking `clamp`.
/// - Metadata collapses on tiny canvases (<6 rows), activity only renders
///   while busy and when the canvas leaves room (<7 rows hides it), so the
///   input always keeps its minimum.
/// - Completion (while typing, i.e. idle) takes rows before the live tail;
///   the tail absorbs whatever is left.
/// - One blank `gap_rows` separates the tail from the sections below it, but
///   only when it would not eliminate the tail's last visible row.
/// - The section heights always sum to exactly `height`, so `Layout::split`
///   yields contiguous, in-bounds, non-overlapping rects.
pub(crate) fn live_layout(
    height: u16,
    busy: bool,
    desired_prompt_lines: usize,
    requested_completion_rows: u16,
) -> LiveLayout {
    let input_cap = 3u16.max(height.saturating_sub(2));
    let input_cap = if busy {
        input_cap.min(height.saturating_sub(4)).max(3)
    } else {
        input_cap
    };
    let desired_prompt_rows = desired_prompt_lines.saturating_add(2) as u16;
    // `clamp` panics when the lower bound exceeds the upper bound, and the
    // 3-row input minimum can exceed a degenerate canvas after ratatui clamps
    // the viewport to a 1–2 row terminal. Degrade the minimum instead.
    let upper = input_cap.min(height);
    let prompt_rows = desired_prompt_rows.clamp(3.min(upper), upper);

    let available = height.saturating_sub(prompt_rows);
    // On tiny canvases the metadata/activity rows collapse entirely so the
    // input keeps its 3-row minimum and the slack rows stay blank.
    let metadata = if height >= 6 && available >= 2 { 1 } else { 0 };
    let after_metadata = available - metadata;
    let activity = if busy && height >= 7 && after_metadata >= 2 {
        1
    } else {
        0
    };
    let after_activity = after_metadata - activity;
    let completion_rows = requested_completion_rows.min(after_activity);
    let mut tail_rows = after_activity - completion_rows;
    // A blank row between the live tail and whatever sits below it (activity
    // line, completion popup, metadata, or the input box) — never spent when
    // it would hide the tail's last visible row.
    let has_below = activity > 0 || completion_rows > 0 || metadata > 0;
    let gap_rows = u16::from(tail_rows > 1 && has_below);
    tail_rows -= gap_rows;

    LiveLayout {
        prompt_rows,
        completion_rows,
        activity_rows: activity,
        metadata_rows: metadata,
        gap_rows,
        tail_rows,
    }
}

impl crate::Tui {
    pub(crate) fn draw(&mut self) -> Result<()> {
        self.terminal.draw(|frame| {
            // `frame.area()` is the H-row inline viewport; ratatui already
            // auto-resized it, so layout can only ever address rows within it.
            let area = frame.area();
            let horizontal = render::horizontal_pad(area.width);
            let vertical = render::vertical_pad(area.height);
            let outer = area.inner(Margin {
                horizontal,
                vertical,
            });
            if outer.width == 0 || outer.height == 0 {
                return;
            }
            let canvas = outer.height;

            // Build the live tail from the uncommitted entries only. Each
            // entry keeps its own expansion state (Ctrl+O expands the running
            // tool while it streams). The welcome banner is committed into
            // scrollback at startup, never painted here.
            let committed = self.committed.min(self.transcript.len());
            let live = &self.transcript[committed..];
            let mut tail_lines = Vec::new();
            for entry in live {
                if !tail_lines.is_empty() {
                    render::push_blank(&mut tail_lines, render::SECTION_GAP);
                }
                let expanded = matches!(entry, TranscriptEntry::Tool { expanded: true, .. });
                tail_lines.extend(render::entry_lines(
                    entry,
                    expanded,
                    outer.width as usize,
                    self.theme,
                ));
            }

            let prompt_content_width = outer.width.saturating_sub(4).max(1) as usize;
            let prompt_layout =
                render::prompt_layout(&self.textarea, prompt_content_width, self.theme);
            let requested_completion_rows = self
                .completion
                .as_ref()
                .map(|completion| render::completion_rows(&completion.candidates))
                .unwrap_or(0);
            let budget = live_layout(
                canvas,
                self.busy,
                prompt_layout.lines.len(),
                requested_completion_rows,
            );

            let prompt_inner_height = budget.prompt_rows.saturating_sub(2).max(1) as usize;
            self.prompt_scroll = render::prompt_scroll_for_cursor(
                prompt_layout.cursor_row,
                prompt_layout.lines.len(),
                prompt_inner_height,
            );
            let hidden_above = self.prompt_scroll;
            // Rows hidden below the visible window inside the (possibly
            // scrollable) input box; shown as `+N` on the bottom border.
            let hidden_below = render::prompt_hidden_below(
                prompt_layout.lines.len(),
                self.prompt_scroll,
                prompt_inner_height,
            );

            let constraints = vec![
                Constraint::Length(budget.tail_rows),
                Constraint::Length(budget.gap_rows),
                Constraint::Length(budget.activity_rows),
                Constraint::Length(budget.completion_rows),
                Constraint::Length(budget.metadata_rows),
                Constraint::Length(budget.prompt_rows),
            ];
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(outer);

            // render_tail always shows the newest rows of the live tail.
            // chunks[1] is the blank gap row and stays unrendered.
            if budget.tail_rows > 0 {
                render::render_tail(chunks[0], &tail_lines, self.theme, frame);
            }
            if budget.activity_rows > 0 {
                render::render_activity(
                    chunks[2],
                    self.activity.label(),
                    self.spinner,
                    self.theme,
                    frame,
                );
            }
            if let Some(completion) = self.completion.as_ref()
                && budget.completion_rows > 0
            {
                render::render_completion(
                    chunks[3],
                    &completion.candidates,
                    completion.selected,
                    completion.offset,
                    self.theme,
                    frame,
                );
            }
            if budget.metadata_rows > 0 {
                render_metadata(
                    chunks[4],
                    &self.environment.cwd_display,
                    self.environment.branch.as_deref(),
                    &self.provider,
                    &self.model,
                    self.theme,
                    frame,
                );
            }
            render::render_prompt(
                chunks[5],
                &self.textarea,
                self.prompt_scroll,
                hidden_above,
                hidden_below,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::TerminalOptions;
    use ratatui::backend::TestBackend;

    fn split_constraints(budget: LiveLayout) -> Vec<Constraint> {
        vec![
            Constraint::Length(budget.tail_rows),
            Constraint::Length(budget.gap_rows),
            Constraint::Length(budget.activity_rows),
            Constraint::Length(budget.completion_rows),
            Constraint::Length(budget.metadata_rows),
            Constraint::Length(budget.prompt_rows),
        ]
    }

    #[test]
    fn live_layout_survives_degenerate_canvases() {
        // A 1–2 row canvas (terminal shrunk below the fixed viewport height)
        // must not panic `clamp` and must still partition the canvas exactly.
        for height in [1u16, 2, 3, 4] {
            for busy in [false, true] {
                let budget = live_layout(height, busy, 1, 0);
                let total = budget.prompt_rows
                    + budget.completion_rows
                    + budget.activity_rows
                    + budget.metadata_rows
                    + budget.gap_rows
                    + budget.tail_rows;
                assert_eq!(total, height, "height {height} busy {busy}");
                // Everything except the (border-only) input box collapses.
                assert_eq!(budget.metadata_rows, 0);
                assert_eq!(budget.activity_rows, 0);
                assert_eq!(budget.completion_rows, 0);
                assert!(budget.prompt_rows >= 3.min(height));
                assert!(budget.prompt_rows <= height);
            }
        }
    }

    #[test]
    fn inline_viewport_reflows_the_live_region_after_resize() {
        // ratatui's `autoresize` runs inside every `draw()`: when the terminal
        // size changes it re-anchors the inline viewport at the new width and
        // clamps its height to the terminal. Our layout is computed purely
        // from `frame.area()`, so the live region reflows with no cached
        // widths anywhere. This pins that contract.
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: ratatui::Viewport::Inline(7),
            },
        )
        .unwrap();
        terminal
            .draw(|frame| {
                assert_eq!(frame.area().width, 80);
                assert_eq!(frame.area().height, 7);
            })
            .unwrap();

        // Shrinking the terminal re-anchors the viewport at the new width;
        // the fixed height is untouched while it still fits.
        terminal.backend_mut().resize(50, 20);
        terminal
            .draw(|frame| {
                assert_eq!(frame.area().width, 50);
                assert_eq!(frame.area().height, 7);
            })
            .unwrap();

        // Growing again is symmetric.
        terminal.backend_mut().resize(100, 30);
        terminal
            .draw(|frame| {
                assert_eq!(frame.area().width, 100);
                assert_eq!(frame.area().height, 7);
            })
            .unwrap();

        // A terminal shorter than the fixed viewport height clamps the
        // viewport instead of drawing past the screen edge.
        terminal.backend_mut().resize(60, 4);
        terminal
            .draw(|frame| {
                assert_eq!(frame.area().width, 60);
                assert_eq!(frame.area().height, 4);
            })
            .unwrap();
    }

    #[test]
    fn live_layout_puts_one_blank_row_between_tail_and_sections_below() {
        // A busy 14-row canvas with a minimal prompt leaves room for tail,
        // gap, activity, metadata, and the input box.
        let budget = live_layout(14, true, 1, 0);
        assert_eq!(budget.activity_rows, 1);
        assert_eq!(budget.metadata_rows, 1);
        assert_eq!(budget.prompt_rows, 3);
        assert_eq!(budget.gap_rows, 1, "streaming text needs breathing room");
        assert!(budget.tail_rows >= 1);

        // Nothing below the tail on a tiny canvas: no gap to spend.
        let budget = live_layout(5, true, 1, 0);
        assert_eq!(budget.metadata_rows, 0);
        assert_eq!(budget.activity_rows, 0);
        assert_eq!(budget.gap_rows, 0);

        // The gap never eliminates the tail's last visible row.
        let budget = live_layout(6, true, 1, 0);
        assert_eq!(budget.metadata_rows, 1);
        assert_eq!(budget.tail_rows, 1);
        assert_eq!(budget.gap_rows, 1);
        let budget = live_layout(7, true, 1, 0);
        assert_eq!(budget.activity_rows, 1);
        assert_eq!(budget.tail_rows, 1);
        assert_eq!(budget.gap_rows, 1);
    }

    /// The fixed-canvas budget yields non-overlapping, in-bounds, contiguous
    /// rects for every terminal height in the supported band (including the
    /// degenerate 1–4 row canvases ratatui clamps to when the terminal shrinks
    /// below the fixed viewport height), busy or not, with a completion popup
    /// open or not, and with a prompt grown past its cap. Sections may be
    /// zero-height (e.g. tail on a tiny idle canvas) but must never overlap or
    /// escape the viewport.
    #[test]
    fn live_layout_rects_stay_in_bounds_and_do_not_overlap() {
        for height in 1u16..=16 {
            for busy in [false, true] {
                for completion in [0u16, 3, 8] {
                    for prompt_lines in [1usize, 3, 20] {
                        let budget = live_layout(height, busy, prompt_lines, completion);
                        // The input box keeps its 3-row minimum whenever the
                        // canvas has the rows to spare, and never exceeds the
                        // canvas; the budget never exceeds the canvas.
                        assert!(
                            budget.prompt_rows >= 3.min(height),
                            "height {height} busy {busy} gave prompt {}",
                            budget.prompt_rows
                        );
                        assert!(
                            budget.prompt_rows <= height,
                            "height {height} busy {busy} gave prompt {}",
                            budget.prompt_rows
                        );
                        let total = budget.prompt_rows
                            + budget.completion_rows
                            + budget.activity_rows
                            + budget.metadata_rows
                            + budget.gap_rows
                            + budget.tail_rows;
                        assert_eq!(total, height, "height {height} busy {busy}");

                        // Split a real inline viewport and check the geometry.
                        let backend = TestBackend::new(80, height);
                        let mut terminal = Terminal::with_options(
                            backend,
                            TerminalOptions {
                                viewport: ratatui::Viewport::Inline(height),
                            },
                        )
                        .unwrap();
                        terminal
                            .draw(|frame| {
                                let chunks = Layout::default()
                                    .direction(Direction::Vertical)
                                    .constraints(split_constraints(budget))
                                    .split(frame.area());
                                let mut previous_bottom = 0u16;
                                for chunk in chunks.iter() {
                                    assert_eq!(chunk.x, 0);
                                    assert_eq!(chunk.width, 80);
                                    assert!(chunk.y >= previous_bottom);
                                    assert!(chunk.y + chunk.height <= height);
                                    previous_bottom = chunk.y + chunk.height;
                                }
                                let last = chunks.last().unwrap();
                                assert_eq!(last.y + last.height, height);
                            })
                            .unwrap();
                    }
                }
            }
        }
    }
}
