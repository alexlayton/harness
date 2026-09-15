use super::{MAX_DIRECTORY_SUGGESTIONS, MuxLayout, MuxUi, Overlay};
use crate::app::line_to_ansi;
use crate::render;
use anyhow::Result;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::terminal;
use ratatui_core::buffer::Buffer;
use ratatui_core::layout::Rect;
use ratatui_core::style::{Modifier, Style};
use ratatui_core::text::{Line, Span};
use std::fmt::Write as _;
use std::io::{Stdout, Write};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

impl MuxUi {
    /// Render a complete, fixed-height sidebar. Each returned row has exactly
    /// `sidebar_width + 1` cells, keeping its right border at a fixed boundary.
    pub(super) fn sidebar_lines(
        &self,
        sidebar_width: u16,
        height: u16,
        muted: Style,
        accent: Style,
    ) -> Vec<Line<'static>> {
        let total_width = sidebar_width.saturating_add(1) as usize;
        if height == 0 || total_width == 0 {
            return vec![];
        }
        let border = |text: &str| Span::styled(text.to_owned(), muted);
        if total_width == 1 {
            let mut rows = (0..height)
                .map(|_| Line::from(border("│")))
                .collect::<Vec<_>>();
            rows[0] = Line::from(border("╭"));
            if height > 1 {
                rows[height as usize - 1] = Line::from(border("╰"));
            }
            return rows;
        }
        let interior = total_width.saturating_sub(2);
        let framed = |text: &str, style: Style| {
            Line::from(vec![
                border("│"),
                Span::styled(fit(text, interior), style),
                border("│"),
            ])
        };
        let ruled = |left: &str, label: &str, right: &str, style: Style| {
            let label = if label.width() <= interior {
                label.to_owned()
            } else {
                fit(label, interior)
            };
            let fill = "─".repeat(interior.saturating_sub(label.width()));
            Line::from(vec![
                border(left),
                Span::styled(format!("{label}{fill}"), style),
                border(right),
            ])
        };
        let mut rows = (0..height).map(|_| framed("", muted)).collect::<Vec<_>>();
        rows[0] = ruled("╭", "─ SESSIONS ", "╮", muted.add_modifier(Modifier::BOLD));
        if height == 1 {
            return rows;
        }
        rows[height as usize - 1] = Line::from(border(&format!("╰{}╯", "─".repeat(interior))));
        if height == 2 {
            return rows;
        }

        for (n, slot) in self
            .slots
            .iter()
            .skip(self.sidebar_scroll)
            .take(self.visible_rows())
            .enumerate()
        {
            let y = 2 + n;
            let index = self.sidebar_scroll + n;
            let selected = Some(index) == self.selected;
            let text = fit(
                &format!(
                    " {} {} {} {}",
                    if selected { "›" } else { " " },
                    session_ordinal(index),
                    slot.name,
                    slot.status.glyph()
                ),
                interior,
            );
            rows[y] = framed(
                &text,
                if selected {
                    accent.add_modifier(Modifier::BOLD)
                } else {
                    muted
                },
            );
        }
        let visible = self
            .slots
            .len()
            .saturating_sub(self.sidebar_scroll)
            .min(self.visible_rows());
        let new_y = 3 + visible;
        let footer_start = height.saturating_sub(1 + self.sidebar_footer_rows()) as usize;
        if new_y < footer_start {
            rows[new_y] = framed("   + New agent", muted);
        }

        if self.prefix && self.sidebar_footer_rows() == 6 {
            rows[footer_start] = ruled(
                "├",
                "─ MUX › ^Space ",
                "┤",
                accent.add_modifier(Modifier::BOLD),
            );
            for (offset, command) in [
                " n  new      w  worktree",
                " c  current  d  directory",
                " x  close    j  next",
                " k  prev     1–9 jump",
                " ?  help",
            ]
            .iter()
            .enumerate()
            {
                rows[footer_start + 1 + offset] = framed(command, muted);
            }
        } else {
            rows[footer_start] = ruled("├", "", "┤", muted);
            rows[footer_start + 1] = framed(" ^Space  mux", muted);
        }
        rows
    }

    pub(super) fn compose_frame(
        &mut self,
        width: u16,
        height: u16,
    ) -> (Buffer, Option<(u16, u16)>) {
        self.width = width;
        self.height = height;
        let layout = self.layout(width, height);
        let theme = render::Theme::default();
        let muted = Style::default()
            .fg(theme.muted_text)
            .add_modifier(Modifier::DIM);
        let accent = Style::default().fg(theme.accent);
        let mut buffer = Buffer::empty(Rect::new(0, 0, width, height));

        if let Some((_, sidebar_width)) = layout.sidebar {
            for (y, line) in self
                .sidebar_lines(sidebar_width, layout.body_height, muted, accent)
                .iter()
                .enumerate()
            {
                buffer.set_line(0, y as u16, line, sidebar_width.saturating_add(1));
            }
        }

        let pane_cursor = if let Some(i) = self.selected {
            let frame = self.slots[i]
                .pane
                .render(layout.main_width, layout.body_height);
            for y in 0..layout.body_height {
                let line = frame.lines.get(y as usize).cloned().unwrap_or_default();
                buffer.set_line(layout.main_x, y, &line, layout.main_width);
            }
            Some(pane_cursor_position(layout, &frame))
        } else {
            if layout.body_height > 4 {
                let y = layout.body_height / 2;
                buffer.set_string(layout.main_x, y, " No agents yet", muted);
                buffer.set_string(
                    layout.main_x,
                    y + 1,
                    " Press Ctrl+Space then n to create one.",
                    muted,
                );
            }
            None
        };

        let cursor = if let Some(overlay) = &self.overlay {
            draw_overlay(&mut buffer, layout, overlay, theme)
        } else {
            pane_cursor
        };
        (
            buffer,
            cursor.and_then(|cursor| bounded_cursor(cursor, width, height)),
        )
    }

    pub(super) fn draw(&mut self, out: &mut Stdout) -> Result<()> {
        let (width, height) = terminal::size().unwrap_or((self.width, self.height));
        let (frame, cursor) = self.compose_frame(width, height);
        let mut output = Hide.to_string();
        for y in 0..height {
            let line = buffer_row(&frame, y, width);
            let _ = write!(output, "{}{}", MoveTo(0, y), line_to_ansi(&line));
        }
        if let Some((x, y)) = cursor {
            let _ = write!(output, "{}{}", MoveTo(x, y), Show);
        }
        out.write_all(output.as_bytes())?;
        out.flush()?;
        Ok(())
    }
}

/// Convert one Ratatui buffer row back to styled text without emitting the
/// reset continuation cells that follow a wide grapheme. Writing those cells
/// would make the physical terminal row wider than its buffer geometry.
pub(super) fn buffer_row(buffer: &Buffer, y: u16, width: u16) -> Line<'static> {
    let mut spans = Vec::new();
    let mut x = 0;
    while x < width {
        let cell = &buffer[(x, y)];
        let symbol = cell.symbol();
        spans.push(Span::styled(symbol.to_owned(), cell.style()));
        let cells = symbol.width().max(1).min(u16::MAX as usize) as u16;
        x = x.saturating_add(cells);
    }
    Line::from(spans)
}

pub(super) fn pane_cursor_position(layout: MuxLayout, frame: &crate::PaneFrame) -> (u16, u16) {
    (
        layout.main_x.saturating_add(frame.cursor_col),
        frame.cursor_row,
    )
}

pub(super) fn bounded_cursor(cursor: (u16, u16), width: u16, height: u16) -> Option<(u16, u16)> {
    (width > 0 && height > 0).then(|| (cursor.0.min(width - 1), cursor.1.min(height - 1)))
}

pub(super) fn session_ordinal(index: usize) -> String {
    (index + 1).to_string()
}

fn fit(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let cw = c.width().unwrap_or(0);
        if used + cw > width {
            break;
        }
        out.push(c);
        used += cw
    }
    out.push_str(&" ".repeat(width.saturating_sub(used)));
    out
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct OverlayFrame {
    pub(super) x: u16,
    pub(super) y: u16,
    pub(super) width: u16,
    pub(super) rows: Vec<String>,
}

pub(super) fn overlay_frame(layout: MuxLayout, overlay: &Overlay) -> OverlayFrame {
    let title = match overlay {
        Overlay::NewMenu { .. } | Overlay::Current { .. } => "New agent",
        Overlay::Worktree { .. } => "New worktree agent",
        Overlay::Directory { .. } => "Choose directory",
        Overlay::Close => "Close agent?",
        Overlay::ConfirmExit => "Exit Harness?",
        Overlay::ConfirmDuplicate { .. } => "Duplicate workspace?",
        Overlay::Help => "Mux help",
    };
    let content_lines = match overlay {
        Overlay::NewMenu { selected } => ["New worktree", "Current directory", "Another directory"]
            .iter()
            .enumerate()
            .flat_map(|(index, label)| {
                [
                    format!("{} {label}", if index == *selected { '›' } else { ' ' }),
                    String::new(),
                ]
            })
            .collect::<Vec<_>>(),
        Overlay::Worktree { branch, keep } => vec![
            format!("Branch / name: {branch}"),
            format!(
                "Retain for future runs: {} (Tab)",
                if *keep { "yes" } else { "no" }
            ),
        ],
        Overlay::Current { name } => vec![format!("Session name: {name}")],
        // Reserve a compact candidate list so asynchronous scans do not move
        // or resize the editor while results filter down as the user types.
        Overlay::Directory {
            input,
            suggestions,
            selected,
        } => std::iter::once(format!("Directory: {input}"))
            .chain((0..MAX_DIRECTORY_SUGGESTIONS).map(|index| {
                suggestions.get(index).map_or_else(String::new, |path| {
                    format!(
                        "{} {}",
                        if index == *selected { '›' } else { ' ' },
                        path.display()
                    )
                })
            }))
            .collect(),
        Overlay::Close => vec!["Enter to close; Esc to cancel".into()],
        Overlay::ConfirmExit => {
            vec!["All running agents will stop. Enter to exit; Esc to cancel".into()]
        }
        Overlay::ConfirmDuplicate { .. } => vec![
            "A direct agent already uses this directory. Enter to create anyway; Esc to cancel"
                .into(),
        ],
        Overlay::Help => vec![
            "Ctrl+Space: n menu · w worktree · c current".into(),
            "d directory · j/k switch · 1-9 jump".into(),
            "x close · ? help".into(),
            "Mouse: click sessions/new/menu choices; wheel scroll".into(),
        ],
    };
    let width = if layout.main_width < 10 {
        layout.main_width
    } else {
        layout.main_width.saturating_sub(4).clamp(10, 58)
    };
    let height = (content_lines.len() + 4).min(layout.body_height as usize) as u16;
    let x = layout.main_x + layout.main_width.saturating_sub(width) / 2;
    let y = layout.body_height.saturating_sub(height) / 2;
    let mut rows = Vec::with_capacity(height as usize);
    for row in 0..height {
        let text = match (width, height, row) {
            (0, _, _) => String::new(),
            (1, _, _) => "│".into(),
            (_, 1, _) => format!("╭{}╮", "─".repeat(width.saturating_sub(2) as usize)),
            (_, _, 0) => {
                let interior = width.saturating_sub(2) as usize;
                let label = format!("─ {title} ");
                let rule = if label.width() <= interior {
                    format!("{label}{}", "─".repeat(interior - label.width()))
                } else {
                    fit(&label, interior)
                };
                format!("╭{rule}╮")
            }
            (_, _, r) if r == height - 1 => {
                format!("╰{}╯", "─".repeat(width.saturating_sub(2) as usize))
            }
            (_, _, 1) => "│".into(),
            _ => format!(
                "│ {}",
                content_lines
                    .get((row - 2) as usize)
                    .map(String::as_str)
                    .unwrap_or("")
            ),
        };
        let mut fitted = fit(&text, width as usize);
        if width >= 2 {
            let right = if row == 0 {
                '╮'
            } else if row == height - 1 {
                '╯'
            } else {
                '│'
            };
            fitted.pop();
            fitted.push(right);
        }
        rows.push(fitted);
    }
    OverlayFrame { x, y, width, rows }
}

pub(super) fn overlay_cursor(frame: &OverlayFrame, overlay: &Overlay) -> Option<(u16, u16)> {
    let (label, input) = match overlay {
        Overlay::Worktree { branch, .. } => ("Branch / name: ", branch.as_str()),
        Overlay::Current { name } => ("Session name: ", name.as_str()),
        Overlay::Directory { input, .. } => ("Directory: ", input.as_str()),
        _ => return None,
    };
    // Editable content is on the first content row, after the left border and
    // its padding. Clamp the caret to the final opaque interior cell when the
    // value is wider than the modal.
    if frame.width < 3 || frame.rows.len() <= 3 {
        return None;
    }
    let column = 2usize + label.width() + input.width();
    Some((
        frame.x + column.min(frame.width.saturating_sub(2) as usize) as u16,
        frame.y + 2,
    ))
}

pub(super) fn draw_overlay(
    buffer: &mut Buffer,
    layout: MuxLayout,
    overlay: &Overlay,
    theme: render::Theme,
) -> Option<(u16, u16)> {
    let frame = overlay_frame(layout, overlay);
    let cursor = overlay_cursor(&frame, overlay);
    let primary = Style::default().fg(theme.primary_text);
    let muted = Style::default()
        .fg(theme.muted_text)
        .add_modifier(Modifier::DIM);
    let accent = Style::default().fg(theme.accent);
    for (row, text) in frame.rows.iter().enumerate() {
        let directory_selection = match overlay {
            Overlay::Directory { selected, .. } => row == selected + 3,
            _ => false,
        };
        let style = if row == 0 || row + 1 == frame.rows.len() {
            muted
        } else if directory_selection {
            accent.add_modifier(Modifier::BOLD)
        } else if matches!(overlay, Overlay::Directory { .. }) && row >= 3 {
            muted
        } else if matches!(overlay, Overlay::NewMenu { selected: 0 }) && row == 2
            || matches!(overlay, Overlay::NewMenu { selected: 1 }) && row == 4
            || matches!(overlay, Overlay::NewMenu { selected: 2 }) && row == 6
        {
            accent.add_modifier(Modifier::BOLD)
        } else {
            primary
        };
        let y = frame.y + row as u16;
        buffer.set_line(
            frame.x,
            y,
            &Line::from(Span::styled(text.clone(), style)),
            frame.width,
        );
        // Keep every modal outline the same muted gray as the sidebar,
        // independently of highlighted or primary-colored content.
        if row == 0 || row + 1 == frame.rows.len() {
            for x in frame.x..frame.x.saturating_add(frame.width) {
                buffer[(x, y)].set_style(muted);
            }
        } else if frame.width > 0 {
            buffer[(frame.x, y)].set_style(muted);
            if frame.width > 1 {
                buffer[(frame.x + frame.width - 1, y)].set_style(muted);
            }
        }
    }
    cursor
}
