use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use textwrap::wrap;

/// Return the stable prefix (through the last blank line) and the remainder.
/// Markdown blocks ending at a blank line can safely be rendered without being
/// changed by a later token, while the remainder stays in the live viewport.
pub fn split_stable_prefix(input: &str) -> (String, String) {
    let Some(index) = input.rfind("\n\n") else {
        return (String::new(), input.to_owned());
    };
    let split = index + 2;
    (input[..split].to_owned(), input[split..].to_owned())
}

pub fn stable_prefix(input: &str) -> Option<(&str, &str)> {
    input.rfind("\n\n").map(|index| {
        let split = index + 2;
        (&input[..split], &input[split..])
    })
}

/// Number of terminal rows needed for a plain string at `width` columns.
pub fn wrap_count(input: &str, width: usize) -> usize {
    let width = width.max(1);
    input
        .split('\n')
        .map(|line| {
            if line.is_empty() {
                1
            } else {
                wrap(line, width).len().max(1)
            }
        })
        .sum::<usize>()
        .max(1)
}

pub fn insert_markdown<B: Backend>(
    terminal: &mut Terminal<B>,
    markdown: &str,
    width: u16,
) -> std::io::Result<()> {
    if markdown.is_empty() {
        return Ok(());
    }
    let rendered = tui_markdown::from_str(markdown);
    let plain = rendered
        .lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.to_string())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    // tui-markdown returns ratatui-core's Text while ratatui 0.29 exposes its
    // own compatible Text type. Rebuild lines at this crate boundary.
    let lines = rendered
        .lines
        .iter()
        .map(|line| {
            Line::from(
                line.spans
                    .iter()
                    .map(|span| Span::raw(span.content.to_string()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let height = wrap_count(&plain, width as usize) as u16;
    terminal.insert_before(height.max(1), |buffer| {
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(buffer.area, buffer);
    })
}

pub fn insert_reasoning<B: Backend>(
    terminal: &mut Terminal<B>,
    reasoning: &str,
    width: u16,
) -> std::io::Result<()> {
    if reasoning.is_empty() {
        return Ok(());
    }
    let mut lines = Vec::new();
    for (index, line) in reasoning.lines().enumerate() {
        let marker = if index == 0 { "… thinking: " } else { "  " };
        lines.push(Line::from(Span::styled(
            format!("{marker}{line}"),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM | Modifier::ITALIC),
        )));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "… thinking: ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM | Modifier::ITALIC),
        )));
    }
    let plain = reasoning.to_owned();
    let height = wrap_count(&plain, width as usize).saturating_add(1) as u16;
    terminal.insert_before(height.max(1), |buffer| {
        Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .render(buffer.area, buffer);
    })
}

pub fn insert_user<B: Backend>(
    terminal: &mut Terminal<B>,
    input: &str,
    width: u16,
) -> std::io::Result<()> {
    let mut lines = Vec::new();
    for (index, line) in input.lines().enumerate() {
        let prefix = if index == 0 { "> " } else { "  " };
        lines.push(Line::from(Span::styled(
            format!("{prefix}{line}"),
            Style::default().add_modifier(Modifier::BOLD),
        )));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "> ",
            Style::default().add_modifier(Modifier::BOLD),
        )));
    }
    let height = wrap_count(input, width as usize).saturating_add(1) as u16;
    terminal.insert_before(height.max(1), |buffer| {
        Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .render(buffer.area, buffer);
    })
}

pub fn insert_tool_finished<B: Backend>(
    terminal: &mut Terminal<B>,
    summary: &str,
    ok: bool,
    duration_ms: u64,
    error: Option<&str>,
    width: u16,
) -> std::io::Result<()> {
    let symbol = if ok { "✓" } else { "✗" };
    let duration = if duration_ms >= 1_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{duration_ms}ms")
    };
    let mut text = format!("{symbol} {summary}");
    if let Some(error) = error {
        text.push_str(" — ");
        text.push_str(error.lines().next().unwrap_or(error));
    }
    text.push_str(&format!(" ({duration})"));
    let style = if ok {
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM)
    } else {
        Style::default().fg(Color::Red)
    };
    let height = wrap_count(&text, width as usize) as u16;
    terminal.insert_before(height.max(1), |buffer| {
        Paragraph::new(Line::from(Span::styled(text, style)))
            .wrap(Wrap { trim: false })
            .render(buffer.area, buffer);
    })
}

pub fn render_live(
    area: Rect,
    pending_reasoning: &str,
    pending_text: &str,
    running_tool: Option<(&str, usize)>,
    frame: &mut ratatui::Frame<'_>,
) {
    let mut lines = Vec::new();
    if !pending_reasoning.is_empty() {
        for (index, line) in pending_reasoning.lines().enumerate() {
            lines.push(Line::from(Span::styled(
                if index == 0 {
                    format!("… {line}")
                } else {
                    format!("  {line}")
                },
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM | Modifier::ITALIC),
            )));
        }
    }
    if !pending_text.is_empty() {
        lines.extend(pending_text.lines().map(Line::from));
    }
    if let Some((summary, spinner)) = running_tool {
        const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        lines.push(Line::from(Span::styled(
            format!("{} {}", FRAMES[spinner % FRAMES.len()], summary),
            Style::default().fg(Color::Yellow),
        )));
    }
    if lines.is_empty() {
        return;
    }
    let visible = lines
        .into_iter()
        .rev()
        .take(area.height as usize)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(Text::from(visible)).wrap(Wrap { trim: false }),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_prefix_uses_last_blank_line() {
        let (stable, pending) = split_stable_prefix("one\n\ntwo\n\nthree");
        assert_eq!(stable, "one\n\ntwo\n\n");
        assert_eq!(pending, "three");
    }

    #[test]
    fn wrapping_counts_unicode_and_empty_lines() {
        assert_eq!(wrap_count("", 10), 1);
        assert_eq!(wrap_count("abcdefgh", 4), 2);
        assert_eq!(wrap_count("éééé", 2), 2);
    }
}
