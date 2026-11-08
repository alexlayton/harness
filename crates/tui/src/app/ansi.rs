//! Serialization of styled rows to ANSI. Content is sanitized here, so only
//! generated styles ever reach the terminal.

use super::*;

/// Serialize one styled ratatui line to ANSI. Ratatui styles describe each
/// span independently, while terminal attributes persist until reset, so a
/// reset is required when adjacent effective styles differ. Without it the
/// input prefix's bold attribute leaks into the first text row even though
/// continuation rows correctly use normal weight.
pub(crate) fn line_to_ansi(line: &Line<'_>) -> String {
    let mut out = String::new();
    let mut previous = None;
    let mut styled = false;
    for span in &line.spans {
        let content = render::sanitize_terminal_text(span.content.as_ref());
        if content.is_empty() {
            continue;
        }
        let style = line.style.patch(span.style);
        if previous.is_some_and(|previous| previous != style) {
            out.push_str("\u{1b}[0m");
            styled = true;
        }
        if previous != Some(style) {
            let prefix = style_prefix(&style);
            styled |= !prefix.is_empty();
            out.push_str(&prefix);
        }
        out.push_str(&content);
        previous = Some(style);
    }
    if styled {
        out.push_str("\u{1b}[0m");
    }
    out
}

pub(super) fn style_prefix(style: &Style) -> String {
    let mut out = String::new();
    if let Some(color) = style.fg {
        let _ = write!(out, "{}", SetForegroundColor(ansi_color(color)));
    }
    if let Some(color) = style.bg {
        let _ = write!(out, "{}", SetBackgroundColor(ansi_color(color)));
    }
    for (modifier, attribute) in MODIFIER_ATTRIBUTES {
        if style.add_modifier.contains(*modifier) {
            let _ = write!(out, "{}", SetAttribute(*attribute));
        }
    }
    out
}

/// ratatui → crossterm modifier pairs for [`style_prefix`].
pub(super) const MODIFIER_ATTRIBUTES: &[(Modifier, Attribute)] = &[
    (Modifier::BOLD, Attribute::Bold),
    (Modifier::DIM, Attribute::Dim),
    (Modifier::ITALIC, Attribute::Italic),
    (Modifier::UNDERLINED, Attribute::Underlined),
    (Modifier::SLOW_BLINK, Attribute::SlowBlink),
    (Modifier::RAPID_BLINK, Attribute::RapidBlink),
    (Modifier::REVERSED, Attribute::Reverse),
    (Modifier::HIDDEN, Attribute::Hidden),
    (Modifier::CROSSED_OUT, Attribute::CrossedOut),
];

/// ratatui-core → crossterm color mapping, mirroring ratatui's own
/// crossterm-backend mapping so the theme renders identically.
pub(super) fn ansi_color(color: ratatui_core::style::Color) -> AnsiColor {
    use ratatui_core::style::Color as CoreColor;
    match color {
        CoreColor::Reset => AnsiColor::Reset,
        CoreColor::Black => AnsiColor::Black,
        CoreColor::Red => AnsiColor::DarkRed,
        CoreColor::Green => AnsiColor::DarkGreen,
        CoreColor::Yellow => AnsiColor::DarkYellow,
        CoreColor::Blue => AnsiColor::DarkBlue,
        CoreColor::Magenta => AnsiColor::DarkMagenta,
        CoreColor::Cyan => AnsiColor::DarkCyan,
        CoreColor::Gray => AnsiColor::Grey,
        CoreColor::DarkGray => AnsiColor::DarkGrey,
        CoreColor::LightRed => AnsiColor::Red,
        CoreColor::LightGreen => AnsiColor::Green,
        CoreColor::LightYellow => AnsiColor::Yellow,
        CoreColor::LightBlue => AnsiColor::Blue,
        CoreColor::LightMagenta => AnsiColor::Magenta,
        CoreColor::LightCyan => AnsiColor::Cyan,
        CoreColor::White => AnsiColor::White,
        CoreColor::Rgb(r, g, b) => AnsiColor::Rgb { r, g, b },
        CoreColor::Indexed(value) => AnsiColor::AnsiValue(value),
    }
}

/// Write one row: the shared left gutter plus the line's ANSI serialization.
/// Blank rows become gutter-width space runs, which render identically.
pub(super) fn write_row(buffer: &mut String, line: &Line<'_>, gutter: usize, width: usize) {
    if gutter > 0 {
        buffer.push_str(&" ".repeat(gutter.min(width)));
    }
    let content_width = width.saturating_sub(gutter);
    let line = render::fit_line_to_width(line, content_width);
    buffer.push_str(&line_to_ansi(&line));
}
