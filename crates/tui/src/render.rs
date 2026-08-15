use crate::commands::Candidate;
use crate::state::{ToolRecord, ToolStatus, TranscriptEntry};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui_core::style::{Color as CoreColor, Modifier as CoreModifier, Style as CoreStyle};
use tui_markdown::{AlertKind, Options, StyleSheet, from_str_with_options};
use tui_textarea::TextArea;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// The semantic palette used by every renderer. Keeping these roles together
/// prevents individual widgets from slowly acquiring unrelated colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    pub background: Color,
    pub primary_text: Color,
    pub assistant_text: Color,
    pub secondary_text: Color,
    pub muted_text: Color,
    pub dim_text: Color,
    pub accent: Color,
    pub code_background: Color,
    pub tool_background: Color,
    pub tool_running_border: Color,
    pub tool_success_border: Color,
    pub tool_failure_border: Color,
    pub selection: Color,
    pub focus: Color,
    pub error: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            // Use the terminal's default colours for most backgrounds/text so
            // the UI respects the user's terminal theme instead of imposing a
            // dark background everywhere.
            background: Color::Reset,
            primary_text: Color::Reset,
            assistant_text: Color::Reset,
            secondary_text: Color::Reset,
            muted_text: Color::DarkGray,
            dim_text: Color::DarkGray,
            accent: Color::Cyan,
            code_background: Color::Reset,
            tool_background: Color::Reset,
            tool_running_border: Color::DarkGray,
            tool_success_border: Color::Green,
            tool_failure_border: Color::Red,
            selection: Color::Blue,
            focus: Color::Blue,
            error: Color::Red,
        }
    }
}

pub const MAX_COMPLETION_ROWS: usize = 8;
pub const ACTIVITY_FRAMES: &[&str] = &["·", "∙", "•", "●", "•", "∙"];
const USER_PREFIX: &str = "› ";
const ASSISTANT_PREFIX: &str = "‹ ";
pub const SECTION_GAP: usize = 1;
pub const BLOCK_GAP: usize = 1;
pub const DEFAULT_TAIL_LINES: usize = 4;

/// The key descriptions shown on the empty-session screen. This is kept next
/// to input rendering so the welcome screen cannot drift from the keymap.
pub const KEYMAP: &[(&str, &str)] = &[
    ("Enter", "Submit prompt"),
    ("Shift+Enter", "Insert newline"),
    ("↑ / ↓", "Move through prompt"),
    ("k / j", "Scroll transcript"),
    ("Mouse wheel", "Scroll transcript"),
    ("PageUp / PageDown", "Scroll transcript"),
    ("End / Ctrl+End", "Return to bottom"),
    ("Ctrl+O", "Expand / collapse tool"),
    ("Ctrl+C", "Cancel / quit"),
    ("Esc", "Close transient UI"),
    ("/", "Commands"),
    ("@", "File references"),
];

/// Compatibility wrapper retained for embedders that used the old live-tail
/// type. The retained-mode renderer uses `TranscriptEntry::Tool` directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailTool {
    pub record: ToolRecord,
    pub expanded: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct MarkdownTheme {
    theme: Theme,
}

fn core_color(color: Color) -> CoreColor {
    match color {
        Color::Reset => CoreColor::Reset,
        Color::Black => CoreColor::Black,
        Color::Red => CoreColor::Red,
        Color::Green => CoreColor::Green,
        Color::Yellow => CoreColor::Yellow,
        Color::Blue => CoreColor::Blue,
        Color::Magenta => CoreColor::Magenta,
        Color::Cyan => CoreColor::Cyan,
        Color::Gray => CoreColor::Gray,
        Color::DarkGray => CoreColor::DarkGray,
        Color::LightRed => CoreColor::LightRed,
        Color::LightGreen => CoreColor::LightGreen,
        Color::LightYellow => CoreColor::LightYellow,
        Color::LightBlue => CoreColor::LightBlue,
        Color::LightMagenta => CoreColor::LightMagenta,
        Color::LightCyan => CoreColor::LightCyan,
        Color::White => CoreColor::White,
        Color::Rgb(red, green, blue) => CoreColor::Rgb(red, green, blue),
        Color::Indexed(value) => CoreColor::Indexed(value),
    }
}

fn core_fg(color: Color) -> CoreStyle {
    CoreStyle::default().fg(core_color(color))
}

impl StyleSheet for MarkdownTheme {
    fn heading(&self, _level: u8) -> CoreStyle {
        core_fg(self.theme.accent).add_modifier(CoreModifier::BOLD)
    }

    fn code(&self) -> CoreStyle {
        core_fg(self.theme.primary_text).bg(core_color(self.theme.code_background))
    }

    fn link(&self) -> CoreStyle {
        core_fg(self.theme.accent).add_modifier(CoreModifier::UNDERLINED)
    }

    fn blockquote(&self) -> CoreStyle {
        core_fg(self.theme.muted_text)
    }

    fn heading_meta(&self) -> CoreStyle {
        core_fg(self.theme.dim_text)
    }

    fn metadata_block(&self) -> CoreStyle {
        core_fg(self.theme.muted_text)
    }

    fn html(&self) -> CoreStyle {
        core_fg(self.theme.dim_text)
    }

    fn math_inline(&self) -> CoreStyle {
        core_fg(self.theme.accent).add_modifier(CoreModifier::ITALIC)
    }

    fn math_display(&self) -> CoreStyle {
        core_fg(self.theme.accent)
    }

    fn table_header(&self) -> CoreStyle {
        core_fg(self.theme.primary_text).add_modifier(CoreModifier::BOLD)
    }

    fn table_cell(&self) -> CoreStyle {
        core_fg(self.theme.assistant_text)
    }

    fn table_border(&self) -> CoreStyle {
        core_fg(self.theme.dim_text)
    }

    fn image_alt(&self) -> CoreStyle {
        core_fg(self.theme.dim_text).add_modifier(CoreModifier::ITALIC)
    }

    fn alert(&self, kind: AlertKind) -> CoreStyle {
        let color = match kind {
            AlertKind::Note => self.theme.accent,
            AlertKind::Tip => self.theme.tool_success_border,
            AlertKind::Important => self.theme.accent,
            AlertKind::Warning => Color::Yellow,
            AlertKind::Caution => self.theme.tool_failure_border,
        };
        core_fg(color)
    }
}

fn dim_style(theme: Theme) -> Style {
    Style::default().fg(theme.dim_text)
}

fn muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text)
}

fn primary_style(theme: Theme) -> Style {
    Style::default().fg(theme.primary_text)
}

fn assistant_style(theme: Theme) -> Style {
    Style::default().fg(theme.assistant_text)
}

fn message_prefix_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD)
}

fn error_style(theme: Theme) -> Style {
    Style::default().fg(theme.error)
}

fn blank_line() -> Line<'static> {
    Line::from("")
}

fn push_blank(lines: &mut Vec<Line<'static>>, count: usize) {
    lines.extend(std::iter::repeat_with(blank_line).take(count));
}

fn push_span(line: &mut Vec<Span<'static>>, value: impl Into<String>, style: Style) {
    let value = value.into();
    if value.is_empty() {
        return;
    }
    if let Some(last) = line.last_mut()
        && last.style == style
    {
        last.content.to_mut().push_str(&value);
        return;
    }
    line.push(Span::styled(value, style));
}

fn line_with_style(value: impl Into<String>, style: Style) -> Line<'static> {
    Line::from(Span::styled(value.into(), style))
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum()
}

fn convert_color(color: CoreColor) -> Color {
    match color {
        CoreColor::Reset => Color::Reset,
        CoreColor::Black => Color::Black,
        CoreColor::Red => Color::Red,
        CoreColor::Green => Color::Green,
        CoreColor::Yellow => Color::Yellow,
        CoreColor::Blue => Color::Blue,
        CoreColor::Magenta => Color::Magenta,
        CoreColor::Cyan => Color::Cyan,
        CoreColor::Gray => Color::Gray,
        CoreColor::DarkGray => Color::DarkGray,
        CoreColor::LightRed => Color::LightRed,
        CoreColor::LightGreen => Color::LightGreen,
        CoreColor::LightYellow => Color::LightYellow,
        CoreColor::LightBlue => Color::LightBlue,
        CoreColor::LightMagenta => Color::LightMagenta,
        CoreColor::LightCyan => Color::LightCyan,
        CoreColor::White => Color::White,
        CoreColor::Rgb(red, green, blue) => Color::Rgb(red, green, blue),
        CoreColor::Indexed(value) => Color::Indexed(value),
    }
}

fn convert_modifier(modifier: CoreModifier) -> Modifier {
    let mut converted = Modifier::empty();
    for (source, target) in [
        (CoreModifier::BOLD, Modifier::BOLD),
        (CoreModifier::DIM, Modifier::DIM),
        (CoreModifier::ITALIC, Modifier::ITALIC),
        (CoreModifier::UNDERLINED, Modifier::UNDERLINED),
        (CoreModifier::SLOW_BLINK, Modifier::SLOW_BLINK),
        (CoreModifier::RAPID_BLINK, Modifier::RAPID_BLINK),
        (CoreModifier::REVERSED, Modifier::REVERSED),
        (CoreModifier::HIDDEN, Modifier::HIDDEN),
        (CoreModifier::CROSSED_OUT, Modifier::CROSSED_OUT),
    ] {
        if modifier.contains(source) {
            converted.insert(target);
        }
    }
    converted
}

fn convert_style(style: CoreStyle) -> Style {
    let mut converted = Style::default();
    if let Some(color) = style.fg {
        converted = converted.fg(convert_color(color));
    }
    if let Some(color) = style.bg {
        converted = converted.bg(convert_color(color));
    }
    converted
        .add_modifier(convert_modifier(style.add_modifier))
        .remove_modifier(convert_modifier(style.sub_modifier))
}

fn span_style(base: Style, line_style: Style, span_style: Style) -> Style {
    base.patch(line_style).patch(span_style)
}

/// Wrap a styled Ratatui text value while preserving span styles. Text is
/// wrapped at whitespace when possible; a single word is split only when it
/// is wider than the available line. This is the common measurement/rendering
/// path for Markdown and transcript scrolling.
pub fn wrap_text(text: &Text<'_>, width: usize, base: Style) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut result = Vec::new();

    for source_line in &text.lines {
        let line_base = base.patch(source_line.style);
        let mut source_chars = Vec::<(char, Style, usize)>::new();
        for source_span in &source_line.spans {
            let style = span_style(line_base, Style::default(), source_span.style);
            source_chars.extend(source_span.content.chars().map(|character| {
                let character_width = UnicodeWidthChar::width(character).unwrap_or(1).max(1);
                (character, style, character_width)
            }));
        }

        if source_chars.is_empty() {
            result.push(Line::from("").style(line_base));
            continue;
        }

        let mut current = Vec::<(char, Style, usize)>::new();
        let mut current_width = 0usize;
        let mut pending_whitespace = Vec::<(char, Style, usize)>::new();
        let mut pending_width = 0usize;
        let mut index = 0usize;

        while index < source_chars.len() {
            let is_whitespace = source_chars[index].0.is_whitespace();
            let start = index;
            while index < source_chars.len()
                && source_chars[index].0.is_whitespace() == is_whitespace
            {
                index += 1;
            }
            let group = &source_chars[start..index];
            let group_width = group.iter().map(|(_, _, width)| *width).sum::<usize>();

            if is_whitespace {
                if current.is_empty() {
                    current.extend_from_slice(group);
                    current_width = current_width.saturating_add(group_width);
                } else {
                    pending_whitespace.extend_from_slice(group);
                    pending_width = pending_width.saturating_add(group_width);
                }
                continue;
            }

            if !current.is_empty()
                && current_width
                    .saturating_add(pending_width)
                    .saturating_add(group_width)
                    > width
            {
                result.push(wrapped_line(std::mem::take(&mut current)));
                current_width = 0;
                pending_whitespace.clear();
                pending_width = 0;
            }

            if !pending_whitespace.is_empty() {
                current_width = current_width.saturating_add(pending_width);
                current.append(&mut pending_whitespace);
                pending_width = 0;
            }

            for &(character, style, character_width) in group {
                if current_width > 0 && current_width.saturating_add(character_width) > width {
                    result.push(wrapped_line(std::mem::take(&mut current)));
                    current_width = 0;
                }
                current.push((character, style, character_width));
                current_width = current_width.saturating_add(character_width);
            }
        }

        // Trailing whitespace is not visible and should not create an extra
        // whitespace-only row when it happens to land at the wrap boundary.
        if current.is_empty() {
            result.push(Line::from("").style(line_base));
        } else {
            result.push(wrapped_line(current));
        }
    }

    if result.is_empty() {
        result.push(Line::from("").style(base));
    }
    result
}

fn wrapped_line(chars: Vec<(char, Style, usize)>) -> Line<'static> {
    let mut spans = Vec::new();
    for (character, style, _) in chars {
        push_span(&mut spans, character.to_string(), style);
    }
    Line::from(spans)
}

fn plain_text(value: &str, style: Style) -> Text<'static> {
    Text::from(
        value
            .split('\n')
            .map(|line| line_with_style(line.to_owned(), style))
            .collect::<Vec<_>>(),
    )
}

fn owned_markdown(markdown: &str, theme: Theme) -> Text<'static> {
    let options = Options::new(MarkdownTheme { theme });
    let rendered = from_str_with_options(markdown, &options);
    let lines = rendered
        .lines
        .iter()
        .map(|line| {
            let mut owned = Line::from(
                line.spans
                    .iter()
                    .map(|span| Span::styled(span.content.to_string(), convert_style(span.style)))
                    .collect::<Vec<_>>(),
            );
            owned.style = convert_style(line.style);
            owned
        })
        .collect::<Vec<_>>();
    Text::from(lines)
}

fn prefix_message_lines(
    lines: Vec<Line<'static>>,
    prefix: &str,
    theme: Theme,
) -> Vec<Line<'static>> {
    let prefix_style = message_prefix_style(theme);
    let continuation = " ".repeat(UnicodeWidthStr::width(prefix));
    let mut has_prefix = false;
    lines
        .into_iter()
        .map(|line| {
            if line_width(&line) == 0 {
                return line;
            }
            let prefix = if has_prefix {
                Span::raw(continuation.clone())
            } else {
                has_prefix = true;
                Span::styled(prefix.to_owned(), prefix_style)
            };
            Line::from(
                std::iter::once(prefix)
                    .chain(line.spans)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn message_content_width(width: usize) -> usize {
    width
        .saturating_sub(UnicodeWidthStr::width(USER_PREFIX))
        .max(1)
}

fn reasoning_lines(reasoning: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = plain_text(reasoning, muted_style(theme).add_modifier(Modifier::ITALIC));
    wrap_text(&text, width, Style::default())
}

fn markdown_lines(markdown: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = owned_markdown(markdown, theme);
    prefix_message_lines(
        wrap_text(&text, message_content_width(width), assistant_style(theme)),
        ASSISTANT_PREFIX,
        theme,
    )
}

fn user_lines(input: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    prefix_message_lines(
        wrap_text(
            &plain_text(input, primary_style(theme)),
            message_content_width(width),
            Style::default(),
        ),
        USER_PREFIX,
        theme,
    )
}

fn notice_lines(notice: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = plain_text(notice, dim_style(theme));
    wrap_text(&text, width.saturating_sub(2).max(1), Style::default())
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "· " } else { "  " };
            let mut spans = vec![Span::styled(prefix, dim_style(theme))];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect()
}

fn error_lines(error: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = plain_text(error, error_style(theme));
    wrap_text(&text, width.saturating_sub(2).max(1), Style::default())
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "✗ " } else { "  " };
            let mut spans = vec![Span::styled(prefix, error_style(theme))];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect()
}

fn title_tail(name: &str, summary: &str) -> String {
    let summary = summary.trim();
    let without_name = summary
        .strip_prefix(name)
        .map(|rest| rest.strip_prefix(':').unwrap_or(rest).trim_start())
        .unwrap_or(summary);
    if without_name.is_empty() {
        name.to_owned()
    } else {
        format!("{name} · {without_name}")
    }
}

fn duration_text(duration_ms: u64) -> String {
    if duration_ms >= 1_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{duration_ms}ms")
    }
}

fn tool_border_style(record: &ToolRecord, theme: Theme) -> Style {
    let color = match record.status {
        ToolStatus::Running => theme.tool_running_border,
        ToolStatus::Success => theme.tool_success_border,
        ToolStatus::Failure => theme.tool_failure_border,
    };
    Style::default().fg(color)
}

fn tool_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.secondary_text)
}

fn tool_header_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.primary_text)
        .add_modifier(Modifier::BOLD)
}

fn tool_hint_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM)
}

fn push_tool_body_line(
    lines: &mut Vec<Line<'static>>,
    content: Line<'static>,
    width: usize,
    border: Style,
    body: Style,
) {
    let inner_width = width.saturating_sub(4).max(1);
    let wrapped = wrap_text(&Text::from(content), inner_width, body);
    for wrapped_line in wrapped {
        let used = line_width(&wrapped_line);
        let mut spans = vec![Span::styled("│ ", border)];
        spans.extend(wrapped_line.spans);
        if used < inner_width {
            spans.push(Span::styled(" ".repeat(inner_width - used), body));
        }
        spans.push(Span::styled(" │", border));
        lines.push(Line::from(spans));
    }
}

fn push_tool_plain_body(
    lines: &mut Vec<Line<'static>>,
    value: &str,
    width: usize,
    border: Style,
    body: Style,
) {
    let text = plain_text(value, body);
    for line in wrap_text(&text, width.saturating_sub(4).max(1), Style::default()) {
        push_tool_body_line(lines, line, width, border, body);
    }
}

fn tool_top_line(title: &str, width: usize, border: Style, title_style: Style) -> Line<'static> {
    if width <= 3 {
        return Line::from(Span::styled("─".repeat(width), border));
    }
    let available = width.saturating_sub(4);
    let mut title = title.to_owned();
    while UnicodeWidthStr::width(title.as_str()) > available {
        title.pop();
    }
    let used = 3 + UnicodeWidthStr::width(title.as_str());
    let fill = width.saturating_sub(used + 1);
    Line::from(vec![
        Span::styled("╭─ ", border),
        Span::styled(title, title_style),
        Span::styled(format!(" {}╮", "─".repeat(fill.saturating_sub(1))), border),
    ])
}

fn tool_bottom_line(width: usize, border: Style) -> Line<'static> {
    if width <= 2 {
        return Line::from(Span::styled("─".repeat(width), border));
    }
    Line::from(Span::styled(
        format!("╰{}╯", "─".repeat(width.saturating_sub(2))),
        border,
    ))
}

fn output_tail(output: &str) -> Vec<String> {
    let lines = output.lines().map(str::to_owned).collect::<Vec<_>>();
    if lines.len() <= DEFAULT_TAIL_LINES {
        return lines;
    }
    let omitted = lines.len() - DEFAULT_TAIL_LINES;
    let mut result = vec![format!("… {omitted} lines above")];
    result.extend(lines.into_iter().skip(omitted));
    result
}

fn tool_box_lines(
    record: &ToolRecord,
    expanded: bool,
    width: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    let border = tool_border_style(record, theme);
    let body = tool_text_style(theme);
    let title_style = tool_header_style(theme);
    let mut lines = vec![tool_top_line(
        &title_tail(&record.name, &record.summary),
        width,
        border,
        title_style,
    )];

    if expanded {
        push_tool_body_line(
            &mut lines,
            line_with_style("args", tool_header_style(theme)),
            width,
            border,
            body,
        );
        if record.args.trim().is_empty() {
            push_tool_plain_body(&mut lines, "(none)", width, border, body);
        } else {
            push_tool_plain_body(&mut lines, &record.args, width, border, body);
        }
        push_tool_body_line(
            &mut lines,
            line_with_style("output", tool_header_style(theme)),
            width,
            border,
            body,
        );
        if record.output.is_empty() {
            push_tool_plain_body(&mut lines, "(empty)", width, border, body);
        } else {
            push_tool_plain_body(&mut lines, &record.output, width, border, body);
        }
        if let Some(error) = record.error.as_deref() {
            push_tool_body_line(
                &mut lines,
                line_with_style("error", error_style(theme)),
                width,
                border,
                body,
            );
            push_tool_plain_body(&mut lines, error, width, border, error_style(theme));
        }
        if !record.status.is_running() {
            push_tool_plain_body(
                &mut lines,
                &format!("completed in {}", duration_text(record.duration_ms)),
                width,
                border,
                tool_hint_style(theme),
            );
        }
    } else {
        match record.status {
            ToolStatus::Running => {
                push_tool_plain_body(&mut lines, "running…", width, border, body);
            }
            ToolStatus::Success | ToolStatus::Failure => {
                let tail = output_tail(&record.output);
                if tail.is_empty() {
                    push_tool_plain_body(&mut lines, "(no output)", width, border, body);
                } else {
                    for line in tail {
                        push_tool_plain_body(&mut lines, &line, width, border, body);
                    }
                }
                if let Some(error) = record.error.as_deref() {
                    let preview = error.lines().next().unwrap_or(error);
                    push_tool_plain_body(&mut lines, preview, width, border, error_style(theme));
                }
            }
        }
        push_tool_body_line(
            &mut lines,
            line_with_style("ctrl + o to expand", tool_hint_style(theme)),
            width,
            border,
            tool_hint_style(theme),
        );
    }

    lines.push(tool_bottom_line(width, border));
    lines
}

fn welcome_lines(width: usize, theme: Theme) -> Vec<Line<'static>> {
    const ASCII_TITLE: &[&str] = &[
        "  ██   ██  █████  ██████  ███    ██ ███████ ███████ ███████",
        "  ██   ██ ██   ██ ██   ██ ████   ██ ██      ██      ██",
        "  ███████ ███████ ██████  ██ ██  ██ █████   ███████ ███████",
        "  ██   ██ ██   ██ ██   ██ ██  ██ ██ ██           ██      ██",
        "  ██   ██ ██   ██ ██   ██ ██   ████ ███████ ███████ ███████",
    ];
    let title_width = ASCII_TITLE
        .iter()
        .map(|line| UnicodeWidthStr::width(*line))
        .max()
        .unwrap_or(0);
    let mut lines = Vec::new();
    if width >= title_width + 2 {
        lines.extend(
            ASCII_TITLE.iter().map(|line| {
                line_with_style(*line, primary_style(theme).add_modifier(Modifier::BOLD))
            }),
        );
    } else {
        lines.push(line_with_style(
            "Harness",
            primary_style(theme).add_modifier(Modifier::BOLD),
        ));
    }
    push_blank(&mut lines, 2);

    let label_width = KEYMAP
        .iter()
        .map(|(label, _)| UnicodeWidthStr::width(*label))
        .max()
        .unwrap_or(0);
    for (label, description) in KEYMAP {
        let padding = " ".repeat(label_width.saturating_sub(UnicodeWidthStr::width(*label)) + 4);
        lines.push(Line::from(vec![
            Span::styled((*label).to_owned(), dim_style(theme)),
            Span::raw(padding),
            Span::styled((*description).to_owned(), muted_style(theme)),
        ]));
    }
    lines
}

/// Build all transcript rows at the current width. Every returned line is
/// already wrapped, so the scroll offset is measured in the same rows that are
/// ultimately painted.
pub fn transcript_lines(
    entries: &[TranscriptEntry],
    show_welcome: bool,
    width: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut lines = if show_welcome {
        welcome_lines(width, theme)
    } else {
        Vec::new()
    };

    for entry in entries {
        if !lines.is_empty() {
            push_blank(&mut lines, SECTION_GAP);
        }
        match entry {
            TranscriptEntry::User { text, .. } => {
                lines.extend(user_lines(text, theme, width));
            }
            TranscriptEntry::Assistant {
                markdown,
                reasoning,
                ..
            } => {
                if !reasoning.is_empty() {
                    lines.extend(reasoning_lines(reasoning, theme, width));
                    if !markdown.is_empty() {
                        push_blank(&mut lines, BLOCK_GAP);
                    }
                }
                if !markdown.is_empty() {
                    lines.extend(markdown_lines(markdown, theme, width));
                }
            }
            TranscriptEntry::Tool {
                record, expanded, ..
            } => {
                push_blank(&mut lines, BLOCK_GAP);
                lines.extend(tool_box_lines(record, *expanded, width, theme));
                push_blank(&mut lines, BLOCK_GAP);
            }
            TranscriptEntry::Notice { text, .. } => {
                lines.extend(notice_lines(text, theme, width));
            }
            TranscriptEntry::Error { text, .. } => {
                lines.extend(error_lines(text, theme, width));
            }
        }
    }

    if lines.is_empty() {
        lines.push(blank_line());
    }
    lines
}

pub fn render_transcript_lines(
    area: Rect,
    lines: &[Line<'static>],
    offset: usize,
    theme: Theme,
    frame: &mut Frame<'_>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let start = offset.min(lines.len().saturating_sub(area.height as usize));
    let visible = lines
        .iter()
        .skip(start)
        .take(area.height as usize)
        .cloned()
        .collect::<Vec<_>>();
    frame.render_widget(
        Block::default().style(Style::default().bg(theme.background)),
        area,
    );
    frame.render_widget(Paragraph::new(Text::from(visible)), area);
}

pub fn render_transcript(
    area: Rect,
    entries: &[TranscriptEntry],
    show_welcome: bool,
    offset: usize,
    theme: Theme,
    frame: &mut Frame<'_>,
) -> usize {
    let lines = transcript_lines(entries, show_welcome, area.width as usize, theme);
    let count = lines.len();
    render_transcript_lines(area, &lines, offset, theme, frame);
    count
}

pub fn render_new_content_indicator(area: Rect, theme: Theme, frame: &mut Frame<'_>) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let text = "↓ new content below · End to follow";
    let width = UnicodeWidthStr::width(text);
    let x = area
        .x
        .saturating_add(area.width.saturating_sub(width as u16));
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            text,
            dim_style(theme).add_modifier(Modifier::DIM),
        )))
        .style(Style::default().bg(theme.background)),
        Rect {
            x,
            y: area.y,
            width: area.width.saturating_sub(x.saturating_sub(area.x)),
            height: 1,
        },
    );
}

pub fn render_activity(
    area: Rect,
    label: &str,
    frame_index: usize,
    theme: Theme,
    frame: &mut Frame<'_>,
) {
    if area.width == 0 || area.height == 0 || ACTIVITY_FRAMES.is_empty() {
        return;
    }
    let marker = ACTIVITY_FRAMES[frame_index % ACTIVITY_FRAMES.len()];
    let line = Line::from(vec![
        Span::styled(format!("{marker} "), Style::default().fg(theme.accent)),
        Span::styled(label, dim_style(theme).add_modifier(Modifier::DIM)),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(theme.background)),
        area,
    );
}

pub fn input_block(theme: Theme, focused: bool) -> Block<'static> {
    let border = if focused { theme.focus } else { theme.dim_text };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .padding(Padding::horizontal(1))
}

#[derive(Clone, Debug)]
pub struct PromptLayout {
    pub lines: Vec<Line<'static>>,
    pub cursor_row: usize,
    pub cursor_col: usize,
}

fn prompt_push_char(line: &mut Vec<Span<'static>>, character: char, style: Style) {
    push_span(line, character.to_string(), style);
}

/// Convert the logical textarea into wrapped visual rows and calculate the
/// cursor's visual location. The cursor is painted as a reversed cell rather
/// than relying on the textarea widget's horizontal-scrolling renderer.
pub fn prompt_layout(textarea: &TextArea<'_>, width: usize, theme: Theme) -> PromptLayout {
    let width = width.max(1);
    let (cursor_line, cursor_col) = textarea.cursor();
    let mut lines = Vec::new();
    let mut cursor_row = 0;
    let mut cursor_visual_col = 0;

    for (line_index, source) in textarea.lines().iter().enumerate() {
        let characters = source.chars().collect::<Vec<_>>();
        let mut current = Vec::<Span<'static>>::new();
        let mut current_width = 0usize;
        let mut row_index = lines.len();
        let normal = primary_style(theme);

        if line_index == cursor_line && cursor_col == 0 {
            cursor_row = row_index;
            cursor_visual_col = 0;
        }

        for (character_index, character) in characters.iter().copied().enumerate() {
            let character_width = UnicodeWidthChar::width(character).unwrap_or(1).max(1);
            if current_width > 0 && current_width + character_width > width {
                lines.push(Line::from(std::mem::take(&mut current)));
                current_width = 0;
                row_index = lines.len();
                if line_index == cursor_line && cursor_col == character_index {
                    cursor_row = row_index;
                    cursor_visual_col = 0;
                }
            }
            if line_index == cursor_line && cursor_col == character_index {
                cursor_row = row_index;
                cursor_visual_col = current_width;
            }
            prompt_push_char(&mut current, character, normal);
            current_width = current_width.saturating_add(character_width);
        }

        if line_index == cursor_line && cursor_col >= characters.len() {
            if current_width >= width && !current.is_empty() {
                lines.push(Line::from(std::mem::take(&mut current)));
                row_index = lines.len();
                current_width = 0;
            }
            cursor_row = row_index;
            cursor_visual_col = current_width;
        }

        if current.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(current));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }

    // Paint the cursor cell. At the end of a line, append a space so the
    // cursor remains visible even when there is no character to reverse.
    if let Some(line) = lines.get_mut(cursor_row) {
        let mut rebuilt = Vec::new();
        let mut position = 0usize;
        let mut painted = false;
        for span in &line.spans {
            for character in span.content.chars() {
                let character_width = UnicodeWidthChar::width(character).unwrap_or(1).max(1);
                let style = if !painted && position == cursor_visual_col {
                    painted = true;
                    span.style.bg(theme.focus).fg(theme.background)
                } else {
                    span.style
                };
                prompt_push_char(&mut rebuilt, character, style);
                position = position.saturating_add(character_width);
            }
        }
        if !painted && position == cursor_visual_col {
            push_span(
                &mut rebuilt,
                " ",
                Style::default().bg(theme.focus).fg(theme.background),
            );
        }
        line.spans = rebuilt;
    }

    PromptLayout {
        lines,
        cursor_row,
        cursor_col: cursor_visual_col,
    }
}

/// Return the prompt scroll offset that keeps the cursor row visible inside a
/// prompt body with `inner_height` visible rows.
///
/// The cursor can sit on its own fresh visual row when the previous row
/// filled the width exactly; this helper guarantees that row is painted even
/// for a one-row prompt area.
pub fn prompt_scroll_for_cursor(
    cursor_row: usize,
    line_count: usize,
    inner_height: usize,
) -> usize {
    let inner_height = inner_height.max(1);
    let max_scroll = line_count.saturating_sub(inner_height);
    if cursor_row < inner_height {
        0
    } else {
        (cursor_row + 1 - inner_height).min(max_scroll)
    }
}

pub fn render_prompt(
    area: Rect,
    textarea: &TextArea<'_>,
    scroll_top: usize,
    theme: Theme,
    frame: &mut Frame<'_>,
) -> PromptLayout {
    let block = input_block(theme, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return PromptLayout {
            lines: Vec::new(),
            cursor_row: 0,
            cursor_col: 0,
        };
    }

    let mut layout = prompt_layout(textarea, inner.width as usize, theme);
    if textarea.is_empty() {
        let placeholder = textarea.placeholder_text();
        let mut spans = vec![Span::styled(
            " ",
            Style::default().bg(theme.focus).fg(theme.background),
        )];
        spans.push(Span::styled(
            placeholder.to_owned(),
            dim_style(theme).add_modifier(Modifier::DIM),
        ));
        layout.lines = vec![Line::from(spans)];
        layout.cursor_row = 0;
        layout.cursor_col = 0;
    }

    let start = scroll_top.min(layout.lines.len().saturating_sub(inner.height as usize));
    let visible = layout
        .lines
        .iter()
        .skip(start)
        .take(inner.height as usize)
        .cloned()
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(Text::from(visible)), inner);
    layout
}

pub fn completion_rows(candidates: &[Candidate]) -> u16 {
    candidates.len().min(MAX_COMPLETION_ROWS) as u16
}

pub fn render_completion(
    area: Rect,
    candidates: &[Candidate],
    selected: usize,
    offset: usize,
    theme: Theme,
    frame: &mut Frame<'_>,
) {
    if area.width == 0 || area.height == 0 || candidates.is_empty() {
        return;
    }
    let visible = candidates
        .iter()
        .enumerate()
        .skip(offset)
        .take(area.height as usize)
        .collect::<Vec<_>>();
    let value_width = candidates
        .iter()
        .map(|candidate| UnicodeWidthStr::width(candidate.value.as_str()))
        .max()
        .unwrap_or(0);
    let mut lines = Vec::new();
    for (index, candidate) in visible {
        let selected = index == selected;
        let value_style = Style::default()
            .fg(if selected {
                theme.focus
            } else {
                theme.primary_text
            })
            .add_modifier(if selected {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });
        let description_style = Style::default().fg(theme.muted_text);
        let padding =
            value_width.saturating_sub(UnicodeWidthStr::width(candidate.value.as_str())) + 2;
        let used = value_width + padding + UnicodeWidthStr::width(candidate.description.as_str());
        let mut spans = vec![
            Span::styled(candidate.value.clone(), value_style),
            Span::styled(" ".repeat(padding), description_style),
            Span::styled(candidate.description.clone(), description_style),
        ];
        if used < area.width as usize {
            spans.push(Span::styled(
                " ".repeat(area.width as usize - used),
                Style::default(),
            ));
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ToolStatus, TranscriptEntry};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use tui_textarea::TextArea;

    fn record(status: ToolStatus) -> ToolRecord {
        ToolRecord {
            name: "bash".into(),
            args: "{\"command\":\"cargo test\"}".into(),
            summary: "bash: cargo test".into(),
            ok: !matches!(status, ToolStatus::Failure),
            duration_ms: 1_200,
            output: "first\nsecond\nthird\nfourth\nfifth".into(),
            error: matches!(status, ToolStatus::Failure).then(|| "failed".into()),
            status,
        }
    }

    #[test]
    fn markdown_preserves_formatting_styles() {
        let text = owned_markdown("**bold** *italic* `code`", Theme::default());
        let styles = text
            .lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.style)
            .collect::<Vec<_>>();
        assert!(
            styles
                .iter()
                .any(|style| style.add_modifier.contains(Modifier::BOLD))
        );
        assert!(
            styles
                .iter()
                .any(|style| style.add_modifier.contains(Modifier::ITALIC))
        );
        assert!(
            styles
                .iter()
                .any(|style| style.bg == Some(Theme::default().code_background))
        );
    }

    #[test]
    fn reasoning_lines_keep_continuations_aligned_with_the_first_line() {
        let lines = reasoning_lines("first\nsecond", Theme::default(), 40);
        let values = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec!["first", "second"]);
    }

    #[test]
    fn message_blocks_use_consistent_prefixes_on_wrapped_lines() {
        let user = user_lines("I can still", Theme::default(), 8);
        let assistant = markdown_lines("I can still", Theme::default(), 8);
        let user_values = user
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let assistant_values = assistant
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert_eq!(user_values, vec!["› I can", "  still"]);
        assert_eq!(assistant_values, vec!["‹ I can", "  still"]);
    }

    #[test]
    fn collapsed_tools_show_the_tail_and_hint() {
        let lines = tool_box_lines(&record(ToolStatus::Success), false, 50, Theme::default());
        let value = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(value.contains("fifth"));
        assert!(!value.contains("first"));
        assert!(value.contains("ctrl + o to expand"));
    }

    #[test]
    fn tool_border_color_communicates_state() {
        let theme = Theme::default();
        assert_eq!(
            tool_border_style(&record(ToolStatus::Running), theme).fg,
            Some(theme.tool_running_border)
        );
        assert_eq!(
            tool_border_style(&record(ToolStatus::Success), theme).fg,
            Some(theme.tool_success_border)
        );
        assert_eq!(
            tool_border_style(&record(ToolStatus::Failure), theme).fg,
            Some(theme.tool_failure_border)
        );
    }

    #[test]
    fn prompt_wraps_and_keeps_cursor_visible() {
        let mut textarea = TextArea::from(["abcdefgh"]);
        textarea.move_cursor(tui_textarea::CursorMove::End);
        let layout = prompt_layout(&textarea, 4, Theme::default());
        assert!(layout.lines.len() >= 2);
        assert_eq!(layout.cursor_row, 2);
        assert_eq!(layout.cursor_col, 0);
    }

    #[test]
    fn prompt_cursor_stays_visible_in_a_one_row_prompt_when_a_word_wraps() {
        // A single word that exactly fills each wrapped row places the cursor
        // on a fresh (initially empty) visual row after the wrap boundary. The
        // layout must still scroll a 1-row prompt so that row is painted.
        let mut textarea = TextArea::from(["abcdefgh"]);
        textarea.move_cursor(tui_textarea::CursorMove::End);
        let layout = prompt_layout(&textarea, 4, Theme::default());
        assert_eq!(layout.lines.len(), 3);
        assert_eq!(layout.cursor_row, 2);
        assert_eq!(layout.cursor_col, 0);

        let inner_height = 1; // a one-row prompt body
        let scroll = prompt_scroll_for_cursor(layout.cursor_row, layout.lines.len(), inner_height);
        assert!(
            layout.cursor_row >= scroll && layout.cursor_row < scroll + inner_height,
            "cursor row {} outside visible window [{}, {})",
            layout.cursor_row,
            scroll,
            scroll + inner_height
        );
        assert_eq!(scroll, 2);
    }

    #[test]
    fn prompt_scroll_helper_keeps_cursor_in_window() {
        let cases = [
            // (cursor_row, line_count, inner_height)
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (5, 20, 4),
            (19, 20, 4),
            (0, 10, 3),
        ];
        for (cursor_row, line_count, inner_height) in cases {
            let scroll = prompt_scroll_for_cursor(cursor_row, line_count, inner_height);
            assert!(
                cursor_row >= scroll && cursor_row < scroll + inner_height.max(1),
                "cursor {cursor_row} outside window [{scroll}, {})",
                scroll + inner_height.max(1)
            );
            assert!(scroll <= line_count.saturating_sub(inner_height.max(1)));
        }
    }

    #[test]
    fn activity_indicator_renders_the_current_frame_and_label() {
        let backend = TestBackend::new(24, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_activity(frame.area(), "Reasoning...", 3, Theme::default(), frame);
            })
            .unwrap();
        assert_eq!(terminal.backend().buffer()[(0, 0)].symbol(), "●");
        assert_eq!(terminal.backend().buffer()[(2, 0)].symbol(), "R");
    }

    #[test]
    fn transcript_can_render_into_a_test_backend() {
        let entries = vec![TranscriptEntry::User {
            id: 1,
            text: "hello".into(),
        }];
        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_transcript(frame.area(), &entries, false, 0, Theme::default(), frame);
            })
            .unwrap();
        assert_eq!(terminal.backend().buffer()[(0, 0)].symbol(), "›");
    }
}
