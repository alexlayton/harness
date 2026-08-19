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

// ---------------------------------------------------------------------------
// Spacing design system
//
// Every blank row and blank column the UI inserts comes from here so that
// committed scrollback, the live viewport, and separators share one rhythm:
//
// - `horizontal_pad`/`vertical_pad` define the gutter around content. The
//   live viewport applies them as a `Margin`; committed lines are wrapped at
//   `content_width` and then prefixed with the same gutter by `indent_lines`,
//   so text keeps its wrap when it moves from the live tail into scrollback.
// - `SECTION_GAP` is the blank-line count between transcript entries — one
//   uniform gap between any two entries, tool boxes included.
// - `BLOCK_GAP` is the blank-line count between blocks *inside* one entry
//   (reasoning → markdown).
// ---------------------------------------------------------------------------

/// Blank lines between transcript entries. Tool boxes rely on this same gap
/// rather than adding their own, so the rhythm between entries is uniform.
pub const SECTION_GAP: usize = 1;
/// Blank lines between blocks within a single entry (reasoning → markdown).
pub const BLOCK_GAP: usize = 1;
/// Rows of collapsed tool output kept in a tool box.
pub const DEFAULT_TAIL_LINES: usize = 4;

/// Columns of gutter on each side of content, live and committed alike.
pub fn horizontal_pad(width: u16) -> u16 {
    if width >= 80 {
        2
    } else if width >= 40 {
        1
    } else {
        0
    }
}

/// Blank rows above and below the live viewport's sections.
pub fn vertical_pad(height: u16) -> u16 {
    if height >= 8 { 1 } else { 0 }
}

/// The width content is wrapped at once the gutter is reserved on both sides.
pub(crate) fn content_width(width: u16) -> usize {
    width.saturating_sub(2 * horizontal_pad(width)).max(1) as usize
}

/// Prefix every non-blank line with `pad` spaces so committed scrollback
/// keeps the same gutter the live viewport draws with `Margin`. Blank
/// separator lines stay zero-width.
pub(crate) fn indent_lines(lines: &mut [Line<'static>], pad: u16) {
    if pad == 0 {
        return;
    }
    let gutter = " ".repeat(pad as usize);
    for line in lines.iter_mut() {
        let blank = line
            .spans
            .iter()
            .all(|span| span.content.as_ref().trim().is_empty());
        if !blank {
            line.spans.insert(0, Span::raw(gutter.clone()));
        }
    }
}

/// Key descriptions shown to users. Rendered inside a startup welcome banner
/// committed into scrollback at first draw; kept here so the keymap cannot
/// drift from the input handling.
pub const KEYMAP: &[(&str, &str)] = &[
    ("Enter", "Submit prompt"),
    ("Shift+Enter", "Insert newline"),
    ("↑ / ↓", "Prompt history (empty prompt)"),
    ("Tab", "Focus running tool"),
    ("Ctrl+O", "Expand / collapse running tool"),
    ("Ctrl+C", "Cancel turn / quit"),
    ("Esc", "Close transient UI"),
    ("/", "Commands"),
    ("@", "File references"),
];

/// The startup banner committed into scrollback on first draw (once).
/// Inclusive of the ASCII title and the keymap so it mirrors what the input
/// handler accepts; afterwards it is immutable scrollback like any other
/// committed content.
pub(crate) fn welcome_lines(width: usize, theme: Theme) -> Vec<Line<'static>> {
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

pub(crate) fn push_blank(lines: &mut Vec<Line<'static>>, count: usize) {
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
///
/// The wrapping logic deliberately lives here rather than in ratatui.  ratatui
/// 0.29 keeps its styled reflow machinery private (`widgets::reflow` is a
/// private `mod`, exposing `WordWrapper`/`LineComposer` only internally), and
/// `Paragraph::line_count`/`wrap` are gated behind
/// `#[instability::unstable(feature = "rendered-line-info")]` with the design
/// explicitly marked "not stable".  A hand-rolled wrapper is also required
/// because the transcript renderer needs the wrapped `Line`s up front to
/// compute scroll heights, not at render time.
///
/// The break logic is pinned to `textwrap`'s greedy first-fit algorithm by a
/// differential test; the intentional differences are whitespace handling:
/// leading whitespace is preserved (and can occupy its own row), internal
/// whitespace runs are preserved, and trailing whitespace is dropped at a
/// wrap boundary or the end of the line.
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
    // The surrounding decorations are `╭─ ` (3 columns), a separating space,
    // and `╮` (1 column): 5 columns in total. Reserve them before deciding how
    // much of the title fits, so the rendered line never exceeds `width`.
    let available = width.saturating_sub(5);
    let mut title = title.to_owned();
    while UnicodeWidthStr::width(title.as_str()) > available {
        title.pop();
    }
    let title_width = UnicodeWidthStr::width(title.as_str());
    let fill = width.saturating_sub(title_width + 5);
    Line::from(vec![
        Span::styled("╭─ ", border),
        Span::styled(title, title_style),
        Span::styled(format!(" {}╮", "─".repeat(fill)), border),
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
            line_with_style("call", tool_header_style(theme)),
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

/// Build the wrapped rows for a single transcript entry at the current width.
/// `expanded` drives tool-box rendering: the commit path always passes
/// `false`, the live viewport passes the entry's own expansion state. This is
/// the one builder both the commit pipeline and the live tail share, so a
/// finalized entry commits into scrollback with the same pixels it rendered
/// live (committed collapsed, tools included).
pub fn entry_lines(
    entry: &TranscriptEntry,
    expanded: bool,
    width: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    match entry {
        TranscriptEntry::User { text, .. } => lines.extend(user_lines(text, theme, width)),
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
        TranscriptEntry::Tool { record, .. } => {
            // No extra gap here: entries are separated uniformly by
            // `SECTION_GAP`, so tool boxes do not double the padding.
            lines.extend(tool_box_lines(record, expanded, width, theme));
        }
        TranscriptEntry::Notice { text, .. } => lines.extend(notice_lines(text, theme, width)),
        TranscriptEntry::Error { text, .. } => lines.extend(error_lines(text, theme, width)),
    }
    lines
}

/// Paint the live tail: the newest rows of `lines`, keeping the bottom of the
/// content visible while it grows. The terminal's own scrollback holds the
/// committed history, so there is no user scrolling in the live region.
pub fn render_tail(area: Rect, lines: &[Line<'static>], theme: Theme, frame: &mut Frame<'_>) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let visible_rows = area.height as usize;
    let offset = lines.len().saturating_sub(visible_rows) as u16;
    frame.render_widget(
        Block::default().style(Style::default().bg(theme.background)),
        area,
    );
    frame.render_widget(
        Paragraph::new(Text::from(Vec::from(lines))).scroll((offset, 0)),
        area,
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

/// The input-box border. `hidden_above`/`hidden_below` are rows scrolled out
/// of the visible window on each side; when nonzero the adjacent border gains
/// a dim `+N` title (exactly `+N` — no arrow, no "more").
pub fn input_block(
    theme: Theme,
    focused: bool,
    hidden_above: usize,
    hidden_below: usize,
) -> Block<'static> {
    let border = if focused { theme.focus } else { theme.dim_text };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .padding(Padding::horizontal(1));
    if hidden_above > 0 {
        block = block.title_top(Line::from(Span::styled(
            format!("+{hidden_above}"),
            dim_style(theme),
        )));
    }
    if hidden_below > 0 {
        block = block.title_bottom(Line::from(Span::styled(
            format!("+{hidden_below}"),
            dim_style(theme),
        )));
    }
    block
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

/// Rows hidden *below* the visible window inside a scrollable prompt body
/// with `inner_height` visible rows at scroll offset `scroll_top`. Shown as
/// `+N` on the bottom border of the input box. For a scroll within bounds
/// (`scroll_top <= line_count - inner_height`) this is exactly
/// `line_count - scroll_top - inner_height`; saturating arithmetic keeps it
/// defined for any caller-provided offset.
pub fn prompt_hidden_below(line_count: usize, scroll_top: usize, inner_height: usize) -> usize {
    line_count.saturating_sub(scroll_top + inner_height)
}

pub fn render_prompt(
    area: Rect,
    textarea: &TextArea<'_>,
    scroll_top: usize,
    hidden_above: usize,
    hidden_below: usize,
    theme: Theme,
    frame: &mut Frame<'_>,
) -> PromptLayout {
    let block = input_block(theme, true, hidden_above, hidden_below);
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
    use proptest::prelude::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::widgets::Widget;
    use tui_textarea::TextArea;

    fn record(status: ToolStatus) -> ToolRecord {
        ToolRecord {
            name: "bash".into(),
            args: "bash: cargo test (timeout 30s)".into(),
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
        assert!(!value.contains("all"));
    }

    #[test]
    fn expanded_tools_show_the_human_recap_not_raw_json() {
        let lines = tool_box_lines(&record(ToolStatus::Success), true, 60, Theme::default());
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
        assert!(value.contains("call"));
        assert!(value.contains("bash: cargo test (timeout 30s)"));
        assert!(!value.contains("\"command\""));
        assert!(value.contains("completed in 1.2s"));
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
    fn input_block_shows_plus_n_titles_when_rows_are_hidden() {
        let theme = Theme::default();

        // No hidden rows: no `+N` on either border.
        let backend = TestBackend::new(12, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                input_block(theme, true, 0, 0).render(frame.area(), frame.buffer_mut());
            })
            .unwrap();
        let top = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .take(12)
            .map(|cell| cell.symbol())
            .collect::<String>();
        let bottom = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .skip(24)
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(!top.contains('+'));
        assert!(!bottom.contains('+'));
    }

    #[test]
    fn input_block_shows_top_and_bottom_plus_n() {
        let theme = Theme::default();
        let backend = TestBackend::new(12, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                input_block(theme, true, 3, 2).render(frame.area(), frame.buffer_mut());
            })
            .unwrap();
        let content = terminal.backend().buffer().content();
        let top: String = content.iter().take(12).map(|cell| cell.symbol()).collect();
        let bottom: String = content.iter().skip(24).map(|cell| cell.symbol()).collect();
        assert!(top.contains("+3"), "top border should show +3, got {top:?}");
        assert!(
            bottom.contains("+2"),
            "bottom border should show +2, got {bottom:?}"
        );

        // Rendered titles are dim.
        let dim_on_top = content.iter().take(12).any(|cell| {
            cell.symbol() == "+" && cell.style().fg == Some(theme.dim_text)
                || cell.symbol().starts_with('+') && cell.style().fg == Some(theme.dim_text)
        });
        assert!(dim_on_top, "+N title should be dimmed");
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
    fn prompt_hidden_below_counts_rows_outside_the_visible_window() {
        // Everything below the visible window is hidden; nothing above counts.
        assert_eq!(prompt_hidden_below(10, 0, 4), 6);
        assert_eq!(prompt_hidden_below(10, 3, 4), 3);
        assert_eq!(prompt_hidden_below(10, 6, 4), 0);
        // A scroll offset past the bottom hides nothing below.
        assert_eq!(prompt_hidden_below(10, 7, 4), 0);
        assert_eq!(prompt_hidden_below(10, 12, 4), 0);
        // The window may be taller than the content.
        assert_eq!(prompt_hidden_below(4, 0, 4), 0);
        assert_eq!(prompt_hidden_below(4, 1, 6), 0);
        // Degenerate empty inputs stay defined.
        assert_eq!(prompt_hidden_below(0, 0, 4), 0);
        assert_eq!(prompt_hidden_below(0, 0, 0), 0);
    }

    #[test]
    fn prompt_window_partitions_the_lines() {
        // For any in-bounds scroll, hidden-above + visible + hidden-below
        // partition the prompt lines exactly: no row is lost or shown twice.
        for line_count in 0..=12usize {
            for inner_height in 1..=5usize {
                let max_scroll = line_count.saturating_sub(inner_height);
                for scroll in 0..=max_scroll {
                    let hidden_above = scroll;
                    let hidden_below = prompt_hidden_below(line_count, scroll, inner_height);
                    let visible = inner_height.min(line_count.saturating_sub(scroll));
                    assert_eq!(
                        hidden_above + visible + hidden_below,
                        line_count,
                        "lines {line_count} inner {inner_height} scroll {scroll}"
                    );
                }
            }
        }
    }

    #[test]
    fn render_prompt_paints_plus_n_on_both_borders_when_the_input_overflows() {
        // A 30-line prompt in a 3-row inner box: the cursor forces a scroll
        // that hides rows above *and* below the visible window.
        let theme = Theme::default();
        let mut textarea = TextArea::from((0..30).map(|i| format!("line {i}")).collect::<Vec<_>>());
        // Park the cursor mid-prompt so rows are hidden above *and* below the
        // visible window (a cursor at the end hides rows above only).
        textarea.move_cursor(tui_textarea::CursorMove::Jump(15, 0));
        let layout = prompt_layout(&textarea, 20, theme);
        let inner_height = 3usize;
        let scroll = prompt_scroll_for_cursor(layout.cursor_row, layout.lines.len(), inner_height);
        let hidden_above = scroll;
        let hidden_below = prompt_hidden_below(layout.lines.len(), scroll, inner_height);
        assert!(hidden_above > 0, "expected rows hidden above the window");
        assert!(hidden_below > 0, "expected rows hidden below the window");

        let backend = TestBackend::new(40, 7);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_prompt(
                    frame.area(),
                    &textarea,
                    scroll,
                    hidden_above,
                    hidden_below,
                    theme,
                    frame,
                );
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        let top: String = buffer
            .content()
            .iter()
            .take(40)
            .map(|c| c.symbol())
            .collect();
        let bottom: String = buffer
            .content()
            .iter()
            .skip(40 * 6)
            .map(|c| c.symbol())
            .collect();
        assert!(
            top.contains(&format!("+{hidden_above}")),
            "top border should show +{hidden_above}, got {top:?}"
        );
        assert!(
            bottom.contains(&format!("+{hidden_below}")),
            "bottom border should show +{hidden_below}, got {bottom:?}"
        );
        // The visible window shows exactly the scrolled slice of the prompt:
        // rows 1..=3 of the buffer are the 3 visible text rows, each starting
        // at column 2 (border `│`, padding space, then the line).
        let visible_row_text = |buffer_row: usize| -> String {
            buffer
                .content()
                .iter()
                .skip(40 * buffer_row + 2)
                .take(20)
                .map(|c| c.symbol())
                .collect::<String>()
                .trim_end()
                .to_owned()
        };
        for (offset, buffer_row) in (1..=3).enumerate() {
            let expected: String = layout.lines[scroll + offset]
                .spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect();
            assert_eq!(
                visible_row_text(buffer_row),
                expected,
                "visible row at buffer row {buffer_row}"
            );
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
    fn entry_lines_never_exceed_width() {
        let long_command = format!("bash: {}", "x".repeat(500));
        let entries = [
            TranscriptEntry::User {
                id: 1,
                text: "a ".repeat(200),
            },
            TranscriptEntry::Assistant {
                id: 2,
                markdown: "**bold** `code` ".repeat(60),
                reasoning: "thinking ".repeat(80),
                streaming: false,
            },
            TranscriptEntry::Tool {
                id: 3,
                record: record(ToolStatus::Success),
                expanded: true,
            },
            TranscriptEntry::Tool {
                id: 4,
                record: ToolRecord {
                    summary: long_command,
                    ..record(ToolStatus::Success)
                },
                expanded: false,
            },
            TranscriptEntry::Notice {
                id: 5,
                text: "notice ".repeat(100),
            },
            TranscriptEntry::Error {
                id: 6,
                text: "error ".repeat(100),
            },
        ];
        for width in 8usize..=140 {
            for (index, entry) in entries.iter().enumerate() {
                let lines = entry_lines(entry, false, width, Theme::default());
                for line in &lines {
                    let w = line_width(line);
                    assert!(
                        w <= width,
                        "entry {index} line is {w} wide but width is {width}: {}",
                        span_contents(line)
                    );
                }
            }
        }
    }

    #[test]
    fn content_width_reserves_the_gutter_on_both_sides() {
        assert_eq!(content_width(80), 76);
        assert_eq!(content_width(100), 96);
        assert_eq!(content_width(79), 77);
        assert_eq!(content_width(40), 38);
        assert_eq!(content_width(39), 39);
        assert_eq!(content_width(20), 20);
        assert_eq!(content_width(2), 2);
    }

    #[test]
    fn indent_lines_prefixes_content_but_not_blank_separators() {
        let mut lines = vec![
            line_with_style("hello", Style::default()),
            blank_line(),
            Line::from(vec![Span::raw("a"), Span::raw("b")]),
        ];
        indent_lines(&mut lines, 2);
        let values: Vec<String> = lines.iter().map(span_contents).collect();
        assert_eq!(values, vec!["  hello", "", "  ab"]);

        // A zero pad is a no-op.
        indent_lines(&mut lines, 0);
        assert_eq!(span_contents(&lines[0]), "  hello");
    }

    #[test]
    fn committed_lines_wrapped_at_content_width_stay_within_the_terminal() {
        // entry_lines wraps at content_width; adding the gutter must never
        // push a committed line past the real terminal width.
        let entries = [
            TranscriptEntry::User {
                id: 1,
                text: "a ".repeat(100),
            },
            TranscriptEntry::Assistant {
                id: 2,
                markdown: "**bold** `code` ".repeat(40),
                reasoning: "thinking ".repeat(60),
                streaming: false,
            },
            TranscriptEntry::Tool {
                id: 3,
                record: record(ToolStatus::Success),
                expanded: false,
            },
            TranscriptEntry::Notice {
                id: 4,
                text: "notice ".repeat(60),
            },
        ];
        for width in [20u16, 40, 60, 80, 120] {
            let pad = horizontal_pad(width);
            for entry in &entries {
                let mut lines = entry_lines(entry, false, content_width(width), Theme::default());
                indent_lines(&mut lines, pad);
                for line in &lines {
                    let used = line_width(line);
                    assert!(
                        used <= width as usize,
                        "line is {used} wide but terminal is {width}: {}",
                        span_contents(line)
                    );
                }
            }
        }
    }

    fn span_contents(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    fn wrapped_values(text: &str, width: usize) -> Vec<String> {
        wrap_text(&Text::from(text), width, Style::default())
            .iter()
            .map(span_contents)
            .collect()
    }

    #[test]
    fn wrap_text_preserves_leading_and_internal_whitespace_but_drops_trailing() {
        // Leading and internal runs survive; the trailing space is dropped.
        let lines = wrap_text(&Text::from("  a  b "), 40, Style::default());
        assert_eq!(span_contents(&lines[0]), "  a  b");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn wrap_text_drops_trailing_whitespace_at_boundaries_and_line_end() {
        // A pending space before a word that does not fit is dropped at the
        // wrap boundary, and trailing whitespace never starts a new row.
        assert_eq!(wrapped_values("hello world", 6), vec!["hello", "world"]);
        assert_eq!(wrapped_values("hello ", 20), vec!["hello"]);
    }

    #[test]
    fn wrap_text_treats_tabs_as_whitespace() {
        // A tab is whitespace: it separates words and is dropped at a wrap
        // boundary like any other pending whitespace.
        assert_eq!(wrapped_values("a\tb", 2), vec!["a", "b"]);
    }

    #[test]
    fn wrap_text_breaks_overwide_words_by_display_width() {
        assert_eq!(wrapped_values("abcdefgh", 4), vec!["abcd", "efgh"]);
    }

    #[test]
    fn wrap_text_uses_display_width_for_cjk_breaks() {
        // Each CJK char is 2 columns wide, so 3 chars fit in a 6-column line.
        assert_eq!(wrapped_values("日本語日本語", 6), vec!["日本語", "日本語"]);
    }

    #[test]
    fn wrap_text_never_loses_characters_of_unbroken_input() {
        // ZWJ emoji sequences, combining marks, and CJK are broken at the
        // character level by this wrapper; the round-trip property holds
        // regardless of where the breaks land.
        for input in ["👨‍👩‍👧‍👦", "café\u{301}", "日本語テキスト", "aeiou"]
        {
            let lines = wrap_text(&Text::from(input), 3, Style::default());
            let joined: String = lines.iter().map(span_contents).collect();
            assert_eq!(joined, input, "round-trip failed for {input:?}");
        }
    }

    proptest! {
        /// Differential test against textwrap's greedy first-fit wrapping.
        /// ratatui's reflow machinery is private, so this pins the custom
        /// wrapper's break positions to a battle-tested reference.  The domain
        /// is restricted to single-space-separated words (no leading or
        /// trailing whitespace) because whitespace handling is an intentional,
        /// separately-tested difference.
        #[test]
        fn wrap_text_break_positions_match_textwrap(
            words in proptest::collection::vec("[a-z]{1,6}", 1..10),
            width in 1usize..24,
        ) {
            let input = words.join(" ");
            let ours = wrapped_values(&input, width);

            let reference = textwrap::wrap(
                &input,
                textwrap::Options::new(width)
                    .break_words(true)
                    .wrap_algorithm(textwrap::WrapAlgorithm::FirstFit)
                    .word_separator(textwrap::WordSeparator::AsciiSpace),
            );

            // textwrap never emits trailing whitespace; ours drops pending
            // whitespace at boundaries too, so trim defensively on both sides.
            let ours: Vec<String> = ours.iter().map(|line| line.trim_end().to_owned()).collect();
            let reference: Vec<String> =
                reference.iter().map(|line| line.trim_end().to_owned()).collect();
            prop_assert_eq!(ours, reference, "input {:?} at width {}", input, width);
        }
    }
}
