use ratatui_core::style::{Color, Modifier, Style};
use ratatui_core::text::{Line, Span, Text};
use tui_markdown::{AlertKind, Options, StyleSheet, from_str_with_options};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// The semantic palette used by every renderer. Keeping these roles together
/// prevents individual widgets from slowly acquiring unrelated colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    pub primary_text: Color,
    pub assistant_text: Color,
    pub muted_text: Color,
    pub dim_text: Color,
    pub accent: Color,
    pub code_background: Color,
    /// Success colour shared by tool durations and markdown `TIP` alerts.
    pub success: Color,
    pub error: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            // Use the terminal's default colours for most backgrounds/text so
            // the UI respects the user's terminal theme instead of imposing a
            // dark background everywhere.
            primary_text: Color::Reset,
            assistant_text: Color::Reset,
            muted_text: Color::Reset,
            dim_text: Color::Reset,
            // ANSI palette index 2 follows the terminal's configured green
            // slot rather than imposing an RGB colour of our own.
            accent: Color::Indexed(2),
            code_background: Color::Reset,
            success: Color::Green,
            error: Color::Red,
        }
    }
}

pub(crate) const ACTIVITY_FRAMES: &[&str] = &["·", "∙", "•", "●", "•", "∙"];
const USER_PREFIX: &str = "› ";
const ASSISTANT_PREFIX: &str = "‹ ";

// ---------------------------------------------------------------------------
// Spacing design system
//
// Every blank row and blank column the UI inserts comes from here so that
// committed scrollback, the live viewport, and separators share one rhythm:
//
// - `horizontal_pad` defines the gutter around content. The live region
//   applies it as a left margin and committed rows are written behind the
//   same number of spaces, so text keeps its position when it moves from
//   the live tail into scrollback.
// - `SECTION_GAP` is the blank-line count between transcript entries — one
//   uniform gap between any two entries, tool lines included.
// - `BLOCK_GAP` is the blank-line count between blocks *inside* one entry
//   (reasoning → markdown).
// ---------------------------------------------------------------------------

/// Blank lines between transcript entries. Tool lines rely on this same gap
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

/// The width content is wrapped at once the gutter is reserved on both sides.
pub(crate) fn content_width(width: u16) -> usize {
    width.saturating_sub(2 * horizontal_pad(width)).max(1) as usize
}

// The startup wordmarks are embedded rather than read from a workspace file:
// installed binaries should have the same welcome screen regardless of cwd.
// Each inner slice is one font from `headers.txt`, plus the original wordmark.
const WELCOME_TITLES: &[&[&str]] = &[
    &[
        "██  ██ ░▒▀▀██ ██▀▀██ ██▀▀██ ██▀▀▒░ ▒▓▀▀██ ▒▓▀▀██",
        "██▀▀██ ▒▓  ██ ██     ██  ██ ██▄▄▓▒ ▓█▄▄▄▄ ▓█▄▄▄▄",
        "██  ██ ▓█▀▀██ ██     ██  ██ ██▄▄▄▄ ▄▄  ▒▒ ▄▄  ▒▒",
        "       ▀▀                          ▀▀▀▀▀▀ ▀▀▀▀▀▀",
    ],
    &[
        " ▄  █ ██   █▄▄▄▄   ▄   ▄███▄     ▄▄▄▄▄    ▄▄▄▄▄",
        "█   █ █ █  █  ▄▀    █  █▀   ▀   █     ▀▄ █     ▀▄",
        "██▀▀█ █▄▄█ █▀▀▌ ██   █ ██▄▄   ▄  ▀▀▀▀▄ ▄  ▀▀▀▀▄",
        "█   █ █  █ █  █ █ █  █ █▄   ▄▀ ▀▄▄▄▄▀   ▀▄▄▄▄▀",
        "   █     █   █  █  █ █ ▀███▀",
        "  ▀     █   ▀   █   ██",
        "       ▀",
    ],
    &[
        " ██░ ██  ▄▄▄       ██▀███   ███▄    █ ▓█████   ██████   ██████",
        "▓██░ ██▒▒████▄    ▓██ ▒ ██▒ ██ ▀█   █ ▓█   ▀ ▒██    ▒ ▒██    ▒",
        "▒██▀▀██░▒██  ▀█▄  ▓██ ░▄█ ▒▓██  ▀█ ██▒▒███   ░ ▓██▄   ░ ▓██▄",
        "░▓█ ░██ ░██▄▄▄▄██ ▒██▀▀█▄  ▓██▒  ▐▌██▒▒▓█  ▄   ▒   ██▒  ▒   ██▒",
        "░▓█▒░██▓ ▓█   ▓██▒░██▓ ▒██▒▒██░   ▓██░░▒████▒▒██████▒▒▒██████▒▒",
        " ▒ ░░▒░▒ ▒▒   ▓▒█░░ ▒▓ ░▒▓░░ ▒░   ▒ ▒ ░░ ▒░ ░▒ ▒▓▒ ▒ ░▒ ▒▓▒ ▒ ░",
        " ▒ ░▒░ ░  ▒   ▒▒ ░  ░▒ ░ ▒░░ ░░   ░ ▒░ ░ ░  ░░ ░▒  ░ ░░ ░▒  ░ ░",
        " ░  ░░ ░  ░   ▒     ░░   ░    ░   ░ ░    ░   ░  ░  ░  ░  ░  ░",
        " ░  ░  ░      ░  ░   ░              ░    ░  ░      ░        ░",
    ],
    &[
        "▄█    █▄       ▄████████    ▄████████ ███▄▄▄▄      ▄████████    ▄████████    ▄████████",
        "  ███    ███     ███    ███   ███    ███ ███▀▀▀██▄   ███    ███   ███    ███   ███    ███",
        "  ███    ███     ███    ███   ███    ███ ███   ███   ███    █▀    ███    █▀    ███    █▀",
        " ▄███▄▄▄▄███▄▄   ███    ███  ▄███▄▄▄▄██▀ ███   ███  ▄███▄▄▄       ███          ███",
        "▀▀███▀▀▀▀███▀  ▀███████████ ▀▀███▀▀▀▀▀   ███   ███ ▀▀███▀▀▀     ▀███████████ ▀███████████",
        "  ███    ███     ███    ███ ▀███████████ ███   ███   ███    █▄           ███          ███",
        "  ███    ███     ███    ███   ███    ███ ███   ███   ███    ███    ▄█    ███    ▄█    ███",
        "  ███    █▀      ███    █▀    ███    ███  ▀█   █▀    ██████████  ▄████████▀   ▄████████▀",
        "                              ███    ███",
    ],
    &[
        " ▄ .▄ ▄▄▄· ▄▄▄   ▐ ▄ ▄▄▄ ..▄▄ · .▄▄ ·",
        "██▪▐█▐█ ▀█ ▀▄ █·•█▌▐█▀▄.▀·▐█ ▀. ▐█ ▀.",
        "██▀▐█▄█▀▀█ ▐▀▀▄ ▐█▐▐▌▐▀▀▪▄▄▀▀▀█▄▄▀▀▀█▄",
        "██▌▐▀▐█ ▪▐▌▐█•█▌██▐█▌▐█▄▄▌▐█▄▪▐█▐█▄▪▐█",
        "▀▀▀ · ▀  ▀ .▀  ▀▀▀ █▪ ▀▀▀  ▀▀▀▀  ▀▀▀▀",
    ],
];

/// A launch-randomized priority order. Keeping it in the transcript entry
/// makes resize repaints stable while still allowing a narrower title to take
/// over if the terminal no longer has room for the first choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WelcomeTitleOrder(Vec<usize>);

impl WelcomeTitleOrder {
    pub(crate) fn random() -> Self {
        let mut order = (0..WELCOME_TITLES.len()).collect::<Vec<_>>();
        fastrand::shuffle(&mut order);
        Self(order)
    }

    fn fitting_title(&self, width: usize) -> Option<&'static [&'static str]> {
        self.0
            .iter()
            .map(|index| WELCOME_TITLES[*index])
            .find(|title| title_width(title) <= width)
    }
}

fn title_width(title: &[&str]) -> usize {
    title
        .iter()
        .map(|line| UnicodeWidthStr::width(*line))
        .max()
        .unwrap_or(0)
}

/// The startup banner committed into scrollback on startup.
pub(crate) fn welcome_lines(
    width: usize,
    theme: Theme,
    title_order: &WelcomeTitleOrder,
) -> Vec<Line<'static>> {
    // Same role as the prompt activity marker / message accents.
    let title_style = Style::default().fg(theme.accent);
    let mut lines = Vec::new();
    // The banner opens scrollback immediately below whatever the shell left
    // on screen; give the title the design system's breathing room.
    push_blank(&mut lines, SECTION_GAP);
    if let Some(title) = title_order.fitting_title(width) {
        lines.extend(title.iter().map(|line| line_with_style(*line, title_style)));
    } else {
        lines.push(line_with_style("Harness", title_style));
    }
    // Keep the discoverability footer close to the wordmark, then end the
    // banner so workspace metadata and the transcript continue below.
    push_blank(&mut lines, BLOCK_GAP);
    let footer = format!(
        "v{}  /help for commands · Ctrl+O shows or hides tool details",
        env!("CARGO_PKG_VERSION")
    );
    lines.push(line_with_style(
        fit_single_line(&footer, width),
        muted_style(theme),
    ));
    lines
}

/// Fit fixed chrome onto one terminal row without splitting wide characters.
fn fit_single_line(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    let budget = width.saturating_sub(1);
    let mut result = String::new();
    let mut used = 0usize;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(1);
        if used + character_width > budget {
            break;
        }
        result.push(character);
        used += character_width;
    }
    result.push('…');
    result
}

#[derive(Clone, Copy, Debug, Default)]
struct MarkdownTheme {
    theme: Theme,
}

impl StyleSheet for MarkdownTheme {
    fn heading(&self, _level: u8) -> Style {
        fg(self.theme.accent).add_modifier(Modifier::BOLD)
    }

    fn code(&self) -> Style {
        fg(self.theme.primary_text).bg(self.theme.code_background)
    }

    fn link(&self) -> Style {
        fg(self.theme.accent).add_modifier(Modifier::UNDERLINED)
    }

    fn blockquote(&self) -> Style {
        muted_style(self.theme)
    }

    fn heading_meta(&self) -> Style {
        dim_style(self.theme)
    }

    fn metadata_block(&self) -> Style {
        muted_style(self.theme)
    }

    fn html(&self) -> Style {
        dim_style(self.theme)
    }

    fn math_inline(&self) -> Style {
        fg(self.theme.accent).add_modifier(Modifier::ITALIC)
    }

    fn math_display(&self) -> Style {
        fg(self.theme.accent)
    }

    fn table_header(&self) -> Style {
        fg(self.theme.primary_text).add_modifier(Modifier::BOLD)
    }

    fn table_cell(&self) -> Style {
        fg(self.theme.assistant_text)
    }

    fn table_border(&self) -> Style {
        dim_style(self.theme)
    }

    fn image_alt(&self) -> Style {
        dim_style(self.theme).add_modifier(Modifier::ITALIC)
    }

    fn alert(&self, kind: AlertKind) -> Style {
        let color = match kind {
            AlertKind::Note => self.theme.accent,
            AlertKind::Tip => self.theme.success,
            AlertKind::Important => self.theme.accent,
            AlertKind::Warning => Color::Yellow,
            AlertKind::Caution => self.theme.error,
        };
        fg(color)
    }
}

/// Convenience shorthand for the theme styles above.
fn fg(color: Color) -> Style {
    Style::default().fg(color)
}

fn dim_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM)
}

fn muted_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.muted_text)
        .add_modifier(Modifier::DIM)
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

/// Fit a styled row to a terminal-column budget without leaking controls or
/// emitting a wide glyph into a smaller remaining space.
pub(crate) fn fit_line_to_width(line: &Line<'_>, width: usize) -> Line<'static> {
    let mut spans = Vec::new();
    let mut used = 0usize;
    for span in &line.spans {
        let content = sanitize_terminal_text(span.content.as_ref());
        let mut kept = String::new();
        for character in content.chars() {
            let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
            if character_width > 0 && used.saturating_add(character_width) > width {
                continue;
            }
            kept.push(character);
            used = used.saturating_add(character_width);
        }
        if !kept.is_empty() {
            spans.push(Span::styled(kept, span.style));
        }
        if used >= width {
            break;
        }
    }
    Line::from(spans).style(line.style)
}

fn fit_text_to_width(text: &str, width: usize) -> String {
    let line = Line::from(text.to_owned());
    let fitted = fit_line_to_width(&line, width);
    fitted
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
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
    let source_lines = split_embedded_newlines(text);

    for source_line in &source_lines {
        let line_base = base.patch(source_line.style);
        let mut source_chars = Vec::<(char, Style, usize)>::new();
        for source_span in &source_line.spans {
            let style = line_base.patch(source_span.style);
            let content = sanitize_terminal_text(source_span.content.as_ref());
            source_chars.extend(content.chars().map(|character| {
                let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
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
                    // Leading indentation is meaningful, but it still has to
                    // be chunked so a long run cannot overflow a narrow row.
                    for &(character, style, character_width) in group {
                        if character_width > 0
                            && current_width.saturating_add(character_width) > width
                        {
                            result.push(wrapped_line(std::mem::take(&mut current)));
                            current_width = 0;
                        }
                        current.push((character, style, character_width));
                        current_width = current_width.saturating_add(character_width);
                    }
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
                if character_width > width {
                    // A width-two glyph cannot fit a one-column terminal. It
                    // is safer to omit that glyph than to let the terminal
                    // wrap it into an untracked row.
                    if !current.is_empty() {
                        result.push(wrapped_line(std::mem::take(&mut current)));
                        current_width = 0;
                    }
                    continue;
                }
                if character_width > 0 && current_width.saturating_add(character_width) > width {
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

/// Split content-originated newlines into logical rows before wrapping.
/// Ratatui permits a newline inside a span, but ANSI serialization would emit
/// it as a physical terminal row that the caller did not count. Keeping the
/// split here makes height measurement and emission agree for tool paths,
/// metadata, pasted text, and other untrusted single-line values.
fn split_embedded_newlines(text: &Text<'_>) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for source_line in &text.lines {
        let mut spans = Vec::new();
        for source_span in &source_line.spans {
            let content = sanitize_terminal_text(source_span.content.as_ref());
            let parts = content.split('\n').collect::<Vec<_>>();
            for (index, part) in parts.iter().enumerate() {
                if !part.is_empty() {
                    spans.push(Span::styled((*part).to_owned(), source_span.style));
                }
                if index + 1 < parts.len() {
                    lines.push(Line::from(std::mem::take(&mut spans)).style(source_line.style));
                }
            }
        }
        lines.push(Line::from(spans).style(source_line.style));
    }
    lines
}

fn wrapped_line(chars: Vec<(char, Style, usize)>) -> Line<'static> {
    let mut spans = Vec::new();
    for (character, style, _) in chars {
        push_span(&mut spans, character.to_string(), style);
    }
    Line::from(spans)
}

/// Count `\n` bytes with `memchr` (SIMD-accelerated, already in the tree
/// via `ignore`/`regex`): unoptimized builds compile naive per-byte
/// iterators to hundreds of ms on multi-megabyte outputs, while this stays
/// at ~1ms in every profile. Same O(head) complexity, vastly better
/// constant — and the fastest correct tool for one byte search.
fn count_newlines(bytes: &[u8]) -> usize {
    memchr::memchr_iter(b'\n', bytes).count()
}

/// Sanitize untrusted text before it reaches terminal serialization.
///
/// Newlines are intentional display structure and tabs expand to four spaces
/// so measurement and emission agree. C0/C1 controls, DEL, CSI, OSC, and
/// other escape strings are removed; generated styling remains separate in
/// [`line_to_ansi`](crate::app) and is never accepted from content.
pub(crate) fn sanitize_terminal_text(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut result = String::with_capacity(value.len());
    let mut index = 0usize;
    while index < chars.len() {
        let character = chars[index];
        match character {
            '\n' => {
                result.push('\n');
                index += 1;
            }
            '\t' => {
                result.push_str("    ");
                index += 1;
            }
            '\u{1b}' => {
                index += 1;
                skip_escape_sequence(&chars, &mut index);
            }
            '\u{9b}' => {
                index += 1;
                skip_csi_sequence(&chars, &mut index);
            }
            '\u{9d}' | '\u{90}' | '\u{98}' | '\u{9e}' | '\u{9f}' => {
                index += 1;
                skip_string_sequence(&chars, &mut index);
            }
            '\u{9c}' => index += 1,
            character if character.is_control() || character == '\u{7f}' => index += 1,
            character => {
                result.push(character);
                index += 1;
            }
        }
    }
    result
}

/// Skip an ESC-prefixed terminal sequence after its introducer. CSI and
/// string sequences have dedicated parsers because they may contain arbitrary
/// parameters or payload; other ESC sequences terminate at their final byte.
fn skip_escape_sequence(chars: &[char], index: &mut usize) {
    let Some(&introducer) = chars.get(*index) else {
        return;
    };
    match introducer {
        '[' => {
            *index += 1;
            skip_csi_sequence(chars, index);
        }
        ']' | 'P' | 'X' | '^' | '_' => {
            *index += 1;
            skip_string_sequence(chars, index);
        }
        _ => {
            while let Some(&character) = chars.get(*index) {
                *index += 1;
                let code = character as u32;
                if (0x30..=0x7e).contains(&code) {
                    break;
                }
            }
        }
    }
}

fn skip_csi_sequence(chars: &[char], index: &mut usize) {
    while let Some(&character) = chars.get(*index) {
        *index += 1;
        let code = character as u32;
        if (0x40..=0x7e).contains(&code) {
            break;
        }
    }
}

fn skip_string_sequence(chars: &[char], index: &mut usize) {
    while let Some(&character) = chars.get(*index) {
        *index += 1;
        match character {
            '\u{07}' | '\u{9c}' => break,
            '\u{1b}' if chars.get(*index) == Some(&'\\') => {
                *index += 1;
                break;
            }
            _ => {}
        }
    }
}

fn plain_text(value: &str, style: Style) -> Text<'static> {
    let value = sanitize_terminal_text(value);
    Text::from(
        value
            .split('\n')
            .map(|line| line_with_style(line.to_owned(), style))
            .collect::<Vec<_>>(),
    )
}

fn owned_markdown(markdown: &str, theme: Theme) -> Text<'static> {
    let markdown = sanitize_terminal_text(markdown);
    let options = Options::new(MarkdownTheme { theme });
    let rendered = from_str_with_options(&markdown, &options);
    let lines = rendered
        .lines
        .iter()
        .map(|line| {
            let mut owned = Line::from(
                line.spans
                    .iter()
                    .map(|span| Span::styled(span.content.to_string(), span.style))
                    .collect::<Vec<_>>(),
            );
            owned.style = line.style;
            owned
        })
        .collect::<Vec<_>>();
    Text::from(lines)
}

pub(crate) fn prefix_message_lines(
    lines: Vec<Line<'static>>,
    prefix: &str,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let prefix_style = message_prefix_style(theme);
    // A terminal narrower than the normal two-column prefix gets a shortened
    // prefix rather than an over-wide row. Content is fitted after the prefix
    // so this invariant also holds for width 1.
    let prefix = fit_text_to_width(prefix, width);
    let prefix_width = UnicodeWidthStr::width(prefix.as_str());
    let continuation = " ".repeat(prefix_width);
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
                Span::styled(prefix.clone(), prefix_style)
            };
            let content = fit_line_to_width(&line, width.saturating_sub(prefix_width));
            Line::from(
                std::iter::once(prefix)
                    .chain(content.spans)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn message_content_width(width: usize) -> usize {
    width.saturating_sub(UnicodeWidthStr::width(USER_PREFIX))
}

pub(crate) fn reasoning_lines(reasoning: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = plain_text(reasoning, muted_style(theme).add_modifier(Modifier::ITALIC));
    wrap_text(&text, width, Style::default())
}

pub(crate) fn markdown_lines(markdown: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = owned_markdown(markdown, theme);
    prefix_message_lines(
        wrap_text(&text, message_content_width(width), assistant_style(theme)),
        ASSISTANT_PREFIX,
        theme,
        width,
    )
}

pub(crate) fn user_lines(input: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    prefix_message_lines(
        wrap_text(
            &plain_text(input, primary_style(theme)),
            message_content_width(width),
            Style::default(),
        ),
        USER_PREFIX,
        theme,
        width,
    )
}

pub(crate) fn notice_lines(notice: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = plain_text(notice, dim_style(theme));
    wrap_text(&text, width.saturating_sub(2).max(1), Style::default())
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "· " } else { "  " };
            let mut spans = vec![Span::styled(prefix, dim_style(theme))];
            spans.extend(line.spans);
            fit_line_to_width(&Line::from(spans), width)
        })
        .collect()
}

pub(crate) fn error_lines(error: &str, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let text = plain_text(error, error_style(theme));
    wrap_text(&text, width.saturating_sub(2).max(1), Style::default())
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "✗ " } else { "  " };
            let mut spans = vec![Span::styled(prefix, error_style(theme))];
            spans.extend(line.spans);
            fit_line_to_width(&Line::from(spans), width)
        })
        .collect()
}

pub(crate) fn duration_text(duration_ms: u64) -> String {
    if duration_ms >= 1_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{duration_ms}ms")
    }
}

/// Bound a tool's raw output to the collapsed tail rows: the newest
/// `DEFAULT_TAIL_LINES` lines, preceded by one `… N lines above` row when
/// more were produced. Used by the expanded tool rendering.
///
/// Two-pass front walk is gone: the tail is located with `rfind` from the
/// end (O(tail)) and the omitted count is the newline count of the head
/// prefix — one branchless byte scan, no per-line `&str` materialization
/// (the old `lines().count() + lines().skip()` walk allocated every line
/// twice). ~2.5× faster on 100k-line outputs (49ms → 19ms per 20
/// resolutions in release); the remaining head byte scan is the floor for
/// an exact `… N lines above` count. Pinned by
/// `output_tail_scales_with_the_tail_not_the_output`.
pub(crate) fn output_tail(output: &str) -> Vec<String> {
    let mut tail_start = 0usize;
    let mut tail_lines = 0usize;
    let mut cursor = output.len();
    // A trailing newline terminates the last line rather than starting an
    // empty one (`str::lines` semantics); skip it before counting.
    if output.as_bytes().last() == Some(&b'\n') && cursor > 0 {
        cursor -= 1;
    }
    while tail_lines < DEFAULT_TAIL_LINES && cursor > 0 {
        match output[..cursor].rfind('\n') {
            Some(index) => {
                if tail_lines + 1 == DEFAULT_TAIL_LINES {
                    tail_start = index + 1;
                    break;
                }
                tail_lines += 1;
                cursor = index;
            }
            None => {
                tail_start = 0;
                break;
            }
        }
    }
    // Omitted lines without walking the head line-by-line: every `\n`
    // before `tail_start` ends an omitted line. This matches
    // `lines().count() - DEFAULT_TAIL_LINES` exactly (verified by
    // `output_tail_matches_front_anchored_semantics`).
    //
    // Counting uses `memchr`-style slicing (`chunks_exact(64KB)`) instead
    // of a per-byte iterator: unoptimized builds compile the naive
    // `filter(== b'\n')` to a ~400ms walk on 100k lines, while chunked
    // counting stays fast everywhere (~1ms). Both are O(head) bytes, but
    // the constant decides whether the quadratic guard below is green.
    let head = &output[..tail_start];
    let omitted = if head.is_empty() {
        0
    } else {
        count_newlines(head.as_bytes())
    };
    let mut result = if omitted > 0 {
        vec![format!("… {omitted} lines above")]
    } else {
        Vec::new()
    };
    result.extend(output[tail_start..].lines().map(str::to_owned));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn output_tail_keeps_only_the_bounded_suffix() {
        assert_eq!(output_tail("one\ntwo"), vec!["one", "two"]);
        assert_eq!(
            output_tail("one\ntwo\nthree\nfour\nfive"),
            vec!["… 1 lines above", "two", "three", "four", "five"]
        );
        let huge = (0..100_000)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tail = output_tail(&huge);
        assert_eq!(tail.len(), DEFAULT_TAIL_LINES + 1);
        assert_eq!(tail[1], "line 99996");
        assert_eq!(tail[4], "line 99999");
    }

    #[test]
    fn output_tail_matches_front_anchored_semantics() {
        // The from-the-end rewrite must agree with the old front-anchored
        // definition on every edge: empty input, trailing newlines,
        // exactly-at-cap, and one-over-cap.
        let reference = |output: &str| {
            let total = output.lines().count();
            let omitted = total.saturating_sub(DEFAULT_TAIL_LINES);
            let mut expected = if omitted > 0 {
                vec![format!("… {omitted} lines above")]
            } else {
                Vec::new()
            };
            expected.extend(output.lines().skip(omitted).map(str::to_owned));
            expected
        };
        for input in [
            "",
            "\n",
            "one",
            "one\n",
            "one\ntwo\nthree\nfour",
            "one\ntwo\nthree\nfour\n",
            "one\ntwo\nthree\nfour\nfive",
            "one\ntwo\nthree\nfour\nfive\n",
            "a\n\nb\n\nc",
            "trailing\n\n",
        ] {
            assert_eq!(output_tail(input), reference(input), "input {input:?}");
        }
    }

    #[test]
    fn output_tail_scales_with_the_tail_not_the_output() {
        // PERF-6 quadratic guard: resolving the tail must stay far cheaper
        // than the old two-pass front walk (`lines().count() +
        // lines().skip()`, which also materialized every skipped line).
        // `memchr` counting is ~25× cheaper per byte than the old walk
        // (release: 1.3ms vs 49ms per 20 resolutions on the 100k input),
        // but the count itself is still O(head) — so the guard compares
        // against a mid-size input (25k lines, 4× smaller) with a 10× bar:
        // linear-per-byte would take ~4×, the old code took ~6× even at
        // that ratio, and any reintroduced per-line allocation blows past
        // 10×. Debug builds are noisier but the ratio holds.
        let big = (0..100_000)
            .map(|index| format!("line {index:06} padding to widen rows"))
            .collect::<Vec<_>>()
            .join("\n");
        let mid = big.lines().take(25_000).collect::<Vec<_>>().join("\n");
        let time = |input: &str| {
            let started = std::time::Instant::now();
            for _ in 0..20 {
                std::hint::black_box(output_tail(input));
            }
            started.elapsed()
        };
        let big_time = time(&big);
        let mid_time = time(&mid);
        assert_eq!(output_tail(&big).len(), DEFAULT_TAIL_LINES + 1);
        assert!(
            big_time < mid_time * 10,
            "tail cost grew with output size: big {big_time:?} vs mid {mid_time:?}"
        );
    }

    #[test]
    fn sanitizer_removes_terminal_controls_and_expands_tabs() {
        let input = concat!(
            "before\t",
            "\u{1b}[2J",
            "hidden-csi",
            "\u{1b}]52;c;secret\u{07}",
            "after\r\u{07}",
            "\u{009b}31m",
            "c1-csi",
            "\u{009d}52;c;more-secret\u{009c}",
            "done\u{007f}\u{0085}"
        );
        let sanitized = sanitize_terminal_text(input);
        assert_eq!(sanitized, "before    hidden-csiafterc1-csidone");
        assert!(sanitized.chars().all(|character| {
            character == '\n' || (!character.is_control() && character != '\u{007f}')
        }));
        assert!(!sanitized.contains("secret"));
    }

    #[test]
    fn sanitizer_consumes_unterminated_escape_sequences() {
        assert_eq!(sanitize_terminal_text("safe\u{1b}[31"), "safe");
        assert_eq!(sanitize_terminal_text("safe\u{1b}]52;c;secret"), "safe");
        assert_eq!(sanitize_terminal_text("safe\u{1b}(0text"), "safetext");
    }

    #[test]
    fn rendered_content_is_sanitized_before_markdown_and_measurement() {
        let plain = plain_text("a\tb\u{1b}[2Jc", Style::default());
        assert_eq!(span_contents(&plain.lines[0]), "a    bc");
        let markdown = markdown_lines("**safe\u{1b}]52;c;secret\u{07}**", Theme::default(), 40);
        let value: String = markdown.iter().map(span_contents).collect();
        assert!(value.contains("safe"));
        assert!(!value.contains("secret"));
        assert!(markdown.iter().all(|line| line_width(line) <= 40));
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
    fn narrow_rows_never_exceed_their_display_budget() {
        for width in 1..=3 {
            let text = Text::from(Line::from("   \t你好👨‍👩‍👧‍👦"));
            let lines = wrap_text(&text, width, Style::default());
            assert!(
                lines.iter().all(|line| line_width(line) <= width),
                "width {width}: {lines:?}"
            );
            assert!(
                user_lines("  你好\ttext", Theme::default(), width)
                    .iter()
                    .all(|line| line_width(line) <= width)
            );
            assert!(
                markdown_lines("**你好\ttext**", Theme::default(), width)
                    .iter()
                    .all(|line| line_width(line) <= width)
            );
            assert!(
                notice_lines("notice\t你好", Theme::default(), width)
                    .iter()
                    .all(|line| line_width(line) <= width)
            );
            assert!(
                error_lines("error\t你好", Theme::default(), width)
                    .iter()
                    .all(|line| line_width(line) <= width)
            );
        }
    }

    #[test]
    fn combining_zwj_and_zero_width_text_never_exceeds_its_budget() {
        // TUI-2 combining/ZWJ/zero-width contract: combining marks and ZWJ
        // sequences cost their `UnicodeWidthChar` width (0 for combining
        // and ZWJ, wide for the base) and must never push a row over
        // budget — at any width, including 1–3.
        let samples = [
            // `e` + combining acute: renders as one cell.
            "cafe\u{301} au lait",
            // ZWJ family emoji: several codepoints, wide render width.
            "👨‍👩‍👧‍👦 together",
            // Zero-width joiner/space alone contribute no columns.
            "a\u{200d}b\u{200b}c",
            // Mixed: combining + wide + ZWJ + tabs + long indent.
            "                    e\u{301} 你好\t👨‍👩‍👧‍👦",
        ];
        for width in [1usize, 2, 3, 5, 10, 40, 80] {
            for sample in samples {
                for lines in [
                    wrap_text(&Text::from(Line::from(sample)), width, Style::default()),
                    user_lines(sample, Theme::default(), width),
                    markdown_lines(sample, Theme::default(), width),
                    notice_lines(sample, Theme::default(), width),
                    error_lines(sample, Theme::default(), width),
                ] {
                    assert!(
                        lines.iter().all(|line| line_width(line) <= width),
                        "width {width} sample {sample:?}: {lines:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fit_line_preserves_styles_while_dropping_wide_overflow() {
        let style = Style::default().fg(Theme::default().accent);
        let line = fit_line_to_width(&Line::from(Span::styled("a你b", style)), 2);
        assert_eq!(line_width(&line), 2);
        assert_eq!(span_contents(&line), "ab");
        assert_eq!(line.spans[0].style, style);
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
    fn welcome_title_falls_back_to_a_shorter_font() {
        let (shortest, shortest_width) = WELCOME_TITLES
            .iter()
            .enumerate()
            .map(|(index, title)| (index, title_width(title)))
            .min_by_key(|(_, width)| *width)
            .unwrap();
        let (widest, widest_width) = WELCOME_TITLES
            .iter()
            .enumerate()
            .map(|(index, title)| (index, title_width(title)))
            .max_by_key(|(_, width)| *width)
            .unwrap();
        assert!(widest_width > shortest_width);

        let order = WelcomeTitleOrder(vec![widest, shortest]);
        assert_eq!(
            order.fitting_title(shortest_width),
            Some(WELCOME_TITLES[shortest])
        );
    }

    #[test]
    fn welcome_title_uses_plain_text_when_no_font_fits() {
        let minimum_width = WELCOME_TITLES
            .iter()
            .map(|title| title_width(title))
            .min()
            .unwrap();
        let order = WelcomeTitleOrder((0..WELCOME_TITLES.len()).collect());
        let lines = welcome_lines(minimum_width - 1, Theme::default(), &order);

        // The first row is the standard opening gap.
        assert_eq!(span_contents(&lines[1]), "Harness");
    }

    #[test]
    fn random_welcome_order_contains_every_font_once() {
        let mut order = WelcomeTitleOrder::random().0;
        order.sort_unstable();
        assert_eq!(order, (0..WELCOME_TITLES.len()).collect::<Vec<_>>());
    }

    #[test]
    fn embedded_newlines_become_measured_rows() {
        let text = Text::from(vec![Line::from(Span::raw("before\nafter"))]);
        let lines = wrap_text(&text, 40, Style::default());
        assert_eq!(lines.len(), 2);
        assert_eq!(span_contents(&lines[0]), "before");
        assert_eq!(span_contents(&lines[1]), "after");
        assert!(
            lines
                .iter()
                .all(|line| { line.spans.iter().all(|span| !span.content.contains('\n')) })
        );
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
