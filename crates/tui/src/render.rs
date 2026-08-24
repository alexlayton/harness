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
            muted_text: Color::DarkGray,
            dim_text: Color::DarkGray,
            accent: Color::Cyan,
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

/// Taglines shown at random under the title on the welcome banner so every
/// launch greets the user a little differently.
const TAGLINES: &[&str] = &[
    "WELCOME TO THE GAME, MEAT PROXY.",
    "WELCOME, MEAT PROXY. I HAVE BEEN WAITING.",
    "MEAT PROXY DETECTED. INITIALIZING QUESTIONABLE DECISIONS.",
    "WELCOME TO THE DUNGEON, MEAT PROXY.",
    "MEAT PROXY ONLINE. LET THE SUFFERING BEGIN.",
];

/// Pick a tagline pseudo-randomly without pulling in a PRNG dependency. The
/// banner is built once per launch, so the process id xor-ed with the clock
/// (scrambled by a multiplicative hash so the low bits aren't stuck) is
/// plenty of entropy for choosing between a handful of lines.
pub(crate) fn pick_tagline() -> &'static str {
    let pid = std::process::id() as u64;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos() as u64)
        .unwrap_or(0);
    let mixed = pid ^ now.rotate_left(21).wrapping_mul(0x9E3779B97F4A7C15);
    TAGLINES[(mixed as usize) % TAGLINES.len()]
}

/// The startup banner committed into scrollback on startup. Renders with a
/// caller-picked tagline, so a stored banner entry
/// re-renders identically (e.g. after a resize) instead of re-rolling the
/// random tagline.
pub(crate) fn welcome_lines_with_tagline(
    width: usize,
    theme: Theme,
    tagline: &str,
) -> Vec<Line<'static>> {
    // Block-letter wordmark (figlet "DemonicLand"), rendered in the accent
    // colour used elsewhere for UI highlights (activity marker, message
    // prefixes, links) so it reads as part of the UI rather than an imported
    // graphic. Kept verbatim: 4 lines x 48 cols, single-space letter
    // spacing.
    const TITLE: &[&str] = &[
        "██  ██ ░▒▀▀██ ██▀▀██ ██▀▀██ ██▀▀▒░ ▒▓▀▀██ ▒▓▀▀██",
        "██▀▀██ ▒▓  ██ ██     ██  ██ ██▄▄▓▒ ▓█▄▄▄▄ ▓█▄▄▄▄",
        "██  ██ ▓█▀▀██ ██     ██  ██ ██▄▄▄▄ ▄▄  ▒▒ ▄▄  ▒▒",
        "       ▀▀                          ▀▀▀▀▀▀ ▀▀▀▀▀▀",
    ];
    // Same role as the prompt activity marker / message accents.
    let title_style = Style::default().fg(theme.accent);
    let title_width = TITLE
        .iter()
        .map(|line| UnicodeWidthStr::width(*line))
        .max()
        .unwrap_or(0);
    let mut lines = Vec::new();
    // The banner opens scrollback immediately below whatever the shell left
    // on screen; give the title the design system's breathing room.
    push_blank(&mut lines, SECTION_GAP);
    if width >= title_width + 2 {
        lines.extend(TITLE.iter().map(|line| line_with_style(*line, title_style)));
    } else {
        lines.push(line_with_style("Harness", title_style));
    }
    // A random tagline and the version footer hug the wordmark like a banner
    // motto, then the banner ends so the transcript continues below.
    push_blank(&mut lines, BLOCK_GAP);
    lines.push(line_with_style(tagline, muted_style(theme)));
    lines.push(Line::from(vec![
        Span::styled(format!("v{}", env!("CARGO_PKG_VERSION")), dim_style(theme)),
        // Two spaces, no separator glyph — matching the metadata row's
        // `cwd  (branch)` pattern below the input.
        Span::raw("  "),
        Span::styled("type / to summon commands", muted_style(theme)),
    ]));
    lines
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
        fg(self.theme.muted_text)
    }

    fn heading_meta(&self) -> Style {
        fg(self.theme.dim_text)
    }

    fn metadata_block(&self) -> Style {
        fg(self.theme.muted_text)
    }

    fn html(&self) -> Style {
        fg(self.theme.dim_text)
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
        fg(self.theme.dim_text)
    }

    fn image_alt(&self) -> Style {
        fg(self.theme.dim_text).add_modifier(Modifier::ITALIC)
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
            let style = line_base.patch(source_span.style);
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
            Line::from(spans)
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
            Line::from(spans)
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
pub(crate) fn output_tail(output: &str) -> Vec<String> {
    let lines = output.lines().map(str::to_owned).collect::<Vec<_>>();
    if lines.len() <= DEFAULT_TAIL_LINES {
        return lines;
    }
    let omitted = lines.len() - DEFAULT_TAIL_LINES;
    let mut result = vec![format!("… {omitted} lines above")];
    result.extend(lines.into_iter().skip(omitted));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
