use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use textwrap::wrap;

/// The amount of whitespace used before a new conversation section.
pub const SECTION_GAP: usize = 2;
/// The amount of whitespace used around secondary blocks such as tools and
/// reasoning.
pub const BLOCK_GAP: usize = 1;

/// The key descriptions are deliberately kept in one place.  The welcome
/// header is rendered from this table and the same table is mirrored in the
/// README's keys section.
pub const KEYMAP: &[(&str, &str)] = &[
    ("Enter", "Send message"),
    ("Shift+Enter", "Newline"),
    ("↑ / ↓", "History"),
    ("Esc", "Interrupt"),
    ("Ctrl+O", "Expand tool call"),
    ("Ctrl+C", "Quit"),
];

const MAX_DETAIL_OUTPUT_LINES: usize = 200;
const MAX_LIVE_DETAIL_ROWS: usize = 12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolRecord {
    pub name: String,
    /// Pretty-printed arguments supplied by the agent.  Keeping this as text
    /// means the TUI does not need to depend on serde_json.
    pub args: String,
    pub summary: String,
    pub ok: bool,
    pub duration_ms: u64,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailTool {
    pub record: ToolRecord,
    pub expanded: bool,
}

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

/// Number of terminal rows needed for the final styled text block.
///
/// Heights must be calculated from the same `Text` that is passed to
/// `Paragraph`; counting the source string separately is especially error
/// prone for labels, indentation, blank lines, and wrapped spans.
pub fn text_height<W: TryInto<usize>>(text: &Text<'_>, width: W) -> u16 {
    let width = width.try_into().ok().unwrap_or(1).max(1);
    let rows = text
        .lines
        .iter()
        .map(|line| {
            let plain = line
                .spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>();
            wrap_count(&plain, width)
        })
        .sum::<usize>()
        .max(1);
    rows.min(u16::MAX as usize) as u16
}

fn label_style() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn dim_style() -> Style {
    Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::DIM)
}

fn reasoning_style() -> Style {
    dim_style().add_modifier(Modifier::ITALIC)
}

fn error_style() -> Style {
    Style::default().fg(Color::Red)
}

fn blank_line() -> Line<'static> {
    Line::from("")
}

fn plain_line(value: impl Into<String>, style: Style) -> Line<'static> {
    Line::from(Span::styled(value.into(), style))
}

fn push_blank_lines(lines: &mut Vec<Line<'static>>, count: usize) {
    lines.extend((0..count).map(|_| blank_line()));
}

fn text(lines: Vec<Line<'static>>) -> Text<'static> {
    Text::from(lines)
}

/// Build the one-time welcome/header block.
pub fn build_header(_width: u16) -> Text<'static> {
    let label_width = KEYMAP
        .iter()
        .map(|(label, _)| label.chars().count())
        .max()
        .unwrap_or(0);
    let mut lines = vec![plain_line("Harness", label_style()), blank_line()];
    for (label, description) in KEYMAP {
        let padding = " ".repeat(label_width.saturating_sub(label.chars().count()) + 4);
        lines.push(Line::from(vec![
            Span::styled((*label).to_owned(), dim_style()),
            Span::raw(padding),
            Span::raw((*description).to_owned()),
        ]));
    }
    text(lines)
}

/// Build a user message with a bold role label and regular message text.
pub fn build_user(input: &str) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, SECTION_GAP);
    lines.push(plain_line("User", label_style()));
    for line in input.split('\n') {
        lines.push(plain_line(line.to_owned(), Style::default()));
    }
    text(lines)
}

/// Build the role label for an assistant/model segment.  The label is kept
/// separate from markdown so it can be committed exactly once per segment.
pub fn build_model_label() -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, SECTION_GAP);
    lines.push(plain_line("Model", label_style()));
    text(lines)
}

fn reasoning_lines(reasoning: &str) -> Vec<Line<'static>> {
    reasoning
        .split('\n')
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "thinking · " } else { "  " };
            plain_line(format!("{prefix}{line}"), reasoning_style())
        })
        .collect()
}

/// Build a subdued reasoning block.  Reasoning is intentionally separated
/// from primary model text by one blank line on either side.
pub fn build_reasoning(reasoning: &str) -> Text<'static> {
    if reasoning.is_empty() {
        return Text::default();
    }
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    lines.extend(reasoning_lines(reasoning));
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
}

fn duration_text(duration_ms: u64) -> String {
    if duration_ms >= 1_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{duration_ms}ms")
    }
}

fn tool_compact_lines(
    name: &str,
    summary: &str,
    ok: bool,
    duration_ms: u64,
    error: Option<&str>,
) -> Vec<Line<'static>> {
    let header_style = if ok { Style::default() } else { error_style() };
    let marker = if ok { "" } else { "✗ " };
    let header = format!("{marker}Tool · {name}");
    vec![
        Line::from(vec![
            Span::styled(header, header_style),
            Span::styled(format!(" ({})", duration_text(duration_ms)), dim_style()),
        ]),
        plain_line(format!("  {summary}"), dim_style()),
    ]
    .into_iter()
    .chain(error.filter(|_| !ok).map(|value| {
        plain_line(
            format!("  {}", value.lines().next().unwrap_or(value)),
            dim_style(),
        )
    }))
    .collect()
}

/// Build the compact representation of a completed tool call.
pub fn build_tool_finished(
    name: &str,
    summary: &str,
    ok: bool,
    duration_ms: u64,
    error: Option<&str>,
) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    lines.extend(tool_compact_lines(name, summary, ok, duration_ms, error));
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
}

/// Build the compact representation from a stored tool record.
pub fn build_tool_compact(record: &ToolRecord) -> Text<'static> {
    build_tool_finished(
        &record.name,
        &record.summary,
        record.ok,
        record.duration_ms,
        record.error.as_deref(),
    )
}

/// Build an error block.  The complete error is retained in scrollback while
/// the first line receives the compact red marker used by the UI.
pub fn build_error(error: &str) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    let mut error_lines = error.split('\n');
    if let Some(first) = error_lines.next() {
        lines.push(plain_line(format!("✗ {first}"), error_style()));
    } else {
        lines.push(plain_line("✗", error_style()));
    }
    lines.extend(error_lines.map(|line| plain_line(format!("  {line}"), error_style())));
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
}

fn capped_output_lines(output: &str) -> Vec<String> {
    let lines = output.lines().collect::<Vec<_>>();
    if lines.len() <= MAX_DETAIL_OUTPUT_LINES {
        return lines.into_iter().map(str::to_owned).collect();
    }

    let omitted = lines.len() - MAX_DETAIL_OUTPUT_LINES;
    let mut result = lines[..MAX_DETAIL_OUTPUT_LINES / 2]
        .iter()
        .map(|line| (*line).to_owned())
        .collect::<Vec<_>>();
    result.push(format!("… {omitted} lines omitted"));
    result.extend(
        lines[lines.len() - MAX_DETAIL_OUTPUT_LINES / 2..]
            .iter()
            .map(|line| (*line).to_owned()),
    );
    result
}

fn push_indented(lines: &mut Vec<Line<'static>>, value: &str, indent: &str) {
    if value.is_empty() {
        lines.push(plain_line(format!("{indent}(empty)"), dim_style()));
    } else {
        lines.extend(
            value
                .split('\n')
                .map(|line| plain_line(format!("{indent}{line}"), dim_style())),
        );
    }
}

/// Build the optional full detail view for a completed tool call.
pub fn build_tool_detail(record: &ToolRecord, _width: u16) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    lines.push(plain_line(
        format!("Tool · {} — details", record.name),
        dim_style(),
    ));
    lines.push(plain_line("  args", dim_style()));
    push_indented(&mut lines, &record.args, "    ");
    lines.push(plain_line("  output", dim_style()));
    let output = capped_output_lines(&record.output);
    if output.is_empty() {
        lines.push(plain_line("    (empty)", dim_style()));
    } else {
        for line in output {
            lines.push(plain_line(format!("    {line}"), dim_style()));
        }
    }
    if let Some(error) = &record.error {
        lines.push(plain_line("  error", dim_style()));
        push_indented(&mut lines, error, "    ");
    }
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
}

/// Insert an already-built text block into immutable terminal scrollback.
pub fn insert_text<'a, B: Backend>(
    terminal: &mut Terminal<B>,
    text: Text<'a>,
    width: u16,
) -> std::io::Result<()> {
    if text.lines.is_empty() {
        return Ok(());
    }
    let height = text_height(&text, width).max(1);
    terminal.insert_before(height, move |buffer| {
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .render(buffer.area, buffer);
    })
}

pub fn insert_header<B: Backend>(terminal: &mut Terminal<B>, width: u16) -> std::io::Result<()> {
    insert_text(terminal, build_header(width), width)
}

pub fn insert_markdown<B: Backend>(
    terminal: &mut Terminal<B>,
    markdown: &str,
    width: u16,
) -> std::io::Result<()> {
    if markdown.is_empty() {
        return Ok(());
    }
    insert_text(terminal, build_markdown(markdown), width)
}

/// Convert markdown to owned ratatui text.  The conversion at this crate
/// boundary intentionally preserves the terminal text while making the
/// returned value independent of the input stream's lifetime.
pub fn build_markdown(markdown: &str) -> Text<'static> {
    let rendered = tui_markdown::from_str(markdown);
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
    text(lines)
}

pub fn insert_user<B: Backend>(
    terminal: &mut Terminal<B>,
    input: &str,
    width: u16,
) -> std::io::Result<()> {
    insert_text(terminal, build_user(input), width)
}

pub fn insert_model_label<B: Backend>(
    terminal: &mut Terminal<B>,
    width: u16,
) -> std::io::Result<()> {
    insert_text(terminal, build_model_label(), width)
}

pub fn insert_reasoning<B: Backend>(
    terminal: &mut Terminal<B>,
    reasoning: &str,
    width: u16,
) -> std::io::Result<()> {
    if reasoning.is_empty() {
        return Ok(());
    }
    insert_text(terminal, build_reasoning(reasoning), width)
}

pub fn insert_tool_finished<B: Backend>(
    terminal: &mut Terminal<B>,
    name: &str,
    summary: &str,
    ok: bool,
    duration_ms: u64,
    error: Option<&str>,
    width: u16,
) -> std::io::Result<()> {
    insert_text(
        terminal,
        build_tool_finished(name, summary, ok, duration_ms, error),
        width,
    )
}

pub fn insert_tool_detail<B: Backend>(
    terminal: &mut Terminal<B>,
    record: &ToolRecord,
    width: u16,
) -> std::io::Result<()> {
    insert_text(terminal, build_tool_detail(record, width), width)
}

pub fn insert_error<B: Backend>(
    terminal: &mut Terminal<B>,
    error: &str,
    width: u16,
) -> std::io::Result<()> {
    insert_text(terminal, build_error(error), width)
}

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

fn is_blank(line: &Line<'_>) -> bool {
    line.spans.iter().all(|span| span.content.is_empty())
}

fn without_outer_gaps(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    while lines.first().is_some_and(is_blank) {
        lines.remove(0);
    }
    while lines.last().is_some_and(is_blank) {
        lines.pop();
    }
    lines
}

fn focused_line(line: Line<'static>, focused: bool) -> Line<'static> {
    if !focused {
        return line;
    }
    let mut spans = vec![Span::styled("> ", Style::default().fg(Color::Cyan))];
    spans.extend(line.spans);
    Line::from(spans)
}

fn live_tool_lines(entry: &TailTool, focused: bool, width: u16) -> Vec<Line<'static>> {
    if !entry.expanded {
        let mut lines = tool_compact_lines(
            &entry.record.name,
            &entry.record.summary,
            entry.record.ok,
            entry.record.duration_ms,
            entry.record.error.as_deref(),
        );
        if let Some(first) = lines.first_mut() {
            let current = std::mem::take(first);
            *first = focused_line(current, focused);
        }
        return lines;
    }

    let lines = without_outer_gaps(build_tool_detail(&entry.record, width).lines.to_vec());
    let mut visible = Vec::new();
    let mut rows = 0usize;
    let mut truncated = false;
    for line in lines {
        let plain = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        let line_rows = wrap_count(&plain, width as usize);
        if rows + line_rows > MAX_LIVE_DETAIL_ROWS {
            truncated = true;
            break;
        }
        rows += line_rows;
        visible.push(line);
    }
    if truncated {
        let note = plain_line("    … more details; Ctrl+O to dump", dim_style());
        let note_plain = note
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        let note_rows = wrap_count(&note_plain, width as usize);
        while rows + note_rows > MAX_LIVE_DETAIL_ROWS && !visible.is_empty() {
            let removed = visible.pop().expect("visible is non-empty");
            let removed_plain = removed
                .spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>();
            rows = rows.saturating_sub(wrap_count(&removed_plain, width as usize));
        }
        visible.push(note);
    }
    let mut lines = visible;
    if let Some(first) = lines.first_mut() {
        let current = std::mem::take(first);
        *first = focused_line(current, focused);
    }
    lines
}

fn append_pending_reasoning(lines: &mut Vec<Line<'static>>, reasoning: &str) {
    if reasoning.is_empty() {
        return;
    }
    push_blank_lines(lines, BLOCK_GAP);
    lines.extend(reasoning_lines(reasoning));
    push_blank_lines(lines, BLOCK_GAP);
}

fn append_pending_text(lines: &mut Vec<Line<'static>>, pending_text: &str) {
    if pending_text.is_empty() {
        return;
    }
    lines.extend(
        pending_text
            .split('\n')
            .map(|line| plain_line(line.to_owned(), Style::default())),
    );
}

/// Render the redrawable bottom live area.  Finished tail entries are placed
/// before pending reasoning/text, and a running tool is always last.
pub fn render_live_with_tail(
    area: Rect,
    tail: &[TailTool],
    focused_tool: Option<usize>,
    pending_reasoning: &str,
    pending_text: &str,
    running_tool: Option<(&str, &str, usize)>,
    frame: &mut ratatui::Frame<'_>,
) {
    let mut lines = Vec::new();
    for (index, entry) in tail.iter().enumerate() {
        push_blank_lines(&mut lines, BLOCK_GAP);
        lines.extend(live_tool_lines(
            entry,
            focused_tool == Some(index),
            area.width,
        ));
        push_blank_lines(&mut lines, BLOCK_GAP);
    }
    append_pending_reasoning(&mut lines, pending_reasoning);
    append_pending_text(&mut lines, pending_text);
    if let Some((name, summary, spinner)) = running_tool {
        push_blank_lines(&mut lines, BLOCK_GAP);
        lines.push(Line::from(vec![Span::styled(
            format!("{} Tool · {name}", FRAMES[spinner % FRAMES.len()]),
            Style::default().fg(Color::Yellow),
        )]));
        lines.push(plain_line(format!("  {summary}"), dim_style()));
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

/// Compatibility wrapper for callers that only have the pre-tail running tool
/// representation.
pub fn render_live(
    area: Rect,
    pending_reasoning: &str,
    pending_text: &str,
    running_tool: Option<(&str, usize)>,
    frame: &mut ratatui::Frame<'_>,
) {
    let mut lines = Vec::new();
    append_pending_reasoning(&mut lines, pending_reasoning);
    append_pending_text(&mut lines, pending_text);
    if let Some((summary, spinner)) = running_tool {
        lines.push(Line::from(Span::styled(
            format!("{} {summary}", FRAMES[spinner % FRAMES.len()]),
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

    fn plain(text: &Text<'_>) -> String {
        text.lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

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

    #[test]
    fn header_uses_keymap_and_styles_roles() {
        let header = build_header(80);
        let value = plain(&header);
        assert!(value.starts_with("Harness\n\n"));
        assert!(value.contains("Ctrl+O"));
        assert_eq!(header.lines[0].spans[0].style.add_modifier, Modifier::BOLD);
        assert!(
            header.lines[2].spans[0]
                .style
                .add_modifier
                .contains(Modifier::DIM)
        );
    }

    #[test]
    fn user_and_reasoning_have_labels_and_gaps() {
        let user = build_user("hello\nworld");
        assert_eq!(plain(&user), "\n\nUser\nhello\nworld");
        let reasoning = build_reasoning("checking\nfiles");
        assert_eq!(plain(&reasoning), "\nthinking · checking\n  files\n");
    }

    #[test]
    fn compact_tool_has_name_summary_duration_and_error_preview() {
        let tool = build_tool_finished(
            "bash",
            "bash: cargo test",
            false,
            1_300,
            Some("first line\nsecond line"),
        );
        let value = plain(&tool);
        assert!(value.contains("✗ Tool · bash (1.3s)"));
        assert!(value.contains("  bash: cargo test"));
        assert!(value.contains("  first line"));
        assert!(!value.contains("second line"));
    }

    #[test]
    fn detail_caps_middle_output_and_keeps_error() {
        let output = (0..205)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let record = ToolRecord {
            name: "bash".into(),
            args: "{\n  \"command\": \"cargo test\"\n}".into(),
            summary: "cargo test".into(),
            ok: false,
            duration_ms: 10,
            output,
            error: Some("complete error\nwith context".into()),
        };
        let detail = plain(&build_tool_detail(&record, 80));
        assert!(detail.contains("Tool · bash — details"));
        assert!(detail.contains("line 0"));
        assert!(detail.contains("line 204"));
        assert!(detail.contains("… 5 lines omitted"));
        assert!(detail.contains("with context"));
    }

    #[test]
    fn text_height_counts_final_text_lines_and_gaps() {
        let user = build_user("abcdefgh");
        assert_eq!(text_height(&user, 4), 2 + 1 + 2);
    }
}
