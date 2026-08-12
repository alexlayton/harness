use crate::commands::Candidate;
use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Widget, Wrap};
use textwrap::wrap;

/// The amount of whitespace used before a new conversation section.
pub const SECTION_GAP: usize = 2;
/// The amount of whitespace used around secondary blocks such as tools and
/// reasoning.
pub const BLOCK_GAP: usize = 1;

/// Subtle background grouping everything that belongs to a tool call.
pub const TOOL_BG: Color = Color::Rgb(34, 35, 41);
/// Brighter variant marking the keyboard-focused tool call.
pub const TOOL_FOCUSED_BG: Color = Color::Rgb(46, 48, 58);
/// Faint violet background grouping model reasoning separately from tools.
pub const THINKING_BG: Color = Color::Rgb(30, 28, 38);
/// Background of the completion popup; the selected row is brighter.
pub const COMPLETION_BG: Color = Color::Rgb(28, 29, 34);
pub const COMPLETION_SELECTED_BG: Color = Color::Rgb(58, 60, 74);
/// Shared number of completion rows visible below the editor.
pub const MAX_COMPLETION_ROWS: usize = 8;

/// The key descriptions are deliberately kept in one place so the welcome
/// header doubles as the keybinding reference.
pub const KEYMAP: &[(&str, &str)] = &[
    ("Enter", "Send message"),
    ("Shift+Enter", "Newline"),
    ("↑ / ↓", "History"),
    ("Tab", "Focus tool calls"),
    ("Esc", "Interrupt"),
    ("Ctrl+O", "Expand / collapse tool"),
    ("/", "Commands"),
    ("@", "File references"),
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

fn plain_line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>()
}

/// Number of terminal rows needed for the final styled text block.
///
/// Heights must be calculated from the same `Text` that is passed to
/// `Paragraph`; counting the source string separately is especially error
/// prone for prefixes, indentation, blank lines, and wrapped spans.
pub fn text_height<W: TryInto<usize>>(text: &Text<'_>, width: W) -> u16 {
    let width = width.try_into().ok().unwrap_or(1).max(1);
    let rows = text
        .lines
        .iter()
        .map(|line| wrap_count(&plain_line_text(line), width))
        .sum::<usize>()
        .max(1);
    rows.min(u16::MAX as usize) as u16
}

fn label_style() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn user_text_style() -> Style {
    Style::default().fg(Color::White)
}

/// Base foreground color for assistant output.
fn model_text_style() -> Style {
    Style::default().fg(Color::Rgb(170, 178, 192))
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

/// Base style of a tool block; applied to the whole render area so the
/// background band spans the full terminal width.
pub fn tool_bg_style() -> Style {
    Style::default().bg(TOOL_BG)
}

/// The bordered box around the message editor.
pub fn input_block() -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray))
        .padding(Padding::horizontal(1))
}

fn tool_name_style() -> Style {
    Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD)
}

fn tool_summary_style() -> Style {
    Style::default().fg(Color::Gray)
}

fn tool_section_style() -> Style {
    Style::default()
        .fg(Color::Gray)
        .add_modifier(Modifier::BOLD)
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

/// Build a user message with a white foreground.
pub fn build_user(input: &str) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, SECTION_GAP);
    for line in input.split('\n') {
        lines.push(plain_line(line.to_owned(), user_text_style()));
    }
    text(lines)
}

fn reasoning_lines(reasoning: &str) -> Vec<Line<'static>> {
    reasoning
        .split('\n')
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "" } else { "  " };
            plain_line(format!("{prefix}{line}"), reasoning_style())
        })
        .collect()
}

/// Build the body of a subdued reasoning block.  Spacing and the background
/// band are added by the insertion/live-area paths so they stay consistent.
pub fn build_reasoning(reasoning: &str) -> Text<'static> {
    if reasoning.is_empty() {
        return Text::default();
    }
    text(reasoning_lines(reasoning))
}

fn duration_text(duration_ms: u64) -> String {
    if duration_ms >= 1_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{duration_ms}ms")
    }
}

/// Drop a redundant `<name>:` / `<name> ` prefix so a compact line reads
/// "● bash · cargo test" rather than "● bash · bash: cargo test".
fn summary_tail<'a>(name: &str, summary: &'a str) -> &'a str {
    match summary.strip_prefix(name) {
        Some(rest) => rest.strip_prefix(':').unwrap_or(rest).trim_start(),
        None => summary,
    }
}

fn status_dot(ok: bool) -> (&'static str, Style) {
    if ok {
        ("●", Style::default().fg(Color::Green))
    } else {
        ("●", Style::default().fg(Color::Red))
    }
}

/// The compact representation of a completed tool call: a single status line,
/// plus a one-line error preview for failures.  No outer gap lines are
/// included so the caller can shade the block as a unit.
fn tool_compact_lines(
    name: &str,
    summary: &str,
    ok: bool,
    duration_ms: u64,
    error: Option<&str>,
) -> Vec<Line<'static>> {
    let (dot, dot_style) = status_dot(ok);
    let mut header = vec![
        Span::styled(format!(" {dot} "), dot_style),
        Span::styled(name.to_owned(), tool_name_style()),
    ];
    let tail = summary_tail(name, summary);
    if !tail.is_empty() {
        header.push(Span::styled(format!(" · {tail}"), tool_summary_style()));
    }
    header.push(Span::styled(
        format!(" · {}", duration_text(duration_ms)),
        dim_style(),
    ));
    let mut lines = vec![Line::from(header)];
    if !ok && let Some(error) = error {
        lines.push(plain_line(
            format!("    {}", error.lines().next().unwrap_or(error)),
            error_style(),
        ));
    }
    lines
}

/// Build an informational command/agent notice block.
pub fn build_notice(notice: &str) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    let mut notice_lines = notice.split('\n');
    if let Some(first) = notice_lines.next() {
        lines.push(plain_line(format!("· {first}"), dim_style()));
    } else {
        lines.push(plain_line("·", dim_style()));
    }
    lines.extend(notice_lines.map(|line| plain_line(format!("  {line}"), dim_style())));
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
}

/// Build the one-line echo for a command submitted from the editor.
pub fn build_command_echo(command: &str) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    lines.push(plain_line(format!("⌘ {command}"), dim_style()));
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
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

/// Body of the full detail view for a completed tool call.  No outer gap
/// lines are included so the block can be shaded as a unit.
fn tool_detail_lines(record: &ToolRecord) -> Vec<Line<'static>> {
    let (dot, dot_style) = status_dot(record.ok);
    let mut lines = vec![Line::from(vec![
        Span::styled(format!(" {dot} "), dot_style),
        Span::styled(record.name.clone(), tool_name_style()),
        Span::styled(" — details".to_owned(), dim_style()),
    ])];
    lines.push(plain_line("  args", tool_section_style()));
    push_indented(&mut lines, &record.args, "    ");
    lines.push(plain_line("  output", tool_section_style()));
    let output = capped_output_lines(&record.output);
    if output.is_empty() {
        lines.push(plain_line("    (empty)", dim_style()));
    } else {
        for line in output {
            lines.push(plain_line(format!("    {line}"), dim_style()));
        }
    }
    if let Some(error) = &record.error {
        lines.push(plain_line("  error", tool_section_style()));
        lines.extend(
            error
                .split('\n')
                .map(|line| plain_line(format!("    {line}"), error_style())),
        );
    }
    lines
}

/// Build the gapped detail view.  Used by tests and as documentation of the
/// committed layout; live and scrollback paths shade `tool_detail_lines`.
pub fn build_tool_detail(record: &ToolRecord, _width: u16) -> Text<'static> {
    let mut lines = Vec::new();
    push_blank_lines(&mut lines, BLOCK_GAP);
    lines.extend(tool_detail_lines(record));
    push_blank_lines(&mut lines, BLOCK_GAP);
    text(lines)
}

/// Insert an already-built text block into immutable terminal scrollback.
pub fn insert_text<'a, B: Backend>(
    terminal: &mut Terminal<B>,
    text: Text<'a>,
    width: u16,
) -> std::io::Result<()> {
    insert_text_styled(terminal, text, width, Style::default())
}

/// Insert a text block whose base style fills the whole render area.  Span
/// styles patch over the base style, so a background color set here becomes a
/// full-width band behind every (including wrapped) row.
pub fn insert_text_styled<'a, B: Backend>(
    terminal: &mut Terminal<B>,
    text: Text<'a>,
    width: u16,
    base: Style,
) -> std::io::Result<()> {
    if text.lines.is_empty() {
        return Ok(());
    }
    let height = text_height(&text, width).max(1);
    terminal.insert_before(height, move |buffer| {
        Paragraph::new(text)
            .style(base)
            .wrap(Wrap { trim: false })
            .render(buffer.area, buffer);
    })
}

/// Insert plain (unshaded) blank separator rows into scrollback.
fn insert_gap<B: Backend>(terminal: &mut Terminal<B>, count: usize) -> std::io::Result<()> {
    if count == 0 {
        return Ok(());
    }
    terminal.insert_before(count.min(u16::MAX as usize) as u16, |_| {})
}

/// Insert the blank rows separating a user turn from the next transcript block.
pub fn insert_section_gap<B: Backend>(terminal: &mut Terminal<B>) -> std::io::Result<()> {
    insert_gap(terminal, SECTION_GAP)
}

/// Insert a tool block: unshaded gap rows around a full-width shaded body.
fn insert_shaded_block<B: Backend>(
    terminal: &mut Terminal<B>,
    body: Vec<Line<'static>>,
    width: u16,
    base_style: Style,
) -> std::io::Result<()> {
    insert_gap(terminal, BLOCK_GAP)?;
    insert_text_styled(terminal, text(body), width, base_style)?;
    insert_gap(terminal, BLOCK_GAP)
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
    insert_text_styled(
        terminal,
        build_markdown(markdown),
        width,
        model_text_style(),
    )
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

pub fn insert_reasoning<B: Backend>(
    terminal: &mut Terminal<B>,
    reasoning: &str,
    width: u16,
) -> std::io::Result<()> {
    if reasoning.is_empty() {
        return Ok(());
    }
    insert_shaded_block(
        terminal,
        reasoning_lines(reasoning),
        width,
        Style::default().bg(THINKING_BG),
    )
}

pub fn insert_notice<B: Backend>(
    terminal: &mut Terminal<B>,
    notice: &str,
    width: u16,
) -> std::io::Result<()> {
    if notice.is_empty() {
        return Ok(());
    }
    insert_text(terminal, build_notice(notice), width)
}

pub fn insert_command_echo<B: Backend>(
    terminal: &mut Terminal<B>,
    command: &str,
    width: u16,
) -> std::io::Result<()> {
    insert_text(terminal, build_command_echo(command), width)
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
    insert_shaded_block(
        terminal,
        tool_compact_lines(name, summary, ok, duration_ms, error),
        width,
        tool_bg_style(),
    )
}

pub fn insert_tool_detail<B: Backend>(
    terminal: &mut Terminal<B>,
    record: &ToolRecord,
    width: u16,
) -> std::io::Result<()> {
    insert_shaded_block(terminal, tool_detail_lines(record), width, tool_bg_style())
}

pub fn insert_error<B: Backend>(
    terminal: &mut Terminal<B>,
    error: &str,
    width: u16,
) -> std::io::Result<()> {
    insert_text(terminal, build_error(error), width)
}

/// Render the completion popup below the editor.  The app keeps the complete
/// candidate set and supplies an offset so this function only has to render
/// the visible window.
pub fn render_completion(
    area: Rect,
    candidates: &[Candidate],
    selected: usize,
    offset: usize,
    frame: &mut ratatui::Frame<'_>,
) {
    if area.height == 0 || candidates.is_empty() {
        return;
    }
    let visible = candidates
        .iter()
        .enumerate()
        .skip(offset)
        .take((area.height as usize).min(MAX_COMPLETION_ROWS))
        .collect::<Vec<_>>();
    if visible.is_empty() {
        return;
    }
    let value_width = candidates
        .iter()
        .map(|candidate| candidate.value.chars().count())
        .max()
        .unwrap_or(0);
    let lines = visible
        .into_iter()
        .map(|(index, candidate)| {
            let is_selected = index == selected;
            let row_bg = if is_selected {
                COMPLETION_SELECTED_BG
            } else {
                COMPLETION_BG
            };
            let value_style = Style::default().bg(row_bg).add_modifier(if is_selected {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });
            let value_style = if is_selected {
                value_style.fg(Color::White)
            } else {
                value_style
            };
            let description_style = dim_style().bg(row_bg);
            let padding =
                " ".repeat(value_width.saturating_sub(candidate.value.chars().count()) + 2);
            let mut spans = vec![
                Span::styled(candidate.value.clone(), value_style),
                Span::styled(padding, description_style),
                Span::styled(candidate.description.clone(), description_style),
            ];
            if is_selected {
                // Extend the selected row's brighter background to the edge.
                let used = value_width + 2 + candidate.description.chars().count();
                let rest = (area.width as usize).saturating_sub(used);
                spans.push(Span::styled(" ".repeat(rest), Style::default().bg(row_bg)));
            }
            Line::from(spans)
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(Text::from(lines)).style(Style::default().bg(COMPLETION_BG)),
        area,
    );
}

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

fn focused_line(line: Line<'static>) -> Line<'static> {
    let mut spans = vec![Span::styled("> ", Style::default().fg(Color::Cyan))];
    spans.extend(line.spans);
    Line::from(spans)
}

fn live_tool_lines(entry: &TailTool, focused: bool, width: u16) -> Vec<Line<'static>> {
    let mut lines = if !entry.expanded {
        tool_compact_lines(
            &entry.record.name,
            &entry.record.summary,
            entry.record.ok,
            entry.record.duration_ms,
            entry.record.error.as_deref(),
        )
    } else {
        capped_live_detail_lines(&entry.record, width)
    };
    if focused && let Some(first) = lines.first_mut() {
        let current = std::mem::take(first);
        *first = focused_line(current);
    }
    lines
}

/// The expanded in-place view is capped so a chatty tool cannot push the rest
/// of the live area off screen; the full dump remains available via Tab + d.
fn capped_live_detail_lines(record: &ToolRecord, width: u16) -> Vec<Line<'static>> {
    let mut visible = Vec::new();
    let mut rows = 0usize;
    let mut truncated = false;
    for line in tool_detail_lines(record) {
        let line_rows = wrap_count(&plain_line_text(&line), width as usize);
        if rows + line_rows > MAX_LIVE_DETAIL_ROWS {
            truncated = true;
            break;
        }
        rows += line_rows;
        visible.push(line);
    }
    if truncated {
        let note = plain_line("    … truncated — Tab, then d for full output", dim_style());
        let note_rows = wrap_count(&plain_line_text(&note), width as usize);
        while rows + note_rows > MAX_LIVE_DETAIL_ROWS && !visible.is_empty() {
            let removed = visible.pop().expect("visible is non-empty");
            rows = rows.saturating_sub(wrap_count(&plain_line_text(&removed), width as usize));
        }
        visible.push(note);
    }
    visible
}

fn append_pending_reasoning(
    lines: &mut Vec<Line<'static>>,
    flags: &mut Vec<RowKind>,
    reasoning: &str,
) {
    if reasoning.is_empty() {
        return;
    }
    push_gap_flags(lines, flags, BLOCK_GAP);
    push_flagged(lines, flags, reasoning_lines(reasoning), RowKind::Thinking);
    push_gap_flags(lines, flags, BLOCK_GAP);
}

fn append_pending_text(
    lines: &mut Vec<Line<'static>>,
    flags: &mut Vec<RowKind>,
    pending_text: &str,
) {
    if pending_text.is_empty() {
        return;
    }
    let text = pending_text
        .split('\n')
        .map(|line| plain_line(line.to_owned(), model_text_style()))
        .collect();
    push_flagged(lines, flags, text, RowKind::Plain);
}

/// Patches a background style onto whole rows.  Rendered after the text so
/// glyphs and foreground colors are preserved while the background band is
/// extended to the full width.
struct RowShade {
    rows: Vec<bool>,
    style: Style,
}

impl Widget for RowShade {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for (offset, shaded) in self.rows.iter().enumerate() {
            let Ok(offset) = u16::try_from(offset) else {
                break;
            };
            if offset >= area.height {
                break;
            }
            if !shaded {
                continue;
            }
            buf.set_style(
                Rect {
                    y: area.y + offset,
                    height: 1,
                    ..area
                },
                self.style,
            );
        }
    }
}

fn push_gap_flags(lines: &mut Vec<Line<'static>>, flags: &mut Vec<RowKind>, count: usize) {
    for _ in 0..count {
        lines.push(blank_line());
        flags.push(RowKind::Plain);
    }
}

fn push_flagged(
    lines: &mut Vec<Line<'static>>,
    flags: &mut Vec<RowKind>,
    added: Vec<Line<'static>>,
    kind: RowKind,
) {
    flags.extend(std::iter::repeat_n(kind, added.len()));
    lines.extend(added);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowKind {
    Plain,
    Tool,
    FocusedTool,
    Thinking,
}

/// Render the redrawable bottom live area.  Finished tail entries are placed
/// before pending reasoning/text, and a running tool is always last.  Tool and
/// reasoning rows are shaded so secondary blocks stand out from the transcript.
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
    let mut flags = Vec::new();
    for (index, entry) in tail.iter().enumerate() {
        push_gap_flags(&mut lines, &mut flags, BLOCK_GAP);
        let focused = focused_tool == Some(index);
        let kind = if focused {
            RowKind::FocusedTool
        } else {
            RowKind::Tool
        };
        push_flagged(
            &mut lines,
            &mut flags,
            live_tool_lines(entry, focused, area.width),
            kind,
        );
        push_gap_flags(&mut lines, &mut flags, BLOCK_GAP);
    }
    append_pending_reasoning(&mut lines, &mut flags, pending_reasoning);
    append_pending_text(&mut lines, &mut flags, pending_text);
    if let Some((name, summary, spinner)) = running_tool {
        push_gap_flags(&mut lines, &mut flags, BLOCK_GAP);
        let tail = summary_tail(name, summary);
        let mut spans = vec![
            Span::styled(
                format!(" {} ", FRAMES[spinner % FRAMES.len()]),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(name.to_owned(), tool_name_style()),
        ];
        if !tail.is_empty() {
            spans.push(Span::styled(format!(" · {tail}"), tool_summary_style()));
        }
        push_flagged(
            &mut lines,
            &mut flags,
            vec![Line::from(spans)],
            RowKind::Tool,
        );
    }
    if lines.is_empty() {
        return;
    }

    let start = lines.len().saturating_sub(area.height as usize);
    let visible_lines: Vec<Line<'static>> = lines.split_off(start);
    let visible_flags: Vec<RowKind> = flags.split_off(start);

    // Map line flags onto wrapped terminal rows so the shading covers the
    // same wrapped layout the Paragraph produces.
    let mut tool_rows = Vec::new();
    let mut focused_rows = Vec::new();
    let mut thinking_rows = Vec::new();
    for (line, flag) in visible_lines.iter().zip(visible_flags.iter()) {
        let rows = wrap_count(&plain_line_text(line), area.width as usize);
        tool_rows.extend(std::iter::repeat_n(
            matches!(*flag, RowKind::Tool | RowKind::FocusedTool),
            rows,
        ));
        focused_rows.extend(std::iter::repeat_n(*flag == RowKind::FocusedTool, rows));
        thinking_rows.extend(std::iter::repeat_n(*flag == RowKind::Thinking, rows));
    }

    frame.render_widget(
        Paragraph::new(Text::from(visible_lines)).wrap(Wrap { trim: false }),
        area,
    );
    frame.render_widget(
        RowShade {
            rows: tool_rows,
            style: tool_bg_style(),
        },
        area,
    );
    frame.render_widget(
        RowShade {
            rows: focused_rows,
            style: Style::default().bg(TOOL_FOCUSED_BG),
        },
        area,
    );
    frame.render_widget(
        RowShade {
            rows: thinking_rows,
            style: Style::default().bg(THINKING_BG),
        },
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(text: &Text<'_>) -> String {
        text.lines
            .iter()
            .map(|line| plain_line_text(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn plain_lines(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| plain_line_text(line))
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
    fn header_uses_keymap_and_styles_labels() {
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
    fn user_and_reasoning_use_colour_without_markers() {
        let user = build_user("hello\nworld");
        assert_eq!(plain(&user), "\n\nhello\nworld");
        assert_eq!(
            user.lines[2].spans[0].style.fg,
            Some(Color::White),
            "user text is white"
        );
        let reasoning = build_reasoning("checking\nfiles");
        assert_eq!(plain(&reasoning), "checking\n  files");
        assert_eq!(
            reasoning.lines[0].spans[0].style.fg,
            Some(Color::DarkGray),
            "reasoning text remains dim"
        );
    }

    #[test]
    fn summary_tail_strips_redundant_name_prefix() {
        assert_eq!(summary_tail("bash", "bash: cargo test"), "cargo test");
        assert_eq!(summary_tail("read", "read src/main.rs"), "src/main.rs");
        assert_eq!(summary_tail("bash", "bash"), "");
        assert_eq!(summary_tail("bash", "unrelated"), "unrelated");
    }

    #[test]
    fn compact_tool_is_single_status_line_plus_error_preview() {
        let ok = tool_compact_lines("bash", "bash: cargo test", true, 1_300, None);
        assert_eq!(ok.len(), 1);
        let value = plain_lines(&ok);
        assert!(
            value.contains("● bash · cargo test · 1.3s"),
            "got: {value:?}"
        );

        let failed = tool_compact_lines(
            "bash",
            "bash: cargo test",
            false,
            10,
            Some("first line\nsecond line"),
        );
        assert_eq!(failed.len(), 2);
        let value = plain_lines(&failed);
        assert!(value.contains("first line"));
        assert!(!value.contains("second line"));
        assert_eq!(failed[0].spans[0].style.fg, Some(Color::Red));
        assert_eq!(ok[0].spans[0].style.fg, Some(Color::Green));
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
        assert!(detail.contains("bash — details"));
        assert!(detail.contains("line 0"));
        assert!(detail.contains("line 204"));
        assert!(detail.contains("… 5 lines omitted"));
        assert!(detail.contains("with context"));
    }

    #[test]
    fn live_expansion_is_capped_and_points_at_dump() {
        let record = ToolRecord {
            name: "bash".into(),
            args: "{}".into(),
            summary: "cargo test".into(),
            ok: true,
            duration_ms: 10,
            output: (0..100)
                .map(|index| format!("line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
            error: None,
        };
        let lines = capped_live_detail_lines(&record, 80);
        let value = plain_lines(&lines);
        assert!(value.contains("truncated"), "got: {value:?}");
        assert!(lines.len() <= MAX_LIVE_DETAIL_ROWS);
    }

    #[test]
    fn row_shade_patches_background_without_touching_text() {
        let area = Rect::new(0, 0, 10, 3);
        let mut buffer = Buffer::empty(area);
        Paragraph::new("hi").render(area, &mut buffer);
        RowShade {
            rows: vec![true, false, true],
            style: tool_bg_style(),
        }
        .render(area, &mut buffer);
        assert_eq!(buffer[(0, 0)].bg, TOOL_BG);
        assert_eq!(buffer[(0, 0)].symbol(), "h");
        assert_ne!(buffer[(0, 1)].bg, TOOL_BG);
        assert_eq!(buffer[(9, 2)].bg, TOOL_BG, "band spans full width");
    }

    #[test]
    fn text_height_counts_final_text_lines_and_gaps() {
        let user = build_user("abcdefgh");
        assert_eq!(text_height(&user, 4), 2 + 2);
    }

    #[test]
    fn live_tail_shades_tool_rows_and_marks_focus() {
        use ratatui::backend::TestBackend;

        let record = |name: &str| ToolRecord {
            name: name.into(),
            args: "{}".into(),
            summary: format!("{name}: do things"),
            ok: true,
            duration_ms: 12,
            output: String::new(),
            error: None,
        };
        let tail = vec![
            TailTool {
                record: record("read"),
                expanded: false,
            },
            TailTool {
                record: record("bash"),
                expanded: false,
            },
        ];

        let backend = TestBackend::new(40, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_live_with_tail(
                    frame.area(),
                    &tail,
                    Some(1),
                    "",
                    "hello",
                    Some(("write", "write x.rs", 0)),
                    frame,
                );
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let bg_at = |row: u16| buffer[(0, row)].bg;
        // gap, tool, gap, gap, focused tool, gap, text, gap, running tool.
        assert_eq!(bg_at(1), TOOL_BG, "first tail tool shaded");
        assert_eq!(bg_at(4), TOOL_FOCUSED_BG, "focused tail tool highlighted");
        assert_eq!(buffer[(0, 4)].symbol(), ">", "focus marker present");
        assert_ne!(bg_at(6), TOOL_BG, "pending text not shaded");
        assert_eq!(bg_at(8), TOOL_BG, "running tool shaded");
        assert_eq!(buffer[(39, 1)].bg, TOOL_BG, "band spans the full width");
        assert_eq!(buffer[(3, 4)].symbol(), "●", "dot shifted by focus marker");
        assert_eq!(buffer[(3, 4)].fg, Color::Green);
    }
}
