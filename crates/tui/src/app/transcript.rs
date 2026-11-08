//! Transcript retention and the rendering of final entries (messages, tool
//! lines, metadata) into terminal rows.

use super::*;

/// Keep the session-global chrome when a conversation switch replaces the
/// transcript: the startup banner, metadata lines, and the most recent
/// boundary separator. Everything else was already printed into scrollback;
/// it only loses its place in the resize-repaint source, not on screen.
pub(super) fn retain_chrome(transcript: &mut Vec<Entry>) {
    let last_separator = transcript
        .iter()
        .rposition(|entry| matches!(entry, Entry::Separator { .. }));
    let mut index = 0usize;
    transcript.retain(|entry| {
        let keep = match entry {
            Entry::Banner { .. } | Entry::Metadata { .. } => true,
            Entry::Separator { .. } => Some(index) == last_separator,
            _ => false,
        };
        index += 1;
        keep
    });
}

/// Restore the terminal when the UI thread panics.
///
/// The hook is process-global, but only a panic on the thread that owns the
/// UI may restore it. Tokio catches panics in spawned tasks (for example the
/// agent task) and the UI keeps running; restoring cooked mode there would
/// break the live TUI. The hook holds a `Weak` reference, so it does nothing
/// after the UI is dropped, and it uses `try_lock` because the panic can
/// happen while the UI thread holds the terminal lock.
/// Byte budget for `transcript`. It is only the source for resize and Ctrl+O
/// repaints, which show one screen of history; older entries are already in
/// terminal scrollback. Without a bound a long session keeps every message
/// and tool result in memory.
pub(super) const TRANSCRIPT_BUDGET_BYTES: usize = 4 * 1024 * 1024;

/// Approximate retained bytes of one entry.
pub(super) fn entry_bytes(entry: &Entry) -> usize {
    match entry {
        Entry::Banner { .. } => 0,
        Entry::Metadata {
            cwd,
            branch,
            provider,
            model,
            context_files,
            mcp_servers,
            skills,
        } => {
            cwd.len()
                + branch.as_ref().map_or(0, String::len)
                + provider.len()
                + model.len()
                + context_files
                    .iter()
                    .chain(mcp_servers)
                    .chain(skills)
                    .map(String::len)
                    .sum::<usize>()
        }
        Entry::User { text } | Entry::Notice { text } | Entry::Error { text } => text.len(),
        Entry::Assistant {
            markdown,
            reasoning,
            ..
        } => markdown.len() + reasoning.len(),
        Entry::Tool { record } => {
            record.name.len()
                + record.summary.len()
                + record.output.len()
                + record.error.as_ref().map_or(0, String::len)
        }
        Entry::Separator { label } => label.len(),
    }
}

/// Move printed entries into the transcript, then drop the oldest
/// conversation entries while it exceeds [`TRANSCRIPT_BUDGET_BYTES`]. The
/// banner, metadata, and the newest entry are always kept.
pub(super) fn commit_to_transcript(transcript: &mut Vec<Entry>, pending: &mut Vec<Entry>) {
    if pending.is_empty() {
        return;
    }
    transcript.append(pending);
    trim_transcript(transcript, TRANSCRIPT_BUDGET_BYTES);
}

pub(super) fn trim_transcript(transcript: &mut Vec<Entry>, budget: usize) {
    let mut total = transcript.iter().map(entry_bytes).sum::<usize>();
    if total <= budget {
        return;
    }
    let last = transcript.len().saturating_sub(1);
    let mut index = 0usize;
    transcript.retain(|entry| {
        let keep = total <= budget
            || index == last
            || matches!(entry, Entry::Banner { .. } | Entry::Metadata { .. });
        if !keep {
            total = total.saturating_sub(entry_bytes(entry));
        }
        index += 1;
        keep
    });
}

/// Snapshot entries that fit the budget, newest last.
pub(super) struct SnapshotWindow {
    pub(super) kept: Vec<SessionSnapshotEntry>,
    pub(super) dropped: usize,
}

pub(super) fn snapshot_entry_bytes(entry: &SessionSnapshotEntry) -> usize {
    match entry {
        SessionSnapshotEntry::User { text } => text.len(),
        SessionSnapshotEntry::Assistant {
            markdown,
            reasoning,
        } => markdown.len() + reasoning.len(),
        SessionSnapshotEntry::Tool {
            name,
            summary,
            output,
            error,
            ..
        } => {
            name.len()
                + summary.len()
                + output
                    .len()
                    .min(crate::state::MAX_RETAINED_TOOL_OUTPUT_BYTES)
                + error.as_ref().map_or(0, |error| {
                    error.len().min(crate::state::MAX_RETAINED_TOOL_ERROR_BYTES)
                })
        }
    }
}

/// Keep the newest snapshot entries whose retained size fits `budget` (at
/// least one entry), in their original order.
pub(super) fn newest_snapshot_entries(
    entries: Vec<SessionSnapshotEntry>,
    budget: usize,
) -> SnapshotWindow {
    let mut total = 0usize;
    let mut start = entries.len();
    while start > 0 {
        let bytes = snapshot_entry_bytes(&entries[start - 1]);
        if start < entries.len() && total.saturating_add(bytes) > budget {
            break;
        }
        total = total.saturating_add(bytes);
        start -= 1;
    }
    let dropped = start;
    SnapshotWindow {
        kept: entries.into_iter().skip(dropped).collect(),
        dropped,
    }
}

/// Build the wrapped rows for one final or in-flight entry. Shared by the
/// pending path (print above the region) and the resize repaint (window of
/// history), which is why entries are stored at the source level.
/// `tools_expanded` is the global Ctrl+O state; committed entries render
/// collapsed unless the toggle is on, and the progress snapshot used the
/// same global for its in-flight tool line so both agree after a repaint.
/// Render only the newest history rows that can fit above the live region.
///
/// A resize used to render every transcript entry and discard almost all of
/// it afterward. Walking backward lets a large session stop as soon as the
/// viewport is full. The marker intentionally does not claim an exact hidden
/// row count: discovering that count would require the full-history pass this
/// helper is designed to avoid.
pub(super) fn history_window(
    entries: &[Entry],
    budget: usize,
    width: usize,
    theme: Theme,
    tools_expanded: bool,
) -> Vec<Line<'static>> {
    if budget == 0 || entries.is_empty() {
        return Vec::new();
    }

    let mut newest_first = Vec::<Vec<Line<'static>>>::new();
    let mut used = 0usize;
    let mut hidden = false;
    for (index, entry) in entries.iter().enumerate().rev() {
        let lines = entry_lines(entry, width, theme, tools_expanded);
        let gap = usize::from(!newest_first.is_empty()) * render::SECTION_GAP;
        if used.saturating_add(gap).saturating_add(lines.len()) <= budget {
            used = used.saturating_add(gap).saturating_add(lines.len());
            newest_first.push(lines);
            continue;
        }

        // A single entry can be taller than the viewport. Keep its newest
        // rows rather than rendering older entries that cannot be visible.
        let available = budget.saturating_sub(used.saturating_add(gap));
        let entry_had_lines = !lines.is_empty();
        if available > 0 {
            let start = lines.len().saturating_sub(available);
            let mut suffix = Vec::with_capacity(available.min(lines.len()));
            suffix.extend(lines.into_iter().skip(start));
            newest_first.push(suffix);
        }
        hidden = index > 0 || entry_had_lines;
        break;
    }

    let mut result = Vec::new();
    for (index, lines) in newest_first.iter().rev().enumerate() {
        if index > 0 {
            render::push_blank(&mut result, render::SECTION_GAP);
        }
        result.extend(lines.iter().cloned());
    }
    if hidden {
        let marker = Line::from(Span::styled(
            "… older rows above",
            Style::default()
                .fg(theme.dim_text)
                .add_modifier(Modifier::DIM),
        ));
        if result.len() >= budget {
            result.drain(..result.len() - budget.saturating_sub(1));
        }
        result.insert(0, marker);
    }
    result
}

pub(super) fn entry_lines(
    entry: &Entry,
    width: usize,
    theme: Theme,
    tools_expanded: bool,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    match entry {
        Entry::Banner { title_order } => render::welcome_lines(width, theme, title_order),
        Entry::Metadata {
            cwd,
            branch,
            provider,
            model,
            context_files,
            mcp_servers,
            skills,
        } => metadata_lines(
            cwd,
            branch.as_deref(),
            provider,
            model,
            context_files,
            mcp_servers,
            skills,
            theme,
        )
        .into_iter()
        .map(|line| render::fit_line_to_width(&line, width))
        .collect(),
        Entry::User { text } => render::user_lines(text, theme, width),
        Entry::Assistant {
            markdown,
            reasoning,
            continued,
        } => {
            let mut lines = Vec::new();
            if !reasoning.is_empty() {
                lines.extend(render::reasoning_lines(reasoning, theme, width));
            }
            if !markdown.is_empty() {
                if !lines.is_empty() {
                    render::push_blank(&mut lines, render::BLOCK_GAP);
                }
                lines.extend(render::markdown_lines_continued(
                    markdown, theme, width, *continued,
                ));
            }
            lines
        }
        Entry::Tool { record } => tool_lines(record, tools_expanded, width, theme),
        Entry::Notice { text } => render::notice_lines(text, theme, width),
        Entry::Error { text } => render::error_lines(text, theme, width),
        Entry::Separator { label } => vec![separator_line(label, width, theme)],
    }
}

/// The simplified tool call: one line with the tool type and its primary
/// parameter — `$ git status`, `read file.txt` — plus a duration on success
/// or a `✗` error preview on failure. Collapsed this is the compact line;
/// expanded it adds the bounded output tail, later error lines, and a
/// `running…` marker, each indented two spaces.
pub(super) fn tool_lines(
    record: &ToolRecord,
    expanded: bool,
    width: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    // Agent summaries read `bash: <command>` / `read <path>` / …; bash reads
    // as a shell line, every other tool as `<verb> <param>` with the verb in
    // the same accent as the `$` so all tool lines share one signature.
    let marker_style = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    let command = record
        .summary
        .strip_prefix("bash:")
        .map(|rest| rest.trim_start())
        .filter(|rest| !rest.is_empty());
    match command {
        Some(command) => {
            spans.push(Span::styled("$ ", marker_style));
            spans.push(Span::styled(
                command.to_owned(),
                Style::default().fg(theme.primary_text),
            ));
        }
        None => {
            let summary = record.summary.trim();
            let (verb, rest) = summary.split_once(' ').unwrap_or((summary, ""));
            spans.push(Span::styled(verb.to_owned(), marker_style));
            if !rest.is_empty() {
                spans.push(Span::styled(
                    format!(" {rest}"),
                    Style::default().fg(theme.primary_text),
                ));
            }
        }
    }
    match record.status {
        ToolStatus::Running => {}
        ToolStatus::Success => spans.push(Span::styled(
            format!(" · {}", render::duration_text(record.duration_ms)),
            Style::default()
                .fg(theme.dim_text)
                .add_modifier(Modifier::DIM),
        )),
        ToolStatus::Failure => {
            let preview = record
                .error
                .as_deref()
                .and_then(|error| error.lines().next())
                .unwrap_or("failed");
            spans.push(Span::styled("  ✗ ", Style::default().fg(theme.error)));
            spans.push(Span::styled(
                preview.to_owned(),
                Style::default().fg(theme.error),
            ));
        }
    }
    // The marker/verb is part of the spans, so no extra message prefix is
    // applied; the wrapper only reflows the line to the available width.
    let summary_lines = render::wrap_text(
        &Text::from(Line::from(spans)),
        width.saturating_sub(INPUT_PREFIX_WIDTH).max(1),
        Style::default(),
    );
    if !expanded {
        return summary_lines;
    }

    // Expanded: the compact line, then the bounded output tail, later error
    // lines, and a trailing `running…` marker, each indented two spaces
    // and dimmed so they read as details under the summary.
    let dim_style = Style::default()
        .fg(theme.dim_text)
        .add_modifier(Modifier::DIM);
    let mut lines = summary_lines;
    if !record.output.is_empty() {
        for tail in render::output_tail_after(&record.output, record.omitted_lines) {
            push_detail_line(&mut lines, &tail, dim_style, width);
        }
    }
    if let Some(error) = record.error.as_deref() {
        for detail in error.lines().skip(1).take(render::DEFAULT_TAIL_LINES) {
            push_detail_line(&mut lines, detail, Style::default().fg(theme.error), width);
        }
    }
    if matches!(record.status, ToolStatus::Running) {
        push_detail_line(&mut lines, "running…", dim_style, width);
    }
    lines
}

/// One indented detail row under an expanded tool line, wrapped to the
/// content width with the given style applied to every span.
pub(super) fn push_detail_line(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    style: Style,
    width: usize,
) {
    for wrapped in render::wrap_text(
        &Text::from(Line::from(Span::styled(text.to_owned(), style))),
        width.saturating_sub(INPUT_PREFIX_WIDTH).max(1),
        Style::default(),
    ) {
        lines.push(Line::from(
            std::iter::once(Span::raw("  "))
                .chain(wrapped.spans)
                .collect::<Vec<_>>(),
        ));
    }
}

/// How many context/MCP/skill entries each header row shows before folding into
/// `+N more`. Skills are cheap to accumulate (one per directory), so the cap
/// keeps a skill-heavy workspace from pushing the transcript out of view.
pub(super) const METADATA_MAX_ENTRIES: usize = 4;

pub(super) fn fold_entries(entries: &[String]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let shown = entries.len().min(METADATA_MAX_ENTRIES);
    let mut text = entries[..shown]
        .iter()
        .map(|entry| render::sanitize_terminal_text(entry))
        .collect::<Vec<_>>()
        .join(", ");
    let hidden = entries.len() - shown;
    if hidden > 0 {
        let _ = write!(text, " · +{hidden} more");
    }
    Some(text)
}

/// An accent bar and aligned, muted labels make startup details easy to scan.
/// Values use the terminal's normal text colour; optional context, skill, and
/// MCP rows remain capped at [`METADATA_MAX_ENTRIES`] names.
#[expect(
    clippy::too_many_arguments,
    reason = "the renderer keeps each metadata row explicit for focused tests"
)]
pub(super) fn metadata_lines(
    cwd: &str,
    branch: Option<&str>,
    provider: &str,
    model: &str,
    context_files: &[String],
    mcp_servers: &[String],
    skills: &[String],
    theme: Theme,
) -> Vec<Line<'static>> {
    let cwd = render::sanitize_terminal_text(cwd);
    let cwd = match branch.map(render::sanitize_terminal_text) {
        Some(branch) => format!("{cwd}  ({branch})"),
        None => cwd,
    };
    let mut lines = vec![
        metadata_row("cwd", cwd, theme),
        metadata_row("provider", render::sanitize_terminal_text(provider), theme),
        metadata_row("model", render::sanitize_terminal_text(model), theme),
    ];
    for (label, entries) in [
        ("context", context_files),
        ("skills", skills),
        ("mcps", mcp_servers),
    ] {
        if let Some(text) = fold_entries(entries) {
            lines.push(metadata_row(label, text, theme));
        }
    }
    lines
}

pub(super) fn metadata_row(label: &str, value: String, theme: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled("│ ", Style::default().fg(theme.accent)),
        Span::styled(
            format!("{label:<8}  "),
            Style::default()
                .fg(theme.dim_text)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(value, Style::default().fg(theme.primary_text)),
    ])
}

/// A left-oriented rule marking a session boundary. An empty label is the
/// unlabelled initial boundary below the full startup header.
pub(super) fn separator_line(label: &str, width: usize, theme: Theme) -> Line<'static> {
    let label = render::sanitize_terminal_text(label);
    let text = if label.is_empty() {
        "─".repeat(width)
    } else {
        let prefix = format!("── {label} ");
        let prefix = ghost_to_width(&prefix, width);
        let remaining = width.saturating_sub(UnicodeWidthStr::width(prefix.as_str()));
        format!("{prefix}{}", "─".repeat(remaining))
    };
    Line::from(Span::styled(
        text,
        Style::default()
            .fg(theme.dim_text)
            .add_modifier(Modifier::DIM),
    ))
}

/// The busy indicator row: an animated marker plus the activity label.
pub(super) fn activity_line_with_label(label: &str, spinner: usize, theme: Theme) -> Line<'static> {
    let marker = render::ACTIVITY_FRAMES[spinner % render::ACTIVITY_FRAMES.len()];
    Line::from(vec![
        Span::styled(format!("{marker} "), Style::default().fg(theme.accent)),
        Span::styled(
            label.to_owned(),
            Style::default()
                .fg(theme.dim_text)
                .add_modifier(Modifier::DIM),
        ),
    ])
}
