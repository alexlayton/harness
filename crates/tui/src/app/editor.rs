//! The prompt editor: pure helpers over a `String` and a byte cursor, plus the
//! wrapped input layout and its inverse for mouse clicks.

use super::*;

/// The visual rows of the input line plus the cursor's position within them.
pub(super) struct InputLayout {
    pub(super) rows: Vec<Row>,
    pub(super) cursor_row: usize,
    pub(super) cursor_col: usize,
}

/// Many terminals deliver pasted line breaks as CR (or CRLF). The sanitizer
/// drops CR as a control character, which joined every pasted line into
/// one; turn both forms into LF first.
pub(super) fn normalize_paste_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Wrap the input draft into visual rows (first row prefixed with `› `,
/// continuations indented to match) and locate the cursor inside them. The
/// real terminal cursor is placed at `cursor_col` columns into `cursor_row`.
///
/// `usage_trailer` is appended, dim, to the empty-input placeholder row;
/// nothing is shown while typing (the counter reappears after a submit
/// because the input empties again). `ghost` is the fish-style dim suffix
/// preview painted after the cursor (only at end-of-input, clamped to the row
/// width), and `completion_hint` is one dim candidate row below the input.
#[cfg(test)]
pub(super) fn input_layout(
    input: &str,
    cursor: usize,
    width: usize,
    theme: Theme,
    usage_trailer: &str,
    ghost: &str,
    completion_hint: &str,
) -> InputLayout {
    input_layout_with_row_width(
        input,
        cursor,
        width,
        width.saturating_add(INPUT_PREFIX_WIDTH),
        theme,
        usage_trailer,
        ghost,
        completion_hint,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn input_layout_with_row_width(
    input: &str,
    cursor: usize,
    width: usize,
    row_width: usize,
    theme: Theme,
    usage_trailer: &str,
    ghost: &str,
    completion_hint: &str,
) -> InputLayout {
    let width = width.max(1);
    let row_width = row_width.max(1);
    let cursor = cursor.min(input.len());
    let prefix_style = ContentStyle::default()
        .fg(theme.accent)
        .add_modifier(Attribute::Bold);
    let text_style = ContentStyle::default().fg(theme.primary_text);
    let dim_style = ContentStyle::default()
        .fg(theme.dim_text)
        .add_modifier(Attribute::Dim);

    if input.is_empty() {
        // Placeholder row; the terminal cursor sits right after the prefix.
        // When usage is available the counter trailer trails it, dim.
        let mut spans = vec![
            Run::new(INPUT_PREFIX, prefix_style),
            Run::new(PLACEHOLDER, dim_style),
        ];
        if !usage_trailer.is_empty() {
            spans.push(Run::new(usage_trailer.to_owned(), dim_style));
        }
        let mut rows = vec![render::fit_line_to_width(&Row::from(spans), row_width)];
        if !completion_hint.is_empty() {
            rows.push(render::fit_line_to_width(
                &Row::from(vec![
                    Run::plain(INPUT_CONTINUATION),
                    Run::new(completion_hint.to_owned(), dim_style),
                ]),
                row_width,
            ));
        }
        return InputLayout {
            rows,
            cursor_row: 0,
            cursor_col: 0,
        };
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut cursor_row = 0usize;
    let mut cursor_col = 0usize;

    let mut byte_base = 0usize;
    for logical in input.split('\n') {
        let line_end = byte_base + logical.len();
        let cursor_here = cursor >= byte_base && cursor <= line_end;
        // Layout units are grapheme clusters, matching editing and wrap.
        let chars: Vec<(&str, usize)> = logical
            .graphemes(true)
            .map(|cluster| (cluster, render::grapheme_width(cluster)))
            .collect();
        let cursor_char = if cursor_here {
            Some(input[byte_base..cursor].graphemes(true).count())
        } else {
            None
        };

        // Greedy wrap; rows are append-stable, so the cursor can be located
        // afterwards by walking the row boundaries.
        let mut visual: Vec<Vec<(&str, usize)>> = Vec::new();
        let mut current: Vec<(&str, usize)> = Vec::new();
        let mut current_width = 0usize;
        for &(character, character_width) in &chars {
            if character_width > width {
                if !current.is_empty() {
                    visual.push(std::mem::take(&mut current));
                    current_width = 0;
                }
                continue;
            }
            if character_width > 0 && current_width + character_width > width {
                visual.push(std::mem::take(&mut current));
                current_width = 0;
            }
            current.push((character, character_width));
            current_width += character_width;
        }
        visual.push(current);

        if let Some(cursor_char) = cursor_char {
            if cursor_char >= chars.len() {
                // End of this logical line. A row filled to exactly the wrap
                // width gets a fresh row for the cursor, mirroring
                // `render::prompt_layout`.
                let last = visual.len() - 1;
                let last_width: usize = visual[last].iter().map(|(_, w)| *w).sum();
                if last_width >= width && !visual[last].is_empty() {
                    cursor_row = rows.len() + visual.len();
                    cursor_col = 0;
                    visual.push(Vec::new());
                } else {
                    cursor_row = rows.len() + last;
                    cursor_col = last_width;
                }
            } else {
                let mut index = 0usize;
                for (row_index, row) in visual.iter().enumerate() {
                    if index + row.len() > cursor_char {
                        cursor_row = rows.len() + row_index;
                        cursor_col = row.iter().take(cursor_char - index).map(|(_, w)| *w).sum();
                        break;
                    }
                    index += row.len();
                }
            }
        }

        for (row_index, row) in visual.iter().enumerate() {
            let text: String = row.iter().map(|(c, _)| *c).collect();
            let lead = if rows.is_empty() && row_index == 0 {
                Run::new(INPUT_PREFIX, prefix_style)
            } else {
                Run::plain(INPUT_CONTINUATION)
            };
            rows.push(render::fit_line_to_width(
                &Row::from(vec![lead, Run::new(text, text_style)]),
                row_width,
            ));
        }

        byte_base = line_end + 1;
    }

    if rows.is_empty() {
        rows.push(Row::from(Run::new(INPUT_PREFIX, prefix_style)));
    }

    // The list hint sits below the draft so opening completion never moves
    // the input or its real terminal cursor.
    if !completion_hint.is_empty() {
        rows.push(render::fit_line_to_width(
            &Row::from(vec![
                Run::plain(INPUT_CONTINUATION),
                Run::new(completion_hint.to_owned(), dim_style),
            ]),
            row_width,
        ));
    }
    // Ghost preview: append the dim suffix after the typed text on the
    // cursor's row. Only at end-of-input, clamped to the remaining width so
    // the row never overflows; the cursor position is untouched.
    if !ghost.is_empty()
        && cursor >= input.len()
        && let Some(row) = rows.get_mut(cursor_row)
    {
        let used = row
            .runs
            .iter()
            .map(|span| render::text_width(span.text.as_str()))
            .sum();
        let available = row_width.saturating_sub(used);
        let fits = ghost_to_width(ghost, available);
        if !fits.is_empty() {
            row.runs.push(Run::new(fits, dim_style));
        }
    }

    InputLayout {
        rows,
        cursor_row,
        cursor_col,
    }
}

/// Fit completion values onto exactly one visual row. Candidates begin at the
/// highlighted item and wrap around, so the value Tab will accept is never
/// hidden behind the `… +N` counter.
pub(super) fn fit_completion_hint(
    candidates: &[Candidate],
    selected: usize,
    width: usize,
) -> String {
    if candidates.is_empty() || width == 0 {
        return String::new();
    }
    let selected = selected.min(candidates.len() - 1);
    let values = candidates[selected..]
        .iter()
        .chain(&candidates[..selected])
        .map(|candidate| candidate.value.as_str())
        .collect::<Vec<_>>();

    let mut best = String::new();
    for shown in 1..=values.len() {
        let mut hint = values[..shown].join(" · ");
        let hidden = values.len() - shown;
        if hidden > 0 {
            let _ = write!(hint, " … +{hidden}");
        }
        if UnicodeWidthStr::width(hint.as_str()) > width {
            break;
        }
        best = hint;
    }
    if !best.is_empty() {
        return best;
    }

    // Even one provider-qualified ID can be wider than a small terminal.
    // Reserve the last cell for an ellipsis rather than letting the terminal
    // soft-wrap a row the live-region model intentionally counts as one.
    if width == 1 {
        return "…".into();
    }
    let mut hint = ghost_to_width(values[0], width - 1);
    hint.push('…');
    hint
}

/// Truncate `text` (in display columns) to fit `width`, only ever cutting at
/// whole characters. Returns empty when nothing fits.
pub(super) fn ghost_to_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut result = String::new();
    let mut used = 0usize;
    for cluster in text.graphemes(true) {
        let cluster_width = render::grapheme_width(cluster).max(1);
        if used + cluster_width > width {
            break;
        }
        result.push_str(cluster);
        used += cluster_width;
    }
    result
}

/// The byte offset of the `column`-th character (0-based) into `input`.
pub(super) fn byte_index_at_char(input: &str, column: usize) -> usize {
    input
        .char_indices()
        .nth(column)
        .map(|(index, _)| index)
        .unwrap_or(input.len())
}

/// Byte offset in `input` for a click on layout row `row` (counted exactly as
/// [`input_layout_with_row_width`] wraps for the same `cursor`) at display
/// column `col` within the text area. A click past the end of a row lands at
/// its end; rows beyond the text (such as the completion hint) yield `None`.
///
/// `cursor` matters because the layout adds an empty row for a cursor parked
/// at the end of a logical line that exactly fills the wrap width; hit testing
/// must count that row too or every row below it would be off by one.
pub(super) fn input_offset_at(
    input: &str,
    cursor: usize,
    width: usize,
    row: usize,
    col: usize,
) -> Option<usize> {
    let width = width.max(1);
    let cursor = cursor.min(input.len());
    let mut rows_seen = 0usize;
    let mut base = 0usize;
    for logical in input.split('\n') {
        let line_end = base + logical.len();
        // Per visual row: (byte offset, width) per cluster plus the byte
        // offset just past the row, greedily wrapped like the layout.
        let mut visual: Vec<(Vec<(usize, usize)>, usize)> = Vec::new();
        let mut current: Vec<(usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut offset = base;
        for cluster in logical.graphemes(true) {
            let cluster_width = render::grapheme_width(cluster);
            if cluster_width > width {
                // The layout drops oversized clusters and breaks the row.
                if !current.is_empty() {
                    visual.push((std::mem::take(&mut current), offset));
                    used = 0;
                }
                offset += cluster.len();
                continue;
            }
            if cluster_width > 0 && used + cluster_width > width {
                visual.push((std::mem::take(&mut current), offset));
                used = 0;
            }
            current.push((offset, cluster_width));
            used += cluster_width;
            offset += cluster.len();
        }
        let last_full = used >= width && !current.is_empty();
        visual.push((current, line_end));
        if last_full && cursor == line_end {
            // Cursor-only row, mirroring the layout.
            visual.push((Vec::new(), line_end));
        }
        if row < rows_seen + visual.len() {
            let index = row - rows_seen;
            let (cells, row_end) = &visual[index];
            let mut acc = 0usize;
            for &(start, cell) in cells {
                if col < acc + cell.max(1) {
                    return Some(start);
                }
                acc += cell;
            }
            // Past the last cluster: end of this visual row, which for a
            // wrapped row is where the next row's text begins.
            let next_start = visual
                .get(index + 1)
                .and_then(|(next, _)| next.first())
                .map(|&(start, _)| start);
            return Some(next_start.unwrap_or(*row_end));
        }
        rows_seen += visual.len();
        base = line_end + 1;
    }
    None
}

/// Clip the input rows to a window containing the cursor when the draft is
/// taller than the screen can show. Returns the visible rows and the cursor
/// row within them.
pub(super) fn clip_input(
    input: &InputLayout,
    height: usize,
    busy: bool,
    tool_rows: usize,
) -> (Vec<Row>, usize, usize) {
    let cap = height
        .saturating_sub(usize::from(busy) + tool_rows + 2)
        .max(1);
    if input.rows.len() <= cap {
        return (input.rows.clone(), input.cursor_row, 0);
    }
    // Window pinned so the cursor row stays visible, biased toward showing
    // as much above it as fits.
    let start = input
        .cursor_row
        .saturating_sub(cap - 1)
        .min(input.rows.len() - cap);
    (
        input.rows[start..start + cap].to_vec(),
        input.cursor_row - start,
        start,
    )
}

/// Insert text at the cursor, normalizing CRLF/CR paste payloads to `\n`.
pub(super) fn insert_text(input: &mut String, cursor: &mut usize, text: &str) {
    if text.is_empty() {
        return;
    }
    let text = if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_owned()
    };
    input.insert_str(*cursor, &text);
    *cursor += text.len();
}

pub(super) fn delete_backward(input: &mut String, cursor: &mut usize) {
    if let Some(cluster) = input[..*cursor].graphemes(true).next_back() {
        let len = cluster.len();
        *cursor -= len;
        input.drain(*cursor..*cursor + len);
    }
}

pub(super) fn delete_forward(input: &mut String, cursor: &mut usize) {
    if let Some(len) = input[*cursor..].graphemes(true).next().map(str::len) {
        input.drain(*cursor..*cursor + len);
    }
}

pub(super) fn move_left(input: &str, cursor: &mut usize) {
    if let Some(cluster) = input[..*cursor].graphemes(true).next_back() {
        *cursor -= cluster.len();
    }
}

pub(super) fn move_right(input: &str, cursor: &mut usize) {
    if let Some(len) = input[*cursor..].graphemes(true).next().map(str::len) {
        *cursor += len;
    }
}

/// Byte offsets of the start and end of the logical line containing `cursor`.
pub(super) fn line_bounds(input: &str, cursor: usize) -> (usize, usize) {
    let start = input[..cursor].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = input[cursor..]
        .find('\n')
        .map(|i| cursor + i)
        .unwrap_or(input.len());
    (start, end)
}

/// Move the cursor one logical line up (`delta < 0`) or down, preserving the
/// character column. Returns `None` at the outer edges.
pub(super) fn vertical_move(input: &str, cursor: usize, delta: i32) -> Option<usize> {
    let before = &input[..cursor];
    let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let column = before[line_start..].graphemes(true).count();
    let line_index = before.matches('\n').count() as i32;
    let total_lines = input.matches('\n').count() as i32 + 1;
    let target = line_index + delta;
    if target < 0 || target >= total_lines {
        return None;
    }
    let mut position = 0usize;
    for _ in 0..target {
        position += input[position..].find('\n').expect("target line exists") + 1;
    }
    let line_end = input[position..]
        .find('\n')
        .map(|i| position + i)
        .unwrap_or(input.len());
    let byte_column: usize = input[position..line_end]
        .graphemes(true)
        .take(column)
        .map(str::len)
        .sum();
    Some(position + byte_column)
}
