//! Reflow the boxed tables produced by tui-markdown before ordinary line wrapping.
//! Its table renderer sizes columns to their longest cells without a width input;
//! wrapping the completed box as prose splits borders from their cells.

use ratatui_core::style::Style;
use ratatui_core::text::{Line, Span, Text};
use unicode_width::UnicodeWidthStr;

use super::wrap_text;

pub(super) fn fit_tables(text: Text<'static>, width: usize) -> Text<'static> {
    let mut output = Vec::new();
    let mut lines = text.lines.into_iter().peekable();
    while let Some(line) = lines.next() {
        let Some(top_index) = line
            .spans
            .iter()
            .position(|span| span.content.starts_with('┌') && span.content.ends_with('┐'))
        else {
            output.push(line);
            continue;
        };
        let prefix = line.spans[..top_index].to_vec();
        let top = &line.spans[top_index];
        let columns = top.content.chars().filter(|&ch| ch == '┬').count() + 1;
        let border_style = top.style;
        let mut group = vec![line];
        while let Some(next) = lines.peek() {
            let finished = next
                .spans
                .iter()
                .any(|span| span.content.starts_with('└') && span.content.ends_with('┘'));
            // Do not swallow unrelated blocks if the Markdown is incomplete.
            if !next.spans.iter().any(|span| span.content.as_ref() == "│")
                && !next
                    .spans
                    .iter()
                    .any(|span| span.content.starts_with('├') || span.content.starts_with('└'))
            {
                break;
            }
            group.push(lines.next().expect("peeked line"));
            if finished {
                break;
            }
        }
        let valid = group
            .last()
            .is_some_and(|line| line.spans.iter().any(|span| span.content.starts_with('└')));
        if !valid || columns == 0 {
            output.extend(group);
            continue;
        }
        let continuation = group
            .get(1)
            .map(|line| {
                line.spans
                    .iter()
                    .take_while(|span| {
                        span.content.as_ref() != "│" && !span.content.starts_with('└')
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let indent = prefix_width(&prefix).max(prefix_width(&continuation));
        let available = width.saturating_sub(indent);
        let rows = group
            .iter()
            .filter(|line| line.spans.iter().any(|span| span.content.as_ref() == "│"))
            .map(|line| cells(line, columns))
            .collect::<Option<Vec<_>>>();
        let Some(rows) = rows else {
            output.extend(group);
            continue;
        };
        let alignments = (0..columns)
            .map(|column| alignment(&group, column))
            .collect::<Vec<_>>();
        let mut widths = vec![1usize; columns];
        for row in &rows {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(
                    cell.iter()
                        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                        .sum(),
                );
            }
        }
        // Leave a fitting table alone, including its original alignment.
        if group.iter().all(|line| {
            line.spans
                .iter()
                .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                .sum::<usize>()
                <= width
        }) {
            output.extend(group);
            continue;
        }
        // Each boxed column needs one content column, two padding columns,
        // and a shared border. Below that limit, use an unboxed vertical layout.
        if available < columns.saturating_mul(4).saturating_add(1) {
            let mut first = true;
            let body = if rows.len() > 1 {
                &rows[1..]
            } else {
                &rows[..]
            };
            for (row_index, row) in body.iter().enumerate() {
                if row_index > 0 {
                    output.push(Line::from(""));
                }
                for (column, cell) in row.iter().enumerate() {
                    let mut content = Vec::new();
                    if rows.len() > 1 && !rows[0][column].is_empty() {
                        content.extend(rows[0][column].iter().cloned());
                        content.push(Span::raw(": "));
                    }
                    content.extend(cell.iter().cloned());
                    let wrapped = wrap_text(
                        &Text::from(Line::from(content)),
                        available.max(1),
                        Style::default(),
                    );
                    for line in wrapped {
                        output.push(with_prefix(
                            if first { &prefix } else { &continuation },
                            line,
                        ));
                        first = false;
                    }
                }
            }
            continue;
        }
        let budget = available - (3 * columns + 1);
        if widths.iter().sum::<usize>() > budget {
            // Cap the widest columns together, without one iteration per
            // character in a potentially very long cell.
            let mut low = 1;
            let mut high = *widths.iter().max().unwrap_or(&1);
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if widths.iter().map(|&w| w.min(mid)).sum::<usize>() <= budget {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            let natural = widths.clone();
            for width in &mut widths {
                *width = (*width).min(low);
            }
            let mut remaining = budget - widths.iter().sum::<usize>();
            for (width, original) in widths.iter_mut().zip(natural) {
                if remaining == 0 {
                    break;
                }
                if *width < original {
                    *width += 1;
                    remaining -= 1;
                }
            }
        }
        output.push(with_prefix(
            &prefix,
            border(&widths, '┌', '┬', '┐', border_style),
        ));
        for (index, row) in rows.iter().enumerate() {
            if index == 1 {
                output.push(with_prefix(
                    &continuation,
                    border(&widths, '├', '┼', '┤', border_style),
                ));
            }
            let wrapped = row
                .iter()
                .zip(&widths)
                .map(|(cell, &w)| {
                    wrap_text(&Text::from(Line::from(cell.clone())), w, Style::default())
                })
                .collect::<Vec<_>>();
            let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
            for row_line in 0..height {
                let mut spans = vec![Span::styled("│", border_style)];
                for (column, &w) in widths.iter().enumerate() {
                    let cell = wrapped[column].get(row_line);
                    let content_width = cell.map_or(0, |line| {
                        line.spans
                            .iter()
                            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                            .sum()
                    });
                    let extra = w - content_width;
                    let left = match alignments[column] {
                        Alignment::Left => 0,
                        Alignment::Right => extra,
                        Alignment::Center => extra / 2,
                    };
                    spans.push(Span::raw(" ".repeat(left + 1)));
                    if let Some(cell) = cell {
                        spans.extend(cell.spans.iter().cloned());
                    }
                    spans.push(Span::raw(" ".repeat(extra - left + 1)));
                    spans.push(Span::styled("│", border_style));
                }
                output.push(with_prefix(&continuation, Line::from(spans)));
            }
        }
        output.push(with_prefix(
            &continuation,
            border(&widths, '└', '┴', '┘', border_style),
        ));
    }
    Text::from(output)
}

#[derive(Clone, Copy)]
enum Alignment {
    Left,
    Center,
    Right,
}

fn alignment(group: &[Line<'_>], column: usize) -> Alignment {
    for line in group {
        let borders = line
            .spans
            .iter()
            .enumerate()
            .filter_map(|(index, span)| (span.content.as_ref() == "│").then_some(index))
            .collect::<Vec<_>>();
        let (Some(&start), Some(&end)) = (borders.get(column), borders.get(column + 1)) else {
            continue;
        };
        let value = line.spans[start + 1..end]
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        let left = value.len() - value.trim_start_matches(' ').len();
        let right = value.len() - value.trim_end_matches(' ').len();
        if left > 1 && right > 1 {
            return Alignment::Center;
        }
        if left > 1 {
            return Alignment::Right;
        }
        if right > 1 {
            return Alignment::Left;
        }
    }
    Alignment::Left
}

fn prefix_width(prefix: &[Span<'_>]) -> usize {
    prefix
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum()
}

fn with_prefix(prefix: &[Span<'static>], line: Line<'static>) -> Line<'static> {
    Line::from(prefix.iter().cloned().chain(line.spans).collect::<Vec<_>>())
}

fn border(widths: &[usize], left: char, middle: char, right: char, style: Style) -> Line<'static> {
    let mut value = left.to_string();
    for (index, width) in widths.iter().enumerate() {
        value.push_str(&"─".repeat(width + 2));
        value.push(if index + 1 == widths.len() {
            right
        } else {
            middle
        });
    }
    Line::from(Span::styled(value, style))
}

fn cells(line: &Line<'static>, count: usize) -> Option<Vec<Vec<Span<'static>>>> {
    let mut cells = Vec::new();
    let mut current = None;
    for span in &line.spans {
        if span.content.as_ref() == "│" {
            if let Some(cell) = current.replace(Vec::new()) {
                cells.push(trim_cell(cell));
            }
        } else if let Some(cell) = &mut current {
            cell.push(span.clone());
        }
    }
    (cells.len() == count).then_some(cells)
}

fn trim_cell(mut cell: Vec<Span<'static>>) -> Vec<Span<'static>> {
    // The library's alignment and box padding are whitespace spans. Retain
    // inline styles and rebuild only the layout whitespace for the new width.
    for span in &mut cell {
        span.content = span.content.trim_start_matches(' ').to_owned().into();
        if !span.content.is_empty() {
            break;
        }
    }
    for span in cell.iter_mut().rev() {
        span.content = span.content.trim_end_matches(' ').to_owned().into();
        if !span.content.is_empty() {
            break;
        }
    }
    cell.retain(|span| !span.content.is_empty());
    cell
}
