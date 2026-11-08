//! Width-aware layout: [`Document`] + [`Theme`] → terminal [`Row`]s.

use crossterm::style::{ContentStyle, Stylize};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::doc::{AlertKind, Alignment, Block, BlockKind, Cell, Document, Inline, ListItem};
use crate::row::{Row, Run, str_width};
use crate::syntax_theme::SyntaxTheme;
use crate::theme::{Theme, patch};
use crate::wrap::wrap;

/// Blank rows between sibling blocks.
const BLOCK_GAP: usize = 1;

/// Lay out a whole document. Every returned row is at most `width` columns.
pub fn render(document: &Document, theme: &Theme, width: usize) -> Vec<Row> {
    render_blocks(&document.blocks, theme, width)
}

/// Lay out a slice of top-level blocks, e.g. only the unstable tail of a
/// streaming document (see [`Document::stable_len`]). Separate calls do not
/// insert the inter-block gap between their outputs; callers joining them
/// should add one blank row.
pub fn render_blocks(blocks: &[Block], theme: &Theme, width: usize) -> Vec<Row> {
    let width = width.max(1);
    Layout { theme }
        .blocks(blocks, width, theme.text, 0, BLOCK_GAP)
        .into_iter()
        .map(|row| {
            if row.width() > width {
                row.fit(width)
            } else {
                row
            }
        })
        .collect()
}

/// Parse and lay out in one call.
pub fn render_str(source: &str, theme: &Theme, width: usize) -> Vec<Row> {
    render(&crate::parse(source), theme, width)
}

struct Layout<'t> {
    theme: &'t Theme,
}

impl Layout<'_> {
    fn blocks(
        &self,
        blocks: &[Block],
        width: usize,
        base: ContentStyle,
        depth: usize,
        gap: usize,
    ) -> Vec<Row> {
        let mut out = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            if index > 0 {
                out.extend(std::iter::repeat_with(Row::new).take(gap));
            }
            out.extend(self.block(&block.kind, width, base, depth));
        }
        out
    }

    fn block(&self, kind: &BlockKind, width: usize, base: ContentStyle, depth: usize) -> Vec<Row> {
        let theme = self.theme;
        match kind {
            BlockKind::Paragraph(content) => wrap(&self.runs(content, base), width),
            BlockKind::Heading { level, content } => {
                let style = patch(base, theme.heading(*level));
                let mut runs = Vec::new();
                if theme.symbols.heading_marker {
                    runs.push(Run::new(
                        format!("{} ", "#".repeat(usize::from(*level))),
                        style,
                    ));
                }
                runs.extend(self.runs(content, style));
                wrap(&runs, width)
            }
            BlockKind::CodeBlock { lang, code } => {
                self.code_block(lang.as_deref(), code, width, base)
            }
            BlockKind::List {
                start,
                tight,
                items,
            } => self.list(*start, *tight, items, width, base, depth),
            BlockKind::Quote { alert, blocks } => self.quote(*alert, blocks, width, base, depth),
            BlockKind::Table {
                alignments,
                header,
                rows,
            } => self.table(alignments, header, rows, width, base),
            BlockKind::FootnoteDefinition { label, blocks } => {
                let marker = format!("[{}] ", clean(label));
                let marker_width = str_width(&marker);
                if width <= marker_width {
                    // No room for the label: keep the content.
                    return self.blocks(blocks, width, base, depth, BLOCK_GAP);
                }
                let rows = self.blocks(blocks, width - marker_width, base, depth, BLOCK_GAP);
                hang(
                    rows,
                    Row::from(Run::new(marker, patch(base, theme.footnote))),
                    marker_width,
                )
            }
            BlockKind::Html(html) => {
                let style = patch(base, theme.html);
                html.trim_end_matches('\n')
                    .split('\n')
                    .flat_map(|line| hard_wrap(&clean(line), width, style, None))
                    .collect()
            }
            BlockKind::Rule => vec![Row::from(Run::new(
                theme.symbols.rule.to_string().repeat(width),
                patch(base, theme.rule),
            ))],
        }
    }

    // ----- inline content ------------------------------------------------

    fn runs(&self, inlines: &[Inline], style: ContentStyle) -> Vec<Run> {
        let mut out = Vec::new();
        self.inline_into(inlines, style, &mut out);
        while let Some(last) = out.last_mut() {
            let trimmed = last.text.trim_end_matches('\n').len();
            last.text.truncate(trimmed);
            if !last.text.is_empty() {
                break;
            }
            out.pop();
        }
        out
    }

    fn inline_into(&self, inlines: &[Inline], style: ContentStyle, out: &mut Vec<Run>) {
        let theme = self.theme;
        for inline in inlines {
            match inline {
                Inline::Text(text) => push(out, text, style),
                Inline::Code(code) => push(out, code, patch(style, theme.code)),
                Inline::Emphasis(content) => {
                    self.inline_into(content, patch(style, theme.emphasis), out)
                }
                Inline::Strong(content) => {
                    self.inline_into(content, patch(style, theme.strong), out)
                }
                Inline::Strikethrough(content) => {
                    self.inline_into(content, patch(style, theme.strikethrough), out)
                }
                Inline::Link { url, content } => {
                    self.inline_into(content, patch(style, theme.link), out);
                    let text = Inline::plain_text(content);
                    let bare = url.strip_prefix("mailto:").unwrap_or(url);
                    if theme.symbols.link_urls && !url.is_empty() && text != *url && text != bare {
                        push(out, &format!(" ({url})"), patch(style, theme.link_url));
                    }
                }
                Inline::Image { url, alt } => {
                    let alt = Inline::plain_text(alt);
                    let label = if alt.is_empty() { url } else { &alt };
                    push(out, &format!("[image: {label}]"), patch(style, theme.image));
                }
                Inline::Math { display, tex } => {
                    let math = patch(style, theme.math);
                    if *display {
                        if out.last().is_some_and(|run| !run.text.ends_with('\n')) {
                            push(out, "\n", style);
                        }
                        push(out, tex.trim(), math);
                        push(out, "\n", style);
                    } else {
                        push(out, tex, math);
                    }
                }
                Inline::FootnoteRef(label) => {
                    push(out, &format!("[{label}]"), patch(style, theme.footnote))
                }
                // Inline HTML stays on one row; a newline would add one.
                Inline::Html(html) => push(out, &html.replace('\n', " "), patch(style, theme.html)),
                Inline::SoftBreak => push(out, " ", style),
                Inline::HardBreak => push(out, "\n", style),
            }
        }
    }

    // ----- containers ----------------------------------------------------

    fn code_block(
        &self,
        lang: Option<&str>,
        code: &str,
        width: usize,
        base: ContentStyle,
    ) -> Vec<Row> {
        let style = patch(base, self.theme.code_block);
        let symbols = &self.theme.symbols;
        let gutter = if width > symbols.code_indent + 2 {
            symbols.code_indent
        } else {
            0
        };
        let inner = width - gutter;
        let pad = style.background_color.is_some().then_some(inner);
        let label = patch(base, self.theme.code_lang);
        let lang = lang.filter(|lang| !lang.is_empty());
        let mut rows = Vec::new();
        if symbols.code_fence {
            rows.extend(hard_wrap(
                &format!("```{}", clean(lang.unwrap_or(""))),
                width,
                label,
                None,
            ));
        } else if let Some(lang) = lang {
            rows.extend(hard_wrap(&clean(lang), width, label, None));
        }
        let highlighted = lang
            .zip(self.theme.syntax_theme.as_ref())
            .and_then(|(lang, theme)| highlight_code(lang, code, theme));
        let lines: Vec<Vec<Run>> = match highlighted {
            Some(lines) => lines
                .into_iter()
                .map(|runs| {
                    runs.into_iter()
                        .map(|run| Run::new(clean(&run.text), patch(style, run.style)))
                        .collect()
                })
                .collect(),
            None => code
                .trim_end_matches('\n')
                .split('\n')
                .map(|line| vec![Run::new(clean(line), style)])
                .collect(),
        };
        for line in &lines {
            for content in hard_wrap_runs(line, inner, pad.map(|pad| (pad, style))) {
                let mut row = Row::plain(" ".repeat(gutter));
                row.extend(content);
                rows.push(row);
            }
        }
        if symbols.code_fence {
            rows.extend(hard_wrap("```", width, label, None));
        }
        rows
    }

    fn list(
        &self,
        start: Option<u64>,
        tight: bool,
        items: &[ListItem],
        width: usize,
        base: ContentStyle,
        depth: usize,
    ) -> Vec<Row> {
        let theme = self.theme;
        let ordered = |index: usize| {
            start.map(|n| format!("{}{}", n + index as u64, theme.symbols.ordered_suffix))
        };
        let number_width = (0..items.len())
            .filter_map(ordered)
            .map(|n| str_width(&n))
            .max()
            .unwrap_or(0);
        let markers: Vec<(String, ContentStyle)> = items
            .iter()
            .enumerate()
            .map(|(index, item)| match item.task {
                Some(true) => (
                    format!("{}{}", theme.symbols.task_done, theme.symbols.marker_gap),
                    patch(base, theme.task_done),
                ),
                Some(false) => (
                    format!("{}{}", theme.symbols.task_todo, theme.symbols.marker_gap),
                    patch(base, theme.task_todo),
                ),
                None => match ordered(index) {
                    Some(number) => (
                        format!("{number:>number_width$}{}", theme.symbols.marker_gap),
                        patch(base, theme.list_marker),
                    ),
                    None => (
                        format!(
                            "{}{}",
                            theme.symbols.bullet(depth),
                            theme.symbols.marker_gap
                        ),
                        patch(base, theme.list_marker),
                    ),
                },
            })
            .collect();
        let gap = if tight { 0 } else { BLOCK_GAP };
        let mut out = Vec::new();
        for (index, (item, (marker, style))) in items.iter().zip(markers).enumerate() {
            if index > 0 {
                out.extend(std::iter::repeat_with(Row::new).take(gap));
            }
            // Each item hangs under its own marker; ordered numbers are
            // already right-aligned to a shared width.
            let marker_width = str_width(&marker);
            if width <= marker_width {
                // No room for the marker: keep the content.
                out.extend(self.blocks(&item.blocks, width, base, depth + 1, gap));
                continue;
            }
            let child_width = width - marker_width;
            let mut rows = self.blocks(&item.blocks, child_width, base, depth + 1, gap);
            if rows.is_empty() {
                rows.push(Row::new());
            }
            out.extend(hang(rows, Row::from(Run::new(marker, style)), marker_width));
        }
        out
    }

    fn quote(
        &self,
        alert: Option<AlertKind>,
        blocks: &[Block],
        width: usize,
        base: ContentStyle,
        depth: usize,
    ) -> Vec<Row> {
        let theme = self.theme;
        let (bar, inner_base) = match alert {
            Some(kind) => (patch(base, theme.alert(kind)), base),
            None => (patch(base, theme.quote_marker), patch(base, theme.quote)),
        };
        let bar_text = theme.symbols.quote_bar.as_str();
        let bar_width = str_width(bar_text) + 1;
        let mut rows = Vec::new();
        if let Some(kind) = alert {
            rows.extend(hard_wrap(
                kind.label(),
                width.saturating_sub(bar_width).max(1),
                bar.bold(),
                None,
            ));
        }
        if width <= bar_width {
            // No room for the bar: keep the content.
            rows.extend(self.blocks(blocks, width, inner_base, depth, BLOCK_GAP));
            return rows;
        }
        let child_width = width - bar_width;
        rows.extend(self.blocks(blocks, child_width, inner_base, depth, BLOCK_GAP));
        rows.into_iter()
            .map(|content| {
                let mut row = Row::new();
                if content.is_empty() {
                    row.push(bar_text, bar);
                } else {
                    row.push(format!("{bar_text} "), bar);
                    row.extend(content);
                }
                row
            })
            .collect()
    }

    // ----- tables --------------------------------------------------------

    fn table(
        &self,
        alignments: &[Alignment],
        header: &[Cell],
        body: &[Vec<Cell>],
        width: usize,
        base: ContentStyle,
    ) -> Vec<Row> {
        let columns = header
            .len()
            .max(body.iter().map(Vec::len).max().unwrap_or(0));
        if columns == 0 {
            return Vec::new();
        }
        let header_style = patch(base, self.theme.table_header);
        let cells = |row: &[Cell], style| -> Vec<Vec<Run>> {
            (0..columns)
                .map(|i| row.get(i).map(|c| self.runs(c, style)).unwrap_or_default())
                .collect()
        };
        let head = cells(header, header_style);
        let body: Vec<_> = body.iter().map(|row| cells(row, base)).collect();

        let overhead = 3 * columns + 1;
        if width < overhead + 3 * columns {
            return stacked(&head, &body, width, base);
        }
        let natural = |runs: &Vec<Run>| {
            wrap(runs, usize::MAX / 4)
                .iter()
                .map(Row::width)
                .max()
                .unwrap_or(0)
        };
        let mut widths: Vec<usize> = (0..columns)
            .map(|i| {
                std::iter::once(&head[i])
                    .chain(body.iter().map(|row| &row[i]))
                    .map(natural)
                    .max()
                    .unwrap_or(0)
                    .max(1)
            })
            .collect();
        // Shrink the widest column until the table fits. Columns first give
        // up width down to their longest word, so words only break once
        // every column is at that floor.
        let longest_word = |runs: &Vec<Run>| {
            let text: String = runs.iter().map(|run| run.text.as_str()).collect();
            text.split_whitespace().map(str_width).max().unwrap_or(0)
        };
        let floors: Vec<usize> = (0..columns)
            .map(|i| {
                std::iter::once(&head[i])
                    .chain(body.iter().map(|row| &row[i]))
                    .map(longest_word)
                    .max()
                    .unwrap_or(0)
                    .clamp(1, widths[i])
            })
            .collect();
        let available = width - overhead;
        // If even the longest words can't all fit, a boxed table would have to
        // break words mid-way (`Coun/t`); the stacked layout reads better.
        if floors.iter().sum::<usize>() > available {
            return stacked(&head, &body, width, base);
        }
        let mut total: usize = widths.iter().sum();
        while total > available {
            let widest = (0..columns)
                .filter(|&i| widths[i] > floors[i])
                .max_by_key(|&i| widths[i])
                .or_else(|| {
                    (0..columns)
                        .filter(|&i| widths[i] > 1)
                        .max_by_key(|&i| widths[i])
                })
                .expect("the stacked fallback guarantees 1 column each fits");
            widths[widest] -= 1;
            total -= 1;
        }

        let border = patch(base, self.theme.table_border);
        let chars = &self.theme.symbols.table;
        let rule = |left: char, mid: char, right: char| {
            let mut line = String::from(left);
            for (i, w) in widths.iter().enumerate() {
                if i > 0 {
                    line.push(mid);
                }
                line.push_str(&chars.horizontal.to_string().repeat(w + 2));
            }
            line.push(right);
            Row::from(Run::new(line, border))
        };
        let mut out = vec![rule(chars.top_left, chars.top_mid, chars.top_right)];
        out.extend(table_row(
            &head,
            &widths,
            alignments,
            chars.vertical,
            border,
            base,
        ));
        if !body.is_empty() {
            out.push(rule(chars.mid_left, chars.cross, chars.mid_right));
            for row in &body {
                out.extend(table_row(
                    row,
                    &widths,
                    alignments,
                    chars.vertical,
                    border,
                    base,
                ));
            }
        }
        out.push(rule(
            chars.bottom_left,
            chars.bottom_mid,
            chars.bottom_right,
        ));
        out
    }
}

fn table_row(
    cells: &[Vec<Run>],
    widths: &[usize],
    alignments: &[Alignment],
    vertical: char,
    border: ContentStyle,
    base: ContentStyle,
) -> Vec<Row> {
    let wrapped: Vec<Vec<Row>> = cells
        .iter()
        .zip(widths)
        .map(|(cell, &w)| wrap(cell, w))
        .collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
    (0..height)
        .map(|line| {
            let mut row = Row::new();
            row.push(vertical.to_string(), border);
            for (column, &w) in widths.iter().enumerate() {
                let content = wrapped[column]
                    .get(line)
                    .map(|row| row.fit(w))
                    .unwrap_or_default();
                let pad = w - content.width();
                let (left, right) = match alignments.get(column).copied().unwrap_or_default() {
                    Alignment::Right => (pad, 0),
                    Alignment::Center => (pad / 2, pad - pad / 2),
                    Alignment::Left | Alignment::None => (0, pad),
                };
                row.push(" ".repeat(left + 1), base);
                row.extend(content);
                row.push(" ".repeat(right + 1), base);
                row.push(vertical.to_string(), border);
            }
            row
        })
        .collect()
}

/// Narrow fallback: one `Header: value` line per cell, records separated by
/// blank rows.
fn stacked(
    head: &[Vec<Run>],
    body: &[Vec<Vec<Run>>],
    width: usize,
    base: ContentStyle,
) -> Vec<Row> {
    let mut out = Vec::new();
    if body.is_empty() {
        for cell in head {
            out.extend(wrap(cell, width));
        }
        return out;
    }
    for (index, record) in body.iter().enumerate() {
        if index > 0 {
            out.push(Row::new());
        }
        for (label, value) in head.iter().zip(record) {
            let mut runs = label.clone();
            if !label.is_empty() {
                let style = label.last().map_or(base, |run| run.style);
                runs.push(Run::new(": ", style));
            }
            runs.extend(value.iter().cloned());
            out.extend(wrap(&runs, width));
        }
    }
    out
}

// ----- helpers -----------------------------------------------------------

/// Prefix the first row with `first` and indent the rest by `indent`
/// columns. Blank rows stay empty so no trailing whitespace is emitted.
fn hang(rows: Vec<Row>, first: Row, indent: usize) -> Vec<Row> {
    rows.into_iter()
        .enumerate()
        .map(|(index, content)| {
            if index == 0 {
                let mut row = first.clone();
                if content.is_empty()
                    && let Some(last) = row.runs.last_mut()
                {
                    // Empty item: no trailing gap after the marker.
                    last.text.truncate(last.text.trim_end().len());
                }
                row.extend(content);
                row
            } else if content.is_empty() {
                content
            } else {
                let mut row = Row::plain(" ".repeat(indent));
                row.extend(content);
                row
            }
        })
        .collect()
}

/// Split one line at exactly `width` columns, preserving every character
/// (code must survive verbatim). `pad` right-pads each row to that width.
fn hard_wrap(text: &str, width: usize, style: ContentStyle, pad: Option<usize>) -> Vec<Row> {
    hard_wrap_runs(&[Run::new(text, style)], width, pad.map(|pad| (pad, style)))
}

/// [`hard_wrap`] over styled runs. `pad` is `(width, style)` of the fill.
fn hard_wrap_runs(runs: &[Run], width: usize, pad: Option<(usize, ContentStyle)>) -> Vec<Row> {
    let width = width.max(1);
    let mut rows = vec![Row::new()];
    let mut used = 0;
    for run in runs {
        for cluster in run.text.graphemes(true) {
            let w = cluster.width();
            if used > 0 && used + w > width {
                rows.push(Row::new());
                used = 0;
            }
            rows.last_mut().expect("non-empty").push(cluster, run.style);
            used += w;
        }
    }
    if let Some((pad, style)) = pad {
        for row in &mut rows {
            row.push(" ".repeat(pad.saturating_sub(row.width())), style);
        }
    }
    rows
}

#[cfg(feature = "highlight")]
fn highlight_code(lang: &str, code: &str, theme: &SyntaxTheme) -> Option<Vec<Vec<Run>>> {
    crate::highlight::highlight(lang, code, theme)
}

#[cfg(not(feature = "highlight"))]
fn highlight_code(_: &str, _: &str, _: &SyntaxTheme) -> Option<Vec<Vec<Run>>> {
    None
}

/// Strip terminal controls so document text can never inject escape
/// sequences; tabs become four spaces. `\n` is kept as a row break.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\t' => out.push_str("    "),
            '\n' => out.push('\n'),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// Append sanitized text, merging into the previous run when styles match.
fn push(out: &mut Vec<Run>, text: &str, style: ContentStyle) {
    let text = clean(text);
    if text.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(last) if last.style == style => last.text.push_str(&text),
        _ => out.push(Run::new(text, style)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(source: &str, width: usize) -> Vec<String> {
        render_str(source, &Theme::default(), width)
            .iter()
            .map(Row::text)
            .collect()
    }

    #[test]
    fn blocks_are_separated_by_one_blank_row() {
        assert_eq!(
            lines("# Title\n\nbody\n\n---", 5),
            ["Title", "", "body", "", "─────"]
        );
    }

    #[test]
    fn lists_hang_wrapped_lines_under_their_text() {
        assert_eq!(
            lines("- alpha beta gamma\n- two\n  - nested", 12),
            ["• alpha beta", "  gamma", "• two", "  ◦ nested"]
        );
        assert_eq!(lines("9. a\n10. b", 20), [" 9. a", "10. b"]);
        assert_eq!(
            lines("- [x] done\n- [ ] todo", 20),
            ["[x] done", "[ ] todo"]
        );
    }

    #[test]
    fn loose_lists_keep_blank_rows() {
        assert_eq!(lines("- a\n\n- b", 20), ["• a", "", "• b"]);
    }

    #[test]
    fn quotes_and_alerts_get_a_bar() {
        assert_eq!(lines("> one two three", 9), ["│ one two", "│ three"]);
        assert_eq!(lines("> [!TIP]\n> yes", 20), ["│ Tip", "│ yes"]);
    }

    #[test]
    fn code_blocks_are_verbatim_and_hard_wrapped() {
        assert_eq!(
            lines("```sh\nls  -la\nabcdefghij\n```", 8),
            ["sh", "  ls  -l", "  a", "  abcdef", "  ghij"]
        );
    }

    #[cfg(feature = "highlight")]
    #[test]
    fn highlighted_code_keeps_text_and_adds_colour() {
        let source = "```rust\nfn main() { let s = \"a long string literal\"; }\n```";
        let theme = Theme::default();
        let rows = render_str(source, &theme, 12);
        let plain = render_str(
            source,
            &Theme {
                syntax_theme: None,
                ..Theme::default()
            },
            12,
        );
        let text = |rows: &[Row]| rows.iter().map(Row::text).collect::<Vec<_>>();
        assert_eq!(text(&rows), text(&plain));
        assert!(rows.iter().all(|row| row.width() <= 12));
        let coloured = rows
            .iter()
            .flat_map(|row| &row.runs)
            .filter(|run| run.style.foreground_color.is_some())
            .count();
        assert!(coloured > 2);
        assert!(
            plain
                .iter()
                .flat_map(|r| &r.runs)
                .all(|r| r.style.foreground_color.is_none())
        );
    }

    #[cfg(feature = "custom-themes")]
    #[test]
    fn custom_and_named_syntax_themes_drive_colours() {
        const TMTHEME: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>name</key><string>Mine</string>
<key>settings</key><array>
<dict><key>settings</key><dict><key>foreground</key><string>#010203</string></dict></dict>
<dict><key>scope</key><string>storage</string><key>settings</key><dict><key>foreground</key><string>#FF0000</string></dict></dict>
</array></dict></plist>"#;
        let source = "```rust\nfn x() {}\n```";
        let colours = |theme: SyntaxTheme| {
            let theme = Theme {
                syntax_theme: Some(theme),
                ..Theme::default()
            };
            render_str(source, &theme, 40)[1]
                .runs
                .iter()
                .filter_map(|run| run.style.foreground_color)
                .collect::<Vec<_>>()
        };
        let custom = colours(SyntaxTheme::from_tmtheme(TMTHEME).unwrap());
        assert_eq!(
            custom[0],
            crossterm::style::Color::Rgb { r: 255, g: 0, b: 0 }
        );
        assert!(custom.contains(&crossterm::style::Color::Rgb { r: 1, g: 2, b: 3 }));
        assert_ne!(
            colours(SyntaxTheme::named("Solarized (dark)")),
            colours(SyntaxTheme::default())
        );
        assert!(colours(SyntaxTheme::named("missing")).is_empty());
    }

    #[cfg(feature = "custom-themes")]
    #[test]
    fn syntax_theme_validation_and_errors() {
        assert!(SyntaxTheme::default().is_available());
        assert!(!SyntaxTheme::named("nope").is_available());
        assert!(SyntaxTheme::builtin_names().contains(&"InspiredGitHub"));
        assert!(SyntaxTheme::from_tmtheme("not a plist").is_err());
        assert!(SyntaxTheme::from_tmtheme_file("/no/such/file").is_err());
        assert_eq!(SyntaxTheme::from("x"), SyntaxTheme::named("x"));
    }

    #[test]
    fn unknown_language_falls_back_to_plain() {
        let rows = render_str("```nope\nx = 1\n```", &Theme::default(), 20);
        assert_eq!(rows[1].text(), "  x = 1");
        assert!(
            rows[1]
                .runs
                .iter()
                .all(|r| r.style.foreground_color.is_none())
        );
    }

    #[test]
    fn links_show_their_url_once() {
        assert_eq!(lines("[docs](http://x.y)", 40), ["docs (http://x.y)"]);
        assert_eq!(lines("<http://x.y>", 40), ["http://x.y"]);
    }

    #[test]
    fn tables_fit_the_width() {
        let source = "| name | description |\n|---|--:|\n| a | the quick brown fox jumps |\n";
        assert_eq!(
            lines(source, 80),
            [
                "┌──────┬───────────────────────────┐",
                "│ name │               description │",
                "├──────┼───────────────────────────┤",
                "│ a    │ the quick brown fox jumps │",
                "└──────┴───────────────────────────┘",
            ]
        );
        let narrow = lines(source, 24);
        assert!(
            narrow.iter().all(|line| str_width(line) <= 24),
            "{narrow:#?}"
        );
        assert!(
            narrow
                .iter()
                .all(|line| line.ends_with(['┐', '│', '┤', '┘']))
        );
        assert_eq!(
            lines(source, 10),
            [
                "name: a",
                "descriptio",
                "n: the",
                "quick",
                "brown fox",
                "jumps"
            ]
        );
    }

    #[test]
    fn narrow_tables_stack_instead_of_breaking_words() {
        let source = "| Name | Count |\n|---|---|\n| a | 1 |\n";
        let rows = lines(source, 14);
        assert_eq!(rows, ["Name: a", "Count: 1"]);
        assert!(lines(source, 17)[0].starts_with('┌'));
    }

    #[test]
    fn classic_symbols_match_markdown_source() {
        let theme = Theme {
            symbols: crate::Symbols::classic(),
            ..Theme::default()
        };
        let text = |source: &str| {
            render_str(source, &theme, 30)
                .iter()
                .map(Row::text)
                .collect::<Vec<_>>()
        };
        assert_eq!(text("## Hi\n\n- a\n  - b"), ["## Hi", "", "- a", "  - b"]);
        assert_eq!(text("```rs\nlet x;\n```"), ["```rs", "let x;", "```"]);
    }

    #[test]
    fn symbols_are_configurable() {
        let mut theme = Theme::default();
        theme.symbols.ordered_suffix = ")".into();
        theme.symbols.quote_bar = ">".into();
        theme.symbols.rule = '=';
        theme.symbols.table = crate::TableChars::ascii();
        theme.symbols.link_urls = false;
        let text = |source: &str| {
            render_str(source, &theme, 12)
                .iter()
                .map(Row::text)
                .collect::<Vec<_>>()
        };
        assert_eq!(text("1. a"), ["1) a"]);
        assert_eq!(text("> q"), ["> q"]);
        assert_eq!(text("---"), ["=".repeat(12)]);
        assert_eq!(text("[x](http://y)"), ["x"]);
        assert_eq!(text("|a|\n|-|\n|b|")[0], "+---+");
    }

    #[test]
    fn with_accent_recolours_but_keeps_attributes() {
        use crossterm::style::Attribute;
        let theme = Theme::default().with_accent(crossterm::style::Color::Green);
        assert_eq!(
            theme.heading(2).foreground_color,
            Some(crossterm::style::Color::Green)
        );
        assert!(theme.heading(2).attributes.has(Attribute::Bold));
        assert!(theme.link.attributes.has(Attribute::Underlined));
    }

    #[test]
    fn controls_are_stripped() {
        assert_eq!(lines("a\u{1b}[31mb\u{7}", 20), ["a[31mb"]);
    }

    #[test]
    fn styles_reach_rows() {
        let rows = render_str("**b** `c`", &Theme::default(), 20);
        let theme = Theme::default();
        assert_eq!(rows[0].runs[0].style, patch(theme.text, theme.strong));
        assert_eq!(rows[0].runs[2].style, patch(theme.text, theme.code));
    }

    #[test]
    fn tiny_widths_keep_all_text() {
        let strip = |rows: Vec<String>| {
            rows.concat()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        };
        for source in [
            "> text",
            "- abc",
            "1. abc",
            "> [!NOTE]\n> hi",
            "[^1]: note\n\n[^1]",
        ] {
            for width in 1..=4 {
                let got = strip(lines(source, width));
                for word in ["text", "abc", "hi"] {
                    if source.contains(word) {
                        assert!(got.contains(word), "{source:?} @ {width}: {got:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn empty_item_has_no_trailing_space_and_html_stays_on_one_row() {
        assert_eq!(lines("- \n- b", 20), ["•", "• b"]);
        assert_eq!(lines("a <span\nx> b", 40).len(), 1);
    }

    #[test]
    fn empty_highlighted_fence_keeps_one_row() {
        let theme = Theme {
            syntax_theme: Some(SyntaxTheme::default()),
            ..Theme::default()
        };
        let plain = render_str("```rust\n```", &Theme::default(), 20).len();
        assert_eq!(render_str("```rust\n```", &theme, 20).len(), plain);
    }

    #[test]
    fn every_row_fits_at_every_width() {
        let source = include_str!("../examples/sample.md");
        let theme = Theme::default();
        let document = crate::parse(source);
        for width in 1..=100 {
            for row in render(&document, &theme, width) {
                assert!(row.width() <= width, "width {width}: {:?}", row.text());
            }
        }
    }
}
