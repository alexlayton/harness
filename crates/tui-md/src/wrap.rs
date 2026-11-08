//! Styled greedy word wrapping measured in grapheme clusters.

use crossterm::style::ContentStyle;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::row::{Row, Run};

struct Cluster<'a> {
    text: &'a str,
    style: ContentStyle,
    width: usize,
}

/// Wrap styled runs into rows at most `width` columns wide.
///
/// `\n` forces a row break. Lines break at whitespace; a word is split only
/// when it is wider than a whole row (it then fills the current row first, so
/// long URLs/paths don't leave a ragged gap). Whitespace at a wrap point is
/// dropped; leading whitespace of a source line is kept.
pub fn wrap(runs: &[Run], width: usize) -> Vec<Row> {
    let width = width.max(1);
    let mut lines: Vec<Vec<Cluster<'_>>> = vec![Vec::new()];
    for run in runs {
        for (index, part) in run.text.split('\n').enumerate() {
            if index > 0 {
                lines.push(Vec::new());
            }
            let line = lines.last_mut().expect("non-empty");
            line.extend(part.graphemes(true).map(|text| Cluster {
                text,
                style: run.style,
                width: text.width(),
            }));
        }
    }
    let mut rows = Vec::new();
    for line in &lines {
        wrap_line(line, width, &mut rows);
    }
    rows
}

fn is_space(cluster: &Cluster<'_>) -> bool {
    cluster.text.chars().all(char::is_whitespace)
}

fn wrap_line(line: &[Cluster<'_>], width: usize, rows: &mut Vec<Row>) {
    let mut row = Row::new();
    let mut used = 0usize;
    let mut first_row = true;
    let mut pending: &[Cluster<'_>] = &[];
    let mut rest = line;
    while !rest.is_empty() {
        let space = is_space(&rest[0]);
        let len = rest
            .iter()
            .position(|c| is_space(c) != space)
            .unwrap_or(rest.len());
        let (token, tail) = rest.split_at(len);
        rest = tail;
        let token_width: usize = token.iter().map(|c| c.width).sum();
        if space {
            if used == 0 && first_row {
                // Keep the indent, but never wider than the row.
                for cluster in token {
                    if used + cluster.width > width {
                        break;
                    }
                    row.push(cluster.text, cluster.style);
                    used += cluster.width;
                }
            } else {
                pending = token;
            }
            continue;
        }
        let pending_width: usize = pending.iter().map(|c| c.width).sum();
        if used + pending_width + token_width <= width {
            push(&mut row, pending);
            push(&mut row, token);
            used += pending_width + token_width;
        } else if token_width <= width {
            rows.push(std::mem::take(&mut row));
            first_row = false;
            push(&mut row, token);
            used = token_width;
        } else {
            // Over-long word: fill the current row, then split it.
            if used > 0 && used + pending_width < width {
                push(&mut row, pending);
                used += pending_width;
            } else if used > 0 {
                rows.push(std::mem::take(&mut row));
                first_row = false;
                used = 0;
            }
            for cluster in token {
                if used > 0 && used + cluster.width > width {
                    rows.push(std::mem::take(&mut row));
                    first_row = false;
                    used = 0;
                }
                row.push(cluster.text, cluster.style);
                used += cluster.width;
            }
        }
        pending = &[];
    }
    rows.push(row);
}

fn push(row: &mut Row, clusters: &[Cluster<'_>]) {
    for cluster in clusters {
        row.push(cluster.text, cluster.style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(source: &str, width: usize) -> Vec<String> {
        wrap(&[Run::plain(source)], width)
            .iter()
            .map(Row::text)
            .collect()
    }

    #[test]
    fn wraps_at_whitespace() {
        assert_eq!(texts("aa bb cc", 5), ["aa bb", "cc"]);
        assert_eq!(texts("aa bb cc", 80), ["aa bb cc"]);
    }

    #[test]
    fn splits_only_overlong_words() {
        assert_eq!(texts("a abcdefgh", 4), ["a ab", "cdef", "gh"]);
        assert_eq!(texts("abcd", 4), ["abcd"]);
    }

    #[test]
    fn newlines_force_breaks_and_keep_blank_lines() {
        assert_eq!(texts("a\n\nb", 10), ["a", "", "b"]);
    }

    #[test]
    fn keeps_leading_indent_drops_wrap_space() {
        assert_eq!(texts("  a b", 3), ["  a", "b"]);
    }

    #[test]
    fn leading_indent_is_clamped_to_width() {
        assert_eq!(texts("    a", 3), ["   ", "a"]);
        for row in wrap(&[Run::plain("     abc")], 2) {
            assert!(row.width() <= 2);
        }
    }

    #[test]
    fn wide_glyphs_never_overflow() {
        for row in wrap(&[Run::plain("你好世界 hello")], 3) {
            assert!(row.width() <= 3);
        }
    }

    #[test]
    fn styles_survive_wrapping() {
        let bold = ContentStyle {
            attributes: crossterm::style::Attribute::Bold.into(),
            ..ContentStyle::default()
        };
        let rows = wrap(&[Run::plain("x "), Run::new("bold words", bold)], 6);
        assert_eq!(rows[0].runs[1].style, bold);
        assert_eq!(rows[1].runs[0].style, bold);
    }

    /// Pseudo-random fuzz: rows fit, and no non-whitespace text is lost or
    /// reordered.
    #[test]
    fn fuzz_rows_fit_and_preserve_text() {
        let alphabet: Vec<&str> =
            vec!["a", "b", " ", "  ", "\n", "你", "é", "👨‍👩‍👧", "-", "xyzxyzxyz"];
        let mut seed = 0x2545_f491_u64;
        for _ in 0..500 {
            let mut source = String::new();
            for _ in 0..(seed % 40) {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                source.push_str(alphabet[(seed >> 33) as usize % alphabet.len()]);
            }
            let width = 1 + (seed >> 20) as usize % 12;
            let rows = wrap(&[Run::plain(source.clone())], width);
            for row in &rows {
                assert!(row.width() <= width.max(2), "{source:?} @ {width}");
                // Only a lone 2-wide glyph may exceed a width-1 row.
                if row.width() > width {
                    assert_eq!(row.text().graphemes(true).count(), 1);
                }
            }
            let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
            let joined: String = rows.iter().map(Row::text).collect();
            assert_eq!(strip(&joined), strip(&source));
        }
    }
}
