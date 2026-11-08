//! Writing styled rows to the terminal. Content is sanitized and fitted to the
//! width before serialization, so only generated styles ever reach the terminal.

use super::*;

/// Write one row: the shared left gutter plus the row's ANSI serialization.
/// Blank rows become gutter-width space runs, which render identically.
pub(super) fn write_row(buffer: &mut String, line: &Row, gutter: usize, width: usize) {
    if gutter > 0 {
        buffer.push_str(&" ".repeat(gutter.min(width)));
    }
    let content_width = width.saturating_sub(gutter);
    let line = render::fit_line_to_width(line, content_width);
    line.write_ansi(buffer);
}
