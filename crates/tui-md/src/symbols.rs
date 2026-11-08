//! Glyphs and structural choices, separate from colours so a host can match
//! its own look (`-` bullets, `##` headings, fenced code…) without touching
//! the layout code.

/// Characters used to draw table borders. Each must be one column wide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableChars {
    /// Horizontal line segment.
    pub horizontal: char,
    /// Vertical line segment.
    pub vertical: char,
    /// Top-left corner.
    pub top_left: char,
    /// Top edge junction above a column divider.
    pub top_mid: char,
    /// Top-right corner.
    pub top_right: char,
    /// Left edge junction of an inner horizontal rule.
    pub mid_left: char,
    /// Inner junction where rules and dividers cross.
    pub cross: char,
    /// Right edge junction of an inner horizontal rule.
    pub mid_right: char,
    /// Bottom-left corner.
    pub bottom_left: char,
    /// Bottom edge junction below a column divider.
    pub bottom_mid: char,
    /// Bottom-right corner.
    pub bottom_right: char,
}

impl TableChars {
    /// Box-drawing characters (`┌─┬┐`).
    pub fn unicode() -> Self {
        Self {
            horizontal: '─',
            vertical: '│',
            top_left: '┌',
            top_mid: '┬',
            top_right: '┐',
            mid_left: '├',
            cross: '┼',
            mid_right: '┤',
            bottom_left: '└',
            bottom_mid: '┴',
            bottom_right: '┘',
        }
    }

    /// Plain ASCII fallback (`+`, `-`, `|`).
    pub fn ascii() -> Self {
        Self {
            horizontal: '-',
            vertical: '|',
            top_left: '+',
            top_mid: '+',
            top_right: '+',
            mid_left: '+',
            cross: '+',
            mid_right: '+',
            bottom_left: '+',
            bottom_mid: '+',
            bottom_right: '+',
        }
    }
}

/// All glyphs and structural toggles used when laying out a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbols {
    /// Bullet per list nesting depth, cycling. Include any trailing space
    /// you want after the glyph via `marker_gap`.
    pub bullets: Vec<String>,
    /// Text between a list marker and its content.
    pub marker_gap: String,
    /// Appended to an ordered list number (`1.`, `1)`).
    pub ordered_suffix: String,
    /// Marker for a checked task-list item.
    pub task_done: String,
    /// Marker for an unchecked task-list item.
    pub task_todo: String,
    /// Block quote bar; content follows after one space.
    pub quote_bar: String,
    /// Horizontal rule character.
    pub rule: char,
    /// Table border characters.
    pub table: TableChars,
    /// Prefix headings with `#`×level and a space.
    pub heading_marker: bool,
    /// Draw ``` fence rows around code blocks instead of a language label.
    pub code_fence: bool,
    /// Columns code is indented by (dropped when the width is tiny).
    pub code_indent: usize,
    /// Append ` (url)` after link text that differs from its URL.
    pub link_urls: bool,
}

impl Default for Symbols {
    fn default() -> Self {
        Self {
            bullets: vec!["•".into(), "◦".into(), "▪".into()],
            marker_gap: " ".into(),
            ordered_suffix: ".".into(),
            task_done: "[x]".into(),
            task_todo: "[ ]".into(),
            quote_bar: "│".into(),
            rule: '─',
            table: TableChars::unicode(),
            heading_marker: false,
            code_fence: false,
            code_indent: 2,
            link_urls: true,
        }
    }
}

impl Symbols {
    /// The look of plain Markdown source: `-` bullets, `##` headings and
    /// fenced code with no indent (copies back out as valid Markdown).
    pub fn classic() -> Self {
        Self {
            bullets: vec!["-".into()],
            heading_marker: true,
            code_fence: true,
            code_indent: 0,
            ..Self::default()
        }
    }

    pub(crate) fn bullet(&self, depth: usize) -> &str {
        if self.bullets.is_empty() {
            "-"
        } else {
            &self.bullets[depth % self.bullets.len()]
        }
    }
}
