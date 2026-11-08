//! Styled terminal rows: the output primitive of the renderer.
//!
//! A [`Row`] is exactly one physical terminal row made of styled [`Run`]s.
//! Styles are crossterm's own [`ContentStyle`], so there is no colour or
//! attribute mapping layer between the renderer and the terminal.

use std::fmt::Write as _;

use crossterm::Command;
use crossterm::style::{
    Attribute, ContentStyle, SetAttribute, SetBackgroundColor, SetForegroundColor,
    SetUnderlineColor,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// A run of text sharing one style.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Run {
    /// The text of the run.
    pub text: String,
    /// Style applied to the whole run.
    pub style: ContentStyle,
}

impl Run {
    /// Create a run with an explicit style.
    pub fn new(text: impl Into<String>, style: ContentStyle) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    /// Create an unstyled run.
    pub fn plain(text: impl Into<String>) -> Self {
        Self::new(text, ContentStyle::default())
    }

    /// Display width in terminal columns.
    pub fn width(&self) -> usize {
        str_width(&self.text)
    }
}

/// One physical terminal row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Row {
    /// Styled runs, left to right.
    pub runs: Vec<Run>,
}

impl Row {
    /// Create an empty row.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a row holding one unstyled run.
    pub fn plain(text: impl Into<String>) -> Self {
        let mut row = Self::new();
        row.push(text.into(), ContentStyle::default());
        row
    }

    /// Append text, merging into the previous run when the style matches.
    pub fn push(&mut self, text: impl AsRef<str>, style: ContentStyle) {
        let text = text.as_ref();
        if text.is_empty() {
            return;
        }
        match self.runs.last_mut() {
            Some(last) if last.style == style => last.text.push_str(text),
            _ => self.runs.push(Run::new(text, style)),
        }
    }

    /// Append a run, merging with the previous one when styles match.
    pub fn push_run(&mut self, run: Run) {
        self.push(&run.text, run.style);
    }

    /// Append all runs of `other` to this row.
    pub fn extend(&mut self, other: Row) {
        for run in other.runs {
            self.push_run(run);
        }
    }

    /// Whether the row contains no text.
    pub fn is_empty(&self) -> bool {
        self.runs.iter().all(|run| run.text.is_empty())
    }

    /// Display width in terminal columns.
    pub fn width(&self) -> usize {
        self.runs.iter().map(Run::width).sum()
    }

    /// Concatenated text without styling.
    pub fn text(&self) -> String {
        self.runs.iter().map(|run| run.text.as_str()).collect()
    }

    /// Truncate to at most `width` columns without splitting a grapheme
    /// cluster; a wide glyph that would overflow is dropped.
    pub fn fit(&self, width: usize) -> Row {
        let mut out = Row::new();
        let mut used = 0usize;
        'runs: for run in &self.runs {
            let mut kept = String::new();
            for cluster in run.text.graphemes(true) {
                let w = cluster.width();
                if used + w > width {
                    out.push(&kept, run.style);
                    break 'runs;
                }
                kept.push_str(cluster);
                used += w;
            }
            out.push(&kept, run.style);
        }
        out
    }

    /// Serialize to ANSI. Attributes persist in the terminal until reset, so a
    /// full reset is emitted whenever the style changes between runs, and once
    /// at the end if anything was styled. Text is written verbatim: callers
    /// must sanitize untrusted content before it reaches a [`Run`].
    pub fn write_ansi(&self, out: &mut String) {
        let mut previous: Option<ContentStyle> = None;
        let mut styled = false;
        for run in &self.runs {
            if run.text.is_empty() {
                continue;
            }
            if previous != Some(run.style) {
                if styled {
                    out.push_str("\u{1b}[0m");
                }
                let prefix = style_prefix(&run.style);
                styled = !prefix.is_empty();
                out.push_str(&prefix);
            }
            out.push_str(&run.text);
            previous = Some(run.style);
        }
        if styled {
            out.push_str("\u{1b}[0m");
        }
    }

    /// Serialize to a new ANSI string; see [`Row::write_ansi`].
    pub fn to_ansi(&self) -> String {
        let mut out = String::new();
        self.write_ansi(&mut out);
        out
    }
}

impl From<Run> for Row {
    fn from(run: Run) -> Self {
        let mut row = Row::new();
        row.push_run(run);
        row
    }
}

impl From<Vec<Run>> for Row {
    fn from(runs: Vec<Run>) -> Self {
        let mut row = Row::new();
        runs.into_iter().for_each(|run| row.push_run(run));
        row
    }
}

/// Display width of a string in terminal columns.
pub fn str_width(text: &str) -> usize {
    text.graphemes(true).map(UnicodeWidthStr::width).sum()
}

const ATTRIBUTES: &[Attribute] = &[
    Attribute::Bold,
    Attribute::Dim,
    Attribute::Italic,
    Attribute::Underlined,
    Attribute::SlowBlink,
    Attribute::RapidBlink,
    Attribute::Reverse,
    Attribute::Hidden,
    Attribute::CrossedOut,
];

fn style_prefix(style: &ContentStyle) -> String {
    let mut out = String::new();
    if let Some(color) = style.foreground_color {
        let _ = write!(out, "{}", SetForegroundColor(color));
    }
    if let Some(color) = style.background_color {
        let _ = write!(out, "{}", SetBackgroundColor(color));
    }
    if let Some(color) = style.underline_color {
        let _ = SetUnderlineColor(color).write_ansi(&mut out);
    }
    for attribute in ATTRIBUTES {
        if style.attributes.has(*attribute) {
            let _ = write!(out, "{}", SetAttribute(*attribute));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::style::{Color, Stylize};

    #[test]
    fn push_merges_equal_styles() {
        let mut row = Row::new();
        row.push("a", ContentStyle::default());
        row.push("b", ContentStyle::default());
        row.push("", ContentStyle::new().bold());
        assert_eq!(row.runs.len(), 1);
        assert_eq!(row.text(), "ab");
    }

    #[test]
    fn fit_never_splits_wide_glyphs() {
        let row = Row::plain("a你b");
        assert_eq!(row.fit(2).text(), "a");
        assert_eq!(row.fit(3).text(), "a你");
        assert_eq!(row.fit(0).text(), "");
    }

    #[test]
    fn ansi_resets_between_differing_styles() {
        let mut row = Row::new();
        row.push("x", ContentStyle::new().bold());
        row.push("y", ContentStyle::new().with(Color::Red));
        let ansi = row.to_ansi();
        assert!(ansi.starts_with("\u{1b}[1mx\u{1b}[0m"));
        assert!(ansi.ends_with("y\u{1b}[0m"));
    }

    #[test]
    fn plain_rows_emit_no_escapes() {
        assert_eq!(Row::plain("hello").to_ansi(), "hello");
    }
}
