//! Styles for every semantic role the renderer emits.

use crossterm::style::{Color, ContentStyle, Stylize};

use crate::doc::AlertKind;
use crate::symbols::Symbols;
use crate::syntax_theme::SyntaxTheme;

/// Overlay `over` onto `base`: colours in `over` win, attributes accumulate.
pub fn patch(base: ContentStyle, over: ContentStyle) -> ContentStyle {
    let mut attributes = base.attributes;
    attributes.extend(over.attributes);
    ContentStyle {
        foreground_color: over.foreground_color.or(base.foreground_color),
        background_color: over.background_color.or(base.background_color),
        underline_color: over.underline_color.or(base.underline_color),
        attributes,
    }
}

/// Every style the renderer uses. Plain data: construct one, tweak fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Theme {
    /// Glyphs and structural choices (bullets, fences, table borders…).
    pub symbols: Symbols,
    /// Base style for all body text.
    pub text: ContentStyle,
    /// Headings, indexed by level − 1 (levels above 6 use the last).
    pub heading: [ContentStyle; 6],
    /// Emphasised (italic) text.
    pub emphasis: ContentStyle,
    /// Strong (bold) text.
    pub strong: ContentStyle,
    /// Struck-through text.
    pub strikethrough: ContentStyle,
    /// Inline `code`.
    pub code: ContentStyle,
    /// Fenced/indented code block body. A background colour here is padded
    /// across the block's full width.
    pub code_block: ContentStyle,
    /// Colour scheme for fenced code, or `None` to disable highlighting.
    /// An unavailable theme also falls back to plain code.
    /// Without the `highlight` feature this is ignored.
    pub syntax_theme: Option<SyntaxTheme>,
    /// Language label above a fenced code block.
    pub code_lang: ContentStyle,
    /// Link text.
    pub link: ContentStyle,
    /// The ` (url)` suffix after link text that differs from its URL.
    pub link_url: ContentStyle,
    /// Image alt text.
    pub image: ContentStyle,
    /// Overlaid on content inside a block quote.
    pub quote: ContentStyle,
    /// The quote bar `│`.
    pub quote_marker: ContentStyle,
    /// List bullets and ordered-list numbers.
    pub list_marker: ContentStyle,
    /// Marker of a checked task item.
    pub task_done: ContentStyle,
    /// Marker of an unchecked task item.
    pub task_todo: ContentStyle,
    /// Thematic break rows.
    pub rule: ContentStyle,
    /// Table border glyphs.
    pub table_border: ContentStyle,
    /// Table header cell text.
    pub table_header: ContentStyle,
    /// Raw HTML, shown verbatim.
    pub html: ContentStyle,
    /// Math spans (TeX source).
    pub math: ContentStyle,
    /// Footnote references and labels.
    pub footnote: ContentStyle,
    /// `[!NOTE]` alert accent.
    pub alert_note: ContentStyle,
    /// `[!TIP]` alert accent.
    pub alert_tip: ContentStyle,
    /// `[!IMPORTANT]` alert accent.
    pub alert_important: ContentStyle,
    /// `[!WARNING]` alert accent.
    pub alert_warning: ContentStyle,
    /// `[!CAUTION]` alert accent.
    pub alert_caution: ContentStyle,
}

impl Theme {
    /// Style for a heading level; levels are clamped to 1..=6.
    pub fn heading(&self, level: u8) -> ContentStyle {
        self.heading[usize::from(level.clamp(1, 6)) - 1]
    }

    /// Accent style for the given alert kind.
    pub fn alert(&self, kind: AlertKind) -> ContentStyle {
        match kind {
            AlertKind::Note => self.alert_note,
            AlertKind::Tip => self.alert_tip,
            AlertKind::Important => self.alert_important,
            AlertKind::Warning => self.alert_warning,
            AlertKind::Caution => self.alert_caution,
        }
    }
}

impl Theme {
    /// Use one accent colour for headings, links, list markers and quote
    /// bars, keeping their existing bold/underline attributes.
    pub fn with_accent(mut self, accent: Color) -> Self {
        for heading in &mut self.heading {
            heading.foreground_color = Some(accent);
        }
        for style in [
            &mut self.link,
            &mut self.list_marker,
            &mut self.quote_marker,
        ] {
            style.foreground_color = Some(accent);
        }
        self
    }
}

impl Default for Theme {
    fn default() -> Self {
        let s = ContentStyle::new;
        Self {
            symbols: Symbols::default(),
            text: s(),
            heading: [
                s().with(Color::Cyan).bold().underlined(),
                s().with(Color::Cyan).bold(),
                s().bold(),
                s().bold(),
                s().bold(),
                s().bold(),
            ],
            emphasis: s().italic(),
            strong: s().bold(),
            strikethrough: s().crossed_out(),
            code: s().with(Color::Yellow),
            code_block: s(),
            syntax_theme: Some(SyntaxTheme::default()),
            code_lang: s().dim(),
            link: s().with(Color::Blue).underlined(),
            link_url: s().dim(),
            image: s().dim().italic(),
            quote: s().dim(),
            quote_marker: s().dim(),
            list_marker: s().with(Color::Blue),
            task_done: s().with(Color::Green),
            task_todo: s().dim(),
            rule: s().dim(),
            table_border: s().dim(),
            table_header: s().bold(),
            html: s().dim(),
            math: s().italic(),
            footnote: s().dim(),
            alert_note: s().with(Color::Blue),
            alert_tip: s().with(Color::Green),
            alert_important: s().with(Color::Magenta),
            alert_warning: s().with(Color::Yellow),
            alert_caution: s().with(Color::Red),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::style::Attribute;

    #[test]
    fn patch_overrides_colours_and_merges_attributes() {
        let base = ContentStyle::new().with(Color::Red).bold();
        let over = ContentStyle::new().with(Color::Blue).italic();
        let merged = patch(base, over);
        assert_eq!(merged.foreground_color, Some(Color::Blue));
        assert!(merged.attributes.has(Attribute::Bold));
        assert!(merged.attributes.has(Attribute::Italic));
        assert_eq!(
            patch(over, ContentStyle::new()).foreground_color,
            Some(Color::Blue)
        );
    }
}
