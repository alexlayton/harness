//! Syntax highlighting for fenced code blocks (syntect, pure-Rust regexes).
//!
//! Only foreground colour and font style are taken from the syntect theme;
//! backgrounds are left to the terminal (or `Theme::code_block`).

use std::sync::OnceLock;

use crossterm::style::{Attribute, Color, ContentStyle};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, Theme as SyntectTheme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use crate::row::Run;
use crate::syntax_theme::SyntaxTheme;

struct Assets {
    syntaxes: SyntaxSet,
    themes: ThemeSet,
}

fn assets() -> &'static Assets {
    static ASSETS: OnceLock<Assets> = OnceLock::new();
    ASSETS.get_or_init(|| Assets {
        syntaxes: SyntaxSet::load_defaults_newlines(),
        themes: ThemeSet::load_defaults(),
    })
}

/// Load the grammars and themes now. The first highlighted block otherwise
/// pays this (tens of ms); call from a background thread at startup.
pub fn warm_up() {
    let assets = assets();
    // Linking a grammar's contexts is lazy; touch the common ones too.
    for lang in ["rust", "python", "js", "sh"] {
        if let Some(syntax) = assets.syntaxes.find_syntax_by_token(lang)
            && let Some(theme) = assets.themes.themes.get(crate::DEFAULT_SYNTAX_THEME)
        {
            let mut highlighter = HighlightLines::new(syntax, theme);
            let _ = highlighter.highlight_line("x\n", &assets.syntaxes);
        }
    }
}

/// Blocks beyond these limits render plain: highlighting cost is linear but
/// the regex engine can be slow on pathological lines (minified code).
const MAX_LINES: usize = 2000;
const MAX_LINE_BYTES: usize = 2000;

/// Look up a bundled theme.
pub(crate) fn builtin_theme(name: &str) -> Option<&'static SyntectTheme> {
    assets().themes.themes.get(name)
}

/// Names of the bundled syntect themes (unsorted).
pub(crate) fn theme_names() -> Vec<&'static str> {
    assets().themes.themes.keys().map(String::as_str).collect()
}

/// Highlight `code`, returning one run list per source line (no trailing
/// newlines). `None` if the language or theme is unknown, so the caller can
/// fall back to plain rendering. Text is returned unsanitized.
///
/// Nothing is cached: a streaming caller re-highlights the growing block on
/// every render, which is bounded by `MAX_LINES` and fine in practice.
pub(crate) fn highlight(lang: &str, code: &str, theme: &SyntaxTheme) -> Option<Vec<Vec<Run>>> {
    if code.len() > MAX_LINES * 80
        || code.lines().take(MAX_LINES + 1).count() > MAX_LINES
        || code.lines().any(|line| line.len() > MAX_LINE_BYTES)
    {
        return None;
    }
    let assets = assets();
    let syntax = assets
        .syntaxes
        .find_syntax_by_token(lang)
        .or_else(|| assets.syntaxes.find_syntax_by_extension(lang))?;
    let theme = theme.resolve()?;
    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines = Vec::new();
    for line in LinesWithEndings::from(code.trim_end_matches('\n')) {
        let ranges = highlighter.highlight_line(line, &assets.syntaxes).ok()?;
        let mut runs: Vec<Run> = Vec::new();
        for (style, text) in ranges {
            let text = text.trim_end_matches(['\n', '\r']);
            if text.is_empty() {
                continue;
            }
            let style = convert(style);
            match runs.last_mut() {
                Some(last) if last.style == style => last.text.push_str(text),
                _ => runs.push(Run::new(text, style)),
            }
        }
        lines.push(runs);
    }
    if lines.is_empty() {
        // An empty block still occupies one row, like the plain path.
        lines.push(Vec::new());
    }
    Some(lines)
}

fn convert(style: syntect::highlighting::Style) -> ContentStyle {
    let fg = style.foreground;
    let mut out = ContentStyle::new();
    out.foreground_color = Some(Color::Rgb {
        r: fg.r,
        g: fg.g,
        b: fg.b,
    });
    if style.font_style.contains(FontStyle::BOLD) {
        out.attributes.set(Attribute::Bold);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        out.attributes.set(Attribute::Italic);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        out.attributes.set(Attribute::Underlined);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> SyntaxTheme {
        SyntaxTheme::default()
    }

    #[test]
    fn highlights_known_languages_line_by_line() {
        let lines = highlight("rust", "fn main() {}\nlet x = 1;\n", &theme()).unwrap();
        assert_eq!(lines.len(), 2);
        let text: String = lines[0].iter().map(|r| r.text.as_str()).collect();
        assert_eq!(text, "fn main() {}");
        assert!(lines[0].len() > 1, "expected multiple styled runs");
    }

    #[test]
    fn accepts_extensions_and_rejects_unknowns() {
        assert!(highlight("rs", "let x;", &theme()).is_some());
        assert!(highlight("not-a-language", "x", &theme()).is_none());
        assert!(highlight("rust", "x", &SyntaxTheme::named("no-such-theme")).is_none());
    }

    #[test]
    fn oversized_blocks_fall_back_to_plain() {
        assert!(highlight("rust", &"x\n".repeat(MAX_LINES + 1), &theme()).is_none());
        assert!(highlight("rust", &"x".repeat(MAX_LINE_BYTES + 1), &theme()).is_none());
        assert!(highlight("rust", &"x\n".repeat(MAX_LINES), &theme()).is_some());
    }

    #[test]
    fn warm_up_is_callable_repeatedly() {
        warm_up();
        warm_up();
    }

    #[test]
    fn bundled_themes_include_the_default() {
        assert!(theme_names().contains(&crate::syntax_theme::DEFAULT_SYNTAX_THEME));
    }
}
