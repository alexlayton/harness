//! Markdown rendering for crossterm TUIs.
//!
//! Two stages: [`parse`] turns source into a semantic [`Document`]; [`render`]
//! lays it out at a width with a [`Theme`], producing terminal [`Row`]s styled
//! with crossterm's own [`ContentStyle`](crossterm::style::ContentStyle).
//!
//! ```
//! let rows = tui_md::render_str("**hi** there", &tui_md::Theme::default(), 40);
//! assert_eq!(rows[0].text(), "hi there");
//! let _ansi = rows[0].to_ansi();
//! ```

#![warn(missing_docs)]

mod doc;
#[cfg(feature = "highlight")]
mod highlight;
mod layout;
mod parse;
mod row;
mod symbols;
mod syntax_theme;
mod theme;
mod wrap;

pub use doc::{AlertKind, Alignment, Block, BlockKind, Cell, Document, Inline, ListItem};
#[cfg(feature = "highlight")]
pub use highlight::warm_up;
pub use layout::{render, render_blocks, render_str};
pub use parse::parse;
pub use row::{Row, Run, str_width};
pub use symbols::{Symbols, TableChars};
pub use syntax_theme::{DEFAULT_SYNTAX_THEME, SyntaxTheme, SyntaxThemeError};
pub use theme::{Theme, patch};
pub use wrap::wrap;

/// No-op: built without the `highlight` feature.
#[cfg(not(feature = "highlight"))]
pub fn warm_up() {}
