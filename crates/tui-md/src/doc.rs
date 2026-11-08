//! The parsed document: a semantic block tree.
//!
//! The tree records *what* the content is (a heading, a task item, a link),
//! never *how* it looks. Styling and width-dependent layout happen later in
//! [`crate::render`], so one parse can be laid out at any width or theme.

use std::ops::Range;

/// A parsed Markdown document.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Document {
    /// Top-level blocks in source order.
    pub blocks: Vec<Block>,
    /// Start of the first block holding an unresolved reference, which a
    /// later definition could still turn into a link.
    pub(crate) unstable_from: Option<usize>,
}

impl Document {
    /// Byte length of the source prefix whose rendering is final.
    ///
    /// Every top-level block except the last is complete: appending more
    /// source can only change the last block (a paragraph may become a
    /// setext heading, a list or table may grow). A streaming caller can
    /// commit the rows of all blocks before this offset and re-render only
    /// the tail.
    ///
    /// Blocks containing an unresolved reference (`[text][ref]`, `[ref]`,
    /// `[^note]`) are also withheld, because a definition arriving later in
    /// the stream turns them into links or footnotes.
    pub fn stable_len(&self) -> usize {
        let last = self.blocks.last().map_or(0, |block| block.range.start);
        self.unstable_from.map_or(last, |from| from.min(last))
    }
}

/// A block plus the byte range of source it was parsed from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// What the block is.
    pub kind: BlockKind,
    /// Byte range of the source this block was parsed from.
    pub range: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// The kind of a block-level element.
pub enum BlockKind {
    /// A paragraph of inline content.
    Paragraph(Vec<Inline>),
    /// An ATX or setext heading.
    Heading {
        /// Heading level, 1 (largest) to 6.
        level: u8,
        /// Inline content of the heading.
        content: Vec<Inline>,
    },
    /// An indented or fenced code block.
    CodeBlock {
        /// First word of a fenced block's info string.
        lang: Option<String>,
        /// Raw code text.
        code: String,
    },
    /// An ordered or unordered list.
    List {
        /// `Some(n)` for an ordered list starting at `n`.
        start: Option<u64>,
        /// Tight lists have no blank rows between items.
        tight: bool,
        /// The list's items in order.
        items: Vec<ListItem>,
    },
    /// A block quote, optionally a GitHub alert.
    Quote {
        /// GitHub alert (`> [!NOTE]`), if any.
        alert: Option<AlertKind>,
        /// Blocks nested inside the quote.
        blocks: Vec<Block>,
    },
    /// A GFM table.
    Table {
        /// Per-column alignment.
        alignments: Vec<Alignment>,
        /// Header cells.
        header: Vec<Cell>,
        /// Body rows, each a list of cells.
        rows: Vec<Vec<Cell>>,
    },
    /// A footnote definition (`[^label]: ...`).
    FootnoteDefinition {
        /// Footnote label as written in the source.
        label: String,
        /// Blocks making up the footnote body.
        blocks: Vec<Block>,
    },
    /// A raw HTML block, kept verbatim.
    Html(String),
    /// A thematic break (`---`).
    Rule,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// One item of a list.
pub struct ListItem {
    /// `Some(checked)` for a task-list item.
    pub task: Option<bool>,
    /// Blocks making up the item's content.
    pub blocks: Vec<Block>,
}

/// A table cell: a run of inline content.
pub type Cell = Vec<Inline>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Horizontal alignment of a table column.
pub enum Alignment {
    /// No explicit alignment (rendered like left).
    #[default]
    None,
    /// Left-aligned.
    Left,
    /// Centered.
    Center,
    /// Right-aligned.
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// GitHub alert flavour introduced by `> [!KIND]`.
pub enum AlertKind {
    /// `[!NOTE]`: general information.
    Note,
    /// `[!TIP]`: helpful advice.
    Tip,
    /// `[!IMPORTANT]`: key information.
    Important,
    /// `[!WARNING]`: urgent, needs attention.
    Warning,
    /// `[!CAUTION]`: risk of negative outcomes.
    Caution,
}

impl AlertKind {
    /// Human-readable title shown on the alert's first row.
    pub fn label(self) -> &'static str {
        match self {
            Self::Note => "Note",
            Self::Tip => "Tip",
            Self::Important => "Important",
            Self::Warning => "Warning",
            Self::Caution => "Caution",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Inline-level content within a block.
pub enum Inline {
    /// Plain text.
    Text(String),
    /// Inline code span.
    Code(String),
    /// Emphasised (italic) content.
    Emphasis(Vec<Inline>),
    /// Strong (bold) content.
    Strong(Vec<Inline>),
    /// Struck-through content.
    Strikethrough(Vec<Inline>),
    /// A hyperlink.
    Link {
        /// Link destination.
        url: String,
        /// Visible link text.
        content: Vec<Inline>,
    },
    /// An image; terminals show its alt text.
    Image {
        /// Image source URL.
        url: String,
        /// Alt text content.
        alt: Vec<Inline>,
    },
    /// A math span.
    Math {
        /// `true` for display math (`$$..$$`), `false` for inline.
        display: bool,
        /// Raw TeX source.
        tex: String,
    },
    /// A footnote reference (`[^label]`) holding the label.
    FootnoteRef(String),
    /// Raw inline HTML, kept verbatim.
    Html(String),
    /// A soft line break within a paragraph.
    SoftBreak,
    /// A hard line break.
    HardBreak,
}

impl Inline {
    /// Plain text content, ignoring formatting.
    pub fn plain_text(inlines: &[Inline]) -> String {
        let mut out = String::new();
        for inline in inlines {
            match inline {
                Inline::Text(t) | Inline::Code(t) | Inline::Html(t) => out.push_str(t),
                Inline::Math { tex, .. } => out.push_str(tex),
                Inline::FootnoteRef(label) => out.push_str(label),
                Inline::Emphasis(c) | Inline::Strong(c) | Inline::Strikethrough(c) => {
                    out.push_str(&Inline::plain_text(c))
                }
                Inline::Link { content, .. } => out.push_str(&Inline::plain_text(content)),
                Inline::Image { alt, .. } => out.push_str(&Inline::plain_text(alt)),
                Inline::SoftBreak => out.push(' '),
                Inline::HardBreak => out.push('\n'),
            }
        }
        out
    }
}
