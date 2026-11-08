//! pulldown-cmark events → [`Document`].

use std::iter::Peekable;
use std::ops::Range;

use pulldown_cmark::{
    Alignment as CmAlignment, BlockQuoteKind, BrokenLink, CodeBlockKind, Event, Options, Parser,
    Tag, TagEnd,
};

use crate::doc::{AlertKind, Alignment, Block, BlockKind, Cell, Document, Inline, ListItem};

/// Parse CommonMark plus the GFM extensions an agent is likely to emit:
/// tables, task lists, strikethrough, footnotes, alerts, and math.
pub fn parse(source: &str) -> Document {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_MATH);
    options.insert(Options::ENABLE_GFM);
    // Unresolved `[text][ref]` / `[ref]` / `[^note]` stay literal text, but a
    // definition arriving later would turn them into links, so remember
    // where the first one is (see `Document::stable_len`).
    // The parser does not report undefined footnote references, so treat
    // any `[^` as potentially unresolved (conservative, and rare).
    let first_unresolved = std::cell::Cell::new(source.find("[^"));
    let mut note = |link: BrokenLink<'_>| {
        let start = link.span.start;
        first_unresolved.set(Some(first_unresolved.get().map_or(start, |s| s.min(start))));
        None
    };
    let mut builder = Builder {
        events: Parser::new_with_broken_link_callback(source, options, Some(&mut note))
            .into_offset_iter()
            .peekable(),
        items: Vec::new(),
        containers: 0,
    };
    let blocks = builder.blocks();
    drop(builder);
    // The block holding the first unresolved reference can still change.
    let unstable_from = first_unresolved.get().and_then(|offset| {
        blocks
            .iter()
            .rfind(|block| block.range.start <= offset)
            .map(|block| block.range.start)
    });
    Document {
        blocks,
        unstable_from,
    }
}

struct Builder<'a, I: Iterator<Item = (Event<'a>, Range<usize>)>> {
    events: Peekable<I>,
    /// Open list items, innermost last.
    items: Vec<ItemState>,
    /// Nesting depth of non-item block containers (quotes, footnotes), so a
    /// paragraph only loosens the item it is a direct child of.
    containers: usize,
}

#[derive(Default)]
struct ItemState {
    task: Option<bool>,
    loose: bool,
    /// `Builder::containers` when the item opened.
    depth: usize,
}

fn is_inline_event(event: &Event<'_>) -> bool {
    match event {
        Event::Text(_)
        | Event::Code(_)
        | Event::InlineMath(_)
        | Event::DisplayMath(_)
        | Event::InlineHtml(_)
        | Event::FootnoteReference(_)
        | Event::SoftBreak
        | Event::HardBreak
        | Event::TaskListMarker(_) => true,
        Event::Start(tag) => matches!(
            tag,
            Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Link { .. }
                | Tag::Image { .. }
                | Tag::Superscript
                | Tag::Subscript
        ),
        _ => false,
    }
}

impl<'a, I: Iterator<Item = (Event<'a>, Range<usize>)>> Builder<'a, I> {
    /// Parse blocks until an `End` event (left unconsumed) or end of input.
    fn blocks(&mut self) -> Vec<Block> {
        let mut blocks = Vec::new();
        while let Some((event, range)) = self.events.peek() {
            if matches!(event, Event::End(_)) {
                break;
            }
            if is_inline_event(event) {
                // Tight list items carry inline content without a paragraph.
                let start = range.start;
                let (content, end) = self.inlines();
                if !content.is_empty() {
                    blocks.push(Block {
                        kind: BlockKind::Paragraph(content),
                        range: start..end.max(start),
                    });
                }
                continue;
            }
            let (event, range) = self.events.next().expect("peeked");
            if let Some(block) = self.block(event, range) {
                blocks.push(block);
            }
        }
        blocks
    }

    /// [`Self::blocks`] inside a non-item container (quote, footnote, …).
    fn nested_blocks(&mut self) -> Vec<Block> {
        self.containers += 1;
        let blocks = self.blocks();
        self.containers -= 1;
        blocks
    }

    /// Consume the `End` event closing the current container.
    fn end(&mut self) {
        self.events.next();
    }

    fn block(&mut self, event: Event<'a>, range: Range<usize>) -> Option<Block> {
        let kind = match event {
            Event::Rule => BlockKind::Rule,
            Event::Html(html) => BlockKind::Html(html.into_string()),
            Event::Start(tag) => self.container(tag)?,
            // Stray events outside any block: nothing to render.
            _ => return None,
        };
        Some(Block { kind, range })
    }

    fn container(&mut self, tag: Tag<'a>) -> Option<BlockKind> {
        let kind = match tag {
            Tag::Paragraph => {
                let depth = self.containers;
                if let Some(item) = self.items.last_mut()
                    && item.depth == depth
                {
                    item.loose = true;
                }
                let (content, _) = self.inlines();
                self.end();
                BlockKind::Paragraph(content)
            }
            Tag::Heading { level, .. } => {
                let (content, _) = self.inlines();
                self.end();
                BlockKind::Heading {
                    level: level as u8,
                    content,
                }
            }
            Tag::BlockQuote(kind) => {
                let blocks = self.nested_blocks();
                self.end();
                BlockKind::Quote {
                    alert: kind.map(|kind| match kind {
                        BlockQuoteKind::Note => AlertKind::Note,
                        BlockQuoteKind::Tip => AlertKind::Tip,
                        BlockQuoteKind::Important => AlertKind::Important,
                        BlockQuoteKind::Warning => AlertKind::Warning,
                        BlockQuoteKind::Caution => AlertKind::Caution,
                    }),
                    blocks,
                }
            }
            Tag::CodeBlock(kind) => {
                let lang = match kind {
                    CodeBlockKind::Fenced(info) => info
                        .split_whitespace()
                        .next()
                        .map(|lang| lang.split(',').next().unwrap_or(lang).to_owned()),
                    CodeBlockKind::Indented => None,
                };
                BlockKind::CodeBlock {
                    lang,
                    code: self.raw_text(),
                }
            }
            Tag::HtmlBlock => BlockKind::Html(self.raw_text()),
            Tag::List(start) => self.list(start),
            Tag::FootnoteDefinition(label) => {
                let blocks = self.nested_blocks();
                self.end();
                BlockKind::FootnoteDefinition {
                    label: label.into_string(),
                    blocks,
                }
            }
            Tag::Table(alignments) => self.table(alignments),
            // Anything else (not enabled, or not block-level here): keep its
            // content, drop the wrapper.
            _ => {
                let mut blocks = self.nested_blocks();
                self.end();
                return match blocks.len() {
                    0 => None,
                    1 => Some(blocks.remove(0).kind),
                    _ => Some(BlockKind::Quote {
                        alert: None,
                        blocks,
                    }),
                };
            }
        };
        Some(kind)
    }

    /// Concatenate text up to and including the closing `End`.
    fn raw_text(&mut self) -> String {
        let mut text = String::new();
        for (event, _) in self.events.by_ref() {
            match event {
                Event::End(_) => break,
                Event::Text(t) | Event::Html(t) | Event::Code(t) => text.push_str(&t),
                Event::SoftBreak | Event::HardBreak => text.push('\n'),
                _ => {}
            }
        }
        text
    }

    fn list(&mut self, start: Option<u64>) -> BlockKind {
        let mut items = Vec::new();
        let mut tight = true;
        while let Some((event, _)) = self.events.next() {
            match event {
                Event::Start(Tag::Item) => {
                    self.items.push(ItemState {
                        depth: self.containers,
                        ..ItemState::default()
                    });
                    let blocks = self.blocks();
                    self.end();
                    let state = self.items.pop().unwrap_or_default();
                    tight &= !state.loose;
                    items.push(ListItem {
                        task: state.task,
                        blocks,
                    });
                }
                Event::End(_) => break,
                _ => {}
            }
        }
        BlockKind::List {
            start,
            tight,
            items,
        }
    }

    fn table(&mut self, alignments: Vec<CmAlignment>) -> BlockKind {
        let alignments = alignments
            .into_iter()
            .map(|alignment| match alignment {
                CmAlignment::None => Alignment::None,
                CmAlignment::Left => Alignment::Left,
                CmAlignment::Center => Alignment::Center,
                CmAlignment::Right => Alignment::Right,
            })
            .collect();
        let mut header = Vec::new();
        let mut rows = Vec::new();
        while let Some((event, _)) = self.events.next() {
            match event {
                Event::Start(Tag::TableHead) => header = self.cells(),
                Event::Start(Tag::TableRow) => rows.push(self.cells()),
                Event::End(TagEnd::Table) => break,
                _ => {}
            }
        }
        BlockKind::Table {
            alignments,
            header,
            rows,
        }
    }

    /// Cells up to and including the row/head `End`.
    fn cells(&mut self) -> Vec<Cell> {
        let mut cells = Vec::new();
        while let Some((event, _)) = self.events.next() {
            match event {
                Event::Start(Tag::TableCell) => {
                    let (content, _) = self.inlines();
                    self.end();
                    cells.push(content);
                }
                Event::End(_) => break,
                _ => {}
            }
        }
        cells
    }

    /// Parse inlines until a non-inline event (left unconsumed). Returns the
    /// content and the end offset of the last consumed event.
    fn inlines(&mut self) -> (Vec<Inline>, usize) {
        let mut out = Vec::new();
        let mut end = 0;
        while let Some((event, _)) = self.events.peek() {
            if !is_inline_event(event) {
                break;
            }
            let (event, range) = self.events.next().expect("peeked");
            end = range.end;
            let inline = match event {
                Event::Text(t) => Inline::Text(t.into_string()),
                Event::Code(t) => Inline::Code(t.into_string()),
                Event::InlineMath(t) => Inline::Math {
                    display: false,
                    tex: t.into_string(),
                },
                Event::DisplayMath(t) => Inline::Math {
                    display: true,
                    tex: t.into_string(),
                },
                Event::InlineHtml(t) => Inline::Html(t.into_string()),
                Event::FootnoteReference(label) => Inline::FootnoteRef(label.into_string()),
                Event::SoftBreak => Inline::SoftBreak,
                Event::HardBreak => Inline::HardBreak,
                Event::TaskListMarker(checked) => {
                    if let Some(item) = self.items.last_mut() {
                        item.task.get_or_insert(checked);
                    }
                    continue;
                }
                Event::Start(tag) => {
                    let (content, _) = self.inlines();
                    if let Some((_, range)) = self.events.next() {
                        end = range.end;
                    }
                    match tag {
                        Tag::Emphasis => Inline::Emphasis(content),
                        Tag::Strong => Inline::Strong(content),
                        Tag::Strikethrough => Inline::Strikethrough(content),
                        Tag::Link { dest_url, .. } => Inline::Link {
                            url: dest_url.into_string(),
                            content,
                        },
                        Tag::Image { dest_url, .. } => Inline::Image {
                            url: dest_url.into_string(),
                            alt: content,
                        },
                        _ => Inline::Emphasis(content),
                    }
                }
                _ => continue,
            };
            out.push(inline);
        }
        (out, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<BlockKind> {
        parse(source).blocks.into_iter().map(|b| b.kind).collect()
    }

    fn text(t: &str) -> Inline {
        Inline::Text(t.to_owned())
    }

    #[test]
    fn paragraphs_and_inline_formatting() {
        assert_eq!(
            kinds("a **b** *c* `d` ~~e~~"),
            vec![BlockKind::Paragraph(vec![
                text("a "),
                Inline::Strong(vec![text("b")]),
                text(" "),
                Inline::Emphasis(vec![text("c")]),
                text(" "),
                Inline::Code("d".into()),
                text(" "),
                Inline::Strikethrough(vec![text("e")]),
            ])]
        );
    }

    #[test]
    fn headings_record_level() {
        assert_eq!(
            kinds("## Hi"),
            vec![BlockKind::Heading {
                level: 2,
                content: vec![text("Hi")]
            }]
        );
    }

    #[test]
    fn fenced_code_keeps_language_and_text() {
        assert_eq!(
            kinds("```rust ignore\nfn main() {}\n```"),
            vec![BlockKind::CodeBlock {
                lang: Some("rust".into()),
                code: "fn main() {}\n".into()
            }]
        );
    }

    #[test]
    fn tight_and_loose_lists() {
        let BlockKind::List { tight, items, .. } = &kinds("- a\n- b\n")[0] else {
            panic!()
        };
        assert!(*tight);
        assert_eq!(items.len(), 2);
        let BlockKind::List { tight, start, .. } = &kinds("1. a\n\n2. b\n")[0] else {
            panic!()
        };
        assert!(!*tight);
        assert_eq!(*start, Some(1));
    }

    #[test]
    fn quote_inside_tight_item_keeps_list_tight() {
        let BlockKind::List { tight, .. } = &kinds("- > q\n- b\n")[0] else {
            panic!()
        };
        assert!(*tight);
    }

    #[test]
    fn task_items_and_nested_lists() {
        let BlockKind::List { items, .. } = &kinds("- [x] done\n  - [ ] inner\n- plain")[0] else {
            panic!()
        };
        assert_eq!(items[0].task, Some(true));
        assert_eq!(items[1].task, None);
        let BlockKind::List { items: inner, .. } = &items[0].blocks[1].kind else {
            panic!()
        };
        assert_eq!(inner[0].task, Some(false));
    }

    #[test]
    fn alerts_and_quotes() {
        assert!(matches!(
            &kinds("> [!WARNING]\n> careful")[0],
            BlockKind::Quote { alert: Some(AlertKind::Warning), blocks } if blocks.len() == 1
        ));
        assert!(matches!(
            &kinds("> plain")[0],
            BlockKind::Quote { alert: None, .. }
        ));
    }

    #[test]
    fn tables() {
        let BlockKind::Table {
            alignments,
            header,
            rows,
        } = &kinds("| a | b |\n|:-|-:|\n| 1 | 2 |\n")[0]
        else {
            panic!()
        };
        assert_eq!(alignments, &[Alignment::Left, Alignment::Right]);
        assert_eq!(header, &vec![vec![text("a")], vec![text("b")]]);
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn links_images_footnotes_math_rules() {
        let blocks = kinds("[x](http://a) ![alt](i.png) $m$[^1]\n\n---\n\n[^1]: note");
        let BlockKind::Paragraph(inlines) = &blocks[0] else {
            panic!()
        };
        assert!(matches!(&inlines[0], Inline::Link { url, .. } if url == "http://a"));
        assert!(matches!(&inlines[2], Inline::Image { url, .. } if url == "i.png"));
        assert!(matches!(&inlines[4], Inline::Math { display: false, .. }));
        assert!(matches!(&inlines[5], Inline::FootnoteRef(l) if l == "1"));
        assert_eq!(blocks[1], BlockKind::Rule);
        assert!(matches!(&blocks[2], BlockKind::FootnoteDefinition { label, .. } if label == "1"));
    }

    #[test]
    fn stable_len_is_start_of_last_block() {
        let source = "one\n\ntwo\n\nthr";
        let doc = parse(source);
        assert_eq!(&source[doc.stable_len()..], "thr");
        assert_eq!(parse("").stable_len(), 0);
    }

    #[test]
    fn unresolved_references_hold_back_their_block() {
        let source = "[docs][ref]\n\nAnother paragraph\n\nlast";
        assert_eq!(parse(source).stable_len(), 0);
        let source = "ok\n\nsee [^1]\n\nlast";
        assert_eq!(&source[parse(source).stable_len()..], "see [^1]\n\nlast");
        let resolved = "[docs][ref]\n\nmore\n\n[ref]: https://x.y\n\nlast";
        assert!(parse(resolved).stable_len() > 0);
    }
}
