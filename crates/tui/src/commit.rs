//! The commit pipeline: finalized transcript entries are written into the
//! terminal's native scrollback exactly once via `insert_before`. The inline
//! viewport then holds only the uncommitted tail (a streaming assistant
//! and/or a running tool), so committed content is immutable pixels and every
//! frame repaints at most a couple of entries.
//!
//! Invariants (see Phase 5 tests):
//! - `transcript[0..committed]` are all final and already in scrollback.
//! - `transcript[committed..]` contains only non-final entries.
//! - `insert_before` is called only from the commit paths in this module.

use crate::render::{self, Theme};
use crate::state::{EntryId, TranscriptEntry};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget};
use unicode_width::UnicodeWidthStr;

/// An entry is "final" when it will never change again: it can be committed to
/// scrollback now and never redrawn or mutated afterwards.
pub(crate) fn is_final(entry: &TranscriptEntry) -> bool {
    match entry {
        TranscriptEntry::User { .. }
        | TranscriptEntry::Notice { .. }
        | TranscriptEntry::Error { .. } => true,
        TranscriptEntry::Assistant { streaming, .. } => !*streaming,
        TranscriptEntry::Tool { record, .. } => !record.status.is_running(),
    }
}

/// Collect the contiguous run of final entries starting at `committed` into
/// wrapped, gap-separated lines. Returns the lines and the new `committed`
/// index (one past the last final entry). Pure, so the commit boundary can be
/// unit-tested without a TTY; the `insert_before` call stays thin.
pub(crate) fn collect_ready_lines(
    transcript: &[TranscriptEntry],
    committed: usize,
    width: usize,
    theme: Theme,
) -> (Vec<Line<'static>>, usize) {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut new_committed = committed;
    for (index, entry) in transcript.iter().enumerate().skip(committed) {
        if !is_final(entry) {
            break; // the live tail starts here
        }
        if !lines.is_empty() {
            render::push_blank(&mut lines, render::SECTION_GAP);
        }
        // Tool boxes commit collapsed; committed content is immutable.
        lines.extend(render::entry_lines(entry, false, width, theme));
        new_committed = index + 1;
    }
    (lines, new_committed)
}

/// Byte offset where the stable prefix of streaming `markdown` ends, i.e. the
/// start of the trailing in-progress block. Splits only at `\n\n` blank-line
/// boundaries that lie *outside* a code fence (tracking ``` parity while
/// scanning); an opened-but-unclosed fence makes everything from its opener
/// onward one block, so nothing after it is ever split. Returns `None` when
/// the whole text is a single block with no stable prefix yet.
pub(crate) fn stable_block_split_offset(markdown: &str) -> Option<usize> {
    let mut in_fence = false;
    let mut boundary = 0usize;
    let mut consumed = 0usize;
    for line in markdown.split('\n') {
        let trimmed = line.trim_start();
        if in_fence {
            if trimmed.starts_with("```") {
                in_fence = false;
            }
        } else if trimmed.starts_with("```") {
            in_fence = true;
        } else if trimmed.is_empty() {
            // A blank line outside a fence closes a paragraph; the byte just
            // past it (the start of the next line) begins a fresh block.
            boundary = consumed + line.len() + 1;
        }
        consumed += line.len() + 1;
    }
    if boundary > 0 && boundary <= markdown.len() {
        Some(boundary)
    } else {
        None
    }
}

/// Replace the streaming assistant entry `id` with a finalized prefix entry
/// (`markdown[..offset]`, whose `streaming` flag is false) followed by a fresh
/// streaming tail entry (`markdown[offset..]`, new id `tail_id`). Pure
/// transcript surgery so the commit boundary is unit-testable without a TTY.
pub(crate) fn split_streaming_assistant(
    transcript: &mut Vec<TranscriptEntry>,
    id: EntryId,
    offset: usize,
    tail_id: EntryId,
) -> Option<()> {
    let index = transcript.iter().position(|entry| entry.id() == id)?;
    let TranscriptEntry::Assistant {
        markdown,
        reasoning,
        ..
    } = &transcript[index]
    else {
        return None;
    };
    let prefix_entry = TranscriptEntry::Assistant {
        id,
        markdown: markdown[..offset].to_owned(),
        reasoning: reasoning.clone(),
        streaming: false,
    };
    let tail_entry = TranscriptEntry::Assistant {
        id: tail_id,
        markdown: markdown[offset..].to_owned(),
        reasoning: String::new(),
        streaming: true,
    };
    transcript.splice(index..=index, [prefix_entry, tail_entry]);
    Some(())
}

impl crate::Tui {
    /// Commit every finalized entry at the front of the uncommitted tail in a
    /// single `insert_before` call (one call per batch, never per streaming
    /// delta). `committed` advances only on success so the invariant
    /// `transcript[0..committed] is in scrollback` always holds.
    pub(crate) fn commit_ready_entries(&mut self) -> anyhow::Result<()> {
        let width = self.terminal.size().map(|s| s.width).unwrap_or(80);
        let (mut lines, new_committed) = collect_ready_lines(
            &self.transcript,
            self.committed,
            render::content_width(width),
            self.theme,
        );
        if lines.is_empty() {
            return Ok(());
        }
        // Committed lines keep the same gutter as the live viewport: wrapped
        // at `content_width`, then indented so scrollback content never
        // touches the window edge and does not reflow on commit.
        render::indent_lines(&mut lines, render::horizontal_pad(width));
        // Theoretically unreachable: a single snapshot bulk commit exceeding
        // 65_535 rows is a bug elsewhere. Keep the newest rows rather than
        // wrapping the u16 height.
        if lines.len() > u16::MAX as usize {
            self.add_error(format!(
                "scrollback commit exceeded {} rows; truncating older content",
                u16::MAX
            ));
            lines.truncate(u16::MAX as usize);
        }
        if let Err(error) = self.commit_lines(&mut lines) {
            // Surface io failures as an error in the UI, never crash.
            self.add_error(format!("failed to write to scrollback: {error:#}"));
            return Ok(());
        }
        self.committed = new_committed;
        Ok(())
    }

    /// One-line separator written into scrollback, e.g. `── new conversation ──`,
    /// with one blank line of breathing room above and below.
    pub(crate) fn commit_separator(&mut self, label: &str) -> anyhow::Result<()> {
        let width = self.terminal.size().map(|s| s.width).unwrap_or(80);
        let text = format!("── {label} ──");
        let inner = render::content_width(width);
        let pad = inner.saturating_sub(UnicodeWidthStr::width(text.as_str()));
        let left = pad / 2;
        let right = pad - left;
        let mut lines = Vec::new();
        render::push_blank(&mut lines, render::SECTION_GAP);
        lines.push(Line::from(Span::styled(
            format!("{}{}{}", "─".repeat(left), text, "─".repeat(right)),
            Style::default().fg(self.theme.dim_text),
        )));
        render::push_blank(&mut lines, render::SECTION_GAP);
        render::indent_lines(&mut lines, render::horizontal_pad(width));
        self.commit_lines(&mut lines)
    }

    /// Commit the startup welcome banner (title + random tagline + version)
    /// into scrollback exactly once, before the first `draw()`. The transcript
    /// is empty at that point; a later `/load` commits its history below the
    /// banner, which is accepted.
    pub(crate) fn commit_welcome_banner(&mut self) -> anyhow::Result<()> {
        if self.welcome_shown {
            return Ok(());
        }
        self.welcome_shown = true;
        if !self.transcript.is_empty() {
            return Ok(());
        }
        let width = self.terminal.size().map(|s| s.width).unwrap_or(80);
        let mut lines = render::welcome_lines(render::content_width(width), self.theme);
        render::indent_lines(&mut lines, render::horizontal_pad(width));
        if let Err(error) = self.commit_lines(&mut lines) {
            self.add_error(format!("failed to write welcome banner: {error:#}"));
        }
        Ok(())
    }

    /// While an assistant streams, commit the stable prefix of its markdown
    /// into scrollback once it outgrows the live-tail budget, keeping only the
    /// trailing in-progress block live. Long responses then flow into
    /// scrollback incrementally instead of appearing all at once at finalize.
    ///
    /// The prefix is finalized as its own committed assistant entry (spliced
    /// in place of the streaming entry); the end-of-event `commit_ready_entries`
    /// writes it below the viewport in a single `insert_before`. The in-progress
    /// block keeps streaming under a fresh entry id until `ToolCallStarted`/
    /// `TurnFinished` finalizes it.
    pub(crate) fn commit_streamed_prefix(&mut self) -> anyhow::Result<()> {
        let Some(id) = self.streaming_assistant else {
            return Ok(());
        };
        let width = self.terminal.size().map(|s| s.width).unwrap_or(80);
        let content = render::content_width(width);
        let index = match self.transcript.iter().position(|entry| entry.id() == id) {
            Some(index) => index,
            None => return Ok(()),
        };
        let TranscriptEntry::Assistant {
            markdown,
            reasoning,
            ..
        } = &self.transcript[index]
        else {
            return Ok(());
        };
        let Some(offset) = stable_block_split_offset(markdown) else {
            return Ok(()); // single block so far; nothing stable to commit
        };
        if offset == 0 {
            return Ok(());
        }
        // Only make room once the completed prefix already overflows the live
        // tail budget; never call the commit path on every streaming delta.
        let budget = self.streamed_tail_budget();
        // Snapshot the prefix (including any completed reasoning) to measure
        // exactly the rows the eventual commit will render.
        let prefix_entry = TranscriptEntry::Assistant {
            id,
            markdown: markdown[..offset].to_owned(),
            reasoning: reasoning.clone(),
            streaming: false,
        };
        let prefix_height = render::entry_lines(&prefix_entry, false, content, self.theme).len();
        if prefix_height <= budget {
            return Ok(());
        }
        let tail_id = self.allocate_id();
        split_streaming_assistant(&mut self.transcript, id, offset, tail_id);
        self.streaming_assistant = Some(tail_id);
        Ok(())
    }

    /// The live-tail row budget applied while streaming: the tail region of a
    /// busy fixed-height viewport held at a minimal single-line prompt.
    fn streamed_tail_budget(&self) -> usize {
        // Mirror `draw`: ratatui clamps the inline viewport to the terminal
        // rows when the terminal shrinks below the fixed height, and the live
        // sections live inside `Margin { vertical }`, which removes one row at
        // the top *and* the bottom of the viewport.
        let rows = self.terminal.size().map(|s| s.height).unwrap_or(24);
        let height = self.viewport_height.min(rows);
        let canvas = height.saturating_sub(2 * render::vertical_pad(height));
        crate::layout::live_layout(canvas, true, 1, 0)
            .tail_rows
            .max(1) as usize
    }

    /// Render `lines` into a temporary buffer and splice it above the inline
    /// viewport. `insert_before` (default ratatui features) clears the
    /// viewport afterwards, so callers must `draw()` in the same event-loop
    /// iteration — the event loop already repaints after every event.
    fn commit_lines(&mut self, lines: &mut [Line<'static>]) -> anyhow::Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let height = lines.len().min(u16::MAX as usize) as u16;
        self.terminal
            .insert_before(height, |buf| {
                Paragraph::new(Text::from(lines.to_owned())).render(buf.area, buf);
            })
            .map_err(anyhow::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ToolRecord, ToolStatus};

    fn record(status: ToolStatus) -> ToolRecord {
        ToolRecord {
            name: "bash".into(),
            args: "{}".into(),
            summary: "bash: echo hi".into(),
            ok: !matches!(status, ToolStatus::Failure),
            duration_ms: 1,
            output: "hi".into(),
            error: None,
            status,
        }
    }

    #[test]
    fn is_final_table() {
        let entries = [
            (
                TranscriptEntry::User {
                    id: 1,
                    text: "hi".into(),
                },
                true,
            ),
            (
                TranscriptEntry::Notice {
                    id: 2,
                    text: "n".into(),
                },
                true,
            ),
            (
                TranscriptEntry::Error {
                    id: 3,
                    text: "e".into(),
                },
                true,
            ),
            (
                TranscriptEntry::Assistant {
                    id: 4,
                    markdown: "".into(),
                    reasoning: "".into(),
                    streaming: true,
                },
                false,
            ),
            (
                TranscriptEntry::Assistant {
                    id: 5,
                    markdown: "done".into(),
                    reasoning: "".into(),
                    streaming: false,
                },
                true,
            ),
            (
                TranscriptEntry::Tool {
                    id: 6,
                    record: record(ToolStatus::Running),
                    expanded: false,
                },
                false,
            ),
            (
                TranscriptEntry::Tool {
                    id: 7,
                    record: record(ToolStatus::Success),
                    expanded: false,
                },
                true,
            ),
            (
                TranscriptEntry::Tool {
                    id: 8,
                    record: record(ToolStatus::Failure),
                    expanded: false,
                },
                true,
            ),
        ];
        for (entry, expected) in entries {
            assert_eq!(is_final(&entry), expected, "entry {entry:?}");
        }
    }

    #[test]
    fn collect_ready_lines_commits_exactly_the_final_prefix() {
        let transcript = vec![
            TranscriptEntry::User {
                id: 1,
                text: "run the tool".into(),
            },
            TranscriptEntry::Assistant {
                id: 2,
                markdown: "let me check".into(),
                reasoning: String::new(),
                streaming: true,
            },
            TranscriptEntry::Tool {
                id: 3,
                record: record(ToolStatus::Running),
                expanded: false,
            },
            TranscriptEntry::Tool {
                id: 4,
                record: record(ToolStatus::Success),
                expanded: false,
            },
        ];
        // Nothing committed yet: the final prefix is just the user message;
        // the streaming assistant and the running tool stay live (the finished
        // tool after them is not reached).
        let (lines, committed) = collect_ready_lines(&transcript, 0, 40, Theme::default());
        assert_eq!(committed, 1);
        assert!(!lines.is_empty());
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("run the tool"));
        assert!(!text.contains("let me check"));
    }

    #[test]
    fn collect_ready_lines_resumes_from_a_committed_offset() {
        let transcript = vec![
            TranscriptEntry::User {
                id: 1,
                text: "old committed message".into(),
            },
            TranscriptEntry::Assistant {
                id: 2,
                markdown: "the answer".into(),
                reasoning: String::new(),
                streaming: false,
            },
            TranscriptEntry::Notice {
                id: 3,
                text: "a note".into(),
            },
        ];
        // Entry 0 is already committed; the next two final entries are picked
        // up and `committed` advances past them, without revisiting entry 0.
        let (lines, committed) = collect_ready_lines(&transcript, 1, 40, Theme::default());
        assert_eq!(committed, 3);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("the answer"));
        assert!(text.contains("a note"));
        assert!(!text.contains("old committed"));
    }

    #[test]
    fn stable_split_ends_before_the_last_completed_paragraph() {
        assert_eq!(stable_block_split_offset("a"), None, "single paragraph");
        assert_eq!(
            stable_block_split_offset("a\n\nb"),
            Some(3),
            "two paragraphs: prefix is the first"
        );
        assert_eq!(
            stable_block_split_offset("a\n\nb\n\nc"),
            Some(6),
            "last boundary wins: prefix is a+b"
        );
        assert_eq!(
            stable_block_split_offset("a\nb"),
            None,
            "single newline is not a paragraph boundary"
        );
        // A trailing separator would leave an empty in-progress block; defer.
        assert_eq!(stable_block_split_offset("a\n\n"), None);
    }

    #[test]
    fn stable_split_never_splits_inside_code_fences() {
        // The blank line inside the fence must not split; the one after the
        // closing fence may.
        let marked = "a\n\n```\nb\n\nc\n```\n\nd";
        let offset = stable_block_split_offset(marked).unwrap();
        assert_eq!(&marked[..offset], "a\n\n```\nb\n\nc\n```\n\n");
        assert_eq!(&marked[offset..], "d");

        // An unclosed fence makes everything from its opener onward one block:
        // only the text before the opener is stable.
        let marked = "a\n\n```\nb\n\nc";
        let offset = stable_block_split_offset(marked).unwrap();
        assert_eq!(&marked[..offset], "a\n\n");
        assert_eq!(&marked[offset..], "```\nb\n\nc");
    }

    #[test]
    fn split_streaming_assistant_replaces_one_entry_with_prefix_and_tail() {
        let mut transcript = vec![TranscriptEntry::Assistant {
            id: 1,
            markdown: "first\n\nsecond tail".into(),
            reasoning: "thinking".into(),
            streaming: true,
        }];
        assert_eq!(
            split_streaming_assistant(&mut transcript, 1, 7, 2),
            Some(())
        );
        assert_eq!(transcript.len(), 2);
        match &transcript[0] {
            TranscriptEntry::Assistant {
                id,
                markdown,
                reasoning,
                streaming,
            } => {
                assert_eq!(*id, 1);
                assert_eq!(markdown, "first\n\n");
                assert_eq!(reasoning, "thinking");
                assert!(!streaming);
            }
            _ => panic!("prefix must stay an assistant entry"),
        }
        match &transcript[1] {
            TranscriptEntry::Assistant {
                id,
                markdown,
                reasoning,
                streaming,
            } => {
                assert_eq!(*id, 2);
                assert_eq!(markdown, "second tail");
                assert!(reasoning.is_empty(), "tail reasoning starts fresh");
                assert!(streaming);
            }
            _ => panic!("tail must stay an assistant entry"),
        }
        // The finalized prefix is now committed by the regular commit pipeline,
        // leaving only the tail live.
        let (_, committed) = collect_ready_lines(&transcript, 0, 40, Theme::default());
        assert_eq!(committed, 1);
    }
}
