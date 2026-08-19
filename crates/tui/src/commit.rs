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
use crate::state::TranscriptEntry;
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

impl crate::Tui {
    /// Commit every finalized entry at the front of the uncommitted tail in a
    /// single `insert_before` call (one call per batch, never per streaming
    /// delta). `committed` advances only on success so the invariant
    /// `transcript[0..committed] is in scrollback` always holds.
    pub(crate) fn commit_ready_entries(&mut self) -> anyhow::Result<()> {
        let width = self.terminal.size().map(|s| s.width as usize).unwrap_or(80);
        let (mut lines, new_committed) =
            collect_ready_lines(&self.transcript, self.committed, width, self.theme);
        if lines.is_empty() {
            return Ok(());
        }
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

    /// One-line separator written into scrollback, e.g. `── new conversation ──`.
    pub(crate) fn commit_separator(&mut self, label: &str) -> anyhow::Result<()> {
        let width = self.terminal.size().map(|s| s.width as usize).unwrap_or(80);
        let text = format!("── {label} ──");
        let pad = width.saturating_sub(UnicodeWidthStr::width(text.as_str()));
        let left = pad / 2;
        let right = pad - left;
        let mut lines = vec![Line::from(Span::styled(
            format!("{}{}{}", "─".repeat(left), text, "─".repeat(right)),
            Style::default().fg(self.theme.dim_text),
        ))];
        self.commit_lines(&mut lines)
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
}
