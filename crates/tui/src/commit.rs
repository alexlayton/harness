//! Byte-offset arithmetic for splitting streaming markdown at stable block
//! boundaries, shared by the incremental scrollback commit.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
