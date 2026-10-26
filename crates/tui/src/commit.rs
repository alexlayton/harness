//! Byte-offset arithmetic for splitting streaming markdown at stable block
//! boundaries, shared by the incremental scrollback commit.

/// Byte offset where the stable prefix of streaming `markdown` ends, i.e. the
/// start of the trailing in-progress block. Splits only at `\n\n` blank-line
/// boundaries that lie *outside* a code fence. Fences follow CommonMark: an
/// opener is at least three backticks or tildes (indented at most three
/// spaces), and only a run of the same character at least as long closes it,
/// so ```` ``` ```` inside a ```` ```` ```` fence or `~~~` fences are handled.
/// An opened-but-unclosed fence makes everything from its opener onward one
/// block, so nothing after it is ever split. Returns `None` when the whole
/// text is a single block with no stable prefix yet.
pub(crate) fn stable_block_split_offset(markdown: &str) -> Option<usize> {
    let mut fence: Option<(char, usize)> = None;
    let mut boundary = 0usize;
    let mut consumed = 0usize;
    for line in markdown.split('\n') {
        match fence {
            Some((character, length)) => {
                if fence_run(line).is_some_and(|(found, run, rest)| {
                    found == character && run >= length && rest.trim().is_empty()
                }) {
                    fence = None;
                }
            }
            None => {
                if let Some((character, run, rest)) = fence_run(line)
                    // A backtick fence's info string cannot contain backticks.
                    && !(character == '`' && rest.contains('`'))
                {
                    fence = Some((character, run));
                } else if line.trim().is_empty() {
                    // A blank line outside a fence closes a paragraph; the
                    // byte just past it begins a fresh block.
                    boundary = consumed + line.len() + 1;
                }
            }
        }
        consumed += line.len() + 1;
    }
    if boundary > 0 && boundary <= markdown.len() {
        Some(boundary)
    } else {
        None
    }
}

/// A fence run at the start of `line`: its character, length (at least 3),
/// and the text after it. Indentation of four or more spaces is code, not a
/// fence.
fn fence_run(line: &str) -> Option<(char, usize, &str)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let character = rest.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let run = rest.chars().take_while(|c| *c == character).count();
    (run >= 3).then(|| (character, run, &rest[run..]))
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

    #[test]
    fn stable_split_follows_fence_character_and_length() {
        // Tilde fences are fences too.
        let marked = "a\n\n~~~\nb\n\nc";
        assert_eq!(stable_block_split_offset(marked), Some(3));
        // A shorter run does not close a longer fence.
        let marked = "a\n\n````\n```\n\nb";
        assert_eq!(stable_block_split_offset(marked), Some(3));
        // A different character does not close it either.
        let marked = "a\n\n```\n~~~\n\nb";
        assert_eq!(stable_block_split_offset(marked), Some(3));
        // A matching closer ends the fence.
        let marked = "a\n\n~~~~\nx\n~~~~\n\nb";
        let offset = stable_block_split_offset(marked).unwrap();
        assert_eq!(&marked[offset..], "b");
    }
}
