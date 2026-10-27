/// Recall the previous history entry. `current` is used only on the first
/// recall to preserve the user's draft for the eventual downward navigation.
pub fn history_previous(
    entries: &[String],
    position: &mut Option<usize>,
    draft: &mut String,
    current: &str,
) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let next = match *position {
        Some(position) => position.saturating_sub(1),
        None => {
            *draft = current.to_owned();
            entries.len() - 1
        }
    };
    *position = Some(next);
    Some(entries[next].clone())
}

/// Recall a newer history entry, or restore the saved draft after the newest
/// entry has been passed.
pub fn history_next(
    entries: &[String],
    position: &mut Option<usize>,
    draft: &str,
) -> Option<String> {
    let current = (*position)?;
    if current + 1 < entries.len() {
        let next = current + 1;
        *position = Some(next);
        Some(entries[next].clone())
    } else {
        *position = None;
        Some(draft.to_owned())
    }
}

/// Largest message kept in input history. Recalling a truncated paste would
/// submit different text, so larger messages are not kept at all.
pub const MAX_HISTORY_ENTRY_BYTES: usize = 64 * 1024;

/// Total byte budget of input history, in addition to the entry count cap.
pub const MAX_HISTORY_BYTES: usize = 1024 * 1024;

/// Add a submitted message to bounded history. Consecutive duplicate
/// submissions are intentionally collapsed. The oldest entries are dropped
/// to respect both `cap` entries and [`MAX_HISTORY_BYTES`].
pub fn push_history(entries: &mut Vec<String>, value: &str, cap: usize) {
    if value.len() > MAX_HISTORY_ENTRY_BYTES || entries.last().is_some_and(|last| last == value) {
        return;
    }
    entries.push(value.to_owned());
    let mut excess = entries.len().saturating_sub(cap);
    let mut bytes = entries[excess..].iter().map(String::len).sum::<usize>();
    while bytes > MAX_HISTORY_BYTES && excess + 1 < entries.len() {
        bytes -= entries[excess].len();
        excess += 1;
    }
    entries.drain(..excess);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_navigation_preserves_and_restores_draft() {
        let entries = vec!["one".into(), "two".into(), "three".into()];
        let mut position = None;
        let mut draft = "draft".to_owned();
        assert_eq!(
            history_previous(&entries, &mut position, &mut draft, "draft"),
            Some("three".into())
        );
        assert_eq!(
            history_previous(&entries, &mut position, &mut draft, "three"),
            Some("two".into())
        );
        assert_eq!(
            history_next(&entries, &mut position, &draft),
            Some("three".into())
        );
        assert_eq!(
            history_next(&entries, &mut position, &draft),
            Some("draft".into())
        );
        assert_eq!(position, None);
    }

    #[test]
    fn history_is_bounded_and_dedupes_consecutive_values() {
        let mut entries = Vec::new();
        push_history(&mut entries, "one", 2);
        push_history(&mut entries, "one", 2);
        push_history(&mut entries, "two", 2);
        push_history(&mut entries, "three", 2);
        assert_eq!(entries, vec!["two", "three"]);
    }

    #[test]
    fn history_respects_its_byte_budget() {
        let mut entries = Vec::new();
        push_history(
            &mut entries,
            &"x".repeat(MAX_HISTORY_ENTRY_BYTES + 1),
            1_000,
        );
        assert!(entries.is_empty(), "oversized entries are not kept");
        let entry = "y".repeat(MAX_HISTORY_ENTRY_BYTES / 2);
        for index in 0..64 {
            push_history(&mut entries, &format!("{index}{entry}"), 1_000);
        }
        assert!(entries.iter().map(String::len).sum::<usize>() <= MAX_HISTORY_BYTES);
        assert!(entries.last().unwrap().starts_with("63"));
    }
}
