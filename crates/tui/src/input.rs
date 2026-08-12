use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputAction {
    Submit,
    Newline,
    Interrupt,
    Quit,
    ExpandDetails,
    PageUp,
    PageDown,
    Bottom,
    FocusTools,
    Edit,
    Ignore,
}

pub fn classify(event: &Event) -> InputAction {
    match event {
        Event::Paste(_) => InputAction::Edit,
        Event::Key(KeyEvent {
            code: KeyCode::Char('c'),
            modifiers,
            ..
        }) if modifiers.contains(KeyModifiers::CONTROL) => InputAction::Interrupt,
        Event::Key(KeyEvent {
            code: KeyCode::Char('d'),
            modifiers,
            ..
        }) if modifiers.contains(KeyModifiers::CONTROL) => InputAction::Quit,
        Event::Key(KeyEvent {
            code: KeyCode::Char(value),
            modifiers,
            ..
        }) if modifiers.contains(KeyModifiers::CONTROL) && value.eq_ignore_ascii_case(&'o') => {
            InputAction::ExpandDetails
        }
        Event::Key(KeyEvent {
            code: KeyCode::Tab,
            modifiers: KeyModifiers::NONE,
            ..
        }) => InputAction::FocusTools,
        Event::Key(KeyEvent {
            code: KeyCode::Esc, ..
        }) => InputAction::Interrupt,
        Event::Key(KeyEvent {
            code: KeyCode::Enter,
            modifiers,
            ..
        }) if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => InputAction::Newline,
        Event::Key(KeyEvent {
            code: KeyCode::Enter,
            ..
        }) => InputAction::Submit,
        Event::Key(KeyEvent {
            code: KeyCode::PageUp,
            ..
        }) => InputAction::PageUp,
        Event::Key(KeyEvent {
            code: KeyCode::PageDown,
            ..
        }) => InputAction::PageDown,
        Event::Key(KeyEvent {
            code: KeyCode::End, ..
        }) => InputAction::Bottom,
        Event::Key(_) => InputAction::Edit,
        _ => InputAction::Ignore,
    }
}

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

/// Add a submitted message to bounded history. Consecutive duplicate
/// submissions are intentionally collapsed.
pub fn push_history(entries: &mut Vec<String>, value: &str, cap: usize) {
    if entries.last().is_some_and(|last| last == value) {
        return;
    }
    entries.push(value.to_owned());
    if entries.len() > cap {
        let excess = entries.len() - cap;
        entries.drain(..excess);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> Event {
        Event::Key(KeyEvent::new(code, modifiers))
    }

    #[test]
    fn classifies_submit_newline_interrupt_quit_tools_and_scrolling() {
        assert_eq!(
            classify(&key(KeyCode::Enter, KeyModifiers::NONE)),
            InputAction::Submit
        );
        assert_eq!(
            classify(&key(KeyCode::Enter, KeyModifiers::ALT)),
            InputAction::Newline
        );
        assert_eq!(
            classify(&key(KeyCode::Esc, KeyModifiers::NONE)),
            InputAction::Interrupt
        );
        assert_eq!(
            classify(&key(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            InputAction::Interrupt
        );
        assert_eq!(
            classify(&key(KeyCode::Char('d'), KeyModifiers::CONTROL)),
            InputAction::Quit
        );
        assert_eq!(
            classify(&key(KeyCode::Char('o'), KeyModifiers::CONTROL)),
            InputAction::ExpandDetails
        );
        assert_eq!(
            classify(&key(KeyCode::PageUp, KeyModifiers::NONE)),
            InputAction::PageUp
        );
        assert_eq!(
            classify(&key(KeyCode::End, KeyModifiers::CONTROL)),
            InputAction::Bottom
        );
        assert_eq!(
            classify(&key(KeyCode::Tab, KeyModifiers::NONE)),
            InputAction::FocusTools
        );
    }

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
}
