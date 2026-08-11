use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputAction {
    Submit,
    Newline,
    Interrupt,
    Quit,
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
        }) if modifiers.contains(KeyModifiers::CONTROL) => InputAction::Quit,
        Event::Key(KeyEvent {
            code: KeyCode::Char('d'),
            modifiers,
            ..
        }) if modifiers.contains(KeyModifiers::CONTROL) => InputAction::Quit,
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
        Event::Key(_) => InputAction::Edit,
        _ => InputAction::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> Event {
        Event::Key(KeyEvent::new(code, modifiers))
    }

    #[test]
    fn classifies_submit_newline_interrupt_and_quit() {
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
            InputAction::Quit
        );
    }
}
