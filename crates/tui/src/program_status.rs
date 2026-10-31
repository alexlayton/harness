//! Program Status Protocol (OSC 7501) reports.
//!
//! The protocol lets a program tell its terminal whether it is idle, working,
//! or finished, and leaves presentation to the terminal. Terminals that do not
//! know the sequence ignore it, so reports are safe to send without feature
//! detection. See <https://www.superlogical.com/rex/docs/build/program-status>.
//!
//! This module only builds the bytes and tracks which report is current. The
//! UI writes them between frames, so a report can never split a frame's
//! escape codes.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD_NO_PAD;

/// Stable program identifier sent as `app`.
pub const APP: &str = "harness";

/// Protocol limits for decoded text (`title` 192 bytes, `msg` 2048 bytes).
const MAX_TITLE_BYTES: usize = 192;
const MAX_MSG_BYTES: usize = 2048;

/// A state this program reports. `Clear` removes the record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Idle,
    Working,
    Done,
    Error,
    Clear,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Working => "working",
            Self::Done => "done",
            Self::Error => "error",
            Self::Clear => "clear",
        }
    }
}

/// Make free text safe for a single-line protocol value: control characters
/// (which make a report invalid) become spaces, the result is trimmed and cut
/// at a character boundary. `None` when nothing is left.
fn sanitize(text: &str, max_bytes: usize) -> Option<String> {
    let mut clean = String::new();
    for ch in text.chars() {
        let ch = if ch.is_control() { ' ' } else { ch };
        if clean.len() + ch.len_utf8() > max_bytes {
            break;
        }
        clean.push(ch);
    }
    let clean = clean.trim();
    (!clean.is_empty()).then(|| clean.to_owned())
}

fn encode_text(text: &str) -> String {
    STANDARD_NO_PAD.encode(text.as_bytes())
}

/// Build one report: `ESC ] 7501 ; state=…:app=harness[:title=…][:msg=…] ESC \`.
/// `Clear` carries no other keys, so it removes the root record outright.
pub fn encode(state: State, title: Option<&str>, msg: Option<&str>) -> String {
    let mut body = format!("state={}", state.label());
    if state != State::Clear {
        body.push_str(":app=");
        body.push_str(APP);
        if let Some(title) = title.and_then(|text| sanitize(text, MAX_TITLE_BYTES)) {
            body.push_str(":title=");
            body.push_str(&encode_text(&title));
        }
        if let Some(msg) = msg.and_then(|text| sanitize(text, MAX_MSG_BYTES)) {
            body.push_str(":msg=");
            body.push_str(&encode_text(&msg));
        }
    }
    format!("\x1b]7501;{body}\x1b\\")
}

/// Tracks the turn lifecycle and yields a report only when it changes.
///
/// A report replaces its record completely, so every report repeats `app` and
/// the title. A finished turn reports `done` (result not yet seen); a failed
/// one reports `error` with a short message; an interrupted one reports
/// `idle`, as the protocol asks.
#[derive(Debug)]
pub struct Tracker {
    state: State,
    title: Option<String>,
    msg: Option<String>,
    /// Failure seen during the current turn, reported when the turn ends.
    failure: Option<String>,
    interrupted: bool,
    /// The last encoded report written, to suppress repeats.
    sent: Option<String>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracker {
    /// A tracker whose first report is `idle`.
    pub fn new() -> Self {
        Self {
            state: State::Idle,
            title: None,
            msg: None,
            failure: None,
            interrupted: false,
            sent: None,
        }
    }

    fn set(&mut self, state: State, msg: Option<String>) {
        self.state = state;
        self.msg = msg;
    }

    /// A turn or foreground operation started or is still running.
    pub fn start_work(&mut self) {
        if self.state != State::Working {
            self.failure = None;
            self.interrupted = false;
            self.set(State::Working, None);
        }
    }

    /// Streaming output: the agent is working, and an earlier failure in this
    /// turn was recovered from.
    pub fn progress(&mut self) {
        self.start_work();
        self.failure = None;
    }

    /// An error event. Reported when the turn ends, since retries or
    /// recovery can still follow it.
    pub fn fail(&mut self, message: &str) {
        self.failure = Some(message.to_owned());
    }

    /// The user cancelled the running turn.
    pub fn interrupt(&mut self) {
        if self.state == State::Working {
            self.interrupted = true;
        }
    }

    /// The turn ended.
    pub fn finish_turn(&mut self) {
        let interrupted = std::mem::take(&mut self.interrupted);
        match self.failure.take() {
            _ if interrupted => self.set(State::Idle, None),
            Some(message) => self.set(State::Error, Some(message)),
            None => self.set(State::Done, None),
        }
    }

    /// An operation (for example `/compact`) ended. Commands are not results
    /// to look at, so success returns to `idle`. A skill runs a whole turn
    /// inside its operation: that turn already reported `done` or `error`, and
    /// the operation ending must not replace it.
    pub fn finish_operation(&mut self) {
        if self.state != State::Working {
            return;
        }
        self.interrupted = false;
        match self.failure.take() {
            Some(message) => self.set(State::Error, Some(message)),
            None => self.set(State::Idle, None),
        }
    }

    /// A new or loaded conversation replaces the old one.
    pub fn reset(&mut self) {
        self.failure = None;
        self.interrupted = false;
        self.set(State::Idle, None);
    }

    /// Set the display-only session title (`None` or blank removes it).
    pub fn set_title(&mut self, title: Option<&str>) {
        self.title = title.and_then(|text| sanitize(text, MAX_TITLE_BYTES));
    }

    /// The report to write now, or `None` when it equals the last one.
    pub fn take_report(&mut self) -> Option<String> {
        let report = encode(self.state, self.title.as_deref(), self.msg.as_deref());
        if self.sent.as_deref() == Some(report.as_str()) {
            return None;
        }
        self.sent = Some(report.clone());
        Some(report)
    }

    /// The report that removes this program's record on exit, so no stale
    /// `done` is left behind. `None` if nothing was ever reported.
    pub fn clear_report(&mut self) -> Option<String> {
        self.sent.take()?;
        Some(encode(State::Clear, None, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(value: &str) -> String {
        String::from_utf8(STANDARD_NO_PAD.decode(value).unwrap()).unwrap()
    }

    #[test]
    fn encodes_state_app_and_base64_text() {
        let report = encode(State::Error, Some("Fix bug"), Some("it broke"));
        let body = report
            .strip_prefix("\x1b]7501;")
            .and_then(|rest| rest.strip_suffix("\x1b\\"))
            .unwrap();
        let pairs: Vec<_> = body.split(':').collect();
        assert_eq!(pairs[0], "state=error");
        assert_eq!(pairs[1], "app=harness");
        assert_eq!(decode(pairs[2].strip_prefix("title=").unwrap()), "Fix bug");
        assert_eq!(decode(pairs[3].strip_prefix("msg=").unwrap()), "it broke");
        assert_eq!(
            encode(State::Clear, Some("x"), None),
            "\x1b]7501;state=clear\x1b\\"
        );
    }

    #[test]
    fn text_is_single_line_and_bounded() {
        assert_eq!(sanitize("a\nb\x1b[0m\tc", 64).as_deref(), Some("a b [0m c"));
        assert_eq!(sanitize(" \n ", 64), None);
        let cut = sanitize(&"é".repeat(200), MAX_TITLE_BYTES).unwrap();
        assert!(cut.len() <= MAX_TITLE_BYTES);
        let report = encode(State::Done, None, Some(&"x".repeat(10_000)));
        assert!(report.len() <= 4096);
    }

    #[test]
    fn turn_outcomes_map_to_states() {
        let mut tracker = Tracker::new();
        assert!(tracker.take_report().unwrap().contains("state=idle"));
        assert!(tracker.take_report().is_none());

        tracker.start_work();
        assert!(tracker.take_report().unwrap().contains("state=working"));
        tracker.finish_turn();
        assert!(tracker.take_report().unwrap().contains("state=done"));

        tracker.start_work();
        tracker.fail("boom");
        tracker.finish_turn();
        let report = tracker.take_report().unwrap();
        assert!(report.contains("state=error") && report.contains("msg="));

        tracker.start_work();
        tracker.interrupt();
        tracker.fail("cancelled");
        tracker.finish_turn();
        assert!(tracker.take_report().unwrap().contains("state=idle"));

        // A skill ends its turn, then its enclosing operation.
        tracker.start_work();
        tracker.finish_turn();
        tracker.finish_operation();
        assert!(tracker.take_report().unwrap().contains("state=done"));
        tracker.start_work();
        tracker.fail("skill failed");
        tracker.finish_turn();
        tracker.finish_operation();
        assert!(tracker.take_report().unwrap().contains("state=error"));

        // A command with no turn returns to idle.
        tracker.start_work();
        tracker.finish_operation();
        assert!(tracker.take_report().unwrap().contains("state=idle"));

        tracker.start_work();
        tracker.fail("transient");
        tracker.progress();
        tracker.finish_turn();
        assert!(tracker.take_report().unwrap().contains("state=done"));
    }

    #[test]
    fn clear_only_after_a_report() {
        let mut tracker = Tracker::new();
        assert!(tracker.clear_report().is_none());
        tracker.take_report();
        assert_eq!(
            tracker.clear_report().as_deref(),
            Some("\x1b]7501;state=clear\x1b\\")
        );
    }
}
