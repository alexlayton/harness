//! Frontend-neutral reduction of one turn's [`AgentEvent`]s into its final
//! answer and failure state, shared by headless mode and the SDK so they
//! cannot drift (ACP shares [`clears_error`]).
//!
//! Prose from a tool-bearing model round is intermediate, not the answer.
//! Concurrent tool calls are tracked by call id rather than by a counter, so
//! a duplicate or unknown finish cannot unbalance the state.

use agent::AgentEvent;
use anyhow::{Result, anyhow, ensure};
use std::collections::HashSet;

/// Whether `event` is a genuine recovery after an error. Metadata, retries,
/// and tool completions do not clear a failure; text or a new tool call does.
pub fn clears_error(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::TextDelta(_) | AgentEvent::ToolCallStarted { .. }
    )
}

/// Accumulates one turn's answer and error state.
#[derive(Debug, Default)]
pub struct TurnOutputReducer {
    active_tools: HashSet<String>,
    pending: String,
    answer: String,
    error: Option<String>,
    finished: bool,
}

impl TurnOutputReducer {
    /// Apply one event. Returns prose that was discarded as intermediate
    /// because a tool round started, so verbose frontends can show it.
    pub fn observe(&mut self, event: &AgentEvent) -> Option<String> {
        let mut intermediate = None;
        if clears_error(event) {
            self.error = None;
        }
        match event {
            AgentEvent::TextDelta(text) => self.pending.push_str(text),
            AgentEvent::ToolCallStarted { call_id, .. } => {
                if !self.pending.is_empty() {
                    intermediate = Some(std::mem::take(&mut self.pending));
                }
                self.active_tools.insert(call_id.clone());
            }
            AgentEvent::ToolCallFinished { call_id, .. } => {
                // An unknown or repeated finish is ignored rather than
                // lowering a counter that belongs to another running call.
                if self.active_tools.remove(call_id) && self.active_tools.is_empty() {
                    self.pending.clear();
                }
            }
            AgentEvent::Error(message) => {
                self.pending.clear();
                self.error = Some(message.clone());
            }
            AgentEvent::TurnFinished => {
                self.answer = if self.active_tools.is_empty() {
                    std::mem::take(&mut self.pending)
                } else {
                    String::new()
                };
                // Nothing may leak into a later turn.
                self.pending.clear();
                self.active_tools.clear();
                self.finished = true;
            }
            _ => {}
        }
        intermediate
    }

    /// Drop buffered prose and in-flight tools: a cancelled round is never an
    /// answer.
    pub fn discard_pending(&mut self) {
        self.pending.clear();
        self.active_tools.clear();
    }

    /// The final answer; empty until a text-only round ended the turn.
    pub fn answer(&self) -> &str {
        &self.answer
    }

    /// The failure that was not followed by a recovery.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Whether `TurnFinished` was observed.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// The answer, or the unrecovered error, or a not-finished error.
    pub fn into_result(self) -> Result<String> {
        if let Some(error) = self.error {
            return Err(anyhow!(error));
        }
        ensure!(self.finished, "Harness stopped before the turn completed");
        Ok(self.answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(id: &str) -> AgentEvent {
        AgentEvent::ToolCallStarted {
            call_id: id.into(),
            name: "read".into(),
            summary: "read".into(),
        }
    }

    fn finish(id: &str) -> AgentEvent {
        AgentEvent::ToolCallFinished {
            call_id: id.into(),
            name: "read".into(),
            summary: "read".into(),
            ok: true,
            duration_ms: 1,
            output: String::new(),
            error: None,
        }
    }

    fn text(value: &str) -> AgentEvent {
        AgentEvent::TextDelta(value.into())
    }

    fn run(events: &[AgentEvent]) -> TurnOutputReducer {
        let mut reducer = TurnOutputReducer::default();
        for event in events {
            reducer.observe(event);
        }
        reducer
    }

    #[test]
    fn answer_excludes_tool_round_prose() {
        let mut reducer = TurnOutputReducer::default();
        assert_eq!(reducer.observe(&text("plan")), None);
        assert_eq!(reducer.observe(&start("a")), Some("plan".into()));
        reducer.observe(&text("more"));
        reducer.observe(&finish("a"));
        reducer.observe(&text("Final"));
        reducer.observe(&AgentEvent::TurnFinished);
        assert_eq!(reducer.into_result().unwrap(), "Final");
    }

    #[test]
    fn concurrent_calls_discard_text_until_all_finish() {
        let reducer = run(&[
            start("a"),
            start("b"),
            finish("a"),
            text("still intermediate"),
            finish("b"),
            text("X"),
            AgentEvent::TurnFinished,
        ]);
        assert_eq!(reducer.answer(), "X");
    }

    #[test]
    fn unknown_or_duplicate_finish_cannot_unbalance_tracking() {
        let reducer = run(&[
            start("a"),
            finish("zzz"),
            text("not yet"),
            AgentEvent::TurnFinished,
        ]);
        // `a` never finished, so the turn has no text-only final round.
        assert_eq!(reducer.answer(), "");
        let reducer = run(&[
            start("a"),
            finish("a"),
            finish("a"),
            text("Y"),
            AgentEvent::TurnFinished,
        ]);
        assert_eq!(reducer.answer(), "Y");
    }

    #[test]
    fn state_resets_at_turn_finished() {
        let mut reducer = run(&[start("a"), AgentEvent::TurnFinished]);
        assert_eq!(reducer.answer(), "");
        reducer.observe(&text("Z"));
        reducer.observe(&AgentEvent::TurnFinished);
        assert_eq!(reducer.answer(), "Z");
    }

    #[test]
    fn error_survives_lifecycle_events_but_not_recovery() {
        let failed = run(&[
            text("partial"),
            AgentEvent::Error("provider failed".into()),
            finish("unrelated"),
            AgentEvent::TurnFinished,
        ]);
        assert_eq!(
            failed.into_result().unwrap_err().to_string(),
            "provider failed"
        );

        let by_text = run(&[
            AgentEvent::Error("x".into()),
            text("ok"),
            AgentEvent::TurnFinished,
        ]);
        assert_eq!(by_text.into_result().unwrap(), "ok");

        let by_tool = run(&[
            AgentEvent::Error("x".into()),
            start("a"),
            finish("a"),
            AgentEvent::TurnFinished,
        ]);
        assert_eq!(by_tool.into_result().unwrap(), "");
    }

    #[test]
    fn clears_error_is_text_or_tool_start_only() {
        assert!(clears_error(&text("t")));
        assert!(clears_error(&start("a")));
        assert!(!clears_error(&AgentEvent::ReasoningDelta("r".into())));
        assert!(!clears_error(&finish("a")));
    }

    #[test]
    fn cancelled_round_has_no_answer_and_unfinished_is_an_error() {
        let mut reducer = run(&[text("half a thought")]);
        reducer.discard_pending();
        reducer.observe(&AgentEvent::TurnFinished);
        assert_eq!(reducer.answer(), "");
        assert!(TurnOutputReducer::default().into_result().is_err());
    }
}
