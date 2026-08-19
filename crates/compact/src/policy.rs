//! Compaction policy for token-aware auto-compaction.
//!
//! Replaces session's old message-count policy: instead of "compact when more
//! than N message-like events are active", compaction is driven by the token
//! pressure on the provider context window.

/// Token-aware compaction policy.
///
/// The `auto` and `context_window` fields are configuration surface that the
/// *agent* acts on (the trigger decision); the planner consumes the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionPolicy {
    /// Auto-compact between turns when the trigger fires. `false` → manual
    /// `/compact` only.
    pub auto: bool,
    /// Auto-compact trigger fraction of the context window (0.8 = 80%).
    pub threshold: f64,
    /// Tokens reserved for the model's response after a trigger fires.
    pub reserve_tokens: u64,
    /// Recent turns kept verbatim (primary keep rule).
    pub keep_recent_turns: usize,
    /// Backstop cap on the verbatim tail, in estimated tokens.
    pub keep_recent_tokens: u64,
    /// Cap on the serialized text fed to the summarizer, in bytes.
    pub max_summary_input_bytes: usize,
    /// Cap on the generated summary itself, in bytes.
    pub max_summary_bytes: usize,
    /// Explicit context window override. `0` = derive from the model's
    /// reported `context_length`, falling back to a conservative default.
    pub context_window: u64,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            auto: true,
            threshold: 0.80,
            reserve_tokens: 16_384,
            keep_recent_turns: 10,
            keep_recent_tokens: 20_000,
            max_summary_input_bytes: 96 * 1024,
            max_summary_bytes: 12 * 1024,
            context_window: 0,
        }
    }
}

impl CompactionPolicy {
    /// Whether `context_tokens` has crossed the auto-compaction trigger.
    /// Returns `false` when auto-compaction is disabled or no window is known.
    pub fn should_auto_compact(&self, context_tokens: u64, context_window: u64) -> bool {
        if !self.auto || context_window == 0 {
            return false;
        }
        let trigger = (context_window as f64 * self.threshold) as u64;
        context_tokens > trigger.saturating_sub(self.reserve_tokens)
    }

    /// The context window to plan against: the config override if set,
    /// otherwise the model-reported value, otherwise a conservative default.
    pub fn resolved_window(&self, model_reported: u64) -> u64 {
        if self.context_window > 0 {
            self.context_window
        } else if model_reported > 0 {
            model_reported
        } else {
            DEFAULT_CONTEXT_WINDOW
        }
    }
}

/// Conservative fallback window when neither a config override nor a
/// model-reported value is available.
pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;

/// Default per-tool-result truncation for the summarizer transcript.
pub const DEFAULT_TOOL_RESULT_CHARS: usize = 2_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let policy = CompactionPolicy::default();
        assert!(policy.auto);
        assert_eq!(policy.threshold, 0.80);
        assert_eq!(policy.reserve_tokens, 16_384);
        assert_eq!(policy.keep_recent_turns, 10);
        assert_eq!(policy.keep_recent_tokens, 20_000);
        assert_eq!(policy.max_summary_input_bytes, 96 * 1024);
        assert_eq!(policy.max_summary_bytes, 12 * 1024);
        assert_eq!(policy.context_window, 0);
    }

    #[test]
    fn trigger_fires_past_threshold_minus_reserve() {
        let policy = CompactionPolicy::default();
        // 128k window → trigger at 0.80*128k - 16k = 86_016.
        assert!(!policy.should_auto_compact(86_016, 128_000));
        assert!(policy.should_auto_compact(86_017, 128_000));
    }

    #[test]
    fn trigger_respects_auto_flag_and_missing_window() {
        let policy = CompactionPolicy {
            auto: false,
            ..CompactionPolicy::default()
        };
        assert!(!policy.should_auto_compact(u64::MAX, 128_000));

        let policy = CompactionPolicy {
            auto: true,
            ..CompactionPolicy::default()
        };
        assert!(!policy.should_auto_compact(u64::MAX, 0));
    }

    #[test]
    fn window_resolution_priority_is_override_then_model_then_default() {
        let policy = CompactionPolicy {
            context_window: 65_536,
            ..CompactionPolicy::default()
        };
        assert_eq!(policy.resolved_window(200_000), 65_536);

        let policy = CompactionPolicy::default();
        assert_eq!(policy.resolved_window(200_000), 200_000);

        assert_eq!(policy.resolved_window(0), DEFAULT_CONTEXT_WINDOW);
    }
}
