use super::persistence::usage_event;
use super::{Agent, AgentEvent, CompactionReason, MAX_OVERFLOW_RECOVERIES, TurnError, send};
use compact::{
    SummaryOutcome, estimate_live_tokens, plan_compaction, summarize as compact_summarize,
};
use llm::{Content, LlmError};
use session::{SessionEvent, usage_summary};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

impl Agent {
    // ------------------------------------------------------------------ compaction

    /// Resolve the provider context window: config override → model-reported
    /// `context_length` → conservative default. Runs once at startup and again
    /// after a model switch; a failed fetch keeps the current value.
    pub(crate) async fn refresh_context_window(&mut self) {
        let resolved = self.compaction.resolved_window(0);
        if self.compaction.context_window > 0 {
            self.context_window = resolved;
            return;
        }
        if let Ok(models) = self.provider.list_models().await
            && let Some(model) = models.iter().find(|model| {
                model.id == self.model || model.name.as_deref() == Some(self.model.as_str())
            })
        {
            self.context_window = self
                .compaction
                .resolved_window(model.context_length.unwrap_or(0));
            return;
        }
        self.context_window = self.compaction.resolved_window(0);
    }

    /// Approximate current context occupation: exact from the last request's
    /// `Done` usage when available, else an estimate over the live session.
    /// `extra_bytes` covers material added since that request (the new user
    /// message); it is small relative to the reserved response slack.
    pub(crate) fn context_tokens_estimate(&self, extra_bytes: usize) -> u64 {
        let base = match self.last_context_tokens {
            Some(exact) => exact,
            None => match self.session.as_ref() {
                Some(state) => estimate_live_tokens(&state.session),
                None => self.estimate_history_tokens(),
            },
        };
        base.saturating_add(compact::estimate::estimate_tokens(extra_bytes))
    }

    /// Estimate context tokens directly from `self.history` (no durable
    /// session / no provider usage yet).
    pub(crate) fn estimate_history_tokens(&self) -> u64 {
        let mut bytes = 0usize;
        for message in &self.history {
            for content in &message.content {
                match content {
                    Content::Text(text) | Content::Reasoning(text) => {
                        bytes = bytes.saturating_add(text.len())
                    }
                    Content::ToolResult { content, .. } => {
                        bytes = bytes.saturating_add(content.len())
                    }
                    Content::Opaque { data, .. } => {
                        bytes = bytes.saturating_add(
                            serde_json::to_string(data)
                                .map(|value| value.len())
                                .unwrap_or(0),
                        );
                    }
                    Content::ToolCall(call) => {
                        bytes = bytes.saturating_add(call.name.len());
                        bytes = bytes.saturating_add(
                            serde_json::to_string(&call.arguments)
                                .map(|rendered| rendered.len())
                                .unwrap_or(0),
                        );
                    }
                }
            }
        }
        compact::estimate::estimate_tokens(bytes)
    }

    /// Whether the pre-turn auto-compaction trigger fires for a turn adding
    /// `user_text`.
    pub(crate) fn should_auto_compact(&self, user_text: &str) -> bool {
        if !self.compaction.auto {
            return false;
        }
        let context = self.context_tokens_estimate(user_text.len());
        self.compaction
            .should_auto_compact(context, self.context_window)
    }

    /// Shared compaction routine used by the pre-turn trigger, manual
    /// `/compact`, and overflow recovery. Plans, summarizes (LLM with a
    /// deterministic fallback), persists the summary + the summarizer's usage,
    /// and rebuilds `self.history` from the new boundary. Returns `false`
    /// (with a `Notice`) when there is nothing to compact or persistence
    /// failed.
    pub(crate) async fn compact_and_reload(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        reason: CompactionReason,
    ) -> Result<bool, TurnError> {
        let Some(state) = self.session.as_ref() else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return Ok(false);
        };
        let session = state.session.clone();

        let estimated = self.context_tokens_estimate(0);
        let Some(plan) = plan_compaction(&session, &self.compaction, estimated) else {
            send(events, AgentEvent::Notice("nothing to compact yet".into()));
            return Ok(false);
        };

        let outcome = compact_summarize(
            self.provider.as_ref(),
            &self.model,
            &plan,
            &self.compaction,
            cancel,
        )
        .await;

        // Persist the summarizer's own usage so session cost totals stay
        // honest and the UI reflects it.
        if let SummaryOutcome::Model { usage, .. } = &outcome {
            let summary = usage_summary(usage);
            self.persist_usage_best_effort(summary, events);
        }

        let compacted_through = plan.boundary;
        let summary = match &outcome {
            SummaryOutcome::Model { text, .. } | SummaryOutcome::Deterministic { text } => text,
        };
        let summary_bytes = summary.len();

        self.persist_event(
            SessionEvent::CompactionSummary {
                summary: summary.clone(),
                compacted_through,
            },
            events,
        )?;

        // This is also the fix for the manual `/compact` no-op: without this
        // rebuild the live conversation would keep stale (uncompacted) history
        // until next restart.
        if let Some(state) = self.session.as_ref() {
            self.history = state.session.context_messages();
            send(events, usage_event(&state.session.metadata.usage));
        }

        send(
            events,
            AgentEvent::CompactionFinished {
                compacted_through,
                summary_bytes,
                auto: reason == CompactionReason::Auto,
                reason,
            },
        );
        Ok(true)
    }

    /// Handle a context-overflow provider error (a 400 whose body matches
    /// context-exceeded patterns) by emergency-compacting and returning
    /// whether the caller should retry the request. Bounded to
    /// `MAX_OVERFLOW_RECOVERIES` per turn.
    pub(crate) async fn try_overflow_recovery(
        &mut self,
        error: &LlmError,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        attempts: &mut usize,
    ) -> Result<bool, TurnError> {
        if !is_context_overflow(error) || *attempts >= MAX_OVERFLOW_RECOVERIES {
            return Ok(false);
        }
        *attempts += 1;
        if self
            .compact_and_reload(events, cancel, CompactionReason::Overflow)
            .await?
        {
            send(
                events,
                AgentEvent::Notice(format!(
                    "overflow recovery: compacted ({attempts}/{MAX_OVERFLOW_RECOVERIES}); retrying"
                )),
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Whether a provider error is a context-window rejection (the request it
/// describes was too large to admit). Proxies vary in both status and
/// phrasing, so we match on the rendered error text; a 400 alone is not
/// enough (it could be a malformed request).
pub(crate) fn is_context_overflow(error: &LlmError) -> bool {
    let body = error.to_string().to_ascii_lowercase();
    const PATTERNS: &[&str] = &[
        "context length",
        "context_length",
        "context window",
        "too many tokens",
        "maximum context",
        "max context",
        "maximum prompt",
        "input is too long",
        "exceeds the maximum",
        "token limit",
    ];
    PATTERNS.iter().any(|pattern| body.contains(pattern))
}
