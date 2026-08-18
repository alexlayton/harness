//! Token-aware context compaction for Harness.
//!
//! What lives here versus the [`session`] crate: `session` owns the durable
//! event log and the `CompactionSummary` event type — the *storage format* of
//! a compaction. This crate owns the *policy, planning, and summarization*:
//!
//! - [`policy::CompactionPolicy`] — the auto-compaction knobs (token-aware,
//!   replacing the old message-count policy).
//! - [`estimate`] — cheap ~4 bytes/token estimation used to pick a cut point.
//! - [`plan::plan_compaction`] — pure cut-point selection over a session.
//! - [`serialize`] — event → text transcript serialization for the summarizer.
//! - [`summarize`] — the LLM summarizer with a deterministic fallback.
//!
//! The agent owns *when* compaction triggers and *how the result is
//! persisted* (appending the summary event and rebuilding provider history),
//! so durable writes stay in one place.

pub mod estimate;
pub mod plan;
pub mod policy;
pub mod serialize;
pub mod summarize;

pub use plan::{CompactionPlan, estimate_live_tokens, plan_compaction};
pub use policy::CompactionPolicy;
pub use summarize::{SummaryOutcome, summarize};
