//! Native library entry points for applications that embed Harness.
//!
//! The CLI and embedded callers share configuration and agent assembly. This
//! library does not install signal handlers, change cwd, or initialize logging.

/// Configuration resolution shared by the CLI and native callers.
pub mod config;
/// Workspace instruction discovery shared by Harness frontends.
pub mod context;
/// One-shot, frontend-independent agent execution.
pub mod sdk;
