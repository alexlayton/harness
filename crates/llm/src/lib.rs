pub mod dialects;
pub mod error;
pub mod http;
pub mod provider;
pub mod providers;
pub mod retry;
pub mod sse;
pub mod types;
pub mod util;

/// Install the `ring` rustls crypto provider once, before any HTTP client is
/// constructed. reqwest's `rustls` feature would otherwise pull in and default
/// to aws-lc-rs; `ring` is pure Rust + a small asm core and links smaller.
/// Installing an already-installed provider is an error in rustls, hence the
/// guard. If the install fails for any other reason the process keeps
/// reqwest's built-in default — correctness never depends on this.
fn install_ring_crypto_provider() {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    if STATE.load(Ordering::Acquire) == 1 {
        return;
    }
    if rustls::crypto::ring::default_provider()
        .install_default()
        .is_ok()
    {
        STATE.store(1, Ordering::Release);
    }
}

pub use error::LlmError;
pub use provider::{EventStream, Provider, RetryCallback};
pub use types::{
    CompletionRequest, Content, Message, ModelInfo, ReasoningEffort, ReasoningPolicy, Role,
    StreamEvent, SubscriptionUsage, SubscriptionUsageWindow, ToolCall, ToolDefinition, Usage,
};
pub use util::as_u64;
pub use util::{truncate_utf8, truncate_utf8_prefix};
