pub mod dialects;
pub mod error;
pub mod http;
pub mod provider;
pub mod providers;
pub mod retry;
pub mod sse;
pub mod types;
pub mod util;

pub use error::LlmError;
pub use provider::{EventStream, Provider, RetryCallback};
pub use types::{
    CompletionRequest, Content, Message, ModelInfo, Role, StreamEvent, ToolCall, ToolDefinition,
    Usage,
};
pub use util::as_u64;
pub use util::{truncate_utf8, truncate_utf8_prefix};
