//! Transport-isolated Model Context Protocol client support for Harness.
//!
//! MCP wire types intentionally stay inside this crate. Connected remote tools
//! are exposed as ordinary [`tools::Tool`] implementations.

mod client;
mod config;
mod names;
mod output;
mod runtime;
mod tool;

pub use config::{McpConfig, McpServerConfig, McpTransportConfig};
pub use names::normalized_tool_name;
pub use runtime::McpRuntime;

use std::path::PathBuf;
use std::time::Duration;

/// Upper bound for MCP process initialization.
pub(crate) const MCP_INITIALIZE_TIMEOUT: Duration = Duration::from_secs(15);
/// Upper bound for one complete tools/list catalogue request.
pub(crate) const MCP_LIST_TIMEOUT: Duration = Duration::from_secs(15);
/// Upper bound spanning one tools/call request and its response.
pub(crate) const MCP_CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// One global deadline for shutting down all MCP services.
pub(crate) const MCP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(4);
/// Fixed stderr read buffer; server diagnostics are discarded by default.
pub(crate) const MCP_STDERR_CHUNK_BYTES: usize = 4096;
/// Maximum JSON bytes in one newline-delimited MCP frame, excluding its final
/// newline. This is enforced while reading stdio, before rmcp deserializes it.
pub(crate) const MCP_MAX_FRAME_BYTES: usize = 1024 * 1024;

/// A named MCP lifecycle error. Its display form deliberately excludes command
/// environment values and HTTP headers.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// A server configuration is invalid.
    #[error("invalid MCP configuration for server `{server}`: {message}")]
    Config { server: String, message: String },
    /// A server could not complete an MCP operation.
    #[error("MCP server `{server}` failed during {operation}: {message}")]
    Operation {
        server: String,
        operation: &'static str,
        message: String,
    },
    /// A remote tool cannot be represented safely as a Harness tool.
    #[error("MCP server `{server}` returned invalid tool `{tool}`: {message}")]
    Tool {
        server: String,
        tool: String,
        message: String,
    },
    /// Registration would violate the existing registry's naming invariant.
    #[error(transparent)]
    Registry(#[from] tools::ToolRegistryError),
}

impl McpError {
    pub(crate) fn operation(
        server: &str,
        operation: &'static str,
        error: impl std::fmt::Display,
    ) -> Self {
        Self::Operation {
            server: server.to_owned(),
            operation,
            message: output::cap_display(error),
        }
    }
}

/// Expand `${NAME}` placeholders without reading or logging environment values.
/// Missing variables are reported by name so configuration mistakes are actionable.
/// The resulting value is bounded by the MCP configuration field limit.
pub fn expand_environment(
    value: &str,
    lookup: impl FnMut(&str) -> Option<String>,
) -> Result<String, String> {
    config::expand_environment(value, lookup)
}

/// Canonical workspace root accepted by the runtime.
pub(crate) fn root_uri(root: &std::path::Path) -> Result<String, McpError> {
    let path = PathBuf::from(root);
    url::Url::from_file_path(path)
        .map(|url| url.into())
        .map_err(|_| McpError::Config {
            server: "<workspace>".into(),
            message: "workspace root cannot be represented as a file URI".into(),
        })
}
