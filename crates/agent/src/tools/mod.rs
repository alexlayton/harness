mod bash;
mod read;
mod registry;
mod write;

pub use bash::{BashTool, truncate_command_output};
pub use read::ReadTool;
pub use registry::ToolRegistry;
pub use write::WriteTool;

use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub summary: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput;
}

pub fn default_registry() -> ToolRegistry {
    ToolRegistry::new(vec![
        Box::new(ReadTool),
        Box::new(WriteTool),
        Box::new(BashTool),
    ])
}

/// A useful one-line preview before a tool has completed.  The completed tool
/// may replace this with a more precise summary, but keeping this pure makes
/// the agent event available immediately.
pub fn call_summary(name: &str, args: &Value) -> String {
    match name {
        "read" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("read {path}"))
            .unwrap_or_else(|| "read".into()),
        "write" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("write {path}"))
            .unwrap_or_else(|| "write".into()),
        "bash" => args
            .get("command")
            .and_then(Value::as_str)
            .map(|command| format!("bash: {}", first_line(command)))
            .unwrap_or_else(|| "bash".into()),
        _ => name.to_owned(),
    }
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value)
}
