use crate::MCP_CALL_TIMEOUT;
use crate::output::flatten;
use crate::{McpError, normalized_tool_name};
use async_trait::async_trait;
use llm::ToolDefinition;
use rmcp::Peer;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CancelledNotification, CancelledNotificationParam,
    ClientRequest, ServerResult, Tool as RemoteTool,
};
use rmcp::service::{PeerRequestOptions, RoleClient};
use serde_json::{Map, Value};
use std::fmt;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tools::{Tool, ToolOutput, ToolPrompt, ToolSpec};

/// Maximum number of remote tools accepted from one server.
pub(crate) const MAX_REMOTE_TOOLS: usize = 256;
/// Maximum aggregate bytes in the registered remote tool definitions.
pub(crate) const MAX_REMOTE_DEFINITION_BYTES: usize = 512 * 1024;
const MAX_TOOL_NAME_BYTES: usize = 256;
const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;
const MAX_SCHEMA_DEPTH: usize = 32;
const MAX_SCHEMA_NODES: usize = 10_000;
const MAX_SCHEMA_STRING_BYTES: usize = 64 * 1024;
const MAX_SCHEMA_BYTES: usize = 256 * 1024;

/// Validate one untrusted remote tool before it is registered or cloned into
/// the local registry. Returns its bounded definition size for catalogue
/// accounting.
pub(crate) fn validate_remote_tool(server: &str, remote: &RemoteTool) -> Result<usize, McpError> {
    let name = remote.name.as_ref();
    if name.trim().is_empty() {
        return Err(McpError::Tool {
            server: server.into(),
            tool: name.to_owned(),
            message: "tool name is empty".into(),
        });
    }
    if name.len() > MAX_TOOL_NAME_BYTES || name.chars().any(char::is_control) {
        return Err(McpError::Tool {
            server: server.into(),
            tool: truncate_name(name),
            message: "tool name is too long or contains control characters".into(),
        });
    }
    if let Some(title) = &remote.title
        && (title.len() > MAX_TOOL_DESCRIPTION_BYTES || title.chars().any(char::is_control))
    {
        return Err(McpError::Tool {
            server: server.into(),
            tool: name.to_owned(),
            message: "tool title is too long or contains control characters".into(),
        });
    }
    if let Some(description) = &remote.description
        && (description.len() > MAX_TOOL_DESCRIPTION_BYTES
            || description.chars().any(char::is_control))
    {
        return Err(McpError::Tool {
            server: server.into(),
            tool: name.to_owned(),
            message: "tool description is too long or contains control characters".into(),
        });
    }
    let mut stats = SchemaStats::default();
    visit_schema_object(&remote.input_schema, 0, &mut stats).map_err(|message| McpError::Tool {
        server: server.into(),
        tool: name.to_owned(),
        message,
    })?;
    if let Some(output) = &remote.output_schema {
        visit_schema_object(output, 0, &mut stats).map_err(|message| McpError::Tool {
            server: server.into(),
            tool: name.to_owned(),
            message,
        })?;
    }
    // Count the compact serialized Harness definition, including the
    // namespaced name and generated description. This prevents a long server
    // name or JSON escaping from bypassing the aggregate catalogue budget.
    let generated_description = format!(
        "MCP tool `{name}` from server `{server}`.{}",
        remote
            .description
            .as_deref()
            .filter(|description| !description.is_empty())
            .map_or(String::new(), |description| format!(" {description}"))
    );
    let definition = serde_json::json!({
        "name": normalized_tool_name(server, name),
        "description": generated_description,
        "parameters": *remote.input_schema,
    });
    serde_json::to_vec(&definition)
        .map(|bytes| bytes.len())
        .map_err(|error| McpError::Tool {
            server: server.into(),
            tool: name.to_owned(),
            message: format!("tool definition cannot be serialized: {error}"),
        })
}

#[derive(Default)]
struct SchemaStats {
    nodes: usize,
    bytes: usize,
}

fn visit_schema_object(
    object: &Map<String, Value>,
    depth: usize,
    stats: &mut SchemaStats,
) -> Result<(), String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(format!(
            "input schema exceeds depth limit of {MAX_SCHEMA_DEPTH}"
        ));
    }
    stats.nodes = stats.nodes.saturating_add(1);
    stats.bytes = stats.bytes.saturating_add(2);
    if stats.nodes > MAX_SCHEMA_NODES {
        return Err(format!(
            "input schema exceeds node limit of {MAX_SCHEMA_NODES}"
        ));
    }
    for (key, value) in object {
        visit_schema_value(value, depth + 1, stats)?;
        stats.bytes = stats.bytes.saturating_add(key.len()).saturating_add(4);
    }
    if stats.bytes > MAX_SCHEMA_BYTES {
        return Err(format!(
            "input schema exceeds byte limit of {MAX_SCHEMA_BYTES}"
        ));
    }
    Ok(())
}

fn visit_schema_value(value: &Value, depth: usize, stats: &mut SchemaStats) -> Result<(), String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(format!(
            "input schema exceeds depth limit of {MAX_SCHEMA_DEPTH}"
        ));
    }
    stats.nodes = stats.nodes.saturating_add(1);
    if stats.nodes > MAX_SCHEMA_NODES {
        return Err(format!(
            "input schema exceeds node limit of {MAX_SCHEMA_NODES}"
        ));
    }
    match value {
        Value::Object(object) => visit_schema_object(object, depth, stats)?,
        Value::Array(values) => {
            stats.bytes = stats.bytes.saturating_add(2);
            for value in values {
                visit_schema_value(value, depth + 1, stats)?;
            }
        }
        Value::String(value) => {
            if value.len() > MAX_SCHEMA_STRING_BYTES {
                return Err(format!(
                    "input schema string exceeds byte limit of {MAX_SCHEMA_STRING_BYTES}"
                ));
            }
            stats.bytes = stats.bytes.saturating_add(value.len() + 2);
        }
        _ => stats.bytes = stats.bytes.saturating_add(8),
    }
    if stats.bytes > MAX_SCHEMA_BYTES {
        return Err(format!(
            "input schema exceeds byte limit of {MAX_SCHEMA_BYTES}"
        ));
    }
    Ok(())
}

fn truncate_name(value: &str) -> String {
    llm::util::truncate_utf8(value, MAX_TOOL_NAME_BYTES)
}

/// Adapter from one discovered MCP tool to Harness's protocol-neutral tool
/// interface. It is intentionally exclusive: server annotations are untrusted.
pub(crate) struct McpTool {
    name: String,
    server: String,
    original_name: String,
    description: String,
    parameters: Value,
    peer: Peer<RoleClient>,
}

impl McpTool {
    pub(crate) fn new(
        server: &str,
        remote: RemoteTool,
        peer: Peer<RoleClient>,
    ) -> Result<Self, McpError> {
        let original_name = remote.name.to_string();
        validate_remote_tool(server, &remote)?;
        if original_name.trim().is_empty() {
            return Err(McpError::Tool {
                server: server.into(),
                tool: original_name,
                message: "tool name is empty".into(),
            });
        }
        let parameters = Value::Object((*remote.input_schema).clone());
        let description = remote
            .description
            .map(|description| description.to_string())
            .unwrap_or_default();
        Ok(Self {
            name: normalized_tool_name(server, &original_name),
            server: server.to_owned(),
            original_name,
            description: truncate(&description, 4 * 1024),
            parameters,
            peer,
        })
    }
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        let mut description = format!(
            "MCP tool `{}` from server `{}`.",
            self.original_name, self.server
        );
        if !self.description.is_empty() {
            description.push(' ');
            description.push_str(&self.description);
        }
        ToolSpec {
            definition: ToolDefinition {
                name: self.name.clone(),
                description,
                parameters: self.parameters.clone(),
            },
            prompt: ToolPrompt::default(),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let arguments = match args {
            Value::Object(arguments) => arguments,
            _ => return self.error("MCP tool arguments must be a JSON object"),
        };
        let params =
            CallToolRequestParams::new(self.original_name.clone()).with_arguments(arguments);
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let deadline = tokio::time::Instant::now().checked_add(MCP_CALL_TIMEOUT);
        let Some(deadline) = deadline else {
            return self.error("MCP tool call deadline overflow");
        };
        let mut handle = tokio::select! {
            _ = cancel.cancelled() => return self.error("MCP tool call cancelled"),
            result = tokio::time::timeout_at(
                deadline,
                self.peer.send_cancellable_request(request, PeerRequestOptions::no_options()),
            ) => match result {
                Ok(Ok(handle)) => handle,
                Ok(Err(error)) => return self.error(format_args!("MCP tools/call failed: {error}")),
                Err(_) => return self.error("MCP tools/call request timed out"),
            }
        };
        tokio::select! {
            _ = cancel.cancelled() => {
                self.cancel_request_bounded(&handle).await;
                self.error("MCP tool call cancelled")
            }
            result = tokio::time::timeout_at(deadline, &mut handle.rx) => match result {
                Ok(Ok(Ok(ServerResult::CallToolResult(result)))) => ToolOutput { content: flatten(&result), is_error: result.is_error.unwrap_or(false), summary: self.summary() },
                Ok(Ok(Ok(_))) => self.error("MCP tools/call returned an unsupported non-final response"),
                Ok(Ok(Err(error))) => self.error(format_args!("MCP tools/call failed: {error}")),
                Ok(Err(_)) => self.error("MCP tools/call connection closed before a response"),
                Err(_) => {
                    self.cancel_request_bounded(&handle).await;
                    self.error("MCP tool call timed out")
                }
            }
        }
    }
}

impl McpTool {
    fn summary(&self) -> String {
        format!("{}:{}", self.server, self.original_name)
    }
    async fn cancel_request_bounded(&self, handle: &rmcp::service::RequestHandle<RoleClient>) {
        let _ = tokio::time::timeout(Duration::from_secs(1), self.cancel_request(handle)).await;
    }

    async fn cancel_request(
        &self,
        handle: &rmcp::service::RequestHandle<RoleClient>,
    ) -> Result<(), rmcp::ServiceError> {
        handle
            .peer
            .send_notification(
                CancelledNotification::new(CancelledNotificationParam::new(
                    Some(handle.id.clone()),
                    Some("Harness cancelled the tool call".into()),
                ))
                .into(),
            )
            .await
    }

    fn error(&self, message: impl fmt::Display) -> ToolOutput {
        ToolOutput {
            content: crate::output::cap_display(message),
            is_error: true,
            summary: self.summary(),
        }
    }
}

fn truncate(value: &str, maximum: usize) -> String {
    if value.len() <= maximum {
        value.into()
    } else {
        format!(
            "{}…",
            llm::util::truncate_utf8(value, maximum.saturating_sub(3))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use std::sync::Arc;

    fn remote(name: &str) -> RemoteTool {
        let mut tool = RemoteTool::default();
        tool.name = Cow::Owned(name.to_owned());
        tool.input_schema = Arc::new(Map::new());
        tool
    }

    #[test]
    fn remote_tool_validation_rejects_control_names() {
        let tool = remote("bad\u{1b}[2J");
        assert!(validate_remote_tool("server", &tool).is_err());
    }

    #[test]
    fn remote_tool_validation_rejects_deep_schemas() {
        let mut schema = serde_json::json!({});
        for _ in 0..(MAX_SCHEMA_DEPTH + 1) {
            schema = serde_json::json!({"nested": schema});
        }
        let mut tool = remote("deep");
        tool.input_schema = Arc::new(schema.as_object().unwrap().clone());
        let error = validate_remote_tool("server", &tool).unwrap_err();
        assert!(error.to_string().contains("depth"));
    }

    #[test]
    fn remote_tool_validation_rejects_oversized_schema_strings() {
        let mut tool = remote("large");
        tool.input_schema = Arc::new(
            [(
                "description".to_owned(),
                Value::String("x".repeat(MAX_SCHEMA_STRING_BYTES + 1)),
            )]
            .into_iter()
            .collect(),
        );
        let error = validate_remote_tool("server", &tool).unwrap_err();
        assert!(error.to_string().contains("string"));
    }

    #[test]
    fn remote_tool_validation_returns_definition_size() {
        let tool = remote("read");
        assert!(validate_remote_tool("server", &tool).unwrap() > 0);
    }
}
