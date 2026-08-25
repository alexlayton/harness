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
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tools::{Tool, ToolOutput, ToolPrompt, ToolSpec};

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
        let mut handle = tokio::select! {
            _ = cancel.cancelled() => return self.error("MCP tool call cancelled"),
            result = self.peer.send_cancellable_request(request, PeerRequestOptions::no_options()) => match result {
                Ok(handle) => handle,
                Err(error) => return self.error(&format!("MCP tools/call failed: {error}")),
            }
        };
        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = self.cancel_request(&handle).await;
                self.error("MCP tool call cancelled")
            }
            result = &mut handle.rx => match result {
                Ok(Ok(ServerResult::CallToolResult(result))) => ToolOutput { content: flatten(&result), is_error: result.is_error.unwrap_or(false), summary: self.summary() },
                Ok(Ok(_)) => self.error("MCP tools/call returned an unsupported non-final response"),
                Ok(Err(error)) => self.error(&format!("MCP tools/call failed: {error}")),
                Err(_) => self.error("MCP tools/call connection closed before a response"),
            }
        }
    }
}

impl McpTool {
    fn summary(&self) -> String {
        format!("{}:{}", self.server, self.original_name)
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

    fn error(&self, message: &str) -> ToolOutput {
        ToolOutput {
            content: message.into(),
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
