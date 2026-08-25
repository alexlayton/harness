use crate::McpError;
use crate::client::HarnessClient;
use crate::config::{McpServerConfig, McpTransportConfig};
use crate::tool::McpTool;
use rmcp::ClientLifecycleMode;
use rmcp::model::Tool as RemoteTool;
use rmcp::service::{RoleClient, RunningService, serve_client_with_lifecycle_and_ct};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tools::ToolRegistry;

/// Connected MCP servers for one agent/session. The runtime owns protocol and
/// child-process lifetimes; tools only retain cloneable request handles.
pub struct McpRuntime {
    servers: Vec<ConnectedServer>,
}

struct ConnectedServer {
    name: String,
    client: RunningService<RoleClient, HarnessClient>,
    tools: Vec<RemoteTool>,
    stderr_task: Option<JoinHandle<()>>,
}

impl McpRuntime {
    /// Connect every configured server and discover its complete static tool
    /// catalogue. Startup is atomic: a failure shuts down already connected
    /// servers before returning the named error.
    pub async fn connect(
        servers: &[McpServerConfig],
        workspace_root: &Path,
        cancel: CancellationToken,
    ) -> Result<Self, McpError> {
        let mut configs = servers.to_vec();
        configs.sort_by(|left, right| left.name.cmp(&right.name));
        crate::McpConfig {
            servers: configs.clone(),
        }
        .validate()?;
        let mut connected = Self {
            servers: Vec::new(),
        };
        for server in &configs {
            if cancel.is_cancelled() {
                connected.shutdown().await;
                return Err(McpError::operation(&server.name, "initialize", "cancelled"));
            }
            match connect_server(server, workspace_root, cancel.clone()).await {
                Ok(server) => connected.servers.push(server),
                Err(error) => {
                    connected.shutdown().await;
                    return Err(error);
                }
            }
        }
        Ok(connected)
    }

    /// Register all discovered remote tools into a registry. The caller must
    /// retain this runtime until every resulting agent tool call has completed.
    pub fn register_into(&self, registry: &mut ToolRegistry) -> Result<(), McpError> {
        for server in &self.servers {
            for remote in &server.tools {
                registry.register(Box::new(McpTool::new(
                    &server.name,
                    remote.clone(),
                    server.client.peer().clone(),
                )?))?;
            }
        }
        Ok(())
    }

    /// Close protocol services, reap stdio children, and stop stderr readers.
    pub async fn shutdown(mut self) {
        for server in &mut self.servers {
            let _ = server
                .client
                .close_with_timeout(std::time::Duration::from_secs(4))
                .await;
            if let Some(task) = server.stderr_task.take() {
                task.abort();
                let _ = task.await;
            }
        }
    }
}

async fn connect_server(
    server: &McpServerConfig,
    workspace_root: &Path,
    cancel: CancellationToken,
) -> Result<ConnectedServer, McpError> {
    let McpTransportConfig::Stdio { command, args, env } = &server.transport else {
        return Err(McpError::operation(
            &server.name,
            "initialize",
            "HTTP transport is not enabled in this build",
        ));
    };
    let mut command = tokio::process::Command::new(command);
    command
        .args(args)
        .current_dir(workspace_root)
        .envs(env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let (transport, stderr) = TokioChildProcess::builder(command.configure(|_| {}))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| McpError::operation(&server.name, "initialize", error))?;
    let stderr_task = stderr.map(|stderr| spawn_stderr_reader(server.name.clone(), stderr));
    let handler = HarnessClient::new(workspace_root)?;
    let client = serve_client_with_lifecycle_and_ct(
        handler,
        transport,
        ClientLifecycleMode::Initialize,
        cancel,
    )
    .await
    .map_err(|error| McpError::operation(&server.name, "initialize", error))?;
    let tools = client
        .peer()
        .list_all_tools()
        .await
        .map_err(|error| McpError::operation(&server.name, "tools/list", error))?;
    tracing::debug!(server = %server.name, tools = tools.len(), "connected MCP server");
    Ok(ConnectedServer {
        name: server.name.clone(),
        client,
        tools,
        stderr_task,
    })
}

fn spawn_stderr_reader(name: String, stderr: tokio::process::ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let line = llm::util::truncate_utf8(&line, 2048);
            tracing::debug!(server = %name, stderr = %line, "MCP server stderr");
        }
    })
}
