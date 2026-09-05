use crate::client::HarnessClient;
use crate::config::{McpServerConfig, McpTransportConfig};
use crate::tool::McpTool;
use crate::{
    MCP_INITIALIZE_TIMEOUT, MCP_LIST_TIMEOUT, MCP_SHUTDOWN_TIMEOUT, MCP_STDERR_CHUNK_BYTES,
    McpError,
};
use rmcp::ClientLifecycleMode;
use rmcp::model::Tool as RemoteTool;
use rmcp::service::{RoleClient, RunningService, serve_client_with_lifecycle_and_ct};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use std::path::Path;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
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

    /// Close protocol services, reap stdio children, and stop stderr readers
    /// under one global deadline rather than one timeout per server.
    pub async fn shutdown(mut self) {
        let mut servers = std::mem::take(&mut self.servers);
        let _ = tokio::time::timeout(MCP_SHUTDOWN_TIMEOUT, async {
            for server in &mut servers {
                let _ = server.client.close_with_timeout(MCP_SHUTDOWN_TIMEOUT).await;
            }
        })
        .await;
        for server in &mut servers {
            if let Some(task) = server.stderr_task.take() {
                task.abort();
                let _ = task.await;
            }
        }
        // Dropping the services after the deadline releases their transports;
        // the configured child process uses kill-on-drop below.
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
        .kill_on_drop(true)
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
    let server_cancel = cancel.child_token();
    let mut client = match tokio::time::timeout(
        MCP_INITIALIZE_TIMEOUT,
        serve_client_with_lifecycle_and_ct(
            handler,
            transport,
            ClientLifecycleMode::Initialize,
            server_cancel.clone(),
        ),
    )
    .await
    {
        Ok(Ok(client)) => client,
        Ok(Err(error)) => {
            abort_stderr_task(stderr_task).await;
            return Err(McpError::operation(&server.name, "initialize", error));
        }
        Err(_) => {
            server_cancel.cancel();
            abort_stderr_task(stderr_task).await;
            return Err(McpError::operation(
                &server.name,
                "initialize",
                "request timed out",
            ));
        }
    };
    let tools = match tokio::time::timeout(MCP_LIST_TIMEOUT, client.peer().list_all_tools()).await {
        Ok(Ok(tools)) => tools,
        Ok(Err(error)) => {
            server_cancel.cancel();
            let _ = client.close_with_timeout(MCP_SHUTDOWN_TIMEOUT).await;
            abort_stderr_task(stderr_task).await;
            return Err(McpError::operation(&server.name, "tools/list", error));
        }
        Err(_) => {
            server_cancel.cancel();
            let _ = client.close_with_timeout(MCP_SHUTDOWN_TIMEOUT).await;
            abort_stderr_task(stderr_task).await;
            return Err(McpError::operation(
                &server.name,
                "tools/list",
                "request timed out",
            ));
        }
    };
    tracing::debug!(server = %server.name, tools = tools.len(), "connected MCP server");
    Ok(ConnectedServer {
        name: server.name.clone(),
        client,
        tools,
        stderr_task,
    })
}

fn spawn_stderr_reader(name: String, mut stderr: tokio::process::ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut buffer = [0u8; MCP_STDERR_CHUNK_BYTES];
        let mut bytes = 0usize;
        while let Ok(read) = stderr.read(&mut buffer).await {
            if read == 0 {
                break;
            }
            bytes = bytes.saturating_add(read);
        }
        if bytes > 0 {
            tracing::debug!(server = %name, bytes, "MCP server emitted stderr");
        }
    })
}

async fn abort_stderr_task(task: Option<JoinHandle<()>>) {
    if let Some(task) = task {
        task.abort();
        let _ = task.await;
    }
}
