use crate::client::HarnessClient;
use crate::config::{McpServerConfig, McpTransportConfig};
use crate::tool::{MAX_REMOTE_DEFINITION_BYTES, MAX_REMOTE_TOOLS, McpTool, validate_remote_tool};
use crate::{
    MCP_INITIALIZE_TIMEOUT, MCP_LIST_TIMEOUT, MCP_SHUTDOWN_TIMEOUT, MCP_STDERR_CHUNK_BYTES,
    McpError,
};
use futures_util::future::join_all;
use rmcp::ClientLifecycleMode;
use rmcp::model::{PaginatedRequestParams, Tool as RemoteTool};
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
        let startup_cancel = cancel.child_token();
        let mut tasks = tokio::task::JoinSet::new();
        for server in configs {
            if startup_cancel.is_cancelled() {
                tasks.abort_all();
                while tasks.join_next().await.is_some() {}
                return Err(McpError::operation(&server.name, "initialize", "cancelled"));
            }
            let workspace_root = workspace_root.to_path_buf();
            let server_cancel = startup_cancel.clone();
            tasks.spawn(async move {
                connect_server(&server, &workspace_root, server_cancel)
                    .await
                    .map(|connected| (server.name.clone(), connected))
            });
        }

        let mut connected = Self {
            servers: Vec::with_capacity(tasks.len()),
        };
        let mut failure = None;
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok((_, server))) => connected.servers.push(server),
                Ok(Err(error)) => {
                    failure = Some(error);
                    startup_cancel.cancel();
                    tasks.abort_all();
                    while tasks.join_next().await.is_some() {}
                    break;
                }
                Err(error) => {
                    failure = Some(McpError::operation(
                        "<mcp>",
                        "initialize",
                        format!("connection task failed: {error}"),
                    ));
                    startup_cancel.cancel();
                    tasks.abort_all();
                    while tasks.join_next().await.is_some() {}
                    break;
                }
            }
        }
        if let Some(error) = failure {
            connected.shutdown().await;
            return Err(error);
        }
        connected
            .servers
            .sort_by(|left, right| left.name.cmp(&right.name));
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
        // Stop diagnostic readers before moving the services into concurrent
        // close futures. This also prevents a global shutdown timeout from
        // dropping JoinHandles and detaching stderr tasks.
        for server in &mut servers {
            if let Some(task) = server.stderr_task.take() {
                task.abort();
            }
        }
        let closes = servers.into_iter().map(|mut server| async move {
            let _ = server.client.close_with_timeout(MCP_SHUTDOWN_TIMEOUT).await;
        });
        let _ = tokio::time::timeout(MCP_SHUTDOWN_TIMEOUT, join_all(closes)).await;
        // Dropping the services after the global deadline releases their
        // transports; the configured child process uses kill-on-drop below.
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
    let tools = match list_tools_bounded(&client, &server.name).await {
        Ok(tools) => tools,
        Err(error) => {
            server_cancel.cancel();
            let _ = client.close_with_timeout(MCP_SHUTDOWN_TIMEOUT).await;
            abort_stderr_task(stderr_task).await;
            return Err(error);
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

async fn list_tools_bounded(
    client: &RunningService<RoleClient, HarnessClient>,
    server: &str,
) -> Result<Vec<RemoteTool>, McpError> {
    let deadline = tokio::time::Instant::now()
        .checked_add(MCP_LIST_TIMEOUT)
        .ok_or_else(|| McpError::operation(server, "tools/list", "deadline overflow"))?;
    let mut tools = Vec::new();
    let mut total_bytes = 0usize;
    let mut cursor = None;
    loop {
        let page = tokio::time::timeout_at(
            deadline,
            client
                .peer()
                .list_tools(Some(PaginatedRequestParams::default().with_cursor(cursor))),
        )
        .await
        .map_err(|_| McpError::operation(server, "tools/list", "request timed out"))?
        .map_err(|error| McpError::operation(server, "tools/list", error))?;
        for tool in page.tools {
            if tools.len() >= MAX_REMOTE_TOOLS {
                return Err(McpError::Operation {
                    server: server.into(),
                    operation: "tools/list",
                    message: format!("tool catalogue exceeds limit of {MAX_REMOTE_TOOLS} entries"),
                });
            }
            total_bytes = total_bytes.saturating_add(validate_remote_tool(server, &tool)?);
            if total_bytes > MAX_REMOTE_DEFINITION_BYTES {
                return Err(McpError::Operation {
                    server: server.into(),
                    operation: "tools/list",
                    message: format!(
                        "tool catalogue exceeds definition limit of {MAX_REMOTE_DEFINITION_BYTES} bytes"
                    ),
                });
            }
            tools.push(tool);
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            return Ok(tools);
        }
    }
}

async fn abort_stderr_task(task: Option<JoinHandle<()>>) {
    if let Some(task) = task {
        task.abort();
        let _ = task.await;
    }
}
