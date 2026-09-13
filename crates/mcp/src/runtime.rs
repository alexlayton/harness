use crate::client::HarnessClient;
use crate::config::{McpServerConfig, McpTransportConfig};
use crate::tool::{MAX_REMOTE_DEFINITION_BYTES, MAX_REMOTE_TOOLS, McpTool, validate_remote_tool};
use crate::{
    MCP_INITIALIZE_TIMEOUT, MCP_LIST_TIMEOUT, MCP_MAX_FRAME_BYTES, MCP_SHUTDOWN_TIMEOUT,
    MCP_STDERR_CHUNK_BYTES, McpError,
};
use futures_util::future::join_all;
use rmcp::ClientLifecycleMode;
use rmcp::model::{
    ErrorData, JsonRpcMessage, PaginatedRequestParams, ProtocolVersion, RequestId,
    Tool as RemoteTool,
};
use rmcp::service::{
    RoleClient, RunningService, RxJsonRpcMessage, TxJsonRpcMessage,
    serve_client_with_lifecycle_and_ct,
};
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, Transport, TransportAdapterIdentity};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, ReadBuf};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tools::ToolRegistry;

/// A bounded newline-delimited reader for MCP stdio.
///
/// rmcp's normal async reader uses a growable line buffer before it invokes
/// serde. Keeping this guard below rmcp's `BufReader` means an overlong frame
/// fails while bytes are still in the transport, while complete valid frames
/// and arbitrary read boundaries retain the normal MCP framing semantics.
struct BoundedFrameReader<R> {
    inner: R,
    max_length: usize,
    frame_length: usize,
    pending: Vec<u8>,
    pending_start: usize,
    failed: bool,
    failure_signal: Arc<AtomicBool>,
    eof: bool,
}

impl<R> BoundedFrameReader<R> {
    #[cfg(test)]
    fn new(inner: R, max_length: usize) -> Self {
        Self::new_with_failure(inner, max_length, Arc::new(AtomicBool::new(false)))
    }

    fn new_with_failure(inner: R, max_length: usize, failure_signal: Arc<AtomicBool>) -> Self {
        Self {
            inner,
            max_length,
            frame_length: 0,
            pending: Vec::new(),
            pending_start: 0,
            failed: false,
            failure_signal,
            eof: false,
        }
    }

    fn has_pending(&self) -> bool {
        self.pending_start < self.pending.len()
    }

    fn copy_pending(&mut self, buffer: &mut ReadBuf<'_>) {
        let available = &self.pending[self.pending_start..];
        let length = available.len().min(buffer.remaining());
        buffer.put_slice(&available[..length]);
        self.pending_start += length;
        if self.pending_start == self.pending.len() {
            self.pending.clear();
            self.pending_start = 0;
        }
    }

    fn process_chunk(&mut self, chunk: &[u8]) {
        for &byte in chunk {
            if byte == b'\n' {
                self.frame_length = 0;
                self.pending.push(byte);
            } else if self.frame_length >= self.max_length {
                // Do not retain the offending byte or any bytes after it. The
                // already pending prefix is delivered first, then the next
                // read reports the terminal framing error.
                self.failed = true;
                self.failure_signal.store(true, Ordering::Release);
                break;
            } else {
                self.frame_length += 1;
                self.pending.push(byte);
            }
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for BoundedFrameReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        if buffer.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        if self.has_pending() {
            self.copy_pending(buffer);
            return Poll::Ready(Ok(()));
        }
        if self.failed {
            return Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("MCP frame exceeds {} bytes", self.max_length),
            )));
        }
        if self.eof {
            return Poll::Ready(Ok(()));
        }

        // A fixed temporary chunk is important: processing a read must not
        // allocate in proportion to an untrusted frame's eventual length.
        let mut chunk = [0u8; 4096];
        let mut chunk_buffer = ReadBuf::new(&mut chunk);
        match Pin::new(&mut self.inner).poll_read(cx, &mut chunk_buffer) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => {
                self.failed = true;
                self.failure_signal.store(true, Ordering::Release);
                Poll::Ready(Err(error))
            }
            Poll::Ready(Ok(())) => {
                let filled = chunk_buffer.filled();
                if filled.is_empty() {
                    self.eof = true;
                    return Poll::Ready(Ok(()));
                }
                self.process_chunk(filled);
                if self.has_pending() {
                    self.copy_pending(buffer);
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Ready(Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("MCP frame exceeds {} bytes", self.max_length),
                    )))
                }
            }
        }
    }
}

type BoundedMcpTransport =
    AsyncRwTransport<RoleClient, BoundedFrameReader<ChildStdout>, ChildStdin>;

/// Stdio transport using rmcp's protocol machinery with a pre-deserialization
/// frame guard. The child remains owned here so close and drop retain the
/// lifecycle guarantees of rmcp's regular child-process transport.
struct BoundedStdioTransport {
    child: Option<Child>,
    transport: BoundedMcpTransport,
    pending_requests: Arc<StdMutex<VecDeque<RequestId>>>,
    frame_failure: Arc<AtomicBool>,
}

impl BoundedStdioTransport {
    fn new(child: Child, stdout: ChildStdout, stdin: ChildStdin) -> Self {
        let frame_failure = Arc::new(AtomicBool::new(false));
        Self {
            child: Some(child),
            transport: AsyncRwTransport::new(
                BoundedFrameReader::new_with_failure(
                    stdout,
                    MCP_MAX_FRAME_BYTES,
                    frame_failure.clone(),
                ),
                stdin,
            ),
            pending_requests: Arc::new(StdMutex::new(VecDeque::new())),
            frame_failure,
        }
    }

    fn forget_response(&self, message: &RxJsonRpcMessage<RoleClient>) {
        let id = match message {
            JsonRpcMessage::Response(response) => Some(&response.id),
            JsonRpcMessage::Error(error) => error.id.as_ref(),
            _ => None,
        };
        let Some(id) = id else { return };
        let mut pending = self
            .pending_requests
            .lock()
            .expect("pending MCP requests lock");
        if let Some(index) = pending.iter().position(|pending_id| pending_id == id) {
            pending.remove(index);
        }
    }

    fn frame_failure_response(&self) -> Option<RxJsonRpcMessage<RoleClient>> {
        if !self.frame_failure.load(Ordering::Acquire) {
            return None;
        }
        let id = self
            .pending_requests
            .lock()
            .expect("pending MCP requests lock")
            .pop_front()?;
        Some(JsonRpcMessage::error(
            ErrorData::invalid_request(
                format!("MCP frame exceeds {MCP_MAX_FRAME_BYTES} bytes"),
                None,
            ),
            Some(id),
        ))
    }
}

impl Transport<RoleClient> for BoundedStdioTransport {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let request_id = match &item {
            JsonRpcMessage::Request(request) => Some(request.id.clone()),
            _ => None,
        };
        if let Some(id) = &request_id {
            self.pending_requests
                .lock()
                .expect("pending MCP requests lock")
                .push_back(id.clone());
        }
        let pending_requests = self.pending_requests.clone();
        let send = self.transport.send(item);
        async move {
            let result = send.await;
            if result.is_err()
                && let Some(id) = request_id
            {
                let mut pending = pending_requests.lock().expect("pending MCP requests lock");
                if let Some(index) = pending.iter().position(|pending_id| pending_id == &id) {
                    pending.remove(index);
                }
            }
            result
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        if let Some(message) = self.transport.receive().await {
            self.forget_response(&message);
            Some(message)
        } else {
            self.frame_failure_response()
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.transport.close().await?;
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        match tokio::time::timeout(Duration::from_secs(3), child.wait()).await {
            Ok(result) => result.map(|_| ()),
            Err(_) => {
                child.start_kill()?;
                child.wait().await.map(|_| ())
            }
        }
    }
}

impl Drop for BoundedStdioTransport {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _ = child.start_kill();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = child.wait().await;
            });
        }
    }
}

/// Connected MCP servers for one agent/session. The runtime owns protocol and
/// child-process lifetimes; tools only retain cloneable request handles.
pub struct McpRuntime {
    servers: Vec<ConnectedServer>,
}

impl Drop for McpRuntime {
    fn drop(&mut self) {
        // Async shutdown is preferred, but assembly cancellation can drop a
        // partially built runtime. Abort diagnostic readers here; the bounded
        // stdio transport owns kill-on-drop child cleanup.
        for server in &mut self.servers {
            if let Some(task) = server.stderr_task.take() {
                task.abort();
            }
        }
    }
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
        loop {
            tokio::select! {
                _ = startup_cancel.cancelled() => {
                    failure = Some(McpError::operation("<mcp>", "initialize", "cancelled"));
                    tasks.abort_all();
                    while tasks.join_next().await.is_some() {}
                    break;
                }
                result = tasks.join_next() => {
                    let Some(result) = result else { break };
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
    match &server.transport {
        McpTransportConfig::Stdio { command, args, env } => {
            let mut command = tokio::process::Command::new(command);
            command
                .kill_on_drop(true)
                .args(args)
                .current_dir(workspace_root)
                .envs(env)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut child = command
                .spawn()
                .map_err(|error| McpError::operation(&server.name, "initialize", error))?;
            let stdin = child.stdin.take().ok_or_else(|| {
                McpError::operation(&server.name, "initialize", "MCP child stdin was not piped")
            })?;
            let stdout = child.stdout.take().ok_or_else(|| {
                McpError::operation(&server.name, "initialize", "MCP child stdout was not piped")
            })?;
            let stderr = child.stderr.take();
            let transport = BoundedStdioTransport::new(child, stdout, stdin);
            let stderr_task = stderr.map(|stderr| spawn_stderr_reader(server.name.clone(), stderr));
            connect_transport(
                server,
                workspace_root,
                cancel,
                transport,
                stderr_task,
                ClientLifecycleMode::Initialize,
            )
            .await
        }
        McpTransportConfig::Http { url, headers } => {
            let headers = headers
                .iter()
                .map(|(name, value)| {
                    let name = http::HeaderName::try_from(name).map_err(|_| {
                        McpError::operation(&server.name, "initialize", "invalid HTTP header name")
                    })?;
                    let value = http::HeaderValue::try_from(value).map_err(|_| {
                        McpError::operation(&server.name, "initialize", "invalid HTTP header value")
                    })?;
                    Ok((name, value))
                })
                .collect::<Result<HashMap<_, _>, McpError>>()?;
            let config = StreamableHttpClientTransportConfig::with_uri(url.clone())
                .custom_headers(headers)
                .max_sse_event_size(MCP_MAX_FRAME_BYTES);
            let transport = StreamableHttpClientTransport::from_config(config);
            connect_transport(
                server,
                workspace_root,
                cancel,
                transport,
                None,
                ClientLifecycleMode::Auto {
                    preferred_versions: vec![ProtocolVersion::V_2026_07_28],
                    legacy_version: Some(ProtocolVersion::V_2025_11_25),
                },
            )
            .await
        }
    }
}

async fn connect_transport<T>(
    server: &McpServerConfig,
    workspace_root: &Path,
    cancel: CancellationToken,
    transport: T,
    stderr_task: Option<JoinHandle<()>>,
    lifecycle: ClientLifecycleMode,
) -> Result<ConnectedServer, McpError>
where
    T: Transport<RoleClient> + 'static,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    let handler = match HarnessClient::new(workspace_root) {
        Ok(handler) => handler,
        Err(error) => {
            abort_stderr_task(stderr_task).await;
            return Err(error);
        }
    };
    let server_cancel = cancel.child_token();
    let initialize = tokio::time::timeout(
        MCP_INITIALIZE_TIMEOUT,
        serve_client_with_lifecycle_and_ct::<_, _, _, TransportAdapterIdentity>(
            handler,
            transport,
            lifecycle,
            server_cancel.clone(),
        ),
    );
    tokio::pin!(initialize);
    let mut client = tokio::select! {
        _ = server_cancel.cancelled() => {
            server_cancel.cancel();
            abort_stderr_task(stderr_task).await;
            return Err(McpError::operation(&server.name, "initialize", "cancelled"));
        }
        result = &mut initialize => match result {
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
        }
    };
    let tools = match list_tools_bounded(&client, &server.name, &server_cancel).await {
        Ok(tools) => tools,
        Err(error) => {
            server_cancel.cancel();
            if !cancel.is_cancelled() {
                let _ = client.close_with_timeout(MCP_SHUTDOWN_TIMEOUT).await;
            }
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
    cancel: &CancellationToken,
) -> Result<Vec<RemoteTool>, McpError> {
    let deadline = tokio::time::Instant::now()
        .checked_add(MCP_LIST_TIMEOUT)
        .ok_or_else(|| McpError::operation(server, "tools/list", "deadline overflow"))?;
    let mut tools = Vec::new();
    let mut total_bytes = 0usize;
    let mut cursor = None;
    loop {
        let page = tokio::select! {
            _ = cancel.cancelled() => {
                return Err(McpError::operation(server, "tools/list", "cancelled"));
            }
            result = tokio::time::timeout_at(
                deadline,
                client
                    .peer()
                    .list_tools(Some(PaginatedRequestParams::default().with_cursor(cursor))),
            ) => result
                .map_err(|_| McpError::operation(server, "tools/list", "request timed out"))?
                .map_err(|error| McpError::operation(server, "tools/list", error))?,
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    struct ChunkedReader {
        data: Vec<u8>,
        position: usize,
        chunk_size: usize,
    }

    impl ChunkedReader {
        fn new(data: impl Into<Vec<u8>>, chunk_size: usize) -> Self {
            assert!(chunk_size > 0);
            Self {
                data: data.into(),
                position: 0,
                chunk_size,
            }
        }
    }

    impl AsyncRead for ChunkedReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buffer: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            if self.position == self.data.len() {
                return Poll::Ready(Ok(()));
            }
            let length = (self.data.len() - self.position)
                .min(self.chunk_size)
                .min(buffer.remaining());
            buffer.put_slice(&self.data[self.position..self.position + length]);
            self.position += length;
            Poll::Ready(Ok(()))
        }
    }

    async fn read_guarded<R: AsyncRead + Unpin>(reader: R) -> (Vec<u8>, Option<io::Error>) {
        let mut reader = reader;
        let mut output = Vec::new();
        let mut buffer = [0u8; 32];
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => return (output, None),
                Ok(length) => output.extend_from_slice(&buffer[..length]),
                Err(error) => return (output, Some(error)),
            }
        }
    }

    #[tokio::test]
    async fn frame_limit_allows_exact_boundary_and_resets_after_newline() {
        let reader =
            BoundedFrameReader::new(ChunkedReader::new(b"12345678\nok\r\n".to_vec(), 1), 8);
        let (output, error) = read_guarded(reader).await;
        assert_eq!(output, b"12345678\nok\r\n");
        assert!(error.is_none());
    }

    #[tokio::test]
    async fn oversized_single_chunk_fails_before_the_delimiter() {
        let reader = BoundedFrameReader::new(ChunkedReader::new(b"123456789\n", 64), 8);
        let (output, error) = read_guarded(reader).await;
        assert_eq!(output, b"12345678");
        let error = error.expect("frame must be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn oversized_frame_split_across_chunks_fails_without_growing_pending_data() {
        let reader = BoundedFrameReader::new(ChunkedReader::new(b"123456789\n", 1), 8);
        let (output, error) = read_guarded(reader).await;
        assert_eq!(output, b"12345678");
        let error = error.expect("chunked frame must be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn valid_frame_before_oversized_frame_is_preserved() {
        let reader = BoundedFrameReader::new(ChunkedReader::new(b"ok\n123456789\n", 3), 8);
        let (output, error) = read_guarded(reader).await;
        assert_eq!(output, b"ok\n12345678");
        assert!(error.is_some());
    }

    #[derive(Clone, Default)]
    struct HttpFixture;

    impl rmcp::ServerHandler for HttpFixture {}

    async fn connect_http_fixture(json_response: bool) {
        use rmcp::transport::streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        };

        let server_cancel = CancellationToken::new();
        let service: StreamableHttpService<HttpFixture, LocalSessionManager> =
            StreamableHttpService::new(
                || Ok(HttpFixture),
                Default::default(),
                StreamableHttpServerConfig::default()
                    .with_legacy_session_mode(false)
                    .with_json_response(json_response)
                    .with_cancellation_token(server_cancel.child_token()),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let saw_header = Arc::new(AtomicBool::new(false));
        let header_probe = axum::middleware::from_fn({
            let saw_header = saw_header.clone();
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let saw_header = saw_header.clone();
                async move {
                    if request
                        .headers()
                        .get("x-harness-test")
                        .is_some_and(|value| value == "present")
                    {
                        saw_header.store(true, Ordering::Release);
                    }
                    next.run(request).await
                }
            }
        });
        let server_task = tokio::spawn({
            let server_cancel = server_cancel.clone();
            async move {
                let router = axum::Router::new()
                    .nest_service("/mcp", service)
                    .layer(header_probe);
                axum::serve(listener, router)
                    .with_graceful_shutdown(async move { server_cancel.cancelled_owned().await })
                    .await
                    .unwrap();
            }
        });
        let config = McpServerConfig {
            name: "http-fixture".into(),
            transport: McpTransportConfig::Http {
                url: format!("http://{address}/mcp"),
                headers: [("X-Harness-Test".into(), "present".into())]
                    .into_iter()
                    .collect(),
            },
        };

        let runtime = McpRuntime::connect(&[config], Path::new("/"), CancellationToken::new())
            .await
            .expect("HTTP MCP server should connect");
        assert_eq!(runtime.servers.len(), 1);
        assert!(saw_header.load(Ordering::Acquire));
        runtime.shutdown().await;
        server_cancel.cancel();
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn streamable_http_accepts_json_responses() {
        connect_http_fixture(true).await;
    }

    #[tokio::test]
    async fn streamable_http_accepts_sse_responses() {
        connect_http_fixture(false).await;
    }

    #[cfg(unix)]
    mod stdio_server_tests {
        use super::*;
        use std::collections::BTreeMap;

        // The fixture deliberately writes both one large printf and many small
        // writes. The production reader must enforce one frame budget in both
        // cases, rather than relying on OS pipe read boundaries.
        const FIXTURE: &str = r#"
set -eu
chunk=$(printf '%01000d' 0 | tr '0' x)
single_huge() {
    prefix=$1
    suffix=$2
    payload=
    i=0
    while [ "$i" -lt 1200 ]; do
        payload=$payload$chunk
        i=$((i + 1))
    done
    printf '%s%s%s\n' "$prefix" "$payload" "$suffix"
}
chunked_huge() {
    prefix=$1
    suffix=$2
    printf '%s' "$prefix"
    i=0
    while [ "$i" -lt 1200 ]; do
        printf '%s' "$chunk"
        i=$((i + 1))
    done
    printf '%s\n' "$suffix"
}
while IFS= read -r line; do
    id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    [ -n "$id" ] || continue
    case "$line" in
        *'"method":"initialize"'*)
            case "$MCP_TEST_MODE" in
                initialize-single)
                    single_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"},\"instructions\":\"" "\"}}"
                    ;;
                initialize-chunked)
                    chunked_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"},\"instructions\":\"" "\"}}"
                    ;;
                *)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}}}\n' "$id"
                    ;;
            esac
            ;;
        *'"method":"tools/list"'*)
            case "$MCP_TEST_MODE" in
                list-single)
                    single_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[{\"name\":\"fixture\",\"description\":\"" "\",\"inputSchema\":{\"type\":\"object\"}}]}}"
                    ;;
                list-chunked)
                    chunked_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[{\"name\":\"fixture\",\"description\":\"" "\",\"inputSchema\":{\"type\":\"object\"}}]}}"
                    ;;
                paged)
                    case "$line" in
                        *'"cursor":"page-2"'*)
                            printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"second","inputSchema":{"type":"object"}}]}}\n' "$id"
                            ;;
                        *)
                            printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"first","inputSchema":{"type":"object"}}],"nextCursor":"page-2"}}\n' "$id"
                            ;;
                    esac
                    ;;
                *)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"fixture","inputSchema":{"type":"object"}}]}}\n' "$id"
                    ;;
            esac
            ;;
        *'"method":"tools/call"'*)
            case "$MCP_TEST_MODE" in
                call-text-single)
                    single_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"" "\"}]}}"
                    ;;
                call-structured-chunked)
                    chunked_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"structuredContent\":{\"payload\":\"" "\"}}}"
                    ;;
                call-image-chunked)
                    chunked_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"image\",\"data\":\"" "\",\"mimeType\":\"image/png\"}]}}"
                    ;;
                call-binary-chunked)
                    chunked_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"resource\",\"resource\":{\"uri\":\"file:///fixture\",\"mimeType\":\"application/octet-stream\",\"blob\":\"" "\"}}]}}"
                    ;;
                call-error-chunked)
                    chunked_huge "{\"jsonrpc\":\"2.0\",\"id\":$id,\"error\":{\"code\":-32000,\"message\":\"" "\"}}"
                    ;;
                *)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"ok"}]}}\n' "$id"
                    ;;
            esac
            ;;
    esac
done
"#;

        fn fixture_server(mode: &str) -> McpServerConfig {
            McpServerConfig {
                name: "fixture".into(),
                transport: McpTransportConfig::Stdio {
                    command: "/bin/sh".into(),
                    args: vec!["-c".into(), FIXTURE.into()],
                    env: BTreeMap::from([(String::from("MCP_TEST_MODE"), mode.into())]),
                },
            }
        }

        async fn connect_fixture(mode: &str) -> Result<McpRuntime, McpError> {
            McpRuntime::connect(
                &[fixture_server(mode)],
                &std::env::current_dir().expect("test workspace should exist"),
                CancellationToken::new(),
            )
            .await
        }

        fn assert_prompt_error(result: Result<McpRuntime, McpError>, operation: &str) {
            let error = match result {
                Ok(_) => panic!("oversized frame must fail"),
                Err(error) => error,
            };
            let rendered = error.to_string();
            assert!(rendered.contains(operation), "{rendered}");
            assert!(rendered.len() <= crate::output::MAX_OUTPUT_BYTES);
        }

        #[tokio::test]
        async fn oversized_initialize_frames_fail_promptly() {
            // 10s, matching the catalogue sibling: the assertion is that
            // frame rejection short-circuits the 15s initialize deadline,
            // not that fixtures spawn in 3s. macOS runners are too slow
            // for the tighter bound and tripped it spuriously.
            for mode in ["initialize-single", "initialize-chunked"] {
                let result = tokio::time::timeout(Duration::from_secs(10), connect_fixture(mode))
                    .await
                    .expect("frame rejection must not wait for initialize timeout");
                assert_prompt_error(result, "initialize");
            }
        }

        #[tokio::test]
        async fn oversized_catalogue_frame_fails_before_tools_are_materialized() {
            for mode in ["list-single", "list-chunked"] {
                let result = tokio::time::timeout(Duration::from_secs(10), connect_fixture(mode))
                    .await
                    .expect("frame rejection must not wait for list timeout");
                assert_prompt_error(result, "tools/list");
            }
        }

        #[tokio::test]
        async fn paginated_catalogues_remain_compatible_with_the_frame_guard() {
            let runtime = connect_fixture("paged")
                .await
                .expect("fixture should connect");
            let mut registry = ToolRegistry::empty();
            runtime
                .register_into(&mut registry)
                .expect("fixture tools should register");
            let names: Vec<_> = registry
                .definitions()
                .into_iter()
                .map(|definition| definition.name)
                .collect();
            assert_eq!(names.len(), 2);
            assert_eq!(names[0], crate::normalized_tool_name("fixture", "first"));
            assert_eq!(names[1], crate::normalized_tool_name("fixture", "second"));
            runtime.shutdown().await;
        }

        #[tokio::test]
        async fn oversized_call_payloads_fail_with_bounded_tool_errors() {
            for mode in [
                "call-text-single",
                "call-structured-chunked",
                "call-image-chunked",
                "call-binary-chunked",
                "call-error-chunked",
            ] {
                let runtime = connect_fixture(mode).await.expect("fixture should connect");
                let mut registry = ToolRegistry::empty();
                runtime
                    .register_into(&mut registry)
                    .expect("fixture tool should register");
                // 15s: same slow-fixture-spawn allowance as the
                // initialize/catalogue siblings. The assertion is that the
                // oversized frame fails as a bounded tool error, not that
                // the fixture round-trips in 5s on macOS runners.
                let output = tokio::time::timeout(
                    Duration::from_secs(15),
                    registry.execute(
                        &crate::normalized_tool_name("fixture", "fixture"),
                        serde_json::json!({}),
                        CancellationToken::new(),
                    ),
                )
                .await
                .expect("oversized call frame must fail promptly");
                assert!(output.is_error, "{mode} unexpectedly succeeded");
                assert!(output.content.len() <= crate::output::MAX_OUTPUT_BYTES);
                runtime.shutdown().await;
            }
        }
    }
}
