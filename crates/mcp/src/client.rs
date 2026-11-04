use crate::{McpError, root_uri};
use rmcp::ClientHandler;
use rmcp::model::{ClientCapabilities, ClientConfig, Implementation, RootsCapabilities};
// Roots are still advertised for compatibility with existing MCP servers.
#[allow(deprecated)]
use rmcp::model::{ListRootsResult, Root};
use rmcp::service::{NotificationContext, RequestContext, RoleClient};

/// Lifecycle state of one connected MCP server after startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerStatus {
    /// The transport is open and tool calls are forwarded.
    Ready,
    /// The transport died unexpectedly (crash, EOF, protocol failure).
    Failed,
    /// The runtime closed the server deliberately.
    Closed,
}

/// Shared, cheaply readable health of one server. Written by the client
/// handler, tool adapters, and the runtime; read by tool calls so a dead
/// server fails fast with a clear error.
#[derive(Debug)]
pub(crate) struct ServerHealth {
    status: std::sync::atomic::AtomicU8,
    tools_stale: std::sync::atomic::AtomicBool,
}

impl Default for ServerHealth {
    fn default() -> Self {
        Self {
            status: std::sync::atomic::AtomicU8::new(0),
            tools_stale: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl ServerHealth {
    pub(crate) fn status(&self) -> ServerStatus {
        match self.status.load(std::sync::atomic::Ordering::Acquire) {
            0 => ServerStatus::Ready,
            1 => ServerStatus::Failed,
            _ => ServerStatus::Closed,
        }
    }

    /// Record `Failed`; a deliberate `Closed` is never downgraded.
    pub(crate) fn mark_failed(&self) {
        let _ = self.status.compare_exchange(
            0,
            1,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        );
    }

    pub(crate) fn mark_closed(&self) {
        self.status.store(2, std::sync::atomic::Ordering::Release);
    }

    /// True once the server announced `tools/list_changed`. The registered
    /// tool set is not refreshed (the registry is fixed per session), so this
    /// only records that the catalogue may differ from the server's.
    pub(crate) fn tools_stale(&self) -> bool {
        self.tools_stale.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn mark_tools_stale(&self) {
        self.tools_stale
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Minimal MCP client handler: only roots are offered. Sampling and elicitation
/// deliberately retain rmcp's safe defaults (unsupported/declined).
pub(crate) struct HarnessClient {
    root_uri: String,
    health: std::sync::Arc<ServerHealth>,
}

impl HarnessClient {
    pub(crate) fn new(
        workspace_root: &std::path::Path,
        health: std::sync::Arc<ServerHealth>,
    ) -> Result<Self, McpError> {
        Ok(Self {
            root_uri: root_uri(workspace_root)?,
            health,
        })
    }
}

impl ClientHandler for HarnessClient {
    fn get_info(&self) -> ClientConfig {
        let mut capabilities = ClientCapabilities::default();
        capabilities.roots = Some(RootsCapabilities::default());
        ClientConfig::new(
            capabilities,
            Implementation::new("harness", env!("CARGO_PKG_VERSION")),
        )
    }

    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        self.health.mark_tools_stale();
        tracing::debug!("MCP tools/list_changed received; catalogue marked stale until reconnect");
        std::future::ready(())
    }

    #[allow(deprecated)]
    fn list_roots(
        &self,
        _context: RequestContext<RoleClient>,
    ) -> impl std::future::Future<Output = Result<ListRootsResult, rmcp::ErrorData>> + Send + '_
    {
        std::future::ready(Ok(ListRootsResult::new(vec![
            Root::new(self.root_uri.clone()).with_name("workspace"),
        ])))
    }
}
