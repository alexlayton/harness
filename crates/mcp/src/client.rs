use crate::{McpError, root_uri};
use rmcp::ClientHandler;
#[allow(deprecated)]
use rmcp::model::{
    ClientCapabilities, ClientInfo, Implementation, ListRootsResult, Root, RootsCapabilities,
};
use rmcp::service::{NotificationContext, RequestContext, RoleClient};

/// Minimal MCP client handler: only roots are offered. Sampling and elicitation
/// deliberately retain rmcp's safe defaults (unsupported/declined).
pub(crate) struct HarnessClient {
    root_uri: String,
}

impl HarnessClient {
    pub(crate) fn new(workspace_root: &std::path::Path) -> Result<Self, McpError> {
        Ok(Self {
            root_uri: root_uri(workspace_root)?,
        })
    }
}

impl ClientHandler for HarnessClient {
    fn get_info(&self) -> ClientInfo {
        let mut capabilities = ClientCapabilities::default();
        capabilities.roots = Some(RootsCapabilities::default());
        ClientInfo::new(
            capabilities,
            Implementation::new("harness", env!("CARGO_PKG_VERSION")),
        )
    }

    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        tracing::debug!("MCP tools/list_changed received; changes apply on reconnect");
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
