//! The externally sourced half of `/reload`: MCP servers, codemode, skills and
//! project context. Provider, model and reasoning are applied by
//! `handle_reload_boundary`.

use super::{Agent, McpSlot, commands::ReloadSettings};

impl Agent {
    /// Apply everything in `settings` that is not a model selection and return
    /// one line per noteworthy outcome (failures and reconnect summaries) for
    /// the reload notice. A failure leaves the previous state in place.
    pub(crate) async fn reload_external(&mut self, settings: &ReloadSettings) -> Vec<String> {
        let mut notes = Vec::new();
        self.project_context = settings.project_context.clone();
        if let Err(error) = tools::reload_skills(&mut self.tools, settings.skills.clone(), true) {
            notes.push(format!("skills not reloaded: {error}"));
        }
        if let Err(error) = self
            .reload_mcp(
                &settings.mcp_servers,
                settings.codemode.as_deref(),
                &mut notes,
            )
            .await
        {
            notes.push(self.secret_masker.mask_text(&error));
        }
        notes
    }

    /// Reconnect MCP servers and rebuild codemode when either changed.
    ///
    /// The new servers connect before anything is touched, so a failure to
    /// start leaves the old tools working. Registration is then undone from
    /// the old runtime if it fails part way.
    pub(super) async fn reload_mcp(
        &mut self,
        servers: &[mcp::McpServerConfig],
        codemode: Option<&[String]>,
        notes: &mut Vec<String>,
    ) -> Result<(), String> {
        let mcp_changed = servers != self.mcp_configs.as_slice();
        let codemode_changed = codemode != self.tools.codemode_selection();
        if !mcp_changed && !codemode_changed {
            return Ok(());
        }

        let mut new_runtime = None;
        if mcp_changed && !servers.is_empty() {
            let mut runtime =
                mcp::McpRuntime::connect(servers, self.tools.workspace_root(), self.cancel.clone())
                    .await
                    .map_err(|error| {
                        format!("MCP reload failed, keeping previous servers: {error}")
                    })?;
            runtime.mask_diagnostics(|text| self.secret_masker.mask_text(text));
            for failure in runtime.failures() {
                notes.push(format!(
                    "MCP server `{}` is unavailable: {}",
                    failure.name, failure.message
                ));
            }
            new_runtime = Some(runtime);
        }

        let snapshot = self.tools.disable_codemode();
        if mcp_changed {
            self.tools.remove_mcp_tools();
            if let Some(runtime) = &new_runtime
                && let Err(error) = runtime.register_into(&mut self.tools)
            {
                let message = format!("MCP reload failed, keeping previous servers: {error}");
                self.rollback_mcp(snapshot);
                shutdown_runtime(new_runtime).await;
                return Err(message);
            }
        }
        if let Some(names) = codemode
            && let Err(error) = self.tools.enable_codemode(names)
        {
            let message = format!("codemode reload failed, keeping previous tools: {error}");
            if mcp_changed {
                self.rollback_mcp(snapshot);
                shutdown_runtime(new_runtime).await;
            } else {
                self.tools.restore_codemode(snapshot);
            }
            return Err(message);
        }

        if mcp_changed {
            let old = swap_runtime(&self.mcp, new_runtime);
            shutdown_runtime(old).await;
            let before = std::mem::replace(&mut self.mcp_configs, servers.to_vec());
            notes.insert(0, mcp_summary(&before, servers));
        }
        Ok(())
    }

    /// Put the previous runtime's tools and codemode back after a failed swap.
    fn rollback_mcp(&mut self, snapshot: tools::CodemodeSnapshot) {
        self.tools.remove_mcp_tools();
        if let Ok(guard) = self.mcp.lock()
            && let Some(old) = guard.as_ref()
        {
            let _ = old.register_into(&mut self.tools);
        }
        self.tools.restore_codemode(snapshot);
    }
}

fn swap_runtime(slot: &McpSlot, new: Option<mcp::McpRuntime>) -> Option<mcp::McpRuntime> {
    match slot.lock() {
        Ok(mut guard) => std::mem::replace(&mut *guard, new),
        Err(poisoned) => std::mem::replace(&mut *poisoned.into_inner(), new),
    }
}

async fn shutdown_runtime(runtime: Option<mcp::McpRuntime>) {
    if let Some(runtime) = runtime {
        runtime.shutdown().await;
    }
}

/// `MCP: +docs (cmd), -safari` style line, or a plain reconnect line when only
/// settings of existing servers changed.
///
/// Each added server is shown with what it will run or contact. A reload can
/// follow an edit the user did not make (the model can write a trusted
/// workspace config), so the notice must not hide the command being started.
pub(super) fn mcp_summary(
    before: &[mcp::McpServerConfig],
    after: &[mcp::McpServerConfig],
) -> String {
    let known = |servers: &[mcp::McpServerConfig], name: &str| {
        servers.iter().any(|server| server.name == name)
    };
    let mut changes: Vec<String> = after
        .iter()
        .filter(|server| !known(before, &server.name))
        .map(|server| {
            format!(
                "+{} ({})",
                server.name,
                describe_transport(&server.transport)
            )
        })
        .chain(
            before
                .iter()
                .filter(|server| !known(after, &server.name))
                .map(|server| format!("-{}", server.name)),
        )
        .collect();
    if changes.is_empty() {
        changes.push("settings changed".into());
    }
    format!(
        "MCP servers reconnected ({} running): {}",
        after.len(),
        changes.join(", ")
    )
}

/// What a server will launch or connect to, without environment values or
/// header values. Only the command line and URL are shown, and the URL goes
/// through `mcp::redact_url`, which drops credentials, query and fragment.
fn describe_transport(transport: &mcp::McpTransportConfig) -> String {
    match transport {
        mcp::McpTransportConfig::Stdio { command, args, .. } => {
            let mut parts = vec![command.display().to_string()];
            parts.extend(args.iter().cloned());
            format!("stdio: {}", parts.join(" "))
        }
        mcp::McpTransportConfig::Http { url, .. } => format!("http: {}", mcp::redact_url(url)),
    }
}
