//! Launch-time MCP server selection layered over the MCP crate's settings.
//!
//! The MCP crate describes how to reach a server; whether a launch connects it
//! is a Harness policy. Servers marked `enabled = false` stay out of every
//! process unless it names them with `--mcp`, so sensitive servers (email,
//! for example) are opt-in per session instead of present everywhere.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// The `[mcp]` section as Harness stores it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileMcpConfig {
    /// Configured servers, including opt-in ones.
    #[serde(default)]
    pub servers: Vec<FileMcpServer>,
}

/// One `[[mcp.servers]]` entry plus its launch policy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileMcpServer {
    /// Connection settings understood by the MCP runtime.
    #[serde(flatten)]
    pub config: mcp::McpServerConfig,
    /// `false` keeps the server out of launches that do not request it with
    /// `--mcp <NAME>`. Omitted from disk when `true` so existing files and
    /// older Harness versions see no change.
    #[serde(default = "enabled_by_default", skip_serializing_if = "is_enabled")]
    pub enabled: bool,
}

fn enabled_by_default() -> bool {
    true
}

fn is_enabled(enabled: &bool) -> bool {
    *enabled
}

impl FileMcpConfig {
    /// Every configured server, regardless of launch policy.
    pub fn runtime(&self) -> mcp::McpConfig {
        mcp::McpConfig {
            servers: self
                .servers
                .iter()
                .map(|server| server.config.clone())
                .collect(),
        }
    }

    /// Validate every server, opt-in ones included, so a typo in a rarely
    /// used entry surfaces on the next launch rather than when requested.
    pub fn validate(&self) -> Result<()> {
        self.runtime().validate().map_err(|error| anyhow!(error))
    }

    /// Servers this launch connects: enabled ones plus those named in
    /// `requested`. Selection happens before `${ENV_VAR}` expansion, so an
    /// opt-in server's credentials are only required when it is requested.
    pub fn select(&self, requested: &[String]) -> Result<mcp::McpConfig> {
        if let Some(unknown) = requested.iter().find(|name| {
            !self
                .servers
                .iter()
                .any(|server| &server.config.name == *name)
        }) {
            return Err(anyhow!(
                "unknown MCP server `{unknown}`; run `harness mcp` to list configured servers"
            ));
        }
        Ok(mcp::McpConfig {
            servers: self
                .servers
                .iter()
                .filter(|server| server.enabled || requested.contains(&server.config.name))
                .map(|server| server.config.clone())
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "fs-server"

[[mcp.servers]]
name = "email"
transport = "http"
url = "https://mail.example.test/mcp"
enabled = false

[mcp.servers.headers]
Authorization = "Bearer ${HARNESS_TEST_UNSET_EMAIL_TOKEN}"
"#;

    #[derive(Deserialize)]
    struct Document {
        mcp: FileMcpConfig,
    }

    fn names(config: &mcp::McpConfig) -> Vec<&str> {
        config
            .servers
            .iter()
            .map(|server| server.name.as_str())
            .collect()
    }

    #[test]
    fn opt_in_servers_connect_only_when_requested() {
        let file: FileMcpConfig = toml::from_str::<Document>(CONFIG).unwrap().mcp;
        file.validate().unwrap();
        assert!(!file.servers[1].enabled);

        let default = file.select(&[]).unwrap();
        assert_eq!(names(&default), ["filesystem"]);
        // The opt-in server's missing credential must not block launches
        // that do not use it.
        assert!(default.resolve_with(|_| None).is_ok());

        let requested = file.select(&["email".into()]).unwrap();
        assert_eq!(names(&requested), ["filesystem", "email"]);
        assert!(requested.resolve_with(|_| None).is_err());

        let error = file.select(&["mail".into()]).unwrap_err().to_string();
        assert!(error.contains("unknown MCP server `mail`"), "{error}");
    }
}
