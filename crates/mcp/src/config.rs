use crate::McpError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

/// The `[mcp]` section of Harness configuration.
#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpConfig {
    /// Servers connected when a Harness agent is assembled.
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

impl fmt::Debug for McpConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpConfig")
            .field("servers", &self.servers)
            .finish()
    }
}

/// One named MCP server.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerConfig {
    /// Stable local name used for diagnostics and tool namespacing.
    pub name: String,
    /// Transport and its non-secret connection settings.
    #[serde(flatten)]
    pub transport: McpTransportConfig,
}

impl fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("transport", &self.transport)
            .finish()
    }
}

/// Supported MCP transport configuration. HTTP is represented so configuration
/// can be validated and preserved, but is not connected until its transport is
/// enabled in a later release.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "lowercase")]
pub enum McpTransportConfig {
    /// Launch an MCP server directly, with no shell involved.
    Stdio {
        command: PathBuf,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    /// Streamable HTTP endpoint (not enabled by this MVP runtime).
    Http {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
}

impl fmt::Debug for McpTransportConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdio { command, args, env } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("args", args)
                .field("env", &format_args!("<{} entries redacted>", env.len()))
                .finish(),
            Self::Http { url, headers } => f
                .debug_struct("Http")
                .field("url", &redact_url_userinfo(url))
                .field(
                    "headers",
                    &format_args!("<{} entries redacted>", headers.len()),
                )
                .finish(),
        }
    }
}

impl McpConfig {
    /// Validate configuration without expanding environment placeholders.
    pub fn validate(&self) -> Result<(), McpError> {
        let mut names = BTreeSet::new();
        for server in &self.servers {
            validate_server(server, &mut names)?;
        }
        Ok(())
    }

    /// Validate configuration and expand environment placeholders in values.
    pub fn resolve_with(
        &self,
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Vec<McpServerConfig>, McpError> {
        let mut names = BTreeSet::new();
        let mut servers = self.servers.clone();
        servers.sort_by(|left, right| left.name.cmp(&right.name));
        for server in &mut servers {
            validate_server(server, &mut names)?;
            match &mut server.transport {
                McpTransportConfig::Stdio { args, env, .. } => {
                    for arg in args {
                        *arg = expand_environment(arg, &mut lookup).map_err(|message| {
                            McpError::Config {
                                server: server.name.clone(),
                                message,
                            }
                        })?;
                    }
                    for value in env.values_mut() {
                        *value = expand_environment(value, &mut lookup).map_err(|message| {
                            McpError::Config {
                                server: server.name.clone(),
                                message,
                            }
                        })?;
                    }
                }
                McpTransportConfig::Http { url, headers } => {
                    *url = expand_environment(url, &mut lookup).map_err(|message| {
                        McpError::Config {
                            server: server.name.clone(),
                            message,
                        }
                    })?;
                    for value in headers.values_mut() {
                        *value = expand_environment(value, &mut lookup).map_err(|message| {
                            McpError::Config {
                                server: server.name.clone(),
                                message,
                            }
                        })?;
                    }
                    if url::Url::parse(url).is_err() {
                        return Err(McpError::Config {
                            server: server.name.clone(),
                            message: "HTTP URL is invalid".into(),
                        });
                    }
                }
            }
        }
        Ok(servers)
    }
}

fn validate_server(server: &McpServerConfig, names: &mut BTreeSet<String>) -> Result<(), McpError> {
    let invalid = |message: &str| McpError::Config {
        server: server.name.clone(),
        message: message.to_owned(),
    };
    if server.name.trim().is_empty() || server.name.contains('\0') {
        return Err(invalid("name must be non-empty and contain no NUL"));
    }
    if !names.insert(server.name.clone()) {
        return Err(invalid("server names must be unique"));
    }
    match &server.transport {
        McpTransportConfig::Stdio { command, args, env } => {
            if command.as_os_str().is_empty() || command.to_string_lossy().contains('\0') {
                return Err(invalid("command must be non-empty and contain no NUL"));
            }
            if args.iter().any(|arg| arg.contains('\0'))
                || env.iter().any(|(key, value)| {
                    key.is_empty() || key.contains('\0') || value.contains('\0')
                })
            {
                return Err(invalid(
                    "arguments and environment must contain no NUL; environment names must be non-empty",
                ));
            }
        }
        McpTransportConfig::Http { url, headers } => {
            if !(url.starts_with("https://") || url.starts_with("http://")) {
                return Err(invalid("HTTP URL must start with http:// or https://"));
            }
            if !url.contains("${") && url::Url::parse(url).is_err() {
                return Err(invalid("HTTP URL is invalid"));
            }
            if headers.keys().any(|key| {
                key.is_empty()
                    || !key.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
                    })
            }) {
                return Err(invalid("HTTP header name is invalid"));
            }
        }
    }
    Ok(())
}

fn redact_url_userinfo(url: &str) -> String {
    match url.find("://") {
        Some(scheme_end) => match url[scheme_end + 3..].find('@') {
            Some(userinfo_end) => format!(
                "{}://<redacted>@{}",
                &url[..scheme_end],
                &url[scheme_end + 4 + userinfo_end..]
            ),
            None => url.to_owned(),
        },
        None => url.to_owned(),
    }
}

pub(crate) fn expand_environment(
    mut value: &str,
    mut lookup: impl FnMut(&str) -> Option<String>,
) -> Result<String, String> {
    let mut output = String::new();
    while let Some(start) = value.find("${") {
        output.push_str(&value[..start]);
        let rest = &value[start + 2..];
        let Some(end) = rest.find('}') else {
            return Err("unterminated ${ENV_VAR} placeholder".into());
        };
        let name = &rest[..end];
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        {
            return Err(format!(
                "invalid environment variable placeholder `${{{name}}}`"
            ));
        }
        let replacement =
            lookup(name).ok_or_else(|| format!("environment variable `{name}` is required"))?;
        output.push_str(&replacement);
        value = &rest[end + 1..];
    }
    output.push_str(value);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_environment_without_touching_process_state() {
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    command: "server".into(),
                    args: vec!["${TOKEN}".into()],
                    env: [("AUTH".into(), "Bearer ${TOKEN}".into())]
                        .into_iter()
                        .collect(),
                },
            }],
        };
        let resolved = config
            .resolve_with(|name| (name == "TOKEN").then(|| "secret".into()))
            .unwrap();
        let McpTransportConfig::Stdio { args, env, .. } = &resolved[0].transport else {
            panic!("expected stdio")
        };
        assert_eq!(args, &["secret"]);
        assert_eq!(env["AUTH"], "Bearer secret");
        assert!(format!("{:?}", resolved[0]).contains("redacted"));
    }

    #[test]
    fn rejects_duplicate_servers_and_missing_environment() {
        let server = McpServerConfig {
            name: "same".into(),
            transport: McpTransportConfig::Stdio {
                command: "server".into(),
                args: vec!["${MISSING}".into()],
                env: BTreeMap::new(),
            },
        };
        assert!(
            McpConfig {
                servers: vec![server.clone(), server]
            }
            .resolve_with(|_| None)
            .is_err()
        );
    }
}
