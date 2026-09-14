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

/// Supported MCP transport configuration.
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
    /// Streamable HTTP endpoint and headers applied to every request.
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
                .field("args", &format_args!("<{} entries redacted>", args.len()))
                .field("env", &format_args!("<{} entries redacted>", env.len()))
                .finish(),
            Self::Http { url, headers } => f
                .debug_struct("Http")
                .field("url", &redact_url(url))
                .field(
                    "headers",
                    &format_args!("<{} entries redacted>", headers.len()),
                )
                .finish(),
        }
    }
}

const MAX_MCP_SERVERS: usize = 64;
const MAX_MCP_CONFIG_BYTES: usize = 1024 * 1024;
const MAX_MCP_FIELD_BYTES: usize = 64 * 1024;
const MAX_SERVER_NAME_BYTES: usize = 256;

impl McpConfig {
    /// Validate configuration without expanding environment placeholders.
    pub fn validate(&self) -> Result<(), McpError> {
        validate_global_limits(self)?;
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
        // Check the caller-owned configuration before cloning it. In
        // particular, a large raw configuration must not be duplicated just
        // to discover that it is already over one of the global limits.
        self.validate()?;

        let mut expansion_budget = MAX_MCP_CONFIG_BYTES;
        let mut servers = self.servers.clone();
        servers.sort_by(|left, right| left.name.cmp(&right.name));
        for server in &mut servers {
            match &mut server.transport {
                McpTransportConfig::Stdio { args, env, .. } => {
                    for arg in args {
                        *arg = expand_with_budget(arg, &mut lookup, &mut expansion_budget)
                            .map_err(|message| McpError::Config {
                                server: server.name.clone(),
                                message,
                            })?;
                    }
                    for value in env.values_mut() {
                        *value = expand_with_budget(value, &mut lookup, &mut expansion_budget)
                            .map_err(|message| McpError::Config {
                                server: server.name.clone(),
                                message,
                            })?;
                    }
                }
                McpTransportConfig::Http { url, headers } => {
                    *url = expand_with_budget(url, &mut lookup, &mut expansion_budget).map_err(
                        |message| McpError::Config {
                            server: server.name.clone(),
                            message,
                        },
                    )?;
                    for value in headers.values_mut() {
                        *value = expand_with_budget(value, &mut lookup, &mut expansion_budget)
                            .map_err(|message| McpError::Config {
                                server: server.name.clone(),
                                message,
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

        // Expansion can introduce NULs, invalid URLs, duplicate-sensitive
        // sizes, and enough serialized data to cross the aggregate limit, so
        // validate the complete resolved form rather than relying on the raw
        // configuration checks above.
        let resolved = McpConfig { servers };
        resolved.validate()?;
        Ok(resolved.servers)
    }
}

fn validate_global_limits(config: &McpConfig) -> Result<(), McpError> {
    if config.servers.len() > MAX_MCP_SERVERS {
        return Err(McpError::Config {
            server: "<mcp>".into(),
            message: format!("too many servers; maximum is {MAX_MCP_SERVERS}"),
        });
    }
    if serde_json::to_vec(config)
        .map(|bytes| bytes.len() > MAX_MCP_CONFIG_BYTES)
        .unwrap_or(true)
    {
        return Err(McpError::Config {
            server: "<mcp>".into(),
            message: format!("configuration exceeds {MAX_MCP_CONFIG_BYTES} bytes"),
        });
    }
    Ok(())
}

fn validate_server(server: &McpServerConfig, names: &mut BTreeSet<String>) -> Result<(), McpError> {
    let invalid = |message: &str| McpError::Config {
        server: server.name.clone(),
        message: message.to_owned(),
    };
    if server.name.trim().is_empty()
        || server.name.contains('\0')
        || server.name.len() > MAX_SERVER_NAME_BYTES
        || server.name.chars().any(char::is_control)
    {
        return Err(invalid(
            "name must be non-empty, at most 256 bytes, and contain no control characters",
        ));
    }
    if !names.insert(server.name.clone()) {
        return Err(invalid("server names must be unique"));
    }
    match &server.transport {
        McpTransportConfig::Stdio { command, args, env } => {
            if command.as_os_str().is_empty() || command.to_string_lossy().contains('\0') {
                return Err(invalid("command must be non-empty and contain no NUL"));
            }
            if args.iter().any(|arg| arg.len() > MAX_MCP_FIELD_BYTES) {
                return Err(field_too_large(server, "argument"));
            }
            if env.iter().any(|(key, value)| {
                key.len() > MAX_MCP_FIELD_BYTES || value.len() > MAX_MCP_FIELD_BYTES
            }) {
                return Err(field_too_large(server, "environment entry"));
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
            if url.len() > MAX_MCP_FIELD_BYTES {
                return Err(field_too_large(server, "HTTP URL"));
            }
            if !(url.starts_with("https://") || url.starts_with("http://")) {
                return Err(invalid("HTTP URL must start with http:// or https://"));
            }
            if !url.contains("${") && url::Url::parse(url).is_err() {
                return Err(invalid("HTTP URL is invalid"));
            }
            let mut normalized_headers = BTreeSet::new();
            for (key, value) in headers {
                if key.len() > MAX_MCP_FIELD_BYTES || http::HeaderName::try_from(key).is_err() {
                    return Err(invalid("HTTP header name is invalid"));
                }
                if !normalized_headers.insert(key.to_ascii_lowercase()) {
                    return Err(invalid(
                        "HTTP header names must be case-insensitively unique",
                    ));
                }
                if value.len() > MAX_MCP_FIELD_BYTES {
                    return Err(field_too_large(server, "HTTP header value"));
                }
                if http::HeaderValue::try_from(value).is_err() {
                    return Err(invalid("HTTP header value is invalid"));
                }
            }
        }
    }
    Ok(())
}

fn field_too_large(server: &McpServerConfig, field: &str) -> McpError {
    McpError::Config {
        server: server.name.clone(),
        message: format!("{field} exceeds {MAX_MCP_FIELD_BYTES} bytes"),
    }
}

fn redact_url(value: &str) -> String {
    let Ok(mut url) = url::Url::parse(value) else {
        return "<redacted URL>".into();
    };
    if url.username() != "" {
        let _ = url.set_username("<redacted>");
    }
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

pub(crate) fn expand_environment(
    value: &str,
    lookup: impl FnMut(&str) -> Option<String>,
) -> Result<String, String> {
    expand_environment_bounded(value, lookup, MAX_MCP_FIELD_BYTES)
}

fn expand_with_budget(
    value: &str,
    lookup: impl FnMut(&str) -> Option<String>,
    budget: &mut usize,
) -> Result<String, String> {
    let limit = (*budget).min(MAX_MCP_FIELD_BYTES);
    let expanded = expand_environment_bounded(value, lookup, limit)?;
    *budget = (*budget).saturating_sub(expanded.len());
    Ok(expanded)
}

fn expand_environment_bounded(
    mut value: &str,
    mut lookup: impl FnMut(&str) -> Option<String>,
    limit: usize,
) -> Result<String, String> {
    let mut output = String::with_capacity(value.len().min(limit));
    while let Some(start) = value.find("${") {
        append_bounded(&mut output, &value[..start], limit)?;
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
        append_bounded(&mut output, &replacement, limit)?;
        value = &rest[end + 1..];
    }
    append_bounded(&mut output, value, limit)?;
    Ok(output)
}

fn append_bounded(output: &mut String, value: &str, limit: usize) -> Result<(), String> {
    if value.len() > limit.saturating_sub(output.len()) {
        return Err(format!("expanded value exceeds {limit} bytes"));
    }
    output.push_str(value);
    Ok(())
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
        let debug = format!("{:?}", resolved[0]);
        assert!(debug.contains("redacted"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn resolves_all_value_surfaces_and_sorts_without_mutating_input() {
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "remote".into(),
                    transport: McpTransportConfig::Http {
                        url: "https://${HOST}/mcp?token=${TOKEN}".into(),
                        headers: [("Authorization".into(), "Bearer ${TOKEN}".into())]
                            .into_iter()
                            .collect(),
                    },
                },
                McpServerConfig {
                    name: "local".into(),
                    transport: McpTransportConfig::Stdio {
                        command: "server".into(),
                        args: vec!["--token=${TOKEN}".into()],
                        env: [("AUTH".into(), "${TOKEN}".into())].into_iter().collect(),
                    },
                },
            ],
        };
        let resolved = config
            .resolve_with(|name| match name {
                "HOST" => Some("example.test".into()),
                "TOKEN" => Some("resolved-secret".into()),
                _ => None,
            })
            .unwrap();

        assert_eq!(resolved[0].name, "local");
        assert_eq!(resolved[1].name, "remote");
        let McpTransportConfig::Http { url, headers } = &resolved[1].transport else {
            panic!("expected HTTP transport")
        };
        assert_eq!(url, "https://example.test/mcp?token=resolved-secret");
        assert_eq!(headers["Authorization"], "Bearer resolved-secret");
        assert_eq!(config.servers[0].name, "remote");
        let debug = format!("{:?}", resolved[1]);
        assert!(!debug.contains("resolved-secret"));
    }

    #[test]
    fn rejects_oversized_expansions_in_every_value_surface() {
        let oversized =
            "oversized-secret".repeat(MAX_MCP_FIELD_BYTES / "oversized-secret".len() + 1);
        let configs = [
            McpConfig {
                servers: vec![McpServerConfig {
                    name: "args".into(),
                    transport: McpTransportConfig::Stdio {
                        command: "server".into(),
                        args: vec!["${TOKEN}".into()],
                        env: BTreeMap::new(),
                    },
                }],
            },
            McpConfig {
                servers: vec![McpServerConfig {
                    name: "env".into(),
                    transport: McpTransportConfig::Stdio {
                        command: "server".into(),
                        args: Vec::new(),
                        env: [("TOKEN_VALUE".into(), "${TOKEN}".into())]
                            .into_iter()
                            .collect(),
                    },
                }],
            },
            McpConfig {
                servers: vec![McpServerConfig {
                    name: "headers".into(),
                    transport: McpTransportConfig::Http {
                        url: "https://example.test/mcp".into(),
                        headers: [("Authorization".into(), "${TOKEN}".into())]
                            .into_iter()
                            .collect(),
                    },
                }],
            },
            McpConfig {
                servers: vec![McpServerConfig {
                    name: "url".into(),
                    transport: McpTransportConfig::Http {
                        url: "https://${TOKEN}/mcp".into(),
                        headers: BTreeMap::new(),
                    },
                }],
            },
        ];

        for config in configs {
            let error = config
                .resolve_with(|_| Some(oversized.clone()))
                .expect_err("oversized expansion must be rejected");
            let rendered = error.to_string();
            assert!(rendered.contains("expanded value exceeds"), "{rendered}");
            assert!(!rendered.contains("oversized-secret"), "{rendered}");
        }
    }

    #[test]
    fn rejects_oversized_pre_resolved_configuration_before_cloning() {
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    command: "server".into(),
                    args: (0..16).map(|_| "x".repeat(MAX_MCP_FIELD_BYTES)).collect(),
                    env: BTreeMap::new(),
                },
            }],
        };

        let error = config
            .resolve_with(|_| None)
            .expect_err("oversized raw config must be rejected");
        assert!(matches!(
            error,
            McpError::Config { server, message }
                if server == "<mcp>" && message.contains("configuration exceeds")
        ));
    }

    #[test]
    fn rejects_aggregate_size_after_expansion() {
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    command: "server".into(),
                    args: (0..16).map(|_| "${TOKEN}".into()).collect(),
                    env: BTreeMap::new(),
                },
            }],
        };
        assert!(config.validate().is_ok());
        let expansion = "x".repeat(MAX_MCP_FIELD_BYTES);

        let error = config
            .resolve_with(|_| Some(expansion.clone()))
            .expect_err("resolved aggregate must be revalidated");
        assert!(matches!(
            error,
            McpError::Config { server, message }
                if server == "<mcp>" && message.contains("configuration exceeds")
        ));

        let debug = format!("{:?}", config);
        assert!(!debug.contains(&expansion));
    }

    #[test]
    fn rejects_nul_introduced_by_expansion_during_final_validation() {
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    command: "server".into(),
                    args: vec!["${TOKEN}".into()],
                    env: BTreeMap::new(),
                },
            }],
        };

        let error = config
            .resolve_with(|_| Some("bad\0value".into()))
            .expect_err("expanded NUL must be rejected");
        assert!(error.to_string().contains("NUL"));
    }

    #[test]
    fn rejects_unsafe_or_case_duplicate_http_headers() {
        for headers in [
            BTreeMap::from([("X-Test".into(), "line one\nline two".into())]),
            BTreeMap::from([
                ("X-Test".into(), "one".into()),
                ("x-test".into(), "two".into()),
            ]),
        ] {
            let config = McpConfig {
                servers: vec![McpServerConfig {
                    name: "remote".into(),
                    transport: McpTransportConfig::Http {
                        url: "https://example.test/mcp".into(),
                        headers,
                    },
                }],
            };
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn debug_redacts_http_credentials_query_and_fragment() {
        let config = McpServerConfig {
            name: "remote".into(),
            transport: McpTransportConfig::Http {
                url: "https://user:password@example.test/path?token=resolved-secret#fragment"
                    .into(),
                headers: BTreeMap::new(),
            },
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("example.test/path"));
        assert!(!debug.contains("password"));
        assert!(!debug.contains("resolved-secret"));
        assert!(!debug.contains("fragment"));
    }

    #[test]
    fn debug_redacts_unparseable_urls() {
        let config = McpServerConfig {
            name: "remote".into(),
            transport: McpTransportConfig::Http {
                url: "https://[not-a-url]?secret=raw".into(),
                headers: BTreeMap::new(),
            },
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("<redacted URL>"));
        assert!(!debug.contains("not-a-url"));
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

    #[test]
    fn rejects_excessive_server_count_and_control_names() {
        let servers = (0..65)
            .map(|index| McpServerConfig {
                name: format!("server-{index}"),
                transport: McpTransportConfig::Stdio {
                    command: "server".into(),
                    args: Vec::new(),
                    env: BTreeMap::new(),
                },
            })
            .collect();
        let config = McpConfig { servers };
        assert!(config.validate().is_err());
        let error = config
            .resolve_with(|_| None)
            .expect_err("the resolver must enforce the server count limit");
        assert!(matches!(
            error,
            McpError::Config { server, message }
                if server == "<mcp>" && message.contains("too many servers")
        ));

        let invalid = McpConfig {
            servers: vec![McpServerConfig {
                name: "bad\nname".into(),
                transport: McpTransportConfig::Stdio {
                    command: "server".into(),
                    args: Vec::new(),
                    env: BTreeMap::new(),
                },
            }],
        };
        assert!(invalid.validate().is_err());
    }
}
