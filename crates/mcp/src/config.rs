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
    /// When true, a failure to start this server aborts startup. Otherwise
    /// the failure is recorded and the remaining servers still connect.
    #[serde(default)]
    pub required: bool,
    /// Transport and its non-secret connection settings.
    #[serde(flatten)]
    pub transport: McpTransportConfig,
}

impl McpServerConfig {
    /// One line saying what this server launches or contacts, safe to show a
    /// user: `stdio: <command> <args>` or `http: <url>`.
    ///
    /// Arguments are resolved (`${VAR}` already expanded), so a token passed
    /// as an argument would otherwise appear verbatim. Values after `=`, the
    /// value following a flag that names a credential, and any configured
    /// environment value are replaced. Positional arguments are kept so the
    /// user can still see what runs. Not a guarantee: a bare positional secret
    /// is indistinguishable from a subcommand, so callers should also apply
    /// their own secret masker.
    pub fn describe(&self) -> String {
        match &self.transport {
            McpTransportConfig::Stdio { command, args, .. } => {
                let mut parts = vec![command.display().to_string()];
                let mut hide_next = false;
                for arg in args {
                    parts.push(describe_arg(arg, &mut hide_next));
                }
                let line = Redactions::for_server(self).apply(&parts.join(" "));
                format!("stdio: {}", truncate_display(&line, 200))
            }
            McpTransportConfig::Http { url, .. } => format!("http: {}", redact_url(url)),
        }
    }
}

/// Flag names that suggest the next argument is a credential.
fn is_sensitive_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    [
        "token",
        "key",
        "secret",
        "password",
        "passwd",
        "auth",
        "credential",
        "cookie",
    ]
    .iter()
    .any(|word| name.contains(word))
}

fn describe_arg(arg: &str, hide_next: &mut bool) -> String {
    if std::mem::take(hide_next) {
        return "<redacted>".into();
    }
    if let Some((name, _)) = arg.split_once('=')
        && !name.is_empty()
        && !name.contains(char::is_whitespace)
        && !name.contains('/')
    {
        return format!("{name}=<redacted>");
    }
    if arg.starts_with('-') && is_sensitive_name(arg) {
        *hide_next = true;
    }
    truncate_display(arg, 80)
}

fn truncate_display(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut shortened: String = text.chars().take(max).collect();
    shortened.push('…');
    shortened
}

impl fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("required", &self.required)
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
        /// Working directory for the server process. Relative paths resolve
        /// against the workspace root, which is also the default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
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
            Self::Stdio {
                command,
                args,
                env,
                cwd,
            } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("cwd", cwd)
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
/// Limit simultaneous connection attempts (including stdio process startup).
/// All configured servers are still visited; admission waits for a slot.
pub(crate) const MAX_CONCURRENT_MCP_STARTUPS: usize = 4;
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
        McpTransportConfig::Stdio {
            command,
            args,
            env,
            cwd,
        } => {
            if let Some(cwd) = cwd
                && (cwd.as_os_str().is_empty()
                    || cwd.to_string_lossy().contains('\0')
                    || cwd.as_os_str().len() > MAX_MCP_FIELD_BYTES)
            {
                return Err(invalid(
                    "cwd must be non-empty, contain no NUL, and fit the field limit",
                ));
            }
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

/// Configured values that must never appear in diagnostics for one server:
/// HTTP header values, URL credentials, query values and fragment, and stdio
/// environment values. Servers and HTTP clients can echo these back in error
/// bodies or URLs; every MCP error and tool-call error passes through
/// [`Redactions::apply`] before it reaches the UI, logs, or the model.
#[derive(Clone, Debug, Default)]
pub(crate) struct Redactions(std::sync::Arc<[String]>);

/// Shorter values (such as `1` or `true`) would redact ordinary text.
const MIN_REDACTED_BYTES: usize = 6;

impl Redactions {
    pub(crate) fn for_server(server: &McpServerConfig) -> Self {
        let mut values = Vec::new();
        match &server.transport {
            McpTransportConfig::Stdio { env, .. } => values.extend(env.values().cloned()),
            McpTransportConfig::Http { url, headers } => {
                for value in headers.values() {
                    values.push(value.clone());
                    // `Authorization: Bearer <token>` can be echoed without
                    // its scheme.
                    if let Some((_, token)) = value.split_once(' ') {
                        values.push(token.to_owned());
                    }
                }
                if let Ok(url) = url::Url::parse(url) {
                    values.push(url.username().to_owned());
                    values.extend(url.password().map(str::to_owned));
                    values.extend(url.query().map(str::to_owned));
                    values.extend(url.fragment().map(str::to_owned));
                    values.extend(url.query_pairs().map(|(_, value)| value.into_owned()));
                }
            }
        }
        values.retain(|value| value.trim().len() >= MIN_REDACTED_BYTES);
        // Longest first, so a value that contains another is replaced whole.
        values.sort_by_key(|value| std::cmp::Reverse(value.len()));
        values.dedup();
        Self(values.into())
    }

    pub(crate) fn apply(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for value in self.0.iter() {
            if text.contains(value.as_str()) {
                text = text.replace(value.as_str(), "[redacted]");
            }
        }
        text
    }
}

/// Display form of a server URL with userinfo, query, and fragment removed;
/// those parts commonly carry tokens. Unparseable URLs are fully hidden.
pub fn redact_url(value: &str) -> String {
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

    fn stdio(args: &[&str], env: &[(&str, &str)]) -> McpServerConfig {
        McpServerConfig {
            name: "s".into(),
            required: false,
            transport: McpTransportConfig::Stdio {
                command: "bunx".into(),
                args: args.iter().map(|arg| (*arg).to_owned()).collect(),
                env: env
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect(),
                cwd: None,
            },
        }
    }

    #[test]
    fn describe_keeps_what_runs_but_hides_argument_values() {
        let server = stdio(
            &[
                "safari-mcp",
                "--token=abc123secret",
                "--api-key",
                "hunter2hunter2",
                "KEY=value-here",
                "--port",
                "9225",
            ],
            &[("OTHER", "env-secret-value")],
        );
        let text = server.describe();
        assert_eq!(
            text,
            "stdio: bunx safari-mcp --token=<redacted> --api-key <redacted> KEY=<redacted> --port 9225"
        );
    }

    #[test]
    fn describe_redacts_environment_values_and_truncates() {
        let text = stdio(
            &["positional-env-secret-value"],
            &[("T", "positional-env-secret-value")],
        )
        .describe();
        assert!(!text.contains("env-secret-value"), "{text}");
        let long = "x".repeat(500);
        assert!(stdio(&[&long], &[]).describe().chars().count() < 260);
    }

    #[test]
    fn describe_redacts_http_url_credentials_and_query() {
        let server = McpServerConfig {
            name: "w".into(),
            required: false,
            transport: McpTransportConfig::Http {
                url: "https://user:pw@example.test/mcp?token=abc".into(),
                headers: BTreeMap::new(),
            },
        };
        let text = server.describe();
        assert!(text.starts_with("http: https://"), "{text}");
        assert!(!text.contains("pw") && !text.contains("abc"), "{text}");
    }

    #[test]
    fn resolves_environment_without_touching_process_state() {
        let config = McpConfig {
            servers: vec![McpServerConfig {
                required: false,
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    cwd: None,
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
                    required: false,
                    name: "remote".into(),
                    transport: McpTransportConfig::Http {
                        url: "https://${HOST}/mcp?token=${TOKEN}".into(),
                        headers: [("Authorization".into(), "Bearer ${TOKEN}".into())]
                            .into_iter()
                            .collect(),
                    },
                },
                McpServerConfig {
                    required: false,
                    name: "local".into(),
                    transport: McpTransportConfig::Stdio {
                        cwd: None,
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
                    required: false,
                    name: "args".into(),
                    transport: McpTransportConfig::Stdio {
                        cwd: None,
                        command: "server".into(),
                        args: vec!["${TOKEN}".into()],
                        env: BTreeMap::new(),
                    },
                }],
            },
            McpConfig {
                servers: vec![McpServerConfig {
                    required: false,
                    name: "env".into(),
                    transport: McpTransportConfig::Stdio {
                        cwd: None,
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
                    required: false,
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
                    required: false,
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
                required: false,
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    cwd: None,
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
                required: false,
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    cwd: None,
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
                required: false,
                name: "server".into(),
                transport: McpTransportConfig::Stdio {
                    cwd: None,
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
                    required: false,
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
            required: false,
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
    fn diagnostics_redact_configured_header_and_url_secrets() {
        let config = McpServerConfig {
            required: false,
            name: "remote".into(),
            transport: McpTransportConfig::Http {
                url: "https://user:password@example.test/path?token=resolved-secret".into(),
                headers: BTreeMap::from([(
                    "Authorization".into(),
                    "Bearer header-secret-value".into(),
                )]),
            },
        };
        let redactions = Redactions::for_server(&config);
        let error = McpError::operation(
            "remote",
            "initialize",
            "HTTP 401: invalid token header-secret-value for \
             https://user:password@example.test/path?token=resolved-secret",
        )
        .redact(&redactions)
        .to_string();
        for secret in ["header-secret-value", "password", "resolved-secret"] {
            assert!(!error.contains(secret), "{secret} leaked: {error}");
        }
        assert!(error.contains("[redacted]"));
    }

    #[test]
    fn debug_redacts_unparseable_urls() {
        let config = McpServerConfig {
            required: false,
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
            required: false,
            name: "same".into(),
            transport: McpTransportConfig::Stdio {
                cwd: None,
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
                required: false,
                name: format!("server-{index}"),
                transport: McpTransportConfig::Stdio {
                    cwd: None,
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
                required: false,
                name: "bad\nname".into(),
                transport: McpTransportConfig::Stdio {
                    cwd: None,
                    command: "server".into(),
                    args: Vec::new(),
                    env: BTreeMap::new(),
                },
            }],
        };
        assert!(invalid.validate().is_err());
    }
}
