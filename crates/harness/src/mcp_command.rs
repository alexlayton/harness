use crate::config::{
    McpArgs, McpCommand, add_http_mcp_server, add_mcp_server, config_path, delete_mcp_server,
    load_file_config, set_mcp_server_enabled,
};
use anyhow::{Result, anyhow};
use mcp::McpTransportConfig;
use std::collections::BTreeMap;
use std::process::ExitCode;

/// Run configuration-only MCP commands without starting an agent or provider.
pub fn run(args: &McpArgs) -> Result<ExitCode> {
    match &args.command {
        Some(McpCommand::Add(args)) => {
            if let Some(url) = &args.url {
                let headers = parse_headers(&args.header)?;
                add_http_mcp_server(&args.name, url, &headers, !args.disabled)?;
            } else {
                add_mcp_server(&args.name, &args.command, !args.disabled)?;
            }
            println!("Added MCP server `{}`.", args.name);
            if args.disabled {
                println!("It connects only with `--mcp {}`.", args.name);
            }
        }
        Some(McpCommand::Enable { name }) => {
            set_mcp_server_enabled(name, true)?;
            println!("MCP server `{name}` now connects in every launch.");
        }
        Some(McpCommand::Disable { name }) => {
            set_mcp_server_enabled(name, false)?;
            println!("MCP server `{name}` now connects only with `--mcp {name}`.");
        }
        Some(McpCommand::Delete { name }) => {
            delete_mcp_server(name)?;
            println!("Deleted MCP server `{name}`.");
        }
        Some(McpCommand::List) | None => list()?,
    }
    Ok(ExitCode::SUCCESS)
}

fn parse_headers(headers: &[String]) -> Result<BTreeMap<String, String>> {
    headers
        .iter()
        .map(|header| {
            let (name, value) = header
                .split_once('=')
                .ok_or_else(|| anyhow!("HTTP header must use NAME=VALUE syntax"))?;
            if name.is_empty() {
                return Err(anyhow!("HTTP header name must not be empty"));
            }
            Ok((name.to_owned(), value.to_owned()))
        })
        .collect()
}

fn list() -> Result<()> {
    let config = load_file_config(&config_path())?;
    let servers = config.mcp.map(|mcp| mcp.servers).unwrap_or_default();
    if servers.is_empty() {
        println!("No MCP servers configured.");
        return Ok(());
    }

    for entry in servers {
        let server = entry.config;
        // Opt-in servers are marked so the list explains why a launch
        // may not connect them.
        let suffix = if entry.enabled { "" } else { "\t(opt-in)" };
        match server.transport {
            McpTransportConfig::Stdio { command, args, .. } => {
                let mut invocation = format!("{:?}", command);
                for arg in args {
                    invocation.push(' ');
                    invocation.push_str(&format!("{arg:?}"));
                }
                println!("{}\tstdio\t{}{}", server.name, invocation, suffix);
            }
            McpTransportConfig::Http { url, .. } => {
                // URLs often carry tokens in userinfo or the query.
                println!("{}\thttp\t{}{}", server.name, mcp::redact_url(&url), suffix);
            }
        }
    }
    Ok(())
}
