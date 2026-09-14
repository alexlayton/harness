use crate::config::{
    McpArgs, McpCommand, add_mcp_server, config_path, delete_mcp_server, load_file_config,
};
use anyhow::Result;
use mcp::McpTransportConfig;
use std::process::ExitCode;

/// Run configuration-only MCP commands without starting an agent or provider.
pub fn run(args: &McpArgs) -> Result<ExitCode> {
    match &args.command {
        Some(McpCommand::Add(args)) => {
            add_mcp_server(&args.name, &args.command)?;
            println!("Added MCP server `{}`.", args.name);
        }
        Some(McpCommand::Delete { name }) => {
            delete_mcp_server(name)?;
            println!("Deleted MCP server `{name}`.");
        }
        Some(McpCommand::List) | None => list()?,
    }
    Ok(ExitCode::SUCCESS)
}

fn list() -> Result<()> {
    let config = load_file_config(&config_path())?;
    let servers = config.mcp.map(|mcp| mcp.servers).unwrap_or_default();
    if servers.is_empty() {
        println!("No MCP servers configured.");
        return Ok(());
    }

    for server in servers {
        match server.transport {
            McpTransportConfig::Stdio { command, args, .. } => {
                let mut invocation = format!("{:?}", command);
                for arg in args {
                    invocation.push(' ');
                    invocation.push_str(&format!("{arg:?}"));
                }
                println!("{}\tstdio\t{}", server.name, invocation);
            }
            McpTransportConfig::Http { url, .. } => {
                println!("{}\thttp\t{}", server.name, url);
            }
        }
    }
    Ok(())
}
