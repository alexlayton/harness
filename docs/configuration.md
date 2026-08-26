# Configuration

Harness reads its normal settings from:

```text
~/.config/harness/config.toml
```

`HARNESS_CONFIG_DIR` changes the configuration directory. If that variable is
not set, Harness uses `$XDG_CONFIG_HOME/harness` when `XDG_CONFIG_HOME` is set.
It otherwise uses `~/.config/harness`.

OAuth credentials are separate from normal settings. They are stored in
`auth.json` in the same configuration directory. API keys remain in the
environment. For provider-specific setup, see
[Providers and authentication](./providers.md).

## Basic settings

```toml
provider = "opencode-go"
model = "gpt-5.6-luna"
rtk = false
```

The command line overrides `provider` and `model` for the current process:

```text
harness --provider openrouter --model openai/<model>
```

Harness requires a provider choice rather than assuming one from an API key.
When no provider is configured, startup lists the available providers and their
setup commands. A successful OAuth login selects that provider only when this
setting is absent.

The `/model` terminal command updates the saved provider and model. Harness
preserves configuration keys that it does not know when it saves these values.

Set `rtk = true` to let the bash tool rewrite supported commands through an
installed `rtk` executable for smaller tool output. This feature is off by
default.

## Terminal UI

To start directly at the input field without printing the welcome banner and
workspace/provider metadata, enable minimal mode:

```toml
[tui]
minimal = true
```

Turn activity, usage, completion, and session output are unchanged.

## Compaction

Harness automatically summarizes older context when token pressure reaches a
configured threshold. All fields are optional. Missing fields use the defaults
shown here:

```toml
[compaction]
auto = true
threshold = 0.80
reserve_tokens = 16384
keep_recent_turns = 10
keep_recent_tokens = 20000
max_summary_input_bytes = 98304
max_summary_bytes = 12288
context_window = 0
```

- `auto`: Enable automatic compaction. Set it to `false` to use only
  `/compact`.
- `threshold`: Trigger fraction of the context window.
- `reserve_tokens`: Keep this amount available for the next model response.
- `keep_recent_turns`: Keep this many recent turns verbatim.
- `keep_recent_tokens`: Limit the estimated size of the verbatim tail.
- `max_summary_input_bytes`: Limit the serialized text sent to the summarizer.
- `max_summary_bytes`: Limit the generated summary.
- `context_window`: Override the provider's reported context window. `0` uses
  provider metadata, then a conservative fallback if metadata is unavailable.

Compaction appends a summary event. It does not rewrite or delete old session
events.

## Subagents

Subagents have bounded nested turns and concurrency. The defaults are:

```toml
[subagents]
max_turns = 25
max_concurrent = 4
```

Set `max_turns = 0` to disable the subagent tool. `max_concurrent` limits
parallel read-only delegations and is clamped to at least one. Workspace-mode
subagents are exclusive and run in sequence.

Subagents do not receive external MCP tools and cannot create more subagents.

## MCP servers

Harness can start external stdio MCP servers from `config.toml`:

```toml
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "/absolute/path/to/mcp-server"
args = ["--root", "."]

[mcp.servers.env]
LOG_LEVEL = "warn"
TOKEN = "${MCP_TOKEN}"
```

Harness expands `${ENV_VAR}` placeholders in arguments and environment values
at startup. A missing variable stops startup. Expanded secrets are not written
back to configuration or session history.

Servers run directly, not through a shell, and use the workspace as their
working directory. Server stdout is reserved for MCP framing. Server stderr
does not enter headless or ACP stdout.

MCP tools are discovered when the session starts. Reconnect to apply server
tool-list changes. Calls are serialized as exclusive operations and run
without a confirmation step. Subagents do not receive MCP tools.

Only stdio transport is enabled. Streamable HTTP and legacy SSE transports are
not enabled.

## Sessions

Normal sessions are stored under:

```text
~/.harness/sessions
```

Sessions are append-only JSONL and are grouped by workspace. `/new`, `/load`,
`/sessions`, and `/export` operate on this durable state. `/export` writes to
the current directory by default.

Use `HARNESS_SESSION_DIR` or `HARNESS_STATE_DIR` to change the session root.
The session format is documented in
[`crates/session/README.md`](../crates/session/README.md).

Headless mode can skip persistence for one run:

```text
harness prompt --no-session "prompt"
```

`--defer-session-sync` syncs at turn boundaries instead of after each event.
This is faster for tool-heavy turns, but a power failure can lose the current
turn's tail.

## Project context and skills

Harness loads project instructions from `AGENTS.md` and `CLAUDE.md` files. It
walks from the current workspace to the Git root. It also checks
`~/.harness/AGENTS.md`, or `$HARNESS_CONFIG_DIR/AGENTS.md` when that variable is
set. Use `--no-context-files` or `HARNESS_NO_CONTEXT_FILES` to disable this
context.

Skills are discovered from project `.harness/skills/` and `.agents/skills/`
directories, plus global `~/.harness/skills` and `~/.agents/skills` locations.
`HARNESS_SKILLS_DIR` overrides the global Harness skills directory.

See [Architecture](../ARCHITECTURE.md#skills-and-project-context) for discovery
order, size limits, and prompt invariants.

## Logging

Set `HARNESS_LOG` to a file path to enable tracing:

```text
HARNESS_LOG=/tmp/harness.log harness
```

Tracing is file-only. This keeps terminal rendering clean and keeps headless
and ACP stdout suitable for machine use.

## Command-line interface

With no subcommand, Harness starts the terminal UI. Other frontends use
subcommands:

```text
harness prompt --provider openrouter "Summarize this workspace"
harness acp --provider openai-codex
```

| Command or option | Purpose |
|---|---|
| `prompt [PROMPT]` | Run one prompt and print only the final answer to stdout. Reads piped stdin when the prompt is omitted. |
| `acp` | Serve ACP over stdio. |
| `login <provider>` | Authenticate with an OAuth provider. |
| `--provider <provider>` | Override the configured provider. |
| `--model <model>` | Override the configured model. |
| `-v`, `--verbose` | In prompt mode, write reasoning and tool progress to stderr. |
| `--resume <selector>` | Resume a session by ID, unique prefix, `latest`, or path in prompt mode. |
| `--no-session` | Do not persist a prompt-mode run. |
| `--defer-session-sync` | Sync session records at turn boundaries. |
| `--no-context-files` | Disable project instruction files. |
