# Harness

Harness is a terminal coding-agent harness. It supports OpenCode Go,
OpenRouter, GitHub Copilot, and the ChatGPT OpenAI Codex subscription endpoint.

Contributing to (or working as an agent in) this repository? Start with
[AGENTS.md](./AGENTS.md) and [ARCHITECTURE.md](./ARCHITECTURE.md).

## GitHub Copilot

Sign in before starting a Copilot session:

```text
harness login github-copilot
# `copilot` is an alias
```

The command displays GitHub's device URL and one-time code and attempts to open
it in a browser. `Ctrl+C` cancels login. Login stores credentials only: it does
not change the selected provider or model in `config.toml`. Use `/model` after
login to select an available model.

GitHub Copilot credentials are kept separately from normal configuration:

```text
~/.config/harness/auth.json
```

`HARNESS_CONFIG_DIR` and `XDG_CONFIG_HOME` apply to this path in the same way
they apply to `config.toml`. The auth directory is created with mode `0700` and
the auth file with mode `0600` on Unix-like systems. The file contains the
short-lived Copilot access token and the GitHub OAuth refresh token. They are
never written to `config.toml` or Harness session files.

Harness stores the normal provider/model selection in:

```text
~/.config/harness/config.toml
```

Conversation history is separate again, under `~/.harness/sessions`. `/new`,
`/load`, `/sessions`, and `/export` operate on that durable conversation state;
`/export` writes to the current directory by default.

GitHub Enterprise domains are supported by the auth crate's domain-aware API.
The standalone command defaults to `github.com`; embedders can pass an optional
domain without changing credential storage or provider routing.

Copilot's device, token, model-policy, and proxy endpoints are client behavior
used by VS Code/Pi rather than a stable public LLM API. Endpoint paths, headers,
model-policy filtering, and static dialect metadata are isolated in
`crates/auth` and `crates/llm/src/providers/github_copilot.rs` so they can be
updated independently. Claude models route through the Anthropic-compatible
`/v1/messages` endpoint, GPT-5-family and MAI models through `/responses`, and
the rest through `/chat/completions`, matching the per-model
`supported_endpoints` metadata Copilot publishes. Free plans gate most premium
models behind billing; when no model is configured, the default prefers one the
plan can serve, and plan-gated requests surface an actionable error instead of
a bare 400.

## OpenAI Codex

Sign in to the ChatGPT subscription provider with browser OAuth (the default),
or use device authorization when a loopback callback is unavailable:

```text
harness login openai-codex
harness login codex --device-code
```

Credentials are stored alongside Copilot in `~/.config/harness/auth.json` with
private Unix permissions. Select `openai-codex` (or `codex`) and one of its
static models through `/model`. Codex uses a dedicated SSE-only provider and
dialect because its subscription protocol differs from the normal OpenAI API.

## MCP tools

Harness can start external stdio MCP servers from `config.toml`:

```toml
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "/absolute/path/to/mcp-server"
args = ["--root", "."]

[mcp.servers.env]
LOG_LEVEL = "warn"
```

`${ENV_VAR}` placeholders in arguments and environment values are expanded at
startup; missing variables fail startup and expanded secrets are never saved to
configuration or session history. Servers run directly (not through a shell)
with the workspace as their cwd. Their tools run without an approval gate,
like built-ins, and are serialized as exclusive calls. Stdio server output is
reserved for MCP framing; server stderr never reaches headless or ACP stdout.

MCP tools are a session-start snapshot: server tool-list changes take effect on
reconnect. Subagents intentionally do not receive external MCP tools. Rich
binary MCP content is represented by bounded metadata/omission markers in the
text-only provider history. Streamable HTTP and legacy SSE transports are not
yet enabled.

## Editor integration (ACP)

Harness ships a third frontend that speaks the [Agent Client Protocol][acp] over
stdio, so ACP-capable editors (Zed natively; Neovim/JetBrains via plugins) can
drive it in place of a standalone agent:

```text
harness --acp
```

Editors spawn `harness --acp` as a subprocess and send `session/new`,
`session/prompt`, and `session/cancel` JSON-RPC messages; Harness answers with
streamed updates and tool-call notifications. Sessions are scoped to the
workspace the editor opens. ACP `session/new` and `session/load` may supply
stdio MCP servers for that session; these replace rather than merge local MCP
configuration. HTTP, SSE, and MCP-over-ACP declarations are rejected clearly.

**Tools run without a confirmation step**, exactly like the TUI and headless
frontends, so an editor that auto-approves prompts is executing your tools as
soon as the model chooses to. There is no permission gate over ACP.

[acp]: https://agentclientprotocol.com/
