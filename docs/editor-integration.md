# Editor integration with ACP

Harness can serve the [Agent Client Protocol][acp] over stdio. ACP-capable
editors can use the same agent stack as the terminal and headless frontends.
Zed supports ACP directly. Neovim, JetBrains IDEs, and other editors can use
compatible plugins.

Configure the editor to start this subprocess:

```text
harness acp
```

The editor sends `session/new`, `session/load`, `session/list`,
`session/delete`, `session/prompt`, and `session/cancel` JSON-RPC messages.
It also accepts `session/set_mode` for its single `work` mode.
Harness returns streamed text, reasoning, usage, and tool-call updates.
Sessions are scoped to the workspace that the editor opens.

`session/load` replays saved user messages, agent messages, reasoning, and
completed tool calls to the editor before it responds. The agent also restores
the saved context for the next turn.

ACP `session/new` and `session/load` can supply stdio or Streamable HTTP MCP
servers for that session. Session declarations replace the MCP servers in local
configuration; they do not merge with them. Harness advertises stdio and HTTP
MCP transport support; it rejects deprecated SSE and MCP-over-ACP server
declarations. ACP gives session assembly a separate 35-second bound, which
leaves room for both the 15-second MCP lifecycle and catalogue deadlines when
they occur sequentially. Disconnect and session deletion retain a short cleanup
bound and cancel the owned agent/MCP tasks before using it as a final fallback.

> [!WARNING]
> Tools run without a permission or confirmation step. An editor that sends a
> prompt can cause Harness to run built-in and configured MCP tools as soon as
> the model requests them.

ACP v1 is supported. ACP v2 is still a draft and is not implemented. Prompts
support text, file links, and embedded text context. Images, audio, and binary
embedded resources are not supported; Harness rejects them instead of dropping
them silently.

Harness does not support authentication through ACP. Sign in to OAuth
providers before the editor starts Harness, or set the API key for an API-key
provider. See [Providers and authentication](./providers.md).

In ACP mode, stdout contains JSON-RPC only. Set `HARNESS_LOG` when you need
file-based diagnostics. Harness does not write normal progress text to stdout
or stderr because editors can treat subprocess output as protocol noise.

[acp]: https://agentclientprotocol.com/
