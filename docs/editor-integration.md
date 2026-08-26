# Editor integration with ACP

Harness can serve the [Agent Client Protocol][acp] over stdio. ACP-capable
editors can use the same agent stack as the terminal and headless frontends.
Zed supports ACP directly. Neovim, JetBrains IDEs, and other editors can use
compatible plugins.

Configure the editor to start this subprocess:

```text
harness acp
```

The editor sends `session/new`, `session/load`, `session/prompt`, and
`session/cancel` JSON-RPC messages. Harness returns streamed text, reasoning,
usage, and tool-call updates. Sessions are scoped to the workspace that the
editor opens.

> [!NOTE]
> `session/load` restores the saved history for the agent, but Harness does not
> replay the old transcript into the editor. The editor shows an empty
> transcript until the next turn. The restored history still supplies context
> to the model.

ACP `session/new` and `session/load` can supply stdio MCP servers for that
session. Session declarations replace the MCP servers in local configuration;
they do not merge with them. Harness rejects HTTP, SSE, and MCP-over-ACP
server declarations.

> [!WARNING]
> Tools run without a permission or confirmation step. An editor that sends a
> prompt can cause Harness to run built-in and configured MCP tools as soon as
> the model requests them.

Harness does not support authentication through ACP. Sign in to OAuth
providers before the editor starts Harness, or set the API key for an API-key
provider. See [Providers and authentication](./providers.md).

In ACP mode, stdout contains JSON-RPC only. Set `HARNESS_LOG` when you need
file-based diagnostics. Harness does not write normal progress text to stdout
or stderr because editors can treat subprocess output as protocol noise.

[acp]: https://agentclientprotocol.com/
