# Harness

Harness is a terminal coding-agent harness. It currently supports OpenCode Go,
OpenRouter, and GitHub Copilot.

Contributing to (or working as an agent in) this repository? Start with
[AGENTS.md](./AGENTS.md) and [ARCHITECTURE.md](./ARCHITECTURE.md).

## GitHub Copilot

Start Harness with the Copilot provider when logging in for the first time:

```text
harness --provider github-copilot
```

Run `/auth` in the TUI. Harness displays GitHub's device URL and one-time code;
open the URL in a browser and authorize the device. Press `Ctrl+C` while it is
waiting to cancel the login without quitting Harness. After login, use `/model`
to select one of the models available to the account. When no model is
configured, Harness defaults to the first model from the account's own
available list that it knows how to route, and switches automatically after a
fresh login if the current model is not entitled to the account.

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
The first TUI flow defaults to `github.com`. For the TUI, set
`HARNESS_GITHUB_ENTERPRISE_DOMAIN` to a hostname or URL before running `/auth`;
blank or unset means `github.com`. Embedding front ends can instead pass the
optional domain directly without changing credential storage or provider
routing.

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
