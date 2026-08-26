# Providers and authentication

Harness supports four providers. Select one with `--provider`, in
`config.toml`, or with `/model` in the terminal UI. If none is selected,
Harness prints setup instructions instead of assuming an API-key provider.

| Provider | Configuration name | Authentication |
|---|---|---|
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` or `OPENCODE_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| GitHub Copilot | `github-copilot` (`copilot`) | Device login |
| OpenAI Codex | `openai-codex` (`codex`) | Browser or device login |

API keys are environment-only secrets. Harness does not write them to
`config.toml` or session files.

## OpenCode Go

Set `OPENCODE_GO_API_KEY`. The older `OPENCODE_API_KEY` name is also accepted
for compatibility with existing OpenCode setups.

```text
harness --provider opencode-go
```

## OpenRouter

Set `OPENROUTER_API_KEY`, then select the provider:

```text
harness --provider openrouter
```

OpenRouter model IDs include the provider prefix, for example
`openai/<model>`.

## GitHub Copilot

Sign in before you start a Copilot session:

```text
harness login github-copilot
# `copilot` is an alias
```

The command shows GitHub's device URL and one-time code. It also tries to open
the URL in a browser. `Ctrl+C` cancels login.

When no provider is configured, a successful login selects GitHub Copilot as
the default provider. It never replaces an existing provider choice. Harness
selects an available default model at startup; use `/model` to change it.

Harness stores the short-lived Copilot access token and GitHub OAuth refresh
token in:

```text
~/.config/harness/auth.json
```

On Unix-like systems, Harness creates the auth directory with mode `0700` and
the auth file with mode `0600`. `HARNESS_CONFIG_DIR` and `XDG_CONFIG_HOME`
change this path as described in [Configuration](./configuration.md).
Credentials are never written to `config.toml` or Harness session files.

GitHub Enterprise domains are supported by the auth crate's domain-aware API.
The standalone login command currently uses `github.com`.

### Protocol status

Copilot's device, token, model-policy, and proxy endpoints are client behavior
used by VS Code and related clients. They are not a stable public LLM API.
Harness isolates endpoint paths, headers, policy filtering, and static model
metadata in `crates/auth` and
`crates/llm/src/providers/github_copilot.rs` so that they can change without
leaking into the agent loop.

Harness selects a wire protocol from Copilot's per-model endpoint metadata:

- Claude models use the Anthropic-compatible `/v1/messages` endpoint.
- GPT-5-family and MAI models use `/responses`.
- Other supported models use `/chat/completions`.

Free plans restrict many premium models. If you do not configure a model,
Harness tries to select one that the signed-in plan can use. Plan-gated
requests return an actionable error instead of a bare HTTP 400 response.

## OpenAI Codex

Use browser OAuth by default:

```text
harness login openai-codex
# `codex` is an alias
```

Use device authorization when a local loopback callback is not available:

```text
harness login codex --device-code
```

Credentials are stored in the same private `auth.json` file as Copilot
credentials. When no provider is configured, a successful login selects
`openai-codex` without replacing any existing choice. Harness uses the
provider's default model until you select another through `/model` or
`config.toml`.

Codex uses a dedicated, SSE-only provider and dialect because its ChatGPT
subscription protocol differs from the normal OpenAI API.

## Change the active model

In the terminal UI, select a model from the active provider:

```text
/model <model>
```

Select a provider and model together:

```text
/model <provider>:<model>
```

The model picker fetches or uses the catalog for the selected provider.
Provider and model changes are saved to `config.toml`.
