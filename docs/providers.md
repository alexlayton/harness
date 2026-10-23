# Providers and authentication

Harness supports five built-in providers and any number of
[OpenAI-compatible endpoints](#openai-compatible-endpoints) that you define.
Select one with `--provider`, in `config.toml`, or with `/model` in the
terminal UI. If none is selected,
Harness prints setup instructions instead of assuming an API-key provider.

| Provider | Configuration name | Authentication |
|---|---|---|
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` or `OPENCODE_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| GitHub Copilot | `github-copilot` (`copilot`) | Device login |
| OpenAI Codex | `openai-codex` (`codex`) | Browser or device login |
| Anthropic Claude | `anthropic-claude` (`claude`) | Browser login |

API keys are environment-only secrets. Harness does not write them to
`config.toml` or session files. Without an explicit model, the configured
fallbacks are `gpt-5.6-luna` for OpenCode Go, `openai/gpt-5.6-luna` for
OpenRouter, `gpt-5.4` for Copilot, and `gpt-5.5` for Codex. Copilot can instead
select a model available to the signed-in account.

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
For example, use `/model gpt-6-astra`, `/model gpt-6-luna`, or
`/model gpt-6.1-sol` if the signed-in account offers them. The picker shows
only models available to that account; GitHub can also restrict models by plan
or organization policy.

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

Harness selects a wire protocol from its per-model catalog, not only from the
model name. Most Claude models use the Anthropic-compatible `/v1/messages`
endpoint, but `claude-fable-5` uses `/chat/completions`. GPT-5-family and MAI
models use `/responses`, as does `grok-4.5`. Other catalogued models can use
`/chat/completions`.

Free plans restrict many premium models. If you do not configure a model,
Harness tries to select one that the signed-in plan can use. Plan-gated
requests return an actionable error instead of a bare HTTP 400 response.

## OpenAI Codex

Use browser OAuth by default:

```text
harness login openai-codex
# `codex` is an alias
```

On a headless server or over SSH, use device authorization instead:

```text
harness login codex --device-code
```

Open the printed URL on another device and enter the printed code. Harness
waits for approval on the server, then saves the credential there. This flow
does not need a local browser or an inbound callback port. OpenAI must permit
device-code sign-in for the account.

Credentials are stored in the same private `auth.json` file as Copilot
credentials. When no provider is configured, a successful login selects
`openai-codex` without replacing any existing choice. Harness uses the
provider's default model until you select another through `/model` or
`config.toml`. For example, use `/model gpt-6.1-sol` to select GPT-6.1 Sol.

Codex uses a dedicated, SSE-only provider and dialect because its ChatGPT
subscription protocol differs from the normal OpenAI API.

## Anthropic Claude

Use your Claude subscription with browser login:

```text
harness login anthropic-claude
harness --provider anthropic-claude
```

This is an unofficial subscription protocol. It can violate provider terms
and put your account at risk. Harness does not fall back to API-key billing.
The default model is `claude-sonnet-4-6`.

A saved Claude login makes the local model catalog available in `/model`,
even when another provider is active. For example:

```text
/model anthropic-claude:claude-sonnet-4-6
```

The catalog does not confirm account access or live model availability.
The older names `anthropic-subscription` and `claude-sub` remain valid aliases.
Existing saved credentials still work; you do not need to log in again.

## OpenAI-compatible endpoints

Define a named provider under `[providers.<name>]` to use any server that
supports the OpenAI Chat Completions API, such as LM Studio, Ollama, vLLM, or a
gateway. Then select it by name:

```toml
provider = "lmstudio"
model = "qwen3-coder-30b"

[providers.lmstudio]
type = "openai-compatible"            # required
base_url = "http://localhost:1234/v1"   # required; API root, including /v1
api_key_env = "LMSTUDIO_API_KEY"        # optional; omit when no key is needed
default_model = "qwen3-coder-30b"       # optional; used when `model` is unset
context_window = 32768                  # optional; see below

[providers.lmstudio.headers]            # optional literal, non-secret values
X-Client = "harness"

[providers.lmstudio.env_headers]        # optional; header = environment variable
X-Api-Token = "MY_GATEWAY_TOKEN"
```

```text
harness --provider lmstudio
/model lmstudio:<model>
```

Harness sends requests to `{base_url}/chat/completions` and reads the model
list from `{base_url}/models`. When `api_key_env` is set, its value is sent as
`Authorization: Bearer <key>`; otherwise no `Authorization` header is sent.
Secrets stay out of `config.toml`: `api_key_env` and `env_headers` name
environment variables, and their values are masked like
[`[secrets]`](./configuration.md#secret-masking) entries.

Local servers usually do not report context lengths. Set `context_window` so
that compaction starts at the right time; a value reported by `/models` takes
precedence, and `[compaction] context_window` overrides both.

Rules:

- A name can contain ASCII letters, digits, `-`, `_`, and `.`. It cannot be a
  built-in provider name or alias.
- `base_url` must be an `http://` or `https://` URL.
- `Content-Type` cannot be overridden. Use either `api_key_env` or an
  `Authorization` header, not both.
- Harness sends only the standard Chat Completions request. It does not send
  provider-specific reasoning controls, so `reasoning_effort` has no effect.

## Reasoning effort

Set a portable reasoning preference in `config.toml` or on the command line:

```toml
reasoning_effort = "high"
```

```text
harness --reasoning-effort high
```

The interactive command supports inline suggestions and Tab completion:

```text
/reasoning
/reasoning high
```

Supported values are `auto`, `off`, `minimal`, `low`, `medium`, `high`, and
`maximum`. OpenAI Responses and Codex translate explicit levels directly;
`maximum` uses their `xhigh` wire value. OpenRouter uses its unified reasoning
extension. Generic Chat-compatible and Anthropic-compatible routes remain
conservative when their endpoint does not advertise a portable effort control;
models can still reason according to their own defaults. Explicit unsupported
values may be rejected by the provider rather than silently downgraded.

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
