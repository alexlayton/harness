# Architecture

Harness is a terminal coding agent. It streams LLM responses, executes tools,
renders output through one of three frontends, and stores an append-only session
log.

This document is a map of stable boundaries and invariants. For implementation
detail, read the linked module or focused documentation for the subsystem you
are changing.

## Crate map

Harness is a Cargo workspace of nine crates. `harness` builds the only binary;
all other crates are libraries.

```text
harness ──┬── agent ──┬── compact ── llm, session
          │           ├── session ── llm
          │           ├── mcp ────── tools, llm
          │           ├── tools ──── llm
          │           └── llm ────── auth
          └── tui
```

| Crate | Responsibility |
|---|---|
| `harness` | CLI, configuration, authentication UX, logging, frontend adapters, and process orchestration. |
| `agent` | Frontend-independent runtime assembly, agent loop, events, retries, tool dispatch, persistence integration, MCP lifecycle, and subagents. |
| `tui` | Terminal rendering, input, slash commands, and path completion. It must not depend on `agent`. |
| `compact` | Token estimation, compaction policy and planning, and summary generation. |
| `session` | Append-only JSONL format, locking, replay, export, and provider-history reconstruction. |
| `llm` | Provider-neutral types, provider clients, retry behavior, SSE handling, and wire-format dialects. |
| `auth` | Copilot and OpenAI Codex OAuth, refresh, and private credential storage. |
| `tools` | Built-in tools, registry, scheduling metadata, path checks, project context, skills, and the subagent interface. |
| `mcp` | External MCP process lifecycle, tool discovery, namespacing, and tool adapters. |

Dependency direction is one-way. Provider wire formats stay in `llm`; auth
storage stays in `auth`; rendering stays in `tui`. Explicit input/event adapters
in `crates/harness/src/tui_adapter.rs` compose the independent `agent` and `tui`
protocols without a dependency in either direction.

## Request flow

The main loop is `Agent::run` in `crates/agent/src/agent/mod.rs`. Its modules
separate events, persistence, tool scheduling, turn streaming, commands, and
compaction.

1. A frontend sends an `InputMessage`.
2. The agent reconstructs provider history from the current `Session` and
   builds a `CompletionRequest` with the system prompt and tool definitions.
3. The provider streams text, reasoning, tool calls, and usage. Bounded retry
   logic handles recoverable stream and tool-argument failures.
4. The registry schedules tool calls by their declared concurrency class. Tool
   events and results are persisted, then the loop requests the next model
   response.
5. A text-only response completes the turn.

Compaction can run before a request or after a context-overflow response. It
appends a summary event and rebuilds provider history; it never rewrites old
session events.

## Frontends

All frontends use `agent::assembly::AgentBuilder`, so provider, tool, session,
MCP, and subagent setup has one implementation.

- **TUI:** direct Crossterm UI with completed output in native terminal
  scrollback. `tui` receives only provider-independent `UiEvent`s.
- **Headless:** `harness prompt "…"` writes only the final answer to stdout.
  Optional progress goes to stderr behind `-v`; stdout purity is an invariant.
- **ACP:** `harness acp` serves Agent Client Protocol over stdio. Stdout is
  JSON-RPC only and tracing remains file-only. Each ACP session owns an agent,
  registry, and session store rooted at the editor workspace.

See `crates/harness/src/headless.rs`, `crates/harness/src/acp.rs`, and
[`docs/editor-integration.md`](./docs/editor-integration.md) for frontend
protocol detail.

## Providers and authentication

`llm::Provider` is the provider-neutral contract used by the agent. Provider
clients own endpoint selection, headers, model metadata, and response parsing.
Dialects in `llm/src/dialects/` translate neutral messages and tools to the
OpenAI Chat, OpenAI Responses, or Anthropic wire formats.

Adding a provider requires:

1. A dialect when no existing wire format applies.
2. A provider implementation in `llm/src/providers/`.
3. Authentication support in `auth` when required.
4. Registration through `ProviderArg` in `crates/harness/src/config.rs`; the
   runtime receives provider construction as an injected factory.

Do not put endpoint paths, provider headers, or wire-format types in `agent`.
See [`docs/providers.md`](./docs/providers.md) for supported providers and
login behavior.

## Tools, MCP, and subagents

A built-in tool implements `tools::Tool`, which combines its JSON definition,
prompt metadata, concurrency class, and executor. The system prompt and
`CompletionRequest.tools` are generated from the same immutable
`ToolRegistry` snapshot; never maintain a second hand-written tool list.

Dedicated path tools confine resolution to the workspace and reject lexical or
symlink escapes. The shell starts in the workspace but is not a sandbox; it can
access anything available to the operating-system user. File mutations also
use process-local locks to prevent overlapping writes.

`mcp` starts configured stdio servers during assembly, discovers their tools,
namespaces them, and registers adapters in the same registry. MCP calls are
exclusive. Tool-list changes require a new connection, and MCP tools are not
passed to subagents.

The subagent schema lives in `tools`, while its runner lives in `agent` to
preserve dependency direction. Important invariants are:

- Read-only children receive only `read`, `find`, and `grep`; unavailable tools,
  not prompt wording, enforce the restriction.
- Workspace children can use normal built-ins and run exclusively.
- Children cannot create subagents.
- Each child gets fresh model context and a bounded turn count.
- Tool call IDs remain stable through scheduling, UI events, and persistence.
- Child sessions link to their parent, but child usage is not added to parent
  totals.

Implementation details are in `crates/agent/src/subagent.rs`,
`crates/tools/src/subagent.rs`, and `crates/mcp/src/`.

## Project context and skills

`crates/tools/src/context_files.rs` loads the global instruction file and the
first matching `AGENTS.md`/`CLAUDE.md` candidate in each directory from the Git
root to the workspace. Nearest files have highest priority. Content is capped
and can be disabled with `--no-context-files` or
`HARNESS_NO_CONTEXT_FILES`.

Skills are discovered from project `.harness/skills/` and `.agents/skills/`
roots plus their global equivalents. Only skill name, description, and path are
put in the system prompt. The model reads a skill body only when needed.

The system prompt is rebuilt on every turn and remains outside session history.
Therefore project instructions and the skill catalogue survive compaction,
while skill content read through a tool is normal compactable history.

See [`docs/configuration.md`](./docs/configuration.md) and
`crates/tools/src/skills.rs` for precedence, limits, and environment overrides.

## Sessions

`session` owns durable state. Its key invariants are:

- One JSON object per line, with a header first and append-only events after it.
- Metadata changes are events; existing history is never rewritten.
- Appends are flushed and normally synced, with a sidecar lock preventing
  concurrent writers.
- `context_messages()` reconstructs valid provider history and handles
  incomplete final tool calls.
- Sessions are grouped by workspace and stored outside the project by default.

See [`crates/session/README.md`](./crates/session/README.md) for the format and
`crates/session/src/store.rs` for locking and durability behavior.

## Configuration and secrets

Normal configuration is in `~/.config/harness/config.toml`, with
`HARNESS_CONFIG_DIR` as an override. OAuth credentials are stored separately in
`auth.json`; API keys remain environment variables. Session roots can be
changed with `HARNESS_SESSION_DIR` or `HARNESS_STATE_DIR`. `HARNESS_LOG` enables
file-only tracing.

Do not write API keys or OAuth credentials to configuration, session history,
tool output, or logs.

## Reading guide

For a focused change, read the crate map and only the relevant section above,
then inspect these sources:

| Change | Start here |
|---|---|
| Agent loop or scheduling | `crates/agent/src/agent/` |
| TUI behavior | `crates/tui/src/app.rs`, `render.rs`, `commands.rs` |
| Provider or wire format | `crates/llm/src/providers/`, `dialects/` |
| Authentication | `crates/auth/src/` |
| Tool behavior or path safety | `crates/tools/src/` |
| MCP integration | `crates/mcp/src/` |
| Compaction | `crates/compact/src/` and `crates/agent/src/agent/compaction.rs` |
| Session durability or format | `crates/session/src/` and `crates/session/README.md` |
| ACP | `crates/harness/src/acp.rs`, `docs/editor-integration.md` |

Historical scratch notes are not authoritative. Verify detailed behavior
against source and tests.
