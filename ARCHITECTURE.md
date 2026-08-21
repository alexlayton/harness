# Architecture

Harness is a terminal coding-agent harness: it drives an LLM through a loop of
streamed completions and workspace-scoped tool calls, renders the result in a
terminal UI (or headless stdout), and persists every event to a durable,
append-only JSONL session log.

The project is a Cargo workspace of 7 crates with a strict dependency
direction. `agent` is the only binary (`harness`); everything else is a
library.

```
agent ──┬── tui      (rendering; no agent dep)
        ├── compact  (compaction policy) ── llm, session
        ├── session  (durable log) ──────── llm
        ├── llm      (providers + dialects)
        ├── auth     (Copilot OAuth; standalone)
        └── tools    (built-in tools) ───── llm
```

Arrows read "depends on". Everything is a library except `agent`, which
builds the `harness` binary.

## Crate map

| Crate | Responsibility | Key public items |
|---|---|---|
| [`agent`] | The `harness` binary and the agent orchestration loop: turn state machine, tool dispatch, retry/recovery, compaction triggers, auth UX, plus two frontends (TUI and headless) | `Agent`, `AgentEvent`, `run_headless`, `Cli`, `Config` |
| [`tui`] | Terminal UI. Renders `UiEvent`s, owns input handling, slash commands, attachments, completion | `CrossTerm`, `InputMessage`, `UiEvent` |
| [`compact`] | Compaction *policy and planning* (when/what to cut, how to summarize); persistence of summaries stays in `session` | `CompactionPolicy`, `plan_compaction`, `summarize` |
| [`session`] | Durable append-only JSONL session log: format, codec, store, export, context reconstruction | `Session`, `SessionStore`, `SessionEvent`, `ExportOptions` |
| [`llm`] | Provider abstraction plus wire-format dialects and provider clients. The only crate that speaks HTTP to model APIs | `Provider`, `CompletionRequest`, `Message`, `StreamEvent` |
| [`auth`] | GitHub Copilot device-flow OAuth, token storage, refresh — independent of `llm`/`agent`/`tui` | `CopilotAuth`, `AuthEvent`, `AuthStore` |
| [`tools`] | The built-in tool set (`read`, `edit`, `write`, `bash`, `find`, `grep`) and the `Tool` trait/registry | `Tool`, `ToolRegistry`, `default_registry` |

Dependency direction is one-way: `agent` may depend on everything; `tui` must
not depend on `agent`; `compact` depends on `llm` and `session`; `session` and
`tools` depend on `llm`; `llm` and `auth` depend only on external crates.

## The main loop

The heart is `Agent::run` in `crates/agent/src/agent.rs`:

1. User input arrives via an mpsc channel of `InputMessage`s (typed text or a
   slash command from either frontend).
2. A *turn* builds a `CompletionRequest` from the session's reconstructed
   context (`Session::context_messages`), the system prompt, and the active
   tool definitions.
3. The `Provider` streams `StreamEvent`s (text/reasoning deltas, tool calls,
   usage). Recoverable failures — malformed tool arguments, dropped streams,
   stalls — re-stream the turn up to a bounded number of times.
4. Tool calls are dispatched through the `ToolRegistry`, results are recorded
   as session events, and the loop repeats until the model produces a final
   assistant message.
5. Every event is appended to the session log as it happens, so a crash loses
   at most the in-flight line.

Compaction is checked around this loop: `compact::plan_compaction` picks a cut
point using token estimates, the agent appends a `CompactionSummary` event
and rebuilds provider history from it. Old events are never deleted.

## Frontends

The interactive UI and the headless runner drive the *same* `Agent` and
differ only in I/O:

- **TUI** (`crates/tui`): direct-crossterm terminal UI — plain rows committed
  into the terminal's native scrollback with a small live region at the bottom.
  The agent's `AgentEvent`s are adapted into `UiEvent`s in
  `crates/agent/src/lib.rs` — this adapter crate is what breaks the would-be
  dependency cycle between `agent` and `tui`.
- **Headless** (`crates/agent/src/headless.rs`): `harness -p "…"`. One prompt,
  run to completion, final answer to stdout, exit. All progress goes to stderr
  behind `-v`. Stdout purity is a hard requirement (pipes/CI).

## Providers and dialects

`llm` separates *what* from *how*:

- `provider.rs` defines the `Provider` trait (`stream`, `list_models`,
  `stream_with_retry`) — the provider-neutral contract the agent sees.
- `dialects/` translates between the neutral message/tool types and each wire
  format: `openai_chat`, `openai_responses`, `anthropic`.
- `providers/` implements endpoint/auth specifics: `openrouter`,
  `github_copilot`, `opencode_go`. Copilot routing is model-aware (Claude →
  `/v1/messages`, GPT-5-family/MAI → `/responses`, others →
  `/chat/completions`).

Adding a provider means: a dialect (if the wire format is new), a provider
client, and registration in `agent/src/config.rs` (`ProviderArg`). Endpoint
URLs, headers, and model-policy metadata must stay inside `llm`/`auth` —
never in `agent`.

## Tools

Tools live in `crates/tools` and implement the `Tool` trait, which pairs a
structured `ToolDefinition` (JSON schema sent to the provider) with `ToolPrompt`
metadata (snippet + guidelines). **The system prompt's tool section is generated
from the same registry snapshot that supplies `CompletionRequest.tools`** —
this is deliberate, so prose and schemas can never drift apart. Never
hand-write tool documentation into the prompt.

All path-taking tools confine resolution to the workspace root captured at
startup; `bash` runs with that root as cwd. `find`/`grep` share one
`FileSearchIndex` (fff-search) created once per process. `bash` optionally
rewrites commands through `rtk` for token-lean output.

## Skills and project context

Both **Agent Skills** discovery and **AGENTS.md** project-context injection live
in `crates/tools` and feed the system prompt:

- `crates/tools/src/skills.rs` — skill discovery, frontmatter parsing, the
  `SkillCatalog`, and the `<available_skills>` prompt XML.
- `crates/tools/src/context_files.rs` — loads and renders AGENTS.md/CLAUDE.md
  as the `<project_context>` block.

Every discovered skill's `SKILL.md` (plus its `scripts/`/`references/`/`assets/`
dirs) is added to `ReadTool`'s allowlist, and the catalog is stored on the
registry for the prompt builder.

### Autodiscovery roots (no explicit paths)

Skills autodiscover with **zero flags or config**: the project walk (`cwd` up
to the git repo root: `.harness/skills` in harness mode and `.agents/skills`
in agents mode), plus the global roots `~/.harness/skills` (or
`$HARNESS_SKILLS_DIR`) and `~/.agents/skills`. Project roots beat global roots;
name collisions keep the earlier (higher-priority) skill. There is deliberately
no `--skill` flag or config `skills` field — the explicit-path mechanism was
removed, leaving pure autodiscovery as the only mechanism; the Agent Skills
standard is the contract.

### The two invariants

1. **System-prompt-lives-outside-history (compaction immunity).** The system
   prompt — including the skills catalog and the `<project_context>` block — is
   rebuilt from scratch every turn and never touches `self.history`. Compaction
   only summarizes history, so this content persists across compaction by
   construction.
2. **Lazy loading (progressive disclosure).** Only skill *name, description,
   and location* ever enter the system prompt. The model loads a skill's body
   by calling `read` on the absolute `<location>` path — the token-efficiency
   point of skills. Skill bodies, once read into history as tool results, are
   compacted like any other tool result (truncated to 2000 chars and folded
   into the summary). That is correct behavior, not something to "fix".

### AGENTS.md context files

`~/.harness/AGENTS.md` (or `$HARNESS_CONFIG_DIR/AGENTS.md`) plus every ancestor
of `cwd` up to the git repo root, nearest last. Per directory, the first
existing candidate wins: `AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`,
`CLAUDE.md`, `CLAUDE.MD`. Files are deduped by canonical path and capped
(32 KiB total, 16 KiB per file). Opt out with `--no-context-files` or
`HARNESS_NO_CONTEXT_FILES`.

## Sessions

Owned by `crates/session`; see `crates/session/README.md` for the full JSONL
format spec. Key invariants:

- Append-only event log; one complete newline-terminated record per write,
  flushed and fsynced. A sidecar lock prevents concurrent writers.
- The first line is a header snapshot; current metadata comes from replaying
  events. Never rewrite history to change a title/model — emit an event.
- `context_messages()` reconstructs provider-ready history, groups tool calls
  with their results, and drops a dangling final tool call rather than send
  invalid history.
- Sessions are workspace-scoped and permission-restricted; the store root can
  be overridden with `HARNESS_SESSION_DIR` / `HARNESS_STATE_DIR`.

## Config and secrets

`~/.config/harness/config.toml` holds provider/model selection
(`HARNESS_CONFIG_DIR` overrides). Copilot credentials live separately in
`~/.config/harness/auth.json` (0600) and are never written to config or
session files. `HARNESS_LOG` enables file-only tracing.

## History and intent

Some historical feature plans live in untracked notes and scratch markdown at
the repo root. They describe intent at time of writing and are **not** a
description of current behavior — verify against the code. ARCHITECTURE.md is
the authoritative map of how the system works today.
