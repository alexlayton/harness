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
reasoning_effort = "auto"
rtk = false
```

The command line overrides `provider`, `model`, and reasoning effort for the
current process:

```text
harness --provider openrouter --model openai/<model> --reasoning-effort high
```

Reasoning accepts `auto`, `off`, `minimal`, `low`, `medium`, `high`, and
`maximum`. `auto` preserves the provider/model default. Named effort is a
portable preference: supported models translate it to their native control,
while providers may reject levels unavailable for a selected model. The TUI
`/reasoning` command displays or changes the active value and persists changes.

Harness requires a provider choice rather than assuming one from an API key.
When no provider is configured, startup lists the available providers and their
setup commands. A successful OAuth login selects that provider only when this
setting is absent.

The `/model` terminal command updates the saved provider and model. Harness
preserves configuration keys that it does not know when it saves these values.
Use `harness --resume-session ID` to reopen a persisted interactive session;
headless runs use `harness prompt --resume ID PROMPT`.

## Herdr TUI integration

This is for the original Herdr TUI. The Herdr desktop app uses
[program status (OSC 7501)](#program-status-osc-7501) instead.

When `HERDR_ENV=1` and `HERDR_PANE_ID`, `HERDR_BIN_PATH`, and
`HERDR_SOCKET_PATH` are all set, Harness reports pane state through the Herdr TUI
CLI at `HERDR_BIN_PATH`. Reports use the stable `harness` source and `Harness`
agent name, report working as soon as a turn starts and idle when ready for
input, and attach the actual agent session ID. Resume argv uses the executable
name `harness` with the active provider, model, reasoning effort, session ID,
and any `--no-context-files`, `--defer-session-sync`, or `--mcp` options. Herdr TUI 0.9.2 or
newer is required to consume the resume argv after `--`. Arguments are checked
against Herdr TUI's limits (64 arguments, 8 KiB, and no apostrophes or control
characters). When launched through `harness worktree BRANCH`, the resume argv
re-enters the same worktree (`worktree BRANCH --dir PATH`), because Herdr TUI
restores the pane in the launch directory and sessions are scoped to the
workspace directory. Repeated reports are coalesced, and when several changes
queue while a Herdr TUI call is running, only the newest state is sent, as Herdr TUI's
guide asks. The session title is sent as display-only pane metadata
(`pane report-metadata --title`) and cleared on exit. Reports are best-effort and use a short timeout; failures do not affect
Harness output and are logged when `HARNESS_LOG` is set. Outside a fully
configured Herdr TUI pane, no integration is activated. TUI and headless lifecycles
report state and release the pane on process exit. ACP remains independent
because it multiplexes multiple editor sessions over a single JSON-RPC process.
The bash tool and its optional `rtk` rewriter remove inherited Herdr TUI pane
variables from child processes, so a nested Harness cannot claim the parent's
pane by accident.
This does not prevent a command from setting those variables itself. See
[Herdr TUI's integration guide](https://herdr.dev/docs/add-herdr-support/) for
Herdr TUI-side behavior and requirements.

## Program status (OSC 7501)

Harness reports what it is doing with the
[Program Status Protocol](https://www.superlogical.com/rex/docs/build/program-status)
(`OSC 7501`), so terminals and multiplexers such as the Herdr app can show
whether it is working, finished, or failed. Terminals that do not know the
sequence ignore it. Reports are on by default; turn them off with:

```toml
program_status = false
```

Every report carries `app=harness` and the session title. States:

| Event | State |
| --- | --- |
| A turn or command starts, or output streams | `working` |
| A turn finishes | `done` (the result is ready to look at) |
| A turn fails | `error`, with a short `msg` |
| The user cancels with Esc | `idle` |
| The TUI exits | `clear` (no stale `done` is left behind) |

Harness has no approval gate, so it never reports `blocked`.

In the TUI a report is written at the start of a frame, so it cannot split the
frame's escape sequences. In `harness prompt` the report goes to stderr, and
only when stderr is a terminal, so stdout stays the answer only. A headless run
leaves a `done` or `error` record on exit, as the protocol recommends. Inside a
Herdr TUI pane Harness sends only the Herdr TUI reports described above.

Set `rtk = true` to let the bash tool rewrite supported commands through an
installed `rtk` executable for smaller tool output. This feature is off by
default.

## Built-in tools

`find`, `grep`, and `multigrep` search a shared, watched workspace index.
`outline` uses Tree-sitter to show declarations and line ranges in supported
source files (Rust, Python, Swift, Go, C, C++, JavaScript, and TypeScript).
Use `read` for the relevant lines after finding a declaration.

`python` runs standalone Monty snippets for calculations and data
transformations. Each call starts with fresh state and has a five-second
execution budget. It does not provide third-party packages, filesystem access,
or host callbacks; use the dedicated file tools for files and bash for commands,
builds, and tests. Monty runs in the Harness process and is **not** a security
boundary or crash-isolated sandbox. No configuration is needed to enable these
tools.

## Codemode

Codemode lets the model write one Python program that calls selected tools and
filters their results. Enable it in `config.toml`:

```toml
[codemode]
enabled = true
```

It is off by default. The older `[code_mode]` table name is still read. The setting applies to the terminal UI, headless mode,
and ACP. Restart Harness to change the selection. With no `tools` list, all
eligible built-in and connected MCP tools are selected. To select fewer tools:

```toml
[codemode]
enabled = true
tools = ["read", "find", "grep", "outline"]
```

Use the registered, namespaced names for MCP tools. Unknown or excluded names
stop startup with an error. `bash`, `python`, `subagent`, and `codemode` cannot
be selected. `bash` and `subagent` stay available as normal model calls when
registered. Codemode replaces the standalone `python` tool: its definition and
prompt guidance are hidden, and calculations run directly in `codemode` without
calling any tools. With codemode disabled, `python` remains available normally.
Other tools not in an explicit selection also remain normal model calls.
Subagents keep their existing tool sets, including standalone `python` for
workspace children; they do not receive codemode or the parent's MCP tools.

Selected tools are removed from the model-facing native tool list. The
`codemode` description lists each one as a compact signature and first sentence
(for example `read(path, offset?, limit?)`), within a small byte budget; tools
past the budget appear by name only. Full descriptions, parameter schemas, and
output schemas stay inside Harness and are not sent with every request. A
snippet reads them on demand through three globals, which are host metadata
and do not count as tool calls: `TOOLS` (name to summary), `describe(name)`
(full description, parameters, and output schema), and `MODULES` (available
and unavailable Python modules). Catalogue construction stops if its
serialized JSON exceeds 512 KiB; select fewer tools if startup reports this
limit. Inside a snippet, each function is available
through `tools["name"]`. This dictionary
preserves MCP names that contain characters which are not valid in Python
identifiers. Use keyword arguments, including `**{"class": value}` when a
parameter name is a Python keyword.

For example, the model can send this source in `codemode`, with a short
`description` label:

```python
notes = {}
for path in ["README.md", "ARCHITECTURE.md"]:
    text = tools["read"](path=path, offset=1, limit=80)
    notes[path] = [line for line in text.splitlines() if "tool" in line.lower()][:5]
notes
```

Calls are **synchronous and sequential**, including MCP calls. Do not use
`await` or `asyncio.gather` for tool functions in this version. The last
expression is the result; printed text is supplementary output. Each snippet
starts with fresh state. There is no persistent REPL.

MCP structured results reach Python as dictionaries, lists, or scalar values.
MCP output schemas are included when the server supplies them. Structured MCP
results larger than 20 KiB are omitted and the function returns bounded display
text instead. Tools without structured results return strings; do not assume
that display text is JSON. Arguments must use JSON-compatible values, finite
numbers, string dictionary keys, and signed 64-bit integers. Values are limited
to 64 levels of nesting. A failed tool raises `RuntimeError` at its call site.

Each snippet has these fixed limits:

- 32 KiB of Python source.
- 50 started nested tool calls and 1,000 interpreter suspensions.
- 50 KiB for each argument object, result, and combined displayed output,
  plus a short truncation marker where needed.
- Five seconds of interpreter execution, excluding tool wait time.
- 120 seconds of elapsed time for the complete workflow.

The complete workflow is exclusive. Nested calls use the existing workspace
and tool restrictions without acquiring the same execution lock twice. Secret
placeholders are restored only in eligible local-tool arguments, never in the
complete Python source or in external MCP arguments. Results and nested audit
records are masked before use.

Nested calls produce live tool events and separate `code_mode_call` /
`code_mode_result` session audit records. These records do not enter model
history. A nested start is persisted before execution; if persistence fails,
the callback does not run. Deferred-sync sessions also sync these records
before acknowledgement. An unfinished start means **execution status unknown**.
On failure, the result reports started calls and their known statuses. Earlier
changes are not rolled back. Inspect partial results before repeating code,
especially after interruption or timeout.

Python has no direct filesystem, environment, network, or host-clock access;
file access goes through the selected tools. As with `python`, Monty runs in
process: it is not crash-isolated, and there is no enforced memory sandbox.
A cancelled interpreter step can finish computing up to its execution limit,
but it cannot start host tools. MCP servers retain their normal external
access; codemode does not make those servers a security sandbox.

## Secret masking

Harness can replace explicitly configured environment-variable values before
conversation content is sent to a model or stored in a new session event:

```toml
[secrets]
env = ["DEPLOY_TOKEN", "AWS_SECRET_ACCESS_KEY"]
```

Harness resolves these variables once at startup. Startup fails if a configured
variable is missing, is not valid Unicode, or contains fewer than eight bytes.
The configuration stores names only. Resolved values stay in process memory and
are not included in debug output.

The model sees stable placeholders such as
`{{harness-secret:DEPLOY_TOKEN}}`. Harness can restore known placeholders in a
temporary argument copy for built-in local tools. The tool call kept in session
history stays masked. Harness does not restore placeholders for subagent prompts,
MCP tools, or other external tools. Tool results are masked before they reach the
UI, model history, or session store. Harness does not modify opaque provider
continuation state because changing it can make it invalid. It drops a complete
opaque item if that item contains an exact configured value.

Harness supports secret placeholders in JSON values, but not in JSON object
keys. Changing a key can merge two fields. Harness replaces an object that has
a secret-bearing key with a safe rejection marker and does not execute a local
tool call that contains that marker. Harness also omits a tool definition or
model-list entry if its routing name contains a configured value. Other
provider-supplied labels and protocol identifiers are masked before display or
persistence. Harness rejects a configured provider or model name that contains
a secret because those routing values cannot be changed before use.

Masking uses exact string matches. It does not detect encoded, split, derived,
or unknown credentials. It also is not a sandbox. In particular, `bash` can
read the process environment and use the network without first returning a
secret to Harness. Use this feature to reduce accidental provider and session
disclosure, not to defend against a malicious model or prompt injection.

Existing append-only session files are not rewritten. When an old session is
loaded, Harness masks its provider and UI views in memory, but any plaintext
that was already written remains in the JSONL file.

## Terminal UI

To start directly at the input field without printing the welcome banner and
workspace/provider metadata, enable minimal mode:

```toml
[tui]
minimal = true
```

Turn activity, usage, completion, and session output are unchanged.

## Prewalk

Prewalk is a one-shot handoff from the active (planning) model to a faster or
cheaper target model. The planning model investigates the workspace, writes a
short numbered plan in its visible reply, and makes the first change. After the
first tool round that calls `edit` or `write` (also inside `codemode`, and also
when the call fails), Harness switches to the target model and tells it to
continue the plan. Other tools, including `bash`, do not end the planning phase.

```toml
[prewalk]
enabled = false
model = "openrouter:openai/<small-model>"
reasoning_effort = "low"
```

`model` is `[<provider>:]<model>`, in the config, on the command line, and with
`/prewalk-model`. A prefix is a provider only when Harness knows it, so a model
id such as `vendor/model:free` keeps its colon. Without a provider, the handoff
keeps the active provider. `enabled = true` requires
`model`. `reasoning_effort` accepts the same values as the top-level setting;
without it, the target keeps the active reasoning effort. A handoff to the same
model still applies a different reasoning effort.

The preference arms each new conversation, not resumed sessions. The handoff
records a normal session model change, but it does not replace the saved
`/model` or `/reasoning` choice. While prewalk is enabled, `/new` after a
handoff returns to the planning model and reasoning effort and arms again.
`/prewalk on` after a handoff does the same in the current conversation. Both
skip the return when you used `/model` or `/reasoning` after the handoff, and
the return does not change the saved choice either. Subagents started after
the handoff use the target model.

| Command | Effect |
|---|---|
| `/prewalk` | Show the current state and target. |
| `/prewalk on` | Arm the handoff for this conversation and save `enabled = true`. After a handoff, first return to the planning model. Saves the target only when none is saved. |
| `/prewalk off` | Cancel a pending handoff and save `enabled = false`. It does not switch back after a handoff. |
| `/prewalk-model [<provider>:]<model>` | Set and save the target, with the same completion as `/model`. A pending handoff uses the new target. |
| `/prewalk-reasoning [inherit\|<level>]` | Set and save the target reasoning effort. `inherit` removes it, so the handoff keeps the active effort. |

Use prewalk for tasks where the hard work is finding the approach and the rest
is bounded. Keep it off for tasks that need strong reasoning during
implementation, such as subtle concurrency or lifecycle changes. ACP sessions
do not arm prewalk.

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
  provider metadata, then a generous 1M fallback if metadata is unavailable
  (some providers report no context lengths).

`threshold` must be finite and between `0.0` and `1.0`. Summary byte limits
must be nonzero and no larger than 16 MiB for input or 1 MiB for output.
`reserve_tokens` must be smaller than the effective context window: an
explicit nonzero `context_window` is checked directly, and an explicit
window also bounds the default reserve (so `context_window = 8000` alone
is rejected). Zero `keep_recent_turns` and `keep_recent_tokens` are
allowed: they request the smallest valid retained provider-history tail.
Invalid values identify their exact `[compaction]` key and prevent startup.

Compaction asks the active model for a bounded summary and uses a deterministic
local fallback when that request fails. It appends a summary event; it does
not rewrite or delete old session events.

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
Read-only children receive exactly `read`, `outline`, `find`, `grep`, and `multigrep`;
workspace children receive the normal built-ins. Exclusion from the registry
is the enforcement, not prompt wording.

## MCP servers

Harness connects external stdio and Streamable HTTP MCP servers from
`config.toml`. Manage them from the command line with:

```text
harness mcp
harness mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem .
harness mcp add remote --url https://example.com/mcp
harness mcp add private --url https://example.com/mcp \
  --header 'Authorization=Bearer ${MCP_TOKEN}'
harness mcp delete filesystem
harness mcp add email --disabled --url https://example.com/mail-mcp
harness mcp enable email
harness mcp disable email
```

`harness mcp` and `harness mcp list` list the configured servers. For stdio,
the `--` separates Harness options from flags passed unchanged to the server
executable. Servers are launched directly, so add the executable and each
argument as a separate command-line value rather than supplying one shell
command string. Repeat `--header NAME=VALUE` to configure HTTP request headers.
The commands update `config.toml` atomically and preserve unrelated settings.

### Per-session servers

A server with `enabled = false` is configured but not connected by default.
A launch connects it only when it names the server with `--mcp`; repeat the
flag for several servers:

```text
harness mcp add email --disabled --url https://example.com/mail-mcp
harness --mcp email
harness --mcp email --mcp calendar prompt "Summarize today's mail"
```

```toml
[[mcp.servers]]
name = "email"
transport = "http"
url = "https://example.com/mail-mcp"
enabled = false
```

`harness mcp enable <name>` and `harness mcp disable <name>` toggle the
setting, and `harness mcp list` marks opt-in servers. Selection happens before
`${ENV_VAR}` expansion, so an opt-in server's credentials are required only in
launches that request it. An unknown `--mcp` name is an error. Servers connect
once per process, so `/new` and `/resume` keep the launch's selection. A Herdr TUI
resume command repeats the same `--mcp` flags.

The equivalent stdio configuration is:

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

Streamable HTTP uses one endpoint and supports both JSON and SSE responses:

```toml
[[mcp.servers]]
name = "private"
transport = "http"
url = "https://example.com/mcp"

[mcp.servers.headers]
Authorization = "Bearer ${MCP_TOKEN}"
```

Harness expands `${ENV_VAR}` placeholders in stdio arguments/environment and
HTTP URLs/header values at startup. A missing variable stops startup. Expanded
secrets are not written back to configuration or session history. Prefer
`https://` for non-local endpoints and keep bearer tokens in environment
variables rather than literal configuration values.

Servers run directly, not through a shell, and use the workspace as their
working directory. Server stdout is reserved for MCP framing. Server stderr
does not enter headless or ACP stdout.

MCP tools are discovered when the session starts. Reconnect to apply server
tool-list changes. Calls are serialized as exclusive operations and run
without a confirmation step. Subagents do not receive MCP tools.

Streamable HTTP first uses the draft `2026-07-28` discovery and request-metadata
flow, then falls back to the initialized `2025-11-25` Streamable HTTP protocol
for compatible older servers. If a legacy server closes the discovery request
instead of returning a JSON-RPC rejection, Harness reconnects before attempting
legacy initialization. Both JSON and request-scoped SSE responses are accepted.
The deprecated HTTP+SSE transport is not supported. ACP clients may
declare stdio or HTTP servers; legacy SSE and MCP-over-ACP entries are rejected
before any connection is attempted.

MCP has fixed safety limits rather than per-server TOML overrides: initialize
and catalogue requests have 15-second deadlines, calls have a 60-second
deadline, and shutdown has a four-second global deadline. ACP session assembly
uses its own 35-second bound so sequential initialize and catalogue requests
can both use their full MCP deadlines; this is separate from the short ACP
session shutdown/delete cleanup bound. A server may expose
at most 256 tools with at most 512 KiB of aggregate definitions. Schemas are
limited to depth 32, 10,000 nodes, 64 KiB strings, and 256 KiB total size.
Structured/text/error output is compacted (never pretty-printed) and capped
at 20 KiB with a truncation notice; when structured and text carry the same
payload only one representation is kept. Every newline-delimited stdio frame
and HTTP SSE event is capped at 1 MiB before rmcp deserializes it, including
initialization/discovery, pagination, calls, and protocol errors. Stderr is read
in bounded 4 KiB chunks, counted but discarded by default, and never logged
with secrets.

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

With `--no-session`, Harness does not create a session store, header, event
file, or resumable history. Automatic compaction is disabled because there is
no durable session to append a summary to; `/compact` reports that it is
unavailable, and an over-window provider request cannot recover by compacting
the ephemeral history.

## Shell execution

The shell starts in the workspace but is not a sandbox. Bash calls are
exclusive, use a 120-second default timeout (maximum 86,400 seconds), and cap
output at 2,000 lines or 50 KiB. On Linux, when a writable cgroup-v2 hierarchy
is available, each command gets a private cgroup before the shell is executed.
Timeout, cancellation, future drop, and shell exit use `cgroup.kill` plus the
process group (SIGTERM, escalating to SIGKILL after a 500 ms grace period), so
even a `setsid` descendant is terminated before output draining and cannot
mutate the workspace after tool return. Hosts without cgroup-v2 delegation
fall back to process-group cleanup; that path cannot contain a descendant that
calls `setsid`. macOS has the same process-group limitation, and other
platforms only guarantee direct child termination. Standard output and error
drain concurrently under one shared one-second deadline, which bounds Harness
waiting even on those best-effort paths.

## Project context and skills

Harness loads project instructions from `AGENTS.md` and `CLAUDE.md` files. It
walks from the current workspace to the Git root. It also checks
`~/.harness/AGENTS.md`, or `$HARNESS_CONFIG_DIR/AGENTS.md` when that variable is
set. Use `--no-context-files` or `HARNESS_NO_CONTEXT_FILES` to disable this
context.

Skills are discovered from project `.harness/skills/` and `.agents/skills/`
directories, plus global `~/.harness/skills` and `~/.agents/skills` locations.
`HARNESS_SKILLS_DIR` overrides the global Harness skills directory.
Symlinks inside a skills directory work when their target is in the same
directory tree. Harness ignores a symlinked skill file or skill directory
that points outside it, and the same rule applies to `AGENTS.md` and
`CLAUDE.md` files.

See [Architecture](../ARCHITECTURE.md#project-context-and-skills) for discovery
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
| `acp` | Serve ACP over stdio (stdout carries JSON-RPC protocol traffic only). |
| `worktree <branch>` | Run in a dedicated Git worktree (see the [README](../README.md#usage-overview)). |
| `mcp` | List, add, or delete MCP servers (see [MCP servers](#mcp-servers)). |
| `login <provider>` | Authenticate with an OAuth provider. |
| `--provider <provider>` | Override the configured provider. |
| `--model <model>` | Override the configured model. |
| `--reasoning-effort <level>` | Override reasoning (`auto`, `off`, `minimal`, `low`, `medium`, `high`, or `maximum`). |
| `--prewalk[=BOOL]` | Arm (or with `=false`, suppress) [prewalk](#prewalk) for the startup conversation, also when resumed. |
| `--prewalk-model <model>` | Override the prewalk target for this process and arm the startup conversation. |
| `--prewalk-reasoning-effort <level>` | Override the reasoning effort that prewalk hands off with. |
| `-v`, `--verbose` | In prompt mode, write reasoning and tool progress to stderr. |
| `--resume <selector>` | Resume a session by ID, unique prefix, `latest`, or path in prompt mode. |
| `--no-session` | Do not persist a prompt-mode run. |
| `--defer-session-sync` | Sync session records at turn boundaries. |
| `--no-context-files` | Disable project instruction files. |
| `--mcp <name>` | Also connect a server marked `enabled = false`; repeatable. |
