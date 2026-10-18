# Using Harness from other languages

## Scope and conclusion

This records the initial source-only feasibility study. No native extension,
WebAssembly build, benchmark, or target compatibility test was run during that
study. A native Python MVP was added afterward; see the
[Python library guide](../python/README.md) for the implemented `harness.run`
API. The remaining text describes the repository before that MVP; its broader
SDK interfaces remain proposals.

**A native Python import is feasible without a rewrite of the agent loop.**
Use PyO3 to expose Rust objects to Python, and maturin to build and package the
extension. WebAssembly is not a shortcut to this result: the complete runtime
needs native operating-system services.

For a script that needs Harness now, keep the process interface. Use
`harness prompt` for one final answer, or `harness acp` for structured events
and a long-lived process. A Python package can hide either process interface
behind an import. An import alone does not mean execution is in-process.

## Options

| Approach | Present support | Main work | Assessment |
|---|---|---|---|
| Python wrapper around `harness prompt` | CLI exists | Python process wrapper | Smallest option for one-shot scripts. |
| Python wrapper around `harness acp` | Structured protocol exists | ACP client, event conversion, process lifecycle | Good option for streaming and several turns. |
| Native Python extension | Rust libraries exist; no Python export layer | PyO3 bindings, runtime ownership, stable SDK interface, wheels | Recommended route for true in-process use. |
| Bindings for other native languages | Rust libraries exist; no foreign-language ABI | Shared SDK plus bindings for each language | Feasible after the SDK boundary is defined. |
| Browser WebAssembly | No explicit target support found | Dependency separation and host-backed services | Large architectural change for a reduced runtime. |
| WASI | No explicit target support found | Target/runtime assessment and host capabilities | Not a drop-in build of the existing runtime. |

## Why native Python is practical

The [crate map](../ARCHITECTURE.md#crate-map) already separates the agent from
the terminal frontend. Only `harness` is a binary. The other workspace crates
are Rust libraries, but the workspace currently disables package publication.

The useful entry point is
[`agent::assembly::AgentBuilder`](../crates/agent/src/assembly.rs), not the CLI
`main` function. It accepts a provider, model, tool registry, and cancellation
token. It also accepts project context, session state, secret policy,
compaction policy, and MCP settings.

`AssembledAgent::run` accepts channels for
[`InputMessage` and `AgentEvent`](../crates/agent/src/agent/events.rs). They
already cover prompts, interrupts, text deltas, tool activity, usage, sessions,
and completion boundaries. The caller supplies the Tokio runtime. Durable
sessions and MCP connections are optional. Subagents are enabled by default;
a minimal binding must explicitly disable them if they are outside its scope.

Tool construction accepts an explicit workspace path. The ACP frontend already
has code to assemble separate workspace sessions and own their agent tasks.
This gives an implementation example for an SDK handle; see
[`build_session_stack` and `spawn_agent`](../crates/harness/src/acp.rs).

A native extension would remove the **Harness subprocess**. It would not remove
child processes used by Bash or stdio MCP tools. Those tools must be excluded
if the application requires no process creation.

## What a native SDK still needs

### 1. A library configuration boundary

Provider selection, configuration resolution, and provider factories are in
[`crates/harness/src/config.rs`](../crates/harness/src/config.rs), a private
module of the binary. The agent builder expects these inputs to be resolved.

A small proof of concept can construct one provider directly. A reusable SDK
should share configuration and provider assembly with the CLI. It should accept
explicit workspace paths, credentials or auth handles, state roots, context
policy, and MCP settings. Keep CLI argument parsing, signals, terminal output,
and logging installation outside the SDK.

Do not change Python's current directory or mutate its environment to configure
an agent. Most core paths can already be explicit, but skill/context discovery
still reads global environment or home settings. Session exports resolve
relative paths against the process directory. These defaults need an explicit
policy for embedded use.

### 2. Runtime and object ownership

Use a long-lived native runtime. Do not create and destroy a Tokio runtime for
each prompt. Release the Python GIL during blocking waits. For asynchronous
methods, bridge Tokio futures to Python `asyncio`, for example with
`pyo3-async-runtimes`.

The SDK handle should own its input sender, event receiver, cancellation token,
and agent task. Provide explicit `close` and `aclose` operations. A Python
finalizer alone cannot reliably complete asynchronous cleanup.

Closing the input channel drains an active turn; it is not immediate
cancellation. A turn interrupt and complete runtime shutdown are different
operations. Await the task so MCP cleanup can finish. Only one live agent can
own a given durable session.

The default HTTP client is process-global
([`streaming_client`](../crates/llm/src/http.rs)). Repeated runtime teardown and
Python interpreter shutdown need a focused lifecycle check. Start with normal
CPython support; do not promise subinterpreter support without validation.

### 3. Stable results and events

Do not expose internal Rust enums as a permanent foreign-language contract.
Define SDK result and event types with explicit success, cancellation, and
error outcomes. `TurnFinished` is a lifecycle boundary, not proof of success.
The current events also do not have prompt request IDs or a serialized schema.

The final answer is not the concatenation of every `TextDelta`. A model can
emit intermediate text before tool calls. Preserve the final-answer rules in
[`headless.rs`](../crates/harness/src/headless.rs), but put shared result
assembly in a library rather than calling a terminal output function.

Input and event channels are currently unbounded. Define an event buffering
limit and behavior when a consumer is slow or disappears. Reject or serialize
concurrent prompts on one handle; use separate handles for parallel agents.
A failed event send does not currently stop the agent automatically.

### 4. Packaging and safety

Build native wheels for each supported operating system and CPU architecture,
or provide a supported source-build path. PyO3's limited-API support may reduce
Python-version wheel variants, subject to the APIs selected. Start with the
native platforms Harness already uses; bindings do not repair existing
platform limits. Workspace read/write/edit operations currently fail closed
on non-Unix platforms.

In-process use shares memory and failure risk with Python. A subprocess remains
useful for fault isolation and forced termination. Neither interface is a
security sandbox: Bash retains the operating-system access of the host user.
The current ACP interface runs tools without a permission confirmation step.

Do not assume a large speed increase. In-process use removes process startup
and protocol overhead, but model/network/tool time may dominate. Benchmark the
actual script before making a performance claim. The main benefits are typed
integration, task control, state reuse, and later support for Python tools.
Python tool callbacks can be a later step; they need GIL, async, error, and
cancellation handling, not just a function pointer.

## Proposed Python interface

This example shows the intended shape only. No such package was added:

```python
from harness import Agent

with Agent(workspace="/absolute/path/to/project", provider="...", model="...") as agent:
    result = agent.prompt("Inspect the project and summarize the risks.")
    print(result.text)
```

Also provide an asynchronous handle, a structured event stream, turn interrupt,
optional session resume, and deterministic shutdown. Keep the public interface
small enough that a process-backed implementation could use the same result
and event types. That lets users choose isolation or native execution without
rewriting their application.

## Why WebAssembly requires more work

### Browser target: `wasm32-unknown-unknown`

The present dependency graph has native requirements even when a feature is
unused at runtime:

- [Tokio features](../Cargo.toml) include multi-thread runtime, signals, and
  processes. The tools crate also enables filesystem and process operations.
- [Provider HTTP](../crates/llm/src/http.rs) uses native TCP settings and native
  TLS setup. A browser needs compatible HTTP streaming, CORS, and header rules.
- [OAuth](../crates/auth/src/openai_codex.rs) includes a loopback TCP callback
  server. Credential files and native auth storage need different backends.
- [Bash](../crates/tools/src/bash.rs) and
  [stdio MCP](../crates/mcp/src/runtime.rs) start native child processes.
- [Workspace files](../crates/tools/src/vfs.rs) use Unix directory handles for
  safe path traversal. The non-Unix branch deliberately rejects operations.
- [Search](../crates/tools/src/find.rs) uses threads and a filesystem watcher.
- [Session storage](../crates/session/src/store.rs) uses files and advisory
  locks. Agent persistence currently holds a concrete `SessionStore`.
- [Tools dependencies](../crates/tools/Cargo.toml) include native-backed search,
  Tree-sitter grammars, and Monty. Their target support needs separate checks.

Disabling Bash, MCP, or sessions through runtime settings does not remove their
compile-time dependencies. Cargo features or separate portable crates are
needed. Existing provider streams also require `Send`; browser JavaScript
futures may require different adapters.

A practical browser design is a reduced core with host interfaces for model
streams, tools, storage, time, and randomness. Provider-neutral types, dialect
conversion, SSE parsing, session replay, and compaction policy are candidates
for portable code. Their current crates still include native dependencies.
Keep provider secrets on a trusted backend where possible. A browser cannot
use local workspace tools without an explicit host service or granted file
capability.

### WASI

WASI can supply selected file, clock, and randomness capabilities. HTTP, sockets,
threads, and process support depend on the target and host runtime. Standard
WASI does not provide the local shell environment that Harness expects, and
its targets do not select the current Unix workspace-file implementation.

WASI could be a capability host after the same service boundaries are defined.
It should not be chosen merely to make Python imports possible. Python would
still need a Wasm runtime and host adapters.

## Recommended sequence

1. **Current script:** use `harness prompt` for a final answer. If structured
   streaming or several turns are required, evaluate an ACP client around
   `harness acp`. See [editor integration](./editor-integration.md).
2. **Native proof of concept:** add a PyO3 extension with one explicit-key
   provider, an explicit workspace, one prompt, optional persistence, interrupt,
   and close. Omit MCP, subagents, and Python tool callbacks initially. Check a
   real import, event delivery, and cleanup on the development machine.
3. **Shared SDK:** move common configuration and task/result ownership into a
   Rust library. Share it between frontends and bindings. Add bounded events,
   session resume, error types, and wheel packaging as needed.
4. **Other languages:** add language-specific bindings over that SDK. Examples
   are Node bindings or a small C ABI with explicit handle/memory ownership.
   PyO3 covers Python, not every language; Rust types are not a stable C ABI.
5. **Wasm only for a concrete requirement:** define the required host
   capabilities and port a reduced core. Do not start with a full-runtime port.

The immediate decision is therefore not Python versus Wasm. It is **a
process-backed SDK versus a native SDK**. Both can offer a Python import;
only the second runs the Harness agent in the Python process.
