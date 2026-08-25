# AGENTS.md

Guidance for coding agents (and humans) working in this repository.

## What this is

Harness is a terminal coding-agent harness written in Rust: a workspace of 8
crates that streams LLM responses, runs workspace-scoped tools, and persists
durable sessions. Before a non-trivial change, use
[ARCHITECTURE.md](./ARCHITECTURE.md) as an index: read the crate map and only
the section relevant to the task. Read the complete file only for changes that
cross several subsystems.

## Commands

```sh
cargo fmt                       # required before every commit
cargo clippy --workspace        # keep it clean
cargo build                     # builds only the `agent` crate (default-members)
cargo build --workspace         # everything
cargo test --workspace          # all tests; plain `cargo test` covers agent only
cargo run -p agent -- -p "hi"   # run the `harness` binary headless
cargo run -p agent              # run the TUI
```

GitHub CI runs the workspace checks on Linux and macOS. There is no
`clippy.toml` or Rust toolchain pin; follow rustfmt and Clippy defaults (stable,
edition 2024).

## Conventions

- **Conventional commits** with a scope, usually the crate or feature:
  `feat(tui): …`, `fix(copilot): …`, `chore: …`.
- **Doc comments on public items.** Existing code comments frequently explain
  *why*, including design rationale ("intentionally", "so that…"). Match that.
- **Tests are colocated** with the code in `#[cfg(test)]` modules inside
  `src/`; there is no `tests/` directory. Pure functions are favored so they
  can be unit-tested without network or provider access.
- **New providers** must not leak wire formats into `agent`. Add a dialect
  (wire format translation) in `llm/src/dialects/` and endpoint/auth handling
  in `llm/src/providers/`, then register it in `agent/src/config.rs`.
- **New tools** implement the `Tool` trait (spec + prompt metadata) in
  `crates/tools` and get registered in `default_registry`. Tools must confine
  paths to the workspace root.
- **Skills context**: skills autodiscover from `.harness/skills/` and
  `.agents/skills/` (project walk to the git root) plus the global
  `~/.harness/skills`. Project instructions are injected from `AGENTS.md` /
  `CLAUDE.md`, including the global `~/.harness/AGENTS.md` (opt out with
  `--no-context-files`).
- **Env overrides for tests**: `HARNESS_CONFIG_DIR`, `HARNESS_SESSION_DIR`,
  `HARNESS_STATE_DIR`, `HARNESS_LOG` (file-only tracing). Use them to avoid
  touching real `~/.config/harness` and `~/.harness/sessions`.

## Repository notes

- Untracked scratch files at the root (`todo.md`, `fixes.md`, `inline-ui.md`,
  `session.jsonl`, `ascii.txt`) are working notes, not source; don't wire them
  into anything.
- `session.jsonl` in the root is a stray export; the real store lives under
  `~/.harness/sessions`.

## Things that will bite you

- Tool descriptions, JSON schemas, and the system prompt are generated from
  the same `ToolRegistry` snapshot — never hand-write tool docs into the
  prompt.
- Sessions are append-only JSONL. Never rewrite history; emit events.
- Headless mode must keep stdout to the answer only; all chatter goes to
  stderr behind `-v`.
- The TUI crate must not depend on the agent crate (the event adapter in
  `agent/src/lib.rs` exists precisely to avoid that cycle).
