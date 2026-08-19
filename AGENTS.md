# AGENTS.md

Guidance for coding agents (and humans) working in this repository.

## What this is

Harness is a terminal coding-agent harness written in Rust: a workspace of 7
crates that streams LLM responses, runs workspace-scoped tools, and persists
durable sessions. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the crate map,
data flow, and invariants before making non-trivial changes — it explains where
each responsibility lives and why.

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

There is no CI, no clippy.toml, and no rust-toolchain pin; follow rustfmt and
clippy defaults (stable, edition 2024).

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
