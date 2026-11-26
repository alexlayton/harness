# AGENTS.md

Guidance for coding agents (and humans) working in this repository.

## What this is

Harness is a terminal coding-agent harness written in Rust: a workspace of 11
crates that streams LLM responses, runs workspace-scoped tools, and persists
durable sessions. Before a non-trivial change, use
[ARCHITECTURE.md](./ARCHITECTURE.md) as an index: read the crate map and only
the section relevant to the task. Read the complete file only for changes that
cross several subsystems.

## Commands

```sh
cargo fmt                       # required before every commit
cargo clippy --workspace        # keep it clean
cargo build                     # builds the `harness` binary and `agent` runtime
cargo build --workspace         # everything
cargo test --workspace          # all tests; plain `cargo test` covers host + runtime
cargo run -p harness -- prompt "hi" # run the `harness` binary headless
cargo run -p harness            # run the TUI
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
  in `llm/src/providers/`, then register it in `crates/harness/src/config.rs`.
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

## Git workflow

Solo project, so PRs are for risk, not for human review.

- **Push straight to `main`** for small fixes, single-crate features, docs,
  release bumps, and anything easy to revert. Do this from a clean `main`
  with conventional commits.
- **Open a PR** for: session JSONL format or agent-loop changes (sessions are
  append-only, so a bad format can't be reverted out of users' history),
  `.github/workflows/*` changes, multi-crate or large (~500+ line) changes,
  or when you want the Codex review or macOS CI before merging.
- **Squash-merge PRs** (the only merge style enabled; branches auto-delete).
  The PR title must be a conventional commit.
- **Pre-push hook**: enable once per clone with
  `git config core.hooksPath scripts/hooks`. It runs fmt, clippy
  (`-D warnings`) and tests, replacing the pre-merge CI run. Bypass only in
  emergencies with `--no-verify`.
- **`main` ruleset** blocks force-push, deletion and merge commits. There are
  no required checks or required PRs, deliberately, so direct pushes work.
- **CI** runs on PRs and on `main`, skips docs-only changes, and cancels
  superseded runs on non-main refs. Releases re-run all checks before
  building, so a broken `main` can't ship.
- Agents: don't open a PR unless the change matches the PR criteria above or
  you're asked to.

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
- The TUI crate must not depend on the agent crate. The explicit adapters in
  `crates/harness/src/tui_adapter.rs` keep both frontend protocols independent.
