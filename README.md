![Harness header](assets/header.png)

> **NEW ACHIEVEMENT! YOU BUILT YOUR OWN CODING HARNESS.**
>
> **REWARD:** You get to use the harness. That's it.

99.999% slop. That's five nines! You should probably pick a different
harness.

Harness is a Rust-based coding agent with an emphasis on speed and efficiency.
It streams model responses, runs workspace-scoped file tools, delegates work to
subagents, and stores durable sessions. Use it in a direct terminal UI, in
headless scripts, or from an editor through the Agent Client Protocol (ACP).

> [!WARNING]
> **THE ILLUSION OF CHOICE!**
>
> Harness runs tools without a confirmation step. But you were going to blindly
> click **Accept all** anyway, weren't you? Congratulations. The tedious
> illusion of informed consent has been removed for your convenience.
>
> This applies to the terminal UI, headless mode, ACP, and external MCP tools.
> Harness is not a sandbox. Dedicated file tools restrict writes to the
> workspace, but `read` can open absolute paths outside it. Shell commands and
> MCP tools can access resources available to the operating-system user. Run
> Harness only in workspaces and environments where you accept that behavior.
>
> A permission system would be nice. There isn't one yet.

## The gimmick: spend fewer tokens

Harness has built-in support for [RTK (Rust Token Killer)][rtk]. When enabled,
the bash tool asks an installed `rtk` binary to rewrite supported commands into
token-optimized equivalents. Unsupported commands, rewrite failures, and a
missing RTK binary fall back to the original command. RTK is optional and off
by default; set `rtk = true` in `config.toml` to enable it.

The `find`, `grep`, and `multigrep` tools use the [fff-search crate][fff]. They
share one lazy, watched workspace index. The `grep` tool uses fff-search's
ripgrep-compatible engine directly, so the model can search files without
building a shell pipeline. For source navigation, `outline` uses Tree-sitter to
list declarations and their line ranges before the model reads a full file.

For small calculations and data transformations, `python` runs a fresh snippet
in [Monty][monty] without a separate Python installation. It has no file access,
third-party packages, or host callbacks. It runs in the Harness process, so it
is **not** a security sandbox for untrusted code; use the file tools for files
and the shell for commands, builds, and tests.

See [Configuration][configuration] for tool limits, the RTK setting, and other
advanced options.

## Highlights

- A fast, direct-crossterm terminal UI that keeps completed output in native
  terminal scrollback.
- An experimental terminal multiplexer for running and switching between
  several independent agents without PTYs or split panes.
- Headless output for scripts and pipelines.
- ACP support for compatible editors.
- OpenCode Go, OpenRouter, GitHub Copilot, and OpenAI Codex subscription
  providers.
- Indexed search, Tree-sitter source outlines, Monty Python calculations,
  workspace-scoped file tools, an unrestricted shell, optional MCP servers,
  and bounded subagents.
- Model-assisted context compaction with a deterministic local fallback.
- Automatic `AGENTS.md`/`CLAUDE.md` context and Agent Skills discovery.
- Append-only JSONL sessions with reload, export, and context compaction.

For scripting, `harness prompt` writes only the final answer to stdout;
verbose progress and diagnostics go to stderr. The `harness acp` frontend
writes only JSON-RPC protocol traffic to stdout, keeping both frontends safe
for pipelines and editor transports.

## Documentation

- [Providers and authentication][providers]
- [Configuration][configuration]
- [Editor integration (ACP)][editor-integration]
- [Architecture and contributor guide][architecture]
- [Session format][session-format]

## Installation

The installer downloads the latest release, verifies its SHA-256 checksum, and
puts `harness` in `~/.local/bin` (or a directory passed with `--bin-dir`):

```sh
curl -fsSL https://raw.githubusercontent.com/alexlayton/harness/main/install.sh | bash
```

To install a specific release, pass the version through to the script:

```sh
curl -fsSL https://raw.githubusercontent.com/alexlayton/harness/main/install.sh \
  | bash -s -- --version v0.5.0
```

The script uses `curl` rather than a browser, so macOS downloads normally do
not acquire a browser quarantine attribute. It does not invoke `sudo` or modify
shell startup files.

Alternatively, download the archive from [GitHub Releases][releases]. Each
archive contains the `harness` binary, this README, its header image, and the
license. Verify the archive with `SHA256SUMS`, then put the binary in a
directory on your `PATH`.

Initial releases provide these builds:

- macOS on Apple silicon (`aarch64-apple-darwin`)
- macOS on Intel (`x86_64-apple-darwin`)
- Linux x86-64 with glibc 2.35 or newer (`x86_64-unknown-linux-gnu`)

The GNU Linux artifact is built on Ubuntu 22.04 to keep that minimum runtime
baseline stable. Windows is not supported in the initial release.

Homebrew users can install the source formula:

```text
brew install alexlayton/tap/harness
```

## Quick start

The shortest setup uses a GitHub Copilot subscription. Sign in, then run one
prompt in the current workspace:

```text
harness login github-copilot
harness prompt "Summarize this workspace"
```

The login command shows a device URL and one-time code. When no provider is
configured, login selects Copilot without replacing an existing choice.
Harness selects an available Copilot model when no model is configured. See
[Providers and authentication][providers] for API-key providers and OpenAI
Codex.

Start the terminal UI with the same provider:

```text
harness
```

To run several agents in one terminal, start the experimental mux frontend:

```text
harness mux
```

## Usage overview

Start the terminal UI by running `harness`. Use `/help` to list its commands.
The main commands include session management, model selection, usage reporting,
compaction, and discovered skills.

Start the terminal UI in a dedicated Git worktree, creating the branch from the
current `HEAD` when it does not already exist:

```text
harness worktree feat/new-feature
```

Use `--start-point REV` (also accepted as `--base REV`) to create a missing
branch from another commit, and `--dir PATH` to select its location. Automatic
worktrees live below `~/.harness/worktrees`, or
`$HARNESS_STATE_DIR/worktrees` when the state root is overridden.

Harness removes a clean ephemeral worktree when the frontend exits but always
leaves the branch in place. Modified or untracked files cause the worktree to
be retained rather than force-removed. Ignored-only files do not pin an
ephemeral worktree and are removed with it, so use `--keep` if an ignored file
contains data you need. `--keep` makes retention sticky across later Harness
runs at the same branch and path; `--ephemeral` clears that policy and
restores automatic cleanup.

A new worktree contains committed Git state, not uncommitted changes from the
launch checkout; Harness warns when it detects those changes. To run a
headless prompt with the same lifecycle, nest the prompt command:

```text
harness worktree feat/new-feature prompt --no-session "Implement the change"
```

Sessions remain scoped to the physical worktree path. Recreating an automatic
worktree at its stable default path sees its previous sessions, while choosing
a different `--dir` creates a separate session namespace.

Mux starts one persisted agent in the launch directory. Press `Ctrl+Space`
then `n` to add an agent for the current directory, another directory, or a
new worktree. The same prefix supports `w`, `c`, and `d` to open those
workspace forms directly, `j`/`k` and `1`–`9` for switching, `x` to close,
and `?` for help. Mux roster ordering is process-local, while each
conversation uses the normal durable session store. Mux-created worktrees are
retained when an agent closes. Their creation dialog defaults to retaining
that worktree for future runs too; toggling the policy off clears an older
sticky marker but does not remove the worktree when the mux slot closes.

Run one prompt without the terminal UI:

```text
harness prompt "Summarize this workspace"
```

Run Harness as an ACP subprocess for an editor:

```text
harness acp
```

For provider selection, sign-in commands, configuration paths, MCP setup, and
advanced settings, use the documentation links above.

## Development

Contributing to this repository, or running an agent in it? Read
[AGENTS.md][agents] first. [ARCHITECTURE.md][architecture] explains the crate
map, data flow, and project invariants.

Development uses the current stable Rust toolchain. The project does not yet
declare a minimum supported Rust version (MSRV).

Source package managers can avoid the expensive fat LTO used for official
release artifacts while retaining an optimized, stripped binary:

```text
cargo install --locked --profile fast-release --path crates/harness
```

The common workspace checks are:

```text
cargo fmt --check
cargo clippy --workspace
cargo test --workspace
cargo build --workspace
```

## License

Harness is available under the [MIT License](./LICENSE).

[agents]: https://github.com/alexlayton/harness/blob/main/AGENTS.md
[architecture]: https://github.com/alexlayton/harness/blob/main/ARCHITECTURE.md
[configuration]: https://github.com/alexlayton/harness/blob/main/docs/configuration.md
[editor-integration]: https://github.com/alexlayton/harness/blob/main/docs/editor-integration.md
[fff]: https://crates.io/crates/fff-search
[monty]: https://pydantic.dev/docs/monty/
[providers]: https://github.com/alexlayton/harness/blob/main/docs/providers.md
[releases]: https://github.com/alexlayton/harness/releases
[rtk]: https://github.com/rtk-ai/rtk
[session-format]: https://github.com/alexlayton/harness/blob/main/crates/session/README.md
