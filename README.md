```text
██  ██ ░▒▀▀██ ██▀▀██ ██▀▀██ ██▀▀▒░ ▒▓▀▀██ ▒▓▀▀██
██▀▀██ ▒▓  ██ ██     ██  ██ ██▄▄▓▒ ▓█▄▄▄▄ ▓█▄▄▄▄
██  ██ ▓█▀▀██ ██     ██  ██ ██▄▄▄▄ ▄▄  ▒▒ ▄▄  ▒▒
       ▀▀                          ▀▀▀▀▀▀ ▀▀▀▀▀▀
```

> **NEW ACHIEVEMENT! YOU BUILT YOUR OWN CODING HARNESS.**
>
> **REWARD:** You get to use the harness. That is all.

100% slop. You probably should not use it.

Harness is a Rust-based coding agent with an emphasis on speed and efficiency.
It streams model responses, runs workspace-scoped file tools, delegates work to
subagents, and stores durable sessions. Use it in a direct terminal UI, in
headless scripts, or from an editor through the Agent Client Protocol (ACP).

> [!WARNING]
> **NEW ACHIEVEMENT: THE ILLUSION OF CHOICE!**
>
> Harness runs tools without a confirmation step. But you were going to blindly
> click **Accept all** anyway, weren't you? Congratulations. The tedious
> illusion of informed consent has been removed for your convenience.
>
> This applies to the terminal UI, headless mode, ACP, and external MCP tools.
> Harness is not a sandbox. The dedicated file tools restrict paths to the
> workspace, but shell commands and MCP tools can access all resources available
> to the current operating-system user. Run Harness only in workspaces and
> environments where you accept that behavior.

## The gimmick: spend fewer tokens

Harness has built-in support for [RTK (Rust Token Killer)][rtk]. When enabled,
the bash tool asks an installed `rtk` binary to rewrite supported commands into
token-optimized equivalents. Unsupported commands, rewrite failures, and a
missing RTK binary fall back to the original command. RTK is optional and off
by default; set `rtk = true` in `config.toml` to enable it.

The `find`, `grep`, and `multigrep` tools use the [fff-search crate][fff]. They
share one lazy, watched workspace index. The `grep` tool uses fff-search's
ripgrep-compatible engine directly, so the model can search files without
building a shell pipeline.

See [Configuration][configuration] for the RTK setting and other advanced
options.

## Highlights

- A fast, direct-crossterm terminal UI that keeps completed output in native
  terminal scrollback.
- Headless output for scripts and pipelines.
- ACP support for compatible editors.
- OpenCode Go, OpenRouter, GitHub Copilot, and OpenAI Codex subscription
  providers.
- Workspace-scoped file tools, an unrestricted shell, optional MCP servers,
  and bounded subagents.
- Automatic `AGENTS.md`/`CLAUDE.md` context and Agent Skills discovery.
- Append-only JSONL sessions with reload, export, and context compaction.

## Documentation

- [Providers and authentication][providers]
- [Configuration][configuration]
- [Editor integration (ACP)][editor-integration]
- [Architecture and contributor guide][architecture]
- [Session format][session-format]

## Installation

Download the archive for your system from [GitHub Releases][releases]. Each
archive contains the `harness` binary, this README, and the license. Verify the
archive with `SHA256SUMS`, then put the binary in a directory on your `PATH`.

Initial releases provide these builds:

- macOS on Apple silicon (`aarch64-apple-darwin`)
- macOS on Intel (`x86_64-apple-darwin`)
- Linux x86-64 with glibc (`x86_64-unknown-linux-gnu`)

Windows is not supported in the initial release.

Homebrew installation is planned, but it is not available yet.

## Quick start

The shortest setup uses a GitHub Copilot subscription. Sign in, then run one
prompt in the current workspace:

```text
harness login github-copilot
harness --provider github-copilot -p "Summarize this workspace"
```

The login command shows a device URL and one-time code. Harness selects an
available Copilot model when no model is configured. See [Providers and
authentication][providers] for API-key providers and OpenAI Codex.

Start the terminal UI with the same provider:

```text
harness --provider github-copilot
```

## Usage overview

Start the terminal UI by running `harness`. Use `/help` to list its commands.
The main commands include session management, model selection, usage reporting,
compaction, and discovered skills.

Run one prompt without the terminal UI:

```text
harness -p "Summarize this workspace"
```

Run Harness as an ACP subprocess for an editor:

```text
harness --acp
```

For provider selection, sign-in commands, configuration paths, MCP setup, and
advanced settings, use the documentation links above.

## Development

Contributing to this repository, or running an agent in it? Read
[AGENTS.md][agents] first. [ARCHITECTURE.md][architecture] explains the crate
map, data flow, and project invariants.

Development uses the current stable Rust toolchain. The project does not yet
declare a minimum supported Rust version (MSRV).

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
[providers]: https://github.com/alexlayton/harness/blob/main/docs/providers.md
[releases]: https://github.com/alexlayton/harness/releases
[rtk]: https://github.com/rtk-ai/rtk
[session-format]: https://github.com/alexlayton/harness/blob/main/crates/session/README.md
