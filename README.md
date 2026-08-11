# Harness

Harness is a small Rust coding-agent harness with an inline ratatui terminal UI. It streams
responses, can inspect and edit the working tree, and runs shell commands sequentially when the
model requests tools.

## Build and run

```sh
cargo build --release
export OPENCODE_GO_API_KEY=...
cargo run --release
```

The default is OpenCode Go with `gpt-5.6-luna` (the OpenAI Responses dialect). OpenRouter is also
available:

```sh
export OPENROUTER_API_KEY=...
cargo run --release -- --provider openrouter
cargo run --release -- --provider opencode-go --model minimax-m3
```

Use `--provider opencode-go|openrouter` and `--model <model-id>` to override the defaults.

## Environment variables

- `OPENCODE_GO_API_KEY` — preferred OpenCode Go Bearer token
- `OPENCODE_API_KEY` — legacy OpenCode-compatible fallback, used when `OPENCODE_GO_API_KEY` is unset
- `OPENROUTER_API_KEY` — OpenRouter Bearer token
- `HARNESS_LOG=<path>` — optional file receiving request, parsing, and retry logs; stdout remains
  owned by the TUI

## Keys

The welcome header and this list use the same keymap:

- **Enter** — Send message
- **Shift+Enter** — Newline
- **↑ / ↓** — History
- **Esc** — Interrupt
- **Ctrl+O** — Expand tool call
- **Ctrl+C** — Quit

Alt+Enter is an undocumented newline fallback for terminals that do not report Shift+Enter, and
Ctrl+D remains a legacy quit alias.

Finished tool calls are compact by default. While idle, press **Ctrl+O** (or **Tab**) to focus the
live tool tail, use **↑ / ↓** to select a call, and press **Enter** or **Space** to expand it. In
focus, **Ctrl+O** dumps the selected detail block to scrollback; with no live tail it dumps the
most recent completed call.
Mouse expansion is intentionally out of scope for the inline viewport: immutable terminal
scrollback cannot provide reliable click regions for historical blocks.

The available tools are `read`, `write`, and `bash`. Relative paths and shell commands use the
process working directory.
