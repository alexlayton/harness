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

- **Enter** — send the current message
- **Shift+Enter** or **Alt+Enter** — insert a newline
- **Esc** — interrupt generation or a running tool
- **Ctrl+C** / **Ctrl+D** — quit and restore the terminal

The available tools are `read`, `write`, and `bash`. Relative paths and shell commands use the
process working directory.
