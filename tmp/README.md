# Local Harness Python example

This uv project uses the native package in `../python`. It does not need the
`harness` executable to run prompts. The existing `.python-version` selects
Python 3.14.

## Run

From this directory:

```sh
uv run main.py
```

The default prompt asks for one short confirmation sentence without tools.
Pass your own prompt as an argument:

```sh
uv run main.py "Explain what this Python example does."
```

Use another workspace or disable session persistence:

```sh
uv run main.py --workspace .. --no-session "Summarize this workspace without changing files."
```

Try only the example skill directory, without configured MCP servers:

```sh
uv run main.py --skill-dir skills --no-mcp --no-session "Use the python-example skill to explain this example."
```

Skill directory paths are relative to the selected workspace. Repeat
`--skill-dir` to supply more directories; the first one wins for duplicate
skill names. Use `--no-skills` instead to disable all skill discovery. Edit
`options["mcp_servers"]` in `main.py` to supply stdio or HTTP MCP servers; the
[Python guide](../python/README.md) includes server dictionary examples.

With no overrides, the script calls `harness.run` with `options=None`. It uses
your normal system `config.toml`, credentials, current directory, and session
settings. Edit `main.py` to try other call options. The full API is documented
in [the Python guide](../python/README.md).

## Setup and import check

The local dependency is declared in `pyproject.toml`; `uv.lock` records it.
If the virtual environment is missing, uv will create it. For a first source
build, limit Cargo to one build worker:

```sh
CARGO_BUILD_JOBS=1 uv sync
uv run python -c "import harness; print('Harness imported')"
```

Source builds require a current stable Rust toolchain. After Python or Rust
library changes, rebuild the dependency when needed:

```sh
CARGO_BUILD_JOBS=1 uv sync --reinstall-package harness-agent
```

Running `main.py` sends a real request to your configured provider and can
incur normal usage charges. It creates a session unless you pass
`--no-session`. Tools run without confirmation and are not sandboxed.
