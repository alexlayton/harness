# Harness for Python (MVP)

Run the Harness coding agent **inside the Python process**:

```python
import harness

answer = harness.run(prompt="Summarize this workspace.")
print(answer)
```

This package uses a native Rust extension. It does not launch the `harness`
executable. Bash tools and stdio MCP servers can still start child processes.
The Python distribution name is `harness-agent`; the import name is `harness`.

## Install with uv

For an existing virtual environment, activate it and install from a local
Harness checkout. This does not require a `pyproject.toml`:

```sh
CARGO_BUILD_JOBS=1 uv pip install /absolute/path/to/harness/python
python script.py
```

For a uv-managed project, use `uv add` from the directory that contains its
`pyproject.toml`:

```sh
CARGO_BUILD_JOBS=1 uv add /absolute/path/to/harness/python
uv run script.py
```

If you want to create a uv project, first run `uv init` in your script's
project directory. An activated virtual environment alone is not a uv project;
`uv add` still requires a `pyproject.toml`.

The MVP requires Python 3.10 or newer and a current stable Rust toolchain for
source installation. It targets the same macOS and Linux systems as Harness.
No prebuilt Python wheels or PyPI release are provided yet. The first install
compiles the Rust dependencies. Later installs can reuse Cargo's build cache.
Local installs use the development profile to avoid expensive release LTO.

For binding development, use an editable install:

```sh
CARGO_BUILD_JOBS=1 uv pip install --python /path/to/project/.venv/bin/python \
  --editable /absolute/path/to/harness/python
```

Rebuild the editable install after Rust changes. For an optimized wheel, run
maturin with `--profile fast-release` and install the resulting wheel with uv.

## Configuration and credentials

Without options, `harness.run` reads the same configuration as the CLI:

- `HARNESS_CONFIG_DIR/config.toml`, if `HARNESS_CONFIG_DIR` is set.
- `$XDG_CONFIG_HOME/harness/config.toml`, if `XDG_CONFIG_HOME` is set.
- Otherwise, `~/.config/harness/config.toml`.

There is no separate Python configuration file. The configured provider,
model, reasoning, RTK, compaction, subagents, MCP servers, and secret policy
apply. A provider must be selected in the configuration or call options.
Missing configuration does not select a provider automatically.

OAuth credentials use the normal `auth.json`. Sign in with the CLI first:

```sh
harness login github-copilot
# Or: harness login openai-codex
```

API-key providers read the usual environment variables. No credentials are
accepted through the options dictionary. See the repository's
[provider guide](../docs/providers.md) and
[configuration guide](../docs/configuration.md).

Each call resolves the current configuration again. Options override only
that call and do not modify `config.toml`. The library does not change the
process directory, mutate environment variables, install signal handlers,
initialize global logging, or report Herdr pane state.

## Options

Pass a dictionary. Every key is optional; omitted keys retain the defaults:

```python
from pathlib import Path
import harness

answer = harness.run(
    prompt="Inspect the project and list the main risks.",
    options={
        "workspace": Path("/absolute/path/to/project"),
        "model": "your-provider-model-id",
        "reasoning_effort": "high",
        "no_session": True,
    },
)
```

| Key | Type | Default / behavior |
|---|---|---|
| `provider` | `str` | Configured provider; supports normal CLI aliases. |
| `model` | `str` | Configured model, or the normal provider default. |
| `reasoning_effort` | `str` | Configured preference; `auto`, `off`, `minimal`, `low`, `medium`, `high`, or `maximum`. |
| `workspace` | `str` or text `PathLike` | Current directory; relative paths use the caller's current directory. |
| `no_session` | `bool` | `False`; create and persist a new session. |
| `resume` | `str` | Resume by ID, unique prefix, `latest`, or path instead of creating a session. |
| `no_context_files` | `bool` | `False`; use normal instruction-file discovery and its environment overrides. |

Unknown keys and invalid option values are rejected. `resume` and
`no_session=True` cannot be combined. `harness.RunOptions` is a `TypedDict`
for editors and type checkers; normal dictionaries also work.

## Results, errors, and lifecycle

`run` is synchronous and returns a `str` with the final answer, without an
added newline. It excludes intermediate model text from tool-bearing rounds.
A successful tool-only turn can return an empty string.

- Invalid inputs raise `TypeError` or `ValueError`.
- Configuration, authentication, provider, tool-initialization, session, and
  unrecovered agent failures raise `RuntimeError`.
- An individual tool failure does not necessarily fail the call: the agent
  can recover and produce an answer.
- Ctrl-C on Python's main thread requests cancellation and waits for session
  and MCP cleanup before raising `KeyboardInterrupt`.

The extension releases the Python GIL while it waits, and uses one native
runtime for the process. It waits for the one-shot agent and MCP shutdown
before returning. A cancelled Monty calculation can continue in a native
blocking task until its bounded execution limit; cancellation does not force
that thread to stop. Sessions use the normal Harness state roots and locks.
Without `resume`, separate calls create separate conversations. Using
`resume="latest"` selects the latest session for the chosen workspace; use
an explicit ID when other agents might create sessions there.

This MVP returns text only. It does not expose streaming, an async API,
Python tool callbacks, a timeout, a session-result object, or explicit
programmatic cancellation. Python subinterpreters are not supported. Do not
use `fork` after the first call initializes Harness; inherited native threads
cannot run in the child, so the library rejects such calls. Use the `spawn`
start method for Python multiprocessing.

> **Warning:** Tools run without confirmation. Harness is not a sandbox.
> Bash has the operating-system access of your Python process. Native
> execution also shares memory and failure risk with your script.

## Small local checks

After installing the package into a virtual environment:

```sh
uv run --project /path/to/your/python/project \
  python -m unittest discover -s /absolute/path/to/harness/python/tests
```

These tests do not call a model or read real credentials. They check native
input validation, normal configuration paths, Ctrl-C during MCP startup, and
safe rejection after fork. All test state uses temporary directories.
