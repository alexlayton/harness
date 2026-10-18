"""Run the Harness agent in the Python process with the normal system config."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TypedDict

from . import _native

__all__ = ["RunOptions", "run"]


class RunOptions(TypedDict, total=False):
    """Per-call overrides; omitted settings use the normal Harness config."""

    provider: str
    model: str
    reasoning_effort: str
    workspace: str | os.PathLike[str]
    no_session: bool
    resume: str
    no_context_files: bool


def run(prompt: str, options: RunOptions | None = None) -> str:
    """Run one prompt and return the final answer without a trailing newline.

    With no options, use the normal config.toml, credentials, current workspace,
    and durable sessions. Overrides apply only to this call. Tools run without
    confirmation and are not sandboxed.

    This call blocks, but releases the Python GIL. Ctrl-C requests cancellation
    and waits for session/MCP cleanup before raising KeyboardInterrupt. Model and
    runtime failures raise RuntimeError; invalid inputs raise TypeError or
    ValueError. A successful tool-only turn can return an empty string.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping or None")
    overrides = dict(options) if options is not None else {}
    if "workspace" in overrides:
        # Convert pathlib.Path without changing the caller's current directory.
        workspace = os.fspath(overrides["workspace"])
        if not isinstance(workspace, str):
            raise TypeError("workspace must be a string or a text PathLike")
        overrides["workspace"] = workspace
    return _native.run(prompt, json.dumps(overrides, allow_nan=False))
