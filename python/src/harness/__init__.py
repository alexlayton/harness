"""Run the Harness agent in the Python process with the normal system config."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Literal, TypedDict

from . import _native

__all__ = ["HttpMcpServer", "RunOptions", "StdioMcpServer", "run"]


class _StdioOptions(TypedDict, total=False):
    args: list[str]
    env: dict[str, str]


class StdioMcpServer(_StdioOptions):
    """A named MCP server launched directly in the selected workspace."""

    name: str
    transport: Literal["stdio"]
    command: str | os.PathLike[str]


class _HttpOptions(TypedDict, total=False):
    headers: dict[str, str]


class HttpMcpServer(_HttpOptions):
    """A named Streamable HTTP MCP endpoint."""

    name: str
    transport: Literal["http"]
    url: str


class RunOptions(TypedDict, total=False):
    """Per-call overrides; omitted settings use the normal Harness config."""

    provider: str
    model: str
    reasoning_effort: str
    workspace: str | os.PathLike[str]
    no_session: bool
    resume: str
    no_context_files: bool
    mcp_servers: list[StdioMcpServer | HttpMcpServer] | None
    skill_dirs: list[str | os.PathLike[str]] | None


def _text_path(value: str | os.PathLike[str], name: str) -> str:
    path = os.fspath(value)
    if not isinstance(path, str):
        raise TypeError(f"{name} must be a string or a text PathLike")
    return path


def run(prompt: str, options: RunOptions | None = None) -> str:
    """Run one prompt and return the final answer without a trailing newline.

    With no options, use the normal config.toml, credentials, current workspace,
    and durable sessions. Overrides apply only to this call. Tools run without
    confirmation and are not sandboxed. For mcp_servers and skill_dirs, omitted
    or None inherits defaults; a supplied list replaces them; [] disables them.

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
        overrides["workspace"] = _text_path(overrides["workspace"], "workspace")
    if overrides.get("skill_dirs") is not None:
        dirs = overrides["skill_dirs"]
        if not isinstance(dirs, (list, tuple)):
            raise TypeError("skill_dirs must be a list of paths or None")
        overrides["skill_dirs"] = [_text_path(path, "skill_dirs entry") for path in dirs]
    if overrides.get("mcp_servers") is not None:
        servers = overrides["mcp_servers"]
        if not isinstance(servers, (list, tuple)):
            raise TypeError("mcp_servers must be a list of server mappings or None")
        normalized = []
        for server in servers:
            if not isinstance(server, Mapping):
                raise TypeError("mcp_servers entries must be mappings")
            server = dict(server)
            if "command" in server:
                server["command"] = _text_path(server["command"], "MCP command")
            normalized.append(server)
        overrides["mcp_servers"] = normalized
    return _native.run(prompt, json.dumps(overrides, allow_nan=False))
