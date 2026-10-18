"""Run a prompt through the native Harness Python library."""

import argparse
from pathlib import Path
import sys

import harness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Reply with one short sentence to confirm that Harness runs from Python. Do not use tools.",
    )
    parser.add_argument("--workspace", type=Path, help="Override the workspace directory")
    parser.add_argument("--no-session", action="store_true", help="Do not save a session")
    parser.add_argument("--no-mcp", action="store_true", help="Disable configured MCP servers")
    skills = parser.add_mutually_exclusive_group()
    skills.add_argument(
        "--skill-dir", dest="skill_dirs", type=Path, action="append",
        help="Replace default skill roots (repeat for more directories)",
    )
    skills.add_argument("--no-skills", action="store_true", help="Disable skill discovery")
    args = parser.parse_args()

    options: harness.RunOptions = {}
    if args.workspace is not None:
        options["workspace"] = args.workspace
    if args.no_session:
        options["no_session"] = True
    if args.no_mcp:
        options["mcp_servers"] = []
    if args.skill_dirs is not None:
        options["skill_dirs"] = args.skill_dirs
    elif args.no_skills:
        options["skill_dirs"] = []

    # To replace MCP servers, set options["mcp_servers"] to server dictionaries.
    # See ../python/README.md for stdio and HTTP examples.
    print("Running Harness with your normal configuration...", file=sys.stderr)
    try:
        # None uses the system config.toml and the normal runtime defaults.
        answer = harness.run(prompt=args.prompt, options=options or None)
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"Harness error: {error}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
