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
    args = parser.parse_args()

    options: harness.RunOptions = {}
    if args.workspace is not None:
        options["workspace"] = args.workspace
    if args.no_session:
        options["no_session"] = True

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
