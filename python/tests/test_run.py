"""Small native API checks: no model requests or real user state."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import harness


class RunTests(unittest.TestCase):
    def test_input_validation(self):
        with self.assertRaises(TypeError):
            harness.run(prompt=123)
        with self.assertRaises(ValueError):
            harness.run(prompt=" \n")
        with self.assertRaises(TypeError):
            harness.run(prompt="hello", options=[])
        with self.assertRaises(ValueError):
            harness.run(prompt="hello", options={"modle": "typo"})
        with self.assertRaises(ValueError):
            harness.run(prompt="hello", options={"no_session": "yes"})
        with self.assertRaises(ValueError):
            harness.run(prompt="hello", options={"reasoning_effort": "invalid"})
        with self.assertRaises(ValueError):
            harness.run(prompt="hello", options={"provider": "invalid"})
        with self.assertRaises(ValueError):
            harness.run(prompt="hello", options={"resume": "latest", "no_session": True})
        for options in (
            {"skill_dirs": "not-a-list"},
            {"skill_dirs": [b"not-a-text-path"]},
            {"mcp_servers": {}},
            {"mcp_servers": ["not-a-mapping"]},
        ):
            with self.subTest(options=options), self.assertRaises(TypeError):
                harness.run(prompt="hello", options=options)
        for options in (
            {"skill_dirs": [""]},
            {"mcp_servers": [{"name": "missing-transport"}]},
            {"mcp_servers": [{"name": "bad-url", "transport": "http", "url": "invalid"}]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                harness.run(prompt="hello", options=options)

    def test_omitted_options_load_system_config(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            config = directory / "config.toml"
            config.write_text('provider = "invalid-test-provider"\n', encoding="utf-8")
            with patch.dict(os.environ, {"HARNESS_CONFIG_DIR": root}):
                for options in (
                    None,
                    {},
                    {"workspace": directory},
                    {"mcp_servers": None, "skill_dirs": None},
                    {"mcp_servers": [], "skill_dirs": []},
                    {
                        "skill_dirs": [directory],
                        "mcp_servers": [{
                            "name": "python", "transport": "stdio", "command": Path(sys.executable),
                        }],
                    },
                ):
                    with self.subTest(options=options):
                        with self.assertRaisesRegex(RuntimeError, "invalid-test-provider"):
                            harness.run(prompt="hello", options=options)
                # No failed call may silently repair or overwrite user settings.
                self.assertEqual(
                    config.read_text(encoding="utf-8"),
                    'provider = "invalid-test-provider"\n',
                )
                self.assertEqual(sorted(path.name for path in directory.iterdir()), ["config.toml"])

    def test_mcp_overrides_skip_stale_servers_and_validate_skill_directories(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            contents = (
                'provider = "opencode-go"\n[[mcp.servers]]\n'
                'name = "stale"\ntransport = "obsolete"\n'
            )
            config = directory / "config.toml"
            config.write_text(contents, encoding="utf-8")
            options = {
                "workspace": directory,
                "skill_dirs": [Path("missing")],
                "no_context_files": True,
                "no_session": True,
            }
            with patch.dict(os.environ, {
                "HARNESS_CONFIG_DIR": root, "OPENCODE_GO_API_KEY": "unused-test-key",
            }):
                with self.assertRaisesRegex(RuntimeError, "parse config file"):
                    harness.run(prompt="hello", options=options)
                for servers in ([], [{
                    "name": "replacement", "transport": "http", "url": "https://example.test/mcp",
                }]):
                    with self.subTest(servers=servers):
                        with self.assertRaisesRegex(RuntimeError, "resolve skill directory"):
                            harness.run(prompt="hello", options={**options, "mcp_servers": servers})
            self.assertEqual(config.read_text(encoding="utf-8"), contents)

    @unittest.skipUnless(os.name == "posix", "requires Unix signals")
    def test_ctrl_c_cancels_native_startup_without_model_request(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            ready = directory / "mcp-started"
            server = (
                "from pathlib import Path; import time; "
                f"Path({str(ready)!r}).write_text('ready'); time.sleep(8)"
            )
            directory.joinpath("config.toml").write_text(
                'provider = "opencode-go"\n'
                '[[mcp.servers]]\nname = "slow-test-server"\ntransport = "stdio"\n'
                f"command = {json.dumps(sys.executable)}\n"
                f"args = {json.dumps(['-c', server])}\n",
                encoding="utf-8",
            )
            script = """
import os
from pathlib import Path
import signal
import threading
import time
import sys
import harness

ready = Path(os.environ["HARNESS_CONFIG_DIR"]) / "mcp-started"
def interrupt():
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    os.kill(os.getpid(), signal.SIGINT)

threading.Thread(target=interrupt, daemon=True).start()
options = {"workspace": ready.parent, "no_session": True, "no_context_files": True}
if os.environ["HARNESS_TEST_OVERRIDE_MCP"] == "1":
    options["mcp_servers"] = [{
        "name": "override-server", "transport": "stdio", "command": Path(sys.executable),
        "args": ["-c", os.environ["HARNESS_TEST_MCP_SERVER"]],
    }]
    options["skill_dirs"] = []
try:
    harness.run(prompt="must cancel before any model request", options=options)
except KeyboardInterrupt:
    assert ready.exists(), "MCP server did not start"
    print("interrupted")
else:
    raise AssertionError("expected KeyboardInterrupt")
"""
            for override in (False, True):
                with self.subTest(override=override):
                    if override:
                        ready.unlink()
                        # An explicit list must replace even stale config entries.
                        directory.joinpath("config.toml").write_text(
                            'provider = "opencode-go"\n[[mcp.servers]]\n'
                            'name = "stale"\ntransport = "obsolete"\n',
                            encoding="utf-8",
                        )
                    result = subprocess.run(
                        [sys.executable, "-c", script],
                        env={
                            **os.environ,
                            "HOME": root,
                            "HARNESS_CONFIG_DIR": root,
                            "HARNESS_STATE_DIR": str(directory / "state"),
                            "HARNESS_SESSION_DIR": str(directory / "sessions"),
                            "HARNESS_SKILLS_DIR": str(directory / "skills"),
                            "OPENCODE_GO_API_KEY": "unused-test-key",
                            "HARNESS_TEST_OVERRIDE_MCP": "1" if override else "0",
                            "HARNESS_TEST_MCP_SERVER": server,
                        },
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), "interrupted")
                    self.assertFalse(directory.joinpath("sessions").exists())

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_forked_child_is_rejected_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.toml").write_text(
                'provider = "invalid-test-provider"\n', encoding="utf-8"
            )
            script = """
import os
import signal
import harness

try:
    harness.run(prompt="initialize runtime without a model request")
except RuntimeError:
    pass
pid = os.fork()
if pid == 0:
    signal.alarm(3)
    try:
        harness.run(prompt="must not wait on inherited runtime threads")
    except RuntimeError as error:
        os._exit(0 if "after fork" in str(error) else 1)
    os._exit(2)
_, status = os.waitpid(pid, 0)
raise SystemExit(os.waitstatus_to_exitcode(status))
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                env={**os.environ, "HARNESS_CONFIG_DIR": root},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
