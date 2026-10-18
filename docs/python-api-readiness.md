# Python API: PR and release review

## Decision

The branch is ready for PR review as a **source-installed native MVP**, subject
to green CI. This is not approval to publish platform wheels or a PyPI release.
The new Linux/Python 3.10 and macOS/Python 3.14 package jobs must run in CI;
local checks covered macOS with Python 3.14 only.

The API is synchronous `harness.run(prompt=..., options=...)`, returning final
answer text. It uses normal configuration and credentials unless supported
call options override them. MCP servers and skill directories can be replaced
or disabled per call. It does not yet expose every runtime setting directly.

## Findings fixed during review

| Finding | Change |
|---|---|
| Uncommitted MCP and skill work | Saved in `abbeac3` (`feat(python): configure MCP servers and skill directories per call`). |
| Python metadata could retain an old release version | Maturin now takes the version from the Rust workspace; the example lockfile no longer fixes a separate Python version. |
| Full MIT notice was not explicitly included in Python artifacts | Added `python/LICENSE` and the package `license-files` declaration; checked the built archive and installed distribution. |
| Cargo-only CI did not test the Python install path | Added package jobs that build an sdist, install it into a fresh environment, check version/license/import, and run the existing Python tests. |
| Retry instructions requested only a continuation, while final-answer collectors discarded failed-round text | Python/headless explicitly select replacement recovery and request a complete final answer. Incremental TUI/ACP output keeps continuation recovery, avoiding duplicate text. A mock-provider test checks both contracts through shared assembly; a headless check verifies replacement output is not duplicated. |

The packaging fixes are in `6ed3cca`; the recovery fix starts in `b9a15b5`.
Recovery policy is explicit through `agent::StreamRecovery`: streaming hosts
keep the default `Continue`, while final-only hosts select `Replace`.

## Local validation

- Built a source archive with maturin through `uv build --sdist`.
- Checked that the archive contains the workspace manifests and lockfile,
  required local Rust sources, Python package, and MIT notice.
- Installed the archive into a fresh virtual environment, outside the checkout.
- Checked native import, package version, and the installed full license text.
- Ran the Python API tests without real credentials or live provider requests.
- Ran focused Rust SDK, MCP replacement, skill-selection/child-propagation,
  recovered-output, and unrecovered-error checks during development/review.
- Ran targeted Clippy, rustfmt, whitespace, and CI YAML syntax checks.

No full workspace suite or full release build was run locally. Those remain
CI responsibilities. The tests use mock providers, temporary configuration,
and short-lived test MCP processes; they do not prove live OAuth refresh or
real provider/MCP interoperability.

## Requested Tokyo weather smoke check

A separate uv consumer project installed the committed feature through a pinned
Git dependency with `subdirectory = "python"`; native import reported version
0.6.1. No standalone Harness executable or wheel host was required.

The requested live run selected only the `weather` MCP entry from
`~/.config/harness/config.toml`, provider `openai-codex`, and model
`gpt-6.1-luna`. The MCP startup completed, but model validation stopped the
turn with:

```text
OpenAI Codex model `gpt-6.1-luna` is not in the supported model catalog
```

The current local catalog includes `gpt-6-luna` and `gpt-6.1-sol`, but not the
requested ID. No fallback model was used, and no weather tool call was recorded.
The Codex request did not reach the provider endpoint. This does not establish
whether the remote account supports the requested model.

The system config was unchanged. Audit events were stored in an isolated
temporary session directory, not the user's normal session store. No credential
values were printed or added to the repository. The live-provider release gate
below remains incomplete: update/verify model support or agree on an alternate
supported model before repeating the request.

## Before merging

- [ ] All existing Rust CI and the new Python package jobs pass.
- [ ] Review the public option names and replacement rules. In particular,
      omitted/`None` lists inherit; supplied lists replace; empty lists disable.
- [ ] Confirm source-only MVP scope and the documented platform/lifecycle limits.

## Before a public Python package release

- [ ] Run a small live-provider smoke check with authorized test credentials:
      a successful prompt, a tool call, session resume, and cancellation. Include
      a real MCP server and an explicit skill directory. Check the affected
      providers' login/refresh behavior as appropriate. Do not store credentials
      in test fixtures or logs.
- [ ] Confirm ownership and availability of the `harness-agent` distribution
      name, the release version policy, and PyPI publishing permissions.
- [ ] If distributing wheels, build and smoke-test every advertised OS/CPU
      combination. ABI3 does not remove OS, architecture, libc, or deployment-
      target constraints. Audit linked native libraries, bundled notices, and
      artifact metadata before upload.
- [ ] Add deliberate Python artifact publication/release automation. The current
      GitHub release workflow still publishes CLI artifacts only; this review
      does not publish packages or add upload credentials.
- [ ] Write release notes that state this is an MVP, list supported platforms,
      and separate Python source installation from CLI installation.

## Documented limits, not additional MVP requirements

No async/streaming API, timeout, Python tool callbacks, result/session object,
or programmatic cancellation is exposed. Provider credentials still use the
normal environment/auth store. Compaction and subagent limits still come from
`config.toml`. Bash and MCP tools are not sandboxed and run without confirmation.
In-process failures share risk with the host script. Python subinterpreters and
fork-after-runtime-start are unsupported. See [the Python guide](../python/README.md).
