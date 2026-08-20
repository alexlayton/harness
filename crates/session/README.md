# Harness sessions

Harness sessions are durable conversation logs owned by the `session` crate.
The session format follows the JSONL conventions implemented here and
documented below: a versioned header, append-only event envelopes, and
explicit compaction summaries that characterize provider context. It is a
Harness-owned format (there is no import/export compatibility with other
tools' private on-disk schemas). The agent owns orchestration and the TUI
only renders session status; neither layer knows the JSONL layout.

## Storage and workspace scope

The default directory is `~/.harness/sessions`. `HARNESS_SESSION_DIR` replaces
that directory, while `HARNESS_STATE_DIR` selects its parent (`$dir/sessions`).
Each workspace is stored in a separate derived directory below the root. The
workspace path is also recorded in the header and is checked when a session is
loaded, so a session from another project cannot accidentally be attached.
Session directories are private (`0700`) and session files are written with
private permissions where supported. Session files can contain source code,
commands, tool output, and reasoning traces; do not share them without
reviewing their contents.

## JSONL format (version 1)

The first line is a header:

```json
{"version":1,"type":"session","session_id":"…","timestamp":"…","data":{"format_version":1,"id":"…",…}}
```

Every later line is an append-only event envelope:

```json
{"version":1,"type":"user_message","session_id":"…","event_id":"…","sequence":1,"timestamp":"…","data":{"role":"user","content":[{"type":"text","text":"…"}]}}
```

The event payloads include user/assistant messages, reasoning, tool calls and
results, model changes, usage, metadata changes, cancellation/errors, and
compaction summaries. Tool output and reasoning are persisted by default.
Unknown event types are retained and exported, but are excluded from provider
context until a reader understands them. IDs and sequence numbers make
ordering and tool-call pairing inspectable.

Writes append one complete newline-terminated record, flush it, and call
`sync_all` before returning. A create-new sidecar lock prevents concurrent
Harness writers from interleaving records. Metadata changes are events rather
than rewrites, preserving the append-only history. The header is the initial
metadata snapshot; current title, model, compaction, and usage values are
obtained by replaying those events. Header/index replacement files use a
temporary file and rename.

A malformed unterminated final line is treated as an interrupted write and is
ignored on load. A malformed middle or newline-terminated line is an error.
Only that clearly incomplete trailing fragment is ignored; the rest of the
file stays authoritative. Loading an exported file by path adopts a copy under
the configured store before the agent appends new events.

## Lifecycle and commands

The agent creates a persisted session at startup. `/new` creates another file
without deleting the old one. `/load <id>`, `/load <unique-prefix>`,
`/load latest`, or `/load <path>` restores event context and metadata. When
startup has just created an empty placeholder, `latest` skips that placeholder
so it resumes the previous non-empty conversation. Saved provider/model
metadata is shown in listings but does **not** replace the
provider/model selected at startup. `/sessions` lists sessions scoped to the
current workspace. `/export [path]` writes canonical JSONL to the requested
path, or to a generated file in the current directory. `/compact` is available
as a deterministic local hook, although the default policy is intentionally
conservative.

Exports are canonical JSONL and can be validated independently by the codec.
`ExportOptions` can omit reasoning/tool output, cap tool output, or apply an
explicit conservative secret redaction pass. Redaction is opt-in because the
normal persisted log is intended to let a user pick up where they left off.

## Context and compaction

`Session::context_messages` reconstructs provider-ready `llm::Message` values.
It groups standalone tool-call events into assistant messages, preserves call
IDs and results, and omits an incomplete final tool call rather than sending
invalid history. The agent writes synthetic cancelled/error tool results when
it can observe an interruption. A deterministic compaction summary is an
ordinary durable `compaction` event with a `compacted_through` sequence
boundary; old events are never deleted. Model-assisted summarization is
intentionally not part of version 1.
