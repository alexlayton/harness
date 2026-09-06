# Harness improvement TODO

This TODO is the implementation follow-up to `improve-plan.md` and the branch audit. Work is ordered by merge risk. Each completed item should have focused regression tests, `cargo fmt`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, and relevant tests before committing.

## Critical findings — implement in this order

### 1. Make handle-relative file mutation portable and robust

**Scope:** `crates/tools/src/file_mutation.rs`, `crates/tools/src/vfs.rs`, write/edit tests.

- Do not use Linux-only `O_TMPFILE`/`AT_EMPTY_PATH` behind a broad `cfg(unix)`; macOS is a CI target.
- Keep the validated parent directory handle and use an unpredictable same-directory temporary name created with `openat(CREATE|EXCL|NOFOLLOW)` where needed.
- Commit with handle-relative `renameat`, preserving atomic replacement and avoiding ancestor re-resolution.
- Provide a Linux optimization only if it has a portable fallback for filesystems without `O_TMPFILE`.
- Preserve existing file permissions for edit operations.
- Ensure cancellation cannot report failure after the destination has already committed.
- Add Linux/macOS-compatible tests for new-file writes, replacement, executable-mode preservation, cancellation, and temporary-file cleanup.

### 2. Replace racy stale sidecar locks

**Scope:** `crates/session/src/store.rs`, `crates/auth/src/storage.rs`.

- Use an advisory lock if it works for all current access patterns; otherwise design an atomic ownership/steal protocol.
- Never check a nonce and then unlink a pathname that may have been replaced.
- Use an unguessable owner nonce, private creation modes (`0700` directories/`0600` files), and propagate metadata/write/sync failures.
- Release only the lock instance still owned by the guard.
- Do not steal locks whose recorded process is alive, regardless of age.
- Add barrier-controlled stale-stealer races, live-owner tests, replacement-owner release tests, permissive-umask tests, and injectable failure coverage.

### 3. Preserve complete tool-call/result groups across compaction

**Scope:** `crates/compact/src/plan.rs`, `crates/session/src/model.rs`, tests.

- Never choose a boundary that leaves a standalone tool result without its call or splits a parallel call batch.
- Use the shared provider-history interpretation for planner and reconstruction.
- Validate compaction boundaries against call/result relationships, not only sequence monotonicity.
- Cover standalone and embedded calls, parallel calls, repeated compactions, post-summary events, malformed/future boundaries, and restart equivalence.

### 4. Retain validated workspace capabilities through path I/O

**Scope:** `crates/tools/src/vfs.rs`, `lib.rs`, `read.rs`, `write.rs`, `edit.rs`, context/skills discovery.

- Do not reopen the workspace root by pathname after lexical/canonical validation.
- Retain the root capability created at registry/tool construction and perform all final opens/commits relative to it.
- Open context and skill files relative to retained validated discovery-root handles with no-follow semantics.
- Ensure skill allowlists do not recanonicalize mutable pathnames at read time.
- Add deterministic barriers between validation and I/O, including root/ancestor replacement and skill-file replacement.
- Keep absolute and `@`-prefixed workspace paths working and preserve error-message contracts.

## High findings — next wave

- Repair UTF-8-byte crash tails in session recovery; preserve valid bytes and repair only an unterminated final fragment.
- Make export redaction recursive and comprehensive, including opaque content, standalone calls, arbitrary secret-bearing argument strings, diagnostics, metadata, and headers.
- Make bash process-group cleanup RAII-safe on future drop/abort, kill direct children on non-Unix, avoid unconditional normal-exit grace delays, and test descendant termination.
- Make auth refresh cross-instance/process safe; persist Copilot credentials before optional discovery; enforce device-flow expiry during requests.
- Pass API secrets into shared HTTP redaction before truncation; preserve retry hints structurally.
- Stop provider adapters immediately after valid terminal events; buffer/validate malformed tool-call batches and preserve Codex opaque-state order end to end.
- Quarantine the agent on deferred-sync, manual compaction, and model-change persistence failures; include pending user input in compaction planning.
- Own and reap ACP sessions on disconnect; move filesystem/MCP assembly out of the JSON-RPC loop; fail deletion on real filesystem errors.
- Enforce MCP frame/catalogue/server-count limits before unbounded materialization and shut down servers concurrently.
- Split untrusted TUI newlines into logical rows and make live/history rendering viewport-bounded and incremental.

## Remaining correctness and acceptance gaps

- Validate compaction reserve/window after defaults are applied and resolve process-level relative config/auth/log paths before worktree `chdir`.
- Detect overlapping exact edit matches, preserve edit file modes, and support advertised absolute/`@` paths.
- Make session listing recover crash-tail files, derive titles consistently, and avoid allocating full event payloads for metadata-only listing.
- Include omission markers within compaction input byte budgets and retain final synthesis reserve in subagent context budgets.
- Ignore empty SSE data frames.
- Finish dependency cleanup, including the duplicate Base64 version, and correct Codex user-agent/version/header duplication.

## Baseline and handoff checks

At the audited baseline, Linux checks passed:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked   # 462 tests
cargo build --workspace --locked
git diff --check
```

Repeat focused tests after each item and the complete checks before handoff. Commit each coherent fix with a conventional one-line subject, for example:

```text
fix(tools): make handle-relative mutations portable
fix(session): make stale-lock ownership atomic
fix(compact): preserve complete tool-call batches
fix(tools): retain validated workspace capabilities
```
