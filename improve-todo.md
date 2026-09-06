# Harness improvement TODO — audit of `improve` vs `improve-plan.md`

Branch: `improve` vs base `9cfa5fc674784092346e682fa4a61fb473590248`.
Scope: 73 commits, 77 files, `+11548/-3187`.
Method: current code + `git diff 9cfa5fc..HEAD` vs plan Implementation / Required tests / Acceptance.

Legend: `DONE` / `PARTIAL` / `MISSING`.

## Phase 1 — Persisted-state correctness (`crates/session`, `crates/compact`)

### SESSION-1 Preserve recent tail after compaction — DONE
- `crates/session/src/model.rs:842 latest_compaction_boundary`, `:864-883 events_after_latest_compaction` returns latest summary first, then `seq > compacted_through && !CompactionSummary` in order. Older summaries excluded.
- `validate_record:~1146-1160` rejects `compacted_through >= record.sequence` (self/future) and `<= previous` (monotonic). `validate_compaction_boundary:1176-1212` requires existing event, `boundary >= summary_sequence` err, splits unresolved `ToolCall` err.
- Single source: `context_messages:742-743` and `plan_compaction:plan.rs:55,72` share `latest_compaction_boundary` + `events_after_latest_compaction`.
- Tests `model.rs:~1555-1700`: `user1,assistant1,user2,assistant2,summary(through=2) → [5,3,4] summary,user2,assistant2`; post-summary append `[5,3,4,6]`; repeat retains only newest `[8,7]`; invalid self `3`/future `99`/repeat `2`; `compaction_boundary_keeps_tool_call_result_pairs [5,2,3,4]`; `compaction_rejects_a_boundary_inside_a_parallel_batch`.

### SESSION-2 Repair incomplete crash tail — DONE
- `crates/session/src/codec.rs:109 TailRecovery{recovered,valid_bytes}`, `decode_session_file:135`, `decode_session_file_bytes:143-181` UTF-8 tail handling, `decode_session_lines:~270-420` tracks `consumed_bytes`, recovers only `recover_trailing && unterminated && last_line` serde err with `valid_bytes=consumed_bytes`.
- Load never mutates; `load_session_file_for_append:store.rs:522-530` returns `(Session,TailRecovery)`.
- `SessionStore::append_event:store.rs:230-340` under `SessionLock`: fast-path / `reconcile_external_tail`, else reload; if `recovered` → `truncate_to_valid_tail:534-544 set_len(valid_bytes)+sync_all` before append. Terminated/middle corruption remain `Err`.
- Tests: `store.rs:1605 append_repairs_an_unterminated_crash_tail`; `codec.rs:654 valid_unterminated_final_line_is_not_truncated`; `store.rs:1451 append_after_valid_unterminated_record_inserts_separator`; `codec.rs:668 terminated_malformed_final_line_is_an_error`; `codec.rs:629 middle_corruption_is_not`; `codec.rs:680 recovery_reports_offset`; `store.rs:1655 append_repairs_utf8_crash_tail (\xc3 split)`.

### SESSION-3 Comprehensive export redaction — DONE
- `crates/session/src/export.rs:~180 transform_record` runs after omission/truncation; `transform_metadata` preserves IDs/numerics, transforms title/provider/model/workspace; `transform_tool_call` preserves id/name.
- Recursive `redact_json:475` (keys `api_key/apikey/authorization/password/secret/token`, recurse arrays/objects, `redact_text` on strings) applied to standalone `ToolCall.arguments`, embedded `ToolCall{arguments}`, `Opaque{data}`, `Unknown`.
- Text `redact_text` on user/assistant `Text`/`Reasoning`, `ToolResult` via `transform_output` (omit→`truncate_utf8`→redact), `transform_reasoning`, `transform_summary`, titles/errors/cancels.
- Tests `export.rs:543,594`: `export_is_loadable_and_can_redact_tool_output`; `redaction_covers_every_secret_bearing_field` sentinel `sentinel-secret-9f3a1c` in header/title/provider/model/workspace, `MetadataChange`, user/assistant, embedded reasoning/opaque/tool-call, standalone `Reasoning`/`ToolCall`, both `ToolResult`s, summary, `TurnCancelled`, `Error`, `Unknown{password}`; asserts `!contains(sentinel)`, decodable, pairing intact, slim export still clean. Caveat: `redact_text:407` is key-anchored heuristic (bare secrets out of scope by design).

### SESSION-4 Lock ownership + private files — DONE
- DONE: `store.rs:SessionLock::acquire` + `auth/storage.rs:AuthFileLock::acquire` use `fs2::FileExt::try_lock_exclusive` + `LOCK_ATTEMPTS/WAIT` loop, `Drop{unlock}` retains sidecar inode. No PID/nonce steal path remains (superseded by advisory).
- DONE: `private_dir_all:store.rs mode 0o700`, `private_file:0o600+create_new`, lock sidecar `0o600`; mirrored `auth/storage.rs`. `ensure_private_*` fail-closed; `sync_parent` after create/rename; Windows ACL documented, no privacy claimed.
- Tests DONE: `concurrent_advisory_lock_contenders_never_overlap (store.rs, auth)`, `advisory_lock_serializes_contenders_and_retains_sidecar_inode`, `permissive_umask_still_yields_private_paths (store.rs, auth)`.
- Tests DONE (fail-closed injection): thread-local `#[cfg(test)]` fault flags + `HardeningFaultGuard` (auto-clear on drop) force each hardening step to return its real `Io` error — running as root makes genuine `chmod`/`fsync` failures untriggerable. `store.rs`: `directory_permission_failure_fails_session_create` (`secure directory permissions`, empty workspace dir), `file_permission_failure_fails_session_append` (`secure file permissions`, no event/file growth, reopen clean), `parent_sync_failure_fails_session_create` (`sync parent directory`). `auth/storage.rs`: `directory_permission_failure_fails_auth_save` (no credential file), `file_permission_failure_fails_auth_save` (previous credentials intact/readable), `parent_sync_failure_fails_auth_save`. Guards are scoped to the write call only (construction hardens/syncs the fresh root too) and dropped before reopen reads (reads also repair permissions, so reading armed would fail by design). Stale-steal-by-nonce tests N/A by design (no nonce protocol). Acceptance (no concurrent entry, no write-then-chmod) met via OS advisory.

### SESSION-5 Tool-call replay validation — DONE
- `model.rs:645 ToolCallTracker{pending,seen,completed,cancelled}`; `validate_tool_call_id:~1215` rejects empty id/name, duplicate pending, global `seen` reuse. Applies to standalone + embedded `StoredContent::ToolCall`; `ToolResult` requires non-empty + `is_pending` + FIFO.
- Shared `ToolCallTracker::record` via `validate_events`/`validate_next_event`/`validate_event_suffix` (`ToolValidationState`+`SuffixTracker`); `TurnCancelled` clears pending but never `seen`; `Error` keeps pending but `seen` blocks reuse.
- Tests `model.rs:~1740-1920`: `completed_tool_call_ids_cannot_be_reused` (with/without second result); `empty_tool_call_ids_and_names_are_rejected`; `cancelled_tool_call_ids_cannot_be_reused`; `standalone_and_embedded_calls_share_one_id_namespace`; `context_reconstruction_never_yields_orphan_results_or_dangling_calls`.

## Phase 2 — Authentication (`crates/auth`)

### AUTH-1 Device-flow polling — DONE
- DONE: `openai_codex.rs:OAUTH_BODY_LIMIT`, `read_bounded_body` used by `request_token_until` + `token_strict`; RFC8628 map (`pending→continue`, `slow_down→+5s`, `expired→Expired`, `denied→sanitized`) before status; `token_strict` 2xx-only no echo used by `refresh`/`login_browser`; cancel/expiry around sleep+request (Tokio-clock deadline: `grant=min(expires_in,3600)`, `Instant::now()+grant`, checks pre/post sleep + `sleep_until` in `request_token_until`).
- Tests DONE (full `login_device` loop, not helper-only): `device_login_pending_then_success_persists` (pending→success persists, reopened handle observes, `Started/Finished`, 3 requests); `device_login_slow_down_delays_the_next_poll` (injectable sleep records `[5,10]` — 5s default + `slow_down→+5s`; recording sleep avoids paused-clock/real-socket starvation); `device_login_expired_token_fails_without_persisting` (`DeviceCodeExpired`, no file); `device_login_access_denied_is_sanitized_without_persisting` (sanitized denial, no `shh-device-secret` echo, no file); `strict_token_exchange_rejects_polling_errors` (`400 authorization_pending` → status-only `400`, no `shh-strict-secret` echo); `cancellation_during_device_poll_aborts_login` (in-flight cancel → `Cancelled`, no file); `malformed_and_oversized_bodies_are_rejected` DONE; `device_poll_errors_map_before_status` DONE (helper-level expired/denied/unrecognized-no-echo). Helper-level `device_poll_pending_then_success` / `slow_down_increases_the_poll_interval` kept as unit contracts, marked superseded.

### AUTH-2 Refresh single-flight — DONE
- DONE: `refresh_lock` (codex + copilot); reload-recheck with short lock scopes; note Codex `&&` vs Copilot `||` divergence (both require the persisted generation to differ from cache AND be unexpired before adopting); `save_openai_codex_if_current` / `save_copilot_if_current` CAS under `AuthFileLock`, callers fall back on `false`; no blocking mutex across net.
- Cross-process lease decision: DOCUMENTED AS CAS-AT-SAVE, no `lease` symbol by design. A lease spanning reload→exchange→save would hold coordination across network I/O (explicitly forbidden: "do not hold a blocking mutex across async network work"); concurrent cross-process exchanges can still double-spend a single-use token, but the loser's stale completion can never overwrite the winner (CAS rejects it) and a retrying loser adopts the winner via reload-recheck. Pinned by `two_handles_racing_refresh_share_one_exchange`.
- Tests DONE (real `ensure_valid` path, single-use rotating fixture — second exchange 401s): copilot `concurrent_refresh_single_flights_on_one_network_call` rewritten (8 waiters + barrier → 1 exchange, all share `access-rotated`, persisted == newest incl. enriched `gpt-5.4`); codex `codex_concurrent_refresh_single_flights_on_one_network_call` (same shape, `refresh-rotated`); copilot `two_handles_racing_refresh_share_one_exchange` (two handles/one file, loser adopts winner, 1 exchange, persisted == winner); `compare_and_save_rejects_stale_refresh_generations` (storage-level CAS).

### AUTH-3 Callback + Copilot login — DONE
- DONE: `CALLBACK_OVERALL_TIMEOUT` 10m, `wait_for_callback`→`with_idle(10s)`, `read_callback_head` 512B chunks through `\r\n\r\n`, `CALLBACK_HEAD_LIMIT` 16KiB; `is_callback_denial` + sanitized denial; `login_with_events_and_persist` persist-before `MODEL_DISCOVERY_TIMEOUT` 15s `fetch_available_model_ids`, best-effort swallow; serial policy off path (`enable_known_models` not in login); `base_url_from_proxy_token`, `copilot_base_url`, `proxy_endpoint` HTTPS-only, reject userinfo/`/?#@`/bad port.
- Tests DONE: `idle_connection_does_not_block_a_later_valid_callback` (idle+fragmented); `live_denial_terminates_promptly` (live valid-state denial → sanitized `denied`, `Login failed` reply, resolves without idle timeout) + `callback_target_parsing_accepts_codes_and_denials` extended with near-miss non-denials (wrong error/missing state/wrong path); proxy parse/reject; `model_list_failure_still_leaves_a_usable_persisted_credential` (500 `/models` → exchanged `access-login` returned + persisted unenriched, `Finished` fires).

## Phase 3 — Tools (`crates/tools`) — DONE

### TOOLS-1 Symlink escapes — DONE
- `context_files.rs:10-16,223-243 read_contained_candidate`: `WorkspaceFs::open_root` canonical root, `canonicalize(candidate)+starts_with`, final `vfs::unix::open_file_relative(O_NOFOLLOW)`. External silently skipped, no contents in diagnostics. Contained symlinks allowed.
- `skills.rs:10-13,299-302,312,391-430 contained_path/is_contained/load_contained_skill/discover_dir`: canonical root once (`:468`), canonical every `SKILL.md`/root `.md`/base-dir, `starts_with`, retained-handle read. Base-dir symlink rejected before descent (`:321-322,351-354`). Allowlist from canonical `file_path/base_dir (:523-543)`; `<location>` canonical (safe).
- Tests `#[cfg(unix)]`: `context_files.rs:497-539 external_context_symlink_is_not_injected (EXTERNAL-SENTINEL-CONTENT)`; `skills.rs:831-890 external_skill_symlinks_are_rejected_but_contained_ones_work (TOP-SECRET-EXTERNAL)` + diagnostics clean.

### TOOLS-2 TOCTOU — DONE
- `vfs.rs:1-27` spike documents handle-relative `openat/O_NOFOLLOW` over capability/`openat2`, open-is-validation; non-Unix fallback explicitly narrows-not-closes.
- `vfs.rs:50-110 WorkspaceFs::open_root` (canonical+`O_DIRECTORY|O_NOFOLLOW` fd) + `split_relative`; `unix::open_dir_relative/open_file_relative/open_parent_relative/open_child_dir/open_child_file/confirm_contained` (dev/ino `..` walk) + `open_file_metadata`.
- `read.rs:187-240 open_target`, `write.rs:150-195 write_validated` + `file_mutation.rs:atomic_write_at (openat CREATE|EXCL temp + renameat same parent_fd)`, `edit.rs:227-292 execute_edit_validated` (handle read→match→re-read→commit). Errors `cannot read/write/edit {path}: {io}` no outside leak.
- Tests (resolve→`remove_dir_all`+`symlink(outside)`→execute): `read.rs:547-579`, `write.rs:284-310` + `write.rs:248-283 retained_workspace_capability_survives_root_path_replacement`, `edit.rs:953-993`, `lib.rs:730-767`.

### TOOLS-3 Bash exclusive + tree kill — DONE
- DONE: `bash.rs:50-62 command_concurrency→Exclusive`, `:129-133 concurrency()`; word-level classifier removed. `MAX_TIMEOUT_SECS=86400 :37-48,72-81,137-160`, `checked_add` → tool error. `process_group(0) :203-210`. `ProcessGroupGuard:461-546 (SIGKILL on drop)` + `terminate_tree (SIGTERM→alive?→KILL_GRACE 500ms→SIGKILL)` on timeout/cancel/drop + after `Exited` for backgrounders; `child.wait()` reaps. `read_bounded_tail :245-305,373-410` + `timeout_at(drain_deadline) join!` one `DRAIN_TIMEOUT=1s`.
- Tests DONE: `bash.rs:707-734 every_bash_invocation_is_exclusive`, `:741 oversized`, `:749 max_rejects_u64_max`, `:767 background_descendant_killed_after_return`, `:815 timeout_kills_descendants`, `:838 cancellation_kills`, `:794 aborting_execution_kills`, `:583 timeout_kills_command`; `held_stdout_and_stderr_share_one_drain_deadline` (setsid-detached survivor holds both pipes past shell exit; ~1s outer sleep + ~1s shared drain asserts 1.5s ≤ elapsed < 2.9s, proving one shared `DRAIN_TIMEOUT` instead of two sequential waits; survivor reaped via pid file).

### TOOLS-4 Exact byte-preserving edit — DONE
- `edit.rs:415-490 apply_edits_exact` via `match_positions (char_indices+starts_with)` exact bytes; `strip_bom:491-495` split+reattach; spans vs original, overlap rejected, apply `rev`; BOM/mixed CRLF/LF preserved; pre-commit re-read (`:266-280` handle, `:343-351` fallback); `unicode-normalization` removed from `Cargo.toml`.
- Tests: `:806 preserves_bom_and_crlf`, `:828 mixed_crlf_outside_spans`, `:840 typographic/trailing-spaces_require_exact`, `:677 multiple_disjoint`, `:877 batch_match_independently`, `:732 rejects_missing_duplicate_overlap_empty`, `:904 proptest byte_preservation_outside_spans`.

## Phase 4 — Provider/streaming (`crates/llm`) — DONE

### LLM-1 Protocol-confirmed completion — DONE
- `openai_chat.rs:280-285,328-352 [DONE]/usage→Done+done=true`, `finish():338 Stream` if `!done`; `openai_responses.rs:282-313,334-341 completed|incomplete→done+drain+Done`, `finish→Stream`; `codex_responses.rs:133-149` reuses `is_done/finish`; `anthropic.rs:415-436,499-506 message_stop→drain+Done`, `failed/error→Stream`, `finish→Stream`.
- Tests: `chat:valid_done_succeeds`, `text_then_eof_without_terminator_fails`, `partial_tool_call_then_eof_fails_no_completed`; `responses:finish_without_terminal_is_stream_error`, `text_then_eof_fails`, `finish_noop_after_completed`; `anthropic:finish_without_message_stop_is_stream_error`; `sse:parses_done_and_chunk_boundaries` split-terminal.

### LLM-2 Incomplete/usage/ordering — DONE
- `openai_responses.rs:282-311 incomplete` terminal, `status+incomplete_details.reason→stop_reason (incomplete: reason)` + usage (all normal-stop by comment); `codex_responses.rs:63-131 convert_input` single ordered pass, replay only `provider==openai-codex`; `anthropic.rs:360-391 saturating input+creation+read→input_tokens`, `cached=read`.
- Tests: `incomplete_terminal_event_succeeds_with_reason_and_usage` (responses+codex); `codex_opaque_state_preserves_order_before_multiple_tool_calls (enc1,c1,enc2,c2, no rs_9)`; `anthropic:full_prompt_usage_sums_all (100+50+80=230, cached 80)`.

### LLM-3 Malformed tool calls — DONE
- `openai_chat.rs:381-420 flush_calls` trim ID/name, empty→`Parse`, duplicate via `seen_ids+validated`, no `ToolCallComplete`; `accumulate_tool_call:355` no synthetic ID; `responses.rs:234-279 pending_calls` held until terminal; `anthropic.rs:447-486 finish_tool` same, `completed_tools` until `message_stop`.
- Tests per dialect: `rejects_missing_id/name|blank`, `rejects_duplicate_ids_in_parallel`, `fragmented_valid_ids_assemble (ca+ll-1→call-1, re+ad→read)`.

### LLM-4 Bounded/correct SSE — DONE
- `sse.rs:36-38,44-77,110-123 MAX_LINE 1MiB, MAX_EVENT 8MiB`; `push_bytes/finish→Result<Stream>`; `dispatch` drops `!has_data||empty` even with `event`, clears `event`; CRLF/multiline/comment/unterminated-final; `SseStream:150` propagates.
- Tests: `empty_data_frames_are_ignored`, `event_only_frames_emit_nothing_and_do_not_leak`, `exactly_at_limit_succeeds_over_limit_fails_across_chunks`, `crlf_and_multiline_data_still_work`, `parses_comments_crlf_and_multiline`.

### LLM-5 Bounded HTTP/redaction/retry — DONE
- `http.rs:144-163,165-184 check_status_with_secret→http_redacted_with_retry_after`; `bounded_error_body` streams `bytes_stream` to `MAX_ERROR_BODY 16KiB`, never `response.text()`; `error.rs:56-112,118-136,142-166 http/truncate_body(_,2048)`, `redacted()` for `Http/Stream/Parse/Auth`, UTF-8 `len<=max`, empty if suffix won't fit, `parse_retry_after_value` delta+http-date, `MAX_RETRY_AFTER 300s`, `retry_after_secs` 429-only; `retry.rs:16-59,61-75 MAX_ATTEMPTS 3`, `max(backoff+jitter,retry_after)`, injectable sleeper, LCG jitter; `providers/github_copilot.rs:543 redact_error→redacted`, `providers/openai_codex.rs:260` same.
- Tests: `http:shared_http_errors_redact_keys_and_preserve_retry_headers`, `stalled_response_body_times_out`; `error:error_bodies_echoing_key_are_redacted`, `error_bodies_stay_bounded_and_utf8 (0,1,2,5,8,12,16,64+4MiB)`, `retry_after_parses_seconds_dates_and_caps`; `retry:retry_counts_attempts_and_honors_retry_after`, `retry_after_overrides_backoff_and_nonretryable_returns`.

## Phase 5 — Agent runtime (`crates/agent`, `crates/compact`)

### AGENT-1 Turn-boundary — DONE (minor test gap)
- `agent/turn.rs:26 execute_turn` single executor (child token, `flush_deferred_sync`, `Shutdown/Quarantine/Continue`; `:62 run_turn_body`, `:459 TurnControl`); `persistence.rs:54 flush_deferred_sync`; `commands.rs:510 handle_invoke_skill→:543 execute_turn`, `:223 handle_compact_session_boundary+:258 flush`, `:276 handle_set_model_boundary+:285 flush`; `mod.rs:288,324,337,361` all return `TurnControl`, run-loop owns single `TurnFinished (:297,:369 quarantine)`.
- Tests: `mod.rs:1476 skill_turn_failure_quarantines_and_never_runs_queued_work`; `:1568 shutdown_during_skill_turn_exits_the_agent`. GAP: no explicit skill/manual-compact/model-change deferred-sync regression.

### AGENT-2 Shared dispatch — DONE (minor test gap)
- `agent/tool_dispatch.rs:64 plan_tool_batches`, `:306 execute_tool_batch<C,H>` generic (`DispatchCancellation/CallState/CallOutcome/BatchOutcome`, `ToolDispatchHooks/Agent/Noop`), `:199 CancellationControl`, `:229 ParentDispatchControl`, `:288 poll_control_now`; ready-drain before `batch_cancel.cancel()` retains `Completed`; `launched && class!=ReadOnly→"cancelled; execution status unknown"` else `"cancelled"`; stable IDs/original-order `slots[index]`; `dispatch_tool_batches:473+` hooks-vs-durable order comment. Child `subagent.rs:30,571,589` reuses same with `Noop`, no-recursion via registry `:250`.
- Tests: `tool_dispatch.rs:615 shared_executor_keeps_original_result_order`, `:640 shared_executor_distinguishes_launched_and_queued_cancellation`; parent ordering/concurrency `mod.rs:513,544,584,635,935,1013`. GAP: no explicit child ready-racing-cancellation retained test.

### AGENT-3 Compaction cancel/no-session — PARTIAL (impl DONE, tests MISSING)
- DONE: `compact/summarize.rs:Cancelled`, pre/post `cancel.is_cancelled()→Cancelled`, `model_summarize select! stream vs cancel`; `agent/compaction.rs:132 Cancelled→Shutdown|Notice("compaction cancelled"), Ok(false)` no fallback/summary; `:143` usage only `Model{usage:Some}`; pre-turn interrupt `turn.rs:70-125 (select! compaction vs cancel vs Interrupt, queued buffer, persist_cancelled+TurnFinished)`; manual `commands.rs:223-264` same; no-session `compaction.rs:79 should_auto_compact false if none`, `:108 Notice unavailable`, `:195 try_overflow Ok(false)`.
- MISSING: pre-cancelled/mid-stream persists-nothing, hanging pre-turn prompt return, no-session small-window no-repeat-failures, missing-usage-no-totals. Existing `mod.rs:2154,2257,2333,2399` trigger/fallback/overflow/manual only.

### AGENT-4 Atomic model change — PARTIAL (impl DONE, tests MISSING)
- DONE: `commands.rs:295 handle_set_model` resolve `next_provider/canonical` w/o mutate→persist `ModelChange?`→commit provider/model+`runner.update_model()` (`:304` comment), `:350 ModelChanged`, reset tokens/window, `:366 spawn_model_metadata`; `:555 spawn_model_metadata` one `5s timeout(list_models)`+cancel, send only `Some`; `compaction.rs:15 apply_model_metadata` same result + stale guard; `mod.rs:236` startup off path, `:250 try_recv` drain + `:268 select metadata vs input` anti-starvation.
- MISSING: failed-persist-leaves-parent+subagent, one-switch→one-request, hanging-metadata-no-freeze, metadata-turn1-before-queued-turn2.

### PERF-1 Unified estimator — PARTIAL
- DONE: `compact/estimate.rs:27 estimate_provider_context_tokens(system,tools,messages)` (system+name/desc/parameters+role+`Text(full)/ToolCall/ToolResult(full)/Opaque(provider+data)`, `Reasoning→0`); `agent/compaction.rs:50 context_tokens_estimate(extra once)`, `:62 estimate_history_tokens` same `system_prompt_with_workspace_context+snapshot.definitions+history` durable/ephemeral; `plan.rs:177 event_tokens` same + prefix-sum `choose_cut`.
- Tests: only `estimate.rs:95` full-result increases. MISSING: durable/in-memory equivalence, active-summary contributes, project-context/schemas trigger, no double-count after `Done`. Note: `should_auto_compact→context→compact_and_reload` re-estimates same range 3×/pre-turn vs "avoid multiple scans".

### PERF-2 Hard byte limits — DONE
- `compact/serialize.rs:29 serialize_events` newest→oldest, separator accounting, oversized-newest `truncate_bytes` prefix, `truncated` flag, `marker_reserve` then `truncate_bytes(text,budget)` UTF-8 loop; `summarize.rs:195 append_with_limit`, `:212 append_file_lists truncate_bytes(combined,max_summary_bytes)`, `:223` deterministic, `:125,134` input cap + `OMISSION_MARKER` (`truncated` used).
- Tests: `serialize.rs:428 transcript_budget_includes_separators_and_oversized_newest (0..=128 len<=budget, é×200)`, `:451` zero-budget + per-item; `summarize.rs:384 len<=max (128)`. GAP: no explicit 1000s-paths test (cap holds via combined truncate).

## Phase 6 — Frontend (`crates/harness`, `crates/tui`)

### HARNESS-1 ACP errors/usage — DONE
- `harness/acp.rs:120-180 PromptTracker{in_flight:{error,cancelled}}` + `mark_error/clear_error` only on `TextDelta` (metadata doesn't clear); `~463,518-522 UsageUpdated→None`, `ContextUsageUpdated→UsageUpdate`; `~1030-1090 forward_events`: `cumulative_cost→UsageUpdate` only, `TurnFinished→EndTurn/Cancelled`, `Error→mark_error`, unconditional `resolve(id,None)` on stream-end; `_context_window` unused (first-model heuristic removed); tests `~1049,1532 EndTurn`, `~1059,1064,1266,1273` billing-vs-context.

### HARNESS-2 ACP lifecycle — DONE
- `acp.rs:89-180 SessionHandle{input_tx,cancel,agent_task,forwarder_task}`; `:189-360 serve` owns `AcpState{sessions,prompts}`, disconnect drains + `shutdown_session` each; `:611,645-720 load_session_inner` trim+`SessionId::parse`+canonical `to_string`+`already loaded` reject; `:720-753 delete_session` reject if `in_flight`, `shutdown_session` under `SESSION_TASK_TIMEOUT=2s` (aborts both after one deadline); `:778 delete_session_everywhere` workspace scan, `NotFound→Ok`; `:887-1020 build_session_stack` via `spawn_blocking`+`timeout`, `acp_mcp_servers` rejects HTTP/SSE, `spawn_agent builder.build()+timeout`, duplicate-after-assembly abort (`~1009` race comment); `new/load_session tokio::spawn` off dispatch.

### HARNESS-3 Headless — DONE
- `harness/headless.rs:120-290 pending_text/active_tools/error_pending/saw_turn_finished`; `TextDelta→push`, `ToolCallStarted→verbose+clear`, `ToolCallFinished+active==0→clear`, `TurnFinished+active==0+!empty→stdout+newline else flush`, `Error→clear+stderr+error_pending=true`, `UsageUpdated/Notice/etc` don't clear (`error_pending=false` only in `TextDelta`); `write!/writeln!+?` propagates; `reasoning_open` delimiter only on `ToolCallStarted/TurnFinished`; `!saw_turn_finished→Err`; `run_headless_resolved agent_task.await.context`; `:130` SIGINT; `main.rs:~130-170 resolved_prompt=resolve_prompt()` before registry/store/bundle, `needs_session_store`, ACP early-return, `run_headless_with_prompt(...,prompt,rendered)` no reload.

### TUI-1 Sanitize — DONE
- `tui/render.rs:561-660 sanitize_terminal_text`: `\n` keep, `\t→4sp`, `ESC/CSI/OSC/string/C0/C1/DEL` via `skip_escape_sequence/skip_csi_sequence/skip_string_sequence`, `is_control/0x7f` drop; used in `plain_text/owned_markdown/fit_line_to_width/wrap_text` + `app.rs:639,725,2497,2524-2547,3103`; `Paste→sanitize`; tests `sanitizer_removes_terminal_controls_and_expands_tabs`, `consumes_unterminated`, `line_to_ansi_sanitizes OSC/CSI`.

### TUI-2 Row-width/restoration — PARTIAL
- DONE: `render.rs:358 fit_line_to_width` + `wrap_text` + `line_width` + `<=width` tests `~926-956`; `prefix_message_lines fit_text_to_width(prefix)` + width-1 invariant; `app.rs:700-713` separate `Home/End` vs `Ctrl+A/E`, `PageUp/Down→{}`; `activity_region_row:Option` from `build.activity_row (:583,2130,3753)`; `TerminalModes{raw,bracketed,keyboard,newline}` + best-effort `restore()` first-error + `Drop` + panic-hook.
- GAP: `metadata_lines(~2520)` builds `cwd(branch)`/`provider·model` without explicit `fit`; combining/ZWJ/zero-width only via `UnicodeWidth` (no explicit ZWJ test); width 1–3 property, height 1–2 spinner-clamp correctness tests not found.

### TUI-3 Commands/completion — DONE
- `tui/commands.rs:parse_command_with_skills` alias only if `rest.is_empty()` + non-colliding else `parse_command` error (never silently drops); `paths.rs:extract_at_prefix token_end` stops at `whitespace/) ] } , ;` (preserves punctuation); `app.rs:1093-1200 request/apply_path_completion` debounced `200ms`+`spawn_blocking`+`generation/cancel`, merges static `candidates_at_cursor` + `find_path_candidates` via `merge_candidates` (`/load ./…` + session IDs); `~3992 direct_provider_prefix_requests_its_model_catalogue`; `tool_lines` expanded `output_tail` + `error.lines().skip(1).take(TAIL)` + `running…`, collapsed first-line only.

## Phase 7 — Token/memory/startup

### PERF-3 MCP bounds — DONE
- `mcp/config.rs`: `Debug` redacts `args/env` counts, `redact_url`, `MAX_SERVERS 64/1MiB`, name/control checks; `runtime.rs`: `MCP_INITIALIZE/LIST/SHUTDOWN_TIMEOUT`, `JoinSet` concurrent connect + sorted order, `join_all+timeout` single global shutdown, `spawn_stderr_reader` bounded `MCP_STDERR_CHUNK_BYTES` byte-count only (`debug bytes`); `list_tools_bounded timeout_at` + `MAX_REMOTE_TOOLS 256` + `validate_remote_tool` byte accounting; `tool.rs`: `MAX_NAME/DESC/SCHEMA_DEPTH 32/NODES 10k/STRING 64k/BYTES 256k`; `output.rs`: `OutputWriter MAX 20k+notice`, `utf8_prefix_len`, compact `to_writer`, `structured_duplicates_text`, `cap_display` same budget.

### PERF-4 Startup context once — DONE
- `main.rs:context::load_context_bundle(workspace,cwd,no_files,interactive?)` once, `rendered/display_paths` reused; `headless_with_prompt` takes `prompt+project_context` (no second discovery; `project_context_for` only in `#[cfg(test)]`); ACP returns before registry/store/context; `resolved_prompt` before store/registry (blank→no mutation); sync `join!` replaced (comment "only ceremony"); worktree env-normalize `set_var` before `chdir`.

### PERF-5 Session append/listing — PARTIAL
- DONE per `completed.md:9` + `session/benches/store.rs`: streamed indexing + incremental append validation (`reconcile only appended tails`, `validate current appends incrementally`, `stream index metadata`, `avoid cloning validation history`, `listing recovery-aware metadata-only`).
- OUTSTANDING: 1k/10k near-linear benchmarks + deeper external-file reconciliation (two-stores-stale, replacement/truncation safe-full-validation, no full-history clone for metadata booleans).

### PERF-6 Subagent + TUI rendering — PARTIAL
- Subagent DONE per `completed.md:10` (`perf(agent): bound subagent context growth`, reserve for final synthesis, truncate old evidence).
- TUI PARTIAL: `StreamMarkdownCache/refresh_stream_markdown_cache`, `history_window` newest→oldest budget-stop + `…older rows above`, `output_tail` bounded tail present; but `output_tail: lines().count()` full walk, no quadratic-guard benchmarks/instrumentation.

## Phase 8 — Cleanup — PARTIAL
- DONE: several dep removals (`eventsource-stream/nom`, `tracing` in `tui`, `tempfile` in `compact`, `ring`/`tokio-util/compat`/`base64` — verify via `cargo tree --duplicates`), redundant compact token surfaces (`estimate_text_tokens`, `estimated_tokens_freed`), shared dispatch/lifecycle/scan-waits refactors (`refactor: remove dead runtime surfaces and share scan waits`, `chore: remove unused session and auth APIs`).
- OUTSTANDING: remaining dead surfaces (`SerializedTranscript::truncated` use-or-remove now used by PERF-2 — confirm, `run_agent` history, OpenCode catalogue, Codex header helper, `FileSearchIndex::_frecency`, session/auth helpers — caller sweep), SSE adapters / model-list fetching / find-grep-wait sharing, CLEANUP-4 test replacements (compaction defaults→policy tests, jitter→deterministic retry, OpenCode catalogue→routing, headless no-session `HARNESS_SESSION_DIR`, ACP load round-trip, synthetic-EOF→strict-terminal).

## Phase 9 — Config/CI/docs

### CONFIG-1 — DONE in code, test-matrix gap
- `harness/config.rs:~153-180 CompactConfig::validate` finite `0.0-1.0`, `1..16MiB/1MiB`, `reserve<window` exact-key errors (`[compaction].threshold…`); `FileConfig{flatten extra}` + `update_config_document (toml_edit::DocumentMut` targeted edit + re-validate) + `save_settings/save_reasoning` preserve unknowns; `lock_config (fs2::lock_exclusive` write+rename) + `0600` temp+rename+`sync` + `TEMP_ENTROPY/splitmix64`; `default_reserve_is_checked…` test.
- GAP: nested `extra` at every level (`[compaction]/[subagents]/[tui]/[mcp]/servers`) + two-mutator preservation tests not confirmed. Note: `completed.md` still lists CONFIG-1 outstanding — reconcile with above.

### CI-1 — PARTIAL
- DONE: locked Cargo checks, all-target Clippy (`cargo clippy --workspace --all-targets --locked -- -D warnings`), native arm64 smoke (per `completed.md`).
- OUTSTANDING: verify `.github/workflows/ci.yml/release.yml` `--locked` everywhere + Clippy exact flags; Linux/glibc baseline (pinned env or musl) + merge/cache speedup.

### DOCS-1 — MISSING
- `completed.md` admits outstanding. Need: model-assisted compaction + deterministic fallback (remove "absent" claim), read-only subagents `multigrep` scope, exact no-session compaction, shell exclusivity + tree-kill, MCP catalogue/output/time limits, Linux baseline, headless-stdout + ACP-purity. Files: `ARCHITECTURE.md`, `crates/session/README.md`, `docs/configuration.md`, `README.md`.

## Remaining TODO (ordered)
1. Phase 2 DONE (AUTH-1/2/3). Next: AGENT-3/AGENT-4 + PERF-1 missing tests; PERF-1 triple-scan fix.
2. TUI-2 metadata fit + width/combining/ZWJ property + height-clamp tests; AGENT-1/AGENT-2 deferred-sync/child-race tests.
3. PERF-5 benches (1k/10k) + external/replacement reconciliation; PERF-6 TUI quadratic guard (fix `lines().count()` walk, add bench).
4. CONFIG-1 nested-extra matrix + concurrent-mutator tests; CLEANUP-1 `cargo tree --duplicates` verify + remaining removals; CLEANUP-2/3/4 sweeps.
5. CI-1 workflow verify + Linux baseline; DOCS-1 contracts.
6. Re-run before handoff: `cargo fmt --all`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo test --workspace --locked`, `cargo build --workspace --locked`, `cargo tree --workspace --duplicates`, `git diff --check`.
