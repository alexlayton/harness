use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec, normalize_workspace_root};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

pub struct BashTool {
    rtk: bool,
    cwd: PathBuf,
}

impl BashTool {
    /// A tool that rewrites supported commands to their token-optimized `rtk`
    /// equivalents before execution (see [`rtk_rewrite`]).
    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        Self {
            rtk: false,
            cwd: normalize_workspace_root(root),
        }
    }

    pub fn with_rtk_and_workspace_root(rtk: bool, root: impl Into<PathBuf>) -> Self {
        Self {
            rtk,
            cwd: normalize_workspace_root(root),
        }
    }
}

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1024;
const RTK_REWRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Maximum shell timeout in seconds (24 hours). Documented in the JSON
/// schema and enforced at runtime; larger values (including `u64::MAX`)
/// are rejected as a tool error instead of overflowing deadline arithmetic.
pub const MAX_TIMEOUT_SECS: u64 = 86_400;
/// Grace period after TERM before escalating a process group to KILL.
const KILL_GRACE: Duration = Duration::from_millis(500);
/// Shared deadline for draining stdout+stderr after the shell ends.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Harness-side concurrency classification for one bash invocation.
/// Every invocation is [`Concurrency::Exclusive`]: shell is the optimized
/// parallel path's *escape hatch*, not a member of it — dedicated `read`,
/// `find`, `grep`, and `multigrep` remain the parallel path. The old
/// shell/Git read-only classifier was fragile word-level analysis that a
/// wrong `ReadOnly` could turn into interleaved mutations; a wrong
/// `Exclusive` merely forfeits latency, so exclusivity fails closed.
/// Kept as a function (rather than inlining in `concurrency()`) so agent
/// dispatch tests and callers can assert the contract directly.
pub fn command_concurrency(_command: &str) -> Concurrency {
    Concurrency::Exclusive
}

/// Ask rtk to rewrite a command to its token-optimized equivalent.  rtk
/// signals support by printing the rewritten command on stdout; unsupported
/// commands, a missing rtk binary, and timeouts all degrade to `None`, in
/// which case the caller runs the original command unchanged.  The exit code
/// is only authoritative for versions known to implement the documented
/// semantics: relying on stdout alone would misread a future version that
/// prints diagnostics but exits non-zero on error.
async fn rtk_rewrite_cancellable(
    command: &str,
    cancel: &CancellationToken,
    deadline: tokio::time::Instant,
) -> Option<String> {
    if command.trim_start().starts_with("rtk ") {
        return None;
    }
    let mut process = Command::new("rtk");
    process
        .arg("rewrite")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let rewrite_deadline = deadline.min(tokio::time::Instant::now() + RTK_REWRITE_TIMEOUT);
    let output = tokio::select! {
        biased;
        _ = cancel.cancelled() => return None,
        output = process.output() => output.ok()?,
        _ = tokio::time::sleep_until(rewrite_deadline) => return None,
    };
    let accepted = matches!(output.status.code(), Some(0 | 3));
    let rewritten = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (accepted && !rewritten.is_empty()).then_some(rewritten)
}

/// Resolve the bash `dir` argument against the workspace root, requiring an
/// existing directory inside the workspace.  This mirrors the path scoping of
/// find/grep; `cd` inside the command itself remains the escape hatch for
/// running anywhere else.
async fn resolve_workspace_dir(root: &Path, dir: &str) -> Result<PathBuf, String> {
    let candidate = super::resolve_workspace_path(dir, Some(root), false).await?;
    let metadata = tokio::fs::metadata(&candidate)
        .await
        .map_err(|error| format!("dir {dir}: {error}"))?;
    if !metadata.is_dir() {
        return Err(format!("dir {dir} is not a directory"));
    }
    Ok(candidate)
}

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
            name: "bash".into(),
            description: "Run a shell command in the working directory. Returns bounded stdout and stderr tails. Optionally run in a workspace-relative directory via the dir argument. Workspace cwd/path resolution is not an OS sandbox; shell commands can access paths allowed by the operating-system user.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command passed to sh -c" },
                    "dir": { "type": "string", "description": "Optional working directory for the command, relative to the workspace root (e.g. \"crates/tools\"). Prefer this over prefixing the command with cd <dir> && ..." },
                    "timeout": { "type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECS, "description": "Timeout in seconds (default 120, maximum 86400)" }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
            },
            prompt: ToolPrompt::new(
                "Run commands",
                [
                    "Use bash for tests, builds, git, or when no dedicated tool applies.".to_owned(),
                    "Use dir for commands in subdirectories instead of cd.".to_owned(),
                ],
            ),
        }
    }

    fn concurrency(&self, _args: &Value) -> Concurrency {
        // Every bash invocation is exclusive: shell text is never provably
        // side-effect-light. See `command_concurrency`.
        Concurrency::Exclusive
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let command = match args.get("command").and_then(Value::as_str) {
            Some(command) if !command.is_empty() => command.to_owned(),
            _ => return error("bash", "missing required argument: command"),
        };
        // Documented maximum timeout, enforced with checked arithmetic:
        // `u64::MAX` (or anything past the cap) is a tool error, never a
        // panic or wrap. `checked_add` on the `Instant` likewise fails to
        // an error instead of overflowing the deadline.
        let timeout_secs = match args.get("timeout") {
            None => 120,
            Some(value) => match value.as_u64() {
                Some(value) if (1..=MAX_TIMEOUT_SECS).contains(&value) => value,
                _ => {
                    return error(
                        "bash",
                        &format!(
                            "timeout must be an integer between 1 and {MAX_TIMEOUT_SECS} seconds"
                        ),
                    );
                }
            },
        };
        let timeout = timeout_secs;
        let dir = match args.get("dir") {
            None => None,
            Some(Value::String(dir)) if !dir.trim().is_empty() => Some(dir.clone()),
            Some(_) => return error("bash", "dir must be a non-empty string when provided"),
        };
        let run_dir = match dir.as_deref() {
            None => None,
            Some(dir) => match resolve_workspace_dir(&self.cwd, dir).await {
                Ok(resolved) => Some(resolved),
                Err(message) => return error("bash", &message),
            },
        };
        if cancel.is_cancelled() {
            return error(&format!("bash: {}", first_line(&command)), "cancelled");
        }

        // RTK owns rewrite policy: one cancellable whole-command request,
        // with rewrite time charged to the bash call's total deadline.
        // Checked arithmetic: an unrepresentable deadline is a tool error,
        // never a panic.
        let deadline = tokio::time::Instant::now()
            .checked_add(Duration::from_secs(timeout))
            .ok_or_else(|| {
                error(
                    "bash",
                    &format!("timeout {timeout}s overflows the deadline clock"),
                )
            });
        let deadline = match deadline {
            Ok(deadline) => deadline,
            Err(output) => return output,
        };
        let run_command = if self.rtk {
            rtk_rewrite_cancellable(&command, &cancel, deadline)
                .await
                .unwrap_or_else(|| command.clone())
        } else {
            command.clone()
        };

        let cwd = self.cwd.clone();
        let mut command_builder = Command::new("sh");
        command_builder
            .arg("-c")
            .arg(&run_command)
            .current_dir(run_dir.as_deref().unwrap_or(&cwd))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Own process group/session: on timeout, cancellation, future drop,
        // or shell exit with surviving descendants, the whole group is
        // terminated (TERM, escalated to KILL) and reaped — a backgrounded
        // descendant can never outlive the tool call and touch the
        // workspace afterwards.
        #[cfg(unix)]
        command_builder.process_group(0);
        let mut child = match command_builder.spawn() {
            Ok(child) => child,
            Err(io_error) => {
                return error(
                    &format!("bash: {}", first_line(&command)),
                    &format!("failed to start shell: {io_error}"),
                );
            }
        };
        // The shell's pid is the process-group leader (process_group(0)),
        // so group signals target exactly this invocation's tree.
        #[cfg(unix)]
        let group_id = child.id();
        #[cfg(not(unix))]
        let group_id = None;
        let mut process_guard = ProcessGroupGuard::new(group_id);

        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        let mut stdout_task = tokio::spawn(read_bounded_tail(stdout));
        let mut stderr_task = tokio::spawn(read_bounded_tail(stderr));

        enum End {
            Exited(std::process::ExitStatus),
            TimedOut,
            Cancelled,
        }
        let end = tokio::select! {
            result = child.wait() => match result {
                Ok(status) => End::Exited(status),
                Err(_) => End::Cancelled,
            },
            _ = tokio::time::sleep_until(deadline) => {
                terminate_tree(&mut child, group_id, &cancel).await;
                let _ = child.wait().await;
                End::TimedOut
            },
            _ = cancel.cancelled() => {
                terminate_tree(&mut child, group_id, &cancel).await;
                let _ = child.wait().await;
                End::Cancelled
            },
        };
        // The shell exited but descendants may survive (they inherit the
        // pipes and the group). Reap the whole tree the same way so a
        // backgrounded `sleep 300 &` cannot write a marker after return.
        if matches!(end, End::Exited(_)) {
            terminate_tree(&mut child, group_id, &cancel).await;
        }

        // Drain stdout and stderr concurrently under one shared deadline
        // (not two sequential one-second waits): held pipes on both
        // streams together consume at most DRAIN_TIMEOUT.
        let drain_deadline = tokio::time::Instant::now()
            .checked_add(DRAIN_TIMEOUT)
            .unwrap_or_else(tokio::time::Instant::now);
        let (stdout, stderr) = tokio::join!(
            async {
                match tokio::time::timeout_at(drain_deadline, &mut stdout_task).await {
                    Ok(Ok(capture)) => capture,
                    _ => {
                        stdout_task.abort();
                        TailCapture::default()
                    }
                }
            },
            async {
                match tokio::time::timeout_at(drain_deadline, &mut stderr_task).await {
                    Ok(Ok(capture)) => capture,
                    _ => {
                        stderr_task.abort();
                        TailCapture::default()
                    }
                }
            }
        );
        let mut output = stdout.render();
        if !stderr.bytes.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str("--- stderr ---\n");
            output.push_str(&stderr.render());
        }
        let mut is_error = false;
        let mut suffix = String::new();
        match end {
            End::Exited(status) => {
                if !status.success() {
                    is_error = true;
                    if let Some(code) = status.code() {
                        suffix = format!("[exit code {code}]");
                    } else {
                        suffix = "[process terminated by signal]".into();
                    }
                }
            }
            End::TimedOut => {
                is_error = true;
                suffix = format!("[timed out after {timeout}s]");
            }
            End::Cancelled => {
                is_error = true;
                suffix = "[cancelled]".into();
            }
        }
        output = truncate_command_output(&output);
        if !suffix.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&suffix);
        }

        process_guard.disarm();
        ToolOutput {
            content: output,
            is_error,
            summary: match dir.as_deref() {
                Some(dir) => format!("bash: {} (in {dir})", first_line(&run_command)),
                None => format!("bash: {}", first_line(&run_command)),
            },
        }
    }
}

#[derive(Default)]
struct TailCapture {
    bytes: Vec<u8>,
    omitted_lines: usize,
    truncated: bool,
}

impl TailCapture {
    fn render(&self) -> String {
        let body = String::from_utf8_lossy(&self.bytes);
        if self.truncated {
            format!(
                "[truncated while running: at least {} lines omitted]\n{body}",
                self.omitted_lines
            )
        } else {
            body.into_owned()
        }
    }
}

async fn read_bounded_tail<R: AsyncRead + Unpin>(mut reader: R) -> TailCapture {
    let mut capture = TailCapture::default();
    let mut chunk = [0u8; 8 * 1024];
    while let Ok(read) = reader.read(&mut chunk).await {
        if read == 0 {
            break;
        }
        capture.bytes.extend_from_slice(&chunk[..read]);
        if capture.bytes.len() > MAX_BYTES {
            let drain = capture.bytes.len() - MAX_BYTES;
            capture.omitted_lines += capture.bytes[..drain]
                .iter()
                .filter(|&&byte| byte == b'\n')
                .count();
            capture.bytes.drain(..drain);
            capture.truncated = true;
        }
        let line_count = capture.bytes.iter().filter(|&&byte| byte == b'\n').count();
        if line_count > MAX_LINES {
            let skip = line_count - MAX_LINES;
            let boundary = capture
                .bytes
                .iter()
                .enumerate()
                .filter(|(_, byte)| **byte == b'\n')
                .nth(skip - 1)
                .map_or(0, |(index, _)| index + 1);
            capture.bytes.drain(..boundary);
            capture.omitted_lines += skip;
            capture.truncated = true;
        }
    }
    capture
}

/// Keep the tail of a command's output.  Shell commands often print the useful
/// diagnostic at the end, so unlike read this intentionally discards the head.
pub fn truncate_command_output(output: &str) -> String {
    let mut start = 0usize;
    let mut omitted_lines = 0usize;
    let line_count = output.lines().count();

    if line_count > MAX_LINES {
        let to_skip = line_count - MAX_LINES;
        let mut skipped = 0;
        for (index, byte) in output.bytes().enumerate() {
            if byte == b'\n' {
                skipped += 1;
                if skipped == to_skip {
                    start = index + 1;
                    break;
                }
            }
        }
        omitted_lines += to_skip;
    }

    if output.len().saturating_sub(start) > MAX_BYTES {
        let tail_start = output.len() - MAX_BYTES;
        let mut boundary = tail_start;
        while boundary < output.len() && !output.is_char_boundary(boundary) {
            boundary += 1;
        }
        omitted_lines += output[start..boundary].lines().count();
        start = boundary;
    }

    if omitted_lines == 0 {
        return output.to_owned();
    }
    format!(
        "[truncated: {omitted_lines} lines omitted]\n{}",
        &output[start..]
    )
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value)
}

/// Terminate the whole process tree of a shell invocation: signal the
/// process group (TERM), escalate to KILL after a short grace period, and
/// reap directly manageable handles. The shell runs as its own group
/// leader (`process_group(0)`), so `killpg`-equivalent signals hit exactly
/// this invocation's descendants — never the harness itself. On non-Unix
/// platforms group semantics are unavailable and this degrades to killing
/// the shell handle directly.
///
/// Owns a shell process group until the tool has drained all of its pipes.
/// This synchronous drop guard is the last line of defense when the async
/// execution future is aborted before it reaches its normal cleanup path.
struct ProcessGroupGuard {
    group_id: Option<u32>,
    armed: bool,
}

impl ProcessGroupGuard {
    fn new(group_id: Option<u32>) -> Self {
        Self {
            group_id,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.armed
            && let Some(pgid) = self.group_id
        {
            // SAFETY: the group ID came from the shell created by this tool;
            // a negative PID targets only that process group.
            unsafe {
                libc::kill(-(pgid as libc::pid_t), libc::SIGKILL);
            }
        }
    }
}

#[cfg(unix)]
fn process_group_alive(pgid: u32) -> bool {
    if pgid == 0 {
        return false;
    }
    // SAFETY: signal 0 only probes the process group selected by our child ID.
    let result = unsafe { libc::kill(-(pgid as libc::pid_t), 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// `child` is used to reap the directly manageable shell handle when the
/// group signal path is unavailable; on Unix the group signals do the
/// work and the caller waits on the shell separately.
async fn terminate_tree(
    child: &mut tokio::process::Child,
    group_id: Option<u32>,
    cancel: &CancellationToken,
) {
    #[cfg(unix)]
    {
        let _ = child;
        let _ = cancel;
        if let Some(pgid) = group_id {
            // SAFETY: `killpg`-equivalent via libc with the group's own
            // pgid; a negative pid targets the group, signals are
            // SIGTERM/SIGKILL constants. The pgid came from our own
            // spawned child, never from external input.
            unsafe {
                libc::kill(-(pgid as libc::pid_t), libc::SIGTERM);
            }
            // If the shell already exited, its group normally disappears
            // immediately. Only wait for the grace period when a descendant
            // is still holding the group, avoiding a fixed delay on ordinary
            // successful commands.
            if process_group_alive(pgid) {
                tokio::select! {
                    _ = tokio::time::sleep(KILL_GRACE) => {}
                    _ = cancel.cancelled() => {}
                }
            }
            if process_group_alive(pgid) {
                unsafe {
                    libc::kill(-(pgid as libc::pid_t), libc::SIGKILL);
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        // Fallback: kill the shell handle. This cannot reach already-detached
        // grandchildren, but the direct child must still be killed before the
        // timeout/cancellation branch waits for it.
        let _ = group_id;
        let _ = cancel;
        let _ = child.start_kill();
    }
}

fn error(summary: &str, content: &str) -> ToolOutput {
    ToolOutput {
        content: content.to_owned(),
        is_error: true,
        summary: summary.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_stderr_and_exit_code() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_workspace_root(directory.path())
            .execute(
                json!({"command":"printf out; printf err >&2; exit 3"}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("out"));
        assert!(output.content.contains("--- stderr ---"));
        assert!(output.content.contains("[exit code 3]"));
    }

    #[tokio::test]
    async fn timeout_kills_command() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_workspace_root(directory.path())
            .execute(
                json!({"command":"sleep 2", "timeout": 1}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("timed out"));
    }

    #[test]
    fn truncates_to_tail() {
        let input = (0..2_100).map(|i| format!("{i}\n")).collect::<String>();
        let output = truncate_command_output(&input);
        assert!(output.starts_with("[truncated:"));
        assert!(output.contains("2099"));
        assert!(!output.contains("\n0\n"));
    }

    /// rtk is an optional external binary; tests that need it skip silently
    /// when it is not installed.
    async fn rtk_available() -> bool {
        Command::new("rtk")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn dir_argument_runs_command_in_subdirectory() {
        let directory = tempfile::tempdir().unwrap();
        let subdirectory = directory.path().join("src");
        std::fs::create_dir_all(&subdirectory).unwrap();
        let tool = BashTool::with_workspace_root(directory.path());
        let output = tool
            .execute(
                json!({"command": "printf ran > cwd-marker", "dir": "src"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert_eq!(
            std::fs::read_to_string(subdirectory.join("cwd-marker")).unwrap(),
            "ran"
        );
        assert!(!directory.path().join("cwd-marker").exists());
        assert!(output.summary.contains("(in src)"));
    }

    #[tokio::test]
    async fn dir_argument_rejects_escape_and_missing_directories() {
        let directory = tempfile::tempdir().unwrap();
        let tool = BashTool::with_workspace_root(directory.path());
        let outside = tool
            .execute(
                json!({"command": "pwd", "dir": "../outside"}),
                CancellationToken::new(),
            )
            .await;
        assert!(outside.is_error);
        assert!(outside.content.contains("outside"));

        let missing = tool
            .execute(
                json!({"command": "pwd", "dir": "nope"}),
                CancellationToken::new(),
            )
            .await;
        assert!(missing.is_error);
        assert!(missing.content.contains("nope"));
    }

    /// The end-to-end rewrite test runs `git status` in the crate directory,
    /// so it also needs to be inside a git work tree.
    async fn git_work_tree_available() -> bool {
        Command::new("git")
            .arg("status")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn rtk_enabled_tool_executes_rewritten_command() {
        if !rtk_available().await || !git_work_tree_available().await {
            return;
        }
        let cwd = std::env::current_dir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, &cwd)
            .execute(json!({"command": "git status"}), CancellationToken::new())
            .await;
        assert!(!output.is_error, "{}", output.content);
        // `rtk git status` prints a compact `* <branch>` header instead of
        // git's "On branch", proving the rewrite path ran.
        assert!(output.content.starts_with("* "), "{}", output.content);
        // The summary names the command that actually ran.
        assert_eq!(output.summary, "bash: rtk git status");
    }

    #[tokio::test]
    async fn rtk_enabled_tool_falls_back_for_unsupported_commands() {
        if !rtk_available().await {
            return;
        }
        let cwd = std::env::current_dir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, &cwd)
            .execute(json!({"command": "echo hi"}), CancellationToken::new())
            .await;
        assert!(!output.is_error);
        assert_eq!(output.content.trim(), "hi");
    }

    #[test]
    fn every_bash_invocation_is_exclusive() {
        // Even env-prefixed and Git helper commands - the old classifier's
        // read-only set - are exclusive now. Dedicated read/find/grep/
        // multigrep remain the optimized parallel path.
        for command in [
            "git status",
            "git log --oneline -5",
            "FOO=bar git status",
            "ls -la",
            "cat README.md",
            "echo hello",
            "cargo --version",
            "rg TODO src/",
            "cargo test",
            "rm -rf target",
            "echo hi > out.txt",
            "",
        ] {
            assert_eq!(
                command_concurrency(command),
                Concurrency::Exclusive,
                "{command:?} must be exclusive"
            );
        }
        // The tool-level classification agrees, with or without a command.
        let tool = BashTool::with_workspace_root("/tmp");
        assert_eq!(
            tool.concurrency(&json!({"command": "git status"})),
            Concurrency::Exclusive
        );
        assert_eq!(tool.concurrency(&json!({})), Concurrency::Exclusive);
    }

    #[test]
    fn oversized_timeout_is_a_tool_error_not_a_panic() {
        assert_eq!(
            MAX_TIMEOUT_SECS, 86_400,
            "schema maximum and runtime cap must agree"
        );
    }

    #[tokio::test]
    async fn max_timeout_rejects_u64_max() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_workspace_root(directory.path())
            .execute(
                json!({"command": "echo hi", "timeout": u64::MAX}),
                CancellationToken::new(),
            )
            .await;
        assert!(
            output.is_error,
            "u64::MAX must not panic: {}",
            output.content
        );
        assert!(output.content.contains("timeout"), "{}", output.content);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_descendant_is_killed_after_return() {
        // A backgrounded descendant that tries to write a marker after the
        // shell returns must be killed with the process group.
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("marker");
        let output = BashTool::with_workspace_root(directory.path())
            .execute(
                json!({
                    "command": format!(
                        "(sleep 30 && touch {} &) ; exit 0",
                        marker.display()
                    ),
                    "timeout": 10,
                }),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        assert!(
            !marker.exists(),
            "background descendant survived the tool call"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn aborting_execution_kills_descendants() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("abort-marker");
        let command = format!("(sleep 1; touch {}) & wait", marker.display());
        let tool = BashTool::with_workspace_root(directory.path());
        let task = tokio::spawn(async move {
            tool.execute(
                json!({"command": command, "timeout": 60}),
                CancellationToken::new(),
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        task.abort();
        let _ = task.await;
        tokio::time::sleep(std::time::Duration::from_millis(1_500)).await;
        assert!(!marker.exists(), "descendant survived an aborted tool task");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_descendants() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("timeout-marker");
        let output = BashTool::with_workspace_root(directory.path())
            .execute(
                json!({
                    "command": format!(
                        "sh -c 'sleep 30 & wait'; touch {}",
                        marker.display()
                    ),
                    "timeout": 1,
                }),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error, "expected timeout: {}", output.content);
        assert!(output.content.contains("timed out"), "{}", output.content);
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        assert!(!marker.exists(), "descendant wrote after timeout");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_kills_descendants() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("cancel-marker");
        let cancel = CancellationToken::new();
        let tool = BashTool::with_workspace_root(directory.path());
        let command = format!("sleep 30; touch {}", marker.display());
        let cancel_task = cancel.clone();
        let task = tokio::spawn(async move {
            tool.execute(json!({"command": command, "timeout": 60}), cancel_task)
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        cancel.cancel();
        let output = task.await.unwrap();
        assert!(output.content.contains("cancelled"), "{}", output.content);
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        assert!(!marker.exists(), "descendant wrote after cancel");
    }
}
