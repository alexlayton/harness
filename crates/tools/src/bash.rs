use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec, normalize_workspace_root};
use async_trait::async_trait;
use llm::ToolDefinition;
use rtk::{
    BeforeStartError, CancellationToken as RtkCancellationToken, ExecuteError, ExecuteOptions,
    MayHaveStartedKind,
};
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
    /// A shell tool rooted at `root`, with in-process RTK execution disabled.
    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        Self {
            rtk: false,
            cwd: normalize_workspace_root(root),
        }
    }

    /// A shell tool rooted at `root`, optionally using RTK's execution API.
    pub fn with_rtk_and_workspace_root(rtk: bool, root: impl Into<PathBuf>) -> Self {
        Self {
            rtk,
            cwd: normalize_workspace_root(root),
        }
    }
}

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1024;

/// Cancels detached RTK work if the agent drops the Bash future while
/// interrupting a tool batch. `spawn_blocking` itself cannot be force-stopped.
struct CancelRtkOnDrop(RtkCancellationToken);

impl Drop for CancelRtkOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// True when a shell operand contains a control-flow keyword as a word, which
/// makes word-level read-only classification unsafe. Conservative: a false
/// positive only serializes execution.
fn has_control_keyword(operand: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "if ",
        "then ",
        "else ",
        "elif ",
        "for ",
        "while ",
        "until ",
        "case ",
        "do ",
        "done ",
        "function ",
        "select ",
    ];
    let bytes = operand.as_bytes();
    KEYWORDS.iter().any(|keyword| {
        bytes
            .windows(keyword.len())
            .any(|window| window == keyword.as_bytes())
    })
}

/// Harness-side concurrency classification for one bash invocation.  This is
/// decided here — never by the model — so scheduling correctness cannot
/// depend on prompt compliance.  It fails closed: anything not provably
/// side-effect-light classifies [`Concurrency::Exclusive`], which merely
/// forfeits latency, while a wrong `ReadOnly` could interleave mutations.
pub fn command_concurrency(command: &str) -> Concurrency {
    match split_readonly_segments(command) {
        Some(segments) if segments.iter().all(|segment| segment_is_read_only(segment)) => {
            Concurrency::ReadOnly
        }
        _ => Concurrency::Exclusive,
    }
}

/// Split a command on top-level separators (`&&`, `;`, and newlines, which
/// are command separators in shell) into operands, refusing any structure a
/// word-level analysis cannot judge: pipes, subshells, command substitution,
/// heredocs, brace groups, redirections, or backgrounding.  Quoting and
/// escapes are respected.  Returns `None` when the command must stay serial;
/// a plain single command yields one segment.
fn split_readonly_segments(command: &str) -> Option<Vec<&str>> {
    let bytes = command.as_bytes();
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        match byte {
            b'\\' => escaped = true,
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'`' => return None,
            b'$' if bytes.get(index + 1) == Some(&b'(') => return None,
            b'<' if matches!(bytes.get(index + 1), Some(b'<') | Some(b'(')) => return None,
            b'<' | b'>' if !in_single && !in_double => return None,
            b'|' | b'(' if !in_single && !in_double => return None,
            b'{' if !in_single && !in_double && bytes.get(index.wrapping_sub(1)) != Some(&b'$') => {
                return None;
            }
            b'&' if !in_single && !in_double => {
                // `&&` separates; a lone `&` backgrounds the command, whose
                // side effects would outlive the tool call — stay serial.
                if bytes.get(index + 1) != Some(&b'&') {
                    return None;
                }
                parts.push(command[start..index].trim());
                start = index + 2;
                index += 2;
                continue;
            }
            b';' | b'\n' | b'\r' if !in_single && !in_double => {
                parts.push(command[start..index].trim());
                start = index + 1;
                index += 1;
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    let tail = command[start..].trim();
    parts.push(tail);
    if parts.iter().any(|part| part.is_empty()) {
        return None;
    }
    if parts.iter().any(|part| has_control_keyword(part)) {
        return None;
    }
    Some(parts)
}

/// Strip one layer of matching quotes so quoted words (`git "status"`)
/// compare equal to their bare form during table lookup.
fn strip_quotes(word: &str) -> &str {
    let mut word = word.trim();
    for quote in ['\'', '"'] {
        if word.len() >= 2 && word.starts_with(quote) && word.ends_with(quote) {
            word = &word[1..word.len() - 1];
        }
    }
    word
}

/// True for `NAME=value` environment assignments preceding a command.
fn is_env_assignment(word: &str) -> bool {
    let Some(equals) = word.find('=') else {
        return false;
    };
    let name = &word[..equals];
    let mut bytes = name.bytes();
    match bytes.next() {
        // A valid identifier starts with a letter or underscore.
        Some(first) if first.is_ascii_alphabetic() || first == b'_' => {}
        _ => return false,
    }
    bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// Git global options that precede the subcommand, mapped to how many
/// following words they consume as values.  Anything unrecognized fails
/// closed.
const GIT_GLOBAL_FLAGS_WITH_VALUES: &[&str] = &["-C", "-c"];
const GIT_GLOBAL_PREFIX_FLAGS: &[&str] = &["--git-dir=", "--work-tree="];
const GIT_GLOBAL_BARE_FLAGS: &[&str] = &["--no-pager", "--literal-pathspecs"];

/// Git subcommands that never mutate repository or worktree state, whatever
/// their arguments (pathspecs and revisions are reads).
const GIT_READ_ONLY_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "blame",
    "ls-files",
    "ls-remote",
    "cat-file",
    "rev-parse",
    "describe",
    "shortlog",
    "whatchanged",
    "merge-base",
    "reflog",
    "show-branch",
    "count-objects",
    "cherry",
    "version",
];

/// Listing-mode flags under which `git branch` / `git tag` are read-only.
/// With any positional argument they create/delete/rename refs, so those
/// fail closed.
const GIT_LIST_MODE_FLAGS: &[&str] = &[
    "-l",
    "--list",
    "-a",
    "-r",
    "-v",
    "-vv",
    "--show-current",
    "--show-ref-names",
    "--merged",
    "--no-merged",
    "--contains",
    "--no-contains",
    "--points-at",
    "--sort",
    "--format",
    "--color",
    "--abbrev",
    "-n",
];

/// Read-only modes of `git config`; any other form may write configuration.
const GIT_CONFIG_READ_FLAGS: &[&str] = &["--get", "--get-all", "--get-regexp", "--list"];

/// Shell commands judged side-effect-light with arbitrary arguments.  This
/// is deliberately an allowlist: unknown commands stay serial.  Deliberately
/// excluded despite common read-only use: `sed` (`-i`, `w`, `r`), `awk`
/// (`system()`, redirections), `xargs` (arbitrary execution), `env` (runs a
/// command), and everything that can spawn processes.
const READ_ONLY_COMMANDS: &[&str] = &[
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "stat",
    "pwd",
    "which",
    "file",
    "du",
    "df",
    "tree",
    "uname",
    "printenv",
    "id",
    "whoami",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "echo",
    "printf",
    "true",
    "false",
    "nl",
    "rev",
    "tac",
    "strings",
    "column",
    "cksum",
    "md5sum",
    "sha1sum",
    "sha256sum",
    "diff",
    "cmp",
    "comm",
    "sort",
    "uniq",
    "cut",
    "rg",
    "grep",
    "fd",
];

/// Version-print subcommands of build-tool binaries; anything else these
/// tools do (builds, installs, package management) stays serial.
const VERSION_ONLY_COMMANDS: &[&str] = &[
    "cargo", "rustc", "rustup", "node", "npm", "npx", "python", "python3", "go",
];

/// Arguments that disqualify an otherwise read-only command: flags that
/// write files (`sort -o`, `git diff --output=`), set state (`date -s`), or
/// execute other programs (`rg --pre`, `fd -x`).  A flag matches exactly or
/// as a `--flag=value` prefix.
const READ_ONLY_COMMAND_EXCLUSIONS: &[(&str, &[&str])] = &[
    ("rg", &["--pre", "--pre-glob"]),
    ("fd", &["-x", "-X", "--exec", "--exec-batch"]),
    ("sort", &["-o", "--output"]),
    ("date", &["-s", "--set"]),
    ("hostname", &["-F", "--file"]),
];

/// Judge one separator-free command operand.  Leading `VAR=value` assignments
/// and a `cd <dir>` prefix are transparent; the remaining command word is
/// looked up in the read-only tables.
fn segment_is_read_only(segment: &str) -> bool {
    let mut words: Vec<&str> = segment
        .split_whitespace()
        .map(strip_quotes)
        .filter(|word| !word.is_empty())
        .collect();
    while words.first().is_some_and(|word| is_env_assignment(word)) {
        words.remove(0);
    }
    let Some(first) = words.first() else {
        return false;
    };
    if *first == "cd" {
        // `cd <dir>` is transparent; bare `cd`, flags, or extra operands mean
        // we did not parse what will really run.
        return words.len() == 2 && !words[1].starts_with('-');
    }
    if *first == "git" {
        return git_invocation_is_read_only(&words[1..]);
    }
    if *first == "find" {
        // find(1) is read-only except for its mutating actions.
        return !words[1..].iter().any(|word| {
            *word == "-delete"
                || *word == "-fls"
                || word.starts_with("-exec")
                || word.starts_with("-ok")
                || word.starts_with("-fprint")
        });
    }
    if READ_ONLY_COMMANDS.contains(first) {
        // The allowlist entry covers arbitrary arguments except for the
        // specific flags tabulated as disqualifiers.
        return !words[1..].iter().any(|word| {
            READ_ONLY_COMMAND_EXCLUSIONS
                .iter()
                .filter(|(command, _)| command == first)
                .flat_map(|(_, flags)| flags.iter())
                .any(|flag| *word == *flag || word.starts_with(&format!("{flag}=")))
        });
    }
    if VERSION_ONLY_COMMANDS.contains(first) {
        return words.len() == 2 && matches!(words[1], "--version" | "-V" | "version");
    }
    false
}

/// Judge a `git` invocation after the leading `git` word: skip known global
/// options, then require the subcommand to be provably read-only.
fn git_invocation_is_read_only(rest: &[&str]) -> bool {
    let mut index = 0;
    while index < rest.len() {
        let word = rest[index];
        if GIT_GLOBAL_FLAGS_WITH_VALUES.contains(&word) {
            index += 2;
        } else if GIT_GLOBAL_PREFIX_FLAGS
            .iter()
            .any(|flag| word.starts_with(flag))
            || GIT_GLOBAL_BARE_FLAGS.contains(&word)
        {
            index += 1;
        } else if word.starts_with('-') {
            return false;
        } else {
            break;
        }
    }
    let Some(subcommand) = rest.get(index) else {
        // Bare `git` prints help: harmless, but pointless to batch.
        return false;
    };
    let args = &rest[index + 1..];
    // `git diff --output=<file>` (and `--output-indicator-*` are fine, but
    // plain `--output` writes a file) fails closed.
    if GIT_READ_ONLY_SUBCOMMANDS.contains(subcommand)
        && matches!(*subcommand, "diff" | "show" | "whatchanged")
        && args
            .iter()
            .any(|arg| arg.starts_with("--output=") || *arg == "--output")
    {
        return false;
    }
    match *subcommand {
        _ if GIT_READ_ONLY_SUBCOMMANDS.contains(subcommand) => true,
        "branch" | "tag" => {
            !args.is_empty()
                && args
                    .iter()
                    .all(|arg| arg.starts_with('-') && GIT_LIST_MODE_FLAGS.contains(arg))
        }
        "config" => args
            .first()
            .is_some_and(|arg| GIT_CONFIG_READ_FLAGS.contains(arg)),
        "remote" => {
            args.iter().all(|arg| matches!(*arg, "-v" | "--verbose"))
                || args.first() == Some(&"get-url") && args.len() == 2
        }
        "worktree" | "stash" => args.first() == Some(&"list") || args.first() == Some(&"show"),
        _ => false,
    }
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

/// Run one command through RTK's synchronous embedding API. Cancellation is
/// bridged explicitly, then the blocking task is awaited so RTK can terminate
/// and reap the complete child process tree before the tool call returns.
async fn execute_with_rtk(
    command: String,
    timeout: u64,
    dir: Option<String>,
    cwd: PathBuf,
    cancel: CancellationToken,
) -> ToolOutput {
    let summary = bash_summary(&command, dir.as_deref());
    let rtk_cancel = RtkCancellationToken::new();
    let _cancel_on_drop = CancelRtkOnDrop(rtk_cancel.clone());
    let options = ExecuteOptions::default()
        .with_cwd(cwd)
        .with_timeout(Duration::from_secs(timeout))
        .with_cancellation(rtk_cancel.clone())
        .with_output_limit(MAX_BYTES);
    let task_command = command.clone();
    let mut task =
        tokio::task::spawn_blocking(move || rtk::execute_with_options(&task_command, &options));

    let joined = tokio::select! {
        biased;
        result = &mut task => result,
        _ = cancel.cancelled() => {
            rtk_cancel.cancel();
            task.await
        }
    };

    match joined {
        Ok(Ok(result)) => {
            tracing::debug!(command = %command, route = ?result.route, "RTK execution completed");
            let suffix =
                (result.exit_code != 0).then(|| format!("[exit code {}]", result.exit_code));
            ToolOutput {
                content: render_rtk_output(
                    &result.stdout,
                    result.stdout_truncated,
                    &result.stderr,
                    result.stderr_truncated,
                    suffix.as_deref(),
                ),
                is_error: result.exit_code != 0,
                summary,
            }
        }
        Ok(Err(error)) => rtk_error_output(error, timeout, summary),
        Err(join_error) => error(
            &summary,
            &format!("RTK execution worker failed: {join_error}"),
        ),
    }
}

fn rtk_error_output(error: ExecuteError, timeout: u64, summary: String) -> ToolOutput {
    let (stdout, stdout_truncated, stderr, stderr_truncated, suffix) = match error {
        ExecuteError::BeforeStart(error) => {
            let suffix = match error {
                BeforeStartError::Cancelled => "[cancelled]".to_owned(),
                BeforeStartError::TimedOut => format!("[timed out after {timeout}s]"),
                BeforeStartError::WorkingDirectory(error) => {
                    format!("[RTK execution failed: working directory: {error}]")
                }
            };
            (String::new(), false, String::new(), false, suffix)
        }
        ExecuteError::MayHaveStarted(error) => {
            let suffix = match error.kind {
                MayHaveStartedKind::Cancelled => "[cancelled]".to_owned(),
                MayHaveStartedKind::TimedOut => format!("[timed out after {timeout}s]"),
                MayHaveStartedKind::Spawn(error) => {
                    format!("[RTK execution failed while spawning: {error}]")
                }
                MayHaveStartedKind::Wait(error) => {
                    format!("[RTK execution failed while waiting: {error}]")
                }
                MayHaveStartedKind::Terminate(error) => {
                    format!("[RTK process termination failed: {error}]")
                }
                MayHaveStartedKind::Capture(error) => {
                    format!("[RTK output capture failed: {error}]")
                }
                MayHaveStartedKind::Filter(error) => {
                    format!("[RTK output filter failed: {error}]")
                }
            };
            (
                error.partial_output.stdout,
                error.partial_output.stdout_truncated,
                error.partial_output.stderr,
                error.partial_output.stderr_truncated,
                suffix,
            )
        }
    };

    ToolOutput {
        content: render_rtk_output(
            &stdout,
            stdout_truncated,
            &stderr,
            stderr_truncated,
            Some(&suffix),
        ),
        is_error: true,
        summary,
    }
}

fn render_rtk_output(
    stdout: &str,
    stdout_truncated: bool,
    stderr: &str,
    stderr_truncated: bool,
    suffix: Option<&str>,
) -> String {
    let mut output = stdout.to_owned();
    if !stderr.is_empty() || stderr_truncated {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("--- stderr ---\n");
        output.push_str(stderr);
    }
    output = truncate_command_output(&output);

    let mut markers = Vec::new();
    if stdout_truncated {
        markers.push(format!(
            "[truncated while running: stdout exceeded the {MAX_BYTES}-byte capture limit]"
        ));
    }
    if stderr_truncated {
        markers.push(format!(
            "[truncated while running: stderr exceeded the {MAX_BYTES}-byte capture limit]"
        ));
    }
    if !markers.is_empty() {
        if !output.is_empty() {
            markers.push(output);
        }
        output = markers.join("\n");
    }

    if let Some(suffix) = suffix {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(suffix);
    }
    output
}

fn bash_summary(command: &str, dir: Option<&str>) -> String {
    match dir {
        Some(dir) => format!("bash: {} (in {dir})", first_line(command)),
        None => format!("bash: {}", first_line(command)),
    }
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
                    "command": { "type": "string", "description": "Shell command to execute" },
                    "dir": { "type": "string", "description": "Optional working directory for the command, relative to the workspace root (e.g. \"crates/tools\"). Prefer this over prefixing the command with cd <dir> && ..." },
                    "timeout": { "type": "integer", "minimum": 1, "description": "Timeout in seconds (default 120)" }
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

    fn concurrency(&self, args: &Value) -> Concurrency {
        match args.get("command").and_then(Value::as_str) {
            Some(command) => command_concurrency(command),
            None => Concurrency::Exclusive,
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let command = match args.get("command").and_then(Value::as_str) {
            Some(command) if !command.is_empty() => command.to_owned(),
            _ => return error("bash", "missing required argument: command"),
        };
        let timeout = match args.get("timeout") {
            None => 120,
            Some(value) => match value.as_u64() {
                Some(value) if value > 0 => value,
                _ => return error("bash", "timeout must be a positive integer"),
            },
        };
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

        let run_cwd = run_dir.unwrap_or_else(|| self.cwd.clone());
        if self.rtk {
            return execute_with_rtk(command, timeout, dir, run_cwd, cancel).await;
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout);
        let mut child = match Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(&run_cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(child) => child,
            Err(io_error) => {
                return error(
                    &format!("bash: {}", first_line(&command)),
                    &format!("failed to start shell: {io_error}"),
                );
            }
        };

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
                let _ = child.kill().await;
                let _ = child.wait().await;
                End::TimedOut
            },
            _ = cancel.cancelled() => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                End::Cancelled
            },
        };

        // A descendant can inherit a pipe after the shell exits. Bound drain
        // time so such a process cannot keep the tool alive indefinitely.
        let stdout = match tokio::time::timeout(Duration::from_secs(1), &mut stdout_task).await {
            Ok(Ok(capture)) => capture,
            _ => {
                stdout_task.abort();
                TailCapture::default()
            }
        };
        let stderr = match tokio::time::timeout(Duration::from_secs(1), &mut stderr_task).await {
            Ok(Ok(capture)) => capture,
            _ => {
                stderr_task.abort();
                TailCapture::default()
            }
        };
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

        ToolOutput {
            content: output,
            is_error,
            summary: bash_summary(&command, dir.as_deref()),
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

    #[test]
    fn rtk_output_preserves_truncation_markers_and_error_suffix() {
        let oversized = "x".repeat(MAX_BYTES);
        let output = render_rtk_output(&oversized, true, "stderr tail", true, Some("[cancelled]"));
        assert!(output.contains("stdout exceeded"));
        assert!(output.contains("stderr exceeded"));
        assert!(output.contains("stderr tail"));
        assert!(output.ends_with("[cancelled]"));
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

    fn run_git(cwd: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
    }

    #[tokio::test]
    async fn rtk_enabled_tool_filters_git_status_in_process() {
        let directory = tempfile::tempdir().unwrap();
        run_git(directory.path(), &["init", "-q"]);
        run_git(
            directory.path(),
            &["config", "user.email", "harness@example.com"],
        );
        run_git(directory.path(), &["config", "user.name", "Harness Test"]);
        std::fs::write(directory.path().join("file.txt"), "first\n").unwrap();
        run_git(directory.path(), &["add", "file.txt"]);
        run_git(directory.path(), &["commit", "-qm", "initial"]);
        std::fs::write(directory.path().join("file.txt"), "first\nsecond\n").unwrap();

        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(json!({"command": "git status"}), CancellationToken::new())
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.starts_with("* "), "{}", output.content);
        assert_eq!(output.summary, "bash: git status");
    }

    #[tokio::test]
    async fn rtk_enabled_tool_filters_wc_without_an_rtk_binary() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("lines.txt"), "one\ntwo\n").unwrap();

        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(
                json!({"command": "wc -l lines.txt"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert_eq!(output.content.trim(), "2");
        assert_eq!(output.summary, "bash: wc -l lines.txt");
    }

    #[tokio::test]
    async fn rtk_enabled_tool_uses_shell_passthrough_for_unsupported_commands() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(json!({"command": "echo hi"}), CancellationToken::new())
            .await;
        assert!(!output.is_error);
        assert_eq!(output.content.trim(), "hi");
    }

    #[tokio::test]
    async fn rtk_shell_passthrough_executes_a_mutation_once() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(
                json!({"command": "printf x >> marker.txt"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert_eq!(
            std::fs::read_to_string(directory.path().join("marker.txt")).unwrap(),
            "x"
        );
    }

    #[tokio::test]
    async fn rtk_enabled_dir_is_used_as_the_execution_cwd() {
        let directory = tempfile::tempdir().unwrap();
        let subdirectory = directory.path().join("src");
        std::fs::create_dir(&subdirectory).unwrap();
        std::fs::write(subdirectory.join("lines.txt"), "one\n").unwrap();

        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(
                json!({"command": "wc -l lines.txt", "dir": "src"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert_eq!(output.content.trim(), "1");
        assert_eq!(output.summary, "bash: wc -l lines.txt (in src)");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rtk_bounded_stdout_retains_the_stream_tail() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(
                json!({"command": "printf START; yes x | head -c 60000; printf END"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("stdout exceeded"));
        assert!(output.content.contains("END"));
        assert!(!output.content.contains("START"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rtk_bounded_stderr_retains_the_stream_tail() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(
                json!({
                    "command": "{ printf START; yes x | head -c 60000; printf END; } >&2"
                }),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("stderr exceeded"));
        assert!(output.content.contains("END"));
        assert!(!output.content.contains("START"));
    }

    #[tokio::test]
    async fn rtk_enabled_timeout_preserves_partial_output() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_rtk_and_workspace_root(true, directory.path())
            .execute(
                json!({"command": "printf started; sleep 5", "timeout": 1}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("started"), "{}", output.content);
        assert!(
            output.content.contains("[timed out after 1s]"),
            "{}",
            output.content
        );
    }

    #[tokio::test]
    async fn harness_cancellation_is_bridged_to_rtk() {
        let directory = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let root = directory.path().to_path_buf();
        let task = tokio::spawn(async move {
            BashTool::with_rtk_and_workspace_root(true, root)
                .execute(json!({"command": "printf started; sleep 5"}), task_cancel)
                .await
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        let output = task.await.unwrap();
        assert!(output.is_error);
        assert!(output.content.contains("started"), "{}", output.content);
        assert!(output.content.contains("[cancelled]"), "{}", output.content);
    }

    #[tokio::test]
    async fn dropping_rtk_tool_future_cancels_detached_blocking_work() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_path_buf();
        let task = tokio::spawn(async move {
            BashTool::with_rtk_and_workspace_root(true, root)
                .execute(
                    json!({
                        "command": "printf started > started.marker; sleep 1; printf finished > finished.marker"
                    }),
                    CancellationToken::new(),
                )
                .await
        });

        let started = directory.path().join("started.marker");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while !started.exists() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(started.exists(), "RTK command did not start");

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        // Wait beyond the command's natural completion time: without the
        // drop guard the detached worker would create the final marker.
        tokio::time::sleep(Duration::from_millis(1_200)).await;
        assert!(
            !directory.path().join("finished.marker").exists(),
            "detached RTK command survived its dropped tool future"
        );
    }

    #[test]
    fn read_only_commands_classify_concurrent() {
        let concurrent = [
            "git status",
            "git log --oneline -5",
            "git diff HEAD~1",
            "git -C crates/tools status",
            "git --no-pager diff",
            "git branch --list",
            "git tag -l",
            "git config --get user.name",
            "git remote -v",
            "git stash list",
            "cd crates/tui && git status",
            "FOO=bar git status",
            "ls -la",
            "cat README.md",
            "head -20 src/main.rs",
            "wc -l crates/*/*.rs",
            "pwd",
            "which cargo",
            "rg TODO src/",
            "grep -rn pattern .",
            "find . -name '*.rs' -maxdepth 2",
            "cargo --version",
            "node --version",
            "echo hello",
            "printf '%s\\n' line",
            "stat Cargo.toml",
            "du -sh target",
            "uname -a",
            "true",
        ];
        for command in concurrent {
            assert_eq!(
                command_concurrency(command),
                Concurrency::ReadOnly,
                "{command:?} should be read-only"
            );
        }
    }

    #[test]
    fn mutating_or_unanalyzable_commands_stay_serial() {
        let exclusive = [
            "date +%Y",
            "hostname",
            "cargo test",
            "cargo build --release",
            "rm -rf target",
            "touch file.txt",
            "mkdir -p a/b",
            "git commit -m x",
            "git checkout main",
            "git add .",
            "git push",
            "git branch new-branch",
            "git tag v1.0.0",
            "git config user.email a@b.c",
            "git worktree add ../wt",
            "find . -name '*.tmp' -delete",
            "find . -name '*.log' -exec rm {} \\",
            "echo hi > out.txt",
            "cat in.txt | sort",
            "sort < input.txt",
            "echo $(date)",
            "echo `date`",
            "sleep 5 & wait",
            "cd .. && rm -rf build",
            "npm install",
            "python script.py",
            "sed -i 's/a/b/' file.txt",
            "xargs ls < files.txt",
            "if true; then echo hi; fi",
            "for f in *; do cat $f; done",
            "ls; rm file",
            "git status && cargo test",
            "", // empty command is rejected at execute time anyway
        ];
        for command in exclusive {
            assert_eq!(
                command_concurrency(command),
                Concurrency::Exclusive,
                "{command:?} must stay serial"
            );
        }
    }

    #[test]
    fn side_effect_flags_on_read_only_commands_fail_closed() {
        let exclusive = [
            "sort -o /etc/passwd input.txt",
            "sort --output=x.txt input.txt",
            "date -s 2000-01-01",
            "date --set=2000-01-01",
            "rg --pre ./hook.sh pattern",
            "fd -x chmod 644 \\{",
            "fd --exec echo",
            "hostname -F hosts.txt",
            "git diff --output=patch.txt",
        ];
        for command in exclusive {
            assert_eq!(
                command_concurrency(command),
                Concurrency::Exclusive,
                "{command:?} must stay serial"
            );
        }
        // Unrelated flags on the same commands stay read-only.
        assert_eq!(
            command_concurrency("sort -u input.txt"),
            Concurrency::ReadOnly
        );
        assert_eq!(command_concurrency("rg -n TODO"), Concurrency::ReadOnly);
    }

    #[test]
    fn quoting_and_escapes_are_respected_by_the_classifier() {
        // Separators inside quotes do not split.
        assert_eq!(command_concurrency("echo 'a && b'"), Concurrency::ReadOnly);
        assert_eq!(command_concurrency("echo \"x; y\""), Concurrency::ReadOnly);
        // A quoted word still matches the command table.
        assert_eq!(command_concurrency("git \"status\""), Concurrency::ReadOnly);
        // An escaped separator does not split either.
        assert_eq!(
            command_concurrency("echo a \\&& echo b"),
            Concurrency::Exclusive
        );
    }
}
