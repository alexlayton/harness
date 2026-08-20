use super::{Tool, ToolOutput, ToolPrompt, ToolSpec, normalize_workspace_root};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::io::AsyncReadExt;
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

/// rtk documents exit 0 for a successful rewrite (exit 1 for unsupported
/// commands) but 0.45.0 exited 3 on success.  Only versions at or above this
/// release are trusted to follow the documented exit-code semantics; older
/// and unrecognized versions fall back to the non-empty-stdout heuristic.
const RTK_TRUSTED_EXIT_CODE_VERSION: (u64, u64, u64) = (0, 46, 0);

static RTK_VERSION: OnceLock<Option<(u64, u64, u64)>> = OnceLock::new();

/// Ask rtk to rewrite a command to its token-optimized equivalent.  rtk
/// signals support by printing the rewritten command on stdout; unsupported
/// commands, a missing rtk binary, and timeouts all degrade to `None`, in
/// which case the caller runs the original command unchanged.  The exit code
/// is only authoritative for versions known to implement the documented
/// semantics: relying on stdout alone would misread a future version that
/// prints diagnostics but exits non-zero on error.
async fn rtk_rewrite(command: &str) -> Option<String> {
    let output = tokio::time::timeout(
        RTK_REWRITE_TIMEOUT,
        Command::new("rtk")
            .arg("rewrite")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    let rewritten = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if rtk_exit_codes_trustworthy().await && !output.status.success() {
        return None;
    }
    if rewritten.is_empty() {
        None
    } else {
        Some(rewritten)
    }
}

/// rtk's compound-command support is selective: some `cd X && cmd` shapes
/// pass through untouched even though the trailing command alone has an rtk
/// equivalent.  When the whole command was not rewritten, split top-level
/// `&&`/`;` groups and rewrite each operand individually, keeping unchanged
/// operands verbatim.  Returns `None` when nothing was rewritten or the
/// command is not safely splittable.
async fn rtk_rewrite_compound(command: &str) -> Option<String> {
    let parts = split_compound(command)?;
    let mut rewritten_any = false;
    let mut output = String::with_capacity(command.len());
    for (index, part) in parts.iter().enumerate() {
        if index % 2 == 0 {
            match rtk_rewrite(part).await {
                Some(rewritten) => {
                    output.push_str(&rewritten);
                    rewritten_any = true;
                }
                None => output.push_str(part),
            }
        } else {
            // Normalize the separator so spacing stays readable after operands
            // were rewritten to different lengths.
            output.push_str(match *part {
                "&&" => " && ",
                ";" => " ; ",
                other => other,
            });
        }
    }
    rewritten_any.then_some(output)
}

/// Split a shell command on top-level `&&` / `;` separators into alternating
/// operands and separators.  Splitting respects single/double quotes and
/// backslash escapes, trims operands, and refuses commands whose structure a
/// naive split would change: pipes, subshells, heredocs, command substitution,
/// brace groups, or control-flow keywords.
fn split_compound(command: &str) -> Option<Vec<&str>> {
    let bytes = command.as_bytes();
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut saw_separator = false;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        match byte {
            b'\\' => {
                escaped = true;
                index += 1;
                continue;
            }
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'`' => return None,
            b'$' if bytes.get(index + 1) == Some(&b'(') => return None,
            b'<' if matches!(bytes.get(index + 1), Some(b'<') | Some(b'(')) => return None,
            b'|' | b'(' if !in_single && !in_double => return None,
            b'{' if !in_single && !in_double && bytes.get(index.wrapping_sub(1)) != Some(&b'$') => {
                return None;
            }
            b'&' | b';' if !in_single && !in_double => {
                let separator_len = match (byte, bytes.get(index + 1)) {
                    (b'&', Some(b'&')) => 2,
                    (b';', _) => 1,
                    _ => 0,
                };
                if separator_len > 0 {
                    parts.push(command[start..index].trim());
                    parts.push(&command[index..index + separator_len]);
                    start = index + separator_len;
                    saw_separator = true;
                    index += separator_len;
                    continue;
                }
            }
            _ => {}
        }
        index += 1;
    }
    if !saw_separator {
        return None;
    }
    let tail = command[start..].trim();
    if tail.is_empty() {
        return None;
    }
    parts.push(tail);
    if parts.iter().any(|part| part.is_empty()) {
        return None;
    }
    if parts
        .iter()
        .step_by(2)
        .any(|part| has_control_keyword(part))
    {
        return None;
    }
    Some(parts)
}

/// True when a shell operand contains a control-flow keyword as a word, which
/// would make per-operand rewriting unsafe.  Conservative: a false positive
/// only skips the rtk split optimization, never alters execution.
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

/// Detect and cache the installed rtk version.  The binary cannot change
/// during the process lifetime, so a miss is cached too.
async fn rtk_version() -> Option<(u64, u64, u64)> {
    if let Some(cached) = RTK_VERSION.get() {
        return *cached;
    }
    let detected = detect_rtk_version().await;
    let _ = RTK_VERSION.set(detected);
    detected
}

async fn detect_rtk_version() -> Option<(u64, u64, u64)> {
    let output = tokio::time::timeout(
        RTK_REWRITE_TIMEOUT,
        Command::new("rtk")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    parse_rtk_version(&String::from_utf8_lossy(&output.stdout))
}

async fn rtk_exit_codes_trustworthy() -> bool {
    match rtk_version().await {
        Some(version) => rtk_exit_codes_trustworthy_for(version),
        None => false,
    }
}

fn rtk_exit_codes_trustworthy_for(version: (u64, u64, u64)) -> bool {
    version >= RTK_TRUSTED_EXIT_CODE_VERSION
}

/// Extract the first `major.minor.patch` triplet from `rtk --version` output
/// (e.g. `rtk 0.45.0` or `0.46.0-beta.1`), tolerating suffixes and arbitrary
/// surrounding text.
fn parse_rtk_version(value: &str) -> Option<(u64, u64, u64)> {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        while index < bytes.len() && !bytes[index].is_ascii_digit() {
            index += 1;
        }
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if start == index {
            return None;
        }
        let mut parts = value[start..].splitn(3, '.');
        let (Some(major), Some(minor)) = (parts.next(), parts.next()) else {
            continue;
        };
        let (Ok(major), Ok(minor)) = (major.parse::<u64>(), minor.parse::<u64>()) else {
            continue;
        };
        let patch = parts
            .next()
            .and_then(|part| {
                let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
                digits.parse::<u64>().ok()
            })
            .unwrap_or(0);
        return Some((major, minor, patch));
    }
    None
}

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
            name: "bash".into(),
            description: "Run a shell command in the working directory. Returns stdout and stderr; output keeps the tail. Optionally run in a workspace-relative directory via the dir argument.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command passed to sh -c" },
                    "dir": { "type": "string", "description": "Optional working directory for the command, relative to the workspace root (e.g. \"crates/tools\"). Prefer this over prefixing the command with cd <dir> && ..." },
                    "timeout": { "type": "integer", "minimum": 1, "description": "Timeout in seconds (default 120)" }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
            },
            prompt: ToolPrompt::new(
                "Execute commands and project operations",
                [
                    "Use bash for tests, builds, git, and operations not covered by a dedicated tool.".to_owned(),
                    "When a command must run in a subdirectory, pass it as the dir argument instead of prefixing the command with cd <dir> && ...".to_owned(),
                ],
            ),
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

        // With rtk enabled, supported commands run through their compact rtk
        // equivalent; anything else runs verbatim.  rtk rewrites most single
        // commands and many `cd X && cmd` compounds, but its compound support
        // is selective; when the whole command comes back untouched we split
        // and rewrite each operand so the real work still gets the rtk form.
        let run_command = if self.rtk {
            match rtk_rewrite(&command).await {
                Some(rewritten) => rewritten,
                None => rtk_rewrite_compound(&command)
                    .await
                    .unwrap_or_else(|| command.clone()),
            }
        } else {
            command.clone()
        };

        let cwd = self.cwd.clone();
        let mut child = match Command::new("sh")
            .arg("-c")
            .arg(&run_command)
            .current_dir(run_dir.as_deref().unwrap_or(&cwd))
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

        let mut stdout = child.stdout.take().expect("stdout was piped");
        let mut stderr = child.stderr.take().expect("stderr was piped");
        let stdout_task = tokio::spawn(async move {
            let mut bytes = Vec::new();
            let result = stdout.read_to_end(&mut bytes).await;
            (result, bytes)
        });
        let stderr_task = tokio::spawn(async move {
            let mut bytes = Vec::new();
            let result = stderr.read_to_end(&mut bytes).await;
            (result, bytes)
        });

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
            _ = tokio::time::sleep(Duration::from_secs(timeout)) => {
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

        let (_, stdout) = stdout_task.await.unwrap_or_else(|_| (Ok(0), Vec::new()));
        let (_, stderr) = stderr_task.await.unwrap_or_else(|_| (Ok(0), Vec::new()));
        let mut output = String::from_utf8_lossy(&stdout).into_owned();
        if !stderr.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str("--- stderr ---\n");
            output.push_str(&String::from_utf8_lossy(&stderr));
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
            summary: match dir.as_deref() {
                Some(dir) => format!("bash: {} (in {dir})", first_line(&run_command)),
                None => format!("bash: {}", first_line(&run_command)),
            },
        }
    }
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
    fn parses_rtk_version_strings() {
        assert_eq!(parse_rtk_version("rtk 0.45.0"), Some((0, 45, 0)));
        assert_eq!(parse_rtk_version("0.46.0"), Some((0, 46, 0)));
        assert_eq!(
            parse_rtk_version("rtk 0.45.0 (abcdef1234)"),
            Some((0, 45, 0))
        );
        assert_eq!(parse_rtk_version("rtk 0.45.0-beta.1"), Some((0, 45, 0)));
        assert_eq!(parse_rtk_version("1.2"), Some((1, 2, 0)));
        assert_eq!(parse_rtk_version("rtk version unknown"), None);
        assert_eq!(parse_rtk_version(""), None);
    }

    #[test]
    fn rtk_exit_code_trust_threshold_isolates_buggy_releases() {
        // 0.45.x exits 3 on success despite documenting 0; only 0.46.0+ is
        // trusted to follow the documented exit-code semantics.
        assert!(rtk_exit_codes_trustworthy_for((0, 46, 0)));
        assert!(rtk_exit_codes_trustworthy_for((1, 0, 0)));
        assert!(!rtk_exit_codes_trustworthy_for((0, 45, 0)));
        assert!(!rtk_exit_codes_trustworthy_for((0, 45, 9)));
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
    async fn rtk_rewrite_rewrites_supported_commands_only() {
        if !rtk_available().await {
            return;
        }
        assert_eq!(
            rtk_rewrite("git status").await.as_deref(),
            Some("rtk git status")
        );
        assert_eq!(rtk_rewrite("echo hi").await, None);
    }

    #[tokio::test]
    async fn compound_rewrite_rewrites_each_operand() {
        if !rtk_available().await {
            return;
        }
        // The trailing `git status` is rewritten even when a whole-command
        // rewrite would have missed the compound.
        assert_eq!(
            rtk_rewrite_compound("cd subdir && git status")
                .await
                .as_deref(),
            Some("cd subdir && rtk git status")
        );
        // Nothing supported anywhere: pass through unchanged.
        assert_eq!(rtk_rewrite_compound("cd a/b/c && npm test").await, None);
    }

    #[test]
    fn splits_compound_commands_safely() {
        assert_eq!(
            split_compound("cd src && cargo build").unwrap(),
            vec!["cd src", "&&", "cargo build"]
        );
        assert_eq!(
            split_compound("cd src && cargo build && cargo test").unwrap(),
            vec!["cd src", "&&", "cargo build", "&&", "cargo test"]
        );
        assert_eq!(
            split_compound("cd \"a b\" && echo hi").unwrap(),
            vec!["cd \"a b\"", "&&", "echo hi"]
        );
        assert_eq!(
            split_compound("cmd1 ; cmd2").unwrap(),
            vec!["cmd1", ";", "cmd2"]
        );
        // Single commands and unsafe constructs are left alone.
        assert_eq!(split_compound("cargo build"), None);
        assert_eq!(split_compound("if true; then echo hi; fi"), None);
        assert_eq!(split_compound("echo $(date) && x"), None);
        assert_eq!(split_compound("grep foo file | wc -l"), None);
        assert_eq!(split_compound("cat <<EOF && x\nEOF"), None);
        assert_eq!(split_compound("{ a && b; }"), None);
        assert_eq!(split_compound("cmd1 &&"), None);
    }

    #[tokio::test]
    async fn dir_argument_runs_command_in_subdirectory() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        let tool = BashTool::with_workspace_root(directory.path());
        let output = tool
            .execute(
                json!({"command": "pwd", "dir": "src"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        let expected = std::fs::canonicalize(directory.path().join("src")).unwrap();
        assert!(
            output
                .content
                .contains(&expected.to_string_lossy().into_owned())
        );
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

    #[tokio::test]
    async fn rtk_disabled_tool_runs_command_verbatim() {
        let directory = tempfile::tempdir().unwrap();
        let output = BashTool::with_workspace_root(directory.path())
            .execute(json!({"command": "echo hi"}), CancellationToken::new())
            .await;
        assert!(!output.is_error);
        assert_eq!(output.content.trim(), "hi");
    }
}
