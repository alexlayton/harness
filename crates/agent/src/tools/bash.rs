use super::{Tool, ToolOutput};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

pub struct BashTool {
    rtk: bool,
}

impl BashTool {
    pub fn new() -> Self {
        Self { rtk: false }
    }

    /// A tool that rewrites supported commands to their token-optimized `rtk`
    /// equivalents before execution (see [`rtk_rewrite`]).
    pub fn with_rtk(rtk: bool) -> Self {
        Self { rtk }
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1024;
const RTK_REWRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Ask rtk to rewrite a command to its token-optimized equivalent.  rtk
/// signals support by printing the rewritten command on stdout; unsupported
/// commands, a missing rtk binary, and timeouts all degrade to `None`, in
/// which case the caller runs the original command unchanged.  The exit code
/// is deliberately ignored: rtk documents 0 for a successful rewrite but
/// 0.45.0 exits 3, so non-empty stdout is the only reliable signal.
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
    if rewritten.is_empty() {
        None
    } else {
        Some(rewritten)
    }
}

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".into(),
            description: "Run a shell command in the working directory.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command passed to sh -c" },
                    "timeout": { "type": "integer", "minimum": 1, "description": "Timeout in seconds (default 120)" }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
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
        if cancel.is_cancelled() {
            return error(&format!("bash: {}", first_line(&command)), "cancelled");
        }

        // With rtk enabled, supported commands run through their compact rtk
        // equivalent; anything else runs verbatim.  The summary keeps naming
        // the original command the model asked for.
        let run_command = if self.rtk {
            rtk_rewrite(&command).await.unwrap_or_else(|| command.clone())
        } else {
            command.clone()
        };

        let cwd = match std::env::current_dir() {
            Ok(cwd) => cwd,
            Err(io_error) => {
                return error(
                    "bash",
                    &format!("cannot determine working directory: {io_error}"),
                );
            }
        };
        let mut child = match Command::new("sh")
            .arg("-c")
            .arg(&run_command)
            .current_dir(cwd)
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
            summary: format!("bash: {}", first_line(&command)),
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
        let output = BashTool::new()
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
        let output = BashTool::new()
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
        let output = BashTool::with_rtk(true)
            .execute(json!({"command": "git status"}), CancellationToken::new())
            .await;
        assert!(!output.is_error, "{}", output.content);
        // `rtk git status` prints a compact `* <branch>` header instead of
        // git's "On branch", proving the rewrite path ran.
        assert!(output.content.starts_with("* "), "{}", output.content);
        // The summary still names the command the model asked for.
        assert_eq!(output.summary, "bash: git status");
    }

    #[tokio::test]
    async fn rtk_enabled_tool_falls_back_for_unsupported_commands() {
        if !rtk_available().await {
            return;
        }
        let output = BashTool::with_rtk(true)
            .execute(json!({"command": "echo hi"}), CancellationToken::new())
            .await;
        assert!(!output.is_error);
        assert_eq!(output.content.trim(), "hi");
    }

    #[tokio::test]
    async fn rtk_disabled_tool_runs_command_verbatim() {
        let output = BashTool::new()
            .execute(json!({"command": "echo hi"}), CancellationToken::new())
            .await;
        assert!(!output.is_error);
        assert_eq!(output.content.trim(), "hi");
    }
}
