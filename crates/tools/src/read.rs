use super::{
    Tool, ToolOutput, ToolPrompt, ToolSpec, expand_tilde, normalize_workspace_root,
    resolve_workspace_path,
};
use async_trait::async_trait;
use llm::ToolDefinition;
use llm::util::truncate_utf8_prefix;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::fs;
use tokio_util::sync::CancellationToken;

pub struct ReadTool {
    workspace_root: Option<PathBuf>,
    /// Absolute paths (files AND dirs) `read` may access for agent skills
    /// (every discovered `SKILL.md` file_path plus skill base_dirs). A path
    /// is readable when it is under the workspace root or under one of these.
    /// Populated from `SkillCatalog::read_paths`; `None` means no allowlist
    /// (arbitrary absolute paths are rejected unless under the workspace).
    allowed_paths: Option<Vec<PathBuf>>,
}

impl ReadTool {
    /// Compatibility constructor: relative paths use the process cwd and
    /// absolute paths retain the historical behavior.
    pub fn new() -> Self {
        Self {
            workspace_root: None,
            allowed_paths: None,
        }
    }

    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: Some(normalize_workspace_root(root)),
            allowed_paths: None,
        }
    }

    /// Add a set of allowed absolute paths (from the skills catalog). These
    /// are canonicalized at call time; a path is readable when it is under
    /// the workspace root or under one of these.
    pub fn with_allowed_paths(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.allowed_paths = Some(paths.into_iter().collect());
        self
    }

    pub fn allowed_paths(&self) -> Option<&[PathBuf]> {
        self.allowed_paths.as_deref()
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1024;

impl ReadTool {
    async fn resolve_path(&self, path: &str) -> Result<PathBuf, String> {
        // First try the workspace-rooted resolution (handles containment,
        // lexical `..`, symlink escapes).
        if let Ok(resolved) =
            resolve_workspace_path(path, self.workspace_root.as_deref(), false).await
        {
            return Ok(resolved);
        }
        // Otherwise, allow an absolute path that is under one of the allowed
        // skill paths (or a `~`-expanded absolute under one of them). This is
        // the pi behavior: `read` can load a skill's SKILL.md from anywhere
        // it was discovered (project or global).
        let candidate = expand_tilde(&PathBuf::from(path));
        if !candidate.is_absolute() {
            return Err(format!("cannot read {path}: outside workspace"));
        }
        // Canonicalize both the candidate and each allowed base so symlink
        // roots (e.g. /tmp -> /private/tmp on macOS) compare equal.
        let canonical = fs::canonicalize(&candidate)
            .await
            .map_err(|e| format!("cannot resolve path {path}: {e}"))?;
        if let Some(allowed) = self.allowed_paths.as_deref() {
            for base in allowed {
                if let Ok(base) = fs::canonicalize(base).await
                    && canonical.starts_with(&base)
                {
                    return Ok(canonical);
                }
            }
        }
        Err("path is outside workspace root and not an allowed skill path".to_owned())
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "read".into(),
                description: "Read a text file, optionally selecting a range of lines. Text files are detected by scanning the first 8 KB; files containing NUL bytes are treated as binary and rejected. Paths may be relative to the working directory or absolute; skill paths (SKILL.md and resources) are readable from any location.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the working directory, or absolute path (including skill paths)" },
                        "offset": { "type": "integer", "minimum": 1, "description": "First 1-indexed line" },
                        "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of lines" }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Read file contents, optionally selecting a range of lines",
                ["Use read to examine file contents instead of cat or sed.".to_owned()],
            ),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let path = match args.get("path").and_then(Value::as_str) {
            Some(path) if !path.is_empty() => path.to_owned(),
            _ => return error("read", "missing required argument: path"),
        };
        let offset = match optional_positive(&args, "offset") {
            Ok(value) => value.unwrap_or(1),
            Err(message) => return error(&format!("read {path}"), &message),
        };
        let limit = match optional_positive(&args, "limit") {
            Ok(value) => value,
            Err(message) => return error(&format!("read {path}"), &message),
        };
        if cancel.is_cancelled() {
            return error(&format!("read {path}"), "cancelled");
        }

        let full_path = match self.resolve_path(&path).await {
            Ok(path) => path,
            Err(message) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {message}"),
                );
            }
        };
        let metadata = match fs::metadata(&full_path).await {
            Ok(metadata) => metadata,
            Err(io_error) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {io_error}"),
                );
            }
        };
        if metadata.is_dir() {
            return error(
                &format!("read {path}"),
                &format!("cannot read {path}: is a directory"),
            );
        }
        let bytes = match fs::read(&full_path).await {
            Ok(bytes) => bytes,
            Err(io_error) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {io_error}"),
                );
            }
        };
        if bytes[..bytes.len().min(8 * 1024)].contains(&0) {
            return error(&format!("read {path}"), "binary file not supported");
        }
        let content = match String::from_utf8(bytes) {
            Ok(content) => content,
            Err(_) => return error(&format!("read {path}"), "binary file not supported"),
        };
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let start = offset.saturating_sub(1).min(total);
        let requested_end = limit
            .map(|limit| start.saturating_add(limit).min(total))
            .unwrap_or(total);
        let mut end = requested_end.min(start.saturating_add(MAX_LINES));
        let mut selected = lines[start..end].join("\n");
        let mut byte_truncated = false;

        if selected.len() > MAX_BYTES {
            selected = truncate_utf8_prefix(&selected, MAX_BYTES).to_owned();
            // The byte limit can cut through a line.  Report the last complete
            // line when possible; the text itself remains useful for a huge line.
            let complete_lines = selected.lines().count();
            end = (start + complete_lines).min(end);
            byte_truncated = true;
        }

        // An explicit limit is intentional selection, not an implementation
        // truncation.  Only the safety caps (or an offset beyond the file)
        // receive the diagnostic notice.
        let truncated = byte_truncated || end < requested_end || (limit.is_none() && end < total);
        if truncated {
            let shown_start = if total == 0 { offset } else { start + 1 };
            let shown_end = if total == 0 {
                offset.saturating_sub(1)
            } else {
                end.max(start + 1)
            };
            let notice = format!("[truncated: showing lines {shown_start}–{shown_end} of {total}]");
            if !selected.is_empty() {
                selected.push('\n');
            }
            selected.push_str(&notice);
        }

        ToolOutput {
            content: selected,
            is_error: false,
            summary: format!("read {path}"),
        }
    }
}

fn optional_positive(args: &Value, name: &str) -> Result<Option<usize>, String> {
    let Some(value) = args.get(name) else {
        return Ok(None);
    };
    let Some(number) = value.as_u64() else {
        return Err(format!("{name} must be a positive integer"));
    };
    if number == 0 || number > usize::MAX as u64 {
        return Err(format!("{name} must be a positive integer"));
    }
    Ok(Some(number as usize))
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
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn reads_ranges_and_reports_truncation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();
        let old = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let output = ReadTool::new()
            .execute(
                json!({"path":"file.txt", "offset":2, "limit":2}),
                CancellationToken::new(),
            )
            .await;
        std::env::set_current_dir(old).unwrap();
        assert_eq!(output.content, "two\nthree");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn rejects_binary_and_missing_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("binary");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"ok\0no").unwrap();
        let output = ReadTool::new()
            .execute(
                json!({"path": dir.path().join("binary")}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("binary"));
    }

    #[tokio::test]
    async fn skill_paths_in_allowlist_are_readable_outside_workspace() {
        let skill_dir = tempdir().unwrap();
        let skill_file = skill_dir.path().join("SKILL.md");
        std::fs::write(
            &skill_file,
            "---\nname: test\ndescription: A test skill\n---\nbody line\n",
        )
        .unwrap();
        let workspace = tempdir().unwrap();
        let tool = ReadTool::with_workspace_root(workspace.path())
            .with_allowed_paths(vec![skill_file.clone(), skill_dir.path().to_path_buf()]);
        // Absolute path to the allowed skill file resolves and reads.
        let output = tool
            .execute(
                json!({"path": skill_file.to_string_lossy()}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("body line"));
        // A resource under the allowed base dir is readable too.
        let resource = skill_dir.path().join("references/guide.md");
        std::fs::create_dir_all(skill_dir.path().join("references")).unwrap();
        std::fs::write(&resource, "reference content").unwrap();
        let output = tool
            .execute(
                json!({"path": resource.to_string_lossy()}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("reference content"));
    }

    #[tokio::test]
    async fn unrelated_absolute_paths_are_rejected_without_workspace() {
        let outside = tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "top secret").unwrap();
        let workspace = tempdir().unwrap();
        let tool = ReadTool::with_workspace_root(workspace.path())
            .with_allowed_paths(vec![workspace.path().join("skills").to_path_buf()]);
        let output = tool
            .execute(
                json!({"path": secret.to_string_lossy()}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(
            output.content.contains("outside workspace"),
            "{}",
            output.content
        );
        assert!(!output.content.contains("top secret"));
    }
}
