use super::{
    Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec, expand_tilde, normalize_workspace_root,
    resolve_workspace_path,
};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncReadExt;
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
        // skill paths (or a `~`-expanded absolute under one of them). `read`
        // can load a discovered skill's files from any location (project or
        // global roots).
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
                description: "Read a text file, optionally selecting a range of lines. Text files are detected by scanning the first 8 KB; files containing NUL bytes are treated as binary and rejected. Output is capped at 2,000 lines and 50 KiB. Paths may be relative to the working directory or absolute; skill paths (SKILL.md and resources) are readable from any location.".into(),
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

    fn concurrency(&self, _args: &Value) -> Concurrency {
        Concurrency::ReadOnly
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
        let file = match fs::File::open(&full_path).await {
            Ok(file) => file,
            Err(io_error) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {io_error}"),
                );
            }
        };
        let selected = match stream_range(file, offset, limit, &cancel).await {
            Ok(content) => content,
            Err(message) => return error(&format!("read {path}"), &message),
        };

        ToolOutput {
            content: selected,
            is_error: false,
            summary: format!("read {path}"),
        }
    }
}

async fn stream_range(
    mut file: fs::File,
    offset: usize,
    limit: Option<usize>,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let mut prefix = vec![0; 8 * 1024];
    let prefix_len = file
        .read(&mut prefix)
        .await
        .map_err(|error| format!("cannot read file: {error}"))?;
    prefix.truncate(prefix_len);
    if prefix.contains(&0) {
        return Err("binary file not supported".into());
    }

    // Replay the detection prefix, then continue from the file's current
    // position. Only selected bytes are retained; skipped lines and the tail
    // after a satisfied explicit limit are never materialized.
    let mut reader = std::io::Cursor::new(prefix).chain(file);
    let mut buffer = [0u8; 8 * 1024];
    let mut output = Vec::new();
    let mut line = 1usize;
    let mut shown = 0usize;
    let wanted = limit.unwrap_or(usize::MAX).min(MAX_LINES);
    let mut truncated = false;
    let mut oversized_line = false;

    'read: loop {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let read = reader
            .read(&mut buffer)
            .await
            .map_err(|error| format!("cannot read file: {error}"))?;
        if read == 0 {
            break;
        }
        for &byte in &buffer[..read] {
            let selected = line >= offset && shown < wanted;
            if byte == b'\n' {
                if selected {
                    shown += 1;
                    if shown >= wanted {
                        break 'read;
                    }
                    if output.len() < MAX_BYTES {
                        output.push(b'\n');
                    }
                }
                line = line.saturating_add(1);
                continue;
            }
            if selected {
                if output.len() >= MAX_BYTES {
                    truncated = true;
                    oversized_line = true;
                    break 'read;
                }
                output.push(byte);
            }
        }
    }

    // A safety line cap is truncation only when the caller did not request an
    // equally narrow explicit range. We stop as soon as the cap is reached.
    if limit.is_none_or(|limit| limit > MAX_LINES) && shown >= MAX_LINES {
        truncated = true;
    }
    let mut text = match String::from_utf8(output) {
        Ok(text) => text,
        Err(error) if truncated && error.utf8_error().error_len().is_none() => {
            let valid = error.utf8_error().valid_up_to();
            let mut bytes = error.into_bytes();
            bytes.truncate(valid);
            String::from_utf8(bytes).expect("prefix ending at valid_up_to is UTF-8")
        }
        Err(_) => return Err("binary file not supported".to_owned()),
    };
    while text.ends_with('\n') {
        text.pop();
    }
    if truncated {
        if !text.is_empty() {
            text.push('\n');
        }
        if oversized_line {
            text.push_str("[truncated: oversized line exceeded the 50 KiB byte limit]");
        } else {
            text.push_str("[truncated: 2,000-line safety limit reached]");
        }
    }
    Ok(text)
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
        let output = ReadTool::with_workspace_root(dir.path())
            .execute(
                json!({"path":"file.txt", "offset":2, "limit":2}),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(output.content, "two\nthree");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn rejects_binary_and_missing_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("binary");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"ok\0no").unwrap();
        let output = ReadTool::with_workspace_root(dir.path())
            .execute(json!({"path": "binary"}), CancellationToken::new())
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
