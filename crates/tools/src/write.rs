use super::file_mutation::{atomic_write, with_file_mutation_lock};
use super::vfs::{WorkspaceFs, split_relative};
use super::{
    Tool, ToolOutput, ToolPrompt, ToolSpec, normalize_workspace_root, resolve_workspace_path,
};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio_util::sync::CancellationToken;

pub struct WriteTool {
    workspace_root: Option<PathBuf>,
    workspace_fs: Option<Arc<WorkspaceFs>>,
}

impl WriteTool {
    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        let root = normalize_workspace_root(root);
        Self {
            workspace_fs: WorkspaceFs::open_root(&root).ok().map(Arc::new),
            workspace_root: Some(root),
        }
    }

    /// Construct a writer using a workspace capability retained by registry
    /// assembly rather than reopening the root pathname for every write.
    pub fn with_workspace_fs(_root: impl Into<PathBuf>, workspace_fs: Arc<WorkspaceFs>) -> Self {
        Self {
            workspace_root: Some(workspace_fs.root().to_path_buf()),
            workspace_fs: Some(workspace_fs),
        }
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "write".into(),
                description: "Create or fully overwrite a text file.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the working directory" },
                        "content": { "type": "string", "description": "Complete file contents" }
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Create or replace files",
                ["Use write only for new files or full rewrites.".to_owned()],
            ),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let path = match args.get("path").and_then(Value::as_str) {
            Some(path) if !path.is_empty() => path.to_owned(),
            _ => return error("write", "missing required argument: path"),
        };
        let Some(content) = args.get("content").and_then(Value::as_str) else {
            return error(
                &format!("write {path}"),
                "missing required argument: content",
            );
        };
        if cancel.is_cancelled() {
            return error(&format!("write {path}"), "cancelled");
        }

        // Resolve lexically first (for precise workspace-relative errors),
        // then commit through the validated parent handle: the temp file is
        // created and renamed relative to the same handle, so the path
        // written is the path that was validated.
        let components =
            match resolve_workspace_path(&path, self.workspace_root.as_deref(), false).await {
                Ok(_) => match split_relative(&path) {
                    Ok(components) => components,
                    Err(io_error) => {
                        return error(
                            &format!("write {path}"),
                            &format!("cannot write {path}: {io_error}"),
                        );
                    }
                },
                Err(message) => {
                    return error(
                        &format!("write {path}"),
                        &format!("cannot write {path}: {message}"),
                    );
                }
            };
        let summary = format!("write {path}");
        let root = self.workspace_root.clone();
        let workspace_fs = self.workspace_fs.clone();
        let Some(result) = with_file_mutation_lock(
            &root
                .as_deref()
                .unwrap_or(std::path::Path::new("."))
                .join(components.join("/")),
            &cancel,
            || async {
                write_validated(
                    &root,
                    workspace_fs.as_deref(),
                    &components,
                    content,
                    &cancel,
                )
                .await
                .map_err(|io_error| format!("cannot write {path}: {io_error}"))
            },
        )
        .await
        else {
            return error(&summary, "cancelled");
        };
        if let Err(message) = result {
            return error(&summary, &message);
        }
        ToolOutput {
            content: format!("wrote {} bytes to {path}", content.len()),
            is_error: false,
            summary,
        }
    }
}

fn error(summary: &str, content: &str) -> ToolOutput {
    ToolOutput {
        content: content.to_owned(),
        is_error: true,
        summary: summary.to_owned(),
    }
}

/// Write `content` through validated handles: open the workspace root,
/// create missing parents handle-relatively, then commit the temp file
/// with a handle-relative rename. Falls back to path-based
/// [`atomic_write`] where handles are unavailable (non-Unix) or no root
/// is configured (compatibility mode).
async fn write_validated(
    root: &Option<PathBuf>,
    workspace_fs: Option<&WorkspaceFs>,
    components: &[String],
    content: &str,
    cancel: &CancellationToken,
) -> Result<(), std::io::Error> {
    if cancel.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "cancelled",
        ));
    }
    #[cfg(unix)]
    if let Some(root) = root.as_deref() {
        let owned_fs;
        let fs = if let Some(workspace_fs) = workspace_fs {
            workspace_fs
        } else {
            owned_fs = WorkspaceFs::open_root(root).map_err(|error| {
                std::io::Error::other(format!("cannot resolve workspace root: {error}"))
            })?;
            &owned_fs
        };
        let (parent_fd, name) = super::vfs::unix::open_parent_relative(fs, components)?;
        // Preserve existing permissions without re-walking names from `/`.
        // This handle-relative metadata check rejects an existing FIFO,
        // socket, device, or directory before any operation can block or the
        // atomic rename could silently replace a special file. It uses
        // fstatat rather than opening the contents, so write-only regular
        // files remain writable.
        let existing = super::vfs::unix::existing_file_permissions(
            std::os::fd::AsFd::as_fd(&parent_fd),
            name.as_str(),
        )?;
        super::file_mutation::atomic_write_at(
            &parent_fd,
            &name,
            content.as_bytes(),
            existing,
            cancel,
        )
        .await?;
        return Ok(());
    }
    // Fallback: reconstruct the lexical path and use the path-based commit.
    let base = root.clone().unwrap_or_else(|| PathBuf::from("."));
    let full_path = components.iter().fold(base, |base, part| base.join(part));
    if cancel.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "cancelled",
        ));
    }
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).await?;
    }
    if cancel.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "cancelled",
        ));
    }
    // Without Unix directory handles this is only a preflight: it rejects
    // known special files but cannot close a rename race or promise
    // cancellation-safe named-pipe behavior during the later path open.
    match fs::metadata(&full_path).await {
        Ok(metadata) if !metadata.is_file() => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    atomic_write(&full_path, content.as_bytes(), cancel).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    fn create_fifo(path: &std::path::Path) {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: the C string remains valid for this libc call.
        let result = unsafe { libc::mkfifo(path.as_ptr(), 0o600) };
        assert_eq!(
            result,
            0,
            "mkfifo failed: {}",
            std::io::Error::last_os_error()
        );
    }

    #[tokio::test]
    async fn creates_parents_and_overwrites() {
        let dir = tempdir().unwrap();
        let tool = WriteTool::with_workspace_root(dir.path());
        let output = tool
            .execute(
                json!({"path": "a/b/file.txt", "content":"first"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/file.txt")).unwrap(),
            "first"
        );
        tool.execute(
            json!({"path": "a/b/file.txt", "content":"second"}),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/file.txt")).unwrap(),
            "second"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn existing_fifo_rejection_is_responsive_to_a_timeout() {
        use std::time::Duration;

        let dir = tempdir().unwrap();
        create_fifo(&dir.path().join("output.fifo"));
        let tool = WriteTool::with_workspace_root(dir.path());
        let task = tokio::spawn(async move {
            tool.execute(
                json!({"path": "output.fifo", "content": "would block"}),
                CancellationToken::new(),
            )
            .await
        });
        let output = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("inspecting a FIFO must not block the Tokio worker")
            .expect("write task panicked");
        assert!(output.is_error);
        assert!(
            output.content.contains("not a regular file"),
            "{}",
            output.content
        );
    }

    /// End-to-end TOCTOU barrier for `write`: resolve, swap an ancestor
    /// for an external symlink, then execute. The handle-relative commit
    /// must refuse and leave both trees unchanged.
    #[cfg(unix)]
    #[tokio::test]
    async fn retained_workspace_capability_survives_root_path_replacement() {
        let workspace = tempdir().unwrap();
        let root = std::fs::canonicalize(workspace.path()).unwrap();
        let moved = root.with_extension("moved");
        let outside = tempdir().unwrap();
        std::fs::write(root.join("file.txt"), "workspace").unwrap();
        std::fs::write(outside.path().join("file.txt"), "external").unwrap();
        let tool = WriteTool::with_workspace_root(&root);

        std::fs::rename(&root, &moved).unwrap();
        std::os::unix::fs::symlink(outside.path(), &root).unwrap();
        let output = tool
            .execute(
                json!({"path": "file.txt", "content": "updated"}),
                CancellationToken::new(),
            )
            .await;

        assert!(!output.is_error, "{}", output.content);
        assert_eq!(
            std::fs::read_to_string(moved.join("file.txt")).unwrap(),
            "updated"
        );
        assert_eq!(
            std::fs::read_to_string(outside.path().join("file.txt")).unwrap(),
            "external"
        );
        std::fs::remove_dir_all(moved).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_cannot_create_through_swapped_ancestor() {
        let workspace = tempdir().unwrap();
        let root = std::fs::canonicalize(workspace.path()).unwrap();
        let outside = tempdir().unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/keep.txt"), "keep").unwrap();
        let tool = WriteTool::with_workspace_root(&root);
        // Warm the resolution path, then swap the ancestor.
        let before = tool
            .execute(
                json!({"path": "sub/warm.txt", "content": "warm"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!before.is_error, "{}", before.content);
        std::fs::remove_dir_all(root.join("sub")).unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("sub")).unwrap();
        let output = tool
            .execute(
                json!({"path": "sub/evil.txt", "content": "evil"}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error, "write escaped: {}", output.content);
        assert!(!outside.path().join("evil.txt").exists());
        assert!(!outside.path().join("sub/evil.txt").exists());
    }
}
