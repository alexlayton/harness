use super::vfs::{WorkspaceFs, split_relative};
use super::{
    Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec, expand_tilde, normalize_workspace_root,
    resolve_workspace_path,
};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

pub struct ReadTool {
    workspace_root: Option<PathBuf>,
    /// The workspace capability retained during registry assembly on Unix.
    /// Workspace files are opened relative to this handle, never by reopening
    /// the root pathname after validation. Non-Unix reads fail closed because
    /// `WorkspaceFs` cannot provide the same capability there.
    workspace_fs: Option<Arc<WorkspaceFs>>,
    /// Validated capabilities for discovered skill roots/files. Each entry
    /// retains the directory handle used for the final read, so an allowlisted
    /// pathname cannot be redirected after discovery.
    allowed_paths: Option<Vec<AllowedPath>>,
}

#[derive(Clone)]
struct AllowedPath {
    base: PathBuf,
    root: Arc<WorkspaceFs>,
    prefix: Vec<String>,
    directory: bool,
}

impl ReadTool {
    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        let root = normalize_workspace_root(root);
        Self {
            workspace_fs: WorkspaceFs::open_root(&root).ok().map(Arc::new),
            workspace_root: Some(root),
            allowed_paths: None,
        }
    }

    /// Construct a workspace-aware reader using a capability retained by the
    /// registry. The handle must have been opened against `root`.
    pub fn with_workspace_fs(_root: impl Into<PathBuf>, workspace_fs: Arc<WorkspaceFs>) -> Self {
        Self {
            workspace_root: Some(workspace_fs.root().to_path_buf()),
            workspace_fs: Some(workspace_fs),
            allowed_paths: None,
        }
    }

    /// Add a set of allowed absolute paths (from the skills catalog). On Unix,
    /// the paths are canonicalized and their parent capabilities are retained;
    /// subsequent reads do not reopen them by name. Non-Unix reads fail closed
    /// because no equivalent capability is available.
    pub fn with_allowed_paths(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut allowed = Vec::new();
        for path in paths {
            let Ok(base) = std::fs::canonicalize(&path) else {
                continue;
            };
            let Ok(metadata) = std::fs::metadata(&base) else {
                continue;
            };
            if metadata.is_dir() {
                let Ok(root) = WorkspaceFs::open_root(&base) else {
                    continue;
                };
                allowed.push(AllowedPath {
                    base,
                    root: Arc::new(root),
                    prefix: Vec::new(),
                    directory: true,
                });
            } else if let Some(parent) = base.parent()
                && let Some(name) = base
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                && let Ok(root) = WorkspaceFs::open_root(parent)
            {
                allowed.push(AllowedPath {
                    base,
                    root: Arc::new(root),
                    prefix: vec![name],
                    directory: false,
                });
            }
        }
        self.allowed_paths = Some(allowed);
        self
    }
}

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1024;

impl ReadTool {
    /// Resolve `path` to workspace-relative components, or to an allowed
    /// skill path. Returns the components plus an optional pre-opened
    /// skill file (skill paths live outside the workspace handle).
    async fn resolve_components(&self, path: &str) -> Result<ReadTarget, String> {
        if let Some(root) = self.workspace_root.as_deref() {
            // Workspace-relative (or absolute-but-inside) paths become
            // components for the handle-relative open below. Absolute
            // paths inside the root are relativized; anything else falls
            // through to the skill allowlist.
            let candidate = PathBuf::from(path);
            let as_relative = if candidate.is_absolute() {
                candidate
                    .strip_prefix(root)
                    .ok()
                    .map(|rel| rel.to_string_lossy().into_owned())
            } else {
                Some(path.to_owned())
            };
            if let Some(relative) = as_relative {
                match split_relative(&relative) {
                    Ok(components) => return Ok(ReadTarget::Workspace(components)),
                    Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {
                        return Err(format!("cannot read {path}: {error}"));
                    }
                    Err(_) => {}
                }
            }
            // Fall back to the legacy validator for its precise
            // workspace-relative error messages (tests assert on them).
            if resolve_workspace_path(path, Some(root), false)
                .await
                .is_ok()
            {
                return split_relative(path)
                    .map(ReadTarget::Workspace)
                    .map_err(|error| format!("cannot read {path}: {error}"));
            }
        }
        // Otherwise, allow an absolute path that is under one of the allowed
        // skill paths (or a `~`-expanded absolute under one of them). `read`
        // can load a discovered skill's files from any location (project or
        // global roots). The candidate is canonicalized first: the
        // allowlist stores canonical paths, and on macOS temp dirs
        // (`/var` -> `/private/var`) would otherwise never prefix-match.
        // Uncanonicalizable paths fall through to the external branch
        // below, which reports the OS error.
        let raw_candidate = expand_tilde(&PathBuf::from(path));
        if !raw_candidate.is_absolute() {
            return Err(format!("cannot read {path}: outside workspace"));
        }
        let candidate =
            std::fs::canonicalize(&raw_candidate).unwrap_or_else(|_| raw_candidate.clone());
        if let Some(allowed) = self.allowed_paths.as_deref() {
            for capability in allowed {
                if capability.directory {
                    if let Ok(relative) = candidate.strip_prefix(&capability.base)
                        && let Ok(mut components) = split_relative(&relative.to_string_lossy())
                    {
                        components.splice(0..0, capability.prefix.iter().cloned());
                        return Ok(ReadTarget::Skill {
                            capability: capability.clone(),
                            components,
                        });
                    }
                } else if candidate == capability.base {
                    return Ok(ReadTarget::Skill {
                        capability: capability.clone(),
                        components: capability.prefix.clone(),
                    });
                }
            }
        }
        // Any other absolute path is readable directly. The shell was never
        // a sandbox, so rejecting an external `read` only pushed the model
        // to `cat` the same file through `bash`; reading it openly keeps the
        // access visible in tool history instead. The usual read limits
        // (text sniffing, line/byte caps, regular-file check) still apply.
        // Prefer the raw (non-canonicalized) path so error messages echo
        // what the caller passed.
        Ok(ReadTarget::External(raw_candidate))
    }
}

/// Where a `read` resolves to: workspace components opened through the
/// validated handle, an allowlisted skill file opened through its retained
/// handle, or an absolute path outside the workspace opened directly.
enum ReadTarget {
    Workspace(Vec<String>),
    Skill {
        capability: AllowedPath,
        components: Vec<String>,
    },
    External(PathBuf),
}

/// Open a resolved target, preserving workspace-relative error messages
/// without leaking outside paths. Workspace files go through the
/// validated handle (`O_NOFOLLOW` per component); skill files use the same
/// retained handle. External paths are opened directly: they carry no
/// containment promise, so a plain open plus a regular-file check (matching
/// what the shell would do) is sufficient. This is only compiled on Unix:
/// non-Unix workspace reads fail closed before resolution or opening.
#[cfg(unix)]
async fn open_target(
    target: &ReadTarget,
    root: Option<&Path>,
    workspace_fs: Option<&WorkspaceFs>,
) -> Result<fs::File, String> {
    match target {
        ReadTarget::External(path) => open_external(path).map(fd_into_tokio_file),
        ReadTarget::Skill {
            capability,
            components,
        } => {
            let fd = super::vfs::unix::open_file_relative(&capability.root, components)
                .map_err(|error| format!("cannot read file: {error}"))?;
            Ok(fd_into_tokio_file(fd))
        }
        ReadTarget::Workspace(components) => {
            let Some(root) = root else {
                return Err("cannot read file: no workspace root".into());
            };
            let owned_fs;
            let fs = if let Some(workspace_fs) = workspace_fs {
                workspace_fs
            } else {
                owned_fs = WorkspaceFs::open_root(root)
                    .map_err(|error| format!("cannot resolve workspace root: {error}"))?;
                &owned_fs
            };
            let fd = super::vfs::unix::open_file_relative(fs, components)
                .map_err(|error| format!("cannot read file: {error}"))?;
            Ok(fd_into_tokio_file(fd))
        }
    }
}

#[cfg(unix)]
/// Open an absolute path outside the workspace. Unlike the workspace walk
/// this follows a symlinked final component (as `cat` would) but still
/// opens with `O_NONBLOCK` so a FIFO can never stall the executor, then
/// refuses anything that is not a regular file via `fstat` on the open
/// handle. Errors name only the failure, never the file contents.
fn open_external(path: &Path) -> Result<std::os::fd::OwnedFd, String> {
    use rustix::fs::{Mode, OFlags};
    let fd = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| format!("cannot read file: {error}"))?;
    let file_type = rustix::fs::FileType::from_raw_mode(
        rustix::fs::fstat(&fd)
            .map_err(|error| format!("cannot read file: {error}"))?
            .st_mode,
    );
    let name = path.to_string_lossy();
    super::vfs::unix::ensure_regular_file(&name, file_type)
        .map_err(|error| format!("cannot read file: {error}"))?;
    Ok(fd)
}

#[cfg(not(unix))]
async fn open_external_file(path: &Path) -> Result<fs::File, String> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|error| format!("cannot read file: {error}"))?;
    if !metadata.is_file() {
        return Err("cannot read file: not a regular file".into());
    }
    tokio::fs::File::open(path)
        .await
        .map_err(|error| format!("cannot read file: {error}"))
}
#[cfg(unix)]
fn fd_into_tokio_file(fd: std::os::fd::OwnedFd) -> fs::File {
    use std::os::fd::IntoRawFd;
    use std::os::unix::io::FromRawFd;
    // SAFETY: `fd` is owned (came from `openat`); transferring it to a
    // `std::fs::File` preserves single ownership, then into tokio.
    let raw = fd.into_raw_fd();
    let std_file = unsafe { std::fs::File::from_raw_fd(raw) };
    fs::File::from_std(std_file)
}

#[async_trait]
impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "read".into(),
                description: "Read a text file, optionally selecting a range of lines. Text files are detected by scanning the first 8 KB; files containing NUL bytes are treated as binary and rejected. Output is capped at 2,000 lines and 50 KiB. Paths may be relative to the working directory or absolute; absolute paths outside the working directory are readable too (skill paths included).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the working directory, or absolute path (readable even outside the working directory)" },
                        "offset": { "type": "integer", "minimum": 1, "description": "First 1-indexed line" },
                        "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of lines" }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Read files",
                ["Prefer read to cat or sed.".to_owned()],
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

        // Resolve first: workspace paths get lexical confinement plus the
        // validated-handle open below; absolute paths outside the workspace
        // fall through to a direct open (see `ReadTarget::External`).
        let target = match self.resolve_components(&path).await {
            Ok(target) => target,
            Err(message) => {
                return error(
                    &format!("read {path}"),
                    &format!("cannot read {path}: {message}"),
                );
            }
        };
        #[cfg(not(unix))]
        {
            // Workspace reads stay fail-closed where handle-relative I/O
            // is unavailable; external reads use a plain open instead.
            if !matches!(target, ReadTarget::External(_)) {
                return error(
                    &format!("read {path}"),
                    &format!(
                        "cannot read {path}: {}",
                        WorkspaceFs::unsupported_operation("read")
                    ),
                );
            }
            let ReadTarget::External(external) = &target else {
                unreachable!("just matched external");
            };
            let file = match open_external_file(external).await {
                Ok(file) => file,
                Err(message) => return error(&format!("read {path}"), &message),
            };
            let selected = match stream_range(file, offset, limit, &cancel).await {
                Ok(content) => content,
                Err(message) => return error(&format!("read {path}"), &message),
            };
            return ToolOutput {
                content: selected,
                is_error: false,
                summary: format!("read {path}"),
            };
        }
        #[cfg(unix)]
        let file = match open_target(
            &target,
            self.workspace_root.as_deref(),
            self.workspace_fs.as_deref(),
        )
        .await
        {
            Ok(file) => file,
            Err(message) => {
                return error(&format!("read {path}"), &message);
            }
        };
        #[cfg(not(unix))]
        let file: fs::File = unreachable!("non-Unix reads fail closed before opening a path");
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

    #[cfg(unix)]
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

    #[cfg(unix)]
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

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fifo_rejection_is_responsive_to_a_timeout() {
        use std::time::Duration;

        let dir = tempdir().unwrap();
        create_fifo(&dir.path().join("input.fifo"));
        let tool = ReadTool::with_workspace_root(dir.path());
        let task = tokio::spawn(async move {
            tool.execute(json!({"path": "input.fifo"}), CancellationToken::new())
                .await
        });
        let output = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("opening a FIFO must not block the Tokio worker")
            .expect("read task panicked");
        assert!(output.is_error);
        assert!(
            output.content.contains("not a regular file"),
            "{}",
            output.content
        );
    }

    #[cfg(unix)]
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

    #[cfg(unix)]
    #[tokio::test]
    async fn absolute_paths_outside_workspace_are_readable() {
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
        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("top secret"));
        // Missing files, FIFOs, and binary files are rejected the same way
        // inside and outside the workspace (and never leak contents).
        let missing = outside.path().join("missing.txt");
        let output = tool
            .execute(
                json!({"path": missing.to_string_lossy()}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error, "{}", output.content);
        create_fifo(&outside.path().join("input.fifo"));
        let output = tool
            .execute(
                json!({"path": outside.path().join("input.fifo").to_string_lossy()}),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error, "{}", output.content);
        assert!(
            output.content.contains("not a regular file"),
            "{}",
            output.content
        );
        // Relative escapes still stay confined.
        let output = tool
            .execute(json!({"path": "../escape.txt"}), CancellationToken::new())
            .await;
        assert!(output.is_error);
        assert!(
            output.content.contains("outside workspace"),
            "{}",
            output.content
        );
    }

    /// Non-Unix intentionally performs no path lookup after an ancestor is
    /// replaced. This deterministic test covers the fail-closed choice
    /// without depending on Windows junction privileges.
    #[cfg(not(unix))]
    #[tokio::test]
    async fn non_unix_read_is_disabled_before_an_ancestor_swap() {
        let workspace = tempdir().unwrap();
        let outside = tempdir().unwrap();
        std::fs::create_dir(workspace.path().join("sub")).unwrap();
        std::fs::write(outside.path().join("secret.txt"), "EXTERNAL-SECRET").unwrap();
        let tool = ReadTool::with_workspace_root(workspace.path());
        std::fs::rename(
            workspace.path().join("sub"),
            workspace.path().join("sub.old"),
        )
        .unwrap();
        std::fs::create_dir(workspace.path().join("sub")).unwrap();

        let output = tool
            .execute(json!({"path": "sub/secret.txt"}), CancellationToken::new())
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("disabled"), "{}", output.content);
        assert!(!output.content.contains("EXTERNAL-SECRET"));
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "EXTERNAL-SECRET"
        );
    }

    /// End-to-end TOCTOU barrier for `read`: resolve, swap an ancestor for
    /// an external symlink, then execute. The handle-relative open must
    /// refuse the swapped tree and leave both files unchanged.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_cannot_read_through_swapped_ancestor() {
        let workspace = tempdir().unwrap();
        let root = std::fs::canonicalize(workspace.path()).unwrap();
        let outside = tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "EXTERNAL-SECRET").unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/file.txt"), "inside-content").unwrap();
        let tool = ReadTool::with_workspace_root(&root);
        // Warm the resolution path, then swap the ancestor.
        let before = tool
            .execute(json!({"path": "sub/file.txt"}), CancellationToken::new())
            .await;
        assert!(!before.is_error, "{}", before.content);
        assert!(before.content.contains("inside-content"));
        std::fs::remove_dir_all(root.join("sub")).unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("sub")).unwrap();
        let output = tool
            .execute(json!({"path": "sub/file.txt"}), CancellationToken::new())
            .await;
        assert!(output.is_error, "read escaped: {}", output.content);
        assert!(
            !output.content.contains("EXTERNAL-SECRET"),
            "leaked: {}",
            output.content
        );
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "EXTERNAL-SECRET"
        );
    }
}
