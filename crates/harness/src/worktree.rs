use crate::config::WorktreeArgs;
use anyhow::{Context, Result, anyhow};
use fs2::FileExt;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Output, Stdio};

/// A prepared Git worktree whose lifetime surrounds one frontend run.
pub(crate) struct WorktreeLease {
    original_cwd: PathBuf,
    repository: Repository,
    worktree_path: PathBuf,
    keep: bool,
    created: bool,
    managed_destination: bool,
    source_was_dirty: bool,
    finished: bool,
    _run_lock: WorktreeRunLock,
}

/// The result of safely releasing a worktree after its frontend exits.
pub(crate) enum CleanupOutcome {
    Removed(PathBuf),
    Kept(PathBuf),
    Retained { path: PathBuf, reason: String },
}

#[derive(Clone, Debug)]
struct Repository {
    root: PathBuf,
    common_dir: PathBuf,
}

#[derive(Debug)]
struct WorktreeEntry {
    path: PathBuf,
    branch: Option<String>,
}

/// OS-backed process lease stored in the repository's common Git directory.
/// Git does not track whether a process is actively using a worktree, so this
/// prevents one Harness run from removing a clean tree beneath another run.
struct WorktreeRunLock {
    file: File,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnsureOutcome {
    Created,
    Reused,
}

/// Create or reuse the requested worktree and enter its corresponding launch
/// directory. The caller must call [`WorktreeLease::finish`] after the frontend
/// shuts down so the process leaves the directory before Git removes it.
pub(crate) fn prepare(args: &WorktreeArgs) -> Result<WorktreeLease> {
    let original_cwd =
        fs::canonicalize(std::env::current_dir().context("resolve the worktree launch directory")?)
            .context("resolve the worktree launch directory")?;
    let repository = Repository::discover(&original_cwd)?;
    validate_branch(&repository, &args.branch)?;

    let relative_cwd = original_cwd
        .strip_prefix(&repository.root)
        .with_context(|| {
            format!(
                "launch directory `{}` is outside repository `{}`",
                original_cwd.display(),
                repository.root.display()
            )
        })?
        .to_path_buf();
    let managed_destination = args.dir.is_none();
    let worktree_path = match &args.dir {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => original_cwd.join(path),
        None => default_destination(&repository, &args.branch)?,
    };
    let worktree_path = normalized_destination(&worktree_path)?;
    if let Ok(existing_path) = fs::canonicalize(&worktree_path)
        && original_cwd.starts_with(&existing_path)
    {
        return Err(anyhow!(
            "already inside selected worktree `{}`; run `harness` directly there",
            existing_path.display()
        ));
    }
    let run_lock = WorktreeRunLock::acquire(&repository, &args.branch)?;

    let source_was_dirty = source_has_changes(&repository)?;
    let ensured = ensure_worktree(
        &repository,
        &args.branch,
        args.start_point.as_deref(),
        &worktree_path,
    )?;
    let created = ensured == EnsureOutcome::Created;
    let resolved_worktree = match fs::canonicalize(&worktree_path) {
        Ok(path) => path,
        Err(error) => {
            rollback_created(&repository, &worktree_path, created);
            return Err(error)
                .with_context(|| format!("resolve worktree `{}`", worktree_path.display()));
        }
    };
    let workspace = resolved_worktree.join(relative_cwd);
    if !workspace.is_dir() {
        rollback_created(&repository, &worktree_path, created);
        return Err(anyhow!(
            "the launch subdirectory `{}` does not exist on branch `{}`",
            workspace.display(),
            args.branch
        ));
    }
    if let Err(error) = std::env::set_current_dir(&workspace) {
        rollback_created(&repository, &worktree_path, created);
        return Err(error).with_context(|| format!("enter worktree `{}`", workspace.display()));
    }
    let keep = match resolve_keep_policy(&repository, &args.branch, &resolved_worktree, args) {
        Ok(keep) => keep,
        Err(error) => {
            let _ = std::env::set_current_dir(&original_cwd);
            rollback_created(&repository, &resolved_worktree, created);
            return Err(error);
        }
    };

    Ok(WorktreeLease {
        original_cwd,
        repository,
        worktree_path: resolved_worktree,
        keep,
        created,
        managed_destination,
        source_was_dirty: created && source_was_dirty,
        finished: false,
        _run_lock: run_lock,
    })
}

impl WorktreeLease {
    pub(crate) fn path(&self) -> &Path {
        &self.worktree_path
    }

    pub(crate) fn was_created(&self) -> bool {
        self.created
    }

    pub(crate) fn source_was_dirty(&self) -> bool {
        self.source_was_dirty
    }

    /// Restore the launch directory, then remove the worktree without force.
    /// Git refusal is a retained outcome rather than an error. Modified or
    /// untracked files keep the tree; ignored-only output follows Git cleanup.
    pub(crate) fn finish(mut self) -> Result<CleanupOutcome> {
        std::env::set_current_dir(&self.original_cwd).with_context(|| {
            format!(
                "leave worktree and restore `{}`",
                self.original_cwd.display()
            )
        })?;
        self.finished = true;

        if self.keep {
            return Ok(CleanupOutcome::Kept(self.worktree_path.clone()));
        }
        let outcome = remove_worktree(&self.repository, &self.worktree_path);
        if self.managed_destination && matches!(outcome, CleanupOutcome::Removed(_)) {
            remove_empty_managed_parent(&self.worktree_path);
        }
        Ok(outcome)
    }
}

impl Drop for WorktreeLease {
    fn drop(&mut self) {
        // Explicit finish reports cleanup failures. This fallback only avoids
        // leaving the process inside the worktree on an unexpected early path.
        if !self.finished {
            let _ = std::env::set_current_dir(&self.original_cwd);
        }
    }
}

impl WorktreeRunLock {
    fn acquire(repository: &Repository, branch: &str) -> Result<Self> {
        let directory = repository.common_dir.join("harness").join("worktree-locks");
        fs::create_dir_all(&directory)
            .with_context(|| format!("create worktree lock directory `{}`", directory.display()))?;
        let path = directory.join(format!(
            "{}-{:016x}.lock",
            slug(branch, "branch"),
            stable_hash(branch.as_bytes())
        ));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open worktree lease `{}`", path.display()))?;

        if let Err(error) = file.try_lock_exclusive() {
            let owner = read_lock_owner(&mut file)
                .map(|pid| format!(" by process {pid}"))
                .unwrap_or_default();
            if error.kind() == std::io::ErrorKind::WouldBlock {
                return Err(anyhow!("branch `{branch}` is already in use{owner}"));
            }
            return Err(error).with_context(|| format!("lock worktree lease `{}`", path.display()));
        }
        file.set_len(0)
            .with_context(|| format!("reset worktree lease `{}`", path.display()))?;
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("seek worktree lease `{}`", path.display()))?;
        writeln!(file, "pid={}", std::process::id())
            .with_context(|| format!("write worktree lease `{}`", path.display()))?;
        file.sync_data()
            .with_context(|| format!("sync worktree lease `{}`", path.display()))?;
        Ok(Self { file })
    }
}

impl Drop for WorktreeRunLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

impl Repository {
    fn discover(cwd: &Path) -> Result<Self> {
        let root_output = git(cwd)
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .context("run Git to discover the repository")?;
        let root = output_text(root_output, "discover the Git repository")?;
        let root = fs::canonicalize(root.trim()).context("resolve the Git repository root")?;

        let common_output = git(&root)
            .args(["rev-parse", "--git-common-dir"])
            .output()
            .context("run Git to discover the common directory")?;
        let common = output_text(common_output, "discover the Git common directory")?;
        let common = PathBuf::from(common.trim());
        let common = if common.is_absolute() {
            common
        } else {
            root.join(common)
        };
        let common_dir = fs::canonicalize(&common)
            .with_context(|| format!("resolve Git common directory `{}`", common.display()))?;

        Ok(Self { root, common_dir })
    }
}

fn validate_branch(repository: &Repository, branch: &str) -> Result<()> {
    let output = git(&repository.root)
        .args(["check-ref-format", "--branch", branch])
        .output()
        .context("run Git to validate the branch")?;
    if output.status.success() {
        return Ok(());
    }
    Err(git_failure(
        &format!("invalid branch name `{branch}`"),
        &output,
    ))
}

fn branch_exists(repository: &Repository, branch: &str) -> Result<bool> {
    let reference = format!("refs/heads/{branch}");
    let output = git(&repository.root)
        .args(["show-ref", "--verify", "--quiet", &reference])
        .output()
        .context("run Git to inspect the branch")?;
    if output.status.success() {
        Ok(true)
    } else if output.status.code() == Some(1) {
        Ok(false)
    } else {
        Err(git_failure(
            &format!("inspect local branch `{branch}`"),
            &output,
        ))
    }
}

fn ensure_worktree(
    repository: &Repository,
    branch: &str,
    start_point: Option<&str>,
    destination: &Path,
) -> Result<EnsureOutcome> {
    let entries = list_worktrees(repository)?;
    if let Some(entry) = entries
        .iter()
        .find(|entry| paths_match(&entry.path, destination))
    {
        if !destination.is_dir() {
            return Err(anyhow!(
                "worktree `{}` is registered but missing; repair or prune it with Git before retrying",
                destination.display()
            ));
        }
        let expected = format!("refs/heads/{branch}");
        if entry.branch.as_deref() != Some(expected.as_str()) {
            let actual = entry.branch.as_deref().unwrap_or("detached HEAD");
            return Err(anyhow!(
                "worktree `{}` uses `{actual}`, not `{expected}`",
                destination.display()
            ));
        }
        if start_point.is_some() {
            return Err(anyhow!(
                "--start-point can only be used when creating a missing branch"
            ));
        }
        return Ok(EnsureOutcome::Reused);
    }

    let expected = format!("refs/heads/{branch}");
    if let Some(entry) = entries
        .iter()
        .find(|entry| entry.branch.as_deref() == Some(expected.as_str()))
    {
        return Err(anyhow!(
            "branch `{branch}` is already checked out at `{}`; run Harness there or choose another branch",
            entry.path.display()
        ));
    }

    if destination.exists() {
        return Err(anyhow!(
            "destination `{}` already exists and is not a registered worktree for this repository",
            destination.display()
        ));
    }

    let exists = branch_exists(repository, branch)?;
    if exists && start_point.is_some() {
        return Err(anyhow!(
            "--start-point can only be used when creating a missing branch"
        ));
    }
    let start_point = start_point.unwrap_or("HEAD");
    if !exists {
        validate_start_point(repository, start_point)?;
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create worktree parent `{}`", parent.display()))?;
    }

    let mut command = git(&repository.root);
    command.arg("worktree").arg("add");
    if exists {
        command.arg("--").arg(destination).arg(branch);
    } else {
        command
            .arg("-b")
            .arg(branch)
            .arg("--")
            .arg(destination)
            .arg(start_point);
    }
    let output = command.output().context("run Git to create the worktree")?;
    if !output.status.success() {
        // Translate the most common race into the same actionable diagnostic
        // used by the preflight check.
        if let Ok(entries) = list_worktrees(repository)
            && let Some(entry) = entries
                .iter()
                .find(|entry| entry.branch.as_deref() == Some(expected.as_str()))
        {
            return Err(anyhow!(
                "branch `{branch}` is already checked out at `{}`; run Harness there or choose another branch",
                entry.path.display()
            ));
        }
        let action = if exists {
            format!("check out existing branch `{branch}` in a worktree")
        } else {
            format!("create branch `{branch}` from `{start_point}` in a worktree")
        };
        return Err(git_failure(&action, &output));
    }
    Ok(EnsureOutcome::Created)
}

fn list_worktrees(repository: &Repository) -> Result<Vec<WorktreeEntry>> {
    let output = git(&repository.root)
        .args(["worktree", "list", "--porcelain", "-z"])
        .output()
        .context("run Git to list worktrees")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("unknown option") || stderr.contains("unknown switch") {
            return Err(anyhow!(
                "could not list Git worktrees: Harness requires Git 2.36 or newer for porcelain worktree output"
            ));
        }
        return Err(git_failure("list Git worktrees", &output));
    }

    let mut entries = Vec::new();
    let mut current: Option<WorktreeEntry> = None;
    for field in output.stdout.split(|byte| *byte == 0) {
        if let Some(path) = field.strip_prefix(b"worktree ") {
            if let Some(entry) = current.take() {
                entries.push(entry);
            }
            current = Some(WorktreeEntry {
                path: path_from_git_bytes(path),
                branch: None,
            });
        } else if let Some(branch) = field.strip_prefix(b"branch ")
            && let Some(entry) = current.as_mut()
        {
            entry.branch = Some(String::from_utf8_lossy(branch).into_owned());
        }
    }
    if let Some(entry) = current {
        entries.push(entry);
    }
    Ok(entries)
}

fn validate_start_point(repository: &Repository, start_point: &str) -> Result<()> {
    let revision = format!("{start_point}^{{commit}}");
    let output = git(&repository.root)
        .args(["rev-parse", "--verify", "--end-of-options"])
        .arg(&revision)
        .output()
        .context("run Git to validate the branch start point")?;
    if output.status.success() {
        Ok(())
    } else {
        Err(git_failure(
            &format!("resolve start point `{start_point}` to a commit"),
            &output,
        ))
    }
}

fn default_destination(repository: &Repository, branch: &str) -> Result<PathBuf> {
    default_destination_in(repository, branch, &default_state_root()?)
}

fn default_destination_in(
    repository: &Repository,
    branch: &str,
    state_root: &Path,
) -> Result<PathBuf> {
    let project_name = repository
        .common_dir
        .parent()
        .and_then(Path::file_name)
        .or_else(|| repository.root.file_name())
        .and_then(OsStr::to_str)
        .unwrap_or("repository");
    let project = format!(
        "{}-{:016x}",
        slug(project_name, "repository"),
        stable_hash(path_hash_bytes(&repository.common_dir))
    );
    let branch = format!(
        "{}-{:016x}",
        slug(branch, "branch"),
        stable_hash(branch.as_bytes())
    );
    Ok(state_root.join("worktrees").join(project).join(branch))
}

fn default_state_root() -> Result<PathBuf> {
    resolve_state_root(
        std::env::var_os("HARNESS_STATE_DIR")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from),
        dirs::home_dir(),
    )
}

fn resolve_state_root(state_override: Option<PathBuf>, home: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = state_override {
        return absolute_path(&path);
    }
    home.map(|home| home.join(".harness"))
        .ok_or_else(|| anyhow!("could not determine the Harness state directory"))
}

fn slug(value: &str, fallback: &str) -> String {
    let mut result = String::new();
    let mut separator = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            if separator && !result.is_empty() {
                result.push('-');
            }
            separator = false;
            if result.len() < 48 {
                result.push(character.to_ascii_lowercase());
            }
        } else {
            separator = true;
        }
    }
    let result = result.trim_matches(['.', '-', '_']);
    if result.is_empty() {
        fallback.to_owned()
    } else {
        result.to_owned()
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(unix)]
fn path_hash_bytes(path: &Path) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes()
}

#[cfg(not(unix))]
fn path_hash_bytes(path: &Path) -> &[u8] {
    path.to_str().unwrap_or_default().as_bytes()
}

#[cfg(unix)]
fn path_from_git_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    PathBuf::from(OsStr::from_bytes(bytes))
}

#[cfg(not(unix))]
fn path_from_git_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    std::path::absolute(path).with_context(|| format!("resolve path `{}`", path.display()))
}

/// Resolve aliases in the existing portion of a destination while allowing
/// Git to create its missing final components.
fn normalized_destination(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path)
            .with_context(|| format!("resolve worktree destination `{}`", path.display()));
    }

    let absolute = absolute_path(path)?;
    let mut existing = absolute.as_path();
    let mut missing = Vec::new();
    while !existing.exists() {
        let name = existing.file_name().ok_or_else(|| {
            anyhow!(
                "worktree destination `{}` has no existing ancestor",
                path.display()
            )
        })?;
        missing.push(name.to_os_string());
        existing = existing.parent().ok_or_else(|| {
            anyhow!(
                "worktree destination `{}` has no existing ancestor",
                path.display()
            )
        })?;
    }
    let mut resolved = fs::canonicalize(existing)
        .with_context(|| format!("resolve destination ancestor `{}`", existing.display()))?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn source_has_changes(repository: &Repository) -> Result<bool> {
    let output = git(&repository.root)
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .output()
        .context("run Git to inspect the launch checkout")?;
    if output.status.success() {
        Ok(!output.stdout.is_empty())
    } else {
        Err(git_failure("inspect the launch checkout", &output))
    }
}

fn retention_marker(repository: &Repository, branch: &str, destination: &Path) -> PathBuf {
    let key = stable_hash(path_hash_bytes(destination));
    repository
        .common_dir
        .join("harness")
        .join("kept-worktrees")
        .join(format!(
            "{}-{:016x}-{key:016x}",
            slug(branch, "branch"),
            stable_hash(branch.as_bytes())
        ))
}

fn resolve_keep_policy(
    repository: &Repository,
    branch: &str,
    destination: &Path,
    args: &WorktreeArgs,
) -> Result<bool> {
    let marker = retention_marker(repository, branch, destination);
    if args.ephemeral {
        match fs::remove_file(&marker) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("remove retention marker `{}`", marker.display()));
            }
        }
        return Ok(false);
    }
    if args.keep {
        let parent = marker
            .parent()
            .expect("retention markers always have a parent");
        fs::create_dir_all(parent).with_context(|| {
            format!("create worktree retention directory `{}`", parent.display())
        })?;
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&marker)
            .with_context(|| format!("write retention marker `{}`", marker.display()))?;
        return Ok(true);
    }
    Ok(marker.is_file())
}

fn remove_empty_managed_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir(parent);
    }
}

fn paths_match(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => absolute_path(left).ok() == absolute_path(right).ok(),
    }
}

fn remove_worktree(repository: &Repository, path: &Path) -> CleanupOutcome {
    let status = git(path)
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .output();
    match status {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => {
            return CleanupOutcome::Retained {
                path: path.to_path_buf(),
                reason: "the worktree contains modified or untracked files".into(),
            };
        }
        Ok(output) if !output.status.success() => {
            return CleanupOutcome::Retained {
                path: path.to_path_buf(),
                reason: git_failure("inspect Git worktree before removal", &output).to_string(),
            };
        }
        Err(error) => {
            return CleanupOutcome::Retained {
                path: path.to_path_buf(),
                reason: format!("run Git to inspect the worktree before removal: {error}"),
            };
        }
        Ok(_) => {}
    }

    let output = git(&repository.root)
        .arg("worktree")
        .arg("remove")
        .arg("--")
        .arg(path)
        .output();
    match output {
        Ok(output) if output.status.success() => CleanupOutcome::Removed(path.to_path_buf()),
        Ok(output) => CleanupOutcome::Retained {
            path: path.to_path_buf(),
            reason: git_failure("remove Git worktree", &output).to_string(),
        },
        Err(error) => CleanupOutcome::Retained {
            path: path.to_path_buf(),
            reason: format!("run Git to remove the worktree: {error}"),
        },
    }
}

fn read_lock_owner(file: &mut File) -> Option<u32> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;
    contents
        .lines()
        .find_map(|line| line.strip_prefix("pid=")?.trim().parse().ok())
}

fn rollback_created(repository: &Repository, path: &Path, created: bool) {
    if created {
        let _ = remove_worktree(repository, path);
    }
}

fn git(cwd: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("git");
    #[cfg(test)]
    {
        let null_device = if cfg!(windows) { "NUL" } else { "/dev/null" };
        command
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", null_device)
            .env("GIT_CONFIG_COUNT", "0")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_CONFIG_PARAMETERS")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_COMMON_DIR")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            .args(["-c", "commit.gpgSign=false"]);
    }
    command
        .arg("-C")
        .arg(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn output_text(output: Output, action: &str) -> Result<String> {
    if !output.status.success() {
        return Err(git_failure(action, &output));
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("Git returned non-UTF-8 output while trying to {action}"))
}

fn git_failure(action: &str, output: &Output) -> anyhow::Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let detail = stderr.trim();
    if detail.is_empty() {
        anyhow!("could not {action} (Git exited with {})", output.status)
    } else {
        anyhow!("could not {action}: {detail}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_name_is_readable_and_collision_resistant() {
        let root = tempdir().unwrap();
        let first = Repository {
            root: root.path().join("one/project"),
            common_dir: root.path().join("one/project/.git"),
        };
        let second = Repository {
            root: root.path().join("two/project"),
            common_dir: root.path().join("two/project/.git"),
        };
        let state_root = root.path().join("state");
        let first_path = default_destination_in(&first, "feat/new-feature", &state_root).unwrap();
        let second_path = default_destination_in(&second, "feat/new-feature", &state_root).unwrap();

        assert_ne!(first_path, second_path);
        assert!(
            first_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("feat-new-feature-")
        );
        assert!(first_path.to_string_lossy().contains("/state/worktrees/"));
    }

    #[test]
    fn state_override_controls_the_automatic_worktree_root() {
        let root = tempdir().unwrap();
        assert_eq!(
            resolve_state_root(Some(root.path().join("custom")), None).unwrap(),
            root.path().join("custom")
        );
        assert_eq!(
            resolve_state_root(None, Some(root.path().join("home"))).unwrap(),
            root.path().join("home/.harness")
        );
    }

    #[test]
    fn creates_missing_branch_and_removes_clean_worktree() {
        let (root, repository) = test_repository();
        let destination = root.path().join("worktrees/feature");

        assert_eq!(
            ensure_worktree(&repository, "feat/new-feature", None, &destination).unwrap(),
            EnsureOutcome::Created
        );
        assert_eq!(
            test_git_text(&destination, &["branch", "--show-current"]),
            "feat/new-feature"
        );
        assert!(branch_exists(&repository, "feat/new-feature").unwrap());
        assert!(matches!(
            remove_worktree(&repository, &destination),
            CleanupOutcome::Removed(_)
        ));
        assert!(!destination.exists());
        assert!(branch_exists(&repository, "feat/new-feature").unwrap());
    }

    #[test]
    fn reuses_matching_worktree_and_retains_dirty_files() {
        let (root, repository) = test_repository();
        let destination = root.path().join("worktrees/feature");
        ensure_worktree(&repository, "feature", None, &destination).unwrap();
        assert_eq!(
            ensure_worktree(&repository, "feature", None, &destination).unwrap(),
            EnsureOutcome::Reused
        );

        fs::write(destination.join("unfinished.txt"), "keep me\n").unwrap();
        let outcome = remove_worktree(&repository, &destination);
        assert!(matches!(outcome, CleanupOutcome::Retained { .. }));
        assert_eq!(
            fs::read_to_string(destination.join("unfinished.txt")).unwrap(),
            "keep me\n"
        );

        force_remove(&repository, &destination);
    }

    #[test]
    fn ignored_build_output_does_not_pin_an_ephemeral_worktree() {
        let (root, repository) = test_repository();
        let destination = root.path().join("worktrees/ignored");
        ensure_worktree(&repository, "ignored", None, &destination).unwrap();
        fs::write(destination.join("local.ignored"), "generated output\n").unwrap();

        assert!(matches!(
            remove_worktree(&repository, &destination),
            CleanupOutcome::Removed(_)
        ));
        assert!(!destination.exists());
    }

    #[test]
    fn process_lock_excludes_another_harness_run_for_the_branch() {
        let (_root, repository) = test_repository();
        let first = WorktreeRunLock::acquire(&repository, "feature").unwrap();
        assert!(WorktreeRunLock::acquire(&repository, "feature").is_err());
        assert!(WorktreeRunLock::acquire(&repository, "other").is_ok());
        drop(first);
        assert!(WorktreeRunLock::acquire(&repository, "feature").is_ok());
    }

    #[test]
    fn creates_a_missing_branch_from_an_explicit_start_point() {
        let (root, repository) = test_repository();
        fs::write(repository.root.join("README.md"), "second\n").unwrap();
        run_test_git(&repository.root, &["add", "README.md"]);
        run_test_git(&repository.root, &["commit", "-m", "second"]);
        let destination = root.path().join("worktrees/from-first");

        ensure_worktree(&repository, "from-first", Some("HEAD^"), &destination).unwrap();
        assert_eq!(
            fs::read_to_string(destination.join("README.md")).unwrap(),
            "initial\n"
        );
        force_remove(&repository, &destination);
    }

    #[test]
    fn reports_where_a_branch_is_already_checked_out() {
        let (root, repository) = test_repository();
        let branch = test_git_text(&repository.root, &["branch", "--show-current"]);
        let destination = root.path().join("worktrees/duplicate");

        let error = ensure_worktree(&repository, &branch, None, &destination)
            .unwrap_err()
            .to_string();
        assert!(error.contains("already checked out at"));
        assert!(error.contains(&repository.root.display().to_string()));
    }

    #[test]
    fn keep_policy_is_sticky_until_explicitly_ephemeral() {
        let (root, repository) = test_repository();
        let destination = root.path().join("worktrees/sticky");
        let mut args = worktree_args("sticky", destination.clone());
        args.keep = true;
        assert!(resolve_keep_policy(&repository, "sticky", &destination, &args).unwrap());

        args.keep = false;
        assert!(resolve_keep_policy(&repository, "sticky", &destination, &args).unwrap());

        args.ephemeral = true;
        assert!(!resolve_keep_policy(&repository, "sticky", &destination, &args).unwrap());
    }

    #[test]
    fn prepare_and_finish_restore_the_launch_directory() {
        const CHILD: &str = "HARNESS_WORKTREE_LIFECYCLE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let destination = PathBuf::from(std::env::var_os("HARNESS_TEST_DEST").unwrap());
            let original = fs::canonicalize(std::env::current_dir().unwrap()).unwrap();
            let args = worktree_args("lifecycle", destination.clone());
            let lease = prepare(&args).unwrap();
            assert_eq!(
                fs::canonicalize(std::env::current_dir().unwrap()).unwrap(),
                fs::canonicalize(destination.join("nested")).unwrap()
            );
            assert!(lease.was_created());
            assert!(matches!(
                lease.finish().unwrap(),
                CleanupOutcome::Removed(_)
            ));
            assert_eq!(
                fs::canonicalize(std::env::current_dir().unwrap()).unwrap(),
                original
            );
            return;
        }

        let (root, repository) = test_repository();
        let destination = root.path().join("worktrees/lifecycle");
        let output = ProcessCommand::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "worktree::tests::prepare_and_finish_restore_the_launch_directory",
            ])
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("HARNESS_TEST_DEST", &destination)
            .current_dir(repository.root.join("nested"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child lifecycle test failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!destination.exists());
        assert!(branch_exists(&repository, "lifecycle").unwrap());
    }

    fn worktree_args(branch: &str, destination: PathBuf) -> WorktreeArgs {
        WorktreeArgs {
            branch: branch.into(),
            start_point: None,
            dir: Some(destination),
            keep: false,
            ephemeral: false,
            command: None,
        }
    }

    fn force_remove(repository: &Repository, destination: &Path) {
        let output = git(&repository.root)
            .args(["worktree", "remove", "--force", "--"])
            .arg(destination)
            .output()
            .unwrap();
        assert!(output.status.success());
    }

    fn test_repository() -> (tempfile::TempDir, Repository) {
        let root = tempdir().unwrap();
        let repository_path = root.path().join("project");
        fs::create_dir(&repository_path).unwrap();
        run_test_git(&repository_path, &["init"]);
        run_test_git(
            &repository_path,
            &["config", "user.email", "harness@example.com"],
        );
        run_test_git(&repository_path, &["config", "user.name", "Harness Tests"]);
        fs::write(repository_path.join("README.md"), "initial\n").unwrap();
        fs::write(repository_path.join(".gitignore"), "*.ignored\n").unwrap();
        fs::create_dir(repository_path.join("nested")).unwrap();
        fs::write(repository_path.join("nested/marker.txt"), "nested\n").unwrap();
        run_test_git(
            &repository_path,
            &["add", "README.md", ".gitignore", "nested/marker.txt"],
        );
        run_test_git(&repository_path, &["commit", "-m", "initial"]);
        let repository = Repository::discover(&repository_path).unwrap();
        (root, repository)
    }

    fn run_test_git(cwd: &Path, args: &[&str]) {
        let output = git(cwd).args(args).output().unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn test_git_text(cwd: &Path, args: &[&str]) -> String {
        let output = git(cwd).args(args).output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }
}
