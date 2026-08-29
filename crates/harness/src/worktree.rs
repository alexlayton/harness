use crate::config::WorktreeArgs;
use anyhow::{Context, Result, anyhow};
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Output, Stdio};
use std::time::{Duration, SystemTime};

/// A prepared Git worktree whose lifetime surrounds one interactive run.
pub(crate) struct WorktreeLease {
    original_cwd: PathBuf,
    repository: Repository,
    worktree_path: PathBuf,
    keep: bool,
    created: bool,
    finished: bool,
    _run_lock: WorktreeRunLock,
}

/// The result of safely releasing a worktree after the TUI exits.
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

/// Process lease stored beside (never inside) the worktree. Git does not track
/// whether a process is actively using a worktree, so this prevents one
/// Harness run from removing a clean tree beneath another run.
struct WorktreeRunLock {
    path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnsureOutcome {
    Created,
    Reused,
}

/// Create or reuse the requested worktree and enter its corresponding launch
/// directory. The caller must call [`WorktreeLease::finish`] after the UI has
/// shut down so the process leaves the directory before Git removes it.
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
    let worktree_path = match &args.dir {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => original_cwd.join(path),
        None => default_destination(&repository, &args.branch)?,
    };
    let worktree_path = absolute_path(&worktree_path)?;
    if let Ok(existing_path) = fs::canonicalize(&worktree_path)
        && original_cwd.starts_with(&existing_path)
    {
        return Err(anyhow!(
            "already inside selected worktree `{}`; run `harness` directly there",
            existing_path.display()
        ));
    }
    let run_lock = WorktreeRunLock::acquire(&worktree_path)?;

    let ensured = ensure_worktree(&repository, &args.branch, &worktree_path)?;
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

    Ok(WorktreeLease {
        original_cwd,
        repository,
        worktree_path: resolved_worktree,
        keep: args.keep,
        created,
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

    /// Restore the launch directory, then remove the worktree without force.
    /// Git refusal is a retained outcome rather than an error; modified,
    /// untracked, or ignored files always keep the worktree in place.
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
        Ok(remove_worktree(&self.repository, &self.worktree_path))
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
    fn acquire(destination: &Path) -> Result<Self> {
        let parent = destination.parent().ok_or_else(|| {
            anyhow!(
                "worktree destination `{}` has no parent directory",
                destination.display()
            )
        })?;
        fs::create_dir_all(parent)
            .with_context(|| format!("create worktree parent `{}`", parent.display()))?;
        let name = destination.file_name().ok_or_else(|| {
            anyhow!(
                "worktree destination `{}` has no directory name",
                destination.display()
            )
        })?;
        let path = parent.join(format!(".{}.harness.lock", name.to_string_lossy()));

        for _ in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    if let Err(error) = writeln!(file, "pid={}", std::process::id()) {
                        let _ = fs::remove_file(&path);
                        return Err(error)
                            .with_context(|| format!("write worktree lease `{}`", path.display()));
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_is_stale(&path) {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    let owner = lock_owner(&path)
                        .map(|pid| format!(" by process {pid}"))
                        .unwrap_or_default();
                    return Err(anyhow!(
                        "worktree `{}` is already in use{owner}",
                        destination.display()
                    ));
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("create worktree lease `{}`", path.display()));
                }
            }
        }
        Err(anyhow!(
            "could not acquire worktree lease `{}`",
            path.display()
        ))
    }
}

impl Drop for WorktreeRunLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
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
        return Ok(EnsureOutcome::Reused);
    }

    if destination.exists() {
        return Err(anyhow!(
            "destination `{}` already exists and is not a registered worktree for this repository",
            destination.display()
        ));
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create worktree parent `{}`", parent.display()))?;
    }

    let exists = branch_exists(repository, branch)?;
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
            .arg("HEAD");
    }
    let output = command.output().context("run Git to create the worktree")?;
    if !output.status.success() {
        let action = if exists {
            format!("check out existing branch `{branch}` in a worktree")
        } else {
            format!("create branch `{branch}` from HEAD in a worktree")
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

fn default_destination(repository: &Repository, branch: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine the home directory"))?;
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
    Ok(home
        .join(".harness")
        .join("worktrees")
        .join(project)
        .join(branch))
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

fn paths_match(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => absolute_path(left).ok() == absolute_path(right).ok(),
    }
}

fn remove_worktree(repository: &Repository, path: &Path) -> CleanupOutcome {
    let status = git(path)
        .args([
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored",
        ])
        .output();
    match status {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => {
            return CleanupOutcome::Retained {
                path: path.to_path_buf(),
                reason: "the worktree contains modified, untracked, or ignored files".into(),
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

fn lock_is_stale(path: &Path) -> bool {
    if let Some(pid) = lock_owner(path) {
        return !pid_is_alive(pid);
    }
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age > Duration::from_secs(5 * 60))
}

fn lock_owner(path: &Path) -> Option<u32> {
    fs::read_to_string(path)
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("pid=")?.trim().parse().ok())
}

fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // SAFETY: signal 0 probes process existence without delivering a signal.
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        // Harness currently ships on Unix. Be conservative on other targets:
        // an existing lock must not be stolen while its owner may be alive.
        true
    }
}

fn rollback_created(repository: &Repository, path: &Path, created: bool) {
    if created {
        let _ = remove_worktree(repository, path);
    }
}

fn git(cwd: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("git");
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
        let first_path = default_destination(&first, "feat/new-feature").unwrap();
        let second_path = default_destination(&second, "feat/new-feature").unwrap();

        assert_ne!(first_path, second_path);
        assert!(
            first_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("feat-new-feature-")
        );
        assert!(
            first_path
                .to_string_lossy()
                .contains("/.harness/worktrees/")
        );
    }

    #[test]
    fn creates_missing_branch_and_removes_clean_worktree() {
        let Some((root, repository)) = test_repository() else {
            return;
        };
        let destination = root.path().join("worktrees/feature");

        assert_eq!(
            ensure_worktree(&repository, "feat/new-feature", &destination).unwrap(),
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
        let Some((root, repository)) = test_repository() else {
            return;
        };
        let destination = root.path().join("worktrees/feature");
        ensure_worktree(&repository, "feature", &destination).unwrap();
        assert_eq!(
            ensure_worktree(&repository, "feature", &destination).unwrap(),
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
    fn ignored_files_are_never_deleted_by_automatic_cleanup() {
        let Some((root, repository)) = test_repository() else {
            return;
        };
        let destination = root.path().join("worktrees/ignored");
        ensure_worktree(&repository, "ignored", &destination).unwrap();
        fs::write(destination.join("local.ignored"), "keep me\n").unwrap();

        assert!(matches!(
            remove_worktree(&repository, &destination),
            CleanupOutcome::Retained { .. }
        ));
        assert_eq!(
            fs::read_to_string(destination.join("local.ignored")).unwrap(),
            "keep me\n"
        );

        force_remove(&repository, &destination);
    }

    #[test]
    fn process_lock_excludes_another_harness_run() {
        let root = tempdir().unwrap();
        let destination = root.path().join("feature");
        let first = WorktreeRunLock::acquire(&destination).unwrap();
        assert!(WorktreeRunLock::acquire(&destination).is_err());
        drop(first);
        assert!(WorktreeRunLock::acquire(&destination).is_ok());
    }

    fn force_remove(repository: &Repository, destination: &Path) {
        let output = git(&repository.root)
            .args(["worktree", "remove", "--force", "--"])
            .arg(destination)
            .output()
            .unwrap();
        assert!(output.status.success());
    }

    fn test_repository() -> Option<(tempfile::TempDir, Repository)> {
        if ProcessCommand::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            return None;
        }
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
        run_test_git(&repository_path, &["add", "README.md", ".gitignore"]);
        run_test_git(&repository_path, &["commit", "-m", "initial"]);
        let repository = Repository::discover(&repository_path).unwrap();
        Some((root, repository))
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
