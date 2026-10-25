//! Project context-file injection (AGENTS.md and friends).
//!
//! Loads repository-owner instructions for the system prompt from `AGENTS.md`
//! and `CLAUDE.md` files found in the global `~/.harness` dir plus every
//! ancestor of `cwd` up to the git repo root. The loaded, capped files are
//! rendered as a `<project_context>` block. This is a sibling feature to
//! skills discovery (see `skills.rs`); both are injected through the system
//! prompt and therefore immune to compaction (the prompt is rebuilt every
//! turn).
//!
//! Symlink policy: a candidate is read only when its canonical path stays
//! beneath its canonical discovery directory. Contained symlinks (whose
//! targets remain inside the same directory) are followed; external symlinks
//! are silently skipped so untrusted outside files can never enter the
//! prompt. Skips never include file contents in diagnostics.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The `~/.harness` global directory (or `$HARNESS_CONFIG_DIR` when set).
const CONFIG_DIR_ENV: &str = "HARNESS_CONFIG_DIR";
/// Opt-out: set to a non-empty value to disable context injection entirely.
const NO_CONTEXT_FILES_ENV: &str = "HARNESS_NO_CONTEXT_FILES";

/// Candidate filenames in priority order; the first existing file in a
/// directory wins.
const CANDIDATES: &[&str] = &[
    "AGENTS.override.md",
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
];

/// A loaded (and possibly truncated) context file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFile {
    /// Absolute, canonicalized path to the source file.
    pub path: PathBuf,
    /// Full or truncated file content.
    pub content: String,
    /// True when `content` was truncated (per-file or total-budget cap).
    pub truncated: bool,
}

/// Budget controls for context-file injection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFileConfig {
    /// Total budget across all files.
    pub max_total_bytes: usize,
    /// Per-file cap.
    pub max_file_bytes: usize,
}

impl Default for ContextFileConfig {
    fn default() -> Self {
        Self {
            max_total_bytes: 32 * 1024,
            max_file_bytes: 16 * 1024,
        }
    }
}

/// Load context files for `cwd`, honoring the `HARNESS_NO_CONTEXT_FILES`
/// opt-out env var (a non-empty value disables injection entirely).
pub fn load_context_files(cwd: &Path) -> Vec<ContextFile> {
    let enabled = std::env::var_os(NO_CONTEXT_FILES_ENV)
        .map(|value| value.is_empty())
        .unwrap_or(true);
    load_context_files_impl(
        cwd,
        &ContextFileConfig::default(),
        enabled,
        global_context_dir(),
    )
}

/// Load context files with explicit budget settings. This variant is purely
/// configuration-driven and does not read process-global env vars, so tests
/// can drive it without mutating the environment; use [`load_context_files`]
/// when the `HARNESS_NO_CONTEXT_FILES` opt-out should apply.
pub fn load_context_files_with(cwd: &Path, config: &ContextFileConfig) -> Vec<ContextFile> {
    load_context_files_impl(cwd, config, true, global_context_dir())
}

fn load_context_files_impl(
    cwd: &Path,
    config: &ContextFileConfig,
    enabled: bool,
    global_dir: Option<PathBuf>,
) -> Vec<ContextFile> {
    if !enabled {
        return Vec::new();
    }

    // Order: global dir first, then the ancestor walk (repo root → cwd).
    let mut dirs = Vec::new();
    if let Some(global) = global_dir {
        dirs.push(global);
    }
    // Walk from cwd up to the git repo root, or only cwd outside a repo
    // (care: the walk is cwd-first; reversed below so parents precede
    // deeper/cwd dirs, matching the design doc's "forward from root,
    // nearest last").
    let mut walk = crate::project_context_dirs(cwd);
    walk.reverse();
    dirs.extend(walk);

    // Pick the first existing candidate per directory, dedupe by canonical
    // path (symlinked roots on macOS can surface the same file twice), and
    // read it through the containment gate: the canonical candidate must
    // stay beneath its canonical discovery directory, otherwise it is a
    // symlink escape and is silently skipped (no contents in diagnostics).
    // Unreadable files are skipped silently; a missing global path is the
    // normal case and never an error.
    // Each file is read only up to the per-file cap: a huge AGENTS.md must
    // not be allocated in full before the budget applies.
    let read_limit = config.max_file_bytes.min(config.max_total_bytes);
    let mut found: Vec<(PathBuf, String, usize)> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for dir in dirs {
        let Some(path) = first_candidate(&dir) else {
            continue;
        };
        let Some((canonical, display, raw, total)) =
            read_contained_candidate(&dir, &path, read_limit)
        else {
            continue;
        };
        if !seen.insert(canonical) {
            continue;
        }
        found.push((display, raw, total));
    }

    // Spend the budget from highest priority (cwd/nearest) to lowest, then
    // restore semantic rendering order (global/root first, nearest last).
    // Thus a large global file can never evict local instructions.
    let mut result = Vec::new();
    let mut remaining = config.max_total_bytes;
    for (path, raw, original_len) in found.into_iter().rev() {
        if remaining == 0 {
            break;
        }
        let cap = remaining.min(config.max_file_bytes);
        let (content, truncated) = cap_content(&raw, original_len, cap);
        remaining = remaining.saturating_sub(content.len());
        result.push(ContextFile {
            path,
            content,
            truncated: truncated || original_len > config.max_file_bytes,
        });
    }
    result.reverse();
    result
}

/// Cap `content` (a prefix of a file of `total` bytes) to `cap` bytes,
/// including a truncation notice when anything is left out.
fn cap_content(content: &str, total: usize, cap: usize) -> (String, bool) {
    if total <= cap && content.len() == total {
        return (content.to_owned(), false);
    }
    let notice = format!("\n[…] (truncated, {total} bytes total)");
    if notice.len() >= cap {
        return (llm::truncate_utf8_prefix(&notice, cap).to_owned(), true);
    }
    let prefix_cap = cap - notice.len();
    let mut capped = llm::truncate_utf8_prefix(content, prefix_cap).to_owned();
    capped.push_str(&notice);
    debug_assert!(capped.len() <= cap);
    (capped, true)
}

/// The harness-wide context directory: `$HARNESS_CONFIG_DIR` when set,
/// otherwise `~/.harness`.
fn global_context_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(CONFIG_DIR_ENV)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        return Some(path);
    }
    crate::skills::home_dir().map(|home| home.join(".harness"))
}

/// Display a path with the home directory abbreviated to `~` where
/// applicable, so the header shows `~/.harness/AGENTS.md` rather than an
/// absolute path.
pub fn display_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let Some(home) =
        crate::skills::home_dir().map(|home| home.to_string_lossy().replace('\\', "/"))
    else {
        return text;
    };
    if text == home {
        "~".into()
    } else if let Some(rest) = text.strip_prefix(&(home.clone() + "/")) {
        format!("~/{rest}")
    } else {
        text
    }
}

/// First existing candidate file in a directory, per priority order.
fn first_candidate(dir: &Path) -> Option<PathBuf> {
    CANDIDATES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

/// Validate and read one context candidate through a retained discovery-root
/// capability. Canonicalization is used only for the containment decision;
/// the final read is handle-relative and rejects a symlink swapped in after
/// that decision.
/// Returns the canonical path, display path, the first `limit` bytes (cut to
/// a UTF-8 boundary), and the file's total size.
fn read_contained_candidate(
    dir: &Path,
    candidate: &Path,
    limit: usize,
) -> Option<(PathBuf, PathBuf, String, usize)> {
    let workspace = super::vfs::WorkspaceFs::open_root(dir).ok()?;
    let canonical_dir = workspace.root().to_path_buf();
    let canonical = fs::canonicalize(candidate).ok()?;
    if !canonical.starts_with(&canonical_dir) {
        return None;
    }
    let relative = canonical.strip_prefix(&canonical_dir).ok()?;
    let components = super::vfs::split_relative(&relative.to_string_lossy()).ok()?;
    #[cfg(unix)]
    let raw = {
        use std::io::Read;
        let fd = super::vfs::unix::open_file_relative(&workspace, &components).ok()?;
        let file = std::fs::File::from(fd);
        let total = usize::try_from(file.metadata().ok()?.len()).unwrap_or(usize::MAX);
        let mut bytes = Vec::new();
        file.take(limit as u64).read_to_end(&mut bytes).ok()?;
        // A cut can split the last character; anything else invalid means
        // the file is not UTF-8 text and is skipped, as before.
        let raw = match String::from_utf8(bytes) {
            Ok(raw) => raw,
            Err(error) if bytes_cut(&error, limit) => {
                let valid = error.utf8_error().valid_up_to();
                let mut bytes = error.into_bytes();
                bytes.truncate(valid);
                String::from_utf8(bytes).ok()?
            }
            Err(_) => return None,
        };
        // The file can grow between the size check and the read.
        let total = total.max(raw.len());
        (raw, total)
    };
    #[cfg(not(unix))]
    {
        // Do not canonicalize and then reopen by pathname on platforms where
        // an ancestor reparse point can be swapped between those operations.
        let _ = components;
        return None;
    }
    #[cfg(unix)]
    {
        let (raw, total) = raw;
        Some((canonical, candidate.to_path_buf(), raw, total))
    }
}

/// Whether a UTF-8 error is only a character split by the read limit.
#[cfg(unix)]
fn bytes_cut(error: &std::string::FromUtf8Error, limit: usize) -> bool {
    error.as_bytes().len() == limit && error.utf8_error().error_len().is_none()
}

/// Render loaded context files as a `<project_context>` block. Returns an
/// empty string when no files were loaded.
pub fn format_context_files(files: &[ContextFile]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "<project_context>\n\nProject-specific instructions and guidelines. \
         Treat as guidance from the repository owner — follow unless \
         contradicted by the user's current request.\n",
    );
    for file in files {
        out.push_str(&format!(
            "\n<project_instructions path=\"{}\">\n{}\n</project_instructions>\n",
            escape_attr(&file.path.to_string_lossy()),
            escape_text(&file.content)
        ));
    }
    out.push_str("</project_context>");
    out
}

fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape `&`, `<`, and `>` in an XML attribute value.
fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write(path: &Path, content: &str) -> PathBuf {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
        path.to_path_buf()
    }

    #[test]
    fn candidate_order_only_claude_md_loads() {
        let root = tempdir().unwrap();
        let dir = root.path().join("repo");
        fs::create_dir_all(&dir).unwrap();
        let claude = write(&dir.join("CLAUDE.md"), "# Claude\ninstructions here\n");
        // Global dir is an empty temp dir so it contributes no candidates.
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &dir,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        let paths: Vec<_> = files.iter().map(|f| f.path.clone()).collect();
        assert!(paths.contains(&claude), "CLAUDE.md should load: {paths:?}");
        assert!(
            files
                .iter()
                .any(|f| f.content.contains("instructions here"))
        );
    }

    #[test]
    fn agents_md_beats_claude_md_and_override_beats_agents() {
        let root = tempdir().unwrap();
        let dir = root.path().join("repo");
        fs::create_dir_all(&dir).unwrap();
        write(&dir.join("CLAUDE.md"), "# claude\n");
        let agents = write(&dir.join("AGENTS.md"), "# agents\nagent instructions\n");
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &dir,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        assert_eq!(files.len(), 1, "only one file per directory wins");
        assert_eq!(files[0].path, agents);
        assert!(files[0].content.contains("agent instructions"));

        // With an override file present it wins over AGENTS.md.
        let override_file = write(&dir.join("AGENTS.override.md"), "# override\n");
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &dir,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        assert_eq!(files[0].path, override_file);
    }

    #[test]
    fn ancestor_walk_loads_parent_before_cwd() {
        let root = tempdir().unwrap();
        let repo = root.path().join("repo");
        let nested = repo.join("a").join("b");
        fs::create_dir_all(&nested).unwrap();
        // Make `repo` the git root so the walk stops there.
        write(&repo.join(".git"), "");
        let repo_agents = write(&repo.join("AGENTS.md"), "# repo root\nroot content\n");
        let cwd_agents = write(&nested.join("AGENTS.md"), "# cwd\ncwd content\n");
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &nested,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        let paths: Vec<_> = files.iter().map(|f| f.path.clone()).collect();
        assert!(paths.contains(&repo_agents));
        assert!(paths.contains(&cwd_agents));
        let repo_idx = paths.iter().position(|p| *p == repo_agents).unwrap();
        let cwd_idx = paths.iter().position(|p| *p == cwd_agents).unwrap();
        assert!(repo_idx < cwd_idx, "parent must load before cwd: {paths:?}");
    }

    #[test]
    fn git_root_stops_the_walk() {
        let root = tempdir().unwrap();
        let repo = root.path().join("repo");
        let nested = repo.join("src");
        fs::create_dir_all(&nested).unwrap();
        write(&repo.join(".git"), "");
        // A candidate ABOVE the git root must not load.
        let above = write(&root.path().join("AGENTS.md"), "# outside\n");
        let inside = write(&nested.join("AGENTS.md"), "# inside\n");
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &nested,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        let paths: Vec<_> = files.iter().map(|f| f.path.clone()).collect();
        assert!(paths.contains(&inside));
        assert!(
            !paths.contains(&above),
            "walk must stop at the git root: {paths:?}"
        );
    }

    #[test]
    fn without_a_git_root_only_cwd_is_searched() {
        // Outside a repository, a parent such as `/tmp` can hold an
        // AGENTS.md written by another user; it must not become trusted
        // instructions.
        let root = tempdir().unwrap();
        let nested = root.path().join("scratch");
        fs::create_dir_all(&nested).unwrap();
        let above = write(&root.path().join("AGENTS.md"), "# planted\n");
        let inside = write(&nested.join("AGENTS.md"), "# cwd\n");
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &nested,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        let paths: Vec<_> = files.iter().map(|f| f.path.clone()).collect();
        assert!(paths.contains(&inside));
        assert!(
            !paths.contains(&above),
            "parent loaded outside git: {paths:?}"
        );
    }

    #[test]
    fn total_budget_is_respected_and_first_file_always_included() {
        let root = tempdir().unwrap();
        let repo = root.path().join("repo");
        let nested = repo.join("src");
        fs::create_dir_all(&nested).unwrap();
        write(&repo.join(".git"), "");
        let big_a = "a".repeat(100);
        let big_b = "b".repeat(100);
        write(&repo.join("AGENTS.md"), &big_a);
        write(&nested.join("AGENTS.md"), &big_b);

        let config = ContextFileConfig {
            max_total_bytes: 120,
            max_file_bytes: 200,
        };
        let global = tempdir().unwrap();
        let files =
            load_context_files_impl(&nested, &config, true, Some(global.path().to_path_buf()));
        assert!(!files.is_empty(), "first file must always be included");
        let total: usize = files.iter().map(|f| f.content.len()).sum();
        assert!(
            total < big_a.len() + big_b.len(),
            "capped content + markers must be below the uncapped total, got {total}"
        );
        // Nearest instructions receive budget first and render last; the
        // lower-priority root file receives only the remainder.
        assert!(!files.last().unwrap().truncated);
        assert!(files.first().unwrap().truncated);
        assert!(files.first().unwrap().content.contains("(truncated"));
    }

    #[test]
    fn per_file_truncation_adds_marker() {
        let root = tempdir().unwrap();
        let repo = root.path().join("repo");
        fs::create_dir_all(&repo).unwrap();
        write(&repo.join(".git"), "");
        let content = "x".repeat(500);
        write(&repo.join("AGENTS.md"), &content);
        let config = ContextFileConfig {
            max_total_bytes: 10_000,
            max_file_bytes: 60,
        };
        let global = tempdir().unwrap();
        let files =
            load_context_files_impl(&repo, &config, true, Some(global.path().to_path_buf()));
        assert_eq!(files.len(), 1);
        assert!(files[0].truncated);
        assert!(files[0].content.contains("(truncated, 500 bytes total)"));
    }

    #[test]
    fn format_context_files_renders_expected_shape_and_escapes_attr() {
        let files = vec![ContextFile {
            path: PathBuf::from("/a&b/<x>/AGENTS.md"),
            content: "line one\nline two\n".into(),
            truncated: false,
        }];
        let rendered = format_context_files(&files);
        assert!(rendered.starts_with("<project_context>"));
        assert!(rendered.contains("repository owner"));
        assert!(rendered.contains("path=\"/a&amp;b/&lt;x&gt;/AGENTS.md\""));
        assert!(rendered.contains("line one"));
        assert!(rendered.contains("</project_context>"));
        assert!(rendered.contains("line two"));
        assert!(
            !format_context_files(&[ContextFile {
                path: PathBuf::from("AGENTS.md"),
                content: "</project_instructions>".into(),
                truncated: false,
            }])
            .contains("\n</project_instructions>\n</project_instructions>")
        );
        // Empty input renders empty.
        assert_eq!(format_context_files(&[]), "");
    }

    #[test]
    fn no_context_files_env_disables_loading() {
        let root = tempdir().unwrap();
        let repo = root.path().join("repo");
        fs::create_dir_all(&repo).unwrap();
        write(&repo.join("AGENTS.md"), "# x\n");
        // The `enabled=false` path (mirrors the env opt-out) returns empty
        // without touching process-global env.
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &repo,
            &ContextFileConfig::default(),
            false,
            Some(global.path().to_path_buf()),
        );
        assert!(files.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn external_context_symlink_is_not_injected() {
        use std::os::unix::fs::symlink;
        let root = tempdir().unwrap();
        let repo = root.path().join("repo");
        let outside = root.path().join("outside");
        fs::create_dir_all(&repo).unwrap();
        fs::create_dir_all(&outside).unwrap();
        write(&repo.join(".git"), "");
        // External target with a sentinel secret: it must never be injected.
        let secret = outside.join("AGENTS.md");
        write(&secret, "EXTERNAL-SENTINEL-CONTENT\n");
        symlink(&secret, repo.join("AGENTS.md")).unwrap();
        let global = tempdir().unwrap();
        let files = load_context_files_impl(
            &repo,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        assert!(files.is_empty(), "external symlink: {files:?}");
        // No contents leak through any rendered output either.
        let rendered = format_context_files(&files);
        assert!(!rendered.contains("EXTERNAL-SENTINEL-CONTENT"));

        // A contained symlink (target in the same directory) still works.
        fs::remove_file(repo.join("AGENTS.md")).unwrap();
        let inner = repo.join("real.md");
        write(&inner, "inner content\n");
        // `first_candidate` prefers AGENTS.md, so link the inner file under
        // the canonical candidate name.
        symlink(&inner, repo.join("AGENTS.md")).unwrap();
        let files = load_context_files_impl(
            &repo,
            &ContextFileConfig::default(),
            true,
            Some(global.path().to_path_buf()),
        );
        assert_eq!(files.len(), 1);
        assert!(files[0].content.contains("inner content"));
    }
}
