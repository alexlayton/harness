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
//! The budget and ordering rules follow the skills/AGENTS.md design doc:
//! global first (lowest priority), then repo root → cwd with the nearest
//! (cwd) file loaded last so it overrides parents.

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
    // Walk from cwd up to the git repo root (care: `walk` is cwd-first;
    // reversed below so parents precede deeper/cwd dirs, matching the
    // design doc's "forward from root, nearest last").
    let mut walk = Vec::new();
    let mut dir = cwd.to_path_buf();
    loop {
        walk.push(dir.clone());
        if dir.join(".git").exists() {
            break;
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent.to_path_buf(),
            _ => break,
        }
    }
    walk.reverse();
    dirs.extend(walk);

    // Pick the first existing candidate per directory, dedupe by canonical
    // path (symlinked roots on macOS can surface the same file twice), and
    // read it. Unreadable files are skipped silently; a missing global path
    // is the normal case and never an error.
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for dir in dirs {
        let Some(path) = first_candidate(&dir) else {
            continue;
        };
        let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !seen.insert(canonical) {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        found.push((path, raw));
    }

    // Spend the budget from highest priority (cwd/nearest) to lowest, then
    // restore semantic rendering order (global/root first, nearest last).
    // Thus a large global file can never evict local instructions.
    let mut result = Vec::new();
    let mut remaining = config.max_total_bytes;
    for (path, raw) in found.into_iter().rev() {
        if remaining == 0 {
            break;
        }
        let cap = remaining.min(config.max_file_bytes);
        let original_len = raw.len();
        let (content, truncated) = cap_content(&raw, cap);
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

fn cap_content(content: &str, cap: usize) -> (String, bool) {
    if content.len() <= cap {
        return (content.to_owned(), false);
    }
    let notice = format!("\n[…] (truncated, {} bytes total)", content.len());
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
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".harness"))
}

/// Display path with `$HOME` abbreviated to `~` where applicable, so the
/// header shows `~/.harness/AGENTS.md` rather than an absolute home path.
pub fn display_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let Some(home) = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(|home| PathBuf::from(home).to_string_lossy().replace('\\', "/"))
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
}
