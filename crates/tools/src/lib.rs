mod bash;
mod edit;
pub mod file_mutation;
mod find;
mod grep;
mod read;
mod registry;
pub mod skills;
mod write;

pub use bash::{BashTool, truncate_command_output};
pub use edit::EditTool;
pub use find::{FileSearchIndex, FindConfig, FindTool};
pub use grep::GrepTool;
pub use read::ReadTool;
pub use registry::{ToolPromptContext, ToolPromptEntry, ToolRegistry, ToolRegistryError};
pub use skills::{
    Skill, SkillCatalog, SkillDiagnostic, SkillSeverity, discover, expand_tilde,
    format_skills_prompt, load_skills_from_dir, parse_frontmatter,
};
pub use write::WriteTool;

use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::Value;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio_util::sync::CancellationToken;

/// Short, model-facing prompt information for a tool.  The JSON schema and
/// full description remain in [`ToolSpec::definition`]; this metadata is
/// intentionally small enough to repeat in the system prompt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolPrompt {
    pub snippet: Option<String>,
    pub guidelines: Vec<String>,
}

impl ToolPrompt {
    pub fn new<I, S>(snippet: impl Into<String>, guidelines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            snippet: Some(snippet.into()),
            guidelines: guidelines.into_iter().map(Into::into).collect(),
        }
    }

    pub fn without_snippet<I, S>(guidelines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            snippet: None,
            guidelines: guidelines.into_iter().map(Into::into).collect(),
        }
    }
}

/// The structured provider definition and the prompt metadata belonging to a
/// tool.  Keeping these together makes it difficult for the system prompt and
/// the provider request to drift apart.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub definition: ToolDefinition,
    pub prompt: ToolPrompt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub summary: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    /// Return the structured definition and optional prompt metadata.  The
    /// default bridges older integrations that implemented only
    /// [`Tool::definition`]; new tools should override this method.
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: self.definition(),
            prompt: ToolPrompt::default(),
        }
    }

    /// Compatibility accessor for callers that only need the structured
    /// definition.  Registry code uses [`Tool::spec`] so it can also build the
    /// dynamic prompt metadata.  Implementations may override either this
    /// method or `spec`.
    fn definition(&self) -> ToolDefinition {
        self.spec().definition
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput;
}

/// Runtime settings shared by all built-in tools in a registry.
#[derive(Clone, Debug)]
pub struct ToolConfig {
    pub cwd: PathBuf,
    pub rtk: bool,
}

impl ToolConfig {
    pub fn new(cwd: impl Into<PathBuf>, rtk: bool) -> Self {
        Self {
            cwd: cwd.into(),
            rtk,
        }
    }

    pub fn from_current_dir(rtk: bool) -> Self {
        Self::new(
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            rtk,
        )
    }
}

impl From<bool> for ToolConfig {
    fn from(rtk: bool) -> Self {
        Self::from_current_dir(rtk)
    }
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self::from_current_dir(false)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolInitError {
    #[error("could not resolve workspace root {path}: {source}")]
    Workspace {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not initialize find index: {0}")]
    Find(String),
    #[error(transparent)]
    Registry(#[from] ToolRegistryError),
}

/// Construct the built-in tools for one workspace.  The index is created
/// once here and shared by every `find` call; it is never initialized per
/// request.
///
/// Skills are discovered from the project walk + global roots; the
/// discovered skill paths are handed to `ReadTool` (pi model: the model
/// reads a skill's `SKILL.md` via `read` on the absolute `<location>`), and
/// the catalog is stored on the registry for the prompt builder.
///
/// The generic argument preserves the old `default_registry(false)` spelling
/// while also accepting the workspace-aware [`ToolConfig`] used by the
/// application.
pub fn default_registry(config: impl Into<ToolConfig>) -> Result<ToolRegistry, ToolInitError> {
    let config = config.into();
    let workspace_root =
        std::fs::canonicalize(&config.cwd).map_err(|source| ToolInitError::Workspace {
            path: config.cwd.clone(),
            source,
        })?;

    let skills = discover_skills_for_config(&workspace_root);
    let read_paths = skills.read_paths.clone();

    let index = Arc::new(
        FileSearchIndex::new(&workspace_root)
            .map_err(|error| ToolInitError::Find(error.to_string()))?,
    );
    let mut registry = ToolRegistry::try_new_with_workspace(
        vec![
            Box::new(
                ReadTool::with_workspace_root(&workspace_root)
                    .with_allowed_paths(read_paths.clone()),
            ),
            Box::new(EditTool::with_workspace_root(&workspace_root)),
            Box::new(WriteTool::with_workspace_root(&workspace_root)),
            Box::new(BashTool::with_rtk_and_workspace_root(
                config.rtk,
                &workspace_root,
            )),
            Box::new(FindTool::new(index.clone())),
            Box::new(GrepTool::new(index.clone())),
        ],
        workspace_root,
    )
    .map_err(ToolInitError::from)?;
    registry.set_skills(skills);
    Ok(registry)
}

pub(crate) fn discover_skills_for_config(workspace_root: &Path) -> SkillCatalog {
    // Project roots: cwd up to git repo root (or filesystem root).
    let mut roots: Vec<(PathBuf, String)> = Vec::new();
    let mut dir = workspace_root.to_path_buf();
    loop {
        roots.push((dir.join(".harness/skills"), "pi".into()));
        roots.push((dir.join(".agents/skills"), "agents".into()));
        // Stop at the git repo root.
        if dir.join(".git").exists() {
            break;
        }
        let parent = dir.parent().map(Path::to_path_buf);
        match parent {
            Some(parent) if parent != dir => dir = parent,
            _ => break,
        }
    }
    // Global: ~/.harness/skills (or $HARNESS_SKILLS_DIR).
    let global = std::env::var_os("HARNESS_SKILLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".harness/skills"))
                .unwrap_or_default()
        });
    if !global.as_os_str().is_empty() {
        roots.push((global, "pi".into()));
    }
    let agents_global = std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(".agents/skills"))
        .unwrap_or_default();
    if !agents_global.as_os_str().is_empty() {
        roots.push((agents_global, "agents".into()));
    }
    discover(&roots)
}

impl Default for ToolRegistry {
    fn default() -> Self {
        default_registry(ToolConfig::from_current_dir(false))
            .expect("the current directory should be a valid tool workspace")
    }
}

pub(crate) fn normalize_workspace_root(root: impl Into<PathBuf>) -> PathBuf {
    let root = root.into();
    std::fs::canonicalize(&root).unwrap_or(root)
}

/// Resolve a tool path.  Workspace-aware tools use this helper to keep paths
/// inside the same root as `bash` and `find`.  The `None` mode is retained for
/// the small compatibility constructors used by library callers and tests;
/// those constructors preserve the historical process-cwd/absolute-path
/// behavior.
pub(crate) async fn resolve_workspace_path(
    value: &str,
    workspace_root: Option<&Path>,
    strip_at_prefix: bool,
) -> Result<PathBuf, String> {
    let value = if strip_at_prefix {
        value.strip_prefix('@').unwrap_or(value)
    } else {
        value
    };
    if value.is_empty() {
        return Err("path must not be empty".into());
    }

    let path = PathBuf::from(value);
    let Some(root) = workspace_root else {
        return Ok(if path.is_absolute() {
            path
        } else {
            std::env::current_dir()
                .map_err(|error| format!("cannot determine working directory: {error}"))?
                .join(path)
        });
    };

    let root = lexical_normalize(root);
    // Canonicalize the workspace root once up front so every containment
    // check below compares like-for-like.  Production callers pass an already
    // canonicalized root; doing it here also keeps a symlinked root safe for
    // embedded and test callers.
    let canonical_root = fs::canonicalize(&root)
        .await
        .map_err(|error| format!("cannot resolve workspace root {}: {error}", root.display()))?;
    let candidate_path = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    let candidate = lexical_normalize(&candidate_path);
    if !candidate.starts_with(&root) {
        return Err(format!("path is outside workspace root {}", root.display()));
    }

    // Lexical containment handles `..`; canonicalizing also prevents a
    // symlink inside the workspace from escaping it.  The fast path covers
    // the common case where the path already exists: one canonicalize, one
    // containment check.
    match fs::canonicalize(&candidate).await {
        Ok(canonical) => {
            if !canonical.starts_with(&canonical_root) {
                return Err(format!(
                    "path resolves outside workspace root {}",
                    root.display()
                ));
            }
            return Ok(candidate);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("cannot resolve path {value}: {error}")),
    }

    // The candidate does not exist yet (a new file).  Its parent directory
    // almost always exists (writing a new file into an existing tree), so try
    // canonicalizing just the parent: one canonicalize, no per-level probing.
    if let Some(parent) = candidate.parent() {
        match fs::canonicalize(parent).await {
            Ok(canonical_parent) => {
                if !canonical_parent.starts_with(&canonical_root) {
                    return Err(format!(
                        "path resolves outside workspace root {}",
                        root.display()
                    ));
                }
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("cannot resolve path {value}: {error}")),
        }
    }

    // Neither the candidate nor its parent exists (e.g. `write` creating a
    // deeply nested directory tree).  Locate the deepest existing ancestor
    // with cheap metadata probes — one stat per level, no symlink resolution
    // — then canonicalize only that single ancestor for the containment
    // check.  This path is rare; the cases above are O(1).
    let mut existing = candidate.clone();
    let canonical = loop {
        match fs::metadata(&existing).await {
            Ok(_) => {
                break fs::canonicalize(&existing)
                    .await
                    .map_err(|error| format!("cannot resolve path {value}: {error}"))?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !existing.pop() {
                    return Err(format!("cannot resolve path {value}: {error}"));
                }
            }
            Err(error) => return Err(format!("cannot resolve path {value}: {error}")),
        }
    };
    if !canonical.starts_with(&canonical_root) {
        return Err(format!(
            "path resolves outside workspace root {}",
            root.display()
        ));
    }
    Ok(candidate)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // Do not pop a filesystem root or a Windows prefix.
                let _ = normalized.pop();
            }
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    normalized
}

/// Build a concise preview used before a tool has completed.
pub fn call_summary(name: &str, args: &Value) -> String {
    match name {
        "read" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("read {path}"))
            .unwrap_or_else(|| "read".into()),
        "edit" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("edit {path}"))
            .unwrap_or_else(|| "edit".into()),
        "write" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("write {path}"))
            .unwrap_or_else(|| "write".into()),
        "bash" => args
            .get("command")
            .and_then(Value::as_str)
            .map(|command| format!("bash: {}", first_line(command)))
            .unwrap_or_else(|| "bash".into()),
        "find" => {
            let query = args.get("query").and_then(Value::as_str);
            let path = args.get("path").and_then(Value::as_str);
            match (query, path) {
                (Some(query), Some(path)) => format!("find {query} in {path}"),
                (Some(query), None) => format!("find {query}"),
                _ => "find".into(),
            }
        }
        "grep" => {
            let pattern = args.get("pattern").and_then(Value::as_str);
            let path = args.get("path").and_then(Value::as_str);
            match (pattern, path) {
                (Some(pattern), Some(path)) => format!("grep {pattern} in {path}"),
                (Some(pattern), None) => format!("grep {pattern}"),
                _ => "grep".into(),
            }
        }
        _ => name.to_owned(),
    }
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value)
}

/// Expanded per-tool recap shown in the transcript. Unlike [`call_summary`]
/// this includes the secondary arguments a user may want to see (line ranges,
/// limits, timeouts) while still staying readable and free of raw JSON. Falls
/// back to [`call_summary`] when there are no interesting extras.
pub fn call_recap(name: &str, args: &Value) -> String {
    let base = call_summary(name, args);
    let extras = match name {
        "read" => {
            let offset = args.get("offset").and_then(Value::as_u64);
            let limit = args.get("limit").and_then(Value::as_u64);
            match (offset, limit) {
                (Some(offset), Some(limit)) if offset != 1 => {
                    // offset is 1-indexed and limit counts lines, so the last
                    // line covered is `offset + limit - 1`.
                    Some(format!("lines {offset}–{}", offset + limit - 1))
                }
                (Some(_), Some(limit)) => Some(format!("lines 1–{}", limit)),
                _ => None,
            }
        }
        "bash" => {
            let dir = args.get("dir").and_then(Value::as_str);
            let timeout = args.get("timeout").and_then(Value::as_u64);
            match (dir, timeout) {
                (Some(dir), Some(timeout)) => Some(format!("in {dir} · timeout {timeout}s")),
                (Some(dir), None) => Some(format!("in {dir}")),
                (None, Some(timeout)) => Some(format!("timeout {timeout}s")),
                (None, None) => None,
            }
        }
        "find" => args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|limit| format!("limit {limit}")),
        "grep" => {
            let limit = args.get("limit").and_then(Value::as_u64);
            let context = args.get("context").and_then(Value::as_u64);
            match (limit, context) {
                (Some(limit), Some(context)) if context > 0 => {
                    Some(format!("limit {limit} · context {context}"))
                }
                (Some(limit), _) => Some(format!("limit {limit}")),
                (None, Some(context)) if context > 0 => Some(format!("context {context}")),
                _ => None,
            }
        }
        "write" | "edit" => {
            // The path in the summary already carries the interesting content.
            None
        }
        _ => {
            // Custom tools have no curated recap; keep their arguments visible
            // instead of hiding them behind the bare tool name.
            let compact = serde_json::to_string(args).unwrap_or_default();
            (compact != "null" && compact != "{}").then(|| format!("args {compact}"))
        }
    };
    match extras {
        Some(extras) => format!("{base} ({extras})"),
        None => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_path_normalization_does_not_escape_root() {
        let root = PathBuf::from("/tmp/workspace");
        assert_eq!(
            lexical_normalize(&root.join("src/../main.rs")),
            PathBuf::from("/tmp/workspace/main.rs")
        );
        assert!(!lexical_normalize(&root.join("../outside")).starts_with(&root));
    }

    #[test]
    fn find_is_in_call_summaries() {
        assert_eq!(
            call_summary("find", &serde_json::json!({"query": "ViewModel"})),
            "find ViewModel"
        );
        assert_eq!(
            call_summary(
                "find",
                &serde_json::json!({"query": "ViewModel", "path": "src"})
            ),
            "find ViewModel in src"
        );
    }

    #[test]
    fn recap_includes_secondary_arguments_human_readably() {
        assert_eq!(
            call_recap("read", &serde_json::json!({"path": "a.rs"})),
            "read a.rs"
        );
        assert_eq!(
            call_recap(
                "read",
                &serde_json::json!({"path": "a.rs", "offset": 2, "limit": 30})
            ),
            "read a.rs (lines 2–31)"
        );
        assert_eq!(
            call_recap(
                "bash",
                &serde_json::json!({"command": "cargo test", "dir": "crates/tui", "timeout": 30})
            ),
            "bash: cargo test (in crates/tui · timeout 30s)"
        );
        assert_eq!(
            call_recap("find", &serde_json::json!({"query": "foo", "limit": 25})),
            "find foo (limit 25)"
        );
        assert_eq!(
            call_recap(
                "grep",
                &serde_json::json!({"pattern": "TODO", "context": 2})
            ),
            "grep TODO (context 2)"
        );
        // Unknown tools have no curated recap; the raw args stay visible.
        assert_eq!(
            call_recap("custom", &serde_json::json!({"a": 1})),
            "custom (args {\"a\":1})"
        );
        // edit/write already carry their path in the summary; nothing to add.
        assert_eq!(
            call_recap("edit", &serde_json::json!({"path": "a.rs"})),
            "edit a.rs"
        );
        // Empty or absent args add no noise.
        assert_eq!(call_recap("custom", &serde_json::json!({})), "custom");
    }

    #[test]
    fn default_registry_has_six_active_tools_and_prompt_metadata() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("main.rs"), "fn main() {}\n").unwrap();
        let registry = default_registry(ToolConfig::new(directory.path(), false)).unwrap();
        assert_eq!(
            registry.active_names(),
            vec!["read", "edit", "write", "bash", "find", "grep"]
        );
        assert_eq!(registry.all_names(), registry.active_names());
        let context = registry.prompt_context();
        assert!(context.snippets.iter().any(|tool| tool.name == "find"));
        assert!(context.snippets.iter().any(|tool| tool.name == "grep"));
        assert!(
            context
                .guidelines
                .iter()
                .any(|guideline| guideline.contains("Use find"))
        );
    }

    #[tokio::test]
    async fn resolve_workspace_path_finds_deepest_existing_ancestor() {
        let directory = tempfile::tempdir().unwrap();
        // Production passes a canonicalized root; mirror that so the
        // containment check is not confused by platform symlinks (e.g.
        // /var -> /private/var on macOS).
        let root = std::fs::canonicalize(directory.path()).unwrap();
        std::fs::create_dir_all(root.join("a")).unwrap();

        // Only `a/` exists; the deep tail resolves lexically after a single
        // canonicalize of the existing ancestor.
        let resolved = resolve_workspace_path("a/b/c/d.txt", Some(&root), false)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("a/b/c/d.txt"));

        // An existing file resolves through the fast path.
        std::fs::write(root.join("keep.txt"), "x").unwrap();
        let resolved = resolve_workspace_path("keep.txt", Some(&root), false)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("keep.txt"));

        // A path that lexically escapes the root is rejected up front.
        let error = resolve_workspace_path("../outside/file.txt", Some(&root), false)
            .await
            .unwrap_err();
        assert!(error.contains("outside"), "{error}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_workspace_path_rejects_symlink_escape_in_deep_paths() {
        let directory = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(directory.path()).unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("link")).unwrap();

        // The symlink is the deepest existing ancestor; canonicalizing it
        // once must surface the escape even though the tail does not exist.
        let error = resolve_workspace_path("link/sub/file.txt", Some(&root), false)
            .await
            .unwrap_err();
        assert!(error.contains("outside"), "{error}");
    }

    #[tokio::test]
    async fn resolve_workspace_path_new_file_in_existing_dir_uses_parent_fast_path() {
        let directory = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(directory.path()).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();

        // Only `src/` exists; the file does not.  The parent fast path
        // resolves it with a single canonicalize of `src/`.
        let resolved = resolve_workspace_path("src/main.rs", Some(&root), false)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("src/main.rs"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_workspace_path_parent_symlink_inside_workspace_resolves() {
        let directory = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(directory.path()).unwrap();
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::os::unix::fs::symlink(root.join("real"), root.join("link")).unwrap();

        // The parent is a symlink inside the workspace; the parent fast path
        // canonicalizes it and must accept the candidate.
        let resolved = resolve_workspace_path("link/new.txt", Some(&root), false)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("link/new.txt"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_workspace_path_parent_symlink_escape_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(directory.path()).unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("escape")).unwrap();

        // The parent fast path canonicalizes the symlink and must surface
        // the escape before the tail is considered.
        let error = resolve_workspace_path("escape/new.txt", Some(&root), false)
            .await
            .unwrap_err();
        assert!(error.contains("outside"), "{error}");
    }
}
