mod bash;
pub mod context_files;
mod edit;
pub mod file_mutation;
mod find;
mod grep;
mod multigrep;
mod read;
mod registry;
pub mod skills;
mod subagent;
mod write;

pub use bash::{BashTool, command_concurrency, truncate_command_output};
pub use context_files::{
    display_path, format_context_files, load_context_files, load_context_files_with,
};
pub use edit::EditTool;
pub use find::{FileSearchIndex, FindConfig, FindTool};
pub use grep::GrepTool;
pub use multigrep::MultiGrepTool;
pub use read::ReadTool;
pub use registry::{
    ToolPromptContext, ToolPromptEntry, ToolRegistry, ToolRegistryError, ToolRegistrySnapshot,
};
pub use skills::{
    Skill, SkillCatalog, SkillDiagnostic, SkillEntry, SkillMode, SkillSeverity, discover,
    expand_tilde, format_skills_prompt, load_skills_from_dir, parse_frontmatter,
};
pub use subagent::{SUBAGENT_TOOL_NAME, SubagentMode, SubagentRunner, SubagentTool};
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

/// Whether one tool invocation may run concurrently with other read-only
/// calls or must occupy the workspace alone.
///
/// Classification is decided by the harness, never the model: the model is
/// only told that batching independent calls is encouraged. It fails closed —
/// anything not provably side-effect-light is [`Concurrency::Exclusive`] —
/// because a wrong `ReadOnly` corrupts workspace state, while a wrong
/// `Exclusive` merely forfeits a little latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Concurrency {
    /// Provably read-only for these arguments: safe to run concurrently
    /// with other `ReadOnly` calls from the same turn.
    ReadOnly,
    /// Not side-effect-free, yet designed for fan-out: safe to run
    /// concurrently with other `Parallel` calls *of the same tool* (each
    /// invocation is expected to stay within its own scope, e.g. one
    /// subagent per crate), but never batched together with `ReadOnly`
    /// calls. Adjacent `Parallel` calls of one tool form their own
    /// concurrent batch; everything else stays in program order.
    Parallel,
    /// May observe or mutate workspace state: runs alone, in program order.
    Exclusive,
}

#[async_trait]
pub trait Tool: Send + Sync {
    /// Return the structured definition and prompt metadata for this tool.
    fn spec(&self) -> ToolSpec;

    /// Harness-side concurrency classification for one invocation. Consulted
    /// by the agent before dispatch to plan concurrent batches; the result
    /// never reaches the model. Defaults to [`Concurrency::Exclusive`] so
    /// custom and unknown tools always serialize; provably read-only tools
    /// override, and argument-sensitive tools (bash) inspect `args`.
    fn concurrency(&self, _args: &Value) -> Concurrency {
        Concurrency::Exclusive
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
/// discovered skill paths are handed to `ReadTool` so the model can load a
/// skill body via `read` on its absolute `<location>`, and the catalog is
/// stored on the registry for the prompt builder.
///
/// The generic argument accepts the workspace-aware [`ToolConfig`].
pub fn default_registry(config: ToolConfig) -> Result<ToolRegistry, ToolInitError> {
    let workspace_root = resolve_registry_workspace(&config.cwd)?;
    let index = Arc::new(
        FileSearchIndex::new(&workspace_root)
            .map_err(|error| ToolInitError::Find(error.to_string()))?,
    );
    default_registry_with_index(config, index)
}

/// Construct the full built-in registry around an assembly-owned search
/// index. Parent and child registries for one workspace should use this path
/// so FFF performs one scan and owns one watcher.
pub fn default_registry_with_index(
    config: ToolConfig,
    index: Arc<FileSearchIndex>,
) -> Result<ToolRegistry, ToolInitError> {
    let workspace_root = resolve_registry_workspace(&config.cwd)?;
    if index.root() != workspace_root {
        return Err(ToolInitError::Find(
            "search index workspace does not match registry".into(),
        ));
    }
    let skills = discover_skills_for_config(&workspace_root);
    let read_paths = skills.read_paths.clone();
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
            Box::new(MultiGrepTool::new(index.clone())),
        ],
        workspace_root.clone(),
    )
    .map_err(ToolInitError::from)?;
    registry.set_skills(skills);
    registry.set_file_search_index(index);
    Ok(registry)
}

/// Construct the read-only subregistry used by `read_only` subagents:
/// `read`, `find`, and `grep` plus the same skill discovery/read allowlists
/// and one shared file index. Deliberately no `edit`/`write`/`bash`: the
/// scheduler class is not a sandbox, so exclusion of mutating tools is the
/// actual enforcement, not prompt wording.
pub fn read_only_registry(config: ToolConfig) -> Result<ToolRegistry, ToolInitError> {
    let workspace_root = resolve_registry_workspace(&config.cwd)?;
    let index = Arc::new(
        FileSearchIndex::new(&workspace_root)
            .map_err(|error| ToolInitError::Find(error.to_string()))?,
    );
    read_only_registry_with_index(config, index)
}

/// Construct the read-only registry with an assembly-owned search index.
pub fn read_only_registry_with_index(
    config: ToolConfig,
    index: Arc<FileSearchIndex>,
) -> Result<ToolRegistry, ToolInitError> {
    let workspace_root = resolve_registry_workspace(&config.cwd)?;
    if index.root() != workspace_root {
        return Err(ToolInitError::Find(
            "search index workspace does not match registry".into(),
        ));
    }
    let skills = discover_skills_for_config(&workspace_root);
    let read_paths = skills.read_paths.clone();
    let mut registry = ToolRegistry::try_new_with_workspace(
        vec![
            Box::new(ReadTool::with_workspace_root(&workspace_root).with_allowed_paths(read_paths)),
            Box::new(FindTool::new(index.clone())),
            Box::new(GrepTool::new(index.clone())),
            Box::new(MultiGrepTool::new(index.clone())),
        ],
        workspace_root,
    )
    .map_err(ToolInitError::from)?;
    registry.set_skills(skills);
    registry.set_file_search_index(index);
    Ok(registry)
}

/// Canonicalized workspace root shared by both registry constructors.
fn resolve_registry_workspace(cwd: &Path) -> Result<PathBuf, ToolInitError> {
    std::fs::canonicalize(cwd).map_err(|source| ToolInitError::Workspace {
        path: cwd.to_path_buf(),
        source,
    })
}

pub(crate) fn discover_skills_for_config(workspace_root: &Path) -> SkillCatalog {
    // Project roots: cwd up to git repo root (or filesystem root).
    let mut roots: Vec<(PathBuf, SkillMode)> = Vec::new();
    let mut dir = workspace_root.to_path_buf();
    loop {
        roots.push((dir.join(".harness/skills"), SkillMode::Harness));
        roots.push((dir.join(".agents/skills"), SkillMode::Agents));
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
            skills::home_dir()
                .map(|home| home.join(".harness/skills"))
                .unwrap_or_default()
        });
    if !global.as_os_str().is_empty() {
        roots.push((global, SkillMode::Harness));
    }
    let agents_global = skills::home_dir()
        .map(|home| home.join(".agents/skills"))
        .unwrap_or_default();
    if !agents_global.as_os_str().is_empty() {
        roots.push((agents_global, SkillMode::Agents));
    }
    let catalog = discover(&roots);
    // Surface discovery diagnostics (frontmatter typos, dropped skills,
    // collisions) so silent drops become visible in the log at startup.
    for diagnostic in &catalog.diagnostics {
        tracing::warn!(
            severity = ?diagnostic.severity,
            path = ?diagnostic.path,
            "{}", diagnostic.message
        );
    }
    catalog
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
        "multigrep" => {
            let count = args
                .get("patterns")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            match args.get("path").and_then(Value::as_str) {
                Some(path) => format!("multigrep {count} patterns in {path}"),
                None => format!("multigrep {count} patterns"),
            }
        }
        "subagent" => {
            // Same helper the tool uses for its live summary, so previews
            // never diverge between dispatch time and snapshot replay.
            let (description, prompt, _mode) = subagent::parse_args(args).unwrap_or((
                None,
                None,
                Ok(subagent::SubagentMode::ReadOnly),
            ));
            format!("subagent: {}", subagent::preview(description, prompt))
        }
        _ => name.to_owned(),
    }
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value)
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
    fn default_registry_has_all_tools_and_prompt_metadata() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("main.rs"), "fn main() {}\n").unwrap();
        let registry = default_registry(ToolConfig::new(directory.path(), false)).unwrap();
        let names: Vec<String> = registry
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect();
        assert_eq!(
            names,
            vec!["read", "edit", "write", "bash", "find", "grep", "multigrep"]
        );
        let context = registry.prompt_context();
        let snippets: Vec<(&str, &str)> = context
            .snippets
            .iter()
            .map(|tool| (tool.name.as_str(), tool.snippet.as_str()))
            .collect();
        assert_eq!(
            snippets,
            vec![
                ("read", "Read files"),
                ("edit", "Apply exact replacements"),
                ("write", "Create or replace files"),
                ("bash", "Run commands"),
                ("find", "Find files and directories"),
                ("grep", "Search file contents"),
                ("multigrep", "Search multiple literal patterns"),
            ]
        );
        assert!(
            context
                .guidelines
                .iter()
                .any(|guideline| guideline.contains("Use find"))
        );
    }

    #[test]
    fn read_only_registry_exposes_no_mutating_tools() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("main.rs"), "fn main() {}\n").unwrap();
        let registry = read_only_registry(ToolConfig::new(directory.path(), false)).unwrap();
        let names: Vec<String> = registry
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect();
        assert_eq!(names, vec!["read", "find", "grep", "multigrep"]);
        // Skill discovery still applies, so skill bodies stay loadable.
        assert_eq!(
            registry.workspace_root(),
            std::fs::canonicalize(directory.path()).unwrap()
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
