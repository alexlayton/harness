mod bash;
mod edit;
pub mod file_mutation;
mod find;
mod read;
mod registry;
mod write;

pub use bash::{BashTool, truncate_command_output};
pub use edit::EditTool;
pub use find::{FileSearchIndex, FindConfig, FindTool};
pub use read::ReadTool;
pub use registry::{ToolPromptContext, ToolPromptEntry, ToolRegistry, ToolRegistryError};
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

/// Construct the five built-in tools for one workspace.  The index is created
/// once here and shared by every `find` call; it is never initialized per
/// request.
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

    let index = Arc::new(
        FileSearchIndex::new(&workspace_root)
            .map_err(|error| ToolInitError::Find(error.to_string()))?,
    );
    ToolRegistry::try_new_with_workspace(
        vec![
            Box::new(ReadTool::with_workspace_root(&workspace_root)),
            Box::new(EditTool::with_workspace_root(&workspace_root)),
            Box::new(WriteTool::with_workspace_root(&workspace_root)),
            Box::new(BashTool::with_rtk_and_workspace_root(
                config.rtk,
                &workspace_root,
            )),
            Box::new(FindTool::new(index)),
        ],
        workspace_root,
    )
    .map_err(ToolInitError::from)
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
    let candidate_path = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    let candidate = lexical_normalize(&candidate_path);
    if !candidate.starts_with(&root) {
        return Err(format!("path is outside workspace root {}", root.display()));
    }

    // Lexical containment handles `..`; canonicalizing the nearest existing
    // ancestor also prevents a symlink inside the workspace from escaping it.
    let mut existing = candidate.clone();
    loop {
        match fs::canonicalize(&existing).await {
            Ok(canonical) => {
                if !canonical.starts_with(&root) {
                    return Err(format!(
                        "path resolves outside workspace root {}",
                        root.display()
                    ));
                }
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !existing.pop() {
                    return Err(format!("cannot resolve path {value}: {error}"));
                }
            }
            Err(error) => return Err(format!("cannot resolve path {value}: {error}")),
        }
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
    fn default_registry_has_five_active_tools_and_prompt_metadata() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("main.rs"), "fn main() {}\n").unwrap();
        let registry = default_registry(ToolConfig::new(directory.path(), false)).unwrap();
        assert_eq!(
            registry.active_names(),
            vec!["read", "edit", "write", "bash", "find"]
        );
        assert_eq!(registry.all_names(), registry.active_names());
        let context = registry.prompt_context();
        assert!(context.snippets.iter().any(|tool| tool.name == "find"));
        assert!(
            context
                .guidelines
                .iter()
                .any(|guideline| guideline.contains("Use find"))
        );
    }
}
