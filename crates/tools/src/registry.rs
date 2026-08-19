use super::{Tool, ToolOutput};
use llm::ToolDefinition;
use llm::util::truncate_utf8;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolRegistryError {
    #[error("duplicate tool name: {0}")]
    DuplicateName(String),
    #[error("tool name must not be empty")]
    EmptyName,
}

struct RegisteredTool {
    name: String,
    spec: super::ToolSpec,
    tool: Box<dyn Tool>,
}

/// Registry of all known tools and the active subset sent to the model.
/// Registration order is retained for deterministic definitions and prompts.
pub struct ToolRegistry {
    tools: Vec<RegisteredTool>,
    active: Vec<String>,
    workspace_root: PathBuf,
    /// Optional skills catalog discovered at startup; used to render the
    /// skills section of the system prompt and to hand read-paths to tools.
    skills: Option<super::skills::SkillCatalog>,
}

impl ToolRegistry {
    /// Construct a registry and reject duplicate names.  `try_new` is the
    /// fallible equivalent useful to callers that do not want a panic when
    /// assembling a dynamic registry.
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self::try_new(tools).expect("tool registry contains duplicate or empty tool names")
    }

    pub fn try_new(tools: Vec<Box<dyn Tool>>) -> Result<Self, ToolRegistryError> {
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::try_new_with_workspace(tools, workspace_root)
    }

    pub fn try_new_with_workspace(
        tools: Vec<Box<dyn Tool>>,
        workspace_root: impl Into<PathBuf>,
    ) -> Result<Self, ToolRegistryError> {
        let mut registry = Self {
            tools: Vec::new(),
            active: Vec::new(),
            workspace_root: workspace_root.into(),
            skills: None,
        };
        for tool in tools {
            registry.register(tool)?;
        }
        Ok(registry)
    }

    pub fn empty() -> Self {
        Self::try_new(Vec::new()).expect("an empty tool registry cannot fail")
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Set the discovered skill catalog (called by `default_registry`).
    pub fn set_skills(&mut self, skills: super::skills::SkillCatalog) {
        self.skills = Some(skills);
    }

    /// The discovered skill catalog, if any.
    pub fn skills(&self) -> Option<&super::skills::SkillCatalog> {
        self.skills.as_ref()
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) -> Result<(), ToolRegistryError> {
        let spec = tool.spec();
        let name = spec.definition.name.clone();
        if name.is_empty() {
            return Err(ToolRegistryError::EmptyName);
        }
        if self.tools.iter().any(|registered| registered.name == name) {
            return Err(ToolRegistryError::DuplicateName(name));
        }
        self.active.push(name.clone());
        self.tools.push(RegisteredTool { name, spec, tool });
        Ok(())
    }

    /// Structured definitions for active tools only.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| self.active.iter().any(|name| name == &tool.name))
            .map(|tool| tool.spec.definition.clone())
            .collect()
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.definitions()
    }

    /// Names of active tools, in deterministic registration order.
    pub fn active_names(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter(|tool| self.active.iter().any(|name| name == &tool.name))
            .map(|tool| tool.name.clone())
            .collect()
    }

    /// Names of all registered tools, including inactive tools.
    pub fn all_names(&self) -> Vec<String> {
        self.tools.iter().map(|tool| tool.name.clone()).collect()
    }

    /// Compatibility alias: historically `names()` meant the tools sent to
    /// the provider, so it continues to report the active subset.
    pub fn names(&self) -> Vec<String> {
        self.active_names()
    }

    /// Activate the requested known tools. Unknown names are ignored, which
    /// makes configuration allowlists forward-compatible with newer tools.
    /// The registry's order, rather than the allowlist's order, controls the
    /// output order.
    pub fn set_active_tools(&mut self, names: &[String]) {
        self.active = self
            .tools
            .iter()
            .filter(|tool| names.iter().any(|name| name == &tool.name))
            .map(|tool| tool.name.clone())
            .collect();
    }

    pub fn prompt_context(&self) -> ToolPromptContext {
        let mut snippets = Vec::new();
        let mut guidelines = Vec::new();
        for tool in self.tools.iter().filter(|tool| {
            self.active
                .iter()
                .any(|active_name| active_name == &tool.name)
        }) {
            let prompt = tool.spec.prompt.clone();
            if let Some(snippet) = prompt.snippet {
                snippets.push(ToolPromptEntry {
                    name: tool.name.clone(),
                    snippet,
                });
            }
            guidelines.extend(prompt.guidelines);
        }
        ToolPromptContext {
            snippets,
            guidelines,
        }
    }

    #[tracing::instrument(
        name = "tool",
        skip_all,
        fields(name = %name, args = %truncate_utf8(&args.to_string(), 512))
    )]
    pub async fn execute(&self, name: &str, args: Value, cancel: CancellationToken) -> ToolOutput {
        let Some(tool) = self.tools.iter().find(|tool| tool.name == name) else {
            return ToolOutput {
                content: format!("unknown tool: {name}"),
                is_error: true,
                summary: name.to_owned(),
            };
        };
        if !self.active.iter().any(|active_name| active_name == name) {
            return ToolOutput {
                content: format!("tool is inactive: {name}"),
                is_error: true,
                summary: name.to_owned(),
            };
        }
        tool.tool.execute(args, cancel).await
    }

    pub async fn dispatch(&self, name: &str, args: Value, cancel: CancellationToken) -> ToolOutput {
        self.execute(name, args, cancel).await
    }
}

/// Prompt-facing tool entry.  It intentionally contains no JSON schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPromptEntry {
    pub name: String,
    pub snippet: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolPromptContext {
    pub snippets: Vec<ToolPromptEntry>,
    pub guidelines: Vec<String>,
}

impl ToolPromptContext {
    pub fn is_empty(&self) -> bool {
        self.snippets.is_empty() && self.guidelines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use llm::ToolDefinition;
    use serde_json::json;

    struct TestTool {
        name: &'static str,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn spec(&self) -> super::super::ToolSpec {
            super::super::ToolSpec {
                definition: ToolDefinition {
                    name: self.name.into(),
                    description: "test".into(),
                    parameters: json!({"type":"object"}),
                },
                prompt: super::super::ToolPrompt {
                    snippet: Some(self.name.into()),
                    guidelines: vec![format!("Use {} for tests.", self.name)],
                },
            }
        }

        async fn execute(&self, _: Value, _: CancellationToken) -> ToolOutput {
            ToolOutput {
                content: "ok".into(),
                is_error: false,
                summary: self.name.into(),
            }
        }
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let result = ToolRegistry::try_new(vec![
            Box::new(TestTool { name: "one" }),
            Box::new(TestTool { name: "one" }),
        ]);
        assert!(matches!(
            result,
            Err(ToolRegistryError::DuplicateName(name)) if name == "one"
        ));
    }

    #[tokio::test]
    async fn inactive_tools_are_not_executable_or_advertised() {
        let mut registry = ToolRegistry::new(vec![
            Box::new(TestTool { name: "one" }),
            Box::new(TestTool { name: "two" }),
        ]);
        registry.set_active_tools(&["one".into()]);
        assert_eq!(registry.active_names(), vec!["one"]);
        assert_eq!(registry.all_names(), vec!["one", "two"]);
        assert_eq!(registry.definitions().len(), 1);
        let output = registry
            .execute("two", json!({}), CancellationToken::new())
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("inactive"));
    }
}
