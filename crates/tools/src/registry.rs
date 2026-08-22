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

/// Registry of all known tools.  Registration order is retained for
/// deterministic definitions and prompts.
pub struct ToolRegistry {
    tools: Vec<RegisteredTool>,
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

    /// Register the subagent tool with an injected runner.  Kept off
    /// [`Self::register`] so callers cannot accidentally advertise a
    /// subagent schema without a working runner behind it.
    pub fn register_subagent(
        &mut self,
        runner: std::sync::Arc<dyn super::subagent::SubagentRunner>,
    ) -> Result<(), ToolRegistryError> {
        self.register(Box::new(super::subagent::SubagentTool::new(runner)))
    }

    /// Whether the subagent tool is available in this registry.
    pub fn has_subagent(&self) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.name == super::subagent::SUBAGENT_TOOL_NAME)
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
        self.tools.push(RegisteredTool { name, spec, tool });
        Ok(())
    }

    /// Structured definitions for every registered tool.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|tool| tool.spec.definition.clone())
            .collect()
    }

    pub fn prompt_context(&self) -> ToolPromptContext {
        let mut snippets = Vec::new();
        let mut guidelines = Vec::new();
        for tool in &self.tools {
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
        tool.tool.execute(args, cancel).await
    }

    /// Harness-side concurrency classification for one invocation. Unknown
    /// tools classify as [`super::Concurrency::Exclusive`], mirroring the
    /// trait default, so a name that misses the registry can never join a
    /// batch. Read-only calls batch together, `Parallel` calls fan out per
    /// tool, everything else serializes.
    pub fn concurrency(&self, name: &str, args: &Value) -> super::Concurrency {
        match self.tools.iter().find(|tool| tool.name == name) {
            Some(tool) => tool.tool.concurrency(args),
            None => super::Concurrency::Exclusive,
        }
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
    async fn registered_tools_are_executable_and_advertised() {
        let registry = ToolRegistry::new(vec![
            Box::new(TestTool { name: "one" }),
            Box::new(TestTool { name: "two" }),
        ]);
        assert_eq!(registry.definitions().len(), 2);
        let output = registry
            .execute("one", json!({}), CancellationToken::new())
            .await;
        assert!(!output.is_error, "{}", output.content);
    }
}
