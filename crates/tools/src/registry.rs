use super::{Tool, ToolOutput};
use llm::ToolDefinition;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Cloneable gate used to serialize workspace-mutating tool executions across
/// otherwise independent registries.
pub type ToolExecutionGate = Arc<Mutex<()>>;

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
    file_search_index: Option<Arc<super::FileSearchIndex>>,
    execution_gate: Option<ToolExecutionGate>,
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
            file_search_index: None,
            execution_gate: None,
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

    /// Configure a shared execution gate for workspace-affecting tools.
    ///
    /// Registries using clones of the same gate serialize every invocation
    /// not classified as [`super::Concurrency::ReadOnly`]. Read-only tools do
    /// not acquire the gate. Without a gate, execution behavior is unchanged.
    pub fn with_execution_gate(mut self, gate: ToolExecutionGate) -> Self {
        self.set_execution_gate(gate);
        self
    }

    /// Set the shared execution gate for workspace-affecting tools.
    ///
    /// The gate is held for the complete tool future. Share one gate between
    /// registries that target the same workspace and use distinct gates for
    /// independent workspaces.
    pub fn set_execution_gate(&mut self, gate: ToolExecutionGate) {
        self.execution_gate = Some(gate);
    }

    /// Return the shared execution gate, when cross-registry coordination is enabled.
    pub fn execution_gate(&self) -> Option<&ToolExecutionGate> {
        self.execution_gate.as_ref()
    }

    /// Set the discovered skill catalog (called by `default_registry`).
    pub fn set_skills(&mut self, skills: super::skills::SkillCatalog) {
        self.skills = Some(skills);
    }

    /// Retain the workspace search index used by built-in search tools so
    /// assembly can inject the same watcher into child registries.
    pub fn set_file_search_index(&mut self, index: std::sync::Arc<super::FileSearchIndex>) {
        self.file_search_index = Some(index);
    }

    pub fn file_search_index(&self) -> Option<&std::sync::Arc<super::FileSearchIndex>> {
        self.file_search_index.as_ref()
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

    /// Capture definitions and prompt metadata from one immutable registry
    /// view. Callers building a completion request must use this instead of
    /// independently reading the two surfaces.
    pub fn snapshot(&self) -> ToolRegistrySnapshot {
        ToolRegistrySnapshot {
            definitions: self.definitions(),
            prompt_context: self.prompt_context(),
        }
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

    // Tool arguments routinely contain commands, source text, and delegated
    // prompts. Never serialize them into tracing fields.
    #[tracing::instrument(name = "tool", skip_all, fields(name = %name))]
    pub async fn execute(&self, name: &str, args: Value, cancel: CancellationToken) -> ToolOutput {
        let Some(tool) = self.tools.iter().find(|tool| tool.name == name) else {
            return ToolOutput {
                content: format!("unknown tool: {name}"),
                is_error: true,
                summary: name.to_owned(),
            };
        };
        if tool.tool.concurrency(&args) == super::Concurrency::ReadOnly {
            return tool.tool.execute(args, cancel).await;
        }
        // A workspace subagent runner acquires this same gate around its
        // complete delegated workflow. Holding the non-reentrant mutex here
        // as well would deadlock before the child could begin. Direct parent
        // mutations continue through the gate below.
        if name == super::subagent::SUBAGENT_TOOL_NAME {
            return tool.tool.execute(args, cancel).await;
        }
        let Some(gate) = &self.execution_gate else {
            return tool.tool.execute(args, cancel).await;
        };
        let _guard = tokio::select! {
            guard = gate.lock() => guard,
            _ = cancel.cancelled() => return ToolOutput {
                content: "cancelled before workspace execution".into(),
                is_error: true,
                summary: super::call_summary(name, &args),
            },
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

/// An immutable provider/prompt view of a registry.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolRegistrySnapshot {
    /// JSON-schema definitions sent to the provider.
    pub definitions: Vec<ToolDefinition>,
    /// Concise metadata rendered into the system prompt.
    pub prompt_context: ToolPromptContext,
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
    use std::sync::Arc;
    use tokio::sync::{Mutex, Notify, mpsc};
    use tokio::time::{Duration, timeout};

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

        fn concurrency(&self, _: &Value) -> super::super::Concurrency {
            super::super::Concurrency::ReadOnly
        }

        async fn execute(&self, _: Value, _: CancellationToken) -> ToolOutput {
            ToolOutput {
                content: "ok".into(),
                is_error: false,
                summary: self.name.into(),
            }
        }
    }

    struct BlockingTool {
        class: super::super::Concurrency,
        entered: mpsc::UnboundedSender<()>,
        release: Arc<Notify>,
    }

    struct GatedWorkspaceRunner {
        gate: ToolExecutionGate,
        entered: mpsc::UnboundedSender<()>,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl super::super::SubagentRunner for GatedWorkspaceRunner {
        async fn run(
            &self,
            _: &str,
            _: &str,
            mode: super::super::SubagentMode,
            _: CancellationToken,
        ) -> Result<String, String> {
            assert_eq!(mode, super::super::SubagentMode::Workspace);
            let _guard = self.gate.lock().await;
            self.entered.send(()).unwrap();
            self.release.notified().await;
            Ok("done".into())
        }
    }

    #[async_trait]
    impl Tool for BlockingTool {
        fn spec(&self) -> super::super::ToolSpec {
            TestTool { name: "block" }.spec()
        }

        fn concurrency(&self, _: &Value) -> super::super::Concurrency {
            self.class
        }

        async fn execute(&self, _: Value, _: CancellationToken) -> ToolOutput {
            self.entered.send(()).unwrap();
            self.release.notified().await;
            ToolOutput {
                content: "ok".into(),
                is_error: false,
                summary: "block".into(),
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

    #[test]
    fn snapshot_contains_matching_definition_and_prompt_surfaces() {
        let registry = ToolRegistry::new(vec![Box::new(TestTool { name: "one" })]);
        let snapshot = registry.snapshot();
        assert_eq!(
            snapshot.definitions[0].name,
            snapshot.prompt_context.snippets[0].name
        );
    }

    #[tokio::test]
    async fn shared_gate_serializes_exclusive_tools_across_registries() {
        let gate = Arc::new(Mutex::new(()));
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let make_registry = || {
            ToolRegistry::new(vec![Box::new(BlockingTool {
                class: super::super::Concurrency::Exclusive,
                entered: entered_tx.clone(),
                release: release.clone(),
            })])
            .with_execution_gate(gate.clone())
        };
        let first = make_registry();
        let second = make_registry();
        let first_task = tokio::spawn(async move {
            first
                .execute("block", json!({}), CancellationToken::new())
                .await
        });
        entered_rx.recv().await.unwrap();
        let second_task = tokio::spawn(async move {
            second
                .execute("block", json!({}), CancellationToken::new())
                .await
        });

        assert!(
            timeout(Duration::from_millis(50), entered_rx.recv())
                .await
                .is_err()
        );
        release.notify_one();
        timeout(Duration::from_secs(1), entered_rx.recv())
            .await
            .unwrap()
            .unwrap();
        release.notify_one();
        first_task.await.unwrap();
        second_task.await.unwrap();
    }

    #[tokio::test]
    async fn workspace_subagents_gate_children_without_nested_deadlock() {
        let gate = Arc::new(Mutex::new(()));
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let make_registry = || {
            let mut registry = ToolRegistry::empty().with_execution_gate(gate.clone());
            registry
                .register_subagent(Arc::new(GatedWorkspaceRunner {
                    gate: gate.clone(),
                    entered: entered_tx.clone(),
                    release: release.clone(),
                }))
                .unwrap();
            registry
        };
        let first = make_registry();
        let second = make_registry();
        let args = json!({"description": "test", "prompt": "test", "mode": "workspace"});
        let first_task = tokio::spawn({
            let args = args.clone();
            async move {
                first
                    .execute("subagent", args, CancellationToken::new())
                    .await
            }
        });
        timeout(Duration::from_secs(1), entered_rx.recv())
            .await
            .expect("outer subagent call deadlocked")
            .unwrap();
        let second_task = tokio::spawn(async move {
            second
                .execute("subagent", args, CancellationToken::new())
                .await
        });
        assert!(
            timeout(Duration::from_millis(50), entered_rx.recv())
                .await
                .is_err(),
            "workspace child mutations overlapped"
        );
        release.notify_one();
        timeout(Duration::from_secs(1), entered_rx.recv())
            .await
            .unwrap()
            .unwrap();
        release.notify_one();
        timeout(Duration::from_secs(1), first_task)
            .await
            .unwrap()
            .unwrap();
        timeout(Duration::from_secs(1), second_task)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cancelled_tool_stops_waiting_for_the_execution_gate() {
        let gate = Arc::new(Mutex::new(()));
        let guard = gate.lock().await;
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let registry = ToolRegistry::new(vec![Box::new(BlockingTool {
            class: super::super::Concurrency::Exclusive,
            entered: entered_tx,
            release: Arc::new(Notify::new()),
        })])
        .with_execution_gate(gate.clone());
        let cancel = CancellationToken::new();
        cancel.cancel();

        let output = timeout(
            Duration::from_millis(50),
            registry.execute("block", json!({}), cancel),
        )
        .await
        .expect("cancelled invocation remained queued");

        assert!(output.is_error);
        assert!(output.content.contains("cancelled"));
        assert!(entered_rx.try_recv().is_err());
        drop(guard);
    }

    #[tokio::test]
    async fn read_only_tools_do_not_acquire_execution_gate() {
        let gate = Arc::new(Mutex::new(()));
        let guard = gate.lock().await;
        let registry = ToolRegistry::new(vec![Box::new(TestTool { name: "one" })])
            .with_execution_gate(gate.clone());

        timeout(
            Duration::from_millis(50),
            registry.execute("one", json!({}), CancellationToken::new()),
        )
        .await
        .expect("read-only invocation waited for execution gate");
        drop(guard);
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
