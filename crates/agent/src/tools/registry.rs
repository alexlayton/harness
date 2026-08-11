use super::{Tool, ToolOutput};
use llm::ToolDefinition;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn empty() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.definitions()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect()
    }

    pub async fn execute(&self, name: &str, args: Value, cancel: CancellationToken) -> ToolOutput {
        let Some(tool) = self
            .tools
            .iter()
            .find(|tool| tool.definition().name == name)
        else {
            return ToolOutput {
                content: format!("unknown tool: {name}"),
                is_error: true,
                summary: name.to_owned(),
            };
        };
        tool.execute(args, cancel).await
    }

    pub async fn dispatch(&self, name: &str, args: Value, cancel: CancellationToken) -> ToolOutput {
        self.execute(name, args, cancel).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        super::default_registry()
    }
}
