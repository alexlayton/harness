use serde_json::Value;

/// The role of a message in a conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A piece of a message.  Reasoning is kept in local history for display/debugging,
/// but every provider deliberately drops it when it builds a request.
#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Text(String),
    Reasoning(String),
    /// Provider-owned state required to continue a conversation. It is never
    /// displayable text and other providers must ignore tags they do not own.
    Opaque {
        provider: String,
        data: Value,
    },
    ToolCall(ToolCall),
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletionRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub cost: Option<f64>,
}

/// Current subscription allowance reported by a provider. This is distinct
/// from [`Usage`], which describes tokens consumed by one model request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubscriptionUsage {
    /// Provider plan name when the endpoint exposes one (for example `plus`).
    pub plan: Option<String>,
    /// Independently resetting allowance windows, in provider display order.
    pub windows: Vec<SubscriptionUsageWindow>,
}

/// One rolling subscription allowance window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubscriptionUsageWindow {
    /// Human-readable window name, such as `rolling`, `weekly`, or `5 hours`.
    pub label: String,
    /// Percentage of the allowance consumed. Providers currently report whole
    /// percentages; `u16` avoids silently wrapping unexpected values over 255.
    pub used_percent: u16,
    /// Provider status when available (OpenCode Go reports `ok`).
    pub status: Option<String>,
    /// Absolute reset time. Kept as provider text because OpenCode returns
    /// RFC 3339 while Codex returns Unix seconds.
    pub resets_at: Option<String>,
    /// Relative reset duration when the provider reports it.
    pub resets_after_seconds: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    TextDelta(String),
    ReasoningDelta(String),
    /// Opaque continuation state returned by a provider (for example Codex
    /// encrypted reasoning). Frontends deliberately never render this.
    OpaqueState {
        provider: String,
        data: Value,
    },
    ToolCallComplete(ToolCall),
    Done {
        stop_reason: Option<String>,
        usage: Option<Usage>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub name: Option<String>,
    pub context_length: Option<u64>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.into())],
        }
    }

    pub fn assistant(content: Vec<Content>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    pub fn tool_result(
        call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: vec![Content::ToolResult {
                tool_call_id: call_id.into(),
                content: content.into(),
                is_error,
            }],
        }
    }
}
