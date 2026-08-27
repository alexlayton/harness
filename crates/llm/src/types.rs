use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::fmt;
use std::str::FromStr;

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

/// Portable reasoning effort requested from a model.
///
/// Providers translate these semantic levels to their own wire values. A
/// provider may reject levels that the selected model does not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Maximum,
}

/// Provider-neutral control over model reasoning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningPolicy {
    /// Preserve the provider/model default behavior.
    #[default]
    Auto,
    /// Do not request extended reasoning.
    Off,
    /// Request a portable effort level.
    Effort(ReasoningEffort),
}

impl ReasoningPolicy {
    /// Stable configuration and CLI spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Effort(ReasoningEffort::Minimal) => "minimal",
            Self::Effort(ReasoningEffort::Low) => "low",
            Self::Effort(ReasoningEffort::Medium) => "medium",
            Self::Effort(ReasoningEffort::High) => "high",
            Self::Effort(ReasoningEffort::Maximum) => "maximum",
        }
    }
}

impl fmt::Display for ReasoningPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ReasoningPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" | "none" => Ok(Self::Off),
            "minimal" => Ok(Self::Effort(ReasoningEffort::Minimal)),
            "low" => Ok(Self::Effort(ReasoningEffort::Low)),
            "medium" => Ok(Self::Effort(ReasoningEffort::Medium)),
            "high" => Ok(Self::Effort(ReasoningEffort::High)),
            "maximum" | "max" | "xhigh" => Ok(Self::Effort(ReasoningEffort::Maximum)),
            _ => Err(format!(
                "invalid reasoning effort `{value}` (expected auto, off, minimal, low, medium, high, or maximum)"
            )),
        }
    }
}

impl Serialize for ReasoningPolicy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ReasoningPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletionRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning: ReasoningPolicy,
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
