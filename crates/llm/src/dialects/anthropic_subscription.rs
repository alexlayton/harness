//! Subscription-only wire adaptation, derived from pi-sub-anthropic/omp (MIT).
//! See THIRD_PARTY_NOTICES.md. The generic Anthropic dialect is unchanged.
use super::{StreamParser, anthropic::AnthropicParser};
use crate::{
    CompletionRequest, Content, LlmError, ReasoningEffort, ReasoningPolicy, Role, StreamEvent,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

pub(crate) const TAG: &str = "anthropic-subscription";
pub(crate) const VERSION: &str = "2.1.280";
const SEED: u64 = 0x4d659218e32a3268;

#[cfg(test)]
#[path = "anthropic_subscription_tests.rs"]
mod tests;

/// JavaScript string indexing selects UTF-16 code units, not Unicode scalars.
/// Node's UTF-8 encoder replaces unpaired surrogates after joining the units.
fn billing(text: &str) -> String {
    let units: Vec<u16> = text.encode_utf16().collect();
    let selected: Vec<u16> = [4, 7, 20]
        .iter()
        .map(|&i| units.get(i).copied().unwrap_or(b'0' as u16))
        .collect();
    let suffix = Sha256::digest(format!(
        "59cf53e54c78{}{VERSION}",
        String::from_utf16_lossy(&selected)
    ));
    let hex = format!("{suffix:x}");
    format!(
        "x-anthropic-billing-header: cc_version={VERSION}.{}; cc_entrypoint=claude-desktop; cch=00000;",
        &hex[..3]
    )
}

/// Serialize exactly once. The billing block's field order and patch offset
/// are explicit, independent of serde_json's map-order feature unification.
pub(crate) fn serialize(req: &CompletionRequest) -> Result<Vec<u8>, LlmError> {
    let mut body = super::anthropic::build_request_body(req);
    let max = req.max_tokens.unwrap_or(64000).min(64000);
    if max == 0 {
        return Err(invalid("max_tokens must be positive"));
    }
    body["max_tokens"] = json!(max);
    let mut prompt = req.system.iter().cloned().collect::<Vec<_>>();
    for message in &req.messages {
        if message.role == Role::System {
            for c in &message.content {
                if let Content::Text(t) = c {
                    prompt.push(t.clone());
                }
            }
        }
    }
    let first = req
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .map(|m| {
            m.content
                .iter()
                .filter_map(|c| {
                    if let Content::Text(t) = c {
                        Some(t.as_str())
                    } else {
                        None
                    }
                })
                .collect::<String>()
        })
        .unwrap_or_default();
    let mut ids = HashMap::<String, String>::new();
    // Hash every original ID, including already-valid ones: punctuation
    // replacement alone aliases a/b and a?b (or a_b) across providers.
    for m in &req.messages {
        for c in &m.content {
            let id = match c {
                Content::ToolCall(c) => Some(&c.id),
                Content::ToolResult { tool_call_id, .. } => Some(tool_call_id),
                _ => None,
            };
            if let Some(id) = id {
                ids.entry(id.clone()).or_insert_with(|| normalize_id(id));
            }
        }
    }
    let mut messages = Vec::new();
    if !prompt.is_empty() {
        messages.push(json!({"role":"user","content":[{"type":"text","text":format!("<system-reminder>\n{}\n</system-reminder>",prompt.join("\n\n"))}]}));
        messages.push(json!({"role":"assistant","content":[{"type":"text","text":"Understood."}]}));
    }
    // Reuse generic grouping, replacing assistant messages individually to
    // keep opaque signed blocks in their original position without duplicating
    // the reasoning/text already retained for display in neutral history.
    let mut history = req.messages.clone();
    for m in &mut history {
        for c in &mut m.content {
            match c {
                Content::ToolCall(call) => {
                    call.id = ids[&call.id].clone();
                    call.name = format!("_{}", call.name);
                }
                Content::ToolResult { tool_call_id, .. } => {
                    *tool_call_id = ids[tool_call_id].clone()
                }
                _ => {}
            }
        }
    }
    // Convert each run so consecutive tool results remain grouped.
    let mut at = 0;
    while at < history.len() {
        let m = &history[at];
        if m.role == Role::Assistant
            && let Some(blocks) = m.content.iter().find_map(|c| match c {
                Content::Opaque { provider, data } if provider == TAG => replay_blocks(data),
                _ => None,
            })
        {
            let mut blocks = blocks;
            for b in &mut blocks {
                if b["type"] == "tool_use" {
                    if let Some(id) = b["id"].as_str() {
                        b["id"] = json!(normalize_id(id));
                    }
                    if let Some(name) = b["name"].as_str() {
                        b["name"] = json!(format!("_{name}"));
                    }
                }
            }
            messages.push(json!({"role":"assistant","content":blocks}));
            at += 1;
            continue;
        }
        let mut end = at + 1;
        if m.role == Role::Tool {
            while end < history.len() && history[end].role == Role::Tool {
                end += 1;
            }
        }
        messages.extend(super::anthropic::convert_messages(&history[at..end]));
        at = end;
    }
    // No more than three cache breakpoints: identity, final tool, final turn.
    if let Some(block) = messages
        .last_mut()
        .and_then(|m| m["content"].as_array_mut())
        .and_then(|b| b.last_mut())
        && block["type"] != "thinking"
        && block["type"] != "redacted_thinking"
    {
        block["cache_control"] = json!({"type":"ephemeral"});
    }
    body["messages"] = json!(messages);
    if let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools {
            let name = tool["name"].as_str().unwrap_or_default();
            tool["name"] = json!(format!("_{name}"));
        }
    }
    body["metadata"] = json!({"user_id":format!("user_{:x}_account_{}_session_{}",Sha256::digest(uuid::Uuid::new_v4().as_bytes()),uuid::Uuid::new_v4(),uuid::Uuid::new_v4())});
    let adaptive = matches!(
        req.model.as_str(),
        "claude-opus-5-5"
            | "claude-sonnet-5-5"
            | "claude-haiku-5-5"
            | "claude-fable-5-1"
            | "claude-opus-5"
            | "claude-sonnet-5"
            | "claude-fable-5"
            | "claude-opus-4-8"
            | "claude-opus-4-7"
            | "claude-opus-4-6"
            | "claude-sonnet-4-6"
    );
    let thinking = adaptive || matches!(req.reasoning, ReasoningPolicy::Effort(_));
    if adaptive && req.reasoning == ReasoningPolicy::Off {
        return Err(invalid(
            "this model uses adaptive thinking; Off is unsupported",
        ));
    }
    if thinking {
        body.as_object_mut().unwrap().remove("temperature");
        if req.temperature.is_some() {
            return Err(invalid(
                "temperature is incompatible with subscription thinking",
            ));
        }
        if adaptive {
            let effort = match req.reasoning {
                ReasoningPolicy::Effort(ReasoningEffort::Minimal | ReasoningEffort::Low) => "low",
                ReasoningPolicy::Effort(ReasoningEffort::Medium) => "medium",
                ReasoningPolicy::Effort(ReasoningEffort::Maximum)
                    if matches!(
                        req.model.as_str(),
                        "claude-opus-5-5"
                            | "claude-sonnet-5-5"
                            | "claude-haiku-5-5"
                            | "claude-fable-5-1"
                            | "claude-opus-5"
                            | "claude-sonnet-5"
                            | "claude-fable-5"
                            | "claude-opus-4-8"
                            | "claude-opus-4-7"
                            | "claude-opus-4-6"
                    ) =>
                {
                    "max"
                }
                _ => "high",
            };
            body["thinking"] = json!({"type":"adaptive","display":"summarized"});
            body["output_config"] = json!({"effort":effort});
        } else {
            if max <= 1024 {
                return Err(invalid("thinking requires max_tokens > 1024"));
            }
            let budget = match req.reasoning {
                ReasoningPolicy::Effort(ReasoningEffort::Minimal) => 1024,
                ReasoningPolicy::Effort(ReasoningEffort::Low) => 4096,
                ReasoningPolicy::Effort(ReasoningEffort::Medium) => 10240,
                ReasoningPolicy::Effort(ReasoningEffort::Maximum) => 49152,
                _ => 20480,
            };
            body["thinking"] = json!({"type":"enabled","budget_tokens":budget.min(max-1)});
        }
    }
    body.as_object_mut().unwrap().remove("system");
    let mut bytes = b"{\"system\":[{\"type\":\"text\",\"text\":".to_vec();
    bytes.extend(
        serde_json::to_vec(&billing(&first))
            .map_err(|_| invalid("billing serialization failed"))?,
    );
    bytes.extend(b"},{\"type\":\"text\",\"text\":\"You are a Claude agent, built on Anthropic's Claude Agent SDK.\",\"cache_control\":{\"type\":\"ephemeral\"}}],");
    let rest = serde_json::to_vec(&body).map_err(|_| invalid("request serialization failed"))?;
    bytes.extend_from_slice(&rest[1..]);
    let offset = bytes
        .windows(9)
        .position(|b| b == b"cch=00000")
        .ok_or_else(|| invalid("missing billing marker"))?
        + 4;
    let hash = xxhash_rust::xxh64::xxh64(&bytes, SEED) & 0xfffff;
    bytes[offset..offset + 5].copy_from_slice(format!("{hash:05x}").as_bytes());
    Ok(bytes)
}
fn normalize_id(id: &str) -> String {
    format!("toolu_{:x}", Sha256::digest(id.as_bytes()))
}
fn invalid(message: &str) -> LlmError {
    LlmError::Parse(format!("Anthropic subscription: {message}"))
}
fn replay_blocks(data: &Value) -> Option<Vec<Value>> {
    let blocks = data.get("blocks")?.as_array()?;
    if blocks.is_empty()
        || !blocks.iter().all(|b| match b["type"].as_str() {
            Some("thinking") => {
                b["thinking"].is_string() && b["signature"].as_str().is_some_and(|s| !s.is_empty())
            }
            Some("redacted_thinking") => b["data"].as_str().is_some_and(|s| !s.is_empty()),
            Some("text") => b["text"].is_string(),
            Some("tool_use") => {
                b["id"].is_string() && b["name"].is_string() && b["input"].is_object()
            }
            _ => false,
        })
    {
        return None;
    }
    Some(blocks.clone())
}

/// Wrapper retains strict generic tool JSON/terminal checks and additionally
/// captures full assistant block order for signed-thinking continuation.
pub(crate) struct SubscriptionParser {
    inner: AnthropicParser,
    blocks: BTreeMap<u64, Value>,
    names: HashMap<String, String>,
    started: bool,
    open: Option<u64>,
}
impl SubscriptionParser {
    pub(crate) fn new(req: &CompletionRequest) -> Self {
        let mut names: HashMap<_, _> = req
            .tools
            .iter()
            .map(|t| (format!("_{}", t.name), t.name.clone()))
            .collect();
        // Prompt guidance and code-mode catalogues use original names. Accept
        // those too when the model omits our wire prefix, but only for tools
        // offered in this request. Exact wire names take precedence if a tool
        // already starts with `_`; never strip prefixes from arbitrary names.
        for tool in &req.tools {
            names
                .entry(tool.name.clone())
                .or_insert_with(|| tool.name.clone());
        }
        Self {
            inner: AnthropicParser::new(),
            blocks: BTreeMap::new(),
            started: false,
            open: None,
            names,
        }
    }
}
impl StreamParser for SubscriptionParser {
    fn parse_event(&mut self, event: &crate::sse::SseEvent) -> Result<Vec<StreamEvent>, LlmError> {
        let v: Value =
            serde_json::from_str(&event.data).map_err(|_| invalid("invalid SSE JSON"))?;
        if self.inner.is_done() {
            return Err(invalid("event after message_stop"));
        }
        let kind = v["type"]
            .as_str()
            .ok_or_else(|| invalid("missing event type"))?;
        if !self.started && !matches!(kind, "message_start" | "ping" | "error") {
            return Err(invalid("event before message_start"));
        }
        let index = if kind.starts_with("content_block_") {
            v["index"]
                .as_u64()
                .ok_or_else(|| invalid("missing block index"))?
        } else {
            0
        };
        match Some(kind) {
            Some("message_start") => {
                if self.started {
                    return Err(invalid("duplicate message_start"));
                }
                self.started = true;
            }
            Some("content_block_start") => {
                if self.open.is_some() || self.blocks.contains_key(&index) {
                    return Err(invalid("overlapping or duplicate block"));
                }
                let block = &v["content_block"];
                // Report the failing field, not the block payload: thinking,
                // tool input, and unknown names can contain private data.
                let problem = match block["type"].as_str() {
                    Some("text") if !block["text"].is_string() => {
                        Some("text block requires string text")
                    }
                    Some("thinking") if !block["thinking"].is_string() => {
                        Some("thinking block requires string thinking")
                    }
                    Some("redacted_thinking")
                        if !block["data"].as_str().is_some_and(|s| !s.is_empty()) =>
                    {
                        Some("redacted thinking block requires nonempty data")
                    }
                    Some("tool_use") => {
                        if !block["id"].as_str().is_some_and(|s| !s.is_empty()) {
                            Some("tool block requires a nonempty id")
                        } else if !block["name"]
                            .as_str()
                            .is_some_and(|s| self.names.contains_key(s))
                        {
                            Some("tool block name is not registered")
                        } else if !block["input"].is_object() {
                            Some("tool block input must be an object")
                        } else {
                            None
                        }
                    }
                    Some("text" | "thinking" | "redacted_thinking") => None,
                    _ => Some("unsupported or missing block type"),
                };
                if let Some(problem) = problem {
                    return Err(invalid(&format!(
                        "invalid content block at index {index}: {problem}"
                    )));
                }
                self.open = Some(index);
                self.blocks.insert(index, block.clone());
            }
            Some("content_block_delta") => {
                if self.open != Some(index) {
                    return Err(invalid("delta outside open block"));
                }
                let block_type = self.blocks[&index]["type"].as_str().unwrap_or_default();
                let field = match (block_type, v["delta"]["type"].as_str()) {
                    ("text", Some("text_delta")) => "text",
                    ("thinking", Some("thinking_delta")) => "thinking",
                    ("thinking", Some("signature_delta")) => "signature",
                    ("tool_use", Some("input_json_delta")) => "partial_json",
                    _ => return Err(invalid("delta does not match block")),
                };
                if !v["delta"][field].is_string() {
                    return Err(invalid("invalid delta value"));
                }
                if let Some(b) = self.blocks.get_mut(&index) {
                    let (key, source) = match v["delta"]["type"].as_str() {
                        Some("text_delta") => ("text", "text"),
                        Some("thinking_delta") => ("thinking", "thinking"),
                        Some("signature_delta") => ("signature", "signature"),
                        _ => ("", ""),
                    };
                    if !key.is_empty() {
                        let mut text = b[key].as_str().unwrap_or_default().to_owned();
                        text.push_str(v["delta"][source].as_str().unwrap_or_default());
                        b[key] = json!(text);
                    }
                }
            }
            Some("content_block_stop") => {
                if self.open != Some(index) {
                    return Err(invalid("stop outside open block"));
                }
                self.open = None;
            }
            Some("message_delta" | "message_stop") if self.open.is_some() => {
                return Err(invalid("message ended with an open block"));
            }
            Some("message_delta" | "message_stop" | "ping" | "error") => {}
            _ => return Err(invalid("unknown event type")),
        }
        let mut events = self.inner.parse_event(event)?;
        if events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallComplete(c) if !c.arguments.is_object()))
        {
            return Err(invalid("tool arguments must be an object"));
        }
        for e in &mut events {
            if let StreamEvent::ToolCallComplete(call) = e {
                call.name = self
                    .names
                    .get(&call.name)
                    .cloned()
                    .ok_or_else(|| invalid("response tool name is not registered"))?;
                for block in self.blocks.values_mut() {
                    if block["type"] == "tool_use" && block["id"].as_str() == Some(&call.id) {
                        block["name"] = json!(call.name);
                        block["input"] = call.arguments.clone();
                    }
                }
            }
        }
        if self.inner.is_done() {
            let data = json!({"blocks":self.blocks.values().collect::<Vec<_>>()});
            // Unsigned thinking is never reclassified as signed state.
            if replay_blocks(&data).is_some() {
                events.insert(
                    0,
                    StreamEvent::OpaqueState {
                        provider: TAG.into(),
                        data,
                    },
                );
            }
        }
        Ok(events)
    }
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        self.inner.finish()
    }
}
