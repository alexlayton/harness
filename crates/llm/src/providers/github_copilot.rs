//! GitHub Copilot provider.
//!
//! Copilot exposes several request dialects behind one token and a
//! credential-specific proxy endpoint.  This module deliberately owns the
//! routing/catalog/header policy instead of changing the existing providers.

use crate::dialects::anthropic::AnthropicMessagesClient;
use crate::dialects::openai_chat::OpenAiChatClient;
use crate::dialects::openai_responses::OpenAiResponsesClient;
use crate::retry::with_retry;
use crate::{
    CompletionRequest, EventStream, LlmError, Message, ModelInfo, Provider, RetryCallback, Role,
};
use auth::{
    COPILOT_EDITOR_PLUGIN_VERSION, COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID,
    COPILOT_USER_AGENT, CopilotAuth,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashSet;
use std::sync::Arc;

pub const PROVIDER_NAME: &str = "github-copilot";

/// Request dialect selected by the static Copilot catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    AnthropicMessages,
    OpenAiChatCompletions,
    OpenAiResponses,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopilotModel {
    pub id: &'static str,
    pub name: &'static str,
    pub dialect: Dialect,
    pub context_length: u64,
    pub reasoning_supported: bool,
    pub max_tokens: u32,
}

/// Static metadata from Pi's Copilot catalog.  The remote policy endpoint
/// tells us which IDs an account can use, but not reliably which wire dialect
/// or limits belong to each model.
pub const COPILOT_MODELS: &[CopilotModel] = &[
    CopilotModel {
        id: "claude-haiku-4.5",
        name: "Claude Haiku 4.5",
        dialect: Dialect::AnthropicMessages,
        context_length: 200_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "claude-opus-4.5",
        name: "Claude Opus 4.5",
        dialect: Dialect::AnthropicMessages,
        context_length: 200_000,
        reasoning_supported: true,
        max_tokens: 32_000,
    },
    CopilotModel {
        id: "claude-opus-4.6",
        name: "Claude Opus 4.6",
        dialect: Dialect::AnthropicMessages,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 32_000,
    },
    CopilotModel {
        id: "claude-opus-4.7",
        name: "Claude Opus 4.7",
        dialect: Dialect::AnthropicMessages,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 32_000,
    },
    CopilotModel {
        id: "claude-opus-4.8",
        name: "Claude Opus 4.8",
        dialect: Dialect::AnthropicMessages,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "claude-opus-5",
        name: "Claude Opus 5",
        dialect: Dialect::AnthropicMessages,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "claude-sonnet-4",
        name: "Claude Sonnet 4",
        dialect: Dialect::AnthropicMessages,
        context_length: 216_000,
        reasoning_supported: true,
        max_tokens: 16_000,
    },
    CopilotModel {
        id: "claude-sonnet-4.5",
        name: "Claude Sonnet 4.5",
        dialect: Dialect::AnthropicMessages,
        context_length: 200_000,
        reasoning_supported: true,
        max_tokens: 32_000,
    },
    CopilotModel {
        id: "claude-sonnet-4.6",
        name: "Claude Sonnet 4.6",
        dialect: Dialect::AnthropicMessages,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 32_000,
    },
    CopilotModel {
        id: "claude-sonnet-5",
        name: "Claude Sonnet 5",
        dialect: Dialect::AnthropicMessages,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "claude-fable-5",
        name: "Claude Fable 5",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gemini-3.1-pro-preview",
        name: "Gemini 3.1 Pro Preview",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "gemini-3.5-flash",
        name: "Gemini 3.5 Flash",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 200_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "gemini-3.6-flash",
        name: "Gemini 3.6 Flash",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "gpt-4.1",
        name: "GPT-4.1",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 128_000,
        reasoning_supported: false,
        max_tokens: 16_384,
    },
    CopilotModel {
        id: "kimi-k2.7-code",
        name: "Kimi K2.7 Code",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 256_000,
        reasoning_supported: true,
        max_tokens: 32_000,
    },
    CopilotModel {
        id: "kimi-k3",
        name: "Kimi K3",
        dialect: Dialect::OpenAiChatCompletions,
        context_length: 1_048_576,
        reasoning_supported: true,
        max_tokens: 131_072,
    },
    CopilotModel {
        id: "gpt-5-mini",
        name: "GPT-5 Mini",
        dialect: Dialect::OpenAiResponses,
        context_length: 264_000,
        reasoning_supported: true,
        max_tokens: 64_000,
    },
    CopilotModel {
        id: "gpt-5.2",
        name: "GPT-5.2",
        dialect: Dialect::OpenAiResponses,
        context_length: 400_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.2-codex",
        name: "GPT-5.2 Codex",
        dialect: Dialect::OpenAiResponses,
        context_length: 400_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.3-codex",
        name: "GPT-5.3 Codex",
        dialect: Dialect::OpenAiResponses,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.4",
        name: "GPT-5.4",
        dialect: Dialect::OpenAiResponses,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.4-mini",
        name: "GPT-5.4 mini",
        dialect: Dialect::OpenAiResponses,
        context_length: 400_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.4-nano",
        name: "GPT-5.4 nano",
        dialect: Dialect::OpenAiResponses,
        context_length: 400_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.5",
        name: "GPT-5.5",
        dialect: Dialect::OpenAiResponses,
        context_length: 1_000_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.6-luna",
        name: "GPT-5.6 Luna",
        dialect: Dialect::OpenAiResponses,
        context_length: 1_050_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.6-sol",
        name: "GPT-5.6 Sol",
        dialect: Dialect::OpenAiResponses,
        context_length: 1_050_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "gpt-5.6-terra",
        name: "GPT-5.6 Terra",
        dialect: Dialect::OpenAiResponses,
        context_length: 1_050_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "grok-4.5",
        name: "Grok 4.5",
        dialect: Dialect::OpenAiResponses,
        context_length: 500_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
    CopilotModel {
        id: "mai-code-1-flash-picker",
        name: "MAI-Code-1-Flash",
        dialect: Dialect::OpenAiResponses,
        context_length: 256_000,
        reasoning_supported: true,
        max_tokens: 128_000,
    },
];

/// Compatibility aliases for callers that refer to this as a static catalog.
pub const STATIC_MODELS: &[CopilotModel] = COPILOT_MODELS;
pub const MODEL_CATALOG: &[CopilotModel] = COPILOT_MODELS;

pub fn model_metadata(model: &str) -> Option<&'static CopilotModel> {
    COPILOT_MODELS.iter().find(|entry| entry.id == model)
}

pub fn dialect_for_model(model: &str) -> Option<Dialect> {
    model_metadata(model).map(|entry| entry.dialect)
}

pub fn known_models() -> Vec<ModelInfo> {
    models_for_available_ids(COPILOT_MODELS.iter().map(|model| model.id))
}

/// Merge a dynamic account allow-list with static routing metadata.  This
/// pure helper is also useful to callers that already fetched `/models`.
pub fn models_for_available_ids<I, S>(available_ids: I) -> Vec<ModelInfo>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let available = available_ids
        .into_iter()
        .map(|id| id.as_ref().to_owned())
        .collect::<HashSet<_>>();
    COPILOT_MODELS
        .iter()
        .filter(|model| available.contains(model.id))
        .map(|model| ModelInfo {
            id: model.id.into(),
            name: Some(model.name.into()),
            context_length: Some(model.context_length),
        })
        .collect()
}

/// Pick a sensible default model for a signed-in account: the first entry,
/// in the account's own ordering, that this build also knows how to route.
/// Returns `None` when the list is empty or contains nothing from the
/// catalog, leaving the caller on the static default.
pub fn default_model_for_available_ids<I, S>(available_ids: I) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    available_ids
        .into_iter()
        .map(|id| id.as_ref().to_owned())
        .find(|id| model_metadata(id).is_some())
}

/// Plan-aware default: free Copilot SKUs gate most premium models behind
/// billing even when the account's model list advertises them, so prefer a
/// model every plan can serve.  Other plans keep the account's own order.
pub fn default_model_for(sku: Option<&str>, available_ids: &[String]) -> Option<String> {
    if is_free_sku(sku) && available_ids.iter().any(|id| id == "gpt-4.1") {
        return Some("gpt-4.1".to_owned());
    }
    default_model_for_available_ids(available_ids.iter())
}

fn is_free_sku(sku: Option<&str>) -> bool {
    sku.is_some_and(|sku| sku.starts_with("free"))
}

fn usable_model_suggestions(available: &[String], sku: Option<&str>) -> Vec<String> {
    if is_free_sku(sku) && available.iter().any(|id| id == "gpt-4.1") {
        return vec!["gpt-4.1".to_owned()];
    }
    models_for_available_ids(available.iter())
        .into_iter()
        .map(|model| model.id)
        .take(3)
        .collect()
}

fn model_gate_error(
    model: &str,
    available: &[String],
    sku: Option<&str>,
    reason: &str,
) -> LlmError {
    let suggestions = usable_model_suggestions(available, sku);
    let hint = if suggestions.is_empty() {
        "run /models to list usable models".to_owned()
    } else {
        format!("try {} (run /models to list all)", suggestions.join(", "))
    };
    LlmError::Auth(format!("GitHub Copilot model `{model}` {reason}; {hint}"))
}

/// Pre-flight the requested model against the account's available list.
/// An empty list (policy endpoint unavailable, or a credential saved before
/// the field existed) defers to the server's own error.  A miss is permanent
/// for this account, so it surfaces as an actionable error instead of the
/// raw 404 the API would return.
fn entitlement_error_for(model: &str, available: &[String], sku: Option<&str>) -> Option<LlmError> {
    if available.is_empty() || available.iter().any(|id| id == model) {
        return None;
    }
    Some(model_gate_error(
        model,
        available,
        sku,
        "is not in this account's available model list",
    ))
}

/// Copilot answers plan-gated models with a bare 400 `model_not_supported`,
/// which says nothing about what to do next.  Translate it into the same
/// actionable shape as the pre-flight check.  Any other error passes through.
fn plan_gate_error(
    error: LlmError,
    model: &str,
    available: &[String],
    sku: Option<&str>,
) -> LlmError {
    if let LlmError::Http {
        status: 400,
        ref body,
    } = error
        && body.contains("model_not_supported")
    {
        return model_gate_error(model, available, sku, "is not usable on this Copilot plan");
    }
    error
}

/// Build the headers shared by Copilot's three LLM dialects.
pub fn copilot_headers(messages: &[Message]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    insert_static_header(&mut headers, "user-agent", COPILOT_USER_AGENT);
    insert_static_header(&mut headers, "editor-version", COPILOT_EDITOR_VERSION);
    insert_static_header(
        &mut headers,
        "editor-plugin-version",
        COPILOT_EDITOR_PLUGIN_VERSION,
    );
    insert_static_header(
        &mut headers,
        "copilot-integration-id",
        COPILOT_INTEGRATION_ID,
    );
    insert_static_header(&mut headers, "openai-intent", "conversation-edits");
    insert_static_header(&mut headers, "x-initiator", copilot_initiator(messages));
    headers
}

pub fn copilot_initiator(messages: &[Message]) -> &'static str {
    match messages.last().map(|message| &message.role) {
        Some(Role::User) | None => "user",
        Some(_) => "agent",
    }
}

fn insert_static_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

#[derive(Clone)]
pub struct GithubCopilotProvider {
    pub auth: Arc<CopilotAuth>,
}

impl std::fmt::Debug for GithubCopilotProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GithubCopilotProvider")
            .field("auth", &self.auth)
            .finish()
    }
}

impl GithubCopilotProvider {
    pub fn new(auth: Arc<CopilotAuth>) -> Self {
        Self { auth }
    }

    pub fn from_default() -> Result<Self, auth::AuthError> {
        Ok(Self::new(Arc::new(CopilotAuth::from_default()?)))
    }

    pub fn model_metadata(model: &str) -> Option<&'static CopilotModel> {
        model_metadata(model)
    }

    pub fn dialect_for_model(model: &str) -> Option<Dialect> {
        dialect_for_model(model)
    }

    async fn stream_once(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        let model = model_metadata(&req.model).ok_or_else(|| {
            LlmError::Parse(format!(
                "GitHub Copilot model `{}` is not in the supported model catalog",
                req.model
            ))
        })?;
        let credential = self.auth.ensure_valid().await.map_err(auth_error)?;
        let sku = auth::sku_from_proxy_token(&credential.access).map(str::to_owned);
        let sku = sku.as_deref();
        if let Some(error) = entitlement_error_for(&req.model, &credential.available_model_ids, sku)
        {
            return Err(error);
        }
        let base_url = self.auth.base_url_for(&credential);
        let headers = copilot_headers(&req.messages);
        let mut request = req.clone();
        if request.max_tokens.is_none() {
            request.max_tokens = Some(model.max_tokens);
        }
        if !model.reasoning_supported {
            request.reasoning = false;
        }
        let access = credential.access;
        let result = match model.dialect {
            // Copilot's Anthropic-compatible endpoint is `/v1/messages`
            // (per the per-model `supported_endpoints` metadata), so the
            // Anthropic dialect client gets the base with the `/v1` prefix.
            Dialect::AnthropicMessages => {
                AnthropicMessagesClient::with_bearer_only_headers(
                    format!("{base_url}/v1"),
                    access.clone(),
                    headers,
                )
                .stream(&request)
                .await
            }
            Dialect::OpenAiChatCompletions => {
                OpenAiChatClient::with_headers(base_url, access.clone(), headers)
                    .stream(&request)
                    .await
            }
            Dialect::OpenAiResponses => {
                OpenAiResponsesClient::with_headers(base_url, access.clone(), headers)
                    .stream(&request)
                    .await
            }
        };
        result
            .map_err(|error| {
                plan_gate_error(error, &req.model, &credential.available_model_ids, sku)
            })
            .map_err(|error| redact_error(error, &access))
    }

    pub async fn stream_with_callback(
        &self,
        req: &CompletionRequest,
        on_retry: RetryCallback,
    ) -> Result<EventStream, LlmError> {
        let callback = on_retry.clone();
        with_retry(
            || async { self.stream_once(req).await },
            move |attempt, error| {
                tracing::warn!(attempt, error = %error, "retrying GitHub Copilot request");
                callback(attempt, error);
            },
        )
        .await
    }

    /// Fetch and filter the account's dynamic model policy list, then merge it
    /// with the static catalog.  No unknown remote model is routed implicitly.
    pub async fn list_models_direct(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let available = self
            .auth
            .refresh_available_model_ids()
            .await
            .map_err(auth_error)?;
        Ok(models_for_available_ids(available))
    }
}

#[async_trait::async_trait]
impl Provider for GithubCopilotProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError> {
        self.stream_with_callback(req, Arc::new(|_, _| {})).await
    }

    async fn stream_with_retry(
        &self,
        req: &CompletionRequest,
        on_retry: RetryCallback,
    ) -> Result<EventStream, LlmError> {
        self.stream_with_callback(req, on_retry).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_direct().await
    }
}

fn auth_error(error: auth::AuthError) -> LlmError {
    LlmError::Auth(error.to_string())
}

fn redact_error(error: LlmError, secret: &str) -> LlmError {
    if secret.is_empty() {
        return error;
    }
    match error {
        LlmError::Http { status, body } => LlmError::Http {
            status,
            body: body.replace(secret, "[redacted]"),
        },
        LlmError::Stream(message) => LlmError::Stream(message.replace(secret, "[redacted]")),
        LlmError::Parse(message) => LlmError::Parse(message.replace(secret, "[redacted]")),
        LlmError::Auth(message) => LlmError::Auth(message.replace(secret, "[redacted]")),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use auth::{AuthStore, CopilotAuth};
    use tempfile::tempdir;

    #[test]
    fn every_representative_model_has_explicit_dialect() {
        assert_eq!(
            dialect_for_model("claude-sonnet-4.6"),
            Some(Dialect::AnthropicMessages)
        );
        assert_eq!(dialect_for_model("gpt-5.4"), Some(Dialect::OpenAiResponses));
        assert_eq!(
            dialect_for_model("gpt-5.4-mini"),
            Some(Dialect::OpenAiResponses)
        );
        assert_eq!(
            dialect_for_model("gpt-5.3-codex"),
            Some(Dialect::OpenAiResponses)
        );
        assert_eq!(
            dialect_for_model("grok-4.5"),
            Some(Dialect::OpenAiResponses)
        );
        assert_eq!(
            dialect_for_model("gemini-3.1-pro-preview"),
            Some(Dialect::OpenAiChatCompletions)
        );
        assert_eq!(
            dialect_for_model("kimi-k3"),
            Some(Dialect::OpenAiChatCompletions)
        );
        assert_eq!(dialect_for_model("unknown"), None);
    }

    #[test]
    fn dynamic_headers_mark_followups_as_agent_initiated() {
        let user = vec![Message::user("hello")];
        let user_headers = copilot_headers(&user);
        assert_eq!(user_headers.get("x-initiator").unwrap(), "user");
        let followup = vec![Message::assistant(vec![crate::Content::Text(
            "answer".into(),
        )])];
        let followup_headers = copilot_headers(&followup);
        assert_eq!(followup_headers.get("x-initiator").unwrap(), "agent");
        assert_eq!(
            followup_headers.get("openai-intent").unwrap(),
            "conversation-edits"
        );
    }

    #[test]
    fn dynamic_model_merge_discards_unknown_remote_ids() {
        let models = models_for_available_ids(["gpt-5.4", "unknown", "kimi-k3"]);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "kimi-k3");
        assert_eq!(models[1].id, "gpt-5.4");
    }

    #[test]
    fn default_model_follows_account_order_and_skips_unknown_ids() {
        assert_eq!(
            default_model_for_available_ids([
                "gpt-5.4-mini-free-auto",
                "gpt-5.6-luna",
                "claude-haiku-4.5",
            ]),
            Some("gpt-5.6-luna".into())
        );
        // Nothing routable: keep the static default.
        assert_eq!(
            default_model_for_available_ids(["unknown", "also-unknown"]),
            None
        );
        assert_eq!(default_model_for_available_ids(Vec::<String>::new()), None);
    }

    #[test]
    fn free_sku_defaults_to_a_plan_servable_model() {
        let available = vec!["gpt-5.6-luna".into(), "gpt-4.1".into()];
        assert_eq!(
            default_model_for(Some("free_limited_copilot"), &available),
            Some("gpt-4.1".into())
        );
        // Paid plans keep the account's own ordering.
        assert_eq!(
            default_model_for(Some("copilot_business"), &available),
            Some("gpt-5.6-luna".into())
        );
        assert_eq!(
            default_model_for(None, &available),
            Some("gpt-5.6-luna".into())
        );
        // Free SKU whose account does not list gpt-4.1 falls through.
        let no_gpt41 = vec!["gpt-5.6-luna".into()];
        assert_eq!(
            default_model_for(Some("free_limited_copilot"), &no_gpt41),
            Some("gpt-5.6-luna".into())
        );
    }

    #[test]
    fn entitlement_check_flags_unlisted_models_with_a_hint() {
        let available = vec!["gpt-5.6-luna".to_owned(), "claude-haiku-4.5".to_owned()];
        let error = entitlement_error_for("gpt-5.4", &available, None).unwrap();
        assert!(matches!(error, LlmError::Auth(_)));
        let message = error.to_string();
        assert!(message.contains("`gpt-5.4`"), "{message}");
        assert!(message.contains("claude-haiku-4.5"), "{message}");
        assert!(message.contains("/models"), "{message}");

        // Entitled models pass the pre-flight.
        assert!(entitlement_error_for("gpt-5.6-luna", &available, None).is_none());
        // An empty list cannot judge; defer to the server's own error.
        assert!(entitlement_error_for("gpt-5.4", &[], None).is_none());
    }

    #[test]
    fn plan_gated_models_map_to_an_actionable_error() {
        let available = vec!["gpt-5.6-luna".to_owned(), "gpt-4.1".to_owned()];
        let error = plan_gate_error(
            LlmError::Http {
                status: 400,
                body: "{\"error\":{\"code\":\"model_not_supported\"}}".into(),
            },
            "claude-haiku-4.5",
            &available,
            Some("free_limited_copilot"),
        );
        let message = error.to_string();
        assert!(
            message.contains("not usable on this Copilot plan"),
            "{message}"
        );
        assert!(message.contains("try gpt-4.1"), "{message}");

        // Unrelated 400s and other statuses pass through untouched.
        let other = || LlmError::Http {
            status: 400,
            body: "bad json".into(),
        };
        assert!(matches!(
            plan_gate_error(other(), "m", &available, None),
            LlmError::Http { status: 400, .. }
        ));
        let not_found = LlmError::Http {
            status: 404,
            body: "model_not_supported".into(),
        };
        assert!(matches!(
            plan_gate_error(not_found, "m", &available, None),
            LlmError::Http { status: 404, .. }
        ));
    }

    #[test]
    fn provider_can_be_constructed_without_a_credential() {
        let directory = tempdir().unwrap();
        let auth =
            Arc::new(CopilotAuth::new(AuthStore::new(directory.path().join("auth.json"))).unwrap());
        let provider = GithubCopilotProvider::new(auth);
        assert_eq!(provider.name(), "github-copilot");
    }

    #[test]
    fn provider_errors_redact_access_tokens() {
        let error = redact_error(
            LlmError::Http {
                status: 401,
                body: "token=access-secret".into(),
            },
            "access-secret",
        );
        assert_eq!(error.to_string(), "http 401: token=[redacted]");
    }

    #[tokio::test]
    async fn missing_auth_is_an_actionable_llm_error() {
        let directory = tempdir().unwrap();
        let auth =
            Arc::new(CopilotAuth::new(AuthStore::new(directory.path().join("auth.json"))).unwrap());
        let provider = GithubCopilotProvider::new(auth);
        let request = crate::CompletionRequest {
            model: "gpt-5.4".into(),
            system: None,
            messages: vec![Message::user("hello")],
            tools: Vec::new(),
            max_tokens: None,
            temperature: None,
            reasoning: true,
        };
        let error = match provider.stream(&request).await {
            Ok(_) => panic!("missing auth unexpectedly produced a stream"),
            Err(error) => error,
        };
        assert!(matches!(error, LlmError::Auth(message) if message.contains("run /auth")));
    }
}
