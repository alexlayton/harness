//! GitHub Copilot's device OAuth flow and short-lived proxy credentials.
//!
//! GitHub Copilot is not a normal API-key provider.  A GitHub OAuth token is
//! exchanged for a short-lived Copilot token, and the token's `proxy-ep` value
//! selects the API host.  Keeping those details here makes the rest of Harness
//! independent from this unstable client protocol.

use crate::device_code::{
    AuthEvent, DeviceCode, PollResult, parse_device_code, parse_u64, required_string,
};
use crate::error::{AuthError, Result};
use crate::storage::{AuthStore, CopilotCredential};
use reqwest::header::{
    ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, USER_AGENT,
};
use reqwest::{Client, Url};
use serde_json::Value;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// The public device-flow client ID used by GitHub's Copilot editors.
pub const GITHUB_DEVICE_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

pub const COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub const COPILOT_EDITOR_VERSION: &str = "vscode/1.107.0";
pub const COPILOT_EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
pub const COPILOT_API_VERSION: &str = "2026-06-01";

/// The known model IDs used for best-effort policy enablement after login.
/// Routing metadata lives in the LLM crate, but policy enablement belongs to
/// authentication and therefore intentionally keeps only IDs here.
pub const KNOWN_MODEL_IDS: &[&str] = &[
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "claude-fable-5",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gpt-4.1",
    "kimi-k2.7-code",
    "kimi-k3",
    "gpt-5-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "grok-4.5",
    "mai-code-1-flash-picker",
];

/// Normalize a GitHub.com or GitHub Enterprise domain.  The stored value is a
/// hostname, not a URL, so it can safely be used to construct the three
/// endpoint families.
pub fn normalize_domain(input: Option<&str>) -> Result<Option<String>> {
    let Some(input) = input else {
        return Ok(None);
    };
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let candidate = if trimmed.contains("://") {
        trimmed.to_owned()
    } else {
        format!("https://{trimmed}")
    };
    let url = Url::parse(&candidate).map_err(|_| AuthError::InvalidDomain(trimmed.to_owned()))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || url.username() != ""
        || url.password().is_some()
        || url.port().is_some()
        || (url.path() != "" && url.path() != "/")
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(AuthError::InvalidDomain(trimmed.to_owned()));
    }
    let host = url
        .host_str()
        .ok_or_else(|| AuthError::InvalidDomain(trimmed.to_owned()))?
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if host.is_empty() || host.chars().any(|character| character.is_whitespace()) {
        return Err(AuthError::InvalidDomain(trimmed.to_owned()));
    }
    Ok(Some(host))
}

/// Endpoint set used by the device flow.  It is public so tests and enterprise
/// integrations can inject a local HTTP fixture without changing production
/// endpoint construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopilotEndpoints {
    pub device_code_url: String,
    pub access_token_url: String,
    pub copilot_token_url: String,
}

impl CopilotEndpoints {
    pub fn for_domain(domain: Option<&str>) -> Result<Self> {
        let domain = normalize_domain(domain)?.unwrap_or_else(|| "github.com".into());
        let api_domain = if domain == "github.com" {
            "api.github.com".to_owned()
        } else {
            format!("api.{domain}")
        };
        Ok(Self {
            device_code_url: format!("https://{domain}/login/device/code"),
            access_token_url: format!("https://{domain}/login/oauth/access_token"),
            copilot_token_url: format!("https://{api_domain}/copilot_internal/v2/token"),
        })
    }
}

#[derive(Clone)]
pub struct GithubCopilotClient {
    http: Client,
    pub endpoints: CopilotEndpoints,
    api_base_url: Option<String>,
}

impl fmt::Debug for GithubCopilotClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GithubCopilotClient")
            .field("endpoints", &self.endpoints)
            .field("api_base_url", &self.api_base_url)
            .finish()
    }
}

impl GithubCopilotClient {
    pub fn new() -> Result<Self> {
        Self::from_endpoints(CopilotEndpoints::for_domain(None)?)
    }

    pub fn with_endpoints(endpoints: CopilotEndpoints) -> Result<Self> {
        Self::from_endpoints(endpoints)
    }

    fn from_endpoints(endpoints: CopilotEndpoints) -> Result<Self> {
        // The auth client shares the process-wide rustls crypto provider
        // installed by `llm`'s HTTP setup; with `rustls-no-provider` a Client
        // built before that install would panic, so mirror the install here.
        // Idempotent: rustls rejects double installs and we ignore the error.
        let _ = rustls::crypto::ring::default_provider().install_default();
        Ok(Self {
            http: Client::builder()
                .user_agent(COPILOT_USER_AGENT)
                .build()
                .map_err(|error| AuthError::Network {
                    endpoint: "GitHub".into(),
                    message: error.to_string(),
                })?,
            endpoints,
            api_base_url: None,
        })
    }

    /// Override the API base used for `/models` and policy requests.  This is
    /// primarily useful for local fixtures and enterprise gateways; normal
    /// callers should let the Copilot token's `proxy-ep` choose the host.
    pub fn with_api_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.api_base_url = Some(base_url.into().trim_end_matches('/').to_owned());
        self
    }

    pub fn for_domain(&self, domain: Option<&str>) -> Result<Self> {
        Ok(Self {
            http: self.http.clone(),
            endpoints: CopilotEndpoints::for_domain(domain)?,
            api_base_url: self.api_base_url.clone(),
        })
    }

    pub async fn start_device_flow(&self, cancel: &CancellationToken) -> Result<DeviceCode> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        headers.insert(USER_AGENT, HeaderValue::from_static(COPILOT_USER_AGENT));
        let request = self
            .http
            .post(&self.endpoints.device_code_url)
            .headers(headers)
            .form(&[
                ("client_id", GITHUB_DEVICE_CLIENT_ID),
                ("scope", "read:user"),
            ]);
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            response = request.send() => response.map_err(|error| network_error(&self.endpoints.device_code_url, error))?,
        };
        let status = response.status();
        if !status.is_success() {
            return Err(AuthError::Http {
                status: status.as_u16(),
                endpoint: endpoint_label(&self.endpoints.device_code_url),
            });
        }
        let value = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            value = response.json::<Value>() => value.map_err(|_| {
                AuthError::InvalidDeviceCode("response is not valid JSON".into())
            })?,
        };
        parse_device_code(&value)
    }

    /// Poll GitHub until a GitHub OAuth token is available.
    pub async fn poll_for_github_access_token(
        &self,
        device: &DeviceCode,
        cancel: &CancellationToken,
    ) -> Result<String> {
        let deadline = Instant::now() + Duration::from_secs(device.expires_in);
        let mut interval = device.interval.max(5);
        // RFC 8628 waits before the first poll.
        loop {
            if Instant::now() >= deadline {
                return Err(AuthError::DeviceCodeExpired);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let wait = Duration::from_secs(interval).min(remaining);
            tokio::select! {
                _ = cancel.cancelled() => return Err(AuthError::Cancelled),
                _ = tokio::time::sleep(wait) => {}
            }
            if Instant::now() >= deadline {
                return Err(AuthError::DeviceCodeExpired);
            }
            match self.poll_once(device, cancel).await? {
                PollResult::Complete { access_token } => return Ok(access_token),
                PollResult::Pending => {}
                PollResult::SlowDown {
                    interval: new_interval,
                } => {
                    interval = new_interval
                        .unwrap_or(interval.saturating_add(5))
                        .max(interval + 5);
                }
            }
        }
    }

    async fn poll_once(
        &self,
        device: &DeviceCode,
        cancel: &CancellationToken,
    ) -> Result<PollResult> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        headers.insert(USER_AGENT, HeaderValue::from_static(COPILOT_USER_AGENT));
        let request = self
            .http
            .post(&self.endpoints.access_token_url)
            .headers(headers)
            .form(&[
                ("client_id", GITHUB_DEVICE_CLIENT_ID),
                ("device_code", device.device_code.as_str()),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ]);
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            response = request.send() => response.map_err(|error| network_error(&self.endpoints.access_token_url, error))?,
        };
        let status = response.status();
        let value = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            value = response.json::<Value>() => value.map_err(|_| AuthError::DeviceFlowFailed("invalid token response".into()))?,
        };
        if !status.is_success() && value.get("error").is_none() {
            return Err(AuthError::Http {
                status: status.as_u16(),
                endpoint: endpoint_label(&self.endpoints.access_token_url),
            });
        }
        if let Some(access_token) = value.get("access_token").and_then(Value::as_str) {
            if access_token.is_empty() {
                return Err(AuthError::DeviceFlowFailed("empty access token".into()));
            }
            return Ok(PollResult::Complete {
                access_token: access_token.to_owned(),
            });
        }
        let Some(error) = value.get("error").and_then(Value::as_str) else {
            return Err(AuthError::DeviceFlowFailed("invalid token response".into()));
        };
        match error {
            "authorization_pending" => Ok(PollResult::Pending),
            "slow_down" => Ok(PollResult::SlowDown {
                interval: value
                    .get("interval")
                    .and_then(|value| parse_u64(value, "interval").ok()),
            }),
            "expired_token" => Err(AuthError::DeviceCodeExpired),
            _other => Err(AuthError::DeviceFlowFailed(
                "device authorization failed".into(),
            )),
        }
    }

    /// Exchange a GitHub OAuth token for a short-lived Copilot token.
    pub async fn exchange_copilot_token(
        &self,
        github_access_token: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<CopilotCredential> {
        self.fetch_copilot_token(github_access_token, enterprise_domain, cancel)
            .await
    }

    /// Refresh a Copilot token from a stored GitHub token.
    pub async fn refresh_copilot_token(
        &self,
        credential: &CopilotCredential,
        cancel: &CancellationToken,
    ) -> Result<CopilotCredential> {
        self.fetch_copilot_token(
            &credential.refresh,
            credential.enterprise_url.as_deref(),
            cancel,
        )
        .await
    }

    /// GET `/copilot_internal/v2/token` with a bearer token and parse the
    /// resulting Copilot credential.  The exchange and refresh flows differ
    /// only in which token is sent and what is threaded through as the
    /// re-exchange token.
    async fn fetch_copilot_token(
        &self,
        bearer_token: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<CopilotCredential> {
        let domain = normalize_domain(enterprise_domain)?;
        let endpoints = if domain.is_some() {
            self.for_domain(domain.as_deref())?.endpoints
        } else {
            self.endpoints.clone()
        };
        let mut headers = copilot_headers();
        headers.insert(
            AUTHORIZATION,
            bearer(bearer_token)
                .ok_or_else(|| AuthError::InvalidCredential("invalid GitHub token".into()))?,
        );
        let request = self.http.get(&endpoints.copilot_token_url).headers(headers);
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            response = request.send() => response.map_err(|error| network_error(&endpoints.copilot_token_url, error))?,
        };
        let status = response.status();
        if !status.is_success() {
            return Err(AuthError::Http {
                status: status.as_u16(),
                endpoint: endpoint_label(&endpoints.copilot_token_url),
            });
        }
        let value = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            value = response.json::<Value>() => value.map_err(|_| AuthError::DeviceFlowFailed("invalid Copilot token response".into()))?,
        };
        parse_copilot_token(&value, bearer_token, domain)
    }

    pub async fn fetch_available_model_ids(
        &self,
        copilot_token: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<Vec<String>> {
        let domain = normalize_domain(enterprise_domain)?;
        let base_url = self
            .api_base_url
            .clone()
            .unwrap_or_else(|| copilot_base_url(copilot_token, domain.as_deref()));
        let url = format!("{base_url}/models");
        let mut headers = copilot_headers();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            bearer(copilot_token)
                .ok_or_else(|| AuthError::InvalidCredential("invalid Copilot token".into()))?,
        );
        headers.insert(
            HeaderName::from_static("x-github-api-version"),
            HeaderValue::from_static(COPILOT_API_VERSION),
        );
        let request = self.http.get(&url).headers(headers);
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            response = request.send() => response.map_err(|error| network_error(&url, error))?,
        };
        let status = response.status();
        if !status.is_success() {
            return Err(AuthError::Http {
                status: status.as_u16(),
                endpoint: endpoint_label(&url),
            });
        }
        let value = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            value = response.json::<Value>() => value.map_err(|_| AuthError::DeviceFlowFailed("invalid Copilot models response".into()))?,
        };
        let fallback = base_url == "https://api.individual.githubcopilot.com";
        parse_available_model_ids_value(&value, fallback)
    }

    /// Enablement is best effort: some accounts reject policy writes even
    /// though already-enabled models work.  Cancellation remains fatal.
    pub async fn enable_known_models(
        &self,
        copilot_token: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        for model_id in KNOWN_MODEL_IDS {
            if cancel.is_cancelled() {
                return Err(AuthError::Cancelled);
            }
            match self
                .enable_model(copilot_token, model_id, enterprise_domain, cancel)
                .await
            {
                Ok(_) => {}
                Err(AuthError::Cancelled) => return Err(AuthError::Cancelled),
                Err(_) => {
                    // Policy enablement is advisory.  Accounts can reject
                    // individual models while still allowing the login and
                    // dynamic model endpoint to succeed.
                }
            }
        }
        Ok(())
    }

    async fn enable_model(
        &self,
        copilot_token: &str,
        model_id: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<bool> {
        if !model_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_'))
        {
            return Ok(false);
        }
        let domain = normalize_domain(enterprise_domain)?;
        let base_url = self
            .api_base_url
            .clone()
            .unwrap_or_else(|| copilot_base_url(copilot_token, domain.as_deref()));
        let url = format!("{base_url}/models/{model_id}/policy");
        let mut headers = copilot_headers();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            bearer(copilot_token)
                .ok_or_else(|| AuthError::InvalidCredential("invalid Copilot token".into()))?,
        );
        headers.insert(
            HeaderName::from_static("openai-intent"),
            HeaderValue::from_static("chat-policy"),
        );
        headers.insert(
            HeaderName::from_static("x-interaction-type"),
            HeaderValue::from_static("chat-policy"),
        );
        let request = self
            .http
            .post(&url)
            .headers(headers)
            .json(&serde_json::json!({"state": "enabled"}));
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            response = request.send() => response.map_err(|error| network_error(&url, error))?,
        };
        Ok(response.status().is_success())
    }

    /// Complete the entire device login, emitting only UI-neutral events.
    pub async fn login_with_events<F>(
        &self,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
        mut emit: F,
    ) -> Result<CopilotCredential>
    where
        F: FnMut(AuthEvent) + Send,
    {
        emit(AuthEvent::Started);
        let domain = normalize_domain(enterprise_domain)?;
        let client = if domain.is_some() {
            self.for_domain(domain.as_deref())?
        } else {
            self.clone()
        };
        let result = async {
            let device = client.start_device_flow(cancel).await?;
            emit(AuthEvent::DeviceCode {
                verification_url: device.verification_url.clone(),
                user_code: device.user_code.clone(),
                expires_in: device.expires_in,
                interval: device.interval,
            });
            let github_token = client.poll_for_github_access_token(&device, cancel).await?;
            emit(AuthEvent::Progress {
                message: "GitHub authorized; exchanging Copilot token...".into(),
            });
            let mut credential = client
                .exchange_copilot_token(&github_token, domain.as_deref(), cancel)
                .await?;
            emit(AuthEvent::Progress {
                message: "Enabling available Copilot models...".into(),
            });
            client
                .enable_known_models(&credential.access, domain.as_deref(), cancel)
                .await?;
            credential.available_model_ids = client
                .fetch_available_model_ids(&credential.access, domain.as_deref(), cancel)
                .await?;
            Ok::<_, AuthError>(credential)
        }
        .await;
        match result {
            Ok(credential) => {
                emit(AuthEvent::Finished);
                Ok(credential)
            }
            Err(error) => {
                emit(AuthEvent::Failed {
                    message: error.to_string(),
                });
                Err(error)
            }
        }
    }
}

/// Parse the token response returned by `/copilot_internal/v2/token`.
pub fn parse_copilot_token(
    value: &Value,
    refresh_token: &str,
    enterprise_domain: Option<String>,
) -> Result<CopilotCredential> {
    let object = value.as_object().ok_or_else(|| {
        AuthError::InvalidCredential("Copilot token response is not an object".into())
    })?;
    let access = required_string(object.get("token"), "token")?;
    let expires_at = object.get("expires_at").ok_or_else(|| {
        AuthError::InvalidCredential("Copilot token response is missing expires_at".into())
    })?;
    let expires_seconds = parse_u64(expires_at, "expires_at")?;
    if expires_seconds == 0 {
        return Err(AuthError::InvalidCredential(
            "Copilot token expires_at is zero".into(),
        ));
    }
    let expires = expires_seconds
        .saturating_mul(1_000)
        .saturating_sub(5 * 60 * 1_000);
    Ok(CopilotCredential::new(
        access,
        refresh_token,
        expires,
        enterprise_domain,
        Vec::new(),
    ))
}

/// Derive the API host from Copilot's semicolon-delimited token metadata.
pub fn base_url_from_proxy_token(token: &str) -> Option<String> {
    let proxy = token.split(';').find_map(|part| {
        part.trim()
            .strip_prefix("proxy-ep=")
            .map(str::trim)
            .filter(|value| !value.is_empty())
    })?;
    let (scheme, host) = if let Ok(url) = Url::parse(proxy) {
        let host = url.host_str()?.to_owned();
        let host = url
            .port()
            .map(|port| format!("{host}:{port}"))
            .unwrap_or(host);
        (url.scheme().to_owned(), host)
    } else {
        ("https".to_owned(), proxy.trim_end_matches('/').to_owned())
    };
    let valid_port = host
        .rsplit_once(':')
        .is_none_or(|(name, port)| !name.contains(':') && port.parse::<u16>().is_ok());
    if host.is_empty()
        || host.contains(['/', '?', '#', '@'])
        || !valid_port
        || host.chars().any(|character| character.is_whitespace())
    {
        return None;
    }
    let api_host = host
        .strip_prefix("proxy.")
        .map(|rest| format!("api.{rest}"))
        .unwrap_or(host);
    if api_host.is_empty() {
        return None;
    }
    Some(format!("{scheme}://{api_host}"))
}

/// The billing SKU from Copilot's semicolon-delimited token metadata (for
/// example `free_limited_copilot`).  Free plans gate most premium models
/// behind billing, so callers use this to bias defaults and error hints
/// toward models such plans can actually serve.
pub fn sku_from_proxy_token(token: &str) -> Option<&str> {
    token.split(';').find_map(|part| {
        let part = part.trim();
        part.strip_prefix("sku=")
            .map(str::trim)
            .filter(|value| !value.is_empty())
    })
}

pub fn copilot_base_url(token: &str, enterprise_domain: Option<&str>) -> String {
    if let Some(url) = base_url_from_proxy_token(token) {
        return url;
    }
    if let Ok(Some(domain)) = normalize_domain(enterprise_domain) {
        return format!("https://copilot-api.{domain}");
    }
    "https://api.individual.githubcopilot.com".into()
}

/// Filter the dynamic `/models` response: picker-enabled models unless the
/// policy state is `disabled`, with policy-enabled models as the fallback on
/// the individual SKU.  Unknown fields are ignored, while unknown model IDs
/// are left for the provider's static catalog to discard.
pub fn parse_available_model_ids_value(
    value: &Value,
    allow_policy_fallback: bool,
) -> Result<Vec<String>> {
    let data = value.get("data").and_then(Value::as_array).ok_or_else(|| {
        AuthError::InvalidCredential("Copilot models response has no data array".into())
    })?;
    let mut picker = Vec::new();
    let mut policy = Vec::new();
    for item in data {
        let Some(object) = item.as_object() else {
            continue;
        };
        let Some(id) = object.get("id").and_then(Value::as_str) else {
            continue;
        };
        let supports_tools = object
            .get("capabilities")
            .and_then(|value| value.get("supports"))
            .and_then(|value| value.get("tool_calls"))
            .and_then(Value::as_bool)
            != Some(false);
        if !supports_tools {
            continue;
        }
        let policy_state = object
            .get("policy")
            .and_then(|value| value.get("state"))
            .and_then(Value::as_str);
        if object.get("model_picker_enabled").and_then(Value::as_bool) == Some(true)
            && policy_state != Some("disabled")
        {
            push_unique(&mut picker, id);
        }
        if policy_state == Some("enabled") {
            push_unique(&mut policy, id);
        }
    }
    if !picker.is_empty() || !allow_policy_fallback {
        Ok(picker)
    } else {
        Ok(policy)
    }
}

pub fn parse_available_model_ids(body: &str, allow_policy_fallback: bool) -> Result<Vec<String>> {
    let value = serde_json::from_str(body)
        .map_err(|_| AuthError::InvalidCredential("invalid Copilot models JSON".into()))?;
    parse_available_model_ids_value(&value, allow_policy_fallback)
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_owned());
    }
}

/// Auth state shared by the provider and the agent.  It caches the current
/// credential but always persists refreshes through `AuthStore`, so a restart
/// observes the same short-lived token state.
#[derive(Clone)]
pub struct CopilotAuth {
    store: AuthStore,
    client: GithubCopilotClient,
    credential: Arc<Mutex<Option<CopilotCredential>>>,
}

impl fmt::Debug for CopilotAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let credential = self.credential.lock().ok().and_then(|value| value.clone());
        formatter
            .debug_struct("CopilotAuth")
            .field("store", &self.store)
            .field(
                "credential",
                &credential.as_ref().map(CopilotCredential::redacted),
            )
            .finish()
    }
}

impl CopilotAuth {
    pub fn new(store: AuthStore) -> Result<Self> {
        let credential = store.copilot()?;
        Ok(Self {
            store,
            client: GithubCopilotClient::new()?,
            credential: Arc::new(Mutex::new(credential)),
        })
    }

    pub fn from_default() -> Result<Self> {
        Self::new(AuthStore::default())
    }

    pub fn store(&self) -> &AuthStore {
        &self.store
    }

    pub fn credential(&self) -> Result<Option<CopilotCredential>> {
        if let Some(credential) = self
            .credential
            .lock()
            .map_err(|_| AuthError::InvalidCredential("credential lock poisoned".into()))?
            .clone()
        {
            return Ok(Some(credential));
        }
        let credential = self.store.copilot()?;
        if let Some(value) = credential.clone() {
            *self
                .credential
                .lock()
                .map_err(|_| AuthError::InvalidCredential("credential lock poisoned".into()))? =
                Some(value);
        }
        Ok(credential)
    }

    pub fn base_url_for(&self, credential: &CopilotCredential) -> String {
        self.client.api_base_url.clone().unwrap_or_else(|| {
            copilot_base_url(&credential.access, credential.enterprise_url.as_deref())
        })
    }

    pub async fn ensure_valid(&self) -> Result<CopilotCredential> {
        let credential = self.credential()?.ok_or(AuthError::NotAuthenticated)?;
        if !credential.is_expired() {
            return Ok(credential);
        }
        if credential.refresh.trim().is_empty() {
            return Err(AuthError::NotAuthenticated);
        }
        self.refresh().await
    }

    pub async fn refresh(&self) -> Result<CopilotCredential> {
        let old = self.credential()?.ok_or(AuthError::NotAuthenticated)?;
        let cancel = CancellationToken::new();
        let mut refreshed = self.client.refresh_copilot_token(&old, &cancel).await?;
        // Token refresh must remain useful if the optional model-policy
        // endpoint is temporarily unavailable.  Keep the last known list and
        // let an explicit model-list refresh report that endpoint failure.
        if let Ok(available_model_ids) = self
            .client
            .fetch_available_model_ids(
                &refreshed.access,
                refreshed.enterprise_url.as_deref(),
                &cancel,
            )
            .await
        {
            refreshed.available_model_ids = available_model_ids;
        } else {
            refreshed.available_model_ids = old.available_model_ids.clone();
        }
        self.store.save_copilot(&refreshed)?;
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::InvalidCredential("credential lock poisoned".into()))? =
            Some(refreshed.clone());
        Ok(refreshed)
    }

    /// Fetch current model policy data and persist the refreshed list without
    /// changing either OAuth token.
    pub async fn refresh_available_model_ids(&self) -> Result<Vec<String>> {
        let credential = self.ensure_valid().await?;
        let cancel = CancellationToken::new();
        let ids = self
            .client
            .fetch_available_model_ids(
                &credential.access,
                credential.enterprise_url.as_deref(),
                &cancel,
            )
            .await?;
        let mut updated = credential;
        updated.available_model_ids = ids.clone();
        self.store.save_copilot(&updated)?;
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::InvalidCredential("credential lock poisoned".into()))? =
            Some(updated);
        Ok(ids)
    }

    pub async fn login(
        &self,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<CopilotCredential> {
        self.login_with_events(enterprise_domain, cancel, |_| {})
            .await
    }

    pub async fn login_with_events<F>(
        &self,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
        emit: F,
    ) -> Result<CopilotCredential>
    where
        F: FnMut(AuthEvent) + Send,
    {
        let domain = normalize_domain(enterprise_domain)?;
        let client = if domain.is_some() {
            self.client.for_domain(domain.as_deref())?
        } else {
            self.client.clone()
        };
        let credential = client
            .login_with_events(domain.as_deref(), cancel, emit)
            .await?;
        self.store.save_copilot(&credential)?;
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::InvalidCredential("credential lock poisoned".into()))? =
            Some(credential.clone());
        Ok(credential)
    }
}

fn copilot_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static(COPILOT_USER_AGENT));
    headers.insert(
        HeaderName::from_static("editor-version"),
        HeaderValue::from_static(COPILOT_EDITOR_VERSION),
    );
    headers.insert(
        HeaderName::from_static("editor-plugin-version"),
        HeaderValue::from_static(COPILOT_EDITOR_PLUGIN_VERSION),
    );
    headers.insert(
        HeaderName::from_static("copilot-integration-id"),
        HeaderValue::from_static(COPILOT_INTEGRATION_ID),
    );
    headers
}

fn bearer(token: &str) -> Option<HeaderValue> {
    HeaderValue::from_str(&format!("Bearer {token}")).ok()
}

fn network_error(endpoint: &str, error: reqwest::Error) -> AuthError {
    AuthError::Network {
        endpoint: endpoint_label(endpoint),
        message: error
            .status()
            .map(|status| format!("HTTP status {}", status.as_u16()))
            .unwrap_or_else(|| "request failed".into()),
    }
}

fn endpoint_label(endpoint: &str) -> String {
    Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_else(|| "GitHub".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn normalizes_github_and_enterprise_domains() {
        assert_eq!(normalize_domain(None).unwrap(), None);
        assert_eq!(
            normalize_domain(Some("github.com")).unwrap(),
            Some("github.com".into())
        );
        assert_eq!(
            normalize_domain(Some("https://GHE.Example.com/")).unwrap(),
            Some("ghe.example.com".into())
        );
        assert!(normalize_domain(Some("https://ghe.example.com/path")).is_err());
    }

    #[test]
    fn constructs_github_endpoints() {
        let endpoints = CopilotEndpoints::for_domain(None).unwrap();
        assert_eq!(
            endpoints.device_code_url,
            "https://github.com/login/device/code"
        );
        assert_eq!(
            endpoints.access_token_url,
            "https://github.com/login/oauth/access_token"
        );
        assert_eq!(
            endpoints.copilot_token_url,
            "https://api.github.com/copilot_internal/v2/token"
        );
    }

    #[test]
    fn parses_proxy_endpoint_and_fallbacks() {
        assert_eq!(
            base_url_from_proxy_token("tid=x;proxy-ep=proxy.individual.githubcopilot.com;exp=1"),
            Some("https://api.individual.githubcopilot.com".into())
        );
        assert_eq!(
            copilot_base_url("no-proxy", Some("ghe.example.com")),
            "https://copilot-api.ghe.example.com"
        );
        assert_eq!(
            copilot_base_url("no-proxy", None),
            "https://api.individual.githubcopilot.com"
        );
    }

    #[test]
    fn parses_billing_sku_from_token_metadata() {
        assert_eq!(
            sku_from_proxy_token("tid=x;sku=free_limited_copilot;chat=1"),
            Some("free_limited_copilot")
        );
        assert_eq!(sku_from_proxy_token("tid=x;sku="), None);
        assert_eq!(sku_from_proxy_token("tid=x;chat=1"), None);
    }

    #[test]
    fn filters_models_by_tool_support_and_policy() {
        let ids = parse_available_model_ids_value(
            &json!({"data": [
                {"id":"gpt-5.4", "model_picker_enabled":true, "policy":{"state":"enabled"}},
                {"id":"no-tools", "model_picker_enabled":true, "capabilities":{"supports":{"tool_calls":false}}},
                {"id":"disabled", "model_picker_enabled":true, "policy":{"state":"disabled"}},
                {"id":"policy-only", "policy":{"state":"enabled"}}
            ]}),
            false,
        )
        .unwrap();
        assert_eq!(ids, vec!["gpt-5.4"]);
        let fallback = parse_available_model_ids_value(
            &json!({"data": [{"id":"policy-only", "policy":{"state":"enabled"}}]}),
            true,
        )
        .unwrap();
        assert_eq!(fallback, vec!["policy-only"]);
    }

    #[test]
    fn token_exchange_applies_refresh_skew_without_exposing_tokens() {
        let credential = parse_copilot_token(
            &json!({"token":"secret-access", "expires_at": 2_000_000}),
            "secret-refresh",
            None,
        )
        .unwrap();
        assert_eq!(credential.expires, 1_999_700_000);
        assert!(!format!("{credential:?}").contains("secret"));
    }

    #[test]
    fn auth_can_be_constructed_without_touching_provider_config() {
        let directory = tempdir().unwrap();
        let store = AuthStore::new(directory.path().join("auth.json"));
        let auth = CopilotAuth::new(store).unwrap();
        assert!(auth.credential().unwrap().is_none());
    }
}
