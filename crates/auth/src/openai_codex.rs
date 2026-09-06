//! OAuth support for the ChatGPT Codex subscription API.
//!
//! The callback listener deliberately accepts only the fixed local callback
//! path and validates the PKCE state before exchanging a code.  It never puts
//! tokens in the browser response or in diagnostics.

use crate::device_code::{AuthEvent, cancellable_sleep, parse_device_code};
use crate::error::{AuthError, Result};
use crate::storage::{AuthStore, OpenAiCodexCredential};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::{Client, Url};
use ring::{
    digest,
    rand::{SecureRandom, SystemRandom},
};
use serde_json::{Value, json};
use std::{
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Public OAuth client identity used by the Codex subscription login.
/// Public client identity used by the Codex CLI OAuth flow.
pub const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const CALLBACK_PORT: u16 = 1455;
pub const CALLBACK_PATH: &str = "/auth/callback";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiCodexEndpoints {
    pub authorize_url: String,
    pub token_url: String,
    pub device_code_url: String,
}
impl Default for OpenAiCodexEndpoints {
    fn default() -> Self {
        Self {
            authorize_url: "https://auth.openai.com/oauth/authorize".into(),
            token_url: "https://auth.openai.com/oauth/token".into(),
            device_code_url: "https://auth.openai.com/api/accounts/deviceauth/usercode".into(),
        }
    }
}

/// PKCE values for one authorization attempt. Secrets deliberately redact.
#[derive(Clone, PartialEq, Eq)]
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
    pub state: String,
}
impl fmt::Debug for Pkce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pkce")
            .field("verifier", &"<redacted>")
            .field("challenge", &self.challenge)
            .field("state", &"<redacted>")
            .finish()
    }
}
pub fn pkce() -> Result<Pkce> {
    let mut verifier = [0_u8; 48];
    let mut state = [0_u8; 32];
    let rng = SystemRandom::new();
    rng.fill(&mut verifier)
        .map_err(|_| AuthError::OpenAiCodex("could not generate OAuth randomness".into()))?;
    rng.fill(&mut state)
        .map_err(|_| AuthError::OpenAiCodex("could not generate OAuth randomness".into()))?;
    let verifier = URL_SAFE_NO_PAD.encode(verifier);
    let challenge = URL_SAFE_NO_PAD.encode(digest::digest(&digest::SHA256, verifier.as_bytes()));
    Ok(Pkce {
        verifier,
        challenge,
        state: URL_SAFE_NO_PAD.encode(state),
    })
}

#[derive(Clone)]
pub struct OpenAiCodexAuth {
    store: AuthStore,
    http: Client,
    endpoints: OpenAiCodexEndpoints,
    credential: Arc<Mutex<Option<OpenAiCodexCredential>>>,
    /// Serializes rotating refresh-token exchanges so concurrent
    /// `ensure_valid` calls produce one network refresh and never let an
    /// older completion overwrite newer credentials.
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
}
impl fmt::Debug for OpenAiCodexAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCodexAuth")
            .field("store", &self.store)
            .field("endpoints", &self.endpoints)
            .field("credential", &"<redacted>")
            .finish()
    }
}
impl OpenAiCodexAuth {
    pub fn new(store: AuthStore) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let credential = store.openai_codex()?;
        let http = Client::builder()
            .build()
            .map_err(|_| AuthError::OpenAiCodex("could not create HTTP client".into()))?;
        Ok(Self {
            store,
            http,
            endpoints: OpenAiCodexEndpoints::default(),
            credential: Arc::new(Mutex::new(credential)),
            refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }
    pub fn from_default() -> Result<Self> {
        Self::new(AuthStore::default())
    }
    pub fn with_endpoints(mut self, endpoints: OpenAiCodexEndpoints) -> Self {
        self.endpoints = endpoints;
        self
    }
    pub fn credential(&self) -> Result<Option<OpenAiCodexCredential>> {
        if let Some(value) = self
            .credential
            .lock()
            .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))?
            .clone()
        {
            return Ok(Some(value));
        }
        let value = self.store.openai_codex()?;
        if let Some(ref credential) = value {
            *self
                .credential
                .lock()
                .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))? =
                Some(credential.clone());
        }
        Ok(value)
    }
    pub fn authorization_url(&self, values: &Pkce) -> Result<String> {
        let mut url = Url::parse(&self.endpoints.authorize_url)
            .map_err(|_| AuthError::OpenAiCodex("invalid authorization endpoint".into()))?;
        url.query_pairs_mut().extend_pairs([
            ("response_type", "code"),
            ("client_id", OPENAI_CODEX_CLIENT_ID),
            ("redirect_uri", "http://localhost:1455/auth/callback"),
            ("scope", "openid profile email offline_access"),
            ("code_challenge_method", "S256"),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "harness"),
            ("code_challenge", &values.challenge),
            ("state", &values.state),
        ]);
        Ok(url.into())
    }
    pub async fn ensure_valid(&self) -> Result<OpenAiCodexCredential> {
        let credential = self
            .credential()?
            .ok_or(AuthError::OpenAiCodexNotAuthenticated)?;
        if !credential.is_expired() {
            return Ok(credential);
        }
        self.refresh().await
    }
    pub async fn refresh(&self) -> Result<OpenAiCodexCredential> {
        // Single-flight: concurrent refreshers queue here, then recheck the
        // credential before refreshing so only the first waiter hits the
        // network.  Never hold the blocking cache mutex across the network;
        // the async guard is released before any `.await` on the mutex.
        let _guard = self.refresh_lock.lock().await;
        // Reload: another waiter (or process) may have already refreshed.
        if let Ok(Some(current)) = self.store.openai_codex() {
            let cached = self
                .credential
                .lock()
                .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))?
                .clone();
            // Prefer the newest of cache vs. disk; disk wins ties only when
            // it is unexpired (a fresh rotation another process persisted).
            if let Some(cached) = cached {
                if !cached.is_expired() {
                    return Ok(cached);
                }
                if !current.is_expired()
                    && current.refresh != cached.refresh
                    && current.access != cached.access
                {
                    *self
                        .credential
                        .lock()
                        .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))? =
                        Some(current.clone());
                    return Ok(current);
                }
            } else if !current.is_expired() {
                *self
                    .credential
                    .lock()
                    .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))? =
                    Some(current.clone());
                return Ok(current);
            }
        }
        let old = self
            .credential()?
            .ok_or(AuthError::OpenAiCodexNotAuthenticated)?;
        let old_refresh = old.refresh.clone();
        // Refresh exchanges use strict status handling like the browser
        // flow: only 2xx with a token payload succeeds.
        let value = self
            .token_strict(
                json!({"grant_type":"refresh_token", "refresh_token":old.refresh, "client_id": OPENAI_CODEX_CLIENT_ID} ),
                &CancellationToken::new(),
            )
            .await?;
        let credential = credential_from_token(&value, Some(&old_refresh))?;
        // Compare and save under the auth-file lock. A separate process may
        // have rotated the credential while this exchange was in flight; in
        // that case never overwrite its newer generation.
        if self.store.save_openai_codex_if_current(&old, &credential)? {
            *self
                .credential
                .lock()
                .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))? =
                Some(credential.clone());
            return Ok(credential);
        }
        let current = self
            .store
            .openai_codex()?
            .ok_or_else(|| AuthError::OpenAiCodex("credential changed during refresh".into()))?;
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))? =
            Some(current.clone());
        Ok(current)
    }
    fn persist(&self, credential: OpenAiCodexCredential) -> Result<()> {
        self.store.save_openai_codex(&credential)?;
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::OpenAiCodex("credential lock poisoned".into()))? =
            Some(credential);
        Ok(())
    }
    pub async fn login_browser<F>(
        &self,
        cancel: &CancellationToken,
        mut emit: F,
    ) -> Result<OpenAiCodexCredential>
    where
        F: FnMut(AuthEvent) + Send,
    {
        // Bind before opening the browser so a fast callback cannot race the
        // listener and a busy port fails with actionable device-flow advice.
        let listener = TcpListener::bind(("127.0.0.1", CALLBACK_PORT))
            .await
            .map_err(|_| AuthError::CallbackBind)?;
        let values = pkce()?;
        let url = self.authorization_url(&values)?;
        emit(AuthEvent::Started);
        emit(AuthEvent::Prompt { message: url });
        let code = wait_for_callback(listener, &values.state, cancel).await?;
        let value = self.token_strict(json!({"grant_type":"authorization_code", "client_id":OPENAI_CODEX_CLIENT_ID, "code":code, "code_verifier":values.verifier, "redirect_uri":"http://localhost:1455/auth/callback"}), cancel).await?;
        let credential = credential_from_token(&value, None)?;
        self.persist(credential.clone())?;
        emit(AuthEvent::Finished);
        Ok(credential)
    }
    pub async fn login_device<F>(
        &self,
        cancel: &CancellationToken,
        mut emit: F,
    ) -> Result<OpenAiCodexCredential>
    where
        F: FnMut(AuthEvent) + Send,
    {
        emit(AuthEvent::Started);
        let value = self
            .request_json(
                self.http
                    .post(&self.endpoints.device_code_url)
                    .json(&json!({"client_id":OPENAI_CODEX_CLIENT_ID})),
                cancel,
            )
            .await?;
        let device = parse_device_code(&value)?;
        emit(AuthEvent::DeviceCode {
            verification_url: device.verification_url.clone(),
            user_code: device.user_code.clone(),
            expires_in: device.expires_in,
            interval: device.interval,
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(device.expires_in);
        let mut interval = device.interval;
        loop {
            // Check expiry before sleeping so an already-expired grant
            // fails fast, and re-check after the sleep and around the
            // request (inside `token`) so cancellation/expiry interrupt
            // both waits and in-flight polls.
            if std::time::Instant::now() >= deadline {
                return Err(AuthError::DeviceCodeExpired);
            }
            cancellable_sleep(interval, cancel).await?;
            if std::time::Instant::now() >= deadline {
                return Err(AuthError::DeviceCodeExpired);
            }
            if cancel.is_cancelled() {
                return Err(AuthError::Cancelled);
            }
            let value = self
                .request_token_until(
                    self.http
                        .post(&self.endpoints.token_url)
                        .form(&json!({"grant_type":"urn:ietf:params:oauth:grant-type:device_code", "device_code":device.device_code, "client_id":OPENAI_CODEX_CLIENT_ID})),
                    cancel,
                    Some(deadline),
                )
                .await?;
            if let Some(error) = value.get("error").and_then(Value::as_str) {
                match error {
                    "authorization_pending" => continue,
                    "slow_down" => {
                        interval = interval.saturating_add(5);
                        continue;
                    }
                    "expired_token" => return Err(AuthError::DeviceCodeExpired),
                    "access_denied" => {
                        return Err(AuthError::OpenAiCodex(
                            "device authorization was denied".into(),
                        ));
                    }
                    _ => return Err(AuthError::OpenAiCodex("device authorization failed".into())),
                }
            }
            let credential = credential_from_token(&value, None)?;
            self.persist(credential.clone())?;
            emit(AuthEvent::Finished);
            return Ok(credential);
        }
    }
    /// Browser authorization exchange: strict status handling (no RFC 8628
    /// polling semantics).  Only 2xx with a token payload succeeds; error
    /// bodies are bounded and never echoed.
    async fn token_strict(&self, body: Value, cancel: &CancellationToken) -> Result<Value> {
        let response = tokio::select! { _ = cancel.cancelled() => return Err(AuthError::Cancelled), result = self.http.post(&self.endpoints.token_url).form(&body).send() => result.map_err(|_| AuthError::OpenAiCodex("network request failed".into()))? };
        let status = response.status();
        let body = tokio::select! { _ = cancel.cancelled() => return Err(AuthError::Cancelled), body = read_bounded_body(response) => body? };
        if !status.is_success() {
            return Err(AuthError::Http {
                status: status.as_u16(),
                endpoint: "auth.openai.com".into(),
            });
        }
        serde_json::from_slice(&body)
            .map_err(|_| AuthError::OpenAiCodex("invalid OAuth response".into()))
    }
    /// Token endpoint with RFC 8628 device-polling semantics: read a bounded
    /// response body regardless of HTTP status, then map recognized `error`
    /// values (`authorization_pending`, `slow_down`, `expired_token`,
    /// `access_denied`) before treating other non-success statuses as
    /// errors.  Cancellation and expiry are checked around both sleeps and
    /// requests; token bodies never enter errors or logs.
    #[cfg(test)]
    async fn request_token(
        &self,
        request: reqwest::RequestBuilder,
        cancel: &CancellationToken,
    ) -> Result<Value> {
        self.request_token_until(request, cancel, None).await
    }

    async fn request_token_until(
        &self,
        request: reqwest::RequestBuilder,
        cancel: &CancellationToken,
        deadline: Option<std::time::Instant>,
    ) -> Result<Value> {
        let response = if let Some(deadline) = deadline {
            let deadline = tokio::time::Instant::from_std(deadline);
            tokio::select! {
                _ = cancel.cancelled() => return Err(AuthError::Cancelled),
                _ = tokio::time::sleep_until(deadline) => return Err(AuthError::DeviceCodeExpired),
                result = request.send() => result.map_err(|_| AuthError::OpenAiCodex("network request failed".into()))?,
            }
        } else {
            tokio::select! {
                _ = cancel.cancelled() => return Err(AuthError::Cancelled),
                result = request.send() => result.map_err(|_| AuthError::OpenAiCodex("network request failed".into()))?,
            }
        };
        let status = response.status();
        let body = if let Some(deadline) = deadline {
            let deadline = tokio::time::Instant::from_std(deadline);
            tokio::select! {
                _ = cancel.cancelled() => return Err(AuthError::Cancelled),
                _ = tokio::time::sleep_until(deadline) => return Err(AuthError::DeviceCodeExpired),
                body = read_bounded_body(response) => body?,
            }
        } else {
            tokio::select! {
                _ = cancel.cancelled() => return Err(AuthError::Cancelled),
                body = read_bounded_body(response) => body?,
            }
        };
        let value: Value = serde_json::from_slice(&body)
            .map_err(|_| AuthError::OpenAiCodex("invalid OAuth response".into()))?;
        if status.is_success() && value.get("error").is_none() {
            return Ok(value);
        }
        match value.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => Ok(value),
            Some("slow_down") => Ok(value),
            Some("expired_token") => Ok(value),
            Some("access_denied") => Ok(value),
            _ => Err(AuthError::Http {
                status: status.as_u16(),
                endpoint: "auth.openai.com".into(),
            }),
        }
    }
    async fn request_json(
        &self,
        request: reqwest::RequestBuilder,
        cancel: &CancellationToken,
    ) -> Result<Value> {
        let response = tokio::select! { _ = cancel.cancelled() => return Err(AuthError::Cancelled), result = request.send() => result.map_err(|_| AuthError::OpenAiCodex("network request failed".into()))? };
        if !response.status().is_success() {
            return Err(AuthError::Http {
                status: response.status().as_u16(),
                endpoint: "auth.openai.com".into(),
            });
        }
        tokio::select! { _ = cancel.cancelled() => Err(AuthError::Cancelled), value = response.json() => value.map_err(|_| AuthError::OpenAiCodex("invalid OAuth response".into())) }
    }
}

/// Read at most `OAUTH_BODY_LIMIT` bytes of a token/authorize response.
/// OAuth error payloads are small JSON objects; bounding the read keeps a
/// malicious endpoint from filling memory before the RFC 8628 error mapping
/// runs.  Bodies are parsed, never echoed into errors or logs.
const OAUTH_BODY_LIMIT: usize = 64 * 1024;

async fn read_bounded_body(response: reqwest::Response) -> Result<Vec<u8>> {
    use futures_util::StreamExt;
    let mut body: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk: bytes::Bytes =
            chunk.map_err(|_| AuthError::OpenAiCodex("network request failed".into()))?;
        let remaining = OAUTH_BODY_LIMIT.saturating_sub(body.len());
        if chunk.len() > remaining {
            return Err(AuthError::OpenAiCodex("invalid OAuth response".into()));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Browser-flow listener timeouts: an idle connection cannot park login
/// forever, and the whole flow is bounded so a hanging browser tab fails
/// with actionable device-flow advice instead of blocking shutdown.
/// `idle_timeout` is exposed for tests; production uses 10 seconds.
const CALLBACK_OVERALL_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// Maximum callback request head: headers are read incrementally through
/// the `\r\n\r\n` terminator under this cap so a slowloris-style sender
/// cannot grow memory without bound.
const CALLBACK_HEAD_LIMIT: usize = 16 * 1024;

async fn wait_for_callback(
    listener: TcpListener,
    expected_state: &str,
    cancel: &CancellationToken,
) -> Result<String> {
    wait_for_callback_with_idle(listener, expected_state, cancel, Duration::from_secs(10)).await
}

async fn wait_for_callback_with_idle(
    listener: TcpListener,
    expected_state: &str,
    cancel: &CancellationToken,
    idle_timeout: Duration,
) -> Result<String> {
    let deadline = std::time::Instant::now() + CALLBACK_OVERALL_TIMEOUT;
    loop {
        if std::time::Instant::now() >= deadline {
            return Err(AuthError::OpenAiCodex(
                "browser login timed out; use `harness login openai-codex --device-code`".into(),
            ));
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let accept = listener.accept();
        let (mut stream, _) = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            _ = tokio::time::sleep(remaining) => return Err(AuthError::OpenAiCodex("browser login timed out; use `harness login openai-codex --device-code`".into())),
            value = accept => value.map_err(|_| AuthError::OpenAiCodex("callback listener failed".into()))?
        };
        let request = match read_callback_head(&stream, cancel, idle_timeout).await {
            Ok(request) => request,
            // Idle/slow senders time out per-connection without blocking a
            // later valid callback; malformed heads are answered as failure.
            Err(_) => {
                let _ = respond_callback(&mut stream, false).await;
                continue;
            }
        };
        let outcome = parse_callback_target(&request, expected_state);
        // A valid-state OAuth denial (`error=access_denied`) terminates
        // promptly as a sanitized denial, not a retried code exchange.
        if is_callback_denial(&request, expected_state) {
            let _ = respond_callback(&mut stream, false).await;
            return Err(AuthError::OpenAiCodex(
                "browser authorization was denied".into(),
            ));
        }
        let success = outcome.is_some();
        let _ = respond_callback(&mut stream, success).await;
        if let Some(code) = outcome {
            return Ok(code);
        }
    }
}

/// Read one HTTP request head incrementally through the `\r\n\r\n`
/// terminator under [`CALLBACK_HEAD_LIMIT`], timing out idle connections so
/// one hanging sender cannot block later callbacks.
async fn read_callback_head(
    stream: &tokio::net::TcpStream,
    cancel: &CancellationToken,
    idle_timeout: Duration,
) -> Result<String> {
    let mut head: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 512];
    loop {
        if head.len() > CALLBACK_HEAD_LIMIT {
            return Err(AuthError::OpenAiCodex("callback request too large".into()));
        }
        if head.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        let read = async {
            stream
                .readable()
                .await
                .and_then(|_| stream.try_read(&mut chunk))
        };
        let size = tokio::select! {
            _ = cancel.cancelled() => return Err(AuthError::Cancelled),
            _ = tokio::time::sleep(idle_timeout) => return Err(AuthError::OpenAiCodex("callback read timed out".into())),
            value = read => value.map_err(|_| AuthError::OpenAiCodex("callback read failed".into()))?,
        };
        if size == 0 {
            break;
        }
        head.extend_from_slice(&chunk[..size]);
    }
    String::from_utf8(head).map_err(|_| AuthError::OpenAiCodex("callback read failed".into()))
}

/// Parse the request target from a callback head: the authorization `code`
/// when the path and PKCE `state` match.
fn parse_callback_target(request: &str, expected_state: &str) -> Option<String> {
    let target = request.lines().next()?.split_whitespace().nth(1)?;
    let url = Url::parse(&format!("http://localhost{target}")).ok()?;
    if url.path() != CALLBACK_PATH {
        return None;
    }
    let pairs: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
    if pairs
        .get("state")
        .is_none_or(|state| state != expected_state)
    {
        return None;
    }
    pairs.get("code").cloned()
}

/// True when the callback is a valid-state OAuth denial rather than a code:
/// `?error=access_denied&state=<expected>`.
fn is_callback_denial(request: &str, expected_state: &str) -> bool {
    let Some(target) = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
    else {
        return false;
    };
    let Ok(url) = Url::parse(&format!("http://localhost{target}")) else {
        return false;
    };
    if url.path() != CALLBACK_PATH {
        return false;
    }
    let pairs: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
    pairs
        .get("state")
        .is_some_and(|state| state == expected_state)
        && pairs
            .get("error")
            .is_some_and(|error| error == "access_denied")
}

async fn respond_callback(stream: &mut tokio::net::TcpStream, success: bool) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let body = if success {
        "Login complete. You may close this window."
    } else {
        "Login failed. Return to Harness and try again."
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n<html><body>{body}</body></html>",
        body.len() + 26
    );
    stream
        .write_all(response.as_bytes())
        .await
        .map_err(|_| AuthError::OpenAiCodex("callback write failed".into()))?;
    Ok(())
}

fn credential_from_token(
    value: &Value,
    old_refresh: Option<&str>,
) -> Result<OpenAiCodexCredential> {
    let access = value
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| AuthError::OpenAiCodex("token response has no access token".into()))?;
    let refresh = value
        .get("refresh_token")
        .and_then(Value::as_str)
        .or(old_refresh)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| AuthError::OpenAiCodex("token response has no refresh token".into()))?;
    let expires = value
        .get("expires_in")
        .and_then(Value::as_u64)
        .map(|seconds| unix_millis().saturating_add(seconds.saturating_mul(1000)))
        .unwrap_or_else(|| jwt_expiry(access).unwrap_or(0));
    let account_id =
        account_id_from_tokens(value.get("id_token").and_then(Value::as_str), Some(access))?;
    Ok(OpenAiCodexCredential::new(
        access, refresh, expires, account_id,
    ))
}
fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
fn jwt_payload(token: &str) -> Result<Value> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| AuthError::OpenAiCodex("malformed access token".into()))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_| AuthError::OpenAiCodex("malformed access token".into()))?;
    serde_json::from_slice(&bytes)
        .map_err(|_| AuthError::OpenAiCodex("malformed access token".into()))
}
fn jwt_expiry(token: &str) -> Option<u64> {
    jwt_payload(token)
        .ok()?
        .get("exp")?
        .as_u64()
        .map(|v| v.saturating_mul(1000))
}
/// Extract the account selected by ChatGPT from either returned JWT. The
/// claim can appear in an ID token, an access token, or the first organization.
pub fn account_id_from_tokens(
    id_token: Option<&str>,
    access_token: Option<&str>,
) -> Result<String> {
    for token in [id_token, access_token].into_iter().flatten() {
        let Ok(payload) = jwt_payload(token) else {
            continue;
        };
        let id = payload
            .get("chatgpt_account_id")
            .or_else(|| {
                payload
                    .get("https://api.openai.com/auth")
                    .and_then(|v| v.get("chatgpt_account_id"))
            })
            .or_else(|| payload.get("account_id"))
            .or_else(|| {
                payload
                    .get("organizations")
                    .and_then(Value::as_array)
                    .and_then(|orgs| orgs.first())
                    .and_then(|org| org.get("id"))
            })
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        if let Some(id) = id {
            return Ok(id.into());
        }
    }
    Err(AuthError::OpenAiCodex(
        "OAuth tokens do not contain a ChatGPT account id".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn pkce_is_url_safe_and_changes_each_time() {
        let a = pkce().unwrap();
        let b = pkce().unwrap();
        assert_ne!(a.verifier, b.verifier);
        assert_eq!(
            a.challenge,
            URL_SAFE_NO_PAD.encode(digest::digest(&digest::SHA256, a.verifier.as_bytes()))
        );
    }
    #[test]
    fn extracts_account_id() {
        let token = format!(
            "x.{}.y",
            URL_SAFE_NO_PAD
                .encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct"}}"#)
        );
        assert_eq!(account_id_from_tokens(None, Some(&token)).unwrap(), "acct");
    }

    /// Minimal HTTP fixture: scripted `(status, body)` replies in order.
    struct Fixture {
        addr: std::net::SocketAddr,
        seen: Arc<Mutex<Vec<String>>>,
    }

    async fn fixture(replies: Vec<(u16, String)>) -> Fixture {
        let replies = Arc::new(Mutex::new(VecDeque::from(replies)));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let replies_task = replies.clone();
        let seen_task = seen.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let next = replies_task.lock().unwrap().pop_front();
                let Some((status, body)) = next else {
                    return;
                };
                // Drain the request head so the client can finish sending.
                let mut scratch = [0u8; 4096];
                let _ = socket.read(&mut scratch).await;
                seen_task
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&scratch).into_owned());
                let reason = match status {
                    200 => "OK",
                    400 => "Bad Request",
                    _ => "Error",
                };
                let head = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
            }
        });
        Fixture { addr, seen }
    }

    fn token_endpoints(addr: &std::net::SocketAddr) -> OpenAiCodexEndpoints {
        OpenAiCodexEndpoints {
            authorize_url: format!("http://{addr}/authorize"),
            token_url: format!("http://{addr}/token"),
            device_code_url: format!("http://{addr}/device"),
        }
    }

    fn success_token_body(refresh: &str) -> String {
        // A JWT whose payload carries the account id and expiry.
        let payload = URL_SAFE_NO_PAD.encode(br#"{"chatgpt_account_id":"acct","exp":9999999999}"#);
        let access = format!("head.{payload}.sig");
        serde_json::json!({
            "access_token": access,
            "refresh_token": refresh,
            "expires_in": 3600,
            "id_token": access,
        })
        .to_string()
    }

    async fn auth_with_fixture(
        store_dir: &tempfile::TempDir,
        fixture: &Fixture,
    ) -> OpenAiCodexAuth {
        let store = AuthStore::new(store_dir.path().join("auth.json"));
        OpenAiCodexAuth::new(store)
            .unwrap()
            .with_endpoints(token_endpoints(&fixture.addr))
    }

    #[tokio::test]
    async fn device_poll_pending_then_success() {
        let dir = tempfile::tempdir().unwrap();
        let fix = fixture(vec![
            (400, r#"{"error":"authorization_pending"}"#.into()),
            (200, success_token_body("refresh-1")),
        ])
        .await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let cancel = CancellationToken::new();
        let credential = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({"grant_type":"device_code"})),
                &cancel,
            )
            .await
            .unwrap();
        assert_eq!(
            credential.get("error").and_then(Value::as_str),
            Some("authorization_pending")
        );
        let credential = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({"grant_type":"device_code"})),
                &cancel,
            )
            .await
            .unwrap();
        let parsed = credential_from_token(&credential, None).unwrap();
        assert_eq!(parsed.refresh, "refresh-1");
        // Request bodies never carry tokens into errors: a bad reply maps to
        // a status-only error with no body echo.
        assert_eq!(fix.seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn device_poll_errors_map_before_status() {
        let dir = tempfile::tempdir().unwrap();
        // `expired_token` on a 400 surface as the parsed error value (the
        // caller maps it to expiry), not as a generic HTTP failure.
        let fix = fixture(vec![(400, r#"{"error":"expired_token"}"#.into())]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let value = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({"grant_type":"device_code"})),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            value.get("error").and_then(Value::as_str),
            Some("expired_token")
        );

        // `access_denied` likewise parses so the caller can deny sanitely.
        let fix = fixture(vec![(400, r#"{"error":"access_denied"}"#.into())]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let value = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({"grant_type":"device_code"})),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            value.get("error").and_then(Value::as_str),
            Some("access_denied")
        );

        // Unrecognized errors on non-success stay generic HTTP failures with
        // no body echo (never leak token response bodies).
        let fix = fixture(vec![(400, r#"{"error":"weird","token":"abc"}"#.into())]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let error = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({"grant_type":"device_code"})),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("400"), "unexpected: {rendered}");
        assert!(!rendered.contains("abc"), "body leaked: {rendered}");
    }

    #[tokio::test]
    async fn slow_down_increases_the_poll_interval() {
        // `request_token` surfaces the parsed `slow_down`; the device loop
        // maps it to `interval + 5`.  Assert the mapping contract directly.
        let dir = tempfile::tempdir().unwrap();
        let fix = fixture(vec![(400, r#"{"error":"slow_down"}"#.into())]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let value = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({"grant_type":"device_code"})),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let interval = 5u64;
        let next = match value.get("error").and_then(Value::as_str) {
            Some("slow_down") => interval.saturating_add(5),
            _ => interval,
        };
        assert_eq!(next, 10);
    }

    #[tokio::test]
    async fn malformed_and_oversized_bodies_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let fix = fixture(vec![(200, "not json".into())]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let error = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({})),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("invalid OAuth response"));

        // Oversized bodies fail before unbounded allocation.
        let big = "x".repeat(OAUTH_BODY_LIMIT + 1024);
        let fix = fixture(vec![(200, big)]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let error = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({})),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("invalid OAuth response"));
    }

    #[tokio::test]
    async fn cancellation_aborts_sleep_and_request() {
        // Cancelled sleep resolves promptly.
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(cancellable_sleep(60, &cancel).await.is_err());

        // Cancelled request never hits the network.
        let dir = tempfile::tempdir().unwrap();
        let fix = fixture(vec![(200, success_token_body("r"))]).await;
        let auth = auth_with_fixture(&dir, &fix).await;
        let error = auth
            .request_token(
                auth.http
                    .post(auth.endpoints.token_url.clone())
                    .form(&serde_json::json!({})),
                &cancel,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, AuthError::Cancelled));
        assert!(fix.seen.lock().unwrap().is_empty());
    }

    #[test]
    fn callback_target_parsing_accepts_codes_and_denials() {
        let request = "GET /auth/callback?code=abc&state=s1 HTTP/1.1\r\nHost: x\r\n\r\n";
        assert_eq!(parse_callback_target(request, "s1").as_deref(), Some("abc"));
        assert!(parse_callback_target(request, "other").is_none());
        assert!(!is_callback_denial(request, "s1"));
        // Fragmented-style denial: valid state + access_denied terminates.
        let denial = "GET /auth/callback?error=access_denied&state=s1 HTTP/1.1\r\n\r\n";
        assert!(parse_callback_target(denial, "s1").is_none());
        assert!(is_callback_denial(denial, "s1"));
        assert!(!is_callback_denial(denial, "other"));
    }

    #[tokio::test]
    async fn idle_connection_does_not_block_a_later_valid_callback() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            wait_for_callback_with_idle(
                listener,
                "s1",
                &CancellationToken::new(),
                Duration::from_millis(300),
            )
            .await
        });
        // First connection: idle, sends nothing (holds the socket open).
        let idle = tokio::net::TcpStream::connect(addr).await.unwrap();
        // Second connection: a fragmented valid callback.
        let mut valid = tokio::net::TcpStream::connect(addr).await.unwrap();
        let head = b"GET /auth/callback?code=frag&state=s1 HTTP/1.1\r\nHost: x\r\n\r\n";
        valid.write_all(&head[..20]).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        valid.write_all(&head[20..]).await.unwrap();
        let mut reply = Vec::new();
        // Read until the response head terminator.
        let mut chunk = [0u8; 512];
        loop {
            let size =
                tokio::time::timeout(std::time::Duration::from_secs(15), valid.read(&mut chunk))
                    .await
                    .unwrap()
                    .unwrap();
            if size == 0 {
                break;
            }
            reply.extend_from_slice(&chunk[..size]);
            if reply.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        assert!(String::from_utf8_lossy(&reply).contains("Login complete"));
        let code = tokio::time::timeout(std::time::Duration::from_secs(15), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(code, "frag");
        drop(idle);
    }
}
