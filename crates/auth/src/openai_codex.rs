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
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
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
        let old = self
            .credential()?
            .ok_or(AuthError::OpenAiCodexNotAuthenticated)?;
        let value = self
            .token(
                json!({"grant_type":"refresh_token", "refresh_token":old.refresh, "client_id": OPENAI_CODEX_CLIENT_ID} ),
                &CancellationToken::new(),
            )
            .await?;
        let credential = credential_from_token(&value, Some(&old.refresh))?;
        self.persist(credential.clone())?;
        Ok(credential)
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
        let value = self.token(json!({"grant_type":"authorization_code", "client_id":OPENAI_CODEX_CLIENT_ID, "code":code, "code_verifier":values.verifier, "redirect_uri":"http://localhost:1455/auth/callback"}), cancel).await?;
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
            if std::time::Instant::now() >= deadline {
                return Err(AuthError::DeviceCodeExpired);
            }
            cancellable_sleep(interval, cancel).await?;
            let value = self.token(json!({"grant_type":"urn:ietf:params:oauth:grant-type:device_code", "device_code":device.device_code, "client_id":OPENAI_CODEX_CLIENT_ID}), cancel).await?;
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
    async fn token(&self, body: Value, cancel: &CancellationToken) -> Result<Value> {
        self.request_json(
            self.http.post(&self.endpoints.token_url).form(&body),
            cancel,
        )
        .await
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

async fn wait_for_callback(
    listener: TcpListener,
    expected_state: &str,
    cancel: &CancellationToken,
) -> Result<String> {
    loop {
        let (mut stream, _) = tokio::select! { _ = cancel.cancelled() => return Err(AuthError::Cancelled), value = listener.accept() => value.map_err(|_| AuthError::OpenAiCodex("callback listener failed".into()))? };
        let mut request = vec![0; 8192];
        let size = tokio::select! { _ = cancel.cancelled() => return Err(AuthError::Cancelled), value = stream.read(&mut request) => value.map_err(|_| AuthError::OpenAiCodex("callback read failed".into()))? };
        let target = std::str::from_utf8(&request[..size])
            .ok()
            .and_then(|s| s.lines().next())
            .and_then(|line| line.split_whitespace().nth(1));
        let outcome = target
            .and_then(|target| Url::parse(&format!("http://localhost{target}")).ok())
            .and_then(|url| (url.path() == CALLBACK_PATH).then_some(url))
            .and_then(|url| {
                let pairs: std::collections::HashMap<_, _> =
                    url.query_pairs().into_owned().collect();
                (pairs
                    .get("state")
                    .is_some_and(|state| state == expected_state))
                .then(|| pairs.get("code").cloned())
                .flatten()
            });
        let (body, code) = match outcome {
            Some(code) => ("Login complete. You may close this window.", Some(code)),
            None => ("Login failed. Return to Harness and try again.", None),
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n<html><body>{body}</body></html>",
            body.len() + 26
        );
        let _ = stream.write_all(response.as_bytes()).await;
        if let Some(code) = code {
            return Ok(code);
        }
    }
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

/// Extract an account from a single JWT for callers that only have an access token.
pub fn account_id_from_jwt(token: &str) -> Result<String> {
    account_id_from_tokens(None, Some(token))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(account_id_from_jwt(&token).unwrap(), "acct");
    }
}
