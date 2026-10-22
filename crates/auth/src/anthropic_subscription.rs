//! Unofficial subscription OAuth, derived from pi-sub-anthropic/omp (MIT).
//! See THIRD_PARTY_NOTICES.md. Token bodies and transport errors never escape.
use crate::storage::AnthropicSubscriptionCredential;
use crate::{AuthError, AuthEvent, AuthStore, Result, pkce};
use reqwest::Client;
use serde_json::{Value, json};
use std::{
    fmt,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;
/// Public OAuth client used by the upstream protocol.
pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const REDIRECT: &str = "http://localhost:54545/callback";
/// Endpoints are injectable for offline fixtures.
#[derive(Clone, Debug)]
pub struct AnthropicSubscriptionEndpoints {
    pub authorize_url: String,
    pub token_url: String,
}
impl Default for AnthropicSubscriptionEndpoints {
    fn default() -> Self {
        Self {
            authorize_url: "https://claude.ai/oauth/authorize".into(),
            token_url: "https://api.anthropic.com/v1/oauth/token".into(),
        }
    }
}
/// Shared, single-flight rotating-token lifecycle backed by private auth.json.
#[derive(Clone)]
pub struct AnthropicSubscriptionAuth {
    store: AuthStore,
    http: Client,
    endpoints: AnthropicSubscriptionEndpoints,
    credential: Arc<Mutex<Option<AnthropicSubscriptionCredential>>>,
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
}
impl fmt::Debug for AnthropicSubscriptionAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicSubscriptionAuth")
            .field("credential", &"<redacted>")
            .finish()
    }
}
impl AnthropicSubscriptionAuth {
    /// Load only this provider's credential.
    pub fn new(store: AuthStore) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let credential = store.anthropic_subscription()?;
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| fail("could not create HTTP client"))?;
        Ok(Self {
            store,
            http,
            endpoints: Default::default(),
            credential: Arc::new(Mutex::new(credential)),
            refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }
    /// Use the normal private credential path.
    pub fn from_default() -> Result<Self> {
        Self::new(AuthStore::default())
    }
    /// Override endpoints for protocol tests.
    pub fn with_endpoints(mut self, endpoints: AnthropicSubscriptionEndpoints) -> Self {
        self.endpoints = endpoints;
        self
    }
    pub fn credential(&self) -> Result<Option<AnthropicSubscriptionCredential>> {
        if let Some(value) = self
            .credential
            .lock()
            .map_err(|_| AuthError::AnthropicSubscription("credential lock poisoned".into()))?
            .clone()
        {
            return Ok(Some(value));
        }
        let value = self.store.anthropic_subscription()?;
        if let Some(ref credential) = value {
            *self.credential.lock().map_err(|_| {
                AuthError::AnthropicSubscription("credential lock poisoned".into())
            })? = Some(credential.clone());
        }
        Ok(value)
    }
    pub async fn ensure_valid(&self) -> Result<AnthropicSubscriptionCredential> {
        let credential = self
            .credential()?
            .ok_or(AuthError::AnthropicSubscriptionNotAuthenticated)?;
        if !credential.is_expired() {
            // A separate Harness process may have refreshed the same file
            // while this handle's cached token was still unexpired. Check
            // for a newer valid generation before returning the cache so a
            // handle does not keep using an obsolete access token until its
            // old expiry.
            if let Some(current) = self.adopt_newer_on_disk(&credential)? {
                return Ok(current);
            }
            return Ok(credential);
        }
        self.refresh().await
    }

    /// Best-effort adoption of a newer persisted generation. Read failures
    /// intentionally leave a usable cached credential in place, matching the
    /// refresh recheck's existing behavior; a refresh or later call can retry.
    fn adopt_newer_on_disk(
        &self,
        cached: &AnthropicSubscriptionCredential,
    ) -> Result<Option<AnthropicSubscriptionCredential>> {
        let Ok(Some(current)) = self.store.anthropic_subscription() else {
            return Ok(None);
        };
        if current.is_newer_generation_than(cached) {
            *self.credential.lock().map_err(|_| {
                AuthError::AnthropicSubscription("credential lock poisoned".into())
            })? = Some(current.clone());
            return Ok(Some(current));
        }
        Ok(None)
    }

    pub async fn refresh(&self) -> Result<AnthropicSubscriptionCredential> {
        // Single-flight: concurrent refreshers queue here, then recheck the
        // credential before refreshing so only the first waiter hits the
        // network.  Never hold the blocking cache mutex across the network;
        // the async guard is released before any `.await` on the mutex.
        let _guard = self.refresh_lock.lock().await;
        // Reload: another waiter (or process) may have already refreshed.
        let mut expired_stored = None;
        if let Ok(Some(current)) = self.store.anthropic_subscription() {
            let cached = self
                .credential
                .lock()
                .map_err(|_| AuthError::AnthropicSubscription("credential lock poisoned".into()))?
                .clone();
            // Prefer a newer valid disk generation over the cache. A
            // generation can change one field at a time: rotating only the
            // refresh token, replacing only the access token, extending its
            // expiry, or updating its account identity must all be adopted.
            // The comparison also prevents an older cached expiry from
            // forcing an unnecessary exchange.
            if let Some(cached) = cached {
                if current.is_newer_generation_than(&cached) {
                    *self.credential.lock().map_err(|_| {
                        AuthError::AnthropicSubscription("credential lock poisoned".into())
                    })? = Some(current.clone());
                    return Ok(current);
                }
                if !cached.is_expired() {
                    return Ok(cached);
                }
            } else if !current.is_expired() {
                *self.credential.lock().map_err(|_| {
                    AuthError::AnthropicSubscription("credential lock poisoned".into())
                })? = Some(current.clone());
                return Ok(current);
            }
            // Both copies have expired. The on-disk generation is still
            // authoritative for the refresh token: another process may
            // have rotated it since this handle cached its old credential.
            expired_stored = Some(current);
        }
        let old = match expired_stored {
            Some(current) => current,
            None => self
                .credential()?
                .ok_or(AuthError::AnthropicSubscriptionNotAuthenticated)?,
        };
        // Refresh exchanges use strict status handling like the browser
        // flow: only 2xx with a token payload succeeds.
        let value = self
            .token(
                json!({"grant_type":"refresh_token", "refresh_token":old.refresh, "client_id": CLIENT_ID} ),
                &CancellationToken::new(),
            )
            .await;
        let value = match value {
            Ok(value) => value,
            Err(error) => {
                if let Some(current) = self.adopt_newer_on_disk(&old)? {
                    return Ok(current);
                }
                return Err(error);
            }
        };
        let credential = credential_from_token(&value, Some(&old))?;
        // Compare and save under the auth-file lock. A separate process may
        // have rotated the credential while this exchange was in flight; in
        // that case never overwrite its newer generation.
        if self
            .store
            .save_anthropic_subscription_if_current(&old, &credential)?
        {
            *self.credential.lock().map_err(|_| {
                AuthError::AnthropicSubscription("credential lock poisoned".into())
            })? = Some(credential.clone());
            return Ok(credential);
        }
        let current = self.store.anthropic_subscription()?.ok_or_else(|| {
            AuthError::AnthropicSubscription("credential changed during refresh".into())
        })?;
        if current.is_expired() {
            return Err(fail("credential changed during refresh; retry login"));
        }
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::AnthropicSubscription("credential lock poisoned".into()))? =
            Some(current.clone());
        Ok(current)
    }
    fn persist(&self, credential: AnthropicSubscriptionCredential) -> Result<()> {
        self.store.save_anthropic_subscription(&credential)?;
        *self
            .credential
            .lock()
            .map_err(|_| AuthError::AnthropicSubscription("credential lock poisoned".into()))? =
            Some(credential);
        Ok(())
    }

    /// Browser PKCE flow. No device-code protocol is known for this provider.
    pub async fn login_browser<F>(
        &self,
        cancel: &CancellationToken,
        mut emit: F,
    ) -> Result<AnthropicSubscriptionCredential>
    where
        F: FnMut(AuthEvent) + Send,
    {
        let listener=TcpListener::bind(("127.0.0.1",54545)).await.map_err(|_|fail("callback port 54545 is occupied; close the other login and retry (or forward localhost:54545 over SSH)"))?;
        let values = pkce()?;
        let mut url = Url::parse(&self.endpoints.authorize_url)
            .map_err(|_| fail("invalid authorize endpoint"))?;
        url.query_pairs_mut().extend_pairs([("response_type","code"),("client_id",CLIENT_ID),("redirect_uri",REDIRECT),("scope","org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"),("code_challenge_method","S256"),("code_challenge",&values.challenge),("state",&values.state)]);
        emit(AuthEvent::Started);
        emit(AuthEvent::Prompt {
            message: url.into(),
        });
        let code = tokio::select! { _=cancel.cancelled()=>return Err(AuthError::Cancelled), result=tokio::time::timeout(Duration::from_secs(600),wait_callback(listener,&values.state))=> result.map_err(|_|fail("browser login timed out"))?? };
        let value=self.token(json!({"grant_type":"authorization_code","client_id":CLIENT_ID,"code":code,"state":values.state,"code_verifier":values.verifier,"redirect_uri":REDIRECT}),cancel).await?;
        let credential = credential_from_token(&value, None)?;
        self.persist(credential.clone())?;
        emit(AuthEvent::Finished);
        Ok(credential)
    }
    async fn token(&self, body: Value, cancel: &CancellationToken) -> Result<Value> {
        let task = async {
            let mut response = self
                .http
                .post(&self.endpoints.token_url)
                .header("accept", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|_| fail("token request failed"))?;
            if !response.status().is_success() {
                return Err(fail(
                    "token exchange rejected; run harness login anthropic-claude again",
                ));
            }
            let mut bytes = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|_| fail("token response failed"))?
            {
                if bytes.len() + chunk.len() > 65536 {
                    return Err(fail("token response too large"));
                }
                bytes.extend_from_slice(&chunk);
            }
            serde_json::from_slice(&bytes).map_err(|_| fail("invalid token JSON"))
        };
        tokio::select! { _=cancel.cancelled()=>Err(AuthError::Cancelled), value=tokio::time::timeout(Duration::from_secs(30),task)=>value.map_err(|_|fail("token exchange timed out"))? }
    }
}
fn fail(message: &str) -> AuthError {
    AuthError::AnthropicSubscription(message.into())
}
fn credential_from_token(
    value: &Value,
    old: Option<&AnthropicSubscriptionCredential>,
) -> Result<AnthropicSubscriptionCredential> {
    let access = value["access_token"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| fail("missing access token"))?;
    let refresh = value["refresh_token"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| old.map(|c| c.refresh.as_str()))
        .ok_or_else(|| fail("missing refresh token"))?;
    let seconds = value["expires_in"]
        .as_u64()
        .filter(|n| *n > 60 && *n <= 365 * 86400)
        .ok_or_else(|| fail("invalid token expiry"))?;
    let expires = crate::storage::unix_millis().saturating_add(seconds * 1000);
    let mut c = AnthropicSubscriptionCredential::new(
        access,
        refresh,
        expires,
        value["account"]["uuid"].as_str().unwrap_or_default(),
    );
    c.email = value["account"]["email_address"]
        .as_str()
        .map(str::to_owned);
    c.org_id = value["organization"]["uuid"].as_str().map(str::to_owned);
    c.org_name = value["organization"]["name"].as_str().map(str::to_owned);
    if let Some(old) = old {
        c.account_id = old.account_id.clone();
        c.email = old.email.clone();
        c.org_id = old.org_id.clone();
        c.org_name = old.org_name.clone();
    }
    Ok(c)
}
/// Parse only an exact callback path and a unique, locally validated state.
fn callback_code(target: &str, state: &str) -> Result<Option<String>> {
    let url =
        Url::parse(&format!("http://localhost{target}")).map_err(|_| fail("invalid callback"))?;
    if url.path() != "/callback" {
        return Ok(None);
    }
    let pairs: Vec<_> = url.query_pairs().collect();
    let one = |key: &str| {
        let mut matches = pairs.iter().filter(|(k, _)| k == key);
        let first = matches.next().map(|(_, v)| v.to_string());
        if matches.next().is_some() {
            None
        } else {
            first
        }
    };
    if one("state").as_deref() != Some(state) {
        return Ok(None);
    }
    if one("error").is_some() {
        return Err(fail("authorization denied"));
    }
    Ok(one("code").filter(|s| !s.trim().is_empty()))
}
async fn wait_callback(listener: TcpListener, state: &str) -> Result<String> {
    loop {
        let (mut socket, _) = listener
            .accept()
            .await
            .map_err(|_| fail("callback listener failed"))?;
        let read = async {
            let mut bytes = Vec::new();
            loop {
                let mut buf = [0u8; 512];
                let n = socket.read(&mut buf).await.ok()?;
                if n == 0 {
                    return None;
                }
                bytes.extend_from_slice(&buf[..n]);
                if bytes.len() > 16384 {
                    return None;
                }
                if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                    return String::from_utf8(bytes).ok();
                }
            }
        };
        let request = tokio::time::timeout(Duration::from_secs(5), read)
            .await
            .ok()
            .flatten();
        let outcome = if let Some(request) = request {
            let mut words = request
                .lines()
                .next()
                .unwrap_or_default()
                .split_whitespace();
            if words.next() == Some("GET") {
                callback_code(words.next().unwrap_or_default(), state)
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        };
        let success = matches!(&outcome, Ok(Some(_)));
        let reply = if success {
            "HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\nLogin complete."
        } else {
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        };
        let _ =
            tokio::time::timeout(Duration::from_secs(1), socket.write_all(reply.as_bytes())).await;
        if let Some(code) = outcome? {
            return Ok(code);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn old() -> AnthropicSubscriptionCredential {
        AnthropicSubscriptionCredential::new("old-access", "old-refresh", 1, "account")
    }
    async fn fixture(
        status: u16,
        body: String,
    ) -> (
        AnthropicSubscriptionEndpoints,
        Arc<AtomicUsize>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let task = tokio::spawn(async move {
            loop {
                let (mut s, _) = listener.accept().await.unwrap();
                let mut all = Vec::new();
                loop {
                    let mut b = [0u8; 4096];
                    let n = s.read(&mut b).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    all.extend_from_slice(&b[..n]);
                    if let Some(pos) = all.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&all[..pos]).to_ascii_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length: ")
                                    .and_then(|v| v.parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if all.len() >= pos + 4 + len {
                            break;
                        }
                    }
                }
                c.fetch_add(1, Ordering::SeqCst);
                s.write_all(format!("HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
            }
        });
        (
            AnthropicSubscriptionEndpoints {
                authorize_url: format!("http://{addr}/authorize"),
                token_url: format!("http://{addr}/token"),
            },
            count,
            task,
        )
    }
    #[test]
    fn malformed_stored_credential_errors_are_redacted() {
        let dir = tempfile::tempdir().unwrap();
        let store = AuthStore::new(dir.path().join("auth.json"));
        // Serde normally quotes this string in an invalid-type diagnostic.
        store.save_provider_value("anthropic-subscription", json!({"type":"oauth", "access":"secret-access", "refresh":"secret-refresh", "expires":"secret-expiry", "accountId":"account"})).unwrap();
        let error = store.anthropic_subscription().unwrap_err();
        assert!(!format!("{error:?} {error}").contains("secret-"));
    }

    #[test]
    fn state_path_and_denial_are_strict() {
        assert_eq!(
            callback_code("/callback?code=c&state=good", "good").unwrap(),
            Some("c".into())
        );
        for target in [
            "/callback-extra?code=c&state=good",
            "/callback?code=c",
            "/callback?code=c&state=bad",
            "/callback?code=c&state=good&state=good",
            "/callback?code=c&code=d&state=good",
            "/callback?code=&state=good",
        ] {
            assert_eq!(callback_code(target, "good").unwrap(), None);
        }
        assert!(callback_code("/callback?error=access_denied&state=good", "good").is_err());
        assert!(
            callback_code("/callback?error=access_denied&state=bad", "good")
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn token_validation_and_identity_preservation() {
        let mut c = old();
        c.org_id = Some("org".into());
        let valid = json!({"access_token":"new-access","refresh_token":"rotated","expires_in":3600,"account":{"uuid":"wrong"},"organization":{"uuid":"wrong"}});
        let next = credential_from_token(&valid, Some(&c)).unwrap();
        assert_eq!(next.account_id, "account");
        assert_eq!(next.org_id, c.org_id);
        assert_eq!(next.refresh, "rotated");
        assert!(!next.is_expired());
        for value in [
            json!({}),
            json!({"access_token":"s","refresh_token":"r","expires_in":0}),
            json!({"access_token":"s","refresh_token":"r","expires_in":-1}),
            json!({"access_token":" ","refresh_token":"r","expires_in":3600}),
        ] {
            assert!(credential_from_token(&value, None).is_err());
        }
        let debug = format!("{next:?}");
        assert!(!debug.contains("new-access"));
        assert!(!debug.contains("rotated"));
    }
    #[tokio::test]
    async fn refresh_single_flight_rotation_disk_adoption_and_private_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = AuthStore::new(dir.path().join("auth.json"));
        store.save_anthropic_subscription(&old()).unwrap();
        store
            .save_provider_value("unrelated", json!({"keep":true}))
            .unwrap();
        let (endpoints, count, task) = fixture(
            200,
            json!({"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600})
                .to_string(),
        )
        .await;
        let auth = AnthropicSubscriptionAuth::new(store.clone())
            .unwrap()
            .with_endpoints(endpoints);
        let (a, b) = tokio::join!(auth.ensure_valid(), auth.ensure_valid());
        assert_eq!(a.unwrap(), b.unwrap());
        assert_eq!(count.load(Ordering::SeqCst), 1);
        let mut next = store.anthropic_subscription().unwrap().unwrap();
        assert_eq!(next.refresh, "new-refresh");
        assert_eq!(next.account_id, "account");
        next.refresh = "disk-rotation".into();
        store.save_anthropic_subscription(&next).unwrap();
        assert_eq!(auth.ensure_valid().await.unwrap(), next);
        assert!(
            !store
                .save_anthropic_subscription_if_current(&old(), &next)
                .unwrap()
        );
        let value: Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(value["unrelated"]["keep"], true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(store.path())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        task.abort();
    }
    #[tokio::test]
    async fn rejected_or_malformed_token_bodies_never_escape() {
        for (status, body) in [
            (400, "secret-token".to_owned()),
            (200, "{not-json secret-token".into()),
            (200, "x".repeat(65537)),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let store = AuthStore::new(dir.path().join("auth.json"));
            store.save_anthropic_subscription(&old()).unwrap();
            let (endpoints, _, task) = fixture(status, body).await;
            let auth = AnthropicSubscriptionAuth::new(store)
                .unwrap()
                .with_endpoints(endpoints);
            let error = auth.ensure_valid().await.unwrap_err().to_string();
            assert!(!error.contains("secret-token"));
            assert!(!error.contains("old-refresh"));
            task.abort();
        }
    }
    #[tokio::test]
    async fn callback_wait_is_cancellable_and_listener_is_released() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        tokio::select! {_ = cancel.cancelled()=>{},_ = wait_callback(listener,"state")=>panic!("unexpected callback")};
        assert!(TcpListener::bind(addr).await.is_ok());
    }
}
