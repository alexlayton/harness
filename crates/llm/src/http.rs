use crate::LlmError;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::Value;
use std::time::Duration;

/// How long a provider connection may take to establish (DNS + TCP + TLS).
/// Bounded so a black-holed connect fails fast instead of parking the turn.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum silence allowed on an in-flight response.  `read_timeout` is a
/// per-read idle timeout: every received chunk (including SSE keep-alive
/// comments such as `: OPENROUTER PROCESSING` or Anthropic `ping` events)
/// resets it, so it never bounds how long a healthy stream may run — only
/// how long it may go quiet.  Without it a stalled SSE connection (server
/// stops sending but keeps the socket open, as seen with free-tier
/// OpenRouter upstreams) hangs `stream.next()` forever and the only way out
/// is a user interrupt.
pub const READ_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// The reqwest client shared by every provider dialect.
///
/// A `read_timeout` (not a total `timeout`) is essential here: a total
/// timeout would abort legitimate multi-minute generations, while the idle
/// timeout fires only on a stream that has gone completely silent.
///
/// Panics only if reqwest cannot initialize its TLS backend, which is a
/// configuration failure worth failing loudly rather than silently
/// reverting to an unbounded client.
pub fn streaming_client() -> Client {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_IDLE_TIMEOUT)
        .build()
        .expect("reqwest client with rustls should initialize")
}

/// Shared HTTP plumbing for the OpenAI-compatible request clients
/// (`OpenAiChatClient`, `OpenAiResponsesClient`).
///
/// Both clients differ only in the endpoint path and the request body builder,
/// so the network, error-mapping, and header logic lives here once.
#[derive(Clone)]
pub struct HttpClient {
    http: Client,
    pub base_url: String,
    pub api_key: String,
    extra_headers: HeaderMap,
}

impl HttpClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::with_headers(base_url, api_key, HeaderMap::new())
    }

    pub fn with_headers(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
    ) -> Self {
        Self::with_client(base_url, api_key, extra_headers, streaming_client())
    }

    /// Construct a client around an explicitly provided HTTP client.  Used by
    /// [`with_headers`] in production and by tests that need short timeouts.
    pub fn with_client(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        extra_headers: HeaderMap,
        http: Client,
    ) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            extra_headers,
        }
    }

    /// Standard bearer + JSON headers for OpenAI-compatible endpoints.
    pub fn headers(&self) -> HeaderMap {
        let mut headers = self.extra_headers.clone();
        if let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", self.api_key)) {
            headers.insert(AUTHORIZATION, value);
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }

    /// POST `body` as JSON to `path` (joined to `base_url`).  Non-success
    /// statuses are mapped to `LlmError::Http` with a bounded error body.
    pub async fn post_json(&self, path: &str, body: &Value) -> Result<reqwest::Response, LlmError> {
        let response = self
            .http
            .post(format!("{}{}", self.base_url, path))
            .headers(self.headers())
            .json(body)
            .send()
            .await
            .map_err(LlmError::Network)?;
        check_status(response).await
    }

    /// GET `path` with the standard headers.
    pub async fn get(&self, path: &str) -> Result<reqwest::Response, LlmError> {
        let response = self
            .http
            .get(format!("{}{}", self.base_url, path))
            .headers(self.headers())
            .send()
            .await
            .map_err(LlmError::Network)?;
        check_status(response).await
    }
}

/// Map a non-success HTTP status to `LlmError::Http` with a bounded body.
/// `LlmError::http` is the single constructor that enforces the truncation
/// invariant, so every status-mapped error (OpenAI dialects and Anthropic
/// alike) is bounded here.
pub(crate) async fn check_status(
    response: reqwest::Response,
) -> Result<reqwest::Response, LlmError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<unable to read response body>".into());
    Err(LlmError::http(status, body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use serde_json::json;

    /// Regression test for the "harness hangs mid-turn" bug: a provider whose
    /// SSE connection stalls (response headers arrive, then nothing — the
    /// socket stays open) used to hang `stream.next()` forever because the
    /// reqwest client had no timeouts.  With `read_timeout` set, the stalled
    /// body read must surface as an error so the agent loop can recover
    /// instead of waiting for a user interrupt.
    #[tokio::test]
    async fn stalled_response_body_times_out_instead_of_hanging() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("listener address");

        // Accept one request, reply with SSE-shaped response headers, then go
        // completely silent while keeping the connection open.  The request
        // body is never read: it is small enough to sit in the socket buffers.
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut scratch = [0u8; 1024];
            // Drain enough of the request head to let the client finish.
            let _ = socket.read(&mut scratch).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
                )
                .await
                .expect("write response head");
            // Hold the connection open without sending any body bytes.
            let _ = socket.read(&mut scratch).await;
        });

        let stalled = Client::builder()
            // Short idle timeout so the test runs in milliseconds; production
            // uses READ_IDLE_TIMEOUT via `streaming_client()`.
            .read_timeout(Duration::from_millis(300))
            .build()
            .expect("build test client");
        let client = HttpClient::with_client(
            format!("http://{addr}"),
            "test-key",
            HeaderMap::new(),
            stalled,
        );

        // The response headers arrive, so this resolves successfully...
        let response = client
            .post_json("/chat/completions", &json!({"model": "demo"}))
            .await
            .expect("response headers should arrive");

        // ...but the first body read must time out rather than hang forever.
        let started = std::time::Instant::now();
        let first = response.bytes_stream().next().await;
        assert!(
            first.as_ref().is_some_and(|item| item.is_err()),
            "silent stream should error, got {first:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "timeout should fire promptly, took {:?}",
            started.elapsed()
        );
    }
}
