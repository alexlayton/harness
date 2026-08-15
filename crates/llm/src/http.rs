use crate::LlmError;
use crate::error::truncate_body;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::Value;

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
        Self {
            http: Client::new(),
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
async fn check_status(response: reqwest::Response) -> Result<reqwest::Response, LlmError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<unable to read response body>".into());
    Err(LlmError::Http {
        status,
        body: truncate_body(&body, 2048),
    })
}
