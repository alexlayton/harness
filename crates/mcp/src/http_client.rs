//! Bounded Streamable HTTP client for MCP servers.
//!
//! rmcp's built-in `reqwest::Client` implementation reads JSON responses and
//! non-success bodies with `Response::json`/`Response::text`, so a server can
//! make Harness allocate an unbounded body before any Harness limit applies.
//! This client keeps the same protocol behavior but counts received bytes
//! (including chunked bodies without `Content-Length`) before it parses
//! anything, and it bounds each SSE event. GET streams and DELETE are already
//! bounded (or do not read a body) in rmcp, so they delegate to it.

use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use http::{HeaderName, HeaderValue};
use reqwest::header::{ACCEPT, CONTENT_TYPE, WWW_AUTHENTICATE};
use rmcp::model::{ClientJsonRpcMessage, JsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::common::http_header::{
    EVENT_STREAM_MIME_TYPE, HEADER_LAST_EVENT_ID, HEADER_MCP_PROTOCOL_VERSION, HEADER_SESSION_ID,
    JSON_MIME_TYPE,
};
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, SseError, StreamableHttpClient, StreamableHttpError,
    StreamableHttpPostResponse,
};
use sse_stream::{Sse, SseStream};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// Largest diagnostic excerpt of an error body kept in an error message.
const ERROR_EXCERPT_BYTES: usize = 2 * 1024;

type HttpError = StreamableHttpError<reqwest::Error>;

/// A `reqwest::Client` wrapper whose POST responses are read under a byte cap.
#[derive(Clone)]
pub(crate) struct BoundedHttpClient {
    inner: reqwest::Client,
    max_body_bytes: usize,
}

impl BoundedHttpClient {
    /// Match rmcp's default client: no idle pooling and no redirects, so
    /// custom headers (often credentials) are never replayed to another host.
    pub(crate) fn new(max_body_bytes: usize) -> Result<Self, reqwest::Error> {
        let inner = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            inner,
            max_body_bytes,
        })
    }
}

/// Outcome of a capped body read.
enum BoundedBody {
    Complete(Vec<u8>),
    TooLarge,
}

async fn read_bounded(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<BoundedBody, reqwest::Error> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Ok(BoundedBody::TooLarge);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if chunk.len() > limit.saturating_sub(body.len()) {
            return Ok(BoundedBody::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(BoundedBody::Complete(body))
}

/// Reject custom headers that would override protocol headers. Mirrors
/// rmcp's reserved list; `MCP-Protocol-Version` is allowed because the
/// worker injects it after initialization.
fn apply_custom_headers(
    mut builder: reqwest::RequestBuilder,
    custom_headers: HashMap<HeaderName, HeaderValue>,
) -> Result<reqwest::RequestBuilder, HttpError> {
    const RESERVED: &[&str] = &["accept", HEADER_SESSION_ID, HEADER_LAST_EVENT_ID];
    for (name, value) in custom_headers {
        if RESERVED
            .iter()
            .any(|reserved| name.as_str().eq_ignore_ascii_case(reserved))
            && !name
                .as_str()
                .eq_ignore_ascii_case(HEADER_MCP_PROTOCOL_VERSION)
        {
            return Err(StreamableHttpError::ReservedHeaderConflict(
                name.as_str().to_owned(),
            ));
        }
        builder = builder.header(name, value);
    }
    Ok(builder)
}

/// `scope="..."` from a `WWW-Authenticate` challenge, if present.
fn scope_from_challenge(header: &str) -> Option<String> {
    let start = header.find("scope=")? + "scope=".len();
    let rest = &header[start..];
    let scope = match rest.strip_prefix('"') {
        Some(quoted) => quoted.split('"').next()?,
        None => rest.split([',', ' ']).next()?,
    };
    (!scope.is_empty()).then(|| scope.to_owned())
}

fn challenge(response: &reqwest::Response) -> Result<Option<String>, HttpError> {
    response
        .headers()
        .get(WWW_AUTHENTICATE)
        .map(|header| {
            header.to_str().map(str::to_owned).map_err(|_| {
                StreamableHttpError::UnexpectedServerResponse(Cow::from(
                    "invalid www-authenticate header value",
                ))
            })
        })
        .transpose()
}

fn excerpt(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    let mut end = text.len().min(ERROR_EXCERPT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    if end < text.len() {
        format!("{}…", &text[..end])
    } else {
        text.into_owned()
    }
}

impl StreamableHttpClient for BoundedHttpClient {
    type Error = reqwest::Error;

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, HttpError> {
        self.get_stream_with_max_sse_event_size(
            uri,
            session_id,
            last_event_id,
            auth_header,
            custom_headers,
            self.max_body_bytes,
        )
        .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, HttpError> {
        // rmcp bounds every raw SSE event on GET streams.
        self.inner
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), HttpError> {
        // DELETE never reads the response body.
        self.inner
            .delete_session(uri, session_id, auth_header, custom_headers)
            .await
    }

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        self.post_message_with_max_sse_event_size(
            uri,
            message,
            session_id,
            auth_header,
            custom_headers,
            self.max_body_bytes,
        )
        .await
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        let body = serde_json::to_vec(&message)?;
        let mut request = self
            .inner
            .post(uri.as_ref())
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "))
            .header(CONTENT_TYPE, JSON_MIME_TYPE);
        if let Some(auth_header) = auth_header {
            request = request.bearer_auth(auth_header);
        }
        request = apply_custom_headers(request, custom_headers)?;
        let session_was_attached = session_id.is_some();
        if let Some(session_id) = session_id {
            request = request.header(HEADER_SESSION_ID, session_id.as_ref());
        }
        let response = request
            .body(body)
            .send()
            .await
            .map_err(StreamableHttpError::Client)?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED
            && let Some(header) = challenge(&response)?
        {
            return Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(
                header,
            )));
        }
        if status == reqwest::StatusCode::FORBIDDEN
            && let Some(header) = challenge(&response)?
        {
            let scope = scope_from_challenge(&header);
            return Err(StreamableHttpError::InsufficientScope(
                InsufficientScopeError::new(header, scope),
            ));
        }
        if matches!(
            status,
            reqwest::StatusCode::ACCEPTED | reqwest::StatusCode::NO_CONTENT
        ) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == reqwest::StatusCode::NOT_FOUND && session_was_attached {
            return Err(StreamableHttpError::SessionExpired);
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
        let response_session_id = response
            .headers()
            .get(HEADER_SESSION_ID)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        // Some servers answer notifications with an empty 200 instead of 202.
        if status.is_success()
            && response.content_length() == Some(0)
            && matches!(
                message,
                ClientJsonRpcMessage::Notification(_)
                    | ClientJsonRpcMessage::Response(_)
                    | ClientJsonRpcMessage::Error(_)
            )
        {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        let is_json = content_type
            .as_deref()
            .is_some_and(|value| value.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()));
        if !status.is_success() {
            let body = match read_bounded(response, self.max_body_bytes)
                .await
                .map_err(StreamableHttpError::Client)?
            {
                BoundedBody::Complete(body) => body,
                BoundedBody::TooLarge => {
                    return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                        format!(
                            "HTTP {status}: response body exceeds {} bytes",
                            self.max_body_bytes
                        ),
                    )));
                }
            };
            // Non-success responses may carry a JSON-RPC error that should
            // reach the caller as a protocol error.
            if is_json
                && let Ok(message @ JsonRpcMessage::Error(_)) =
                    serde_json::from_slice::<ServerJsonRpcMessage>(&body)
            {
                return Ok(StreamableHttpPostResponse::Json(
                    message,
                    response_session_id,
                ));
            }
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                format!("HTTP {status}: {}", excerpt(&body)),
            )));
        }
        match content_type.as_deref() {
            Some(value)
                if value
                    .as_bytes()
                    .starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()) =>
            {
                let stream = bounded_sse_bytes(response.bytes_stream(), max_sse_event_size);
                Ok(StreamableHttpPostResponse::Sse(
                    SseStream::from_bytes_stream(stream).boxed(),
                    response_session_id,
                ))
            }
            Some(_) if is_json => {
                let body = match read_bounded(response, self.max_body_bytes)
                    .await
                    .map_err(StreamableHttpError::Client)?
                {
                    BoundedBody::Complete(body) => body,
                    BoundedBody::TooLarge => {
                        return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                            format!("JSON response exceeds {} bytes", self.max_body_bytes),
                        )));
                    }
                };
                // Like rmcp: a malformed JSON body (for example a 200 reply
                // to a notification without an id) is treated as accepted.
                match serde_json::from_slice::<ServerJsonRpcMessage>(&body) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(
                        message,
                        response_session_id,
                    )),
                    Err(error) => {
                        tracing::warn!(
                            "could not parse MCP JSON response, treating as accepted: {error}"
                        );
                        Ok(StreamableHttpPostResponse::Accepted)
                    }
                }
            }
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum SseBodyError {
    #[error(transparent)]
    Source(reqwest::Error),
    #[error("SSE event exceeded the maximum size of {0} bytes")]
    EventTooLarge(usize),
}

/// Fail the byte stream as soon as one SSE event (the bytes between blank
/// lines, excluding comments) grows past `max_event_size`.
fn bounded_sse_bytes(
    stream: impl futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    max_event_size: usize,
) -> BoxStream<'static, Result<Bytes, SseBodyError>> {
    let mut limiter = SseEventSizeLimiter::new(max_event_size);
    stream
        .scan(false, move |failed, item| {
            let output = if *failed {
                None
            } else {
                Some(match item {
                    Ok(chunk) if limiter.observe(&chunk) => Ok(chunk),
                    Ok(_) => {
                        *failed = true;
                        Err(SseBodyError::EventTooLarge(max_event_size))
                    }
                    Err(error) => {
                        *failed = true;
                        Err(SseBodyError::Source(error))
                    }
                })
            };
            std::future::ready(output)
        })
        .boxed()
}

/// Tracks the retained size of the SSE event being parsed. Line endings may
/// be LF, CR, or CRLF; a blank line ends the event.
struct SseEventSizeLimiter {
    max_size: usize,
    retained_size: usize,
    line_size: usize,
    line_is_comment: bool,
    previous_was_cr: bool,
}

impl SseEventSizeLimiter {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            retained_size: 0,
            line_size: 0,
            line_is_comment: false,
            previous_was_cr: false,
        }
    }

    /// Returns `false` once the current event exceeds the limit.
    fn observe(&mut self, chunk: &[u8]) -> bool {
        for &byte in chunk {
            if self.previous_was_cr {
                self.previous_was_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' => {
                    self.finish_line();
                    self.previous_was_cr = true;
                }
                b'\n' => self.finish_line(),
                _ => {
                    if self.line_size == 0 {
                        self.line_is_comment = byte == b':';
                    }
                    self.line_size = self.line_size.saturating_add(1);
                }
            }
            if self.retained_size.saturating_add(self.line_size) > self.max_size {
                return false;
            }
        }
        true
    }

    fn finish_line(&mut self) {
        if self.line_size == 0 {
            self.retained_size = 0;
        } else if !self.line_is_comment {
            // The SSE parser joins multiple data lines with a newline.
            self.retained_size = self
                .retained_size
                .saturating_add(self.line_size)
                .saturating_add(1);
        }
        self.line_size = 0;
        self.line_is_comment = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_limiter_resets_per_event_and_rejects_oversized_events() {
        let mut limiter = SseEventSizeLimiter::new(20);
        assert!(limiter.observe(b"data: 0123456789\n\n"));
        assert!(limiter.observe(b"data: 0123456789\r\n\r\n"));
        // Comments do not count toward the retained event.
        assert!(limiter.observe(b": ping\n\n"));
        assert!(!limiter.observe(b"data: 0123456789abcdef\n"));
    }

    /// A chunked JSON body without `Content-Length` must be rejected once
    /// the received bytes pass the cap, not after it is fully allocated.
    #[tokio::test]
    async fn oversized_chunked_json_and_error_bodies_are_rejected() {
        use rmcp::model::{ClientRequest, PingRequest, RequestId};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        for status in ["200 OK", "500 Internal Server Error"] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 4096];
                let _ = socket.read(&mut request).await;
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n"
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                for _ in 0..8 {
                    let _ = socket.write_all(b"10\r\n0123456789abcdef\r\n").await;
                }
                let _ = socket.write_all(b"0\r\n\r\n").await;
            });
            let client = BoundedHttpClient::new(64).unwrap();
            let message = ClientJsonRpcMessage::request(
                ClientRequest::PingRequest(PingRequest::default()),
                RequestId::Number(1),
            );
            let error = client
                .post_message(
                    format!("http://{address}/mcp").into(),
                    message,
                    None,
                    None,
                    HashMap::new(),
                )
                .await
                .unwrap_err();
            assert!(
                matches!(&error, StreamableHttpError::UnexpectedServerResponse(text) if text.contains("exceeds")),
                "{status}: {error}"
            );
            server.abort();
        }
    }

    #[test]
    fn scope_is_read_from_quoted_and_bare_challenges() {
        assert_eq!(
            scope_from_challenge(r#"Bearer error="insufficient_scope", scope="a b""#).as_deref(),
            Some("a b")
        );
        assert_eq!(
            scope_from_challenge("Bearer scope=admin, realm=x").as_deref(),
            Some("admin")
        );
        assert_eq!(scope_from_challenge("Bearer realm=x"), None);
    }
}
