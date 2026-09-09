use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http {status}: {body}")]
    Http {
        status: u16,
        body: String,
        retry_after_secs: Option<u64>,
    },
    #[error("network: {0}")]
    Network(#[from] reqwest::Error),
    #[error("stream: {0}")]
    Stream(String),
    #[error("parse: {0}")]
    Parse(String),
    /// Missing or expired credentials (e.g. GitHub Copilot).  Callers can use
    /// this to surface an actionable standalone login command instead of a generic
    /// HTTP or network error.  Never retryable: retrying cannot refresh the
    /// credential.
    #[error("auth: {0}")]
    Auth(String),
}

impl LlmError {
    /// Whether it is safe for the caller to repeat the initial request.
    pub fn is_retryable(&self) -> bool {
        match self {
            // Network errors cover connect failures, timeouts, and — since
            // provider clients now set `read_timeout` — a streaming response
            // body that went silent mid-turn.  Repeating the request is safe
            // in all of those cases: nothing was charged against a partially
            // consumed stream, and the agent's mid-stream recovery relies on
            // this classification to re-stream automatically.
            Self::Network(_) => true,
            Self::Http { status, .. } => *status == 429 || (500..=599).contains(status),
            Self::Auth(_) | Self::Stream(_) | Self::Parse(_) => false,
        }
    }

    /// Bounded `Retry-After` hint (seconds) for 429 responses, if present.
    /// Parses both delta-seconds and HTTP-date forms; unparseable or absurd
    /// values yield `None` so callers fall back to backoff.  Capped so a
    /// malicious header cannot park the retry loop.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::Http {
                status: 429,
                retry_after_secs,
                ..
            } => *retry_after_secs,
            _ => None,
        }
    }

    pub fn http(status: u16, body: impl Into<String>) -> Self {
        Self::Http {
            status,
            body: truncate_body(&body.into(), 2048),
            retry_after_secs: None,
        }
    }

    /// Like [`Self::http`], but redacts the active API key/token before the
    /// error becomes UI-, persistence-, or log-visible.  This is the single
    /// redaction point for provider errors; per-provider `redact_*` helpers
    /// must delegate here rather than reimplementing substitution.
    pub fn http_redacted(status: u16, body: impl Into<String>, secret: &str) -> Self {
        Self::http_redacted_with_retry_after(status, body, secret, None)
    }

    /// Construct a bounded HTTP error while preserving the parsed retry hint
    /// separately from attacker-controlled response text.
    pub fn http_redacted_with_retry_after(
        status: u16,
        body: impl Into<String>,
        secret: &str,
        retry_after_secs: Option<u64>,
    ) -> Self {
        let body = body.into();
        let body = if secret.is_empty() {
            body
        } else {
            body.replace(secret, "[redacted]")
        };
        Self::Http {
            status,
            body: truncate_body(&body, 2048),
            retry_after_secs,
        }
    }

    /// Redact a secret from any error variant (stream/parse/auth bodies can
    /// echo tokens from misbehaving proxies).  Network errors carry no body
    /// and pass through unchanged.
    pub fn redacted(self, secret: &str) -> Self {
        if secret.is_empty() {
            return self;
        }
        match self {
            Self::Http {
                status,
                body,
                retry_after_secs,
            } => Self::Http {
                status,
                body: body.replace(secret, "[redacted]"),
                retry_after_secs,
            },
            Self::Stream(message) => Self::Stream(message.replace(secret, "[redacted]")),
            Self::Parse(message) => Self::Parse(message.replace(secret, "[redacted]")),
            Self::Auth(message) => Self::Auth(message.replace(secret, "[redacted]")),
            other => other,
        }
    }
}

/// Keep provider error bodies useful without allowing a proxy to fill the TUI or
/// a log file with an unbounded response.  The result always satisfies
/// `len() <= max_bytes` (or is empty when even the ellipsis cannot fit):
/// truncation cuts at a UTF-8 boundary *before* appending the suffix, so
/// the suffix never pushes the result over the cap.
pub fn truncate_body(body: &str, max_bytes: usize) -> String {
    const SUFFIX: &str = "\n[truncated]";
    if body.len() <= max_bytes {
        return body.to_owned();
    }
    if max_bytes <= SUFFIX.len() {
        return String::new();
    }
    let mut end = max_bytes - SUFFIX.len();
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{SUFFIX}", &body[..end])
}

/// Maximum `Retry-After` delay honored (5 minutes); larger values fall back
/// to backoff so a malicious header cannot park the loop.
const MAX_RETRY_AFTER_SECS: u64 = 300;

/// Parse a bounded `Retry-After` header value as seconds. Both delta-seconds
/// and HTTP-date forms are accepted; unparseable or over-cap values return
/// `None`.
pub(crate) fn parse_retry_after_value(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(secs) = value.parse::<u64>()
        && secs <= MAX_RETRY_AFTER_SECS
    {
        return Some(secs);
    }
    // HTTP-date form: delay until that instant, bounded and non-negative.
    if let Ok(date) = httpdate::parse_http_date(value) {
        let now = std::time::SystemTime::now();
        let delay = date.duration_since(now).unwrap_or_default().as_secs();
        if delay <= MAX_RETRY_AFTER_SECS {
            return Some(delay);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_errors_are_not_retryable() {
        assert!(!LlmError::Auth("harness login github-copilot".into()).is_retryable());
        assert!(!LlmError::Parse("bad".into()).is_retryable());
        assert!(!LlmError::Stream("gone".into()).is_retryable());
        assert!(LlmError::http(429, "rate limited").is_retryable());
        assert!(LlmError::http(500, "server error").is_retryable());
        assert!(!LlmError::http(401, "unauthorized").is_retryable());
    }

    #[test]
    fn error_bodies_echoing_a_key_are_redacted() {
        let secret = "sk-test-secret-123";
        let error = LlmError::http(500, format!("boom {secret} happened"));
        let redacted = error.redacted(secret);
        assert!(!redacted.to_string().contains(secret));
        let direct = LlmError::http_redacted(401, format!("token {secret}"), secret);
        assert!(!direct.to_string().contains(secret));
        assert!(direct.to_string().contains("[redacted]"));
    }

    #[test]
    fn error_bodies_stay_bounded_and_utf8() {
        // `truncate_body` never splits a char and never exceeds the cap:
        // "aéaéaé" is 9 bytes; cap 4 keeps at most the suffix budget logic.
        for max in [0usize, 1, 2, 5, 8, 12, 16, 64] {
            let out = truncate_body("aéaéaé hello world", max);
            assert!(out.len() <= max, "max={max} got {out:?}");
            assert!(out.is_char_boundary(out.len()));
        }
        assert_eq!(truncate_body("hello", 5), "hello");
        // The 2048-byte cap holds for multi-megabyte bodies.
        let big = "x".repeat(4 * 1024 * 1024);
        assert!(LlmError::http(500, big).to_string().len() <= 2048 + 64);
    }

    #[test]
    fn retry_after_parses_seconds_dates_and_caps() {
        assert_eq!(
            LlmError::http_redacted_with_retry_after(429, "busy", "", Some(5)).retry_after_secs(),
            Some(5)
        );
        assert_eq!(LlmError::http(429, "busy").retry_after_secs(), None);
        // Over the cap falls back to backoff.
        assert_eq!(
            LlmError::http(429, "busy\nretry-after: 99999").retry_after_secs(),
            None
        );
        // HTTP-date form parses when near-future and bounded.
        let soon = httpdate::fmt_http_date(
            std::time::SystemTime::now() + std::time::Duration::from_secs(30),
        );
        let delay = parse_retry_after_value(&soon).unwrap();
        assert!(delay <= 30 + 1, "delay {delay} should be ~30s");
        // Non-429 errors carry no hint.
        assert_eq!(LlmError::http(500, "x").retry_after_secs(), None);
    }
}
