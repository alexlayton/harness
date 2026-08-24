use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http {status}: {body}")]
    Http { status: u16, body: String },
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

    pub fn http(status: u16, body: impl Into<String>) -> Self {
        Self::Http {
            status,
            body: truncate_body(&body.into(), 2048),
        }
    }
}

/// Keep provider error bodies useful without allowing a proxy to fill the TUI or
/// a log file with an unbounded response.
pub fn truncate_body(body: &str, max_bytes: usize) -> String {
    if body.len() <= max_bytes {
        return body.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[truncated]", &body[..end])
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
}
