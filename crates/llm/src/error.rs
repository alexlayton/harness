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
}

impl LlmError {
    /// Whether it is safe for the caller to repeat the initial request.
    pub fn is_retryable(&self) -> bool {
        match self {
            // A Network error is only produced before an HTTP response exists.  It
            // is therefore safe to retry it (and covers connect and timeout errors).
            Self::Network(_) => true,
            Self::Http { status, .. } => *status == 429 || (500..=599).contains(status),
            Self::Stream(_) | Self::Parse(_) => false,
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
