use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Errors returned by credential storage and the GitHub Copilot login flow.
///
/// Error values intentionally never contain access or refresh token values.
/// Response bodies are not retained either: GitHub and Copilot error responses
/// are not a safe place to put secrets supplied by a proxy or an enterprise
/// installation.
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("{operation} {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// Serde messages can quote the offending value, which in the auth file
    /// is often a token. Only the error category and position are kept; the
    /// serde error is deliberately not retained as a source.
    #[error("invalid auth JSON in {}: {category} error at line {line} column {column}", path.display())]
    Json {
        path: PathBuf,
        category: &'static str,
        line: usize,
        column: usize,
    },

    #[error("invalid GitHub Copilot credential: {0}")]
    InvalidCredential(String),

    #[error("invalid GitHub domain: {0}")]
    InvalidDomain(String),

    #[error("invalid GitHub device-code response: {0}")]
    InvalidDeviceCode(String),

    #[error("untrusted verification URL returned by GitHub")]
    UntrustedVerificationUrl,

    #[error("device login expired before authorization completed")]
    DeviceCodeExpired,

    #[error("device login cancelled")]
    Cancelled,

    #[error("GitHub device login failed: {0}")]
    DeviceFlowFailed(String),

    #[error("HTTP {status} while contacting {endpoint}")]
    Http { status: u16, endpoint: String },

    #[error("network while contacting {endpoint}: {message}")]
    Network { endpoint: String, message: String },

    #[error("GitHub Copilot is not authenticated; run `harness login github-copilot`")]
    NotAuthenticated,

    #[error("OpenAI Codex is not authenticated; run `harness login openai-codex`")]
    OpenAiCodexNotAuthenticated,

    #[error("OpenAI Codex OAuth failed: {0}")]
    OpenAiCodex(String),

    #[error(
        "could not bind the OpenAI Codex login callback; use `harness login openai-codex --device-code`"
    )]
    CallbackBind,

    #[error("Anthropic subscription OAuth failed: {0}")]
    AnthropicSubscription(String),

    #[error("Anthropic subscription is not authenticated; run `harness login anthropic-claude`")]
    AnthropicSubscriptionNotAuthenticated,

    #[error("could not acquire auth file lock {}", .0.display())]
    LockUnavailable(PathBuf),
}

pub type Result<T> = std::result::Result<T, AuthError>;

pub(crate) fn io_error(
    operation: &'static str,
    path: impl Into<PathBuf>,
    source: io::Error,
) -> AuthError {
    AuthError::Io {
        operation,
        path: path.into(),
        source,
    }
}

impl From<reqwest::Error> for AuthError {
    fn from(error: reqwest::Error) -> Self {
        // Keep the URL out of this conversion.  A request URL should not carry
        // credentials, and omitting it also keeps error formatting stable for
        // enterprise proxies that include response text in Display.
        Self::Network {
            endpoint: "GitHub".into(),
            message: error
                .status()
                .map(|status| format!("HTTP status {}", status.as_u16()))
                .unwrap_or_else(|| "request failed".into()),
        }
    }
}

impl AuthError {
    /// Build [`AuthError::Json`] without any text from the serde error.
    pub(crate) fn json(path: &std::path::Path, error: &serde_json::Error) -> Self {
        use serde_json::error::Category;
        Self::Json {
            path: path.to_path_buf(),
            category: match error.classify() {
                Category::Io => "I/O",
                Category::Syntax => "syntax",
                Category::Data => "data",
                Category::Eof => "end-of-file",
            },
            line: error.line(),
            column: error.column(),
        }
    }
}
