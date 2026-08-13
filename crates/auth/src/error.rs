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

    #[error("invalid auth JSON in {}: {source}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid GitHub Copilot credential: {0}")]
    InvalidCredential(String),

    #[error("invalid GitHub domain: {0}")]
    InvalidDomain(String),

    #[error("invalid GitHub device-code response: {0}")]
    InvalidDeviceCode(String),

    #[error("untrusted verification URL returned by GitHub")]
    UntrustedVerificationUrl,

    #[error("GitHub device login expired before authorization completed")]
    DeviceCodeExpired,

    #[error("GitHub device login cancelled")]
    Cancelled,

    #[error("GitHub device login failed: {0}")]
    DeviceFlowFailed(String),

    #[error("HTTP {status} while contacting {endpoint}")]
    Http { status: u16, endpoint: String },

    #[error("network while contacting {endpoint}: {message}")]
    Network { endpoint: String, message: String },

    #[error("GitHub Copilot is not authenticated; run /auth")]
    NotAuthenticated,

    #[error("could not acquire auth file lock {}", .0.display())]
    LockUnavailable(PathBuf),
}

pub type Result<T> = std::result::Result<T, AuthError>;

impl AuthError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

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
