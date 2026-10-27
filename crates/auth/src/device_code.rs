//! RFC 8628 device-flow primitives shared by the Copilot implementation.

use crate::error::{AuthError, Result};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use url::Url;

/// Data displayed to the user after GitHub starts device authorization.
#[derive(Clone, PartialEq, Eq)]
pub struct DeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_url: String,
    pub expires_in: u64,
    pub interval: u64,
}

impl std::fmt::Debug for DeviceCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceCode")
            .field("device_code", &"<redacted>")
            .field("user_code", &"<redacted>")
            .field("verification_url", &self.verification_url)
            .field("expires_in", &self.expires_in)
            .field("interval", &self.interval)
            .finish()
    }
}

/// Events emitted while a device flow is running.  The auth crate does not
/// know anything about a terminal; callers decide how to render these values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthEvent {
    Started,
    Prompt {
        message: String,
    },
    DeviceCode {
        verification_url: String,
        user_code: String,
        expires_in: u64,
        interval: u64,
    },
    Progress {
        message: String,
    },
    Finished,
    Failed {
        message: String,
    },
}

/// A result from one OAuth token poll.
#[derive(Clone, PartialEq, Eq)]
pub enum PollResult {
    Pending,
    SlowDown { interval: Option<u64> },
    Complete { access_token: String },
}

impl std::fmt::Debug for PollResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => formatter.write_str("Pending"),
            Self::SlowDown { interval } => formatter
                .debug_struct("SlowDown")
                .field("interval", interval)
                .finish(),
            Self::Complete { .. } => formatter
                .debug_struct("Complete")
                .field("access_token", &"<redacted>")
                .finish(),
        }
    }
}

/// Longest device-code lifetime honored (GitHub uses 15 minutes).  An
/// untrusted larger value is clamped so deadline arithmetic cannot overflow
/// and a login cannot wait for days.
pub const MAX_DEVICE_CODE_LIFETIME_SECS: u64 = 60 * 60;

/// Polling interval bounds (RFC 8628 default is 5 seconds).  The upper bound
/// keeps `slow_down` increments and untrusted intervals from overflowing or
/// parking the login.
pub const MIN_POLL_INTERVAL_SECS: u64 = 5;
pub const MAX_POLL_INTERVAL_SECS: u64 = 120;

/// Clamp an untrusted polling interval into the supported range.
pub fn clamp_poll_interval(interval: u64) -> u64 {
    interval.clamp(MIN_POLL_INTERVAL_SECS, MAX_POLL_INTERVAL_SECS)
}

pub fn parse_device_code(value: &Value) -> Result<DeviceCode> {
    let object = value
        .as_object()
        .ok_or_else(|| AuthError::InvalidDeviceCode("response is not an object".into()))?;
    let device_code = required_string(object.get("device_code"), "device_code")?;
    let user_code = required_string(object.get("user_code"), "user_code")?;
    let verification = object
        .get("verification_uri")
        .or_else(|| object.get("verification_url"))
        .and_then(Value::as_str)
        .ok_or_else(|| AuthError::InvalidDeviceCode("missing verification_uri".into()))?;
    let verification_url = validate_verification_url(verification)?;
    let expires_in = required_u64(object.get("expires_in"), "expires_in")?;
    if expires_in == 0 {
        return Err(AuthError::InvalidDeviceCode(
            "expires_in must be greater than zero".into(),
        ));
    }
    let interval = object
        .get("interval")
        .map(|value| parse_u64(value, "interval"))
        .transpose()?
        .filter(|value| *value > 0)
        .unwrap_or(5);
    Ok(DeviceCode {
        device_code,
        user_code,
        verification_url,
        expires_in: expires_in.min(MAX_DEVICE_CODE_LIFETIME_SECS),
        interval: clamp_poll_interval(interval),
    })
}

/// Only HTTP(S) verification links are accepted.  In particular, a malicious
/// GitHub-compatible endpoint must not make a caller open a `file:` or custom
/// executable URL.
pub fn validate_verification_url(value: &str) -> Result<String> {
    let parsed = Url::parse(value).map_err(|_| AuthError::UntrustedVerificationUrl)?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(AuthError::UntrustedVerificationUrl);
    }
    Ok(parsed.to_string())
}

pub fn required_string(value: Option<&Value>, field: &str) -> Result<String> {
    let value = value
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| AuthError::InvalidDeviceCode(format!("missing {field}")))?;
    Ok(value.to_owned())
}

pub fn required_u64(value: Option<&Value>, field: &str) -> Result<u64> {
    let value = value.ok_or_else(|| AuthError::InvalidDeviceCode(format!("missing {field}")))?;
    parse_u64(value, field)
}

pub fn parse_u64(value: &Value, field: &str) -> Result<u64> {
    if let Some(value) = value.as_u64() {
        return Ok(value);
    }
    if let Some(value) = value.as_i64() {
        return u64::try_from(value)
            .map_err(|_| AuthError::InvalidDeviceCode(format!("{field} must be positive")));
    }
    if let Some(value) = value.as_str() {
        return value
            .parse::<u64>()
            .map_err(|_| AuthError::InvalidDeviceCode(format!("{field} must be a number")));
    }
    Err(AuthError::InvalidDeviceCode(format!(
        "{field} must be a number"
    )))
}

/// Wait for a polling interval while still responding promptly to Ctrl+C.
/// Tokio-aware: uses `tokio::time::sleep`, so paused-time tests
/// auto-advance the virtual clock instead of sleeping wall-clock time.
/// Callers that also enforce a wall-clock expiry deadline must compute
/// that deadline from the same clock domain (see `login_device`).
pub async fn cancellable_sleep(seconds: u64, cancel: &CancellationToken) -> Result<()> {
    tokio::select! {
        _ = cancel.cancelled() => Err(AuthError::Cancelled),
        _ = tokio::time::sleep(std::time::Duration::from_secs(seconds)) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_device_code_and_defaults_poll_interval() {
        let device = parse_device_code(&json!({
            "device_code": "device",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900
        }))
        .unwrap();
        assert_eq!(device.interval, 5);
        assert_eq!(device.verification_url, "https://github.com/login/device");
    }

    #[test]
    fn clamps_untrusted_lifetime_and_interval() {
        // u64::MAX would overflow `Instant + Duration` and `interval + 5`.
        let device = parse_device_code(&json!({
            "device_code": "device",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
            "expires_in": u64::MAX,
            "interval": u64::MAX
        }))
        .unwrap();
        assert_eq!(device.expires_in, MAX_DEVICE_CODE_LIFETIME_SECS);
        assert_eq!(device.interval, MAX_POLL_INTERVAL_SECS);
    }

    #[test]
    fn rejects_untrusted_verification_urls() {
        for url in ["file:///tmp/run", "javascript:alert(1)", "not a url"] {
            assert!(matches!(
                validate_verification_url(url),
                Err(AuthError::UntrustedVerificationUrl)
            ));
        }
    }
}
