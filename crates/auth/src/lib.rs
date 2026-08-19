//! GitHub Copilot authentication for Harness.
//!
//! This crate is intentionally independent of `llm`, `agent`, and `tui`.
//! Callers receive neutral [`AuthEvent`] values and decide how to render or
//! queue input while the device flow is running.

pub mod device_code;
pub mod error;
pub mod github_copilot;
pub mod storage;

pub use device_code::{AuthEvent, DeviceCode, PollResult, validate_verification_url};
pub use error::{AuthError, Result};
pub use github_copilot::{
    COPILOT_API_VERSION, COPILOT_EDITOR_PLUGIN_VERSION, COPILOT_EDITOR_VERSION,
    COPILOT_INTEGRATION_ID, COPILOT_USER_AGENT, CopilotAuth, CopilotEndpoints,
    GITHUB_DEVICE_CLIENT_ID, GithubCopilotClient, KNOWN_MODEL_IDS, base_url_from_proxy_token,
    copilot_base_url, get_base_url_from_token, get_github_copilot_base_url, normalize_domain,
    parse_available_copilot_model_ids, parse_available_model_ids, parse_available_model_ids_value,
    parse_copilot_token, sku_from_proxy_token,
};
pub use storage::{
    AuthEntries, AuthStore, COPILOT_PROVIDER_KEY, CopilotCredential, RedactedCredential, auth_path,
    config_dir,
};
