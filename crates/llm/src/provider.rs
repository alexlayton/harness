use crate::retry::with_retry;
use crate::{CompletionRequest, LlmError, ModelInfo, StreamEvent, SubscriptionUsage};
use futures_core::Stream;
use futures_util::StreamExt;
use std::pin::Pin;
use std::sync::Arc;

pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;
pub type RetryCallback = Arc<dyn for<'a> Fn(u32, &'a LlmError) + Send + Sync>;

/// Attach the active provider credential to an event stream's error boundary.
///
/// SSE parsing happens after a dialect's request method has returned, so
/// redacting only the `Result<EventStream, LlmError>` from that method leaves
/// provider-supplied error payloads exposed.  Every dialect driver uses this
/// adapter before returning its stream; it preserves the original error
/// variants and therefore does not affect retry classification.
pub(crate) fn redact_stream(stream: EventStream, secret: &str) -> EventStream {
    let secret = secret.to_owned();
    Box::pin(stream.map(move |item| item.map_err(|error| error.redacted(&secret))))
}

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Provider identifier used in sessions and logs.  Returns `&str` (rather
    /// than `&'static str`) so dynamically constructed providers (e.g. GitHub
    /// Copilot with a runtime-derived name) can return a stored `String`
    /// reference instead of leaking or registering statics.
    fn name(&self) -> &str;

    /// Perform a single request and return its event stream.  Retrying is the
    /// caller's concern via [`Provider::stream_with_retry`]; this default
    /// implementation wraps [`Provider::stream`] with the shared backoff and
    /// forwards retry notices to the callback.
    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Fetch the current subscription allowance, when this provider has a
    /// usage endpoint. Providers without subscription allowances return `None`.
    async fn subscription_usage(&self) -> Result<Option<SubscriptionUsage>, LlmError> {
        Ok(None)
    }

    /// Run [`Provider::stream`] with retry, notifying `on_retry` before each
    /// repeated attempt so the agent loop can surface the failure in the UI.
    async fn stream_with_retry(
        &self,
        req: &CompletionRequest,
        on_retry: RetryCallback,
    ) -> Result<EventStream, LlmError> {
        let callback = on_retry.clone();
        let provider = self.name().to_owned();
        with_retry(
            || async { self.stream(req).await },
            move |attempt, error| {
                tracing::warn!(provider = %provider, attempt, error = %error, "retrying provider request");
                callback(attempt, error);
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    #[tokio::test]
    async fn late_stream_errors_are_redacted_without_changing_the_variant() {
        let secret = "late-stream-sentinel-token";
        let raw: EventStream = Box::pin(stream::iter([Err(LlmError::Stream(format!(
            "provider echoed {secret}",
        )))]));
        let mut stream = redact_stream(raw, secret);
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, LlmError::Stream(_)));
        assert!(!error.to_string().contains(secret));
        assert!(error.to_string().contains("[redacted]"));
    }
}
