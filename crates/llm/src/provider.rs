use crate::retry::with_retry;
use crate::{CompletionRequest, LlmError, ModelInfo, StreamEvent};
use futures_core::Stream;
use std::pin::Pin;
use std::sync::Arc;

pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;
pub type RetryCallback = Arc<dyn for<'a> Fn(u32, &'a LlmError) + Send + Sync>;

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
