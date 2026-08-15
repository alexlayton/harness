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

    async fn stream(&self, req: &CompletionRequest) -> Result<EventStream, LlmError>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Variant used by the agent loop when it wants retry notices in the UI.
    /// Providers with an HTTP implementation override this; a scripted/mock
    /// provider only needs to implement the three core methods above.
    async fn stream_with_retry(
        &self,
        req: &CompletionRequest,
        on_retry: RetryCallback,
    ) -> Result<EventStream, LlmError> {
        let callback = on_retry.clone();
        with_retry(
            || async { self.stream(req).await },
            move |attempt, error| callback(attempt, error),
        )
        .await
    }
}
