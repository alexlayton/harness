use crate::retry::with_retry;
use crate::{CompletionRequest, LlmError, ModelInfo, StreamEvent};
use futures_core::Stream;
use std::pin::Pin;
use std::sync::Arc;

pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;
pub type RetryCallback = Arc<dyn for<'a> Fn(u32, &'a LlmError) + Send + Sync>;

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &'static str;

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
