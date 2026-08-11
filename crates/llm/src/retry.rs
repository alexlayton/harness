use crate::LlmError;
use crate::error::truncate_body;
use std::future::Future;
use std::time::{SystemTime, UNIX_EPOCH};

/// Run an initial HTTP operation at most three times.  Once a provider has
/// returned an event stream, stream errors are intentionally not passed through
/// this function; they are delivered by the stream itself.
pub async fn with_retry<F, Fut, T, C>(mut operation: F, on_retry: C) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
    C: Fn(u32, &LlmError),
{
    const MAX_ATTEMPTS: u32 = 3;
    let mut attempt = 1;

    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) if error.is_retryable() && attempt < MAX_ATTEMPTS => {
                // attempt is the attempt that just failed.  Expose that value
                // to callers as the retry number (1, then 2).
                on_retry(attempt, &error);
                let base_ms = match attempt {
                    1 => 500,
                    2 => 1_000,
                    _ => 2_000,
                };
                let jitter = jitter_ms();
                tokio::time::sleep(std::time::Duration::from_millis(base_ms + jitter)).await;
                attempt += 1;
            }
            Err(error) => return Err(truncate_http_error(error)),
        }
    }
}

fn truncate_http_error(error: LlmError) -> LlmError {
    match error {
        LlmError::Http { status, body } => LlmError::Http {
            status,
            body: truncate_body(&body, 2048),
        },
        other => other,
    }
}

fn jitter_ms() -> u64 {
    // A dependency-free, small jitter source.  Cryptographic randomness is not
    // needed for scheduling and keeping this helper deterministic-free makes the
    // retry module usable in small binaries and tests.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos() as u64 % 251)
        .unwrap_or(0)
}
