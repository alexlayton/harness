use crate::LlmError;
use crate::error::truncate_body;
use std::future::Future;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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

/// A small (0..251 ms) jitter value used to spread retries.
///
/// Produced by a tiny LCG seeded once per process from the monotonic clock
/// folded with an address-derived constant, so separate harness processes do
/// not share the same retry schedule.  Wall-clock time is deliberately not
/// used: the previous implementation read `SystemTime::now()`
/// `.duration_since(UNIX_EPOCH)`, which panics on systems whose clock is set
/// before 1970 and falls back to a deterministic 0.
fn jitter_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    static STATE: AtomicU64 = AtomicU64::new(0);

    let start = *START.get_or_init(Instant::now);
    let seed = (Instant::now().duration_since(start).as_nanos() as u64)
        ^ address_seed()
        ^ 0x9e37_79b9_7f4a_7c15;
    let mut state = STATE.load(Ordering::Relaxed);
    if state == 0 {
        state = seed;
    }
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    STATE.store(state, Ordering::Relaxed);
    (state >> 33) % 251
}

/// Per-process entropy from ASLR: the address of a stack local differs between
/// process runs (where available), so concurrent harness instances do not all
/// retry on the same schedule.
fn address_seed() -> u64 {
    let local = 0u8;
    &local as *const u8 as usize as u64
}
