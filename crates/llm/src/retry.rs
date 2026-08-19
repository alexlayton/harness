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
///
/// The LCG state lives in a `OnceLock<AtomicU64>` so seeding runs exactly
/// once (no two threads can seed independently), and every step is an atomic
/// read-modify-write via `fetch_update`.  The state is kept non-zero with
/// `.max(1)`, removing the old "re-seed when zero" special case and the
/// non-atomic load/store race it papered over.
fn jitter_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    static STATE: OnceLock<AtomicU64> = OnceLock::new();

    let state = STATE.get_or_init(|| {
        let start = *START.get_or_init(Instant::now);
        let nanos = Instant::now().duration_since(start).as_nanos() as u64;
        let seed = (nanos ^ address_seed() ^ 0x9e37_79b9_7f4a_7c15).max(1);
        AtomicU64::new(seed)
    });

    let value = state
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
            Some(
                s.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    .max(1),
            )
        })
        .unwrap_or(1);

    (value >> 33) % 251
}

/// Per-process entropy from ASLR: the address of a stack local differs between
/// process runs (where available), so concurrent harness instances do not all
/// retry on the same schedule.
fn address_seed() -> u64 {
    let local = 0u8;
    &local as *const u8 as usize as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_values_stay_within_range() {
        for _ in 0..10_000 {
            let value = jitter_ms();
            assert!(value < 251, "jitter {value} outside 0..251");
        }
    }

    #[test]
    fn jitter_is_safe_from_concurrent_callers() {
        let handles = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..1_000 {
                        let value = jitter_ms();
                        assert!(value < 251, "jitter {value} outside 0..251");
                    }
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().expect("jitter thread panicked");
        }
    }
}
