use crate::LlmError;
use std::future::Future;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Run an initial HTTP operation at most three times.  Once a provider has
/// returned an event stream, stream errors are intentionally not passed through
/// this function; they are delivered by the stream itself.
///
/// The wait before retry `n` is `max(backoff(n) + jitter, retry_after)`:
/// exponential backoff plus jitter, or the bounded 429 `Retry-After` hint
/// when it asks for longer (capped in `error.rs` so a malicious header
/// cannot park the loop).  Non-retryable failures return immediately.
pub async fn with_retry<F, Fut, T, C>(operation: F, on_retry: C) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
    C: Fn(u32, &LlmError),
{
    with_retry_sleep(operation, on_retry, sleep_ms).await
}

async fn sleep_ms(ms: u64) {
    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
}

/// Deterministic core: backoff/jitter/`Retry-After` accounting is pure and
/// injectable for tests; production passes the real sleeper.
async fn with_retry_sleep<F, Fut, T, C, S, SFut>(
    mut operation: F,
    on_retry: C,
    sleep: S,
) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
    C: Fn(u32, &LlmError),
    S: Fn(u64) -> SFut,
    SFut: Future<Output = ()>,
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
                sleep(retry_delay_ms(attempt, &error)).await;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Wait before retry `attempt`: `max(backoff + jitter, retry_after_secs)`.
fn retry_delay_ms(attempt: u32, error: &LlmError) -> u64 {
    let base_ms: u64 = match attempt {
        1 => 500,
        2 => 1_000,
        _ => 2_000,
    };
    let backoff = base_ms.saturating_add(jitter_ms());
    let hint_ms = error
        .retry_after_secs()
        .map(|secs| secs.saturating_mul(1_000))
        .unwrap_or(0);
    backoff.max(hint_ms)
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
    use std::sync::{Arc, Mutex};

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

    #[tokio::test]
    async fn retry_counts_attempts_and_honors_retry_after() {
        // Deterministic: injected sleeper records waits, no clock involved.
        let waits = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(Mutex::new(0u32));
        let waits_task = waits.clone();
        let calls_task = calls.clone();
        let result = with_retry_sleep(
            || {
                let calls_task = calls_task.clone();
                async move {
                    let mut calls = calls_task.lock().unwrap();
                    *calls += 1;
                    if *calls < 3 {
                        Err::<u32, _>(LlmError::http(503, "busy"))
                    } else {
                        Ok(*calls)
                    }
                }
            },
            |_, _| {},
            move |ms| {
                let waits_task = waits_task.clone();
                async move {
                    waits_task.lock().unwrap().push(ms);
                }
            },
        )
        .await;
        assert_eq!(result.unwrap(), 3);
        assert_eq!(*calls.lock().unwrap(), 3);
        // Two waits (after attempts 1 and 2), each >= its backoff floor.
        let waits = waits.lock().unwrap().clone();
        assert_eq!(waits.len(), 2);
        assert!(waits[0] >= 500 && waits[0] < 500 + 251 + 1);
        assert!(waits[1] >= 1_000 && waits[1] < 1_000 + 251 + 1);
    }

    #[tokio::test]
    async fn retry_after_overrides_backoff_and_nonretryable_returns() {
        // A 429 with `retry-after: 5` waits max(backoff, 5000) = 5000+.
        let waits = Arc::new(Mutex::new(Vec::new()));
        let waits_task = waits.clone();
        let calls = Arc::new(Mutex::new(0u32));
        let calls_task = calls.clone();
        let result = with_retry_sleep(
            || {
                let calls_task = calls_task.clone();
                async move {
                    let mut calls = calls_task.lock().unwrap();
                    *calls += 1;
                    if *calls == 1 {
                        Err::<u32, _>(LlmError::http(429, "busy\nretry-after: 5"))
                    } else {
                        Ok(*calls)
                    }
                }
            },
            |_, _| {},
            move |ms| {
                let waits_task = waits_task.clone();
                async move {
                    waits_task.lock().unwrap().push(ms);
                }
            },
        )
        .await;
        assert_eq!(result.unwrap(), 2);
        assert_eq!(waits.lock().unwrap().clone(), vec![5_000]);

        // Non-retryable failures return immediately with no waits.
        let waits = Arc::new(Mutex::new(Vec::new()));
        let waits_task = waits.clone();
        let result = with_retry_sleep(
            || async { Err::<u32, _>(LlmError::http(400, "bad")) },
            |_, _| {},
            move |ms| {
                let waits_task = waits_task.clone();
                async move {
                    waits_task.lock().unwrap().push(ms);
                }
            },
        )
        .await;
        assert!(result.is_err());
        assert!(waits.lock().unwrap().is_empty());
    }
}
