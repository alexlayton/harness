//! Blocking session I/O from async code.

/// Run blocking session I/O (appends, `fsync`, bounded lock waits) without
/// stalling the other tasks on this Tokio worker.
///
/// On the multi-thread runtime, `block_in_place` first hands the worker's
/// queued tasks to another thread, so the UI event forwarder, MCP clients,
/// and parallel tools keep running while a disk flush waits. The
/// current-thread runtime (unit tests and some embedders) cannot do that,
/// and calling `block_in_place` there panics, so `f` runs inline as before.
pub(crate) fn blocking_io<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::blocking_io;

    #[test]
    fn runs_inline_without_a_runtime_and_on_current_thread() {
        assert_eq!(blocking_io(|| 1), 1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        assert_eq!(runtime.block_on(async { blocking_io(|| 2) }), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moves_off_the_worker_on_the_multi_thread_runtime() {
        let value = tokio::spawn(async { blocking_io(|| 3) }).await.unwrap();
        assert_eq!(value, 3);
    }
}
