//! Blocking session I/O from async code. The helper lives in `tools`, which
//! uses it for file tools, so both crates share one runtime policy.

pub(crate) use tools::blocking_io;

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
