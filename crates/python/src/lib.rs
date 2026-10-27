//! CPython binding for Harness's native, one-shot execution API.

use harness::sdk::{self, RunOptions};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{OnceLock, mpsc};
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

fn runtime() -> PyResult<&'static Runtime> {
    // Check before touching OnceLock: fork can copy an initialized runtime (or
    // an initialization lock) without the threads that make it usable. The
    // shared provider HTTP pool also cannot be repaired by rebuilding Tokio.
    static OWNER_PID: AtomicU32 = AtomicU32::new(0);
    let pid = std::process::id();
    if let Err(owner) = OWNER_PID.compare_exchange(0, pid, Ordering::Relaxed, Ordering::Relaxed)
        && owner != pid
    {
        return Err(PyRuntimeError::new_err(
            "Harness cannot run after fork once its runtime has started; use multiprocessing with spawn",
        ));
    }
    // Keep one runtime alive for the process: providers share an HTTP pool,
    // and destroying a runtime after every call would invalidate its I/O tasks.
    static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| PyRuntimeError::new_err(format!("start Harness runtime: {error}")))
}

/// Longest wait for session and MCP cleanup after Ctrl-C.
const CLEANUP_AFTER_SIGNAL: Duration = Duration::from_secs(10);

#[pyfunction]
#[pyo3(signature = (prompt, options_json = "{}"))]
fn run(py: Python<'_>, prompt: String, options_json: &str) -> PyResult<String> {
    if prompt.trim().is_empty() {
        return Err(PyValueError::new_err("prompt must not be blank"));
    }
    let options: RunOptions = serde_json::from_str(options_json)
        .map_err(|error| PyValueError::new_err(format!("invalid Harness options: {error}")))?;
    options
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let runtime = runtime()?;
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let (sender, receiver) = mpsc::sync_channel(1);
    runtime.spawn(async move {
        let result = sdk::run(prompt, options, task_cancel).await;
        let _ = sender.send(result);
    });

    // Python remains available to other threads during network/tool waits. Poll
    // its signal state so Ctrl-C cancels the native call, then await session/MCP cleanup
    // before propagating the original Python exception. No Rust signal handler
    // is installed in the host process.
    //
    // Cleanup after Ctrl-C is bounded: a second Ctrl-C, or the cleanup
    // deadline, returns the original exception at once. The task keeps
    // shutting down on the shared runtime in the background.
    py.detach(move || {
        let mut signal_error = None;
        let mut cleanup_deadline = None;
        loop {
            let received = receiver.recv_timeout(Duration::from_millis(50));
            if let Err(error) = Python::attach(|py| py.check_signals()) {
                if let Some(first) = signal_error.take() {
                    // Second signal: stop waiting for cleanup.
                    drop(error);
                    return Err(first);
                }
                cancel.cancel();
                signal_error = Some(error);
                cleanup_deadline = Some(std::time::Instant::now() + CLEANUP_AFTER_SIGNAL);
            }
            if cleanup_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
                && let Some(error) = signal_error.take()
            {
                return Err(error);
            }
            match received {
                Ok(result) => {
                    if let Some(error) = signal_error {
                        return Err(error);
                    }
                    return result.map_err(|error| PyRuntimeError::new_err(format!("{error:#}")));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    cancel.cancel();
                    return Err(signal_error.unwrap_or_else(|| {
                        PyRuntimeError::new_err("Harness execution task stopped unexpectedly")
                    }));
                }
            }
        }
    })
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
