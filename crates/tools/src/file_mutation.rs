//! Shared coordination and commit helpers for tools that mutate files.
//!
//! The lock is deliberately process-local: it prevents two harness tool calls
//! from racing through a read/modify/write cycle for the same file.  The edit
//! tool also performs a content comparison immediately before its atomic
//! rename, so changes made by another process are not silently overwritten in
//! the normal case.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

type FileLock = Arc<tokio::sync::Mutex<()>>;

static FILE_LOCKS: OnceLock<Mutex<HashMap<PathBuf, FileLock>>> = OnceLock::new();
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Serialize mutations targeting the same file.  Different files remain able
/// to proceed concurrently.  `None` means cancellation happened while waiting
/// for the lock; cancellation during `operation` is the operation's concern.
pub async fn with_file_mutation_lock<F, Fut, T>(
    path: &Path,
    cancel: &CancellationToken,
    operation: F,
) -> Option<T>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    if cancel.is_cancelled() {
        return None;
    }

    let key = lock_key(path).await;
    let lock = {
        let locks = FILE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut locks = locks.lock().expect("file mutation lock registry poisoned");
        locks
            .entry(key)
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    };

    let result = tokio::select! {
        guard = lock.lock() => {
            let result = operation().await;
            drop(guard);
            result
        }
        _ = cancel.cancelled() => return None,
    };
    drop(lock);

    // The map otherwise keeps every path ever locked, growing without bound
    // over long sessions.  A lock whose strong count is 1 (only the map holds
    // it) is unreachable: lookups clone under the map mutex, so nobody can be
    // waiting on it.  Sweeping after each operation bounds the map to paths
    // with in-flight or queued edits.
    sweep_unreferenced_file_locks();
    Some(result)
}

/// Drop entries from [`FILE_LOCKS`] that no longer have any reference outside
/// the map itself.  All lookups clone under the map mutex, so a strong count
/// of 1 while the mutex is held means the entry is genuinely unreachable.
fn sweep_unreferenced_file_locks() {
    let locks = FILE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().expect("file mutation lock registry poisoned");
    locks.retain(|_, lock| Arc::strong_count(lock) > 1);
}

/// Write a file through a same-directory temporary file and atomic rename.
/// Keeping the temporary file beside the destination ensures the rename does
/// not cross filesystems.  Existing permissions are copied to the temporary
/// file before it replaces the destination.
pub async fn atomic_write(
    path: &Path,
    contents: &[u8],
    cancel: &CancellationToken,
) -> io::Result<()> {
    check_cancelled(cancel)?;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let permissions = fs::metadata(path)
        .await
        .ok()
        .map(|metadata| metadata.permissions());
    let (temporary_path, mut temporary_file) = create_temporary_file(parent).await?;

    let result = async {
        temporary_file.write_all(contents).await?;
        temporary_file.sync_all().await?;
        drop(temporary_file);

        check_cancelled(cancel)?;
        if let Some(permissions) = permissions {
            fs::set_permissions(&temporary_path, permissions).await?;
        }
        check_cancelled(cancel)?;
        fs::rename(&temporary_path, path).await
    }
    .await;

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path).await;
    }
    result
}

async fn lock_key(path: &Path) -> PathBuf {
    if let Ok(canonical) = fs::canonicalize(path).await {
        return canonical;
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

async fn create_temporary_file(parent: &Path) -> io::Result<(PathBuf, fs::File)> {
    for _ in 0..100 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!(".harness-edit-{}-{counter}.tmp", std::process::id());
        let path = parent.join(name);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique temporary file",
    ))
}

fn check_cancelled(cancel: &CancellationToken) -> io::Result<()> {
    if cancel.is_cancelled() {
        Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
    } else {
        Ok(())
    }
}
