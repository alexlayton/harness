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
        // Poisoning is recoverable: the map is only ever mutated while the
        // mutex is held, so its contents are always consistent even if a
        // panicking thread was interrupted mid-update.
        let mut locks = locks.lock().unwrap_or_else(|poison| poison.into_inner());
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
    let mut locks = locks.lock().unwrap_or_else(|poison| poison.into_inner());
    locks.retain(|_, lock| Arc::strong_count(lock) > 1);
}

/// Write a file through a same-directory temporary file and atomic rename.
/// Keeping the temporary file beside the destination ensures the rename does
/// not cross filesystems. Existing permissions are copied to the temporary
/// file before it replaces the destination.
///
/// This helper is pathname-based and therefore is not a workspace containment
/// primitive: callers handling workspace paths must use `atomic_write_at`
/// on Unix or fail closed on platforms without an equivalent directory-handle
/// API. It remains available for the Unix compatibility path and for callers
/// that deliberately provide their own path-safety boundary.
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
        for chunk in contents.chunks(64 * 1024) {
            check_cancelled(cancel)?;
            temporary_file.write_all(chunk).await?;
        }
        temporary_file.sync_all().await?;
        drop(temporary_file);

        check_cancelled(cancel)?;
        if let Some(permissions) = permissions {
            fs::set_permissions(&temporary_path, permissions).await?;
        }
        check_cancelled(cancel)?;
        fs::rename(&temporary_path, path).await?;
        // Persist the directory entry as well as the file contents so a
        // successful atomic write survives a crash after rename.
        #[cfg(unix)]
        fs::File::open(parent).await?.sync_all().await?;
        Ok(())
    }
    .await;

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path).await;
    }
    result
}

/// Write `contents` to `name` inside the validated parent directory `parent_fd`
/// (Unix): anonymous temp file via `O_TMPFILE`, then `linkat` to the final
/// name (or `renameat` when replacing).  Both the temp creation and the
/// commit are relative to the validated handle, so a swapped ancestor
/// cannot redirect the write.  Returns `Ok(true)` when an existing file was
/// replaced, `Ok(false)` when created.
#[cfg(unix)]
pub async fn atomic_write_at(
    parent_fd: &std::os::fd::OwnedFd,
    name: &str,
    contents: &[u8],
    existing_permissions: Option<std::fs::Permissions>,
    cancel: &CancellationToken,
) -> io::Result<bool> {
    use std::os::fd::AsFd;

    check_cancelled(cancel)?;
    let (tmp, temporary_name) = create_temporary_file_at(parent_fd)?;
    let result = async {
        write_all_chunks(&tmp, contents, cancel).await?;
        rustix::fs::fsync(&tmp).map_err(io::Error::from)?;
        check_cancelled(cancel)?;

        if let Some(permissions) = existing_permissions {
            use std::os::unix::fs::PermissionsExt;
            rustix::fs::fchmod(
                &tmp,
                rustix::fs::Mode::from_bits_truncate(permissions.mode()),
            )
            .map_err(io::Error::from)?;
        }

        // Check and commit through the already validated parent handle. The
        // destination may be replaced atomically, but no ancestor is looked up
        // again by pathname. Recheck the type immediately before rename so a
        // special file racing with the metadata preflight is never silently
        // accepted as an existing write target.
        let existed = match rustix::fs::statat(
            parent_fd.as_fd(),
            name,
            rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(stat) => {
                let file_type = rustix::fs::FileType::from_raw_mode(stat.st_mode);
                super::vfs::unix::ensure_regular_file(name, file_type)?;
                true
            }
            Err(error) if error == rustix::io::Errno::NOENT => false,
            Err(error) => return Err(io::Error::from(error)),
        };
        check_cancelled(cancel)?;
        rustix::fs::renameat(parent_fd.as_fd(), &temporary_name, parent_fd.as_fd(), name)
            .map_err(io::Error::from)?;

        // Cancellation after rename must not turn a committed mutation into a
        // reported failure. The caller can no longer safely retry it.
        rustix::fs::fsync(parent_fd.as_fd()).map_err(io::Error::from)?;
        Ok(existed)
    }
    .await;

    if result.is_err() {
        // If rename already committed, this is simply NotFound. Ignore cleanup
        // errors because the primary operation's error is more useful.
        let _ = rustix::fs::unlinkat(
            parent_fd.as_fd(),
            &temporary_name,
            rustix::fs::AtFlags::empty(),
        );
    }
    result
}

#[cfg(unix)]
fn create_temporary_file_at(
    parent_fd: &std::os::fd::OwnedFd,
) -> io::Result<(std::os::fd::OwnedFd, String)> {
    use std::os::fd::AsFd;
    for _ in 0..100 {
        let name = format!(".harness-edit-{}", uuid::Uuid::new_v4());
        match rustix::fs::openat(
            parent_fd.as_fd(),
            name.as_str(),
            rustix::fs::OFlags::WRONLY
                | rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::EXCL
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        ) {
            Ok(file) => return Ok((file, name)),
            Err(error) if error == rustix::io::Errno::EXIST => continue,
            Err(error) => return Err(io::Error::from(error)),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique handle-relative temporary file",
    ))
}

#[cfg(unix)]
async fn write_all_chunks(
    tmp: &std::os::fd::OwnedFd,
    contents: &[u8],
    cancel: &CancellationToken,
) -> io::Result<()> {
    use std::os::fd::AsFd;
    use tokio::io::AsyncWriteExt;
    // `rustix` fds are blocking; wrap the owned fd in a tokio File without
    // duplicating it by transferring ownership through `File::from`.
    let std_file = fd_to_std_file(tmp.as_fd())?;
    let mut file = tokio::fs::File::from_std(std_file);
    for chunk in contents.chunks(64 * 1024) {
        if cancel.is_cancelled() {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
        }
        file.write_all(chunk).await?;
    }
    file.sync_all().await?;
    // Prevent `file`'s Drop from closing the fd: the caller still owns `tmp`.
    // Re-materialize ownership by forgetting the std wrapper's fd… instead,
    // simply leak-guard: `into_std` below duplicates. Easiest correct path:
    // detach by `mem::forget` after converting back — but tokio File owns
    // the fd. To keep single ownership sound, duplicate first.
    Ok(())
}

#[cfg(unix)]
fn fd_to_std_file(fd: std::os::fd::BorrowedFd<'_>) -> io::Result<std::fs::File> {
    let duplicated = rustix::io::retry_on_intr(|| rustix::io::fcntl_dupfd_cloexec(fd, 0))
        .map_err(io::Error::from)?;
    use std::os::fd::{FromRawFd, IntoRawFd};
    let raw = duplicated.into_raw_fd();
    // SAFETY: `raw` was just duplicated from a live fd and is now owned.
    Ok(unsafe { std::fs::File::from_raw_fd(raw) })
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
