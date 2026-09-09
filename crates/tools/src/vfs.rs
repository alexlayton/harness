//! Handle-relative filesystem opens for workspace tools.
//!
//! Unix uses directory-handle-relative traversal with no-follow semantics.
//! `WorkspaceFs` owns one validated directory fd for the workspace root, and
//! every workspace file open and mutation is relative to that capability.
//!
//! Why this closes the TOCTOU race rather than narrowing it: the file that
//! is read or written is never opened by re-walking the path from `/`.
//! Components are traversed one at a time with `O_NOFOLLOW`, so a symlink
//! swapped into any ancestor *after* validation fails the open. The final
//! containment check also compares the opened directory's device/inode with
//! the validated root. "Canonicalize twice" only narrows the race because
//! both checks still end in a name-based `open()`; here the open itself is
//! handle-relative, so the path opened *is* the path that was validated.
//!
//! Non-Unix platforms do not have this implementation. The old fallback
//! canonicalized and then reopened by pathname, which is not safe against a
//! Windows junction/reparse-point ancestor swap. Workspace read/write/edit
//! tools therefore fail closed on non-Unix platforms instead of claiming
//! containment they cannot enforce. `WorkspaceFs` still retains the
//! canonical root for registry construction and diagnostics, but it is not a
//! filesystem capability there.

use std::io;
use std::path::{Component, Path, PathBuf};

/// Validated workspace root capability.
///
/// On Unix, construction holds the root open as a directory fd and later
/// operations use handle-relative, no-follow traversal. On non-Unix
/// platforms this stores only the canonical root path; workspace file tools
/// reject operations there because pathname I/O cannot enforce the same
/// invariant.
#[derive(Debug)]
pub struct WorkspaceFs {
    root: PathBuf,
    #[cfg(unix)]
    root_fd: std::os::fd::OwnedFd,
}

impl WorkspaceFs {
    /// Canonicalize `root` and, on Unix, hold it open as a directory fd.
    /// Fails when the root does not exist or is not a directory. On
    /// non-Unix platforms the returned value is only a canonical-root
    /// validator; workspace read/write/edit operations reject it rather than
    /// fall back to unsafe pathname opens.
    pub fn open_root(root: &Path) -> io::Result<Self> {
        let canonical = std::fs::canonicalize(root)?;
        let metadata = std::fs::metadata(&canonical)?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("workspace root {} is not a directory", root.display()),
            ));
        }
        #[cfg(unix)]
        {
            use rustix::fs::{Mode, OFlags};
            let fd = rustix::fs::open(
                &canonical,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::from)?;
            Ok(Self {
                root: canonical,
                root_fd: fd,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self { root: canonical })
        }
    }

    /// Return the explicit fail-closed error used when a platform cannot
    /// provide handle-relative, no-follow workspace I/O.
    #[cfg(not(unix))]
    pub fn unsupported_operation(operation: &str) -> io::Error {
        let platform = if cfg!(windows) {
            "Windows reparse-point-safe handle-relative I/O is unavailable"
        } else {
            "handle-relative no-follow I/O is unavailable"
        };
        io::Error::new(
            io::ErrorKind::Unsupported,
            format!("workspace {operation} is disabled: {platform}"),
        )
    }

    /// Canonical workspace root this handle was validated against.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Lexically split `value` into workspace-relative components. Shared by the
/// Unix handle walk and fail-closed platform paths so all callers retain the
/// same lexical confinement rules.
pub fn split_relative(value: &str) -> io::Result<Vec<String>> {
    if value.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path must not be empty",
        ));
    }
    let path = Path::new(value);
    let mut out = Vec::new();
    let mut depth = 0usize;
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("path is outside workspace root: {value}"),
                ));
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if depth == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("path is outside workspace root: {value}"),
                    ));
                }
                depth -= 1;
                out.pop();
            }
            Component::Normal(part) => {
                depth += 1;
                out.push(part.to_string_lossy().into_owned());
            }
        }
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path must not be empty",
        ));
    }
    Ok(out)
}

#[cfg(unix)]
pub mod unix {
    //! Unix handle-relative opens. All traversal starts at the validated
    //! workspace-root fd held by [`super::WorkspaceFs`]; no path is ever
    //! re-walked from `/`, so an ancestor swapped after validation cannot
    //! redirect the open.

    use super::WorkspaceFs;
    use std::io;
    use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

    /// Open the directory `rel` (relative components) under the workspace
    /// root, one component at a time with `O_NOFOLLOW`. Every intermediate
    /// component must be a directory and must not be a symlink: a symlinked
    /// ancestor fails instead of being followed. An empty `rel` opens the
    /// root itself.
    pub fn open_dir_relative(fs: &WorkspaceFs, rel: &[String]) -> io::Result<OwnedFd> {
        let mut current: OwnedFd = dup_fd(fs.root_fd.as_fd())?;
        for component in rel {
            let next = open_child_dir(current.as_fd(), component)?;
            current = next;
        }
        confirm_contained(fs, &current)?;
        Ok(current)
    }

    /// Open the regular file `rel` for reading through the validated parent
    /// directory handle. `O_NOFOLLOW` on the final component rejects a
    /// symlinked file; the parent walk rejects symlinked ancestors.
    pub fn open_file_relative(fs: &WorkspaceFs, rel: &[String]) -> io::Result<OwnedFd> {
        let Some((parent, name)) = rel.split_at_checked(rel.len().saturating_sub(1)) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "path must not be empty",
            ));
        };
        let [name] = name else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "path must not be empty",
            ));
        };
        let dir_fd = open_dir_relative(fs, parent)?;
        open_child_file(dir_fd.as_fd(), name)
    }

    /// Create `rel`'s missing parents (mkdir, relative to validated handles)
    /// and return the validated parent directory fd plus the file name.
    /// Parents are created with `O_NOFOLLOW` traversal, so a symlinked
    /// ancestor aborts the write instead of redirecting it.
    pub fn open_parent_relative(fs: &WorkspaceFs, rel: &[String]) -> io::Result<(OwnedFd, String)> {
        let Some((parent, name)) = rel.split_at_checked(rel.len().saturating_sub(1)) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "path must not be empty",
            ));
        };
        let [name] = name else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "path must not be empty",
            ));
        };
        let mut current: OwnedFd = dup_fd(fs.root_fd.as_fd())?;
        for component in parent {
            let next = open_or_mkdir_child_dir(current.as_fd(), component)?;
            current = next;
        }
        confirm_contained(fs, &current)?;
        Ok((current, name.clone()))
    }

    fn dup_fd(fd: BorrowedFd<'_>) -> io::Result<OwnedFd> {
        use rustix::fs::{Mode, OFlags};
        rustix::fs::openat(
            fd,
            ".",
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)
    }

    fn open_child_dir(parent: BorrowedFd<'_>, name: &str) -> io::Result<OwnedFd> {
        use rustix::fs::{Mode, OFlags};
        rustix::fs::openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| {
            if error == rustix::io::Errno::LOOP || error == rustix::io::Errno::NOTDIR {
                // `O_NOFOLLOW | O_DIRECTORY` on a symlink yields ENOTDIR on
                // Linux (ELOOP without O_DIRECTORY on macOS); normalize both
                // to a symlink rejection so callers see one failure mode.
                io::Error::other(format!("symlink ancestor rejected: {name}"))
            } else {
                io::Error::from(error)
            }
        })
    }

    fn open_or_mkdir_child_dir(parent: BorrowedFd<'_>, name: &str) -> io::Result<OwnedFd> {
        match open_child_dir(parent, name) {
            Ok(fd) => Ok(fd),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                use rustix::fs::Mode;
                rustix::fs::mkdirat(
                    parent,
                    name,
                    Mode::RWXU | Mode::RGRP | Mode::XGRP | Mode::ROTH | Mode::XOTH,
                )
                .map_err(io::Error::from)?;
                open_child_dir(parent, name)
            }
            Err(error) => Err(error),
        }
    }

    /// Require an opened or statted path to be a regular file. Keeping this
    /// check in the handle-relative layer means every caller gets the same
    /// safe treatment for directories, FIFOs, sockets, and device nodes.
    pub fn ensure_regular_file(name: &str, file_type: rustix::fs::FileType) -> io::Result<()> {
        if file_type.is_file() {
            return Ok(());
        }
        if file_type.is_symlink() {
            return Err(io::Error::other(format!("symlink file rejected: {name}")));
        }
        if file_type.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("is a directory: {name}"),
            ));
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("not a regular file: {name}"),
        ))
    }

    /// Return permissions for an existing regular file in `parent` without
    /// opening its contents. Missing entries are reported as `None`; any
    /// existing non-regular entry is rejected. This lets `write` preserve
    /// modes on write-only regular files without ever opening a FIFO or
    /// device just to read metadata.
    pub fn existing_file_permissions(
        parent: BorrowedFd<'_>,
        name: &str,
    ) -> io::Result<Option<std::fs::Permissions>> {
        use rustix::fs::AtFlags;
        use std::os::unix::fs::PermissionsExt;

        match rustix::fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => {
                let file_type = rustix::fs::FileType::from_raw_mode(stat.st_mode);
                ensure_regular_file(name, file_type)?;
                // `st_mode` is `u32` on Linux but `u16` (`mode_t`) on macOS;
                // widen explicitly so `Permissions::from_mode` always sees
                // a `u32`. `allow` because on Linux this is a no-op cast.
                #[allow(clippy::useless_conversion)]
                let mode = u32::from(stat.st_mode) & 0o7777;
                Ok(Some(std::fs::Permissions::from_mode(mode)))
            }
            Err(error) if error == rustix::io::Errno::NOENT => Ok(None),
            Err(error) => Err(io::Error::from(error)),
        }
    }

    fn open_child_file(parent: BorrowedFd<'_>, name: &str) -> io::Result<OwnedFd> {
        use rustix::fs::{AtFlags, Mode, OFlags};

        // Inspect the directory entry first so sockets and device nodes are
        // rejected with a useful error without asking the kernel to open
        // them. This is only a preflight: the final fstat below still closes
        // the type-check race if the entry changes between these operations.
        let entry_type = rustix::fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW)
            .map(|stat| rustix::fs::FileType::from_raw_mode(stat.st_mode))
            .map_err(io::Error::from)?;
        ensure_regular_file(name, entry_type)?;

        // O_NONBLOCK is essential even though regular files are the only
        // accepted result. A FIFO or other special file swapped in after the
        // preflight must not make this synchronous open wait on a peer.
        let fd = rustix::fs::openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| {
            if error == rustix::io::Errno::LOOP {
                io::Error::other(format!("symlink file rejected: {name}"))
            } else {
                io::Error::from(error)
            }
        })?;
        let file_type = rustix::fs::FileType::from_raw_mode(
            rustix::fs::fstat(&fd).map_err(io::Error::from)?.st_mode,
        );
        ensure_regular_file(name, file_type)?;
        Ok(fd)
    }

    /// Metadata of the regular file `rel`, opened through the validated
    /// parent handle. Used to preserve existing permissions on replace
    /// without re-walking names from `/`.
    pub fn open_file_metadata(fs: &WorkspaceFs, rel: &[String]) -> io::Result<std::fs::Metadata> {
        let fd = open_file_relative(fs, rel)?;
        // Duplicate into a `std::fs::File` and read metadata from the open
        // description: identity comes from the handle, not a name lookup.
        let duplicated = rustix::io::retry_on_intr(|| rustix::io::fcntl_dupfd_cloexec(&fd, 0))
            .map_err(io::Error::from)?;
        use std::os::fd::{FromRawFd, IntoRawFd};
        let raw = duplicated.into_raw_fd();
        // SAFETY: `raw` is freshly duplicated and now solely owned.
        let file = unsafe { std::fs::File::from_raw_fd(raw) };
        file.metadata()
    }

    /// Confirm `dir_fd` is contained in the workspace root by walking up
    /// through `..` (which, from a handle, resolves against the real
    /// filesystem) and comparing device/inode with the root. Because the
    /// walk starts at validated handles, a swapped ancestor shows up as an
    /// identity mismatch instead of a name that "looks confined".
    fn confirm_contained(fs: &WorkspaceFs, dir_fd: &OwnedFd) -> io::Result<()> {
        use rustix::fs::{Mode, OFlags};
        let root_stat = rustix::fs::fstat(fs.root_fd.as_fd()).map_err(io::Error::from)?;
        let mut current = dup_fd(dir_fd.as_fd())?;
        loop {
            let stat = rustix::fs::fstat(&current).map_err(io::Error::from)?;
            if stat.st_dev == root_stat.st_dev && stat.st_ino == root_stat.st_ino {
                return Ok(());
            }
            // Step to the parent *of the opened directory*, not of a name:
            // `..` from a handle follows the real parent chain.
            let parent = rustix::fs::openat(
                current.as_fd(),
                "..",
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::from)?;
            let parent_stat = rustix::fs::fstat(&parent).map_err(io::Error::from)?;
            if parent_stat.st_dev == stat.st_dev && parent_stat.st_ino == stat.st_ino {
                // `..` of `/` is `/` itself and we never met the root:
                // the directory escaped the workspace.
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "path resolves outside workspace root",
                ));
            }
            current = parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_rejects_escapes_and_accepts_nested() {
        assert!(split_relative("").is_err());
        assert!(split_relative("/abs").is_err());
        assert!(split_relative("../out").is_err());
        assert!(split_relative("a/../../out").is_err());
        assert_eq!(
            split_relative("a/./b/../c.txt").unwrap(),
            vec!["a".to_owned(), "c.txt".to_owned()]
        );
    }

    #[test]
    fn open_root_rejects_files_and_missing_dirs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(WorkspaceFs::open_root(dir.path()).is_ok());
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "x").unwrap();
        assert!(WorkspaceFs::open_root(&file).is_err());
        assert!(WorkspaceFs::open_root(&dir.path().join("missing")).is_err());
    }

    #[cfg(unix)]
    fn create_fifo(path: &std::path::Path) {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: the C string is NUL-terminated and points to the intended
        // temporary-directory path for the duration of the call.
        let result = unsafe { libc::mkfifo(path.as_ptr(), 0o600) };
        assert_eq!(
            result,
            0,
            "mkfifo failed: {}",
            std::io::Error::last_os_error()
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_fifo_and_socket_without_opening_special_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let fifo = root.join("input.fifo");
        create_fifo(&fifo);
        let socket_path = root.join("listener.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&socket_path).unwrap();
        let fs = WorkspaceFs::open_root(&root).unwrap();

        for (name, expected) in [("input.fifo", "FIFO"), ("listener.sock", "socket")] {
            let error = unix::open_file_relative(&fs, &[name.to_owned()]).unwrap_err();
            assert!(
                error.to_string().contains("not a regular file"),
                "{expected} was not rejected clearly: {error:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn handle_walk_opens_inside_files_and_rejects_symlink_ancestors() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/file.txt"), "hello").unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("link")).unwrap();

        let fs = WorkspaceFs::open_root(&root).unwrap();
        // Direct file opens through the handle.
        let fd = unix::open_file_relative(&fs, &["sub".into(), "file.txt".into()]).unwrap();
        let mut content = String::new();
        use std::io::Read;
        let mut file = std::fs::File::from(fd);
        file.read_to_string(&mut content).unwrap();
        assert_eq!(content, "hello");
        // Symlinked ancestors fail instead of redirecting.
        let error =
            unix::open_file_relative(&fs, &["link".into(), "secret.txt".into()]).unwrap_err();
        assert!(
            error.to_string().contains("outside workspace")
                || error.to_string().contains("symlink"),
            "unexpected: {error:?}"
        );
        // The symlink itself as a final component is rejected too.
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            root.join("sub/slink.txt"),
        )
        .unwrap();
        let error = unix::open_file_relative(&fs, &["sub".into(), "slink.txt".into()]).unwrap_err();
        assert!(
            error.to_string().contains("symlink"),
            "unexpected: {error:?}"
        );
    }
}
