//! Handle-relative filesystem opens for workspace tools.
//!
//! Design (TOOLS-2 spike outcome): of the three options in the plan —
//! capability-based APIs, `openat2`/directory-handle-relative traversal, a
//! small cross-platform abstraction — directory-handle-relative traversal
//! with no-follow semantics wins. It needs no new async runtime support
//! (unlike capability handles, which would force every tool through a new
//! I/O abstraction), it works on both Linux and macOS (unlike `openat2`,
//! which is Linux-only), and the validated-handle construction below is
//! already the small cross-platform abstraction the plan asks for:
//! `WorkspaceFs` owns one validated directory fd for the workspace root,
//! and every open is relative to it.
//!
//! Why this closes the TOCTOU race rather than narrowing it: the file that
//! is read or written is never opened by re-walking the path from `/`.
//! Components are traversed one at a time with `O_NOFOLLOW`, so a symlink
//! swapped into any ancestor *after* validation either fails the open
//! (`ELOOP`) or resolves to a directory fd that fails the final
//! containment check (device/inode comparison against the validated root).
//! "Canonicalize twice" only narrows the race because both checks still end
//! in a name-based `open()`; here the open itself is handle-relative, so
//! the path opened *is* the path that was validated.
//!
//! Non-Unix fallback: without directory fds, traversal degrades to the
//! existing canonicalize-and-check plus a same-directory temporary-file +
//! rename for writes. That narrows but does not close the race; the
//! platform limitation is documented here rather than claimed away.

use std::io;
use std::path::{Component, Path, PathBuf};

/// Validated handle to the workspace root. Construction canonicalizes the
/// root once; every later open is relative to the held directory fd, so the
/// root itself cannot be swapped out from under us.
#[derive(Debug)]
pub struct WorkspaceFs {
    root: PathBuf,
    #[cfg(unix)]
    root_fd: std::os::fd::OwnedFd,
}

impl WorkspaceFs {
    /// Canonicalize `root` and hold it open as a directory fd. Fails when
    /// the root does not exist or is not a directory.
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

    /// Canonical workspace root this handle was validated against.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Lexically split `value` into workspace-relative components. Shared by the
/// Unix handle walk and the non-Unix fallback so both enforce the same
/// lexical confinement.
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

    fn open_child_file(parent: BorrowedFd<'_>, name: &str) -> io::Result<OwnedFd> {
        use rustix::fs::{Mode, OFlags};
        // Read-only open of the validated file. `NOFOLLOW` rejects a final
        // symlink; directories are rejected by checking the opened fd.
        let fd = rustix::fs::openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
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
        if file_type.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("is a directory: {name}"),
            ));
        }
        Ok(fd)
    }

    /// Metadata of the regular file `rel`, opened through the validated
    /// parent handle. Used to preserve existing permissions on replace
    /// without re-walking names from `/`.
    pub fn open_file_metadata(fs: &WorkspaceFs, rel: &[String]) -> Option<std::fs::Metadata> {
        let fd = open_file_relative(fs, rel).ok()?;
        // Duplicate into a `std::fs::File` and read metadata from the open
        // description: identity comes from the handle, not a name lookup.
        let duplicated =
            rustix::io::retry_on_intr(|| rustix::io::fcntl_dupfd_cloexec(&fd, 0)).ok()?;
        use std::os::fd::{FromRawFd, IntoRawFd};
        let raw = duplicated.into_raw_fd();
        // SAFETY: `raw` is freshly duplicated and now solely owned.
        let file = unsafe { std::fs::File::from_raw_fd(raw) };
        file.metadata().ok()
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
