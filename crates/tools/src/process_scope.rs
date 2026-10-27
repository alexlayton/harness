//! Linux process-tree containment shared by the bash tool and MCP stdio
//! servers.

use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Linux cgroup-v2 scope for one process tree. A process group is not a
/// tree boundary: `setsid(2)` lets a descendant leave it. Cgroup membership
/// survives that operation, so `cgroup.kill` is the strong containment path
/// when the host grants this process a writable cgroup hierarchy.
pub struct CgroupScope {
    path: PathBuf,
}

static CGROUP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl CgroupScope {
    /// Create a child cgroup named `<prefix>-<pid>-<n>` under this
    /// process's delegated cgroup, or `None` when cgroup v2 is unavailable
    /// or not writable.
    pub fn new(prefix: &str) -> Option<Self> {
        let root = Path::new("/sys/fs/cgroup");
        if !root.join("cgroup.controllers").is_file() {
            return None;
        }
        // Delegation normally grants write access to this process's current
        // subtree, not to the cgroup-v2 mount root. Resolve `0::<path>` from
        // procfs and reject every non-normal component before joining it.
        let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
        let relative = membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))?;
        let mut base = root.to_path_buf();
        for component in Path::new(relative).components() {
            match component {
                std::path::Component::RootDir | std::path::Component::CurDir => {}
                std::path::Component::Normal(part) => base.push(part),
                std::path::Component::ParentDir | std::path::Component::Prefix(_) => return None,
            }
        }
        for _ in 0..8 {
            let number = CGROUP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{prefix}-{}-{number}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) if path.join("cgroup.kill").is_file() => {
                    return Some(Self { path });
                }
                Ok(()) => {
                    let _ = std::fs::remove_dir(&path);
                }
                Err(_) => return None,
            }
        }
        None
    }

    /// Path of `cgroup.procs`, for [`attach_current_process`] in `pre_exec`.
    pub fn procs_cstring(&self) -> CString {
        CString::new(self.path.join("cgroup.procs").as_os_str().as_bytes())
            .expect("cgroup path cannot contain NUL")
    }

    /// Kill every process in the scope.
    pub fn kill(&self) {
        // cgroup.kill is atomic with respect to membership: unlike a /proc
        // descendant walk, a concurrent fork cannot escape this operation.
        let _ = std::fs::write(self.path.join("cgroup.kill"), b"1\n");
    }

    /// Remove the (now empty) cgroup directory, retrying briefly.
    pub fn cleanup(&self) {
        // A normal path has waited for the shell and cgroup.kill has finished
        // the descendants. Kernel task exit is asynchronous, so retry the
        // removal briefly; this also prevents future-drop cleanup from
        // accumulating empty invocation cgroups.
        for _ in 0..200 {
            match std::fs::remove_dir(&self.path) {
                Ok(()) => return,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        }
    }
}

impl Drop for CgroupScope {
    fn drop(&mut self) {
        self.kill();
        self.cleanup();
    }
}

/// Move the calling process into the cgroup whose `cgroup.procs` is `path`.
/// Only async-signal-safe calls (open, write, close): it is meant to run in
/// a `pre_exec` hook, between fork and exec.
pub fn attach_current_process(path: &CStr) -> std::io::Result<()> {
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }

    let mut digits = [0u8; 20];
    let mut end = digits.len();
    let mut pid = unsafe { libc::getpid() as u64 };
    loop {
        end -= 1;
        digits[end] = b'0' + (pid % 10) as u8;
        pid /= 10;
        if pid == 0 {
            break;
        }
    }
    let bytes = &digits[end..];
    let mut written = 0;
    while written < bytes.len() {
        let result =
            unsafe { libc::write(fd, bytes[written..].as_ptr().cast(), bytes.len() - written) };
        if result < 0 {
            let error = std::io::Error::last_os_error();
            let _ = unsafe { libc::close(fd) };
            return Err(error);
        }
        if result == 0 {
            let _ = unsafe { libc::close(fd) };
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "could not attach shell to cgroup",
            ));
        }
        written += result as usize;
    }
    let close_result = unsafe { libc::close(fd) };
    if close_result < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
