use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EnvironmentInfo {
    pub cwd: PathBuf,
    pub cwd_display: String,
    pub branch: Option<String>,
}

impl EnvironmentInfo {
    pub fn discover(root: PathBuf) -> Self {
        let cwd_display = display_path(&root);
        let branch = git_branch(&root);
        Self {
            cwd: root,
            cwd_display,
            branch,
        }
    }
}

fn display_path(path: &Path) -> String {
    let Some(home) = dirs_home() else {
        return path.to_string_lossy().replace('\\', "/");
    };
    display_path_with_home(path, &home)
}

fn display_path_with_home(path: &Path, home: &Path) -> String {
    let path = path.to_string_lossy().replace('\\', "/");
    let home = home.to_string_lossy().replace('\\', "/");
    if path == home {
        "~".into()
    } else if let Some(rest) = path.strip_prefix(&(home.clone() + "/")) {
        format!("~/{rest}")
    } else {
        path
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn git_branch(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args([
            "-C",
            root.to_string_lossy().as_ref(),
            "branch",
            "--show-current",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!branch.is_empty()).then_some(branch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_home_relative_paths() {
        let home = Path::new("/tmp/home");
        assert_eq!(
            display_path_with_home(Path::new("/tmp/home/project"), home),
            "~/project"
        );
        assert_eq!(display_path_with_home(home, home), "~");
    }
}
