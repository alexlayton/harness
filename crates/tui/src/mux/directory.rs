use std::path::{Path, PathBuf};

pub(super) fn basename(path: &Path) -> String {
    path.file_name()
        .and_then(|v| v.to_str())
        .filter(|v| !v.is_empty())
        .unwrap_or("agent")
        .to_owned()
}
pub(super) fn expand_path(value: &str, cwd: &Path) -> PathBuf {
    let p = if value == "~" || value.starts_with("~/") {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| cwd.to_owned())
            .join(value.trim_start_matches("~/"))
    } else {
        PathBuf::from(value)
    };
    if p.is_absolute() { p } else { cwd.join(p) }
}
pub(super) const MAX_DIRECTORY_SUGGESTIONS: usize = 6;

pub(super) fn directory_suggestions(value: &str, cwd: &Path) -> Vec<PathBuf> {
    let path = expand_path(value, cwd);
    let (parent, needle) = if path.is_dir() {
        (path, "".into())
    } else {
        (
            path.parent().unwrap_or(cwd).to_owned(),
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase(),
        )
    };
    let Ok(read) = std::fs::read_dir(parent) else {
        return vec![];
    };
    let mut found = read
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if !p.is_dir() || name.starts_with('.') {
                return None;
            }
            let lower = name.to_lowercase();
            is_subsequence(&needle, &lower).then_some((
                if lower.starts_with(&needle) { 0 } else { 1 },
                lower,
                p,
            ))
        })
        .collect::<Vec<_>>();
    found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    found.truncate(MAX_DIRECTORY_SUGGESTIONS);
    found.into_iter().map(|(_, _, path)| path).collect()
}
fn is_subsequence(q: &str, s: &str) -> bool {
    let mut chars = s.chars();
    q.chars().all(|c| chars.by_ref().any(|v| v == c))
}
