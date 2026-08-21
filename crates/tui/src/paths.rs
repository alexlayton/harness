//! Filesystem completion for slash-command arguments such as
//! `/export src/ma` or `/load ./session`. Scans are bounded and cancellable
//! so a large directory never blocks the UI thread.

use crate::commands::{Candidate, CandidateKind};
use std::fs;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

/// Maximum number of file/directory candidates returned by one scan.
///
/// This is a completion-performance bound.
const MAX_FILE_CANDIDATES: usize = 100;

/// Directories that are almost always generated output or dependency trees.
/// `.git` is also filtered so it is never offered as a command argument.
const BUILT_IN_IGNORED_DIRECTORIES: &[&str] = &[
    ".git",
    ".cache",
    ".next",
    ".venv",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
];

/// Return filesystem candidates for command arguments such as `/export src/ma`
/// or `/load ./session`. Values preserve the path spelling supplied by the
/// user.
pub fn find_path_candidates(
    root: &Path,
    query: &str,
    cancel: &CancellationToken,
) -> Vec<Candidate> {
    let query = query.replace('\\', "/");
    let Some((display_dir, directory, partial)) = split_path_query(root, &query) else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };

    let mut matches = Vec::new();
    for entry in entries {
        if cancel.is_cancelled() {
            return Vec::new();
        }
        let Ok(entry) = entry else {
            continue;
        };
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.chars().any(char::is_whitespace)
            || BUILT_IN_IGNORED_DIRECTORIES.contains(&name.as_str())
        {
            continue;
        }
        let is_directory = file_type.is_dir();
        let Some(score) = score_path(&name, &partial, is_directory) else {
            continue;
        };
        let value = format!(
            "{}{}{}",
            display_dir,
            name,
            if is_directory { "/" } else { "" }
        );
        matches.push((score, is_directory, value));
    }

    matches.sort_by(
        |(score_a, directory_a, value_a), (score_b, directory_b, value_b)| {
            score_b
                .cmp(score_a)
                .then_with(|| directory_b.cmp(directory_a))
                .then_with(|| {
                    value_a
                        .to_ascii_lowercase()
                        .cmp(&value_b.to_ascii_lowercase())
                })
                .then_with(|| value_a.cmp(value_b))
        },
    );
    matches
        .into_iter()
        .take(MAX_FILE_CANDIDATES)
        .map(|(_, is_directory, value)| Candidate {
            value,
            description: if is_directory {
                "directory".into()
            } else {
                "file".into()
            },
            kind: if is_directory {
                CandidateKind::Directory
            } else {
                CandidateKind::File
            },
        })
        .collect()
}

fn split_path_query(root: &Path, query: &str) -> Option<(String, PathBuf, String)> {
    if query == "~" {
        let home = std::env::var_os("HOME")?;
        return Some(("~/".into(), PathBuf::from(home), String::new()));
    }

    let (display_dir, partial) = if query.ends_with('/') {
        (query.to_owned(), String::new())
    } else if let Some(slash) = query.rfind('/') {
        (query[..=slash].to_owned(), query[slash + 1..].to_owned())
    } else {
        (String::new(), query.to_owned())
    };
    let directory = resolve_path(root, &display_dir)?;
    Some((display_dir, directory, partial))
}

fn resolve_path(root: &Path, display_path: &str) -> Option<PathBuf> {
    if display_path.is_empty() {
        return Some(root.to_path_buf());
    }
    if display_path == "~/" || display_path.starts_with("~/") {
        let home = std::env::var_os("HOME")?;
        return Some(PathBuf::from(home).join(display_path.trim_start_matches("~/")));
    }
    if display_path.starts_with('/') || Path::new(display_path).is_absolute() {
        return Some(PathBuf::from(display_path));
    }
    Some(root.join(display_path))
}

fn score_path(path: &str, query: &str, is_directory: bool) -> Option<i32> {
    let path_lower = path.to_ascii_lowercase();
    let basename = path.rsplit('/').next().unwrap_or(path);
    let basename_lower = basename.to_ascii_lowercase();
    let query_lower = query.to_ascii_lowercase();

    let mut score = if query_lower.is_empty() {
        1
    } else if basename_lower == query_lower {
        10_000
    } else if basename_lower.starts_with(&query_lower) {
        8_000
    } else if basename_lower.contains(&query_lower) {
        6_000
    } else if path_lower == query_lower {
        5_000
    } else if path_lower.starts_with(&query_lower) {
        4_500
    } else if path_lower.contains(&query_lower) {
        4_000
    } else if let Some(fuzzy_score) = subsequence_score(&basename_lower, &query_lower) {
        3_000 + fuzzy_score
    } else {
        2_000 + subsequence_score(&path_lower, &query_lower)?
    };

    if is_directory {
        score += 100;
    }
    Some(score)
}

/// Return a small score when every query character appears in order.
fn subsequence_score(candidate: &str, query: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let mut candidate_chars = candidate.char_indices();
    let mut previous_index = None;
    let mut gaps = 0i32;

    for query_character in query.chars() {
        let (index, _) = candidate_chars.find(|(_, character)| *character == query_character)?;
        if let Some(previous) = previous_index {
            gaps += index.saturating_sub(previous + 1) as i32;
        }
        previous_index = Some(index);
    }

    Some(100 - gaps.min(100))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn values(candidates: &[Candidate]) -> Vec<&str> {
        candidates
            .iter()
            .map(|candidate| candidate.value.as_str())
            .collect()
    }

    #[test]
    fn fuzzy_matching_finds_substrings_and_subsequences() {
        assert!(score_path("SomeViewModel.rs", "ViewModel", false).is_some());
        assert!(score_path("some_view_model.rs", "svm", false).is_some());
        assert!(score_path("src/main.rs", "zz", false).is_none());
    }

    #[test]
    fn command_paths_preserve_the_typed_directory_prefix() {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join("src")).unwrap();
        fs::write(directory.path().join("src/main.rs"), "main\n").unwrap();
        let candidates =
            find_path_candidates(directory.path(), "src/ma", &CancellationToken::new());
        assert_eq!(values(&candidates), vec!["src/main.rs"]);
        assert_eq!(candidates[0].kind, CandidateKind::File);

        // A trailing slash scopes the scan to the directory's contents.
        let candidates = find_path_candidates(directory.path(), "src/", &CancellationToken::new());
        assert_eq!(values(&candidates), vec!["src/main.rs"]);

        // A bare prefix completes the directory itself, slash-terminated.
        let candidates = find_path_candidates(directory.path(), "sr", &CancellationToken::new());
        assert_eq!(values(&candidates), vec!["src/"]);
        assert_eq!(candidates[0].kind, CandidateKind::Directory);
    }
}
