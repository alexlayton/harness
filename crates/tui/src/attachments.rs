//! File-reference completion for `@` mentions.
//!
//! The interactive behavior intentionally mirrors Pi: completion inserts a
//! literal `@relative/path` into the user message. The agent sees that path
//! and can inspect it with its normal tools; the TUI does not eagerly read or
//! inject file contents.

use crate::commands::{Candidate, CandidateKind};
use ignore::{DirEntry, WalkBuilder};
use std::path::Path;
use tokio_util::sync::CancellationToken;

/// Maximum number of file/directory candidates returned by one scan.
///
/// This is a completion-performance bound, not an attachment-content limit.
/// The shared visible popup limit lives in `render::MAX_COMPLETION_ROWS`.
pub const MAX_FILE_CANDIDATES: usize = 100;
const MAX_SCANNED_ENTRIES: usize = 50_000;

/// Directories that are almost always generated output or dependency trees.
/// `.git` is also filtered explicitly because it should never be presented as
/// a project reference, even if an ignore configuration would expose it.
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtPrefix {
    /// Byte offset of the `@` in the current line.
    pub token_start: usize,
    /// Byte offset immediately after the whole current token.
    pub token_end: usize,
    /// Text after `@` up to the cursor.
    pub query: String,
}

/// Find the active `@` token in a line at the supplied byte cursor column.
///
/// A reference begins at the start of the line or after whitespace/punctuation
/// and may not contain whitespace before the cursor. The whole token is
/// returned as the replacement range so accepting a result also works when the
/// cursor is in the middle of an existing reference.
pub fn extract_at_prefix(line: &str, cursor_col: usize) -> Option<AtPrefix> {
    let cursor_col = clamp_char_boundary(line, cursor_col);
    let before_cursor = &line[..cursor_col];
    let at = before_cursor.rfind('@')?;

    if at > 0 {
        let previous = line[..at].chars().next_back()?;
        if previous.is_alphanumeric() || previous == '_' {
            return None;
        }
    }

    let query = &line[at + 1..cursor_col];
    if query.chars().any(char::is_whitespace) {
        return None;
    }

    let token_end = line[cursor_col..]
        .char_indices()
        .find(|(_, character)| character.is_whitespace())
        .map(|(offset, _)| cursor_col + offset)
        .unwrap_or(line.len());

    Some(AtPrefix {
        token_start: at,
        token_end,
        query: query.to_owned(),
    })
}

/// Replace the active `@` token and return the new line plus cursor column.
/// Files get a trailing space when the cursor was at the end of the token;
/// directories retain the active completion context with no trailing space.
pub fn replace_at_token(
    line: &str,
    cursor_col: usize,
    value: &str,
    is_directory: bool,
) -> Option<(String, usize)> {
    let prefix = extract_at_prefix(line, cursor_col)?;
    let cursor_col = clamp_char_boundary(line, cursor_col);
    let suffix = if !is_directory && prefix.token_end == cursor_col {
        " "
    } else {
        ""
    };
    let replacement = format!(
        "{}{}{}{}",
        &line[..prefix.token_start],
        value,
        suffix,
        &line[prefix.token_end..]
    );
    let new_cursor = prefix.token_start + value.len() + suffix.len();
    Some((replacement, new_cursor))
}

/// Walk the workspace and return ranked candidates for the current query.
///
/// Git ignore files are honored, including nested files and negation rules.
/// Global Git excludes, `.git/info/exclude`, and `.ignore` files are disabled
/// deliberately so completion behavior is determined by the workspace.
pub fn find_candidates(root: &Path, query: &str, cancel: &CancellationToken) -> Vec<Candidate> {
    let root = root.to_path_buf();
    let mut builder = WalkBuilder::new(&root);
    builder
        .standard_filters(true)
        .hidden(false)
        .parents(false)
        .ignore(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .require_git(false)
        .follow_links(false)
        .threads(1)
        .filter_entry(|entry| !should_prune(entry));

    let query = query.trim_start_matches('/');
    let (scope, match_query) = split_scoped_query(query);
    let mut matches = Vec::new();
    for (scanned, result) in builder.build().enumerate() {
        if cancel.is_cancelled() {
            return Vec::new();
        }
        if scanned >= MAX_SCANNED_ENTRIES {
            break;
        }

        let Ok(entry) = result else {
            continue;
        };
        if entry.depth() == 0 || is_symlink(&entry) {
            continue;
        }
        let is_directory = entry.file_type().is_some_and(|kind| kind.is_dir());
        let Some(relative) = entry.path().strip_prefix(&root).ok() else {
            continue;
        };
        let Some(relative) = display_path(relative) else {
            continue;
        };

        // Quoted path support is intentionally deferred. Do not offer a result
        // that cannot be inserted as one v1 token.
        if relative.chars().any(char::is_whitespace) {
            continue;
        }

        let score_path_value = if let Some(scope) = scope {
            let scope_prefix = format!("{scope}/");
            if relative == scope || !relative.starts_with(&scope_prefix) {
                continue;
            }
            &relative[scope_prefix.len()..]
        } else {
            &relative
        };
        let Some(score) = score_path(score_path_value, match_query, is_directory) else {
            continue;
        };
        matches.push((score, relative, is_directory));
    }

    matches.sort_by(|(score_a, path_a, dir_a), (score_b, path_b, dir_b)| {
        score_b
            .cmp(score_a)
            .then_with(|| dir_b.cmp(dir_a))
            .then_with(|| {
                path_a
                    .to_ascii_lowercase()
                    .cmp(&path_b.to_ascii_lowercase())
            })
            .then_with(|| path_a.cmp(path_b))
    });

    matches
        .into_iter()
        .take(MAX_FILE_CANDIDATES)
        .map(|(_, path, is_directory)| Candidate {
            value: format!("@{}{}", path, if is_directory { "/" } else { "" }),
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

fn should_prune(entry: &DirEntry) -> bool {
    if is_symlink(entry) {
        return true;
    }
    if !entry.file_type().is_some_and(|kind| kind.is_dir()) {
        return false;
    }
    let Some(name) = entry.file_name().to_str() else {
        return true;
    };
    BUILT_IN_IGNORED_DIRECTORIES.contains(&name)
}

fn is_symlink(entry: &DirEntry) -> bool {
    entry.file_type().is_some_and(|kind| kind.is_symlink())
}

fn display_path(path: &Path) -> Option<String> {
    let mut result = String::new();
    for (index, component) in path.components().enumerate() {
        let component = component.as_os_str().to_str()?;
        if index > 0 {
            result.push('/');
        }
        result.push_str(component);
    }
    (!result.is_empty()).then_some(result)
}

fn split_scoped_query(query: &str) -> (Option<&str>, &str) {
    let Some(slash) = query.rfind('/') else {
        return (None, query);
    };
    let scope = query[..slash].trim_matches('/');
    let match_query = &query[slash + 1..];
    if scope.is_empty() {
        (None, match_query)
    } else {
        (Some(scope), match_query)
    }
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

fn clamp_char_boundary(value: &str, index: usize) -> usize {
    let mut index = index.min(value.len());
    while index > 0 && !value.is_char_boundary(index) {
        index -= 1;
    }
    index
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
    fn extracts_at_tokens_at_boundaries_and_not_inside_words() {
        assert_eq!(
            extract_at_prefix("compare @src/main.rs", 20),
            Some(AtPrefix {
                token_start: 8,
                token_end: 20,
                query: "src/main.rs".into(),
            })
        );
        assert!(extract_at_prefix("alex@example.com", 16).is_none());
        assert_eq!(extract_at_prefix("(@src/ma)", 8).unwrap().query, "src/ma");
    }

    #[test]
    fn replacement_preserves_text_after_the_cursor() {
        let (line, cursor) =
            replace_at_token("see @src/ma now", 10, "@src/main.rs", false).unwrap();
        assert_eq!(line, "see @src/main.rs now");
        assert_eq!(cursor, "see @src/main.rs".len());

        let (line, _) = replace_at_token("@src", 4, "@src/", true).unwrap();
        assert_eq!(line, "@src/");
    }

    #[test]
    fn fuzzy_matching_finds_substrings_and_subsequences() {
        assert!(score_path("SomeViewModel.rs", "ViewModel", false).is_some());
        assert!(score_path("some_view_model.rs", "svm", false).is_some());
        assert!(score_path("src/main.rs", "zz", false).is_none());
    }

    #[test]
    fn scan_honors_nested_gitignore_negation_and_builtin_directories() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join(".gitignore"),
            "target/\n*.secret\n!keep.secret\n",
        )
        .unwrap();
        fs::create_dir_all(directory.path().join("src")).unwrap();
        fs::write(directory.path().join("src/.gitignore"), "ignored.rs\n").unwrap();
        fs::write(directory.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        fs::write(directory.path().join("src/ignored.rs"), "ignored\n").unwrap();
        fs::write(directory.path().join("secret.secret"), "secret\n").unwrap();
        fs::write(directory.path().join("keep.secret"), "keep\n").unwrap();
        fs::create_dir_all(directory.path().join("target")).unwrap();
        fs::write(directory.path().join("target/generated.rs"), "generated\n").unwrap();
        fs::create_dir_all(directory.path().join(".git")).unwrap();
        fs::write(directory.path().join(".git/config"), "internal\n").unwrap();
        fs::write(directory.path().join(".hidden.rs"), "hidden\n").unwrap();

        let candidates = find_candidates(directory.path(), "", &CancellationToken::new());
        let values = values(&candidates);
        assert!(values.contains(&"@src/"));
        assert!(values.contains(&"@src/main.rs"));
        assert!(values.contains(&"@keep.secret"));
        assert!(values.contains(&"@.hidden.rs"));
        assert!(!values.contains(&"@src/ignored.rs"));
        assert!(!values.contains(&"@secret.secret"));
        assert!(!values.contains(&"@target/"));
        assert!(!values.iter().any(|value| value.starts_with("@.git/")));
    }

    #[test]
    fn directories_are_candidates_without_file_space_semantics() {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join("src")).unwrap();
        fs::write(directory.path().join("src/main.rs"), "main\n").unwrap();
        let candidates = find_candidates(directory.path(), "src", &CancellationToken::new());
        let directory_candidate = candidates
            .iter()
            .find(|candidate| candidate.value == "@src/")
            .unwrap();
        assert_eq!(directory_candidate.kind, CandidateKind::Directory);
        let file_candidate = candidates
            .iter()
            .find(|candidate| candidate.value == "@src/main.rs")
            .unwrap();
        assert_eq!(file_candidate.kind, CandidateKind::File);

        let scoped = find_candidates(directory.path(), "src/", &CancellationToken::new());
        assert!(!values(&scoped).contains(&"@src/"));
        assert!(values(&scoped).contains(&"@src/main.rs"));
    }
}
