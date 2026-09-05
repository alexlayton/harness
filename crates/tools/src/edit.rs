use super::file_mutation::{atomic_write, with_file_mutation_lock};
use super::vfs::{WorkspaceFs, split_relative};
use super::{
    Tool, ToolOutput, ToolPrompt, ToolSpec, normalize_workspace_root, resolve_workspace_path,
};
use async_trait::async_trait;
use llm::ToolDefinition;
use llm::util::truncate_utf8;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio_util::sync::CancellationToken;

const MAX_DIFF_LINES: usize = 80;
const MAX_DIFF_INPUT_LINES: usize = 10_000;
const DIFF_CONTEXT_LINES: usize = 2;
const MAX_DIFF_LINE_BYTES: usize = 400;
const MAX_DIFF_OUTPUT_BYTES: usize = 12 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Edit {
    old_text: String,
    new_text: String,
}

#[derive(Clone, Debug)]
struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AppliedEdits {
    base_content: String,
    new_content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DiffSummary {
    text: String,
    added_lines: usize,
    removed_lines: usize,
}

pub struct EditTool {
    workspace_root: Option<PathBuf>,
}

impl EditTool {
    pub fn with_workspace_root(root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: Some(normalize_workspace_root(root)),
        }
    }
}

#[async_trait]
impl Tool for EditTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
            name: "edit".into(),
            description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)"
                    },
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text for one targeted replacement. It must be unique in the original file."
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text for this targeted edit."
                                }
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["path", "edits"],
                "additionalProperties": false
            }),
            },
            prompt: ToolPrompt::new(
                "Apply exact replacements",
                ["Use edit for targeted changes; oldText matches the original file.".to_owned()],
            ),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let (path, edits) = match parse_args(&args) {
            Ok(value) => value,
            Err(message) => return error("edit", &message),
        };
        let summary = format!("edit {path}");
        if cancel.is_cancelled() {
            return error(&summary, "cancelled");
        }

        // Resolve lexically first (for precise workspace-relative errors),
        // then open through the validated workspace handle: the file read,
        // matched, and committed is opened handle-relatively, so an ancestor
        // swapped after validation cannot redirect the edit. The
        // pre-commit re-read below compares against bytes read from the
        // same handle.
        let components =
            match resolve_workspace_path(&path, self.workspace_root.as_deref(), true).await {
                Ok(_) => match split_relative(&path) {
                    Ok(components) => components,
                    Err(io_error) => {
                        return error(&summary, &format!("cannot edit {path}: {io_error}"));
                    }
                },
                Err(message) => return error(&summary, &format!("cannot edit {path}: {message}")),
            };
        let root = self.workspace_root.clone();
        let edit_path_display = path.clone();
        let edit_cancel = cancel.clone();
        let edit_result = with_file_mutation_lock(
            &root
                .as_deref()
                .unwrap_or(std::path::Path::new("."))
                .join(components.join("/")),
            &cancel,
            move || async move {
                execute_edit_validated(
                    &edit_path_display,
                    root.as_deref(),
                    &components,
                    &edits,
                    &edit_cancel,
                )
                .await
            },
        )
        .await;

        let Some(edit_result) = edit_result else {
            return error(&summary, "cancelled");
        };
        match edit_result {
            Ok(result) => {
                let diff = generate_diff(&result.base_content, &result.new_content);
                let mut content = format!(
                    "Successfully replaced {} block(s) in {path} (+{} / -{} lines).",
                    result.replacement_count, diff.added_lines, diff.removed_lines,
                );
                if !diff.text.is_empty() {
                    content.push('\n');
                    content.push_str(&diff.text);
                }
                ToolOutput {
                    content,
                    is_error: false,
                    summary,
                }
            }
            Err(message) => error(&summary, &message),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EditResult {
    base_content: String,
    new_content: String,
    replacement_count: usize,
}

/// Handle-validated edit: open the file through the workspace handle,
/// match against bytes read from that handle, and commit through the
/// validated parent handle. The pre-commit re-read compares handle-read
/// bytes, so a swapped ancestor cannot redirect the result.
async fn execute_edit_validated(
    path: &str,
    root: Option<&Path>,
    components: &[String],
    edits: &[Edit],
    cancel: &CancellationToken,
) -> Result<EditResult, String> {
    check_cancelled(cancel)?;
    #[cfg(unix)]
    if let Some(root) = root {
        use std::io::Read;
        let fs = WorkspaceFs::open_root(root)
            .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        let fd = super::vfs::unix::open_file_relative(&fs, components)
            .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        let mut original_bytes = Vec::new();
        let mut file = std::fs::File::from(fd);
        file.read_to_end(&mut original_bytes)
            .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        check_cancelled(cancel)?;
        if original_bytes.contains(&0) {
            return Err(format!(
                "Could not edit file: {path}. Binary files are not supported."
            ));
        }
        let raw_content = String::from_utf8(original_bytes.clone()).map_err(|_| {
            format!("Could not edit file: {path}. The file is not valid UTF-8 or is binary.")
        })?;
        let (bom, content) = strip_bom(&raw_content);
        let applied = apply_edits_exact(content, edits, path)?;
        check_cancelled(cancel)?;
        let mut final_content = String::with_capacity(bom.len() + applied.new_content.len());
        final_content.push_str(bom);
        final_content.push_str(&applied.new_content);
        // Re-read from the same handle before committing.
        let fd = super::vfs::unix::open_file_relative(&fs, components)
            .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        let mut current_bytes = Vec::new();
        let mut current = std::fs::File::from(fd);
        current
            .read_to_end(&mut current_bytes)
            .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        if current_bytes != original_bytes {
            return Err(format!(
                "Could not edit file: {path}. The file changed while the edit was being prepared; no changes were made."
            ));
        }
        check_cancelled(cancel)?;
        let (parent_fd, name) = super::vfs::unix::open_parent_relative(&fs, components)
            .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        super::file_mutation::atomic_write_at(
            &parent_fd,
            &name,
            final_content.as_bytes(),
            None,
            cancel,
        )
        .await
        .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
        return Ok(EditResult {
            base_content: applied.base_content,
            new_content: applied.new_content,
            replacement_count: edits.len(),
        });
    }
    // Fallback (non-Unix or compatibility mode): path-based edit.
    let base = root
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let target_path = components.iter().fold(base, |base, part| base.join(part));
    execute_edit(path, &target_path, edits, cancel).await
}

async fn execute_edit(
    path: &str,
    target_path: &Path,
    edits: &[Edit],
    cancel: &CancellationToken,
) -> Result<EditResult, String> {
    check_cancelled(cancel)?;

    let metadata = fs::metadata(target_path)
        .await
        .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
    if metadata.is_dir() {
        return Err(format!("Could not edit file: {path}. It is a directory."));
    }

    let original_bytes = fs::read(target_path)
        .await
        .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
    check_cancelled(cancel)?;
    if original_bytes.contains(&0) {
        return Err(format!(
            "Could not edit file: {path}. Binary files are not supported."
        ));
    }
    let raw_content = String::from_utf8(original_bytes.clone()).map_err(|_| {
        format!("Could not edit file: {path}. The file is not valid UTF-8 or is binary.")
    })?;

    // Byte-exact editing: match every `oldText` against the original bytes
    // (BOM-aware, mixed line endings preserved) with no LF normalization
    // and no Unicode or whitespace folding.  All spans are computed against
    // the original content, overlaps rejected, then applied in one pass.
    // Every byte outside the replacement spans is preserved.
    let (bom, content) = strip_bom(&raw_content);
    let applied = apply_edits_exact(content, edits, path)?;
    check_cancelled(cancel)?;

    let mut final_content = String::with_capacity(bom.len() + applied.new_content.len());
    final_content.push_str(bom);
    final_content.push_str(&applied.new_content);
    let final_bytes = final_content.as_bytes();

    // Re-read the snapshot immediately before committing. The per-file queue
    // handles other harness mutations; this check catches ordinary external
    // edits without silently overwriting them.
    let current_bytes = fs::read(target_path)
        .await
        .map_err(|error| format!("Could not edit file: {path}. {error}"))?;
    if current_bytes != original_bytes {
        return Err(format!(
            "Could not edit file: {path}. The file changed while the edit was being prepared; no changes were made."
        ));
    }
    check_cancelled(cancel)?;

    atomic_write(target_path, final_bytes, cancel)
        .await
        .map_err(|error| format!("Could not edit file: {path}. {error}"))?;

    Ok(EditResult {
        base_content: applied.base_content,
        new_content: applied.new_content,
        replacement_count: edits.len(),
    })
}

fn parse_args(args: &Value) -> Result<(String, Vec<Edit>), String> {
    let path = match args.get("path").and_then(Value::as_str) {
        Some(path) if !path.is_empty() => path.to_owned(),
        _ => return Err("missing required argument: path".into()),
    };

    let edits = match args.get("edits") {
        Some(Value::Array(values)) => parse_edit_array(values)?,
        Some(_) => return Err("edits must be an array of replacement objects".into()),
        None => Vec::new(),
    };

    if edits.is_empty() {
        return Err("edits must contain at least one replacement".into());
    }
    Ok((path, edits))
}

fn parse_edit_array(values: &[Value]) -> Result<Vec<Edit>, String> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let Some(object) = value.as_object() else {
                return Err(format!("edits[{index}] must be an object"));
            };
            let old_text = object
                .get("oldText")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("edits[{index}].oldText must be a string"))?;
            let new_text = object
                .get("newText")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("edits[{index}].newText must be a string"))?;
            Ok(Edit {
                old_text: old_text.to_owned(),
                new_text: new_text.to_owned(),
            })
        })
        .collect()
}

/// Match every edit exactly and uniquely against the original content
/// (post-BOM), with no normalization and no fuzzy fallback.  Replacement
/// spans are computed against the original string, checked for overlap,
/// then applied from the end so earlier spans stay valid.  The caller
/// reattaches the BOM, preserving it and every byte outside the spans —
/// including mixed CRLF/LF endings, typographic characters, and trailing
/// whitespace.
fn apply_edits_exact(content: &str, edits: &[Edit], path: &str) -> Result<AppliedEdits, String> {
    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(format!(
                "edits[{index}].oldText must not be empty in {path}."
            ));
        }
    }

    let mut matched_edits = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        let first = content
            .find(&edit.old_text)
            .ok_or_else(|| not_found_error(path, index, edits.len()))?;
        if content[first + edit.old_text.len()..].contains(&edit.old_text) {
            return Err(duplicate_error(
                path,
                index,
                edits.len(),
                count_occurrences(content, &edit.old_text),
            ));
        }
        matched_edits.push(MatchedEdit {
            edit_index: index,
            match_index: first,
            match_length: edit.old_text.len(),
            new_text: edit.new_text.clone(),
        });
    }

    matched_edits.sort_by_key(|edit| edit.match_index);
    for pair in matched_edits.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if previous.match_index + previous.match_length > current.match_index {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                previous.edit_index, current.edit_index
            ));
        }
    }

    let original = content.to_owned();
    let new_content = apply_replacements(content, &matched_edits);
    if original == new_content {
        return Err(no_change_error(path, edits.len()));
    }

    Ok(AppliedEdits {
        base_content: original,
        new_content,
    })
}

fn apply_replacements(content: &str, replacements: &[MatchedEdit]) -> String {
    let mut result = content.to_owned();
    for replacement in replacements.iter().rev() {
        let start = replacement.match_index;
        let end = start + replacement.match_length;
        result.replace_range(start..end, &replacement.new_text);
    }
    result
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    if old_text.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut offset = 0;
    while let Some(index) = content[offset..].find(old_text) {
        count += 1;
        offset += index + old_text.len();
    }
    count
}

fn strip_bom(content: &str) -> (&str, &str) {
    content
        .strip_prefix('\u{feff}')
        .map_or(("", content), |content| ("\u{feff}", content))
}

fn not_found_error(path: &str, index: usize, total: usize) -> String {
    if total == 1 {
        format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    } else {
        format!(
            "Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
    }
}

fn duplicate_error(path: &str, index: usize, total: usize, occurrences: usize) -> String {
    if total == 1 {
        format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    } else {
        format!(
            "Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn no_change_error(path: &str, total: usize) -> String {
    if total == 1 {
        format!("No changes made to {path}. The replacement produced identical content.")
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

fn generate_diff(old_content: &str, new_content: &str) -> DiffSummary {
    let old_line_count = old_content.split('\n').count();
    let new_line_count = new_content.split('\n').count();
    if old_line_count + new_line_count > MAX_DIFF_INPUT_LINES {
        return DiffSummary {
            text: "[diff omitted: file is too large for a bounded preview]".into(),
            added_lines: new_line_count.saturating_sub(old_line_count),
            removed_lines: old_line_count.saturating_sub(new_line_count),
        };
    }

    let parts = diff::lines(old_content, new_content);
    let mut entries = Vec::with_capacity(parts.len());
    let mut added_lines = 0;
    let mut removed_lines = 0;
    for part in parts {
        match part {
            diff::Result::Both(old, _) => entries.push(DiffEntry::Context(old.to_owned())),
            diff::Result::Left(old) => {
                removed_lines += 1;
                entries.push(DiffEntry::Removed(old.to_owned()));
            }
            diff::Result::Right(new) => {
                added_lines += 1;
                entries.push(DiffEntry::Added(new.to_owned()));
            }
        }
    }

    let changed_indices = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| (!entry.is_context()).then_some(index))
        .collect::<Vec<_>>();
    if changed_indices.is_empty() {
        return DiffSummary {
            text: String::new(),
            added_lines,
            removed_lines,
        };
    }

    let mut selected = vec![false; entries.len()];
    for index in changed_indices {
        let start = index.saturating_sub(DIFF_CONTEXT_LINES);
        let end = (index + DIFF_CONTEXT_LINES + 1).min(entries.len());
        selected[start..end].fill(true);
    }

    let mut text = String::new();
    let mut previous_selected = None;
    let mut output_lines = 0;
    let mut truncated = false;
    for (index, entry) in entries.iter().enumerate() {
        if !selected[index] {
            continue;
        }
        if let Some(previous) = previous_selected
            && index > previous + 1
        {
            append_diff_text(&mut text, "  …\n", &mut truncated);
        }
        let (prefix, line) = match entry {
            DiffEntry::Context(line) => ("  ", line.as_str()),
            DiffEntry::Removed(line) => ("- ", line.as_str()),
            DiffEntry::Added(line) => ("+ ", line.as_str()),
        };
        if output_lines >= MAX_DIFF_LINES {
            truncated = true;
            break;
        }
        let display_line = truncate_utf8(line, MAX_DIFF_LINE_BYTES);
        append_diff_text(
            &mut text,
            &format!("{prefix}{display_line}\n"),
            &mut truncated,
        );
        output_lines += 1;
        previous_selected = Some(index);
        if truncated {
            break;
        }
    }
    if truncated && text.len() + "… diff truncated\n".len() <= MAX_DIFF_OUTPUT_BYTES {
        text.push_str("… diff truncated\n");
    }

    DiffSummary {
        text,
        added_lines,
        removed_lines,
    }
}

#[derive(Clone, Debug)]
enum DiffEntry {
    Context(String),
    Removed(String),
    Added(String),
}

impl DiffEntry {
    fn is_context(&self) -> bool {
        matches!(self, Self::Context(_))
    }
}

fn append_diff_text(output: &mut String, value: &str, truncated: &mut bool) {
    if *truncated {
        return;
    }
    if output.len() + value.len() <= MAX_DIFF_OUTPUT_BYTES {
        output.push_str(value);
        return;
    }
    *truncated = true;
}

fn error(summary: &str, content: &str) -> ToolOutput {
    ToolOutput {
        content: content.to_owned(),
        is_error: true,
        summary: summary.to_owned(),
    }
}

fn check_cancelled(cancel: &CancellationToken) -> Result<(), String> {
    if cancel.is_cancelled() {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn replaces_one_block_and_returns_a_bounded_diff() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("main.rs");
        fs::write(&path, "fn main() {\n    println!(\"old\");\n}\n").unwrap();

        let output = EditTool::with_workspace_root(directory.path())
            .execute(
                json!({
                    "path": "main.rs",
                    "edits": [{
                        "oldText": "println!(\"old\")",
                        "newText": "println!(\"new\")"
                    }]
                }),
                CancellationToken::new(),
            )
            .await;

        assert!(!output.is_error, "{}", output.content);
        assert!(output.content.contains("Successfully replaced 1 block"));
        assert!(
            output.content.contains("-     println!(\"old\")"),
            "{}",
            output.content
        );
        assert!(
            output.content.contains("+     println!(\"new\")"),
            "{}",
            output.content
        );
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "fn main() {\n    println!(\"new\");\n}\n"
        );
    }

    #[tokio::test]
    async fn applies_multiple_disjoint_edits_against_the_original() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("file.txt");
        fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        let output = EditTool::with_workspace_root(directory.path())
            .execute(
                json!({
                    "path": "file.txt",
                    "edits": [
                        {"oldText": "alpha", "newText": "one"},
                        {"oldText": "gamma", "newText": "three"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await;

        assert!(!output.is_error, "{}", output.content);
        assert_eq!(fs::read_to_string(path).unwrap(), "one\nbeta\nthree\n");
    }

    #[tokio::test]
    async fn rejects_missing_duplicate_overlap_and_empty_matches() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("file.txt");
        fs::write(&path, "same\nsame\nabcdef\n").unwrap();

        let cases = [
            (
                json!([{"oldText":"missing", "newText":"x"}]),
                "Could not find",
            ),
            (
                json!([{"oldText":"same", "newText":"x"}]),
                "Found 2 occurrences",
            ),
            (
                json!([
                    {"oldText":"abcdef", "newText":"x"},
                    {"oldText":"cde", "newText":"y"}
                ]),
                "overlap",
            ),
            (json!([{"oldText":"", "newText":"x"}]), "must not be empty"),
        ];

        for (edits, expected) in cases {
            let output = EditTool::with_workspace_root(directory.path())
                .execute(
                    json!({"path": "file.txt", "edits": edits}),
                    CancellationToken::new(),
                )
                .await;
            assert!(output.is_error, "unexpected success: {}", output.content);
            assert!(output.content.contains(expected), "{}", output.content);
        }
        assert_eq!(fs::read_to_string(path).unwrap(), "same\nsame\nabcdef\n");
    }

    #[tokio::test]
    async fn preserves_bom_and_crlf_line_endings() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("file.txt");
        fs::write(&path, "\u{feff}first\r\nsecond\r\n").unwrap();

        let output = EditTool::with_workspace_root(directory.path())
            .execute(
                json!({
                    "path": "file.txt",
                    "edits": [{"oldText":"second", "newText":"changed"}]
                }),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert_eq!(
            fs::read(path).unwrap(),
            "\u{feff}first\r\nchanged\r\n".as_bytes()
        );
    }

    #[test]
    fn mixed_crlf_endings_are_preserved_outside_spans() {
        // Only the replaced span changes; untouched CRLF/LF endings stay.
        let content = "first\r\nsecond\r\nthird\nfourth\r\n";
        let edits = vec![Edit {
            old_text: "second".into(),
            new_text: "CHANGED".into(),
        }];
        let result = apply_edits_exact(content, &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "first\r\nCHANGED\r\nthird\nfourth\r\n");
    }

    #[test]
    fn typographic_chars_and_trailing_spaces_require_exact_match() {
        // Curly quotes, dashes, compat chars, and trailing spaces no longer
        // fold: they must match byte-exactly or fail as not-found.
        for (content, old) in [
            ("say \u{2018}hello\u{2019}\n", "say 'hello'"),
            ("a\u{2014}b\n", "a-b"),
            ("caf\u{00e9}\n", "cafe\u{0301}"),
            ("foo\n", "foo  "),
        ] {
            let edits = vec![Edit {
                old_text: old.into(),
                new_text: "x".into(),
            }];
            let error = apply_edits_exact(content, &edits, "file.txt").unwrap_err();
            assert!(
                error.contains("Could not find"),
                "{content:?} vs {old:?}: {error}"
            );
        }
        // Substring matching is exact: "foo" matches inside "foo  ", but
        // trailing spaces in oldText must be present in the file.
        let edits = vec![Edit {
            old_text: "foo".into(),
            new_text: "x".into(),
        }];
        let result = apply_edits_exact("foo  \n", &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "x  \n");
        // Exact duplicates still report uniqueness errors.
        let edits = vec![Edit {
            old_text: "same".into(),
            new_text: "x".into(),
        }];
        let error = apply_edits_exact("same\nsame\n", &edits, "file.txt").unwrap_err();
        assert!(error.contains("Found 2 occurrences"), "{error}");
    }

    #[test]
    fn batch_edits_match_independently_and_overlap_is_rejected() {
        let edits = vec![
            Edit {
                old_text: "alpha".into(),
                new_text: "one".into(),
            },
            Edit {
                old_text: "gamma".into(),
                new_text: "three".into(),
            },
        ];
        let result = apply_edits_exact("alpha\nbeta\ngamma\n", &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "one\nbeta\nthree\n");
        let edits = vec![
            Edit {
                old_text: "abcdef".into(),
                new_text: "x".into(),
            },
            Edit {
                old_text: "cde".into(),
                new_text: "y".into(),
            },
        ];
        let error = apply_edits_exact("abcdef\n", &edits, "file.txt").unwrap_err();
        assert!(error.contains("overlap"), "{error}");
    }

    proptest! {
        /// Every byte outside the replacement spans is unchanged: remove the
        /// replaced spans from both sides and require identical remainders.
        #[test]
        fn byte_preservation_outside_spans(
            lines in prop::collection::vec("[a-zA-Z0-9 \t\u{00e9}\u{2019}\u{2014}]{0,12}", 2..8),
        ) {
            use std::collections::HashSet;
            // Build content with unique lines so spans are unambiguous.
            let mut seen = HashSet::new();
            let mut unique = Vec::new();
            for (i, line) in lines.iter().enumerate() {
                let candidate = format!("{i}_{line}");
                if seen.insert(candidate.clone()) {
                    unique.push(candidate);
                }
            }
            prop_assume!(unique.len() >= 2);
            let content = format!("{}\n", unique.join("\n"));
            // Two disjoint single-line edits.
            let edits = vec![
                Edit { old_text: unique[0].clone(), new_text: "NEW0".into() },
                Edit { old_text: unique[unique.len() - 1].clone(), new_text: "NEW1".into() },
            ];
            let result = apply_edits_exact(&content, &edits, "file.txt").unwrap();
            // Mask out the replaced spans on both sides; remainders match.
            let mut masked_original = content.clone();
            let mut masked_new = result.new_content.clone();
            for (old, new) in [(unique[0].as_str(), "NEW0"), (unique[unique.len()-1].as_str(), "NEW1")] {
                masked_original = masked_original.replacen(old, "\0", 1);
                masked_new = masked_new.replacen(new, "\0", 1);
            }
            prop_assert_eq!(masked_original, masked_new);
        }
    }

    #[test]
    fn diff_is_bounded_for_large_inputs() {
        let old = (0..6_000)
            .map(|line| format!("old {line}\n"))
            .collect::<String>();
        let new = (0..6_000)
            .map(|line| format!("new {line}\n"))
            .collect::<String>();
        let diff = generate_diff(&old, &new);
        assert!(diff.text.contains("diff omitted"));
        assert!(diff.text.len() < 200);
    }

    /// End-to-end TOCTOU barrier for `edit`: resolve, swap an ancestor for
    /// an external symlink, then execute. The handle-relative open must
    /// refuse and leave both trees unchanged.
    #[cfg(unix)]
    #[tokio::test]
    async fn edit_cannot_replace_through_swapped_ancestor() {
        let workspace = tempdir().unwrap();
        let root = std::fs::canonicalize(workspace.path()).unwrap();
        let outside = tempdir().unwrap();
        std::fs::write(outside.path().join("target.txt"), "EXTERNAL").unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/file.txt"), "hello\n").unwrap();
        let tool = EditTool::with_workspace_root(&root);
        // Warm the resolution path, then swap the ancestor.
        let before = tool
            .execute(
                json!({
                    "path": "sub/file.txt",
                    "edits": [{"oldText": "hello", "newText": "warm"}]
                }),
                CancellationToken::new(),
            )
            .await;
        assert!(!before.is_error, "{}", before.content);
        std::fs::write(root.join("sub/file.txt"), "hello\n").unwrap();
        std::fs::remove_dir_all(root.join("sub")).unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("sub")).unwrap();
        let output = tool
            .execute(
                json!({
                    "path": "sub/file.txt",
                    "edits": [{"oldText": "hello", "newText": "evil"}]
                }),
                CancellationToken::new(),
            )
            .await;
        // The swapped ancestor no longer contains the old text (it now
        // points outside); either a not-found or a confinement error is
        // acceptable, but the outside file must be unchanged.
        assert!(output.is_error, "edit escaped: {}", output.content);
        assert_eq!(
            std::fs::read_to_string(outside.path().join("target.txt")).unwrap(),
            "EXTERNAL"
        );
        assert!(!outside.path().join("file.txt").exists());
    }
}
