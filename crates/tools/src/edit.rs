use super::file_mutation::{atomic_write, with_file_mutation_lock};
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
use unicode_normalization::UnicodeNormalization;

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
    /// Set when a match had to fall back to the lenient whitespace mode.
    notice: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineEnding {
    Lf,
    Crlf,
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
                "Make precise file edits with exact text replacement",
                ["Use edit for targeted changes; match oldText against the original file.".to_owned()],
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

        let requested_path =
            match resolve_workspace_path(&path, self.workspace_root.as_deref(), true).await {
                Ok(path) => path,
                Err(message) => return error(&summary, &format!("cannot edit {path}: {message}")),
            };
        let target_path = match fs::canonicalize(&requested_path).await {
            Ok(path) => path,
            Err(io_error) => {
                return error(&summary, &format!("cannot edit {path}: {io_error}"));
            }
        };
        let edit_path = target_path.clone();
        let edit_path_display = path.clone();
        let edit_cancel = cancel.clone();
        let edit_result = with_file_mutation_lock(&target_path, &cancel, move || async move {
            execute_edit(&edit_path_display, &edit_path, &edits, &edit_cancel).await
        })
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
                if let Some(notice) = result.notice.as_deref() {
                    content.push('\n');
                    content.push_str(notice);
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
    notice: Option<String>,
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

    let (bom, content) = strip_bom(&raw_content);
    let original_ending = detect_line_ending(content);
    let normalized_content = normalize_to_lf(content);
    let applied = apply_edits_to_normalized_content(&normalized_content, edits, path)?;
    check_cancelled(cancel)?;

    let restored_content = restore_line_endings(&applied.new_content, original_ending);
    let mut final_content = String::with_capacity(bom.len() + restored_content.len());
    final_content.push_str(bom);
    final_content.push_str(&restored_content);
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
        notice: applied.notice,
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

fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEdits, String> {
    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(format!(
                "edits[{index}].oldText must not be empty in {path}."
            ));
        }
    }

    let normalized_edits = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect::<Vec<_>>();

    // Match every edit against a single base content normalized at the most
    // aggressive mode any edit needs, so all match indices share one
    // coordinate space for the replacement pass.
    let mode = normalized_edits
        .iter()
        .filter_map(|edit| find_best_match(normalized_content, &edit.old_text))
        .map(|result| result.mode)
        .max()
        .unwrap_or(MatchMode::Exact);
    let base_content = match mode {
        MatchMode::Exact => normalized_content.to_owned(),
        MatchMode::Unicode => normalize_for_fuzzy_match(normalized_content),
        MatchMode::LenientWhitespace => normalize_lenient_whitespace(normalized_content),
    };

    let mut matched_edits = Vec::with_capacity(normalized_edits.len());
    for (index, edit) in normalized_edits.iter().enumerate() {
        let Some(found) = find_best_match(&base_content, &edit.old_text) else {
            return Err(not_found_error(path, index, normalized_edits.len()));
        };
        let occurrences = count_occurrences(&base_content, &edit.old_text, found.mode);
        if occurrences > 1 {
            return Err(duplicate_error(
                path,
                index,
                normalized_edits.len(),
                occurrences,
            ));
        }
        matched_edits.push(MatchedEdit {
            edit_index: index,
            match_index: found.index,
            match_length: found.length,
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

    let original = normalized_content.to_owned();
    let new_content = if mode == MatchMode::Exact {
        apply_replacements(&base_content, &matched_edits)
    } else {
        // The base differs from the original (unicode folding and/or
        // whitespace trimming), so rebuild the file from the original,
        // swapping in the base's replaced regions.
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &base_content,
            &matched_edits,
        )?
    };
    if original == new_content {
        return Err(no_change_error(path, normalized_edits.len()));
    }

    let notice = (mode == MatchMode::LenientWhitespace).then(|| {
        "note: oldText was matched leniently — trailing whitespace on matched lines was ignored."
            .to_owned()
    });

    Ok(AppliedEdits {
        base_content: original,
        new_content,
        notice,
    })
}

/// How aggressively an `oldText` match had to be relaxed before it succeeded.
/// `Exact` matches the literal text; `Unicode` folds NFKC and typographic
/// variants; the lenient mode additionally ignores trailing whitespace on
/// each line and is always reported to the user.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MatchMode {
    Exact,
    Unicode,
    LenientWhitespace,
}

#[derive(Clone, Copy, Debug)]
struct FoundMatch {
    index: usize,
    length: usize,
    mode: MatchMode,
}

fn find_best_match(content: &str, old_text: &str) -> Option<FoundMatch> {
    if let Some(index) = content.find(old_text) {
        return Some(FoundMatch {
            index,
            length: old_text.len(),
            mode: MatchMode::Exact,
        });
    }

    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    if !fuzzy_old_text.is_empty()
        && let Some(index) = fuzzy_content.find(&fuzzy_old_text)
    {
        return Some(FoundMatch {
            index,
            length: fuzzy_old_text.len(),
            mode: MatchMode::Unicode,
        });
    }

    let lenient_content = normalize_lenient_whitespace(content);
    let lenient_old_text = normalize_lenient_whitespace(old_text);
    if lenient_old_text.is_empty() {
        return None;
    }
    lenient_content
        .find(&lenient_old_text)
        .map(|index| FoundMatch {
            index,
            length: lenient_old_text.len(),
            mode: MatchMode::LenientWhitespace,
        })
}

fn count_occurrences(content: &str, old_text: &str, mode: MatchMode) -> usize {
    let (haystack, needle) = match mode {
        MatchMode::Exact => (content.to_owned(), old_text.to_owned()),
        MatchMode::Unicode => (
            normalize_for_fuzzy_match(content),
            normalize_for_fuzzy_match(old_text),
        ),
        MatchMode::LenientWhitespace => (
            normalize_lenient_whitespace(content),
            normalize_lenient_whitespace(old_text),
        ),
    };
    if needle.is_empty() {
        return 0;
    }

    let mut count = 0;
    let mut offset = 0;
    while let Some(index) = haystack[offset..].find(&needle) {
        count += 1;
        offset += index + needle.len();
    }
    count
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

#[derive(Clone, Debug)]
struct LineSpan {
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct ReplacementGroup {
    start_line: usize,
    end_line: usize,
    replacements: Vec<MatchedEdit>,
}

fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[MatchedEdit],
) -> Result<String, String> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err("cannot preserve unchanged lines after fuzzy matching".into());
    }

    let mut groups = Vec::<ReplacementGroup>::new();
    for replacement in replacements {
        let range = replacement_line_range(&base_lines, replacement)?;
        if let Some(current) = groups.last_mut()
            && range.0 < current.end_line
        {
            current.end_line = current.end_line.max(range.1);
            current.replacements.push(replacement.clone());
        } else {
            groups.push(ReplacementGroup {
                start_line: range.0,
                end_line: range.1,
                replacements: vec![replacement.clone()],
            });
        }
    }

    let mut result = String::new();
    let mut original_line_index = 0;
    for group in groups {
        for line in &original_lines[original_line_index..group.start_line] {
            result.push_str(line);
        }

        let group_start = base_lines[group.start_line].start;
        let group_end = base_lines[group.end_line - 1].end;
        result.push_str(&apply_replacements_in_range(
            &base_content[group_start..group_end],
            &group.replacements,
            group_start,
        ));
        original_line_index = group.end_line;
    }
    for line in &original_lines[original_line_index..] {
        result.push_str(line);
    }
    Ok(result)
}

fn apply_replacements_in_range(
    content: &str,
    replacements: &[MatchedEdit],
    offset: usize,
) -> String {
    let mut result = content.to_owned();
    for replacement in replacements.iter().rev() {
        let start = replacement.match_index - offset;
        let end = start + replacement.match_length;
        result.replace_range(start..end, &replacement.new_text);
    }
    result
}

fn replacement_line_range(
    lines: &[LineSpan],
    replacement: &MatchedEdit,
) -> Result<(usize, usize), String> {
    let replacement_start = replacement.match_index;
    let replacement_end = replacement.match_index + replacement.match_length;
    let Some(start_line) = lines
        .iter()
        .position(|line| replacement_start >= line.start && replacement_start < line.end)
    else {
        return Err("replacement range is outside the file".into());
    };
    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < replacement_end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        return Err("replacement range is outside the file".into());
    }
    Ok((start_line, end_line + 1))
}

fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            lines.push(&content[start..index + 1]);
            start = index + 1;
        }
    }
    if start < content.len() {
        lines.push(&content[start..]);
    }
    lines
}

fn line_spans(content: &str) -> Vec<LineSpan> {
    let mut spans = Vec::new();
    let mut start = 0;
    for (index, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            spans.push(LineSpan {
                start,
                end: index + 1,
            });
            start = index + 1;
        }
    }
    if start < content.len() {
        spans.push(LineSpan {
            start,
            end: content.len(),
        });
    }
    spans
}

fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Unicode-only fuzzy normalization: NFKC plus folding of typographic quotes,
/// dashes, and whitespace variants.  Trailing whitespace is deliberately
/// preserved, so oldText that includes trailing spaces only matches lines
/// that actually have them.
fn normalize_for_fuzzy_match(text: &str) -> String {
    text.nfkc()
        .collect::<String>()
        .chars()
        .map(|character| match character {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{00a0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
            | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200a}' | '\u{202f}' | '\u{205f}'
            | '\u{3000}' => ' ',
            character => character,
        })
        .collect()
}

/// A separate, lenient mode that additionally ignores trailing whitespace on
/// every line.  Matches that only succeed here are reported to the user.
fn normalize_lenient_whitespace(text: &str) -> String {
    normalize_for_fuzzy_match(text)
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_bom(content: &str) -> (&str, &str) {
    content
        .strip_prefix('\u{feff}')
        .map_or(("", content), |content| ("\u{feff}", content))
}

fn detect_line_ending(content: &str) -> LineEnding {
    let Some(crlf_index) = content.find("\r\n") else {
        return LineEnding::Lf;
    };
    let Some(lf_index) = content.find('\n') else {
        return LineEnding::Lf;
    };
    if crlf_index <= lf_index {
        LineEnding::Crlf
    } else {
        LineEnding::Lf
    }
}

fn restore_line_endings(content: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::Lf => content.to_owned(),
        LineEnding::Crlf => content.replace('\n', "\r\n"),
    }
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
    fn fuzzy_matching_handles_whitespace_and_typographic_variants() {
        let edits = vec![Edit {
            old_text: "let value = 1;".into(),
            new_text: "let value = 2;".into(),
        }];
        let result =
            apply_edits_to_normalized_content("let value = 1;   \n", &edits, "file.rs").unwrap();
        assert_eq!(result.new_content, "let value = 2;   \n");

        let edits = vec![Edit {
            old_text: "say 'hello'".into(),
            new_text: "say 'goodbye'".into(),
        }];
        let result =
            apply_edits_to_normalized_content("say ‘hello’\n", &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "say 'goodbye'\n");
    }

    #[test]
    fn unicode_fuzzy_match_preserves_trailing_whitespace_of_untouched_text() {
        // Quote folding must not silently trim trailing whitespace: the
        // untouched portion of the matched line keeps its trailing spaces.
        let edits = vec![Edit {
            old_text: "say 'hello'".into(),
            new_text: "say 'goodbye'".into(),
        }];
        let result =
            apply_edits_to_normalized_content("say ‘hello’  \n", &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "say 'goodbye'  \n");
        assert!(result.notice.is_none());
    }

    #[test]
    fn lenient_whitespace_matches_are_reported() {
        let edits = vec![Edit {
            old_text: "foo\nbar".into(),
            new_text: "one\ntwo".into(),
        }];
        let result = apply_edits_to_normalized_content("foo  \nbar\n", &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "one\ntwo\n");
        let notice = result
            .notice
            .as_deref()
            .expect("lenient match must be reported");
        assert!(notice.contains("lenient"), "{notice}");
    }

    #[test]
    fn old_text_trailing_spaces_do_not_match_clean_line_silently() {
        // oldText with intentional trailing spaces only matches lines that
        // actually have them; the fallback is the reported lenient mode.
        let edits = vec![Edit {
            old_text: "foo  ".into(),
            new_text: "bar".into(),
        }];
        let result = apply_edits_to_normalized_content("foo\n", &edits, "file.txt").unwrap();
        assert!(result.notice.is_some());
        assert_eq!(result.new_content, "bar\n");
    }

    #[test]
    fn trailing_whitespace_distinguishes_otherwise_identical_lines() {
        // "same  " matches only the line that has the trailing spaces, so
        // the replacement is unambiguous.
        let edits = vec![Edit {
            old_text: "same  ".into(),
            new_text: "changed".into(),
        }];
        let result =
            apply_edits_to_normalized_content("same  \nsame\n", &edits, "file.txt").unwrap();
        assert_eq!(result.new_content, "changed\nsame\n");
        assert!(result.notice.is_none());
    }

    #[test]
    fn preserves_untouched_lines_across_multiline_replacement() {
        let original = "first  \nsecond\nthird  \nfourth\n";
        let base = "first\nsecond\nthird\nfourth\n";
        let replacements = vec![MatchedEdit {
            edit_index: 0,
            match_index: 0,
            match_length: "first\nsecond".len(),
            new_text: "replaced".into(),
        }];
        let output =
            apply_replacements_preserving_unchanged_lines(original, base, &replacements).unwrap();
        assert_eq!(output, "replaced\nthird  \nfourth\n");
    }

    proptest! {
        /// The fuzzy rebuild pass must leave untouched lines byte-identical
        /// to the original and apply every replacement exactly once, even
        /// when replacements span multiple lines.
        #[test]
        fn fuzzy_rebuild_preserves_untouched_lines_and_applies_every_replacement(
            specs in prop::collection::vec(
                ("[a-z]{1,6}", any::<bool>(), any::<bool>(), any::<bool>()),
                2..6,
            ),
        ) {
            let mut base_lines = Vec::with_capacity(specs.len());
            let mut original_lines = Vec::with_capacity(specs.len());
            for (index, (word, trailing, _, _)) in specs.iter().enumerate() {
                // Prefix each line with its index so line content is unique;
                // otherwise identical words make "appears exactly once"
                // assertions ambiguous.
                let line = format!("{index}_{word}");
                base_lines.push(line.clone());
                original_lines.push(format!(
                    "{line}{}",
                    if *trailing { "  " } else { "" }
                ));
            }
            let base_content = format!("{}\n", base_lines.join("\n"));
            let original_content = format!("{}\n", original_lines.join("\n"));
            let spans = line_spans(&base_content);

            // Build non-overlapping replacements.  Each lives on its own
            // line; an extension flag swallows the following lines up to the
            // next line that wants its own replacement.
            let mut replacements = Vec::new();
            let mut skip_until = 0usize;
            for (index, (_, _, has_replacement, extend)) in specs.iter().enumerate() {
                if index < skip_until || !has_replacement {
                    continue;
                }
                let line_start = spans[index].start;
                let line_len = spans[index].end - spans[index].start;
                let start = line_start + (index * 7) % (line_len - 1);
                let mut end = line_start + line_len;
                if *extend {
                    let next = (index + 1..specs.len()).find(|&j| specs[j].2);
                    match next {
                        Some(next) => {
                            end = spans[next].start;
                            skip_until = next;
                        }
                        None => {
                            end = base_content.len();
                            skip_until = specs.len();
                        }
                    }
                } else {
                    skip_until = index + 1;
                }
                replacements.push(MatchedEdit {
                    edit_index: index,
                    match_index: start,
                    match_length: end - start,
                    new_text: format!("newtext{index}"),
                });
            }

            let output = apply_replacements_preserving_unchanged_lines(
                &original_content,
                &base_content,
                &replacements,
            )
            .unwrap();

            // Every replacement's new text appears exactly once.
            for replacement in &replacements {
                prop_assert_eq!(
                    output.matches(&replacement.new_text).count(),
                    1,
                    "replacement {} should appear exactly once",
                    replacement.new_text
                );
            }

            // Lines whose base span no replacement intersects are untouched
            // and must appear byte-identical (exactly once) in the output.
            let original_lines = split_lines_with_endings(&original_content);
            for (index, original_line) in original_lines.iter().enumerate() {
                let touched = replacements.iter().any(|replacement| {
                    replacement.match_index < spans[index].end
                        && spans[index].start < replacement.match_index + replacement.match_length
                });
                if !touched {
                    prop_assert_eq!(
                        output.matches(original_line).count(),
                        1,
                        "untouched line {:?} should appear exactly once",
                        index
                    );
                }
            }
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
}
