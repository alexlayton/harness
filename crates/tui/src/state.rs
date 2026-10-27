/// A complete tool result retained by the UI for collapsed and expanded
/// rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolRecord {
    pub name: String,
    pub summary: String,
    pub ok: bool,
    pub duration_ms: u64,
    /// Newest part of the output, at most [`MAX_RETAINED_TOOL_OUTPUT_BYTES`].
    pub output: String,
    /// Lines dropped from the front of `output` to respect the byte bound.
    /// The expanded view counts them in its `… N lines above` row.
    pub omitted_lines: usize,
    /// First part of the error, at most [`MAX_RETAINED_TOOL_ERROR_BYTES`].
    pub error: Option<String>,
    pub status: ToolStatus,
}

/// Retained bytes of one tool's output. The UI shows only the newest output
/// lines (and a count of earlier ones), so a large result keeps its tail and
/// a count instead of every byte for the rest of the session.
pub const MAX_RETAINED_TOOL_OUTPUT_BYTES: usize = 16 * 1024;

/// Retained bytes of one tool's error. The UI shows its first lines.
pub const MAX_RETAINED_TOOL_ERROR_BYTES: usize = 16 * 1024;

impl ToolRecord {
    /// A finished tool result with output and error reduced to the parts the
    /// UI can display.
    pub fn finished(
        name: String,
        summary: String,
        ok: bool,
        duration_ms: u64,
        output: String,
        error: Option<String>,
    ) -> Self {
        let (output, omitted_lines) = retain_output_tail(output);
        Self {
            name,
            summary,
            ok,
            duration_ms,
            output,
            omitted_lines,
            error: error.map(retain_error_head),
            status: if ok {
                ToolStatus::Success
            } else {
                ToolStatus::Failure
            },
        }
    }
}

/// Keep the last [`MAX_RETAINED_TOOL_OUTPUT_BYTES`] of `output`, starting at
/// a line boundary when one exists, and count the dropped lines.
fn retain_output_tail(output: String) -> (String, usize) {
    if output.len() <= MAX_RETAINED_TOOL_OUTPUT_BYTES {
        return (output, 0);
    }
    let mut start = output.len() - MAX_RETAINED_TOOL_OUTPUT_BYTES;
    while !output.is_char_boundary(start) {
        start += 1;
    }
    if let Some(newline) = output[start..].find('\n')
        && start + newline + 1 < output.len()
    {
        start += newline + 1;
    }
    let omitted = output.as_bytes()[..start]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count();
    (output[start..].to_owned(), omitted)
}

fn retain_error_head(mut error: String) -> String {
    if error.len() > MAX_RETAINED_TOOL_ERROR_BYTES {
        let mut end = MAX_RETAINED_TOOL_ERROR_BYTES;
        while !error.is_char_boundary(end) {
            end -= 1;
        }
        error.truncate(end);
        error.push('…');
    }
    error
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Success,
    Failure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_tool_output_keeps_its_tail_and_counts_dropped_lines() {
        let output = (0..100_000)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        let record = ToolRecord::finished(
            "bash".into(),
            "run".into(),
            true,
            1,
            output.clone(),
            Some("e".repeat(MAX_RETAINED_TOOL_ERROR_BYTES * 4)),
        );
        assert!(record.output.len() <= MAX_RETAINED_TOOL_OUTPUT_BYTES);
        assert!(output.ends_with(&record.output));
        assert!(record.output.starts_with("line "));
        assert_eq!(
            record.omitted_lines + record.output.lines().count(),
            100_000
        );
        assert!(record.error.unwrap().len() <= MAX_RETAINED_TOOL_ERROR_BYTES + '…'.len_utf8());
    }
}
