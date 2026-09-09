use rmcp::model::{CallToolResult, ContentBlock, ResourceContents};
use std::fmt;
use std::io::{self, Write as IoWrite};

pub(crate) const MAX_OUTPUT_BYTES: usize = 20 * 1024;
const OUTPUT_TRUNCATION_NOTICE: &str = "\n[output truncated by Harness]";

/// Byte-budgeted output sink used for both ordinary MCP content and structured
/// JSON. Once the budget is exhausted later blocks are discarded rather than
/// being materialized only to truncate them afterward.
struct OutputWriter {
    output: Vec<u8>,
    truncated: bool,
}

impl OutputWriter {
    fn new() -> Self {
        Self {
            output: Vec::new(),
            truncated: false,
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        if self.truncated {
            return;
        }
        let content_limit = MAX_OUTPUT_BYTES.saturating_sub(OUTPUT_TRUNCATION_NOTICE.len());
        if self.output.len().saturating_add(bytes.len()) <= content_limit {
            self.output.extend_from_slice(bytes);
            return;
        }
        let available = content_limit.saturating_sub(self.output.len());
        self.output
            .extend_from_slice(&bytes[..utf8_prefix_len(bytes, available)]);
        self.truncated = true;
    }

    fn push_str(&mut self, value: &str) {
        self.push_bytes(value.as_bytes());
    }

    fn finish(mut self) -> String {
        if self.truncated {
            let content_limit = MAX_OUTPUT_BYTES.saturating_sub(OUTPUT_TRUNCATION_NOTICE.len());
            self.output.truncate(content_limit);
            while std::str::from_utf8(&self.output).is_err() {
                self.output.pop();
            }
            self.output
                .extend_from_slice(OUTPUT_TRUNCATION_NOTICE.as_bytes());
        }
        String::from_utf8(self.output).unwrap_or_else(|_| "<invalid MCP output>".into())
    }
}

impl fmt::Write for OutputWriter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.push_str(value);
        Ok(())
    }
}

impl IoWrite for OutputWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.push_bytes(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn utf8_prefix_len(bytes: &[u8], max_bytes: usize) -> usize {
    let length = bytes.len().min(max_bytes);
    let mut end = length;
    while end > 0 && std::str::from_utf8(&bytes[..end]).is_err() {
        end -= 1;
    }
    end
}

/// Flatten rich MCP content deterministically because Harness provider history
/// currently stores tool results as text only. Every intermediate write is
/// subject to [`MAX_OUTPUT_BYTES`].
pub(crate) fn flatten(result: &CallToolResult) -> String {
    let mut writer = OutputWriter::new();
    if result.content.is_empty() && result.structured_content.is_none() {
        writer.push_str("MCP tool returned no content.");
        return writer.finish();
    }

    for (index, block) in result.content.iter().enumerate() {
        if index > 0 {
            writer.push_str("\n\n---\n\n");
        }
        flatten_block(&mut writer, block);
    }
    if let Some(structured) = &result.structured_content
        && !structured_duplicates_text(result, structured)
    {
        if !result.content.is_empty() {
            writer.push_str("\n\n---\n\n");
        }
        writer.push_str("Structured result:\n");
        let _ = serde_json::to_writer(&mut writer, structured);
    }
    writer.finish()
}

fn structured_duplicates_text(result: &CallToolResult, structured: &serde_json::Value) -> bool {
    if result.content.len() != 1 {
        return false;
    }
    let ContentBlock::Text(text) = &result.content[0] else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(text.text.trim())
        .is_ok_and(|value| value == *structured)
}

fn flatten_block(writer: &mut OutputWriter, block: &ContentBlock) {
    match block {
        ContentBlock::Text(text) => writer.push_str(&text.text),
        ContentBlock::Image(image) => {
            let _ = write!(
                writer,
                "[image omitted: {} base64 bytes, {}]",
                image.data.len(),
                image.mime_type
            );
        }
        ContentBlock::Audio(audio) => {
            let _ = write!(
                writer,
                "[audio omitted: {} base64 bytes, {}]",
                audio.data.len(),
                audio.mime_type
            );
        }
        ContentBlock::Resource(resource) => match &resource.resource {
            ResourceContents::TextResourceContents { uri, text, .. } => {
                let _ = writeln!(writer, "Resource {uri}:");
                writer.push_str(text);
            }
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => {
                let _ = write!(
                    writer,
                    "[binary resource omitted: {uri}, {} base64 bytes, {}]",
                    blob.len(),
                    mime_type.as_deref().unwrap_or("unknown MIME type")
                );
            }
            _ => writer.push_str("[unsupported resource omitted]"),
        },
        ContentBlock::ResourceLink(resource) => {
            let _ = write!(
                writer,
                "Resource link: {} ({})",
                resource.uri, resource.name
            );
        }
        _ => writer.push_str("[unsupported MCP content omitted]"),
    }
}

/// Cap arbitrary diagnostic text using the same byte budget as tool output.
pub(crate) fn cap_display(value: impl fmt::Display) -> String {
    let mut writer = OutputWriter::new();
    let _ = write!(writer, "{value}");
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_results_are_explicit() {
        assert_eq!(
            flatten(&CallToolResult::default()),
            "MCP tool returned no content."
        );
    }

    #[test]
    fn structured_results_use_compact_json() {
        let mut result = CallToolResult::default();
        result.structured_content = Some(json!({
            "nested": {"answer": 42},
            "items": [1, 2, 3]
        }));
        let output = flatten(&result);
        assert!(output.contains("\"nested\":{"));
        assert!(output.contains("\"answer\":42"));
        assert!(!output.contains("\n  "));
    }

    #[test]
    fn huge_text_and_many_blocks_are_bounded() {
        let result = CallToolResult::success(
            (0..100)
                .map(|_| ContentBlock::text("x".repeat(2_000)))
                .collect(),
        );
        let output = flatten(&result);
        assert!(output.len() <= MAX_OUTPUT_BYTES);
        assert!(output.ends_with(OUTPUT_TRUNCATION_NOTICE));
    }

    #[test]
    fn huge_structured_json_is_bounded() {
        let mut result = CallToolResult::default();
        result.structured_content = Some(json!({
            "payload": "x".repeat(MAX_OUTPUT_BYTES * 2)
        }));
        assert!(flatten(&result).len() <= MAX_OUTPUT_BYTES);
    }

    #[test]
    fn multibyte_output_stays_valid_at_the_boundary() {
        let result =
            CallToolResult::success(vec![ContentBlock::text("é".repeat(MAX_OUTPUT_BYTES))]);
        let output = flatten(&result);
        assert!(output.len() <= MAX_OUTPUT_BYTES);
        assert!(std::str::from_utf8(output.as_bytes()).is_ok());
        assert!(output.ends_with(OUTPUT_TRUNCATION_NOTICE));
    }

    #[test]
    fn prefix_helper_never_splits_utf8() {
        assert_eq!(llm::util::truncate_utf8_prefix("éé", 1), "");
        assert_eq!(llm::util::truncate_utf8_prefix("éé", 2), "é");
    }
}
