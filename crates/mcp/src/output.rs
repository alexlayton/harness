use llm::util::truncate_utf8;
use rmcp::model::{CallToolResult, ContentBlock, ResourceContents};

pub(crate) const MAX_OUTPUT_BYTES: usize = 20 * 1024;

/// Flatten rich MCP content deterministically because Harness provider history
/// currently stores tool results as text only.
pub(crate) fn flatten(result: &CallToolResult) -> String {
    let mut blocks = result.content.iter().map(flatten_block).collect::<Vec<_>>();
    if let Some(structured) = &result.structured_content {
        blocks.push(format!(
            "Structured result:\n{}",
            serde_json::to_string_pretty(structured)
                .unwrap_or_else(|_| "<unserializable JSON>".into())
        ));
    }
    let content = if blocks.is_empty() {
        "MCP tool returned no content.".to_owned()
    } else {
        blocks.join("\n\n---\n\n")
    };
    if content.len() <= MAX_OUTPUT_BYTES {
        return content;
    }
    let mut truncated = truncate_utf8(&content, MAX_OUTPUT_BYTES.saturating_sub(32));
    truncated.push_str("\n[output truncated by Harness]");
    truncated
}

fn flatten_block(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text(text) => text.text.clone(),
        ContentBlock::Image(image) => format!(
            "[image omitted: {} base64 bytes, {}]",
            image.data.len(),
            image.mime_type
        ),
        ContentBlock::Audio(audio) => format!(
            "[audio omitted: {} base64 bytes, {}]",
            audio.data.len(),
            audio.mime_type
        ),
        ContentBlock::Resource(resource) => match &resource.resource {
            ResourceContents::TextResourceContents { uri, text, .. } => {
                format!("Resource {uri}:\n{text}")
            }
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => format!(
                "[binary resource omitted: {uri}, {} base64 bytes, {}]",
                blob.len(),
                mime_type.as_deref().unwrap_or("unknown MIME type")
            ),
            _ => "[unsupported resource omitted]".into(),
        },
        ContentBlock::ResourceLink(resource) => {
            format!("Resource link: {} ({})", resource.uri, resource.name)
        }
        _ => "[unsupported MCP content omitted]".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_results_are_explicit() {
        assert_eq!(
            flatten(&CallToolResult::default()),
            "MCP tool returned no content."
        );
    }
}
