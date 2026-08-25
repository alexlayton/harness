/// Maximum tool-name length accepted by all currently supported providers.
const MAX_NAME_BYTES: usize = 64;

/// Convert an untrusted server/tool pair into a deterministic provider-safe
/// tool name. The hash retains distinction after lossy ASCII sanitization.
pub fn normalized_tool_name(server: &str, tool: &str) -> String {
    let hash = short_hash(&format!("{server}\0{tool}"));
    let prefix = "mcp__";
    let suffix = format!("__{hash:08x}");
    let available = MAX_NAME_BYTES.saturating_sub(prefix.len() + suffix.len());
    let server = sanitize(server);
    let tool = sanitize(tool);
    let half = available / 2;
    let server = truncate_ascii(&server, half.max(1));
    let tool = truncate_ascii(&tool, available.saturating_sub(server.len() + 2).max(1));
    format!("{prefix}{server}__{tool}{suffix}")
}

fn sanitize(value: &str) -> String {
    let mut output = String::new();
    let mut previous_separator = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            output.push(character);
            previous_separator = false;
        } else if !previous_separator {
            output.push('_');
            previous_separator = true;
        }
    }
    let output = output.trim_matches('_');
    if output.is_empty() {
        "tool".to_owned()
    } else {
        output.to_owned()
    }
}

fn truncate_ascii(value: &str, maximum: usize) -> String {
    value.chars().take(maximum).collect()
}

fn short_hash(value: &str) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in value.bytes() {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_are_safe_stable_and_distinct_after_sanitizing() {
        let first = normalized_tool_name("a/b", "x y");
        assert_eq!(first, normalized_tool_name("a/b", "x y"));
        assert_ne!(first, normalized_tool_name("a:b", "x y"));
        assert!(
            first
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        );
        assert!(first.len() <= MAX_NAME_BYTES);
    }
}
