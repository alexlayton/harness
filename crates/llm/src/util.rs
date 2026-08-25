use serde_json::Value;

/// Parse a JSON number into a `u64`, accepting plain integers, negative
/// integers that fit in `u64`, and numeric strings (some providers serialise
/// token counts as strings).
pub fn as_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

/// Truncate `value` to at most `max_bytes` UTF-8 bytes without splitting a
/// multi-byte character, appending an ellipsis when truncation occurs.
/// A `max_bytes` too small to hold the ellipsis yields the empty string.
/// This is the shared implementation of the per-crate `cap_utf8` helpers;
/// keep behaviour changes here so every crate observes them.
pub fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let suffix = "…";
    let mut end = max_bytes.saturating_sub(suffix.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{suffix}", &value[..end])
}

/// Truncate `value` to at most `max_bytes` UTF-8 bytes without splitting a
/// multi-byte character.  Unlike [`truncate_utf8`] the result is the raw
/// prefix with no ellipsis and is borrowed from the input; callers that need
/// the exact prefix (e.g. to append their own notice) should use this.
pub fn truncate_utf8_prefix(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_utf8_returns_input_when_within_budget() {
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello", 5), "hello");
        assert_eq!(truncate_utf8_prefix("hello", 10), "hello");
        assert_eq!(truncate_utf8_prefix("hello", 5), "hello");
    }

    #[test]
    fn truncate_utf8_never_splits_a_multibyte_character() {
        // "é" is two bytes; neither variant may split one.
        let value = "aéaéaé";
        assert_eq!(truncate_utf8(value, 7), "aéa…");
        assert_eq!(truncate_utf8_prefix(value, 5), "aéa");
        assert!(truncate_utf8(value, 3).ends_with('…'));
        assert!(truncate_utf8_prefix(value, 3).is_char_boundary(3));
    }

    #[test]
    fn truncate_utf8_zero_budget_is_empty() {
        assert_eq!(truncate_utf8("hello", 0), "");
        assert_eq!(truncate_utf8_prefix("hello", 0), "");
        // A single-byte ellipsis budget yields just the ellipsis.
        assert_eq!(truncate_utf8("hello", 1), "…");
    }
}
