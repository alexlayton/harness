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
