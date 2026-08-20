//! Token estimation for cut-point selection.
//!
//! Precision is deliberately irrelevant here: the estimator only *selects a
//! cut point with slack*, so a ~4 bytes/token heuristic is more than enough.
//! The compaction *trigger* uses exact provider-reported numbers
//! (`usage.input_tokens` + `output_tokens`), never these estimates.

/// Rough bytes per token. UTF-8 safe in the sense that we count bytes, never
/// split characters.
pub const BYTES_PER_TOKEN: u64 = 4;

/// Estimate the token count of a byte payload.
pub fn estimate_tokens(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(BYTES_PER_TOKEN)
}

/// Estimate the token count of a text payload.
pub fn estimate_text_tokens(text: &str) -> u64 {
    estimate_tokens(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_ceil_of_bytes_over_four() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(4), 1);
        assert_eq!(estimate_tokens(5), 2);
        assert_eq!(estimate_tokens(100), 25);
    }

    #[test]
    fn multibyte_utf8_counts_bytes_not_chars() {
        // "é" is two bytes; a 4-char string of them is 8 bytes → 2 tokens.
        assert_eq!(estimate_text_tokens("éééé"), 2);
    }
}
