//! Contract C — CommitBlock
//!
//! The model is asked to wrap its committed output inside a unique pair of
//! sentinel delimiters:
//!
//! ```text
//! <<<COMMIT:PLAN v1>>>
//! Step 1: …
//! Step 2: …
//! <<<END:PLAN>>>
//! ```
//!
//! Sentinels:
//! - Opening: `<<<COMMIT:{TYPE} v1>>>` (uppercase TYPE, literal " v1")
//! - Closing:  `<<<END:{TYPE}>>>`
//!
//! The sentinels are unusual enough that a small model won't produce them by
//! accident, and distinctive enough that the parser can find them even inside a
//! longer stream.
//!
//! # `must_be_last` enforcement
//!
//! When `must_be_last = true` (the default), the parser additionally checks that
//! there is no non-whitespace content after the closing sentinel.  If there is,
//! the parse fails so a retry can request a clean block.
//!
//! # Retry strategy
//!
//! - Attempt 1: sentinels shown in system prompt with an example.
//! - Attempt 2: stronger reminder with a filled-in template.
//! - Attempt 3+: one-line demand with the exact sentinel format.

/// Build the system-prompt fragment for a commit block contract.
pub fn build_system_instruction(block_type: &str, must_be_last: bool) -> String {
    let upper = block_type.to_uppercase();
    let last_note = if must_be_last {
        "  After the closing sentinel there must be NO additional text."
    } else {
        ""
    };

    format!(
        "\n\nIMPORTANT — OUTPUT FORMAT:\n\
         You may reason freely, but you MUST finish your response with a commit \
         block using EXACTLY these delimiters:\n\
         <<<COMMIT:{upper} v1>>>\n\
         … your content here …\n\
         <<<END:{upper}>>>{last_note}\n\
         Do not alter the sentinel format."
    )
}

/// Build the correction/tightening user message used on retry attempt `n`
/// (1-indexed).
pub fn build_tightening_message(attempt: u32, block_type: &str) -> String {
    let upper = block_type.to_uppercase();
    match attempt {
        1 => format!(
            "Your previous response did not contain a valid commit block. \
             Please respond again and wrap your answer between these exact sentinels \
             (no modifications):\n\
             <<<COMMIT:{upper} v1>>>\n\
             your content\n\
             <<<END:{upper}>>>"
        ),
        _ => format!(
            "STOP. Output ONLY the commit block — nothing before or after it:\n\
             <<<COMMIT:{upper} v1>>>\n\
             your content\n\
             <<<END:{upper}>>>"
        ),
    }
}

/// Parse the sentinel-delimited commit block from the model's raw text.
///
/// Returns `Some(block_text)` — the trimmed content between the sentinels — or
/// `None` if the sentinels are absent, malformed, or (when `must_be_last = true`)
/// followed by non-whitespace content.
pub fn parse(text: &str, block_type: &str, must_be_last: bool) -> Option<String> {
    let upper = block_type.to_uppercase();
    let open_sentinel  = format!("<<<COMMIT:{upper} v1>>>");
    let close_sentinel = format!("<<<END:{upper}>>>");

    // Case-insensitive search: lower everything for positioning, value from original.
    let text_lower  = text.to_lowercase();
    let open_lower  = open_sentinel.to_lowercase();
    let close_lower = close_sentinel.to_lowercase();

    let open_pos = text_lower.find(&open_lower)?;
    let content_start = open_pos + open_sentinel.len();

    let close_pos = text_lower[content_start..].find(&close_lower)?;
    let abs_close_pos  = content_start + close_pos;
    let abs_close_end  = abs_close_pos + close_sentinel.len();

    // Enforce must_be_last: nothing non-whitespace after the closing sentinel.
    if must_be_last {
        let after = &text[abs_close_end..];
        if after.chars().any(|c| !c.is_whitespace()) {
            return None;
        }
    }

    let block_text = text[content_start..abs_close_pos].trim().to_string();
    if block_text.is_empty() {
        return None;
    }

    Some(block_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_parse() {
        let text = "Sure! Here is my plan:\n<<<COMMIT:PLAN v1>>>\nStep 1\nStep 2\n<<<END:PLAN>>>";
        assert_eq!(
            parse(text, "plan", true),
            Some("Step 1\nStep 2".to_string())
        );
    }

    #[test]
    fn must_be_last_rejects_trailing_text() {
        let text = "<<<COMMIT:PLAN v1>>>\nStep 1\n<<<END:PLAN>>>\nextra stuff";
        assert_eq!(parse(text, "plan", true), None);
    }

    #[test]
    fn must_be_last_false_allows_trailing_text() {
        let text = "<<<COMMIT:PLAN v1>>>\nStep 1\n<<<END:PLAN>>>\nextra stuff";
        assert_eq!(parse(text, "plan", false), Some("Step 1".to_string()));
    }

    #[test]
    fn trailing_whitespace_only_is_ok_with_must_be_last() {
        let text = "<<<COMMIT:SUMMARY v1>>>\nContent here\n<<<END:SUMMARY>>>\n   \n";
        assert_eq!(
            parse(text, "summary", true),
            Some("Content here".to_string())
        );
    }

    #[test]
    fn missing_sentinels_returns_none() {
        assert_eq!(parse("just some text", "plan", true), None);
    }

    #[test]
    fn empty_block_returns_none() {
        let text = "<<<COMMIT:PLAN v1>>>\n   \n<<<END:PLAN>>>";
        assert_eq!(parse(text, "plan", true), None);
    }

    #[test]
    fn case_insensitive_sentinel_matching() {
        // Sentinels in mixed case (unusual but tolerated).
        let text = "<<<commit:plan v1>>>\nmy block\n<<<end:plan>>>";
        assert_eq!(parse(text, "plan", true), Some("my block".to_string()));
    }
}
