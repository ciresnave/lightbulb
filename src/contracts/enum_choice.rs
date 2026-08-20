//! Contract A — EnumChoice
//!
//! The most reliable structured-output contract for small LLMs.
//! The model is instructed to end its reply with **exactly one choice** from a
//! fixed list — nothing else on that final line.
//!
//! # Behaviour flags
//!
//! - `case_sensitive` — if `false` (default), matching ignores case.
//! - `allow_index`    — if `true` (default), the model may also output the
//!   1-based numeric index of a choice (e.g. `"1"` for `choices[0]`).
//!
//! # Retry strategy
//!
//! Each retry tightens the instruction progressively:
//! - Attempt 1: polite instruction appended to system prompt.
//! - Attempt 2: firmer reminder added as a fresh user message.
//! - Attempt 3+: minimal one-word demand.

/// Build the system-prompt fragment that teaches the model how to respond.
///
/// This text should be *appended* to the existing system message (or used as
/// the system message if none exists).
pub fn build_system_instruction(
    choices: &[String],
    _case_sensitive: bool,
    allow_index: bool,
) -> String {
    let choice_list = choices
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if allow_index {
                format!("{}. {}", i + 1, c.to_uppercase())
            } else {
                c.to_uppercase()
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");

    let index_note = if allow_index {
        " You may also output just the number (e.g. \"1\")."
    } else {
        ""
    };

    format!(
        "\n\nIMPORTANT — OUTPUT FORMAT:\n\
         You may reason briefly, but your FINAL LINE must contain EXACTLY ONE of \
         the following — nothing else on that line, no punctuation, no extra words:\n\
         {choice_list}{index_note}"
    )
}

/// Build the correction/tightening user message used on retry attempt `n`
/// (1-indexed; attempt 1 means the second inference call).
pub fn build_tightening_message(attempt: u32, choices: &[String], allow_index: bool) -> String {
    let choice_list = choices
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if allow_index {
                format!("{} ({})", c.to_uppercase(), i + 1)
            } else {
                c.to_uppercase()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    match attempt {
        1 => format!(
            "Your previous response did not end with exactly one of the required choices. \
             Please respond again — the VERY LAST LINE must contain ONLY one of: {choice_list}"
        ),
        _ => format!(
            "STOP. Output ONLY the choice — one word or number, nothing else. \
             Valid values: {choice_list}"
        ),
    }
}

/// Parse the model's raw text and return the matched choice (in the original
/// casing from `choices`), or `None` if nothing matches.
///
/// Strategy (most-to-least permissive):
/// 1. Scan lines in **reverse** (last line first, since instructions say to put
///    the choice at the end).
/// 2. For each line: strip leading/trailing punctuation and whitespace, then:
///    a. Exact whole-line match against every choice (respecting `case_sensitive`).
///    b. 1-based numeric index match (when `allow_index = true`).
///    c. Word-boundary match as a fall-through.
/// 3. If no line matches, do a full-text scan as a final fallback.
pub fn parse(
    text: &str,
    choices: &[String],
    case_sensitive: bool,
    allow_index: bool,
) -> Option<String> {
    // Pre-compute normalised versions of each choice for comparison.
    let normed: Vec<(String, &String)> = choices
        .iter()
        .map(|c| {
            let norm = if case_sensitive {
                c.clone()
            } else {
                c.to_lowercase()
            };
            (norm, c)
        })
        .collect();

    let normalise = |s: &str| -> String {
        if case_sensitive {
            s.to_string()
        } else {
            s.to_lowercase()
        }
    };

    // Phase 1 — reverse-line scan.
    for line in text.lines().rev() {
        let cleaned = line
            .trim()
            .trim_matches(|c: char| c.is_ascii_punctuation() || c == '"' || c == '\'');
        let norm_cleaned = normalise(cleaned);

        // 1a. Exact whole-line match.
        for (norm_choice, original) in &normed {
            if norm_cleaned == *norm_choice {
                return Some((*original).clone());
            }
        }

        // 1b. Numeric index ("1" → choices[0]).
        if allow_index {
            if let Ok(idx) = cleaned.trim().parse::<usize>() {
                if idx >= 1 && idx <= choices.len() {
                    return Some(choices[idx - 1].clone());
                }
            }
        }

        // 1c. Word-boundary match.
        for (norm_choice, original) in &normed {
            if contains_word(&norm_cleaned, norm_choice) {
                return Some((*original).clone());
            }
        }
    }

    // Phase 2 — full-text fallback.
    let full_norm = normalise(text);
    for (norm_choice, original) in &normed {
        if contains_word(&full_norm, norm_choice) {
            return Some((*original).clone());
        }
    }

    None
}

/// Returns `true` if `needle` appears in `haystack` as a whole word (bounded
/// by whitespace, punctuation, or string start/end).
fn contains_word(haystack: &str, needle: &str) -> bool {
    let mut start = 0;
    while start + needle.len() <= haystack.len() {
        if let Some(pos) = haystack[start..].find(needle) {
            let abs = start + pos;
            let before_ok = abs == 0
                || haystack
                    .as_bytes()
                    .get(abs - 1)
                    .map_or(true, |b| !b.is_ascii_alphanumeric() && *b != b'_');
            let after_ok = abs + needle.len() >= haystack.len()
                || haystack
                    .as_bytes()
                    .get(abs + needle.len())
                    .map_or(true, |b| !b.is_ascii_alphanumeric() && *b != b'_');

            if before_ok && after_ok {
                return true;
            }
            start = abs + 1;
        } else {
            break;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choices() -> Vec<String> {
        vec!["yes".to_string(), "no".to_string(), "maybe".to_string()]
    }

    #[test]
    fn exact_last_line() {
        assert_eq!(
            parse("Let me think...\nYES", &choices(), false, true),
            Some("yes".to_string())
        );
    }

    #[test]
    fn case_insensitive_default() {
        assert_eq!(
            parse("No.", &choices(), false, true),
            Some("no".to_string())
        );
    }

    #[test]
    fn case_sensitive_requires_exact_case() {
        let caps = vec!["YES".to_string(), "NO".to_string()];
        assert_eq!(parse("yes", &caps, true, false), None);
        assert_eq!(parse("YES", &caps, true, false), Some("YES".to_string()));
    }

    #[test]
    fn allow_index_matches_number() {
        // "2" should resolve to choices[1] = "no"
        assert_eq!(
            parse("My answer:\n2", &choices(), false, true),
            Some("no".to_string())
        );
    }

    #[test]
    fn allow_index_false_ignores_number() {
        assert_eq!(parse("2", &choices(), false, false), None);
    }

    #[test]
    fn inline_word() {
        assert_eq!(
            parse(
                "After analysis my answer is maybe.",
                &choices(),
                false,
                true
            ),
            Some("maybe".to_string())
        );
    }

    #[test]
    fn no_match() {
        assert_eq!(parse("I don't know.", &choices(), false, true), None);
    }

    #[test]
    fn prefers_last_line() {
        // Both "yes" and "no" appear; last line wins.
        assert_eq!(
            parse(
                "First I thought yes.\nFinal answer:\nno",
                &choices(),
                false,
                true
            ),
            Some("no".to_string())
        );
    }

    #[test]
    fn word_boundary_respected() {
        // "nobody" should NOT match "no".
        assert_eq!(parse("nobody came", &choices(), false, true), None);
    }
}
