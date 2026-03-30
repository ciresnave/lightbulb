//! Contract B — TaggedFields
//!
//! The second most reliable structured-output contract.  The model is asked to
//! respond using simple XML-like tags:
//!
//! ```text
//! <MODE>typecheck_error</MODE>
//! <CONF>0.74</CONF>
//! <EVIDENCE>The compiler says \"expected type...\"</EVIDENCE>
//! ```
//!
//! Tag names are matched **case-insensitively**.  Values are preserved in their
//! original casing.  Repeatable tags accumulate multiple values; all other tags
//! keep only the **last** occurrence (models sometimes self-correct inline).
//!
//! # Retry strategy
//!
//! - Attempt 1: tags listed in system prompt with an example.
//! - Attempt 2: reminder with the raw template filled in.
//! - Attempt 3+: strict \"reproduce this exact template\" demand.

use std::collections::HashMap;

/// Build the system-prompt fragment that teaches the model to use tagged fields.
pub fn build_system_instruction(
    allowed_tags: &[String],
    required_tags: &[String],
    _repeatable_tags: &[String],
) -> String {
    let example = allowed_tags
        .iter()
        .map(|f| {
            let upper = f.to_uppercase();
            format!("<{upper}>your {f} here</{upper}>")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let required_note = if required_tags.is_empty() {
        String::new()
    } else {
        let req_list = required_tags
            .iter()
            .map(|t| t.to_uppercase())
            .collect::<Vec<_>>()
            .join(", ");
        format!("  The following tags are REQUIRED: {req_list}.\n")
    };

    format!(
        "\n\nIMPORTANT — OUTPUT FORMAT:\n\
         After your reasoning, output each field using XML-like tags, one per line.\n\
         {required_note}\
         {example}\n\
         Place the tags at the end of your response."
    )
}

/// Build the correction/tightening user message used on retry attempt `n`
/// (1-indexed).
pub fn build_tightening_message(
    attempt: u32,
    allowed_tags: &[String],
    required_tags: &[String],
) -> String {
    let template = allowed_tags
        .iter()
        .map(|f| {
            let upper = f.to_uppercase();
            format!("<{upper}></{upper}>")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let req_note = if required_tags.is_empty() {
        String::new()
    } else {
        let req_list = required_tags
            .iter()
            .map(|t| t.to_uppercase())
            .collect::<Vec<_>>()
            .join(", ");
        format!(" (REQUIRED: {req_list})")
    };

    match attempt {
        1 => format!(
            "Your previous response did not include the required tagged fields{req_note}. \
             Please try again and include ALL required tags filled in:\n{template}"
        ),
        _ => format!(
            "STOP.  Copy-paste this template and fill in each tag — output NOTHING else:\n{template}"
        ),
    }
}

/// Parse `<TAG>value</TAG>` blocks from the model's raw text.
///
/// - Tag names are matched **case-insensitively**.
/// - Leading/trailing whitespace inside values is trimmed.
/// - Tags in `repeatable_tags` accumulate all occurrences (in order).
/// - All other tags keep only the **last** occurrence.
/// - Tags not in `allowed_tags` are ignored.
///
/// Returns a map of `tag_name_as_given → Vec<value>`.  The map may be empty if
/// nothing could be parsed.
pub fn parse(
    text: &str,
    allowed_tags: &[String],
    repeatable_tags: &[String],
) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    let text_lower = text.to_lowercase();

    let repeatable_lower: Vec<String> =
        repeatable_tags.iter().map(|t| t.to_lowercase()).collect();

    for original_tag in allowed_tags {
        let tag_lower = original_tag.to_lowercase();
        let open_tag  = format!("<{tag_lower}>");
        let close_tag = format!("</{tag_lower}>");
        let is_repeatable = repeatable_lower.contains(&tag_lower);

        let mut search_from = 0;
        let mut occurrences: Vec<String> = Vec::new();

        while let Some(open_pos) = text_lower[search_from..].find(&open_tag) {
            let abs_open = search_from + open_pos + open_tag.len();
            if let Some(close_rel) = text_lower[abs_open..].find(&close_tag) {
                let abs_close = abs_open + close_rel;
                let value = text[abs_open..abs_close].trim();
                if !value.is_empty() {
                    occurrences.push(value.to_string());
                }
                search_from = abs_close + close_tag.len();
            } else {
                break;
            }
        }

        if occurrences.is_empty() {
            continue;
        }

        if is_repeatable {
            result.insert(original_tag.clone(), occurrences);
        } else {
            // Last occurrence wins for non-repeatable tags.
            result.insert(original_tag.clone(), vec![occurrences.into_iter().last().unwrap()]);
        }
    }

    result
}

/// Returns `true` if every tag in `required_tags` has at least one value in
/// `parsed`.
pub fn check_cardinality(
    parsed: &HashMap<String, Vec<String>>,
    required_tags: &[String],
) -> bool {
    required_tags
        .iter()
        .all(|t| parsed.get(t).map_or(false, |v| !v.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags() -> Vec<String> {
        vec!["MODE".to_string(), "CONF".to_string(), "EVIDENCE".to_string()]
    }

    #[test]
    fn basic_parse() {
        let text = "<MODE>typecheck_error</MODE>\n<CONF>0.74</CONF>\n<EVIDENCE>see line 42</EVIDENCE>";
        let result = parse(text, &tags(), &[]);
        assert_eq!(
            result.get("MODE").and_then(|v| v.first()).map(String::as_str),
            Some("typecheck_error")
        );
        assert_eq!(
            result.get("CONF").and_then(|v| v.first()).map(String::as_str),
            Some("0.74")
        );
        assert_eq!(
            result.get("EVIDENCE").and_then(|v| v.first()).map(String::as_str),
            Some("see line 42")
        );
    }

    #[test]
    fn case_insensitive_tags() {
        let text = "<mode>runtime_error</mode><conf>0.9</conf>";
        let result = parse(text, &tags(), &[]);
        assert_eq!(
            result.get("MODE").and_then(|v| v.first()).map(String::as_str),
            Some("runtime_error")
        );
        assert_eq!(
            result.get("CONF").and_then(|v| v.first()).map(String::as_str),
            Some("0.9")
        );
    }

    #[test]
    fn last_occurrence_wins_for_non_repeatable() {
        let text = "<MODE>first</MODE> ... <MODE>corrected</MODE>";
        let result = parse(text, &tags(), &[]);
        assert_eq!(
            result.get("MODE").and_then(|v| v.first()).map(String::as_str),
            Some("corrected")
        );
        assert_eq!(result.get("MODE").map(|v| v.len()), Some(1));
    }

    #[test]
    fn repeatable_collects_all() {
        let text = "<EVIDENCE>first clue</EVIDENCE>\n<EVIDENCE>second clue</EVIDENCE>";
        let result = parse(text, &tags(), &["EVIDENCE".to_string()]);
        let vals = result.get("EVIDENCE").expect("EVIDENCE must be present");
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], "first clue");
        assert_eq!(vals[1], "second clue");
    }

    #[test]
    fn missing_field_absent_from_map() {
        let text = "<MODE>lint_error</MODE>";
        let result = parse(text, &tags(), &[]);
        assert!(!result.contains_key("CONF"));
        assert!(!result.contains_key("EVIDENCE"));
    }

    #[test]
    fn check_cardinality_passes_when_required_present() {
        let text = "<MODE>x</MODE><CONF>0.5</CONF>";
        let result = parse(text, &tags(), &[]);
        assert!(check_cardinality(&result, &["MODE".to_string(), "CONF".to_string()]));
        assert!(!check_cardinality(
            &result,
            &["MODE".to_string(), "CONF".to_string(), "EVIDENCE".to_string()]
        ));
    }
}
