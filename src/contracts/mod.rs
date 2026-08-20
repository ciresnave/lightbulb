//! Output Contracts — Reliable structured output from small/weak LLMs.
//!
//! # Problem
//!
//! Small LLMs can't emit strict JSON reliably, but they *can* output a single
//! word from a list or a few XML-like tags if the prompt is sufficiently directed.
//!
//! # Contract types (A → D, roughly ascending complexity)
//!
//! - **`EnumChoice`** — model picks exactly one label from a fixed list.
//!   Most reliable; should be tried first.
//! - **`TaggedFields`** — model fills named `<TAG>…</TAG>` blocks.
//!   Second-most reliable; survives formatting slop well.
//! - **`Json`** — strict JSON (reserved for constrained decoding / strong models).
//!   Not yet enabled.
//!
//! Each contract type lives in its own sub-module and exposes three things:
//!   1. `build_system_instruction` — text injected into the system prompt.
//!   2. `build_tightening_message` — stricter re-prompt used on retry N.
//!   3. `parse` — forgiving parser; returns `None`/empty map on total failure
//!      so the retry loop can act.
//!
//! # Wire format (JSON, in `LightbulbExtensions`)
//!
//! ```json
//! {
//!   "lightbulb": {
//!     "output_contract": {
//!       "type": "enum_choice",
//!       "choices": ["yes", "no", "maybe"],
//!       "case_sensitive": false,
//!       "allow_index": true
//!     },
//!     "max_attempts": 3,
//!     "fallback_contracts": [
//!       { "type": "enum_choice", "choices": ["yes", "no"] }
//!     ]
//!   }
//! }
//! ```
//!
//! A successful response carries an extra top-level `lightbulb_result` field:
//!
//! ```json
//! {
//!   "lightbulb_result": {
//!     "contract_type": "enum_choice",
//!     "attempts": 1,
//!     "parse_success": true,
//!     "model_id": "my-model",
//!     "latency_ms": 142,
//!     "output": { "type": "enum_choice", "choice": "yes" }
//!   }
//! }
//! ```

pub mod commit_block;
pub mod enum_choice;
pub mod executor;
pub mod tagged_fields;
pub mod validation;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Request-side types ────────────────────────────────────────────────────

/// Used by serde to provide a default value of `true` for boolean fields.
fn default_true() -> bool {
    true
}

/// Specification for the desired output contract, supplied by the caller.
///
/// Serialized with an internal `type` tag so JSON looks like:
/// `{"type": "enum_choice", "choices": [...]}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContractSpec {
    /// Model must output **exactly one choice** from the provided list, nothing
    /// else.  Most reliable contract for small LLMs.
    EnumChoice {
        /// The allowed choices (e.g. `["yes", "no", "maybe"]`).
        choices: Vec<String>,
        /// Whether matching is case-sensitive.  Defaults to `false`.
        #[serde(default)]
        case_sensitive: bool,
        /// Whether the model may output the 1-based numeric index of a choice
        /// instead of the choice text (e.g. `"1"` for `choices[0]`).  Defaults
        /// to `true`.
        #[serde(default = "default_true")]
        allow_index: bool,
    },

    /// Model must fill named XML-like tags.
    TaggedFields {
        /// All recognized tag names (e.g. `["MODE", "CONF", "EVIDENCE"]`).
        allowed_tags: Vec<String>,
        /// Tags that **must** be present for the parse to be considered
        /// successful.  Defaults to empty (any populated tag counts).
        #[serde(default)]
        required_tags: Vec<String>,
        /// Tags that may appear **more than once**.  Their values are
        /// collected into a list; all others keep only the last occurrence.
        #[serde(default)]
        repeatable_tags: Vec<String>,
    },

    /// Model must emit a sentinel-delimited commit block:
    /// `<<<COMMIT:{TYPE} v1>>>…<<<END:{TYPE}>>>`.
    CommitBlock {
        /// The block type token embedded in the sentinels (e.g. `"plan"` →
        /// `<<<COMMIT:PLAN v1>>>…<<<END:PLAN>>>`).
        block_type: String,
        /// When `true` (default) the commit block must be the last
        /// non-whitespace content in the response.
        #[serde(default = "default_true")]
        must_be_last: bool,
    },

    /// Reserved — not yet active.
    Json,
}

impl OutputContractSpec {
    /// Human-readable name for logging.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::EnumChoice { .. } => "enum_choice",
            Self::TaggedFields { .. } => "tagged_fields",
            Self::CommitBlock { .. } => "commit_block",
            Self::Json => "json",
        }
    }
}

// ─── Response-side types ───────────────────────────────────────────────────

/// The structured output produced by a contract execution.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContractOutput {
    /// A single choice from the allowed set.
    EnumChoice { choice: String },

    /// Tag → value(s) map parsed from `<TAG>…</TAG>` blocks.  Repeatable
    /// tags accumulate multiple values; all others store exactly one.
    TaggedFields { tags: HashMap<String, Vec<String>> },

    /// The text extracted from a sentinel-delimited commit block.
    CommitBlock { block_text: String },

    /// Raw model text returned when every parse attempt failed.
    Raw { text: String },
}

/// Full result metadata returned alongside the completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ContractResult {
    /// Which contract type was used for the final successful (or fallback) parse.
    pub contract_type: String,

    /// How many inference calls were made (includes retries and fallbacks).
    pub attempts: u32,

    /// Whether the final attempt parsed successfully.
    pub parse_success: bool,

    /// Model identifier from the request.
    pub model_id: String,

    /// Wall-clock time (ms) from first inference call to final result.
    pub latency_ms: u64,

    /// The structured output — or `Raw` if every attempt failed.
    pub output: ContractOutput,
}
