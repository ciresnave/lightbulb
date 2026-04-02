//! Tool call detection and mid-reasoning context injection
//!
//! Detects tool calls in LLM output via configurable pattern matching,
//! pauses generation while preserving the KV cache state, and enables
//! injection of tool results back into the model's active reasoning.
//!
//! # How It Works
//!
//! ```text
//! Model generates: "Let me check the weather <tool_call>get_weather("Paris")</tool_call>"
//!                                             ↑ detected ↑
//!
//! 1. ToolCallDetector matches start+end markers in rolling token buffer
//! 2. Request transitions to AwaitingToolResult (KV cache preserved)
//! 3. Caller executes tool, gets result
//! 4. inject_tool_result() processes result as mini-prefill against preserved cache
//! 5. Model resumes generation with tool result in full attentional context
//! ```
//!
//! # Key Insight
//!
//! The KV cache naturally "freezes" when generation pauses — positions don't
//! advance, tensors remain populated. Injection is just chunked prefill at a
//! non-zero starting position. No new tensor operations needed.

use std::collections::HashMap;

/// A detected tool call extracted from model output.
#[derive(Debug, Clone)]
pub struct DetectedToolCall {
    /// Name of the tool being called.
    pub tool_name: String,

    /// Arguments to the tool (typically JSON or plain text).
    pub tool_args: String,

    /// Start index in `generated_tokens` where the tool call began.
    pub token_start: usize,

    /// End index in `generated_tokens` where the tool call ended.
    pub token_end: usize,
}

/// Snapshot of the model's attention state at the moment of a tool call.
///
/// Records what the model was attending to when it decided to call a tool.
/// Used for diagnostics and future attention-weighted result processing (CR.1.2).
#[derive(Debug, Clone)]
pub struct AttentionSnapshot {
    /// Per-head attention weights from the tool-call-generating token.
    /// Outer vec = heads, inner vec = attention weight per cache position.
    /// Only populated if `capture_attention` is enabled.
    pub head_weights: Option<Vec<Vec<f32>>>,

    /// Which CacheSpan IDs received the most attention, sorted descending.
    /// Each entry is (span_id, total_attention_weight).
    pub attended_spans: Vec<(u64, f32)>,

    /// The cache position at which the tool call was generated.
    pub position: usize,
}

/// Configuration for a single tool call pattern.
///
/// Defines the start and end markers that delimit a tool call in model output,
/// plus a parser function that extracts the tool name and arguments.
pub struct ToolCallPattern {
    /// Start marker text (e.g., "<tool_call>", "[TOOL_CALL]", "```tool\n").
    pub start_marker: String,

    /// End marker text (e.g., "</tool_call>", "[/TOOL_CALL]", "\n```").
    pub end_marker: String,

    /// Parser: given the text between start and end markers, extract (tool_name, tool_args).
    /// Returns None if the content doesn't parse as a valid tool call.
    pub parser: Box<dyn Fn(&str) -> Option<(String, String)> + Send + Sync>,
}

impl std::fmt::Debug for ToolCallPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallPattern")
            .field("start_marker", &self.start_marker)
            .field("end_marker", &self.end_marker)
            .finish()
    }
}

/// Detects tool calls in streaming LLM output by matching configurable patterns
/// against a rolling buffer of recent tokens.
#[derive(Debug)]
pub struct ToolCallDetector {
    /// Registered tool call patterns (checked in order).
    patterns: Vec<ToolCallPattern>,

    /// Rolling token buffer per cache slot (last N tokens).
    token_buffers: HashMap<usize, Vec<u32>>,

    /// Maximum buffer size (how many tokens to retain for matching).
    buffer_size: usize,

    /// Whether detection is enabled.
    enabled: bool,
}

impl ToolCallDetector {
    /// Create a new tool call detector.
    ///
    /// # Arguments
    /// * `buffer_size` - Number of recent tokens to keep per slot for pattern matching.
    ///   Larger buffers catch longer tool calls but use more memory. Default: 50.
    pub fn new(buffer_size: usize) -> Self {
        Self {
            patterns: Vec::new(),
            token_buffers: HashMap::new(),
            buffer_size,
            enabled: true,
        }
    }

    /// Create a detector with common default patterns.
    ///
    /// Registers patterns for:
    /// - `<tool_call>...</tool_call>` (Llama/generic format)
    /// - `[TOOL_CALL]...[/TOOL_CALL]` (bracket format)
    /// - `<|tool_call|>...<|/tool_call|>` (special token format)
    pub fn with_defaults() -> Self {
        let mut detector = Self::new(50);

        // Pattern 1: XML-style <tool_call>name(args)</tool_call>
        detector.add_pattern(ToolCallPattern {
            start_marker: "<tool_call>".to_string(),
            end_marker: "</tool_call>".to_string(),
            parser: Box::new(|content| {
                // Try to parse "name(args)" or "name: args" format
                if let Some(paren_pos) = content.find('(') {
                    let name = content[..paren_pos].trim().to_string();
                    let args = content[paren_pos + 1..]
                        .trim_end_matches(')')
                        .trim()
                        .to_string();
                    Some((name, args))
                } else if let Some(colon_pos) = content.find(':') {
                    let name = content[..colon_pos].trim().to_string();
                    let args = content[colon_pos + 1..].trim().to_string();
                    Some((name, args))
                } else {
                    // Just a tool name with no args
                    Some((content.trim().to_string(), String::new()))
                }
            }),
        });

        // Pattern 2: Bracket format [TOOL_CALL]...[/TOOL_CALL]
        detector.add_pattern(ToolCallPattern {
            start_marker: "[TOOL_CALL]".to_string(),
            end_marker: "[/TOOL_CALL]".to_string(),
            parser: Box::new(|content| {
                // Try JSON parse first, then fallback to name(args)
                if content.trim_start().starts_with('{') {
                    // JSON format: {"name": "...", "args": {...}}
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(content) {
                        let name = v
                            .get("name")
                            .or_else(|| v.get("tool"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let args = v
                            .get("args")
                            .or_else(|| v.get("arguments"))
                            .or_else(|| v.get("input"))
                            .map(|a| a.to_string())
                            .unwrap_or_default();
                        Some((name, args))
                    } else {
                        None
                    }
                } else if let Some(paren_pos) = content.find('(') {
                    let name = content[..paren_pos].trim().to_string();
                    let args = content[paren_pos + 1..]
                        .trim_end_matches(')')
                        .trim()
                        .to_string();
                    Some((name, args))
                } else {
                    Some((content.trim().to_string(), String::new()))
                }
            }),
        });

        // Pattern 3: Special token format <|tool_call|>...<|/tool_call|>
        detector.add_pattern(ToolCallPattern {
            start_marker: "<|tool_call|>".to_string(),
            end_marker: "<|/tool_call|>".to_string(),
            parser: Box::new(|content| {
                // Same parsing as Pattern 2
                if content.trim_start().starts_with('{') {
                    serde_json::from_str::<serde_json::Value>(content)
                        .ok()
                        .and_then(|v| {
                            let name = v
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            let args = v
                                .get("arguments")
                                .map(|a| a.to_string())
                                .unwrap_or_default();
                            Some((name, args))
                        })
                } else {
                    Some((content.trim().to_string(), String::new()))
                }
            }),
        });

        detector
    }

    /// Register a custom tool call pattern.
    pub fn add_pattern(&mut self, pattern: ToolCallPattern) {
        self.patterns.push(pattern);
    }

    /// Enable or disable detection.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Push a new token into the buffer for a cache slot.
    pub fn push_token(&mut self, cache_idx: usize, token: u32) {
        let buffer = self
            .token_buffers
            .entry(cache_idx)
            .or_insert_with(|| Vec::with_capacity(self.buffer_size));

        buffer.push(token);

        // Trim to buffer_size
        if buffer.len() > self.buffer_size {
            let excess = buffer.len() - self.buffer_size;
            buffer.drain(..excess);
        }
    }

    /// Check for a complete tool call in the token buffer.
    ///
    /// Decodes the buffer to text and checks against all registered patterns.
    /// Returns the first match found, or None.
    ///
    /// # Arguments
    /// * `cache_idx` - The cache slot to check
    /// * `tokenizer` - The tokenizer for decoding tokens to text
    /// * `total_generated` - Total number of tokens generated (for token_start/end)
    pub fn check_for_tool_call(
        &self,
        cache_idx: usize,
        tokenizer: &tokenizers::Tokenizer,
        total_generated: usize,
    ) -> Option<DetectedToolCall> {
        if !self.enabled || self.patterns.is_empty() {
            return None;
        }

        let buffer = self.token_buffers.get(&cache_idx)?;
        if buffer.is_empty() {
            return None;
        }

        // Decode buffer to text
        let decoded = tokenizer.decode(buffer, false).ok()?;

        // Check each pattern
        for pattern in &self.patterns {
            if let Some(start_pos) = decoded.find(&pattern.start_marker) {
                let content_start = start_pos + pattern.start_marker.len();
                if let Some(end_offset) = decoded[content_start..].find(&pattern.end_marker) {
                    // Found complete tool call
                    let content = &decoded[content_start..content_start + end_offset];

                    if let Some((tool_name, tool_args)) = (pattern.parser)(content) {
                        // Calculate approximate token positions
                        let buffer_len = buffer.len();
                        let token_end = total_generated;
                        // Rough estimate: tool call started ~(decoded_len - start_pos) chars ago
                        let call_char_len =
                            pattern.start_marker.len() + end_offset + pattern.end_marker.len();
                        let approx_call_tokens =
                            (call_char_len * buffer_len) / decoded.len().max(1);
                        let token_start = token_end.saturating_sub(approx_call_tokens);

                        return Some(DetectedToolCall {
                            tool_name,
                            tool_args,
                            token_start,
                            token_end,
                        });
                    }
                }
            }
        }

        None
    }

    /// Clear the token buffer for a cache slot (e.g., when request completes).
    pub fn clear_slot(&mut self, cache_idx: usize) {
        self.token_buffers.remove(&cache_idx);
    }

    /// Clear all buffers.
    pub fn clear_all(&mut self) {
        self.token_buffers.clear();
    }

    /// Get the number of registered patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

/// Result from a `forward_batch` call that may include tool call signals.
#[derive(Debug, Clone)]
pub enum GenerationResult {
    /// A token was generated normally.
    Token(u32),

    /// A tool call was detected. Generation is paused (KV cache preserved).
    /// The caller should execute the tool and call `inject_tool_result()`.
    ToolCallDetected(DetectedToolCall),

    /// Request completed (EOS or max tokens).
    Completed,

    /// Request is waiting for a tool result (no action needed this step).
    AwaitingToolResult,

    /// Request was inactive this step (e.g., not in Decoding state).
    Inactive,
}

/// Format a tool result for injection into the model's context.
///
/// Wraps the result in delimiters with a reconciliation cue so the model
/// can distinguish tool output from its own generation and properly
/// integrate the result into its reasoning.
///
/// The reconciliation cue is critical: without it, models often treat
/// tool results as supplementary information rather than evidence that
/// may contradict or update prior assumptions.
pub fn format_tool_result(tool_name: &str, result: &str) -> String {
    format!(
        "\n<tool_result tool=\"{}\">\n{}\n</tool_result>\nIntegrate the tool result above into your reasoning. If it conflicts with prior assumptions, revise them before continuing.\n",
        tool_name, result
    )
}

/// Format a tool result with a custom reconciliation instruction.
///
/// Use this when the default reconciliation cue isn't appropriate
/// for the specific tool or model.
pub fn format_tool_result_with_cue(
    tool_name: &str,
    result: &str,
    reconciliation_cue: &str,
) -> String {
    format!(
        "\n<tool_result tool=\"{}\">\n{}\n</tool_result>\n{}\n",
        tool_name, result, reconciliation_cue
    )
}

/// Format a tool result in a chat-model-compatible structure.
///
/// Some chat-tuned models expect tool results in a specific role format
/// that matches their training data. This provides a more structured format.
pub fn format_tool_result_chat(
    tool_name: &str,
    tool_args: &str,
    result: &str,
) -> String {
    format!(
        "\n<tool_response>\n<tool_name>{}</tool_name>\n<tool_input>{}</tool_input>\n<tool_output>{}</tool_output>\n</tool_response>\nContinue your reasoning using the tool output above as new evidence.\n",
        tool_name, tool_args, result
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mock_tokenizer() -> tokenizers::Tokenizer {
        // Create a minimal tokenizer for testing
        // In practice, tests that need real tokenization would use a model's tokenizer
        use tokenizers::models::bpe::BPE;
        tokenizers::Tokenizer::new(BPE::default())
    }

    #[test]
    fn test_detector_creation() {
        let detector = ToolCallDetector::new(30);
        assert_eq!(detector.pattern_count(), 0);
        assert!(detector.enabled);
    }

    #[test]
    fn test_detector_with_defaults() {
        let detector = ToolCallDetector::with_defaults();
        assert_eq!(detector.pattern_count(), 3);
    }

    #[test]
    fn test_push_token_buffer_limit() {
        let mut detector = ToolCallDetector::new(5);
        for i in 0..10 {
            detector.push_token(0, i);
        }
        assert_eq!(detector.token_buffers[&0].len(), 5);
        assert_eq!(detector.token_buffers[&0][0], 5); // First 5 were trimmed
    }

    #[test]
    fn test_clear_slot() {
        let mut detector = ToolCallDetector::new(10);
        detector.push_token(0, 42);
        detector.push_token(1, 43);
        detector.clear_slot(0);
        assert!(!detector.token_buffers.contains_key(&0));
        assert!(detector.token_buffers.contains_key(&1));
    }

    #[test]
    fn test_disabled_detector() {
        let mut detector = ToolCallDetector::with_defaults();
        detector.set_enabled(false);
        detector.push_token(0, 42);
        let tokenizer = make_mock_tokenizer();
        assert!(detector.check_for_tool_call(0, &tokenizer, 1).is_none());
    }

    #[test]
    fn test_format_tool_result() {
        let result = format_tool_result("get_weather", "22°C and sunny in Paris");
        assert!(result.contains("[TOOL_RESULT name=\"get_weather\"]"));
        assert!(result.contains("22°C and sunny in Paris"));
        assert!(result.contains("[/TOOL_RESULT]"));
    }

    #[test]
    fn test_xml_pattern_parser() {
        let detector = ToolCallDetector::with_defaults();
        let parser = &detector.patterns[0].parser;

        // name(args) format
        let result = parser("get_weather(Paris)");
        assert_eq!(result, Some(("get_weather".to_string(), "Paris".to_string())));

        // name: args format
        let result = parser("search: quantum computing");
        assert_eq!(
            result,
            Some(("search".to_string(), "quantum computing".to_string()))
        );

        // name only
        let result = parser("list_files");
        assert_eq!(result, Some(("list_files".to_string(), String::new())));
    }

    #[test]
    fn test_json_pattern_parser() {
        let detector = ToolCallDetector::with_defaults();
        let parser = &detector.patterns[1].parser; // Bracket format with JSON support

        let json_input = r#"{"name": "get_weather", "args": {"city": "Paris"}}"#;
        let result = parser(json_input);
        assert!(result.is_some());
        let (name, args) = result.unwrap();
        assert_eq!(name, "get_weather");
        assert!(args.contains("Paris"));
    }

    #[test]
    fn test_generation_result_variants() {
        // Ensure all variants are constructible
        let _token = GenerationResult::Token(42);
        let _tool = GenerationResult::ToolCallDetected(DetectedToolCall {
            tool_name: "test".to_string(),
            tool_args: "{}".to_string(),
            token_start: 0,
            token_end: 5,
        });
        let _completed = GenerationResult::Completed;
        let _awaiting = GenerationResult::AwaitingToolResult;
        let _inactive = GenerationResult::Inactive;
    }
}
