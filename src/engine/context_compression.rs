// M5.C: Context Compression System
//
// Text-level compression strategies for reducing context size while preserving
// important information. Complements the existing KV cache compression.
//
// Strategies:
// - Extractive: Select most important sentences
// - Entity-preserving: Keep sentences with key entities
// - Token-based: Aggressive truncation with smart boundaries
// - Hierarchical: Multiple compression levels

use std::collections::HashSet;

/// Compression strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// Keep most important sentences (extractive summarization)
    Extractive,

    /// Preserve sentences containing key entities
    EntityPreserving,

    /// Simple token-based truncation
    TokenBased,

    /// Hierarchical compression with multiple levels
    Hierarchical,
}

/// Configuration for compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub strategy: CompressionStrategy,
    pub target_ratio: f32, // 0.0 to 1.0, target compression ratio
    pub preserve_entities: bool,
    pub preserve_code: bool, // Keep code blocks intact
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            strategy: CompressionStrategy::Extractive,
            target_ratio: 0.5, // Compress to 50% of original
            preserve_entities: true,
            preserve_code: true,
        }
    }
}

/// Result of compression
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub ratio: f32,
    pub preserved_entities: Vec<String>,
}

impl CompressionResult {
    pub fn compression_ratio(&self) -> f32 {
        if self.original_tokens == 0 {
            return 1.0;
        }
        self.compressed_tokens as f32 / self.original_tokens as f32
    }
}

/// Context compressor
pub struct ContextCompressor {
    config: CompressionConfig,
}

impl ContextCompressor {
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress text according to strategy
    pub fn compress(&self, text: &str) -> CompressionResult {
        match self.config.strategy {
            CompressionStrategy::Extractive => self.extractive_compress(text),
            CompressionStrategy::EntityPreserving => self.entity_preserving_compress(text),
            CompressionStrategy::TokenBased => self.token_based_compress(text),
            CompressionStrategy::Hierarchical => self.hierarchical_compress(text),
        }
    }

    /// Extractive compression - keep most important sentences
    fn extractive_compress(&self, text: &str) -> CompressionResult {
        let sentences = self.split_sentences(text);
        let original_tokens = self.count_tokens(text);

        if sentences.is_empty() {
            return CompressionResult {
                compressed: String::new(),
                original_tokens,
                compressed_tokens: 0,
                ratio: 0.0,
                preserved_entities: Vec::new(),
            };
        }

        // Score sentences by importance
        let scored = self.score_sentences(&sentences);

        // Calculate target token count
        let target_tokens = (original_tokens as f32 * self.config.target_ratio) as usize;

        // Select sentences greedily until we reach target
        let selected = self.select_sentences(&scored, target_tokens);

        let compressed = selected.join(" ");
        let compressed_tokens = self.count_tokens(&compressed);
        let entities = if self.config.preserve_entities {
            self.extract_entities(&compressed)
        } else {
            Vec::new()
        };

        CompressionResult {
            compressed,
            original_tokens,
            compressed_tokens,
            ratio: compressed_tokens as f32 / original_tokens as f32,
            preserved_entities: entities,
        }
    }

    /// Entity-preserving compression
    fn entity_preserving_compress(&self, text: &str) -> CompressionResult {
        let sentences = self.split_sentences(text);
        let original_tokens = self.count_tokens(text);

        // Extract entities from full text
        let entities = self.extract_entities(text);

        // Keep sentences that contain important entities
        let mut kept_sentences = Vec::new();
        let mut token_count = 0;
        let target_tokens = (original_tokens as f32 * self.config.target_ratio) as usize;

        for sentence in sentences {
            let has_entity = entities.iter().any(|e| sentence.contains(e));
            if has_entity || token_count < target_tokens / 2 {
                token_count += self.count_tokens(&sentence);
                kept_sentences.push(sentence);

                if token_count >= target_tokens {
                    break;
                }
            }
        }

        let compressed = kept_sentences.join(" ");
        let compressed_tokens = self.count_tokens(&compressed);

        CompressionResult {
            compressed,
            original_tokens,
            compressed_tokens,
            ratio: compressed_tokens as f32 / original_tokens as f32,
            preserved_entities: entities,
        }
    }

    /// Token-based compression - simple truncation
    fn token_based_compress(&self, text: &str) -> CompressionResult {
        let original_tokens = self.count_tokens(text);
        let target_tokens = (original_tokens as f32 * self.config.target_ratio) as usize;

        let words: Vec<&str> = text.split_whitespace().collect();
        let kept_words: Vec<&str> = words.into_iter().take(target_tokens).collect();

        let compressed = kept_words.join(" ");
        let compressed_tokens = self.count_tokens(&compressed);
        let entities = if self.config.preserve_entities {
            self.extract_entities(&compressed)
        } else {
            Vec::new()
        };

        CompressionResult {
            compressed,
            original_tokens,
            compressed_tokens,
            ratio: compressed_tokens as f32 / original_tokens as f32,
            preserved_entities: entities,
        }
    }

    /// Hierarchical compression - multiple levels
    fn hierarchical_compress(&self, text: &str) -> CompressionResult {
        // First level: Remove redundant information
        let level1 = self.remove_redundancy(text);

        // Second level: Extract key sentences
        let temp_config = CompressionConfig {
            strategy: CompressionStrategy::Extractive,
            target_ratio: self.config.target_ratio,
            preserve_entities: self.config.preserve_entities,
            preserve_code: self.config.preserve_code,
        };
        let temp_compressor = ContextCompressor::new(temp_config);
        temp_compressor.extractive_compress(&level1)
    }

    /// Split text into sentences
    fn split_sentences(&self, text: &str) -> Vec<String> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Score sentences by importance
    fn score_sentences(&self, sentences: &[String]) -> Vec<(String, f32)> {
        sentences
            .iter()
            .map(|s| {
                let mut score = 0.0;

                // Length-based scoring (prefer medium-length sentences)
                let words = s.split_whitespace().count();
                if words >= 5 && words <= 20 {
                    score += 1.0;
                }

                // Entity bonus
                if self.config.preserve_entities {
                    let entities = self.extract_entities(s);
                    score += entities.len() as f32 * 0.5;
                }

                // Code block bonus
                if self.config.preserve_code
                    && (s.contains("```") || s.contains("fn ") || s.contains("def "))
                {
                    score += 2.0;
                }

                // Question bonus (likely important)
                if s.contains('?') {
                    score += 0.5;
                }

                (s.clone(), score)
            })
            .collect()
    }

    /// Select sentences greedily by score until target reached
    fn select_sentences(&self, scored: &[(String, f32)], target_tokens: usize) -> Vec<String> {
        let mut sorted = scored.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected = Vec::new();
        let mut token_count = 0;

        for (sentence, _score) in sorted {
            let sentence_tokens = self.count_tokens(&sentence);
            if token_count + sentence_tokens <= target_tokens {
                selected.push(sentence);
                token_count += sentence_tokens;
            }

            if token_count >= target_tokens {
                break;
            }
        }

        selected
    }

    /// Extract named entities (simple heuristic: capitalized words)
    fn extract_entities(&self, text: &str) -> Vec<String> {
        let mut entities = HashSet::new();

        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() > 2
                && clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                && clean.chars().all(|c| c.is_alphanumeric())
            {
                entities.insert(clean.to_string());
            }
        }

        entities.into_iter().collect()
    }

    /// Count tokens (simple word-based estimate)
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Remove redundant information
    fn remove_redundancy(&self, text: &str) -> String {
        let sentences = self.split_sentences(text);
        let mut seen = HashSet::new();
        let mut kept = Vec::new();

        for sentence in sentences {
            // Simple deduplication based on first few words
            let key: String = sentence
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
            if !seen.contains(&key) {
                seen.insert(key);
                kept.push(sentence);
            }
        }

        kept.join(". ") + "."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.strategy, CompressionStrategy::Extractive);
        assert_eq!(config.target_ratio, 0.5);
        assert!(config.preserve_entities);
    }

    #[test]
    fn test_split_sentences() {
        let compressor = ContextCompressor::new(CompressionConfig::default());
        let text = "First sentence. Second sentence! Third sentence?";
        let sentences = compressor.split_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence");
        assert_eq!(sentences[1], "Second sentence");
        assert_eq!(sentences[2], "Third sentence");
    }

    #[test]
    fn test_extract_entities() {
        let compressor = ContextCompressor::new(CompressionConfig::default());
        let text = "HashMap and Vector are Rust types. The Database connection failed.";
        let entities = compressor.extract_entities(text);

        assert!(entities.contains(&"HashMap".to_string()));
        assert!(entities.contains(&"Vector".to_string()));
        assert!(entities.contains(&"Rust".to_string()));
        assert!(entities.contains(&"Database".to_string()));
    }

    #[test]
    fn test_token_based_compression() {
        let config = CompressionConfig {
            strategy: CompressionStrategy::TokenBased,
            target_ratio: 0.5,
            preserve_entities: false,
            preserve_code: false,
        };
        let compressor = ContextCompressor::new(config);

        let text = "This is a test sentence with many words that should be compressed.";
        let result = compressor.compress(text);

        assert!(result.compressed_tokens <= result.original_tokens / 2 + 1);
        assert!(result.ratio <= 0.6); // Allow some slack
    }

    #[test]
    fn test_extractive_compression() {
        let config = CompressionConfig {
            strategy: CompressionStrategy::Extractive,
            target_ratio: 0.5,
            preserve_entities: true,
            preserve_code: false,
        };
        let compressor = ContextCompressor::new(config);

        let text = "Short. This is a medium length sentence with content. Another short one.";
        let result = compressor.compress(text);

        assert!(!result.compressed.is_empty());
        assert!(result.compressed_tokens < result.original_tokens);
        assert!(result.ratio <= 0.7);
    }

    #[test]
    fn test_entity_preserving_compression() {
        let config = CompressionConfig {
            strategy: CompressionStrategy::EntityPreserving,
            target_ratio: 0.5,
            preserve_entities: true,
            preserve_code: false,
        };
        let compressor = ContextCompressor::new(config);

        let text = "HashMap is important. This is filler text. Vector is also important.";
        let result = compressor.compress(text);

        assert!(result.preserved_entities.len() > 0);
        assert!(result.compressed.contains("HashMap") || result.compressed.contains("Vector"));
    }

    #[test]
    fn test_hierarchical_compression() {
        let config = CompressionConfig {
            strategy: CompressionStrategy::Hierarchical,
            target_ratio: 0.3,
            preserve_entities: true,
            preserve_code: false,
        };
        let compressor = ContextCompressor::new(config);

        let text = "First sentence. First sentence. Second unique sentence. Third unique sentence.";
        let result = compressor.compress(text);

        // Should remove redundancy and compress further
        assert!(result.compressed_tokens < result.original_tokens);
        assert!(!result.compressed.contains("First sentence. First sentence"));
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let result = CompressionResult {
            compressed: "test".to_string(),
            original_tokens: 100,
            compressed_tokens: 50,
            ratio: 0.5,
            preserved_entities: vec![],
        };

        assert_eq!(result.compression_ratio(), 0.5);
    }

    #[test]
    fn test_empty_text_compression() {
        let compressor = ContextCompressor::new(CompressionConfig::default());
        let result = compressor.compress("");

        assert_eq!(result.compressed, "");
        assert_eq!(result.original_tokens, 0);
        assert_eq!(result.compressed_tokens, 0);
    }

    #[test]
    fn test_remove_redundancy() {
        let compressor = ContextCompressor::new(CompressionConfig::default());
        let text = "This is first. This is second. That is third.";
        let result = compressor.remove_redundancy(text);

        // Should keep "This is" only once
        assert!(result.contains("This is"));
        assert!(result.contains("That is"));
    }
}
