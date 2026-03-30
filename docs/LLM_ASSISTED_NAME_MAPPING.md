# LLM-Assisted Tensor Name Mapping

**Status**: RESEARCH (Post-v1.0)  
**Priority**: MEDIUM  
**Dependencies**: M4.1 (Basic name mapping), M6 (Tool registry)  
**Target Milestone**: M6.5

## Vision

Use a small, fast language model to intelligently map tensor names when regex patterns fail. This enables Lightbulb to handle completely novel architectures without manual pattern engineering.

## Why LLMs Excel at This Task

**Natural Language Understanding**:
- Tensor names are semi-natural language: `"attention_query_projection"`, `"mlp_gate_weight"`
- LLMs trained on code understand these naming conventions
- Can reason about semantic relationships between names

**Fuzzy Matching**:
- Regex is brittle: `"attn_q"` vs `"attention_query"` vs `"q_proj"` requires 3 patterns
- LLMs understand these are equivalent
- Can handle typos, abbreviations, unconventional naming

**Contextual Reasoning**:
- Can use tensor shapes as hints: `[4096, 4096]` likely a projection matrix
- Understands architectural patterns: "if there's `attn_q`, there should be `attn_k` nearby"
- Learns from examples in prompt

## Architecture

### Three-Tier Matching Strategy

```
┌─────────────────────────────────────────────────────┐
│  Tier 1: Regex Pattern Matching (Fast Path)        │
│  - Known architectures (LLaMA, GPT, Mistral)       │
│  - <1ms latency, deterministic                      │
│  - 95% of cases                                     │
└────────────────┬────────────────────────────────────┘
                 │ Fallback if no match
                 ▼
┌─────────────────────────────────────────────────────┐
│  Tier 2: LLM-Assisted Matching (Smart Fallback)    │
│  - Probabilistic matching with confidence           │
│  - ~50-200ms latency (small LLM)                    │
│  - 4% of cases                                      │
└────────────────┬────────────────────────────────────┘
                 │ Fallback if low confidence
                 ▼
┌─────────────────────────────────────────────────────┐
│  Tier 3: User Override (Manual Configuration)      │
│  - Explicit mapping in config file                  │
│  - Last resort for exotic architectures             │
│  - 1% of cases                                      │
└─────────────────────────────────────────────────────┘
```

### LLM Model Selection

**Requirements**:
- **Small**: <1B parameters (must run on CPU efficiently)
- **Fast**: <200ms inference on CPU
- **Code-aware**: Trained on code/technical text
- **Local**: No external API dependencies

**Candidates**:
1. **Phi-2 (2.7B)** - Microsoft's efficient reasoning model
   - Strong code understanding
   - Runs on CPU reasonably fast
   - Good semantic matching

2. **TinyLlama (1.1B)** - Already have it!
   - Already in our test suite
   - Fast enough for this task
   - Knows common ML terminology

3. **CodeT5-small (220M)** - Dedicated code model
   - Extremely fast on CPU
   - Understands variable naming conventions
   - Specialized for code tasks

**Recommendation**: Start with TinyLlama (already available), upgrade to Phi-2 if needed.

## Implementation Design

### 1. LLM-Assisted Mapper Module

```rust
// src/pruning/name_mapping_llm.rs

use crate::pruning::name_mapping::{TensorNameMapper, ComponentType};

pub struct LlmAssistedMapper {
    /// Underlying regex-based mapper (fast path)
    base_mapper: TensorNameMapper,
    
    /// Small LLM for fallback matching
    llm: Option<SmallLlmModel>,
    
    /// Cache of LLM predictions to avoid repeated queries
    prediction_cache: HashMap<String, String>,
    
    /// Confidence threshold (0.0-1.0)
    confidence_threshold: f32,
}

impl LlmAssistedMapper {
    /// Create mapper with optional LLM fallback
    pub fn new(
        tensor_names: &[String],
        llm_path: Option<&Path>,
    ) -> Result<Self> {
        // First try regex-based detection
        let base_mapper = TensorNameMapper::from_tensor_names(tensor_names)?;
        
        // Load LLM if provided
        let llm = if let Some(path) = llm_path {
            Some(SmallLlmModel::load(path)?)
        } else {
            None
        };
        
        Ok(Self {
            base_mapper,
            llm,
            prediction_cache: HashMap::new(),
            confidence_threshold: 0.75, // Require 75% confidence
        })
    }
    
    /// Map abstract name to concrete tensor name
    pub fn map_name(&mut self, abstract_name: &str) -> Result<MappingResult> {
        // Try fast path first
        if let Some(name) = self.base_mapper.map_name(abstract_name) {
            return Ok(MappingResult::Confident(name));
        }
        
        // Fallback to LLM if available
        if let Some(llm) = &self.llm {
            return self.map_with_llm(abstract_name);
        }
        
        // No mapping found
        Ok(MappingResult::NotFound)
    }
    
    /// Use LLM to find best match
    fn map_with_llm(&mut self, abstract_name: &str) -> Result<MappingResult> {
        // Check cache first
        if let Some(cached) = self.prediction_cache.get(abstract_name) {
            return Ok(MappingResult::Confident(cached.clone()));
        }
        
        let llm = self.llm.as_ref().unwrap();
        
        // Build prompt
        let prompt = self.build_matching_prompt(abstract_name);
        
        // Query LLM
        let response = llm.generate(&prompt, GenerateConfig {
            max_tokens: 100,
            temperature: 0.1, // Low temperature for deterministic matching
            top_p: 0.9,
        })?;
        
        // Parse response
        let prediction = self.parse_llm_response(&response)?;
        
        // Check confidence
        if prediction.confidence >= self.confidence_threshold {
            // Cache successful prediction
            self.prediction_cache.insert(
                abstract_name.to_string(),
                prediction.tensor_name.clone(),
            );
            
            Ok(MappingResult::Confident(prediction.tensor_name))
        } else {
            Ok(MappingResult::Uncertain {
                candidate: prediction.tensor_name,
                confidence: prediction.confidence,
            })
        }
    }
    
    /// Build few-shot prompt for LLM
    fn build_matching_prompt(&self, abstract_name: &str) -> String {
        let available_tensors = self.base_mapper.get_all_tensor_names();
        
        format!(
            r#"You are a tensor name matching expert. Given a abstract tensor name and a list of available tensor names, identify the best match.

# Examples:
Abstract: "layer_0.attention.query"
Available: ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"]
Match: blk.0.attn_q.weight
Confidence: 0.95
Reasoning: "attn_q" clearly corresponds to "attention.query"

Abstract: "layer_5.ffn.gate"
Available: ["layers.5.mlp.gate_proj.weight", "layers.5.mlp.up_proj.weight"]
Match: layers.5.mlp.gate_proj.weight
Confidence: 0.90
Reasoning: "gate_proj" matches "ffn.gate"

# Your Task:
Abstract: "{}"
Available: {:?}

Match: "#,
            abstract_name,
            available_tensors.iter().take(20).collect::<Vec<_>>(), // Limit context
        )
    }
    
    /// Parse structured output from LLM
    fn parse_llm_response(&self, response: &str) -> Result<Prediction> {
        // Look for pattern: "Match: <name>\nConfidence: <score>"
        let lines: Vec<&str> = response.lines().collect();
        
        let tensor_name = lines
            .iter()
            .find(|l| l.starts_with("Match:"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .context("Failed to parse tensor name from LLM response")?;
        
        let confidence = lines
            .iter()
            .find(|l| l.starts_with("Confidence:"))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(0.5); // Default to medium confidence
        
        Ok(Prediction {
            tensor_name,
            confidence,
        })
    }
}

pub enum MappingResult {
    /// High-confidence match found
    Confident(String),
    
    /// Low-confidence match (user should verify)
    Uncertain {
        candidate: String,
        confidence: f32,
    },
    
    /// No match found
    NotFound,
}

struct Prediction {
    tensor_name: String,
    confidence: f32,
}
```

### 2. Small LLM Model Wrapper

```rust
// src/model/small_llm.rs

/// Lightweight LLM for tensor name matching
pub struct SmallLlmModel {
    model: LlamaModel, // Reuse existing infrastructure
    tokenizer: Tokenizer,
}

impl SmallLlmModel {
    pub fn load(path: &Path) -> Result<Self> {
        // Load tiny model (TinyLlama, Phi-2, etc.)
        let model = LlamaModel::load(path)?;
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))?;
        
        Ok(Self { model, tokenizer })
    }
    
    pub fn generate(&self, prompt: &str, config: GenerateConfig) -> Result<String> {
        // Simple greedy decoding (no need for fancy sampling)
        let tokens = self.tokenizer.encode(prompt, true)?;
        
        let mut output_tokens = Vec::new();
        let mut current_tokens = tokens.clone();
        
        for _ in 0..config.max_tokens {
            let logits = self.model.forward(&current_tokens)?;
            let next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            
            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
            
            output_tokens.push(next_token);
            current_tokens = vec![next_token];
        }
        
        self.tokenizer.decode(&output_tokens, true)
    }
}

pub struct GenerateConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}
```

### 3. Configuration

```yaml
# lightbulb.yaml
model:
  path: "models/llama-7b.gguf"
  
  # LLM-assisted name mapping (optional)
  name_mapping:
    # Enable LLM fallback for unknown architectures
    use_llm_fallback: true
    
    # Path to small LLM for matching (TinyLlama, Phi-2, etc.)
    llm_model: "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Confidence threshold (0.0-1.0)
    confidence_threshold: 0.75
    
    # Cache LLM predictions to disk
    cache_predictions: true
    cache_path: ".cache/name_mapping.json"
```

### 4. User Experience

**Automatic Fallback (Transparent)**:
```
[INFO] Loading model: models/custom-transformer.gguf
[INFO] Architecture: Unknown (using LLM-assisted mapping)
[INFO] LLM found 24/28 tensor mappings with high confidence
[WARN] Low confidence matches:
  - layer_3.ffn.gate → layers.3.mlp.w1 (confidence: 0.68)
  - layer_7.attention.output → blk.7.attn_o (confidence: 0.72)

Review suggested mappings? [y/N]: n
[INFO] Proceeding with all matches...
[INFO] Model loaded successfully!
```

**Interactive Mode**:
```
[WARN] Uncertain match for "layer_5.attention.query"
[INFO] LLM suggests: "transformer.h.5.attn.c_attn" (confidence: 0.68)
[INFO] Available alternatives:
  1. transformer.h.5.attn.q_proj (manual inspection)
  2. transformer.h.5.attn.c_attn (LLM suggestion)
  3. Skip this tensor (may cause errors)

Choose option [1-3]: 2
[INFO] Using LLM suggestion. Add to config file to avoid this prompt next time.
```

## Performance Characteristics

### Latency Breakdown

**Fast Path (Regex)**: 
- 0.5-1ms per tensor
- 99% cache hit rate after first query
- Total: ~50ms for 100 tensors

**LLM Fallback**:
- 50-200ms per query (TinyLlama on CPU)
- Only invoked for unknown patterns
- Cached after first match
- Total: ~1-2s for 10 unknown tensors (one-time cost at load)

**Overall Impact**:
- Model loading: +0-2s depending on architecture novelty
- Inference: Zero overhead (mapping done at load time)
- Acceptable for model loading scenario

### Accuracy

**Expected Performance**:
- Regex patterns: 100% accuracy (known architectures)
- LLM matching: 85-95% accuracy (depends on model)
- Human verification: 100% accuracy (with override config)

**Failure Modes**:
- LLM suggests wrong tensor (rare, caught by shape validation)
- Low confidence forces user intervention (better than silent failure)
- Completely exotic naming requires manual config (unavoidable)

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_llm_assisted_mapping_llama() {
    let tensor_names = vec![
        "blk.0.attn_q.weight".to_string(),
        "blk.0.attn_k.weight".to_string(),
    ];
    
    let mut mapper = LlmAssistedMapper::new(
        &tensor_names,
        Some(Path::new("models/tinyllama.gguf")),
    ).unwrap();
    
    // Should use fast path (regex)
    let result = mapper.map_name("layer_0.attention.query").unwrap();
    assert!(matches!(result, MappingResult::Confident(_)));
}

#[test]
fn test_llm_fallback_novel_architecture() {
    let tensor_names = vec![
        "custom_attention_query_layer_0_weight".to_string(),
    ];
    
    let mut mapper = LlmAssistedMapper::new(
        &tensor_names,
        Some(Path::new("models/tinyllama.gguf")),
    ).unwrap();
    
    // Should use LLM fallback
    let result = mapper.map_name("layer_0.attention.query").unwrap();
    match result {
        MappingResult::Confident(name) => {
            assert_eq!(name, "custom_attention_query_layer_0_weight");
        }
        MappingResult::Uncertain { candidate, confidence } => {
            assert!(confidence > 0.5);
        }
        MappingResult::NotFound => panic!("Should find match"),
    }
}
```

### Integration Tests

- Load 3 completely novel architectures (create synthetic test models)
- Verify LLM matches 80%+ of tensors correctly
- Confirm inference correctness after mapping
- Test caching behavior (2nd load should be instant)

## Advantages Over Pure Regex

| Aspect | Regex Only | LLM-Assisted |
|--------|-----------|--------------|
| **Known architectures** | Fast, perfect | Same (fast path) |
| **Novel architectures** | Fails | 85-95% success |
| **Maintenance** | Requires pattern engineering | Self-learning from examples |
| **User experience** | Manual config required | Mostly automatic |
| **Latency** | <1ms | 50-200ms (fallback only) |
| **Accuracy** | 100% or 0% (binary) | Probabilistic with confidence |

## Future Enhancements

### 1. Learning from User Corrections

```rust
impl LlmAssistedMapper {
    /// Learn from user override
    pub fn learn_from_correction(&mut self, abstract_name: &str, correct_name: &str) {
        // Add to few-shot examples for future prompts
        self.user_corrections.push((abstract_name.to_string(), correct_name.to_string()));
        
        // Save to persistent cache
        self.save_corrections();
    }
}
```

### 2. Community Crowdsourcing

- Upload anonymized mapping corrections to community database
- Download updated patterns from community
- Federated learning approach (privacy-preserving)

### 3. Multi-Model Ensemble

Use multiple small LLMs and vote on best match:
- TinyLlama (1.1B) - Fast, general purpose
- CodeT5 (220M) - Code-specific
- Domain-specific model trained on model architectures

### 4. Structured Output (JSON)

Instead of free text, force LLM to output JSON:
```json
{
  "match": "blk.0.attn_q.weight",
  "confidence": 0.95,
  "reasoning": "attn_q clearly maps to attention.query",
  "alternatives": [
    {"name": "blk.0.q_proj.weight", "confidence": 0.45}
  ]
}
```

## Implementation Roadmap

**Phase 1: Prototype (2 weeks)**
- [ ] Basic LlmAssistedMapper with TinyLlama
- [ ] Simple prompt engineering
- [ ] Test on 3 architectures
- [ ] Measure accuracy and latency

**Phase 2: Production Ready (2 weeks)**
- [ ] Caching system
- [ ] Confidence thresholds
- [ ] User override workflow
- [ ] Documentation

**Phase 3: Optimization (1 week)**
- [ ] Prompt optimization for better accuracy
- [ ] Faster LLM model (CodeT5?)
- [ ] Parallel batch matching
- [ ] Performance profiling

**Phase 4: Polish (1 week)**
- [ ] Community feedback integration
- [ ] Learning from corrections
- [ ] Comprehensive test suite
- [ ] User guide with examples

## Success Criteria

- ✅ 85%+ accuracy on novel architectures
- ✅ <2s overhead for LLM fallback at load time
- ✅ Zero overhead during inference
- ✅ Clear user experience with confidence scores
- ✅ Backward compatible with regex-only mode

## References

- Phi-2: https://huggingface.co/microsoft/phi-2
- TinyLlama: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- CodeT5: https://huggingface.co/Salesforce/codet5-small
- Few-shot learning for code: https://arxiv.org/abs/2005.14165
