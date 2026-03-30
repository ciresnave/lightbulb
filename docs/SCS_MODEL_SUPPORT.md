# Semantic Coordinate Space (SCS) Model Support

**Status**: PLANNED (Post-v1.0)  
**Target**: v2.0+ (Research Features)  
**Dependencies**: M0-M5 stable core, feature flag system

---

## Overview

This document outlines the infrastructure changes required to support Semantic Coordinate Space (SCS) models alongside traditional transformer models in Lightbulb. SCS represents a fundamentally different LLM architecture that uses coordinate-based semantic representations instead of token sequences.

### What is SCS?

SCS models operate on:
- **Character-level input** → **Variable-length semantic chunks** → **Coordinate sequences** → **Text output**

Instead of predicting the next token, SCS models predict the next **semantic coordinate** in a multi-dimensional space where:
- Each dimension represents a learned relationship type (temporal, causal, spatial, etc.)
- Positions encode semantic meaning
- Dimensional weights encode relationship strengths
- Chunks are variable-length (1-50+ characters) based on semantic certainty

### Key Architectural Differences

| Aspect | Transformer | SCS |
|--------|------------|-----|
| Input | Tokens (BPE/WordPiece) | Characters |
| Processing Unit | Token embeddings | Semantic coordinates |
| Sequence Length | Fixed tokenization | Variable chunking |
| Cache | KV cache (key-value pairs) | Coordinate cache (hierarchical) |
| Generation | Next token prediction | Next coordinate prediction |
| Output | Token IDs → Detokenization | Coordinate sequence → Reconstruction |

---

## Implementation Plan

### Phase 1: Core Infrastructure (4-6 weeks)

#### 1.1 Model Loader Extension

**File**: `src/loaders.rs`

Add SCS model loading capability:

```rust
/// Load SCS model from directory
/// Expects: scs_config.json, char_encoder.safetensors, chunker.safetensors,
///          coord_encoder.safetensors, dimensional_attention.safetensors,
///          coord_decoder.safetensors
pub fn load_scs_model(
    model_dir: &str,
    device: &Device,
) -> Result<ScsModel> {
    // Load SCS-specific configuration
    let config = load_scs_config(model_dir)?;
    
    // Load component weights
    let char_encoder = load_component(model_dir, "char_encoder", &config, device)?;
    let chunker = load_component(model_dir, "chunker", &config, device)?;
    let coordinate_encoder = load_component(model_dir, "coord_encoder", &config, device)?;
    let dimensional_attention = load_component(model_dir, "dimensional_attention", &config, device)?;
    let coordinate_decoder = load_component(model_dir, "coord_decoder", &config, device)?;
    
    Ok(ScsModel {
        char_encoder,
        chunker,
        coordinate_encoder,
        dimensional_attention,
        coordinate_decoder,
        config,
    })
}
```

**New Types**:

```rust
pub struct ScsModel {
    pub char_encoder: CharacterEncoder,
    pub chunker: CertaintyBasedChunker,
    pub coordinate_encoder: CoordinateEncoder,
    pub dimensional_attention: DimensionalAttention,
    pub coordinate_decoder: CoordinateDecoder,
    pub config: ScsConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScsConfig {
    pub num_dimensions: usize,
    pub positions_per_dimension: usize,
    pub dimension_labels: Vec<String>,
    pub certainty_threshold: f32,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub use_hierarchical_cache: bool,
    pub cache_tiers: Vec<CacheTierConfig>,
}
```

#### 1.2 Representation Types

**New File**: `src/model/scs_representations.rs`

Define coordinate space data structures:

```rust
/// A semantic coordinate in SCS space
#[derive(Debug, Clone)]
pub struct SemanticCoordinate {
    /// Position vector in semantic space [num_dimensions]
    pub position: Tensor,
    
    /// Active dimensions and their weights (sparse representation)
    pub dimensional_weights: HashMap<usize, f32>,
    
    /// Chunk span in original text (for reconstruction)
    pub source_span: (usize, usize),
    
    /// Certainty score (from chunker)
    pub certainty: f32,
    
    /// Optional: compositionality info (for storage optimization)
    pub composition: Option<CompositionInfo>,
}

#[derive(Debug, Clone)]
pub struct CompositionInfo {
    /// If additive, store component coordinate IDs instead of full coordinate
    pub is_additive: bool,
    pub component_ids: Vec<usize>,
    pub reconstruction_error: f32,
}

/// Variable-length chunk (output from certainty-based chunker)
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Character sequence
    pub chars: Vec<char>,
    
    /// Span in original text
    pub span: (usize, usize),
    
    /// Certainty score from chunker
    pub certainty: f32,
}
```

#### 1.3 Inference Pipeline

**New File**: `src/model/scs_pipeline.rs`

Implement SCS-specific inference:

```rust
/// SCS inference pipeline
pub struct ScsInferencePipeline {
    model: ScsModel,
    cache: HierarchicalScsCache,
    device: Device,
}

impl ScsInferencePipeline {
    pub fn new(model: ScsModel, device: Device) -> Result<Self> {
        let cache = HierarchicalScsCache::new(&model.config)?;
        Ok(Self { model, cache, device })
    }
    
    /// Process input text and return coordinate sequence
    pub fn encode(&mut self, text: &str) -> Result<Vec<SemanticCoordinate>> {
        // Step 1: Character encoding
        let chars: Vec<char> = text.chars().collect();
        let char_embeddings = self.model.char_encoder.encode(&chars)?;
        
        // Step 2: Certainty-based chunking
        let chunks = self.model.chunker.chunk(&char_embeddings, &chars)?;
        
        // Step 3: Coordinate encoding
        let mut coordinates = Vec::new();
        for chunk in chunks {
            let coord = self.model.coordinate_encoder.encode(&chunk, &self.cache)?;
            coordinates.push(coord);
        }
        
        // Step 4: Update cache with new coordinates
        self.cache.update(&coordinates)?;
        
        Ok(coordinates)
    }
    
    /// Generate next coordinate(s) from current context
    pub fn generate_next(
        &mut self,
        context_coords: &[SemanticCoordinate],
    ) -> Result<SemanticCoordinate> {
        // Use dimensional attention over coordinate sequence
        let attention_output = self.model.dimensional_attention.forward(
            context_coords,
            &self.cache,
        )?;
        
        // Predict next coordinate
        let next_coord = self.model.coordinate_encoder.decode_logits(
            &attention_output
        )?;
        
        Ok(next_coord)
    }
    
    /// Decode coordinate sequence back to text
    pub fn decode(&self, coordinates: &[SemanticCoordinate]) -> Result<String> {
        self.model.coordinate_decoder.decode(coordinates)
    }
    
    /// Full generation loop
    pub fn generate(&mut self, prompt: &str, max_coords: usize) -> Result<String> {
        // Encode prompt
        let mut coordinates = self.encode(prompt)?;
        
        // Generate new coordinates
        for _ in 0..max_coords {
            let next = self.generate_next(&coordinates)?;
            coordinates.push(next);
        }
        
        // Decode to text
        self.decode(&coordinates)
    }
}
```

#### 1.4 Hierarchical Cache

**New File**: `src/cache/hierarchical_scs_cache.rs`

Multi-tier coordinate caching:

```rust
/// Hierarchical cache for semantic coordinates
pub struct HierarchicalScsCache {
    /// Tier 1: Recent context (last 512 coordinates, full precision)
    recent: Vec<SemanticCoordinate>,
    recent_capacity: usize,
    
    /// Tier 2: Mid-range (512-4096 coords, quantized)
    mid_range: Vec<QuantizedCoordinate>,
    mid_capacity: usize,
    
    /// Tier 3: Distant (4096+, summarized)
    distant: Vec<SummaryCoordinate>,
    distant_capacity: usize,
    
    /// Pattern cache for fast chunking
    pattern_cache: PatternCache,
    
    /// Content cache for frequent chunks
    content_cache: LruCache<Vec<char>, SemanticCoordinate>,
}

impl HierarchicalScsCache {
    pub fn new(config: &ScsConfig) -> Result<Self> {
        Ok(Self {
            recent: Vec::with_capacity(config.cache_tiers[0].capacity),
            recent_capacity: config.cache_tiers[0].capacity,
            mid_range: Vec::with_capacity(config.cache_tiers[1].capacity),
            mid_capacity: config.cache_tiers[1].capacity,
            distant: Vec::with_capacity(config.cache_tiers[2].capacity),
            distant_capacity: config.cache_tiers[2].capacity,
            pattern_cache: PatternCache::new(10000),
            content_cache: LruCache::new(100000),
        })
    }
    
    /// Add new coordinate and manage tier transitions
    pub fn update(&mut self, coords: &[SemanticCoordinate]) -> Result<()> {
        for coord in coords {
            self.recent.push(coord.clone());
            
            if self.recent.len() > self.recent_capacity {
                self.compress_recent_to_mid()?;
            }
            
            if self.mid_range.len() > self.mid_capacity {
                self.compress_mid_to_distant()?;
            }
        }
        Ok(())
    }
    
    /// Retrieve all coordinates for attention (with appropriate resolution per tier)
    pub fn get_attention_context(&self) -> AttentionContext {
        AttentionContext {
            recent: &self.recent,
            mid_range: &self.mid_range,
            distant: &self.distant,
        }
    }
}
```

#### 1.5 Coordinate Sampling

**File**: `src/sampling.rs` (extend)

Add coordinate-specific sampling:

```rust
/// Sample next coordinate from coordinate-space logits
pub fn sample_coordinate(
    position_logits: &Tensor,  // [num_dimensions, positions_per_dim]
    weight_logits: &Tensor,    // [num_dimensions]
    params: &SamplingParams,
) -> Result<SemanticCoordinate> {
    // Temperature scaling per dimension
    let mut position_probs = apply_coordinate_temperature(
        position_logits,
        params.temperature,
    )?;
    
    // Top-k/top-p filtering per dimension
    if let Some(k) = params.top_k {
        position_probs = top_k_filter_coordinates(&position_probs, k)?;
    }
    
    if let Some(p) = params.top_p {
        position_probs = top_p_filter_coordinates(&position_probs, p)?;
    }
    
    // Sample position for each dimension
    let mut position = Vec::new();
    let mut weights = HashMap::new();
    
    for dim_id in 0..position_probs.dim(0)? {
        let dim_probs = position_probs.get(dim_id)?;
        let pos = sample_from_logits(
            &dim_probs.to_vec1()?,
            params.seed + dim_id as u64,
        );
        position.push(pos as f32);
        
        // Sample weight (sparsity: many dimensions = 0)
        let weight = sample_weight(&weight_logits, dim_id, params)?;
        if weight > 0.01 {  // Sparsity threshold
            weights.insert(dim_id, weight);
        }
    }
    
    Ok(SemanticCoordinate {
        position: Tensor::from_vec(position, position.len(), &Device::Cpu)?,
        dimensional_weights: weights,
        source_span: (0, 0),  // Generated, no source
        certainty: 1.0,
        composition: None,
    })
}
```

---

### Phase 2: Model Runner Integration (1-2 weeks)

**File**: `src/engine/model_runner.rs`

Extend to support both model types:

```rust
pub struct ModelRunner {
    /// Standard transformer models
    transformers: HashMap<String, Arc<Mutex<BatchedTransformer>>>,
    
    /// SCS coordinate-space models
    scs_models: HashMap<String, Arc<Mutex<ScsInferencePipeline>>>,
    
    /// Model metadata
    model_registry: HashMap<String, ModelMetadata>,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub model_type: ModelType,
    pub context_length: usize,
    pub loaded: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    StandardTransformer,
    ScsCoordinateSpace,
}

impl ModelRunner {
    /// Load model and detect type automatically
    pub async fn load_model(&mut self, model_name: &str, model_dir: &str) -> Result<()> {
        let model_type = detect_model_type_from_dir(model_dir)?;
        
        match model_type {
            ModelType::StandardTransformer => {
                let (model, _cache, _config, device, _mapper) = 
                    load_local_llama(model_dir, None, true, false)?;
                self.transformers.insert(model_name.to_string(), Arc::new(Mutex::new(model)));
            }
            ModelType::ScsCoordinateSpace => {
                let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
                let model = load_scs_model(model_dir, &device)?;
                let pipeline = ScsInferencePipeline::new(model, device)?;
                self.scs_models.insert(
                    model_name.to_string(),
                    Arc::new(Mutex::new(pipeline)),
                );
            }
        }
        
        Ok(())
    }
    
    /// Get SCS pipeline for coordinate generation
    pub async fn get_scs_pipeline(
        &self,
        model_name: &str,
    ) -> Result<tokio::sync::MutexGuard<ScsInferencePipeline>> {
        let pipeline = self.scs_models
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("SCS model not found: {}", model_name))?;
        
        Ok(pipeline.lock().await)
    }
}

/// Detect model type from directory structure
fn detect_model_type_from_dir(model_dir: &str) -> Result<ModelType> {
    let scs_config = Path::new(model_dir).join("scs_config.json");
    let llama_config = Path::new(model_dir).join("config.json");
    
    if scs_config.exists() {
        Ok(ModelType::ScsCoordinateSpace)
    } else if llama_config.exists() {
        Ok(ModelType::StandardTransformer)
    } else {
        anyhow::bail!("Unknown model type in {}", model_dir)
    }
}
```

---

### Phase 3: API Integration (1-2 weeks)

**File**: `src/api/openai/completions.rs`

Support SCS models in API endpoints:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    
    // Standard parameters
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    
    // SCS-specific parameters (optional)
    #[serde(default)]
    pub scs_params: Option<ScsGenerationParams>,
}

#[derive(Debug, serde::Deserialize, Default)]
pub struct ScsGenerationParams {
    /// Generate this many coordinates instead of tokens
    pub max_coordinates: Option<usize>,
    
    /// Which dimensions to emphasize (interpretability control)
    pub dimension_weights: Option<HashMap<String, f32>>,
    
    /// Force specific dimensional constraints
    pub dimension_constraints: Option<Vec<DimensionConstraint>>,
    
    /// Return coordinate sequence in response (for debugging)
    pub return_coordinates: bool,
}

#[derive(Debug, serde::Deserialize)]
pub struct DimensionConstraint {
    pub dimension_name: String,  // e.g., "temporal"
    pub target_value: f32,       // e.g., +1.0 for future
    pub strength: f32,           // how strongly to enforce (0-1)
}

/// Completion handler detects model type and routes appropriately
pub async fn handle_completion(
    req: CompletionRequest,
    model_runner: Arc<ModelRunner>,
) -> Result<CompletionResponse> {
    let model_type = model_runner.get_model_type(&req.model)?;
    
    match model_type {
        ModelType::StandardTransformer => {
            handle_token_completion(req, model_runner).await
        }
        ModelType::ScsCoordinateSpace => {
            handle_scs_completion(req, model_runner).await
        }
    }
}

async fn handle_scs_completion(
    req: CompletionRequest,
    model_runner: Arc<ModelRunner>,
) -> Result<CompletionResponse> {
    let scs_params = req.scs_params.unwrap_or_default();
    
    let mut pipeline = model_runner.get_scs_pipeline(&req.model).await?;
    
    // Apply dimensional constraints if specified
    if let Some(constraints) = scs_params.dimension_constraints {
        pipeline.apply_constraints(&constraints)?;
    }
    
    // Generate
    let max_coords = scs_params.max_coordinates
        .or(req.max_tokens)
        .unwrap_or(100);
    
    let output_text = pipeline.generate(&req.prompt, max_coords)?;
    
    // Optionally include coordinate sequence
    let coordinates = if scs_params.return_coordinates {
        Some(pipeline.get_last_coordinates()?)
    } else {
        None
    };
    
    Ok(CompletionResponse {
        text: output_text,
        coordinates,
        model: req.model,
        usage: compute_usage(&pipeline),
    })
}
```

**File**: `src/api/mod.rs`

Extend configuration:

```rust
#[derive(Debug, Clone)]
pub struct ApiConfig {
    // ... existing fields ...
    
    /// SCS-specific configuration
    pub scs_config: Option<ScsApiConfig>,
}

#[derive(Debug, Clone)]
pub struct ScsApiConfig {
    /// Enable SCS model support
    pub enabled: bool,
    
    /// Default cache tier sizes
    pub recent_cache_size: usize,
    pub mid_cache_size: usize,
    pub distant_cache_size: usize,
    
    /// Enable coordinate sequence logging (for research/debugging)
    pub log_coordinates: bool,
    
    /// Enable interpretability probing
    pub enable_dimension_probes: bool,
    
    /// Pattern cache size
    pub pattern_cache_size: usize,
}

impl Default for ScsApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            recent_cache_size: 512,
            mid_cache_size: 4096,
            distant_cache_size: 16384,
            log_coordinates: false,
            enable_dimension_probes: true,
            pattern_cache_size: 10000,
        }
    }
}
```

---

### Phase 4: Testing & Validation (1 week)

**New File**: `tests/scs_integration_tests.rs`

```rust
#[cfg(test)]
mod scs_tests {
    use super::*;
    
    #[test]
    fn test_load_scs_model() {
        // Test SCS model loading from directory
    }
    
    #[test]
    fn test_character_encoding() {
        // Test character → embedding conversion
    }
    
    #[test]
    fn test_certainty_chunking() {
        // Test variable-length chunking with certainty scores
    }
    
    #[test]
    fn test_coordinate_encoding() {
        // Test chunk → coordinate conversion
    }
    
    #[test]
    fn test_dimensional_attention() {
        // Test coordinate sequence attention
    }
    
    #[test]
    fn test_hierarchical_cache() {
        // Test cache tier transitions (recent → mid → distant)
    }
    
    #[test]
    fn test_coordinate_generation() {
        // Test next coordinate prediction
    }
    
    #[test]
    fn test_coordinate_decoding() {
        // Test coordinate → text reconstruction
    }
    
    #[test]
    fn test_end_to_end_generation() {
        // Test full prompt → generation pipeline
    }
    
    #[test]
    fn test_concurrent_transformer_and_scs() {
        // Verify both model types can run simultaneously
    }
    
    #[test]
    fn test_memory_isolation() {
        // Verify separate memory allocations
    }
}
```

---

## Configuration Example

**Example**: `models/scs-v1/scs_config.json`

```json
{
  "num_dimensions": 150,
  "positions_per_dimension": 1000,
  "dimension_labels": [
    "temporal", "causal", "spatial", "abstract_concrete",
    "emotional_valence", "formality", "agency", "certainty",
    "...additional learned dimensions..."
  ],
  "certainty_threshold": 0.5,
  "min_chunk_size": 3,
  "max_chunk_size": 50,
  "use_hierarchical_cache": true,
  "cache_tiers": [
    {
      "name": "recent",
      "capacity": 512,
      "precision": "full"
    },
    {
      "name": "mid_range",
      "capacity": 4096,
      "precision": "quantized_8bit"
    },
    {
      "name": "distant",
      "capacity": 16384,
      "precision": "summarized"
    }
  ]
}
```

---

## Feature Flag

Add optional compilation:

**File**: `Cargo.toml`

```toml
[features]
default = []
scs = []  # Only compile SCS support if enabled
```

**File**: `src/lib.rs`

```rust
#[cfg(feature = "scs")]
pub mod scs_pipeline;

#[cfg(feature = "scs")]
pub mod hierarchical_scs_cache;

#[cfg(feature = "scs")]
pub use scs_pipeline::ScsInferencePipeline;
```

Usage:

```bash
# Build with SCS support
cargo build --features scs

# Build without SCS (default, no binary bloat)
cargo build
```

---

## Memory Comparison

**Example**: 24GB GPU

### Scenario 1: Transformer Only
- Llama-7B: 6GB weights + 1GB KV cache = 7GB
- Available: 17GB for batching

### Scenario 2: SCS Only
- SCS-v1: 2GB weights + 0.2GB coord cache = 2.2GB
- Available: 21.8GB for batching

### Scenario 3: Both Models
- Llama-7B: 7GB
- SCS-v1: 2.2GB
- Total: 9.2GB used, 14.8GB free
- **Both run independently without interference**

---

## Performance Characteristics

### Expected Throughput (single GPU)

| Model Type | Tokens/Second | Coordinates/Second | Notes |
|------------|---------------|-------------------|-------|
| Transformer alone | 50 | - | Baseline |
| SCS alone | ~100 tok equiv | 20 | Chunks = multiple tokens |
| Both concurrent | 45 (-10%) | 18 (-10%) | Minor GPU contention |

### Latency

| Model Type | Per-Token Latency | Per-Coordinate Latency |
|------------|------------------|----------------------|
| Transformer alone | 100ms | - |
| SCS alone | - | 50ms |
| Both concurrent | 105ms (+5%) | 52ms (+4%) |

**Mitigation for zero contention**: Use separate GPUs

---

## API Usage Examples

### Standard Transformer Request

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### SCS Model Request

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "scs-v1",
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "scs_params": {
      "max_coordinates": 50,
      "dimension_constraints": [
        {
          "dimension_name": "temporal",
          "target_value": 0.0,
          "strength": 0.3
        }
      ],
      "return_coordinates": false
    }
  }'
```

### Concurrent Requests (Both Models)

```bash
# Request 1: Transformer
curl http://localhost:8080/v1/completions \
  -d '{"model": "llama-7b", "prompt": "What is AI?"}' &

# Request 2: SCS (runs in parallel)
curl http://localhost:8080/v1/completions \
  -d '{"model": "scs-v1", "prompt": "Temporal reasoning task"}' &
```

---

## Dependencies

### New Crates

None required—SCS uses existing Candle primitives (`Tensor`, `Device`, `Module`).

### Candle Compatibility

SCS models use the same Candle version as transformers. No version conflicts.

---

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1**: Core Infrastructure | 4-6 weeks | Model loader, representations, pipeline, cache, sampling |
| **Phase 2**: Model Runner | 1-2 weeks | Model type detection, dual-model management |
| **Phase 3**: API Integration | 1-2 weeks | API handlers, configuration, SCS-specific params |
| **Phase 4**: Testing | 1 week | Unit tests, integration tests, concurrent tests |
| **Total** | **7-11 weeks** | Full SCS support |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **SCS models don't exist yet** | High | Wait for SCS training framework completion |
| **Performance worse than expected** | Medium | Feature flag allows disabling; separate compilation |
| **Memory overhead** | Low | Hierarchical cache designed for efficiency |
| **API complexity** | Low | SCS params are optional; backward compatible |
| **Maintenance burden** | Medium | Clear separation; SCS is additive, not invasive |

---

## Success Criteria

✅ **Functional**:
- Load SCS model from directory
- Generate text from SCS model
- Concurrent transformer + SCS requests work

✅ **Performance**:
- SCS memory usage ≤ 3GB for typical model
- Concurrent operation <15% throughput loss vs separate execution
- Coordinate cache compression achieves >5× memory reduction vs naive storage

✅ **Quality**:
- SCS reconstruction quality matches training metrics
- Dimensional constraints work as expected
- No regressions in transformer functionality

---

## Future Enhancements (Post-Initial Implementation)

### Interpretability Tools
- Dimension probe framework for analyzing learned dimensions
- Coordinate visualization for debugging
- Dimensional correlation analysis

### Optimization
- CUDA kernels for coordinate operations
- Sparse coordinate storage optimizations
- Pattern cache preloading from corpus analysis

### Hybrid Models
- Ensemble generation (coordinate + token voting)
- Coordinate-guided token generation
- Cross-architecture knowledge distillation

---

## References

- **SCS Paper**: `semantic_coordinates_paper.md` (research proposal)
- **Lightbulb Architecture**: `ROADMAP.md`, `docs/` (existing documentation)
- **Multi-Model Patterns**: `src/engine/model_runner.rs` (current multi-model support)
- **Cache Architecture**: `src/cache/` (KV cache implementations)
- **API Design**: `src/api/` (OpenAI-compatible endpoints)

---

## Conclusion

SCS model support can be added to Lightbulb as a **fully additive feature** with:
- **Zero breaking changes** to existing transformer functionality
- **Clean separation** via model type detection and routing
- **Optional compilation** via feature flags
- **Independent memory allocation** and processing pipelines

Implementation requires ~2-3 months of development time, primarily building new components rather than modifying existing code. The architecture naturally supports concurrent execution of both model types, making Lightbulb capable of serving diverse AI workloads from a single server.
