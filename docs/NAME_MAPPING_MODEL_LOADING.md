# Dynamic Name Mapping Integration: Model Loading

**Status**: PLANNED  
**Priority**: HIGH  
**Dependencies**: `name_mapping` module (M3.7 COMPLETE)  
**Target Milestone**: M4.1 (Early M4)

## Executive Summary

Integrate dynamic tensor name mapping into Lightbulb's model loading infrastructure to enable automatic support for new model architectures without code changes. This eliminates hardcoded tensor name assumptions and enables Lightbulb to load arbitrary GGUF/safetensors models with minimal configuration.

## Current State (Problems to Solve)

**src/model/loader.rs** currently:
- Hardcodes tensor names for specific architectures (LLaMA, GPT-2, etc.)
- Requires code changes to support new model families
- Fails silently or crashes on unknown architectures
- Duplicates architecture detection logic across modules

**Example Current Code**:
```rust
// Hardcoded assumptions
let q_proj = model.get(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))?;
let k_proj = model.get(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))?;
```

**Problems**:
- Mistral uses `"layers.{}.self_attn.q_proj.weight"` (no "model." prefix)
- Qwen uses `"transformer.h.{}.attn.c_attn.weight"` (fused QKV)
- TinyLlama uses `"blk.{}.attn_q.weight"` (GGUF naming)
- Each requires separate loader code

## Proposed Architecture

### Phase 1: Core Integration (M4.1)

**1. Add NameMapper to ModelLoader**

```rust
// src/model/loader.rs
use crate::pruning::name_mapping::TensorNameMapper;

pub struct ModelLoader {
    name_mapper: Option<TensorNameMapper>,
    // ... existing fields
}

impl ModelLoader {
    pub fn new(model_path: &Path) -> Result<Self> {
        // Load model metadata
        let tensor_names = Self::extract_tensor_names(model_path)?;
        
        // Auto-detect architecture
        let name_mapper = TensorNameMapper::from_tensor_names(&tensor_names)?;
        
        Ok(Self {
            name_mapper: Some(name_mapper),
            // ...
        })
    }
    
    /// Get tensor by abstract name (architecture-agnostic)
    pub fn get_tensor(&self, layer_idx: usize, component: &str) -> Result<Tensor> {
        let abstract_name = format!("layer_{}.{}", layer_idx, component);
        
        if let Some(mapper) = &self.name_mapper {
            if let Some(concrete_name) = mapper.map_name(&abstract_name) {
                return self.load_tensor_by_name(&concrete_name);
            }
        }
        
        // Fallback to old behavior for compatibility
        self.load_tensor_legacy(layer_idx, component)
    }
}
```

**2. Standardize Abstract Component Names**

```rust
// src/model/components.rs (NEW FILE)
pub enum ModelComponent {
    // Attention
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    
    // FFN
    FfnGate,
    FfnUp,
    FfnDown,
    
    // Normalization
    AttentionNorm,
    FfnNorm,
    
    // Embeddings
    TokenEmbedding,
    PositionEmbedding,
    OutputProjection,
}

impl ModelComponent {
    pub fn to_abstract_name(&self, layer_idx: usize) -> String {
        match self {
            Self::AttentionQuery => format!("layer_{}.attention.query", layer_idx),
            Self::AttentionKey => format!("layer_{}.attention.key", layer_idx),
            // ... etc
        }
    }
}
```

**3. Update Model Initialization**

```rust
// src/model/llama.rs (and other model files)
impl LlamaModel {
    pub fn load(loader: &ModelLoader, config: &ModelConfig) -> Result<Self> {
        let mut layers = Vec::new();
        
        for layer_idx in 0..config.num_layers {
            // Use abstract component names - mapper handles architecture
            let q_weight = loader.get_tensor(layer_idx, "attention.query")?;
            let k_weight = loader.get_tensor(layer_idx, "attention.key")?;
            let v_weight = loader.get_tensor(layer_idx, "attention.value")?;
            
            layers.push(TransformerBlock::new(q_weight, k_weight, v_weight)?);
        }
        
        Ok(Self { layers })
    }
}
```

### Phase 2: Enhanced Features (M4.2)

**1. Batch Tensor Loading**

```rust
impl ModelLoader {
    /// Load all tensors for a layer at once
    pub fn get_layer_tensors(&self, layer_idx: usize) -> Result<LayerTensors> {
        if let Some(mapper) = &self.name_mapper {
            let tensor_names = mapper.get_layer_tensors(layer_idx);
            
            let mut tensors = HashMap::new();
            for (component, name) in mapper.map_layer(layer_idx) {
                tensors.insert(component, self.load_tensor_by_name(&name)?);
            }
            
            return Ok(LayerTensors::from_map(tensors)?);
        }
        
        self.get_layer_tensors_legacy(layer_idx)
    }
}
```

**2. Validation and Error Reporting**

```rust
impl ModelLoader {
    /// Validate that all required tensors are present
    pub fn validate_architecture(&self) -> Result<ArchitectureReport> {
        let mapper = self.name_mapper.as_ref()
            .context("No name mapper available")?;
        
        let mut report = ArchitectureReport::new(mapper.architecture);
        
        // Check each layer has required components
        for layer_idx in mapper.layer_indices.iter() {
            for component in REQUIRED_COMPONENTS {
                let abstract_name = format!("layer_{}.{}", layer_idx, component);
                
                if mapper.map_name(&abstract_name).is_none() {
                    report.add_missing(*layer_idx, component);
                }
            }
        }
        
        if !report.is_valid() {
            anyhow::bail!("Model validation failed:\n{}", report);
        }
        
        Ok(report)
    }
}
```

**3. Metadata Extraction**

```rust
impl ModelLoader {
    /// Extract model configuration from architecture
    pub fn infer_config(&self) -> Result<ModelConfig> {
        let mapper = self.name_mapper.as_ref()
            .context("No name mapper available")?;
        
        // Get first layer to determine dimensions
        let layer_0_tensors = self.get_layer_tensors(0)?;
        
        let hidden_size = layer_0_tensors.attention_query.dim(0)?;
        let num_heads = self.infer_num_heads(&layer_0_tensors)?;
        let num_layers = mapper.layer_indices.len();
        
        Ok(ModelConfig {
            hidden_size,
            num_heads,
            num_layers,
            architecture: mapper.architecture,
            // ... more fields
        })
    }
}
```

### Phase 3: Advanced Features (M4.3)

**1. Config-Driven Overrides**

```yaml
# model_config.yaml
architecture: auto  # or "llama", "gpt", "mistral"
name_mapping:
  overrides:
    # Custom mappings for non-standard architectures
    "layer_0.attention.query": "custom.q_proj.0"
    "layer_0.attention.key": "custom.k_proj.0"
  
  # Fallback strategy if mapping fails
  fallback: error  # or "warn", "ignore"
```

**2. Architecture Registry**

```rust
// src/model/registry.rs (NEW)
pub struct ArchitectureRegistry {
    patterns: HashMap<ModelArchitecture, NameMappingPatterns>,
}

impl ArchitectureRegistry {
    /// Register custom architecture pattern
    pub fn register(&mut self, arch: ModelArchitecture, patterns: NameMappingPatterns) {
        self.patterns.insert(arch, patterns);
    }
    
    /// Extend name mapper with registry patterns
    pub fn enhance_mapper(&self, mapper: &mut TensorNameMapper) {
        if let Some(patterns) = self.patterns.get(&mapper.architecture) {
            mapper.add_patterns(patterns);
        }
    }
}
```

## Integration Points

### 1. GGUF Loader (`src/gguf/mod.rs`)
- Extract tensor names from GGUF metadata
- Pass to TensorNameMapper for architecture detection
- Use mapped names for tensor access

### 2. Safetensors Loader (if implemented)
- Similar pattern to GGUF
- Support PyTorch checkpoint naming conventions

### 3. Model Manager (`src/model/manager.rs`)
- Cache name mappers per loaded model
- Share mappers across requests for same model

### 4. Configuration System
- Add `model.architecture` config field
- Support manual architecture override
- Document supported architectures

## Benefits

**For Users**:
- Load any GGUF/safetensors model without code changes
- Clear error messages when tensors are missing
- Automatic architecture detection

**For Developers**:
- Single source of truth for tensor naming
- Easy to add new architectures (just add regex patterns)
- Reduced code duplication

**For Performance**:
- Name mapping happens once at load time (negligible overhead)
- No runtime string manipulation
- Cached lookups for repeated access

## Testing Strategy

**Unit Tests**:
```rust
#[test]
fn test_llama_model_loading() {
    let loader = ModelLoader::new("models/llama-7b.gguf").unwrap();
    assert_eq!(loader.name_mapper.architecture, ModelArchitecture::LLaMA);
    
    let q_proj = loader.get_tensor(0, "attention.query").unwrap();
    assert_eq!(q_proj.dims(), &[4096, 4096]);
}

#[test]
fn test_unknown_architecture_fallback() {
    let loader = ModelLoader::new("models/custom-model.gguf").unwrap();
    // Should detect as Unknown but still work with manual config
}
```

**Integration Tests**:
- Load 5+ different model families (LLaMA, GPT, Mistral, Qwen, Phi)
- Verify inference correctness after loading
- Test partial architecture support (missing optional components)

## Implementation Plan

### Week 1: Core Integration
- [ ] Add `TensorNameMapper` to `ModelLoader`
- [ ] Create `ModelComponent` enum
- [ ] Update `LlamaModel` to use abstract names
- [ ] Add unit tests

### Week 2: Enhanced Features
- [ ] Implement batch tensor loading
- [ ] Add validation and error reporting
- [ ] Add config inference from architecture
- [ ] Update GGUF loader

### Week 3: Polish & Documentation
- [ ] Add config-driven overrides
- [ ] Create architecture registry
- [ ] Write comprehensive docs
- [ ] Add integration tests for 5+ architectures

### Week 4: Validation & Hardening
- [ ] Test on real models (LLaMA, Mistral, Qwen, Phi, GPT)
- [ ] Performance profiling
- [ ] Edge case handling
- [ ] User-facing documentation

## Success Metrics

- ✅ Load 5+ different architectures with same code
- ✅ <1ms overhead for name mapping at load time
- ✅ Clear error messages for missing tensors
- ✅ Zero code changes needed for new GGUF models
- ✅ 100% backward compatibility with existing loaders

## Future Extensions (Post-M4)

1. **LLM-Assisted Mapping** (M6+):
   - Use small LLM to suggest mappings for unknown architectures
   - Probabilistic matching with confidence scores

2. **Community Architecture Database**:
   - Crowdsourced architecture patterns
   - Automatic updates via plugin system

3. **Multi-File Models**:
   - Support sharded models across multiple files
   - Coordinate name mapping across shards

## References

- `src/pruning/name_mapping.rs` - Core name mapping implementation
- `docs/M3_7_NAME_MAPPING.md` - Name mapping architecture (to be created)
- `src/model/loader.rs` - Current model loading infrastructure
- `src/gguf/mod.rs` - GGUF file format support
