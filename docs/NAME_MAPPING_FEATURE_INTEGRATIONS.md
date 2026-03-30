# Name Mapping Integration Plans for Lightbulb Features

**Purpose**: Comprehensive integration plans for extending dynamic name mapping to all Lightbulb modules  
**Status**: PLANNING  
**Dependencies**: M3.7 (Core name mapping), M4.1 (Model loading integration)

---

## Overview

The dynamic name mapping system (M3.7) enables Lightbulb to automatically adapt to new model architectures without code changes. This document outlines integration plans for 5 key features that will benefit from architecture-aware tensor name resolution.

---

## 1. Cache Management Integration (M5.1)

### Current State

**File**: `src/cache/mod.rs` (estimated)

**Problems**:
- KV cache assumes fixed layer indexing (layer 0-31)
- Layer-specific eviction policies hardcoded for specific architectures
- Cannot handle variable layer counts or non-standard numbering
- Relationship-aware cache (N-dimensional token graphs) needs layer boundary detection

**Example Issue**:
```rust
// Hardcoded layer assumptions
for layer_idx in 0..32 {  // What if model has 40 layers? Or 24?
    let kv_cache = self.layer_caches.get_mut(&layer_idx).unwrap();
    // ...
}
```

### Proposed Architecture

**Phase 1: Basic Integration (M5.1)**

Add name mapper to cache manager for layer detection:

```rust
use crate::pruning::name_mapping::{TensorNameMapper, ComponentType};

pub struct CacheManager {
    /// Name mapper for architecture-aware operations
    name_mapper: TensorNameMapper,
    
    /// Layer-specific KV caches
    layer_caches: HashMap<usize, LayerCache>,
    
    /// Relationship graph for token dependencies
    token_graph: TokenRelationshipGraph,
}

impl CacheManager {
    pub fn new(model_tensors: &[String]) -> Result<Self> {
        let name_mapper = TensorNameMapper::from_tensor_names(model_tensors)?;
        
        // Auto-detect layer count
        let num_layers = name_mapper.layer_indices.len();
        
        // Initialize cache for each layer
        let mut layer_caches = HashMap::new();
        for layer_idx in &name_mapper.layer_indices {
            layer_caches.insert(*layer_idx, LayerCache::new());
        }
        
        Ok(Self {
            name_mapper,
            layer_caches,
            token_graph: TokenRelationshipGraph::new(num_layers),
        })
    }
    
    /// Get layer index for a tensor name
    pub fn get_layer_index(&self, tensor_name: &str) -> Option<usize> {
        self.name_mapper.parse_tensor_layer(tensor_name)
    }
    
    /// Apply layer-specific eviction strategy
    pub fn evict_layer(&mut self, layer_idx: usize, strategy: EvictionStrategy) {
        if let Some(cache) = self.layer_caches.get_mut(&layer_idx) {
            match strategy {
                EvictionStrategy::Lru => cache.evict_lru(),
                EvictionStrategy::RelationshipAware => {
                    // Use token graph to identify low-importance tokens
                    let evict_tokens = self.token_graph.get_evictable_tokens(layer_idx);
                    cache.evict_specific(&evict_tokens);
                }
            }
        }
    }
}

pub enum EvictionStrategy {
    Lru,
    RelationshipAware,
}
```

**Phase 2: Relationship-Aware Cache (M5.2)**

Use name mapping to identify layer boundaries in token relationship graph:

```rust
pub struct TokenRelationshipGraph {
    /// N-dimensional graph of token dependencies
    /// Nodes: (batch_idx, seq_idx, layer_idx, token_id)
    nodes: Vec<TokenNode>,
    
    /// Edges: attention relationships between tokens
    edges: Vec<TokenEdge>,
    
    /// Name mapper for layer-aware operations
    name_mapper: Arc<TensorNameMapper>,
}

impl TokenRelationshipGraph {
    /// Identify tokens with low importance in specific layer
    pub fn get_evictable_tokens(&self, layer_idx: usize) -> Vec<TokenId> {
        let layer_nodes = self.nodes.iter()
            .filter(|n| n.layer_idx == layer_idx)
            .collect::<Vec<_>>();
        
        // Find tokens with few outgoing edges (low downstream impact)
        layer_nodes.iter()
            .filter(|n| self.out_degree(n.id) < 2)
            .map(|n| n.token_id)
            .collect()
    }
    
    /// Cross-layer dependency analysis
    pub fn analyze_cross_layer_dependencies(&self) -> Vec<LayerDependency> {
        let mut dependencies = Vec::new();
        
        for layer_idx in &self.name_mapper.layer_indices {
            let downstream_impact = self.compute_downstream_impact(*layer_idx);
            dependencies.push(LayerDependency {
                layer: *layer_idx,
                impact_score: downstream_impact,
            });
        }
        
        dependencies
    }
}

struct TokenNode {
    id: NodeId,
    batch_idx: usize,
    seq_idx: usize,
    layer_idx: usize,
    token_id: TokenId,
}
```

**Phase 3: Dynamic Cache Sizing (M5.3)**

Architecture-aware cache allocation:

```rust
impl CacheManager {
    /// Allocate cache sizes based on model architecture
    pub fn allocate_optimal_cache(&mut self) -> Result<()> {
        let architecture = self.name_mapper.architecture;
        
        match architecture {
            ModelArchitecture::LLaMA => {
                // LLaMA has uniform layer sizes
                let size_per_layer = self.total_cache_budget / self.layer_caches.len();
                for cache in self.layer_caches.values_mut() {
                    cache.resize(size_per_layer)?;
                }
            }
            ModelArchitecture::Mistral => {
                // Mistral has attention sinks in early layers
                for (layer_idx, cache) in &mut self.layer_caches {
                    let size = if *layer_idx < 4 {
                        // Early layers get 2x cache (attention sinks)
                        self.total_cache_budget / (self.layer_caches.len() * 2)
                    } else {
                        self.total_cache_budget / self.layer_caches.len()
                    };
                    cache.resize(size)?;
                }
            }
            _ => {
                // Uniform allocation for unknown architectures
                let size_per_layer = self.total_cache_budget / self.layer_caches.len();
                for cache in self.layer_caches.values_mut() {
                    cache.resize(size_per_layer)?;
                }
            }
        }
        
        Ok(())
    }
}
```

### Integration Points

- **Model Loader** (M4.1): Pass tensor names to cache manager
- **Inference Engine**: Use cache manager's layer detection
- **Memory Manager**: Coordinate cache sizing with model memory
- **Configuration**: Allow per-architecture cache policies

### Testing Strategy

- Unit: Test layer detection for LLaMA, GPT, Mistral
- Integration: Verify cache hit rates with relationship-aware eviction
- Performance: Measure overhead (<5% of cache operations)
- Stress: Test with variable layer counts (24, 32, 40, 80)

### Success Metrics

- ✅ Support variable layer counts automatically
- ✅ Relationship-aware eviction improves cache hit rate by 10-20%
- ✅ <5% overhead for layer detection
- ✅ Works with unknown architectures (generic fallback)

---

## 2. LoRA Integration (M5.4)

### Current State

**Problem**: LoRA adapters have varied naming conventions across sources:
- HuggingFace: `base_model.model.layers.0.self_attn.q_proj.lora_A`
- Custom: `lora.blk.0.attn_q.A`
- PEFT format: `peft.q_proj.adapter_A`

Hardcoded matching logic breaks with new formats.

### Proposed Architecture

**Phase 1: LoRA Name Mapping (M5.4)**

```rust
use crate::pruning::name_mapping::TensorNameMapper;

pub struct LoraAdapter {
    /// Base model name mapper
    base_mapper: TensorNameMapper,
    
    /// LoRA weights (abstract names → matrices)
    adapters: HashMap<String, LoraWeight>,
    
    /// LoRA format (HF, custom, PEFT)
    format: LoraFormat,
}

impl LoraAdapter {
    /// Load LoRA adapter with automatic format detection
    pub fn load(
        adapter_path: &Path,
        base_model_tensors: &[String],
    ) -> Result<Self> {
        let base_mapper = TensorNameMapper::from_tensor_names(base_model_tensors)?;
        
        // Load LoRA tensors
        let lora_tensors = Self::load_safetensors(adapter_path)?;
        
        // Detect LoRA format
        let format = Self::detect_lora_format(&lora_tensors)?;
        
        // Map LoRA tensor names to base model components
        let mut adapters = HashMap::new();
        for (lora_name, tensor) in lora_tensors {
            if let Some(base_component) = Self::map_lora_to_base(
                &lora_name,
                &base_mapper,
                format,
            ) {
                adapters.insert(base_component, LoraWeight {
                    A: tensor.A,
                    B: tensor.B,
                    alpha: tensor.alpha,
                });
            }
        }
        
        Ok(Self {
            base_mapper,
            adapters,
            format,
        })
    }
    
    /// Map LoRA tensor name to base model component
    fn map_lora_to_base(
        lora_name: &str,
        base_mapper: &TensorNameMapper,
        format: LoraFormat,
    ) -> Option<String> {
        match format {
            LoraFormat::HuggingFace => {
                // "base_model.model.layers.5.self_attn.q_proj.lora_A"
                // -> "layer_5.attention.query"
                let parts: Vec<&str> = lora_name.split('.').collect();
                let layer_idx = parts.iter()
                    .find_map(|p| p.strip_prefix("layers"))
                    .and_then(|_| parts.iter().find_map(|p| p.parse::<usize>().ok()))?;
                
                let component = if lora_name.contains("q_proj") {
                    "attention.query"
                } else if lora_name.contains("k_proj") {
                    "attention.key"
                } else if lora_name.contains("v_proj") {
                    "attention.value"
                } else {
                    return None;
                };
                
                let abstract_name = format!("layer_{}.{}", layer_idx, component);
                base_mapper.map_name(&abstract_name)
            }
            LoraFormat::Custom => {
                // "lora.blk.0.attn_q.A" -> already matches base model format
                let concrete_name = lora_name
                    .strip_prefix("lora.")?
                    .strip_suffix(".A")?
                    .to_string();
                Some(concrete_name)
            }
            LoraFormat::Peft => {
                // "peft.q_proj.adapter_A" -> need layer context
                // This is tricky - may need LLM assistance
                None
            }
        }
    }
}

pub enum LoraFormat {
    HuggingFace,
    Custom,
    Peft,
}

struct LoraWeight {
    A: Tensor,
    B: Tensor,
    alpha: f32,
}
```

**Phase 2: Validation and Merging (M5.5)**

```rust
impl LoraAdapter {
    /// Validate LoRA is compatible with base model
    pub fn validate(&self, base_model: &Model) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();
        
        for (component, lora_weight) in &self.adapters {
            // Check if component exists in base model
            if let Some(base_tensor) = base_model.get_tensor(component) {
                // Check shape compatibility
                if Self::is_shape_compatible(&base_tensor, lora_weight) {
                    report.compatible.push(component.clone());
                } else {
                    report.shape_mismatch.push(component.clone());
                }
            } else {
                report.missing_in_base.push(component.clone());
            }
        }
        
        if !report.shape_mismatch.is_empty() || !report.missing_in_base.is_empty() {
            bail!("LoRA adapter incompatible with base model: {:?}", report);
        }
        
        Ok(report)
    }
    
    /// Merge LoRA weights into base model
    pub fn merge_into(&self, base_model: &mut Model, scale: f32) -> Result<()> {
        for (component, lora_weight) in &self.adapters {
            let base_tensor = base_model.get_tensor_mut(component)?;
            
            // W' = W + (B @ A) * (alpha / r) * scale
            let delta = lora_weight.B.matmul(&lora_weight.A)?;
            let scaled_delta = delta * (lora_weight.alpha / lora_weight.B.dim(1) as f32) * scale;
            
            *base_tensor = base_tensor.add(&scaled_delta)?;
        }
        
        Ok(())
    }
}

#[derive(Default)]
struct ValidationReport {
    compatible: Vec<String>,
    shape_mismatch: Vec<String>,
    missing_in_base: Vec<String>,
}
```

### Integration Points

- **Model Loader** (M4.1): Pass base model tensor names to LoRA loader
- **Inference**: Apply LoRA on-the-fly or merge at load time
- **Configuration**: Support multiple LoRA adapters simultaneously

### Testing Strategy

- Unit: Test format detection (HF, custom, PEFT)
- Integration: Load LoRA from 3+ sources, verify mapping correctness
- Validation: Ensure shape compatibility checks work
- Inference: Verify output correctness after LoRA merging

### Success Metrics

- ✅ Support 3+ LoRA formats automatically
- ✅ Detect and report incompatibilities clearly
- ✅ 100% mapping accuracy for common formats
- ✅ <50ms overhead for LoRA loading

---

## 3. Multi-GPU Sharding Integration (M3.6 Enhancement)

### Current State

**File**: `src/distribution/mod.rs` (already implemented in M3.6)

**Current Capability**: 
- Basic layer distribution across GPUs
- Assumes uniform layer sizes
- Hardcoded layer count (32 for LLaMA-7B)

**Limitation**:
Cannot handle heterogeneous architectures (MoE, variable layer sizes, non-standard numbering).

### Proposed Enhancement

**Phase 1: Architecture-Aware Sharding (M3.6.1)**

```rust
use crate::pruning::name_mapping::{TensorNameMapper, ComponentType};

pub struct MultiGpuSharding {
    /// Name mapper for layer detection
    name_mapper: TensorNameMapper,
    
    /// GPU assignments (layer_idx → GPU id)
    assignments: HashMap<usize, usize>,
    
    /// Number of available GPUs
    num_gpus: usize,
}

impl MultiGpuSharding {
    pub fn new(model_tensors: &[String], num_gpus: usize) -> Result<Self> {
        let name_mapper = TensorNameMapper::from_tensor_names(model_tensors)?;
        
        // Auto-detect optimal sharding strategy
        let assignments = Self::compute_assignments(&name_mapper, num_gpus)?;
        
        Ok(Self {
            name_mapper,
            assignments,
            num_gpus,
        })
    }
    
    /// Compute optimal layer-to-GPU assignments
    fn compute_assignments(
        mapper: &TensorNameMapper,
        num_gpus: usize,
    ) -> Result<HashMap<usize, usize>> {
        let num_layers = mapper.layer_indices.len();
        let layers_per_gpu = (num_layers + num_gpus - 1) / num_gpus;
        
        let mut assignments = HashMap::new();
        for (idx, layer_idx) in mapper.layer_indices.iter().enumerate() {
            let gpu_id = idx / layers_per_gpu;
            assignments.insert(*layer_idx, gpu_id.min(num_gpus - 1));
        }
        
        Ok(assignments)
    }
    
    /// Get GPU assignment for tensor
    pub fn get_gpu_for_tensor(&self, tensor_name: &str) -> Option<usize> {
        let layer_idx = self.name_mapper.parse_tensor_layer(tensor_name)?;
        self.assignments.get(&layer_idx).copied()
    }
}
```

**Phase 2: Heterogeneous Sharding (M3.6.2)**

Handle MoE models with variable layer sizes:

```rust
impl MultiGpuSharding {
    /// Compute load-balanced assignments based on tensor sizes
    fn compute_balanced_assignments(
        mapper: &TensorNameMapper,
        tensor_sizes: &HashMap<String, usize>,
        num_gpus: usize,
    ) -> Result<HashMap<usize, usize>> {
        // Calculate size per layer
        let mut layer_sizes: HashMap<usize, usize> = HashMap::new();
        for layer_idx in &mapper.layer_indices {
            let tensors = mapper.get_layer_tensors(*layer_idx);
            let total_size: usize = tensors.iter()
                .filter_map(|name| tensor_sizes.get(name))
                .sum();
            layer_sizes.insert(*layer_idx, total_size);
        }
        
        // Greedy assignment: assign each layer to least-loaded GPU
        let mut gpu_loads = vec![0usize; num_gpus];
        let mut assignments = HashMap::new();
        
        // Sort layers by size (largest first for better balance)
        let mut sorted_layers: Vec<_> = layer_sizes.iter().collect();
        sorted_layers.sort_by_key(|(_, size)| std::cmp::Reverse(**size));
        
        for (layer_idx, size) in sorted_layers {
            // Find GPU with minimum load
            let gpu_id = gpu_loads.iter()
                .enumerate()
                .min_by_key(|(_, load)| *load)
                .map(|(id, _)| id)
                .unwrap();
            
            assignments.insert(*layer_idx, gpu_id);
            gpu_loads[gpu_id] += size;
        }
        
        Ok(assignments)
    }
}
```

### Integration Points

- **Model Loader** (M4.1): Provide tensor sizes for load balancing
- **Inference Engine**: Route computations to correct GPU
- **Memory Manager**: Track per-GPU memory usage

### Testing Strategy

- Unit: Test assignment algorithms with variable layer counts
- Integration: Run inference on 2-4 GPUs with sharding
- Performance: Verify balanced GPU utilization
- Stress: Test with MoE models (heterogeneous layer sizes)

### Success Metrics

- ✅ Support variable layer counts automatically
- ✅ Balanced GPU utilization (max deviation <10%)
- ✅ Works with MoE models
- ✅ <1ms overhead for routing decisions

---

## 4. Quantization Integration (M4 - AWQ/SmoothQuant)

### Current State

**Planned Feature**: M4 will add AWQ and SmoothQuant support

**Need**: Layer-specific quantization levels based on sensitivity analysis

### Proposed Architecture

**Phase 1: Architecture-Aware Quantization (M4.1)**

```rust
use crate::pruning::name_mapping::{TensorNameMapper, ComponentType};

pub struct QuantizationConfig {
    /// Name mapper for component detection
    name_mapper: TensorNameMapper,
    
    /// Per-component quantization levels
    component_configs: HashMap<ComponentType, QuantLevel>,
    
    /// Layer-specific overrides
    layer_overrides: HashMap<usize, QuantLevel>,
}

impl QuantizationConfig {
    pub fn new_mixed_precision(
        model_tensors: &[String],
        target_bits: f32,
    ) -> Result<Self> {
        let name_mapper = TensorNameMapper::from_tensor_names(model_tensors)?;
        
        // Default: attention gets higher precision than FFN
        let mut component_configs = HashMap::new();
        component_configs.insert(ComponentType::AttentionQuery, QuantLevel::Int8);
        component_configs.insert(ComponentType::AttentionKey, QuantLevel::Int8);
        component_configs.insert(ComponentType::AttentionValue, QuantLevel::Int8);
        component_configs.insert(ComponentType::AttentionOutput, QuantLevel::Int8);
        component_configs.insert(ComponentType::FfnGate, QuantLevel::Int4);
        component_configs.insert(ComponentType::FfnUp, QuantLevel::Int4);
        component_configs.insert(ComponentType::FfnDown, QuantLevel::Int4);
        
        Ok(Self {
            name_mapper,
            component_configs,
            layer_overrides: HashMap::new(),
        })
    }
    
    /// Get quantization level for specific tensor
    pub fn get_quant_level(&self, tensor_name: &str) -> QuantLevel {
        // Check layer-specific override first
        if let Some(layer_idx) = self.name_mapper.parse_tensor_layer(tensor_name) {
            if let Some(level) = self.layer_overrides.get(&layer_idx) {
                return *level;
            }
        }
        
        // Fall back to component-level config
        if let Some((_, component)) = self.name_mapper.parse_component(tensor_name) {
            if let Some(level) = self.component_configs.get(&component) {
                return *level;
            }
        }
        
        // Default
        QuantLevel::Int8
    }
}

#[derive(Copy, Clone)]
pub enum QuantLevel {
    Float16,
    Int8,
    Int4,
    Int2,
}
```

**Phase 2: Sensitivity-Based Quantization (M4.2)**

Automatically determine optimal quantization per layer:

```rust
impl QuantizationConfig {
    /// Analyze layer sensitivity and assign quantization levels
    pub fn from_sensitivity_analysis(
        model: &Model,
        calibration_data: &[Tensor],
    ) -> Result<Self> {
        let name_mapper = TensorNameMapper::from_tensor_names(
            &model.tensor_names()
        )?;
        
        // Run calibration forward pass
        let mut layer_sensitivities = HashMap::new();
        for layer_idx in &name_mapper.layer_indices {
            let sensitivity = Self::measure_layer_sensitivity(
                model,
                *layer_idx,
                calibration_data,
            )?;
            layer_sensitivities.insert(*layer_idx, sensitivity);
        }
        
        // Assign quantization levels based on sensitivity
        let mut layer_overrides = HashMap::new();
        for (layer_idx, sensitivity) in layer_sensitivities {
            let level = if sensitivity > 0.8 {
                QuantLevel::Float16  // High sensitivity = keep FP16
            } else if sensitivity > 0.5 {
                QuantLevel::Int8     // Medium = INT8
            } else {
                QuantLevel::Int4     // Low = INT4
            };
            layer_overrides.insert(layer_idx, level);
        }
        
        Ok(Self {
            name_mapper,
            component_configs: HashMap::new(),
            layer_overrides,
        })
    }
}
```

### Integration Points

- **Model Loader** (M4.1): Apply quantization during load
- **Inference**: Use appropriate kernels per quantization level
- **Configuration**: Support per-model quantization profiles

### Testing Strategy

- Unit: Test quantization level assignment
- Integration: Verify mixed precision works correctly
- Accuracy: Measure perplexity degradation per quantization config
- Performance: Measure speedup vs accuracy tradeoff

### Success Metrics

- ✅ Support mixed precision automatically
- ✅ <2% perplexity increase with optimal quantization
- ✅ 1.5-2x speedup vs uniform INT8
- ✅ Works with all supported architectures

---

## 5. Tool Registry Integration (M5 - Tool Infrastructure)

### Current State

**Planned Feature**: M5 will add tool/function calling support

**Need**: Auto-detect model capabilities to register appropriate tools

### Proposed Architecture

**Phase 1: Capability Detection (M5.1)**

```rust
use crate::pruning::name_mapping::ModelArchitecture;

pub struct ToolRegistry {
    /// Registered tools
    tools: HashMap<String, Tool>,
    
    /// Model capabilities detected from architecture
    capabilities: ModelCapabilities,
}

impl ToolRegistry {
    pub fn from_model(model: &Model) -> Result<Self> {
        let name_mapper = TensorNameMapper::from_tensor_names(&model.tensor_names())?;
        
        // Detect capabilities from architecture
        let capabilities = Self::detect_capabilities(&name_mapper)?;
        
        // Auto-register appropriate tools
        let mut tools = HashMap::new();
        if capabilities.supports_text {
            tools.insert("text_generation".into(), Tool::text_generation());
        }
        if capabilities.supports_vision {
            tools.insert("image_understanding".into(), Tool::image_understanding());
        }
        if capabilities.supports_function_calling {
            tools.insert("function_call".into(), Tool::function_call());
        }
        
        Ok(Self { tools, capabilities })
    }
    
    /// Detect model capabilities from architecture
    fn detect_capabilities(mapper: &TensorNameMapper) -> Result<ModelCapabilities> {
        let mut capabilities = ModelCapabilities::default();
        
        // All transformer models support text
        capabilities.supports_text = true;
        
        // Check for vision encoder
        let has_vision = mapper.get_all_tensor_names().iter()
            .any(|name| name.contains("vision") || name.contains("image"));
        capabilities.supports_vision = has_vision;
        
        // Check for function calling head
        let has_function_head = mapper.get_all_tensor_names().iter()
            .any(|name| name.contains("function") || name.contains("tool"));
        capabilities.supports_function_calling = has_function_head;
        
        // Detect context length from positional embeddings
        capabilities.max_context = Self::detect_max_context(mapper);
        
        Ok(capabilities)
    }
}

#[derive(Default)]
pub struct ModelCapabilities {
    supports_text: bool,
    supports_vision: bool,
    supports_function_calling: bool,
    max_context: usize,
}
```

**Phase 2: Dynamic Tool Registration (M5.2)**

```rust
impl ToolRegistry {
    /// Register custom tool for specific model
    pub fn register_tool(&mut self, tool: Tool) -> Result<()> {
        // Validate tool is compatible with model capabilities
        if tool.requires_vision && !self.capabilities.supports_vision {
            bail!("Tool requires vision but model doesn't support it");
        }
        
        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }
    
    /// Get recommended tools for user query
    pub fn get_tools_for_query(&self, query: &str) -> Vec<&Tool> {
        // Simple keyword matching (could use LLM in future)
        self.tools.values()
            .filter(|tool| {
                tool.keywords.iter().any(|kw| query.contains(kw))
            })
            .collect()
    }
}
```

### Integration Points

- **Model Loader** (M4.1): Detect capabilities at load time
- **Inference**: Invoke tools based on user query
- **Configuration**: Allow manual tool registration

### Testing Strategy

- Unit: Test capability detection for various architectures
- Integration: Verify tool invocation works correctly
- Accuracy: Test tool selection for diverse queries

### Success Metrics

- ✅ Auto-detect capabilities for 5+ architectures
- ✅ <10ms overhead for tool registration
- ✅ 90%+ accuracy for tool selection
- ✅ Extensible for custom tools

---

## Implementation Timeline

### Milestone 4 (Current - Q1 2025)
- M4.1: Model Loading Integration (3 weeks)
- M4.2: Quantization Integration (2 weeks)

### Milestone 5 (Q2 2025)
- M5.1: Cache Management Integration (3 weeks)
- M5.2: Relationship-Aware Cache (2 weeks)
- M5.3: Dynamic Cache Sizing (1 week)
- M5.4: LoRA Integration (2 weeks)
- M5.5: LoRA Validation (1 week)

### Milestone 3.6.1 (Parallel with M5)
- M3.6.1: Architecture-Aware Sharding (1 week)
- M3.6.2: Heterogeneous Sharding (1 week)

### Milestone 5 (Continued)
- M5.6: Tool Registry Integration (2 weeks)

### Milestone 6+ (Q3 2025)
- M6.5: LLM-Assisted Name Mapping (4 weeks)

---

## Cross-Feature Benefits

**Unified Architecture Detection**:
All features share the same `TensorNameMapper` instance from model loader, ensuring consistent architecture detection.

**Composability**:
Features work together naturally:
- Cache manager uses layer detection for eviction
- Multi-GPU uses cache manager for memory coordination
- Quantization uses layer detection for mixed precision
- LoRA uses name mapping for adapter compatibility

**Extensibility**:
New architectures automatically work with all integrated features (no code changes needed).

**Performance**:
Single name mapping at load time (<2s) enables zero-overhead architecture-aware operations throughout inference.

---

## Success Criteria (Overall)

- ✅ All 5 features integrated by M5.6
- ✅ Support 10+ model architectures without code changes
- ✅ <5% total overhead for all integrations
- ✅ Comprehensive test coverage (unit + integration)
- ✅ Clear error messages for unsupported patterns
- ✅ Backward compatible with existing code

---

## References

- Core Name Mapping: `docs/NAME_MAPPING_MODEL_LOADING.md`
- LLM-Assisted Mapping: `docs/LLM_ASSISTED_NAME_MAPPING.md`
- Multi-GPU Sharding: M3.6 implementation
- Cache Management: M5 milestone (planned)
- Quantization: M4 milestone (planned)
