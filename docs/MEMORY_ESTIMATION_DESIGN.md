# Unified Memory Estimation Design

**Status**: DESIGN DOCUMENT  
**Affects**: AWQ quantization, Speculative decoding, Model loading  
**Created**: 2025-11-05

---

## Purpose

Provide a unified memory estimation system for Lightbulb that accurately predicts memory usage across:
- Standard FP16/BF16 models
- Quantized models (AWQ, GPTQ, Marlin format)
- Speculative decoding (dual-model scenarios)
- KV caches (single and multi-model)
- Activation memory during inference

**Why unified?**
- AWQ and speculative decoding both need memory accounting
- Prevents double-counting or incorrect estimates
- Enables intelligent model selection and configuration
- Critical for OOM prevention and resource allocation

---

## Memory Components

### 1. Model Weights

```rust
pub enum WeightMemory {
    /// Unquantized weights (FP32, FP16, BF16)
    Unquantized {
        dtype: DType,
        num_parameters: usize,
    },
    
    /// Quantized weights (AWQ, GPTQ, Marlin)
    Quantized {
        /// Original dtype before quantization
        original_dtype: DType,
        /// Number of bits per weight (4, 8)
        bits: usize,
        /// Total parameters
        num_parameters: usize,
        /// Quantization format overhead (scales, zero-points, etc.)
        metadata_bytes: usize,
    },
}

impl WeightMemory {
    /// Calculate weight memory in bytes
    pub fn bytes(&self) -> usize {
        match self {
            WeightMemory::Unquantized { dtype, num_parameters } => {
                let bytes_per_param = match dtype {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::U8 => 1,
                    _ => 4, // Conservative default
                };
                num_parameters * bytes_per_param
            }
            
            WeightMemory::Quantized { bits, num_parameters, metadata_bytes, .. } => {
                // Quantized weights (packed)
                let weight_bytes = (num_parameters * bits) / 8;
                
                // Add metadata (scales, zeros, group indices)
                weight_bytes + metadata_bytes
            }
        }
    }
    
    /// Estimate metadata size for quantized format
    pub fn estimate_metadata(
        num_parameters: usize,
        group_size: i32,
        quant_method: &str,
    ) -> usize {
        if group_size <= 0 {
            // Per-channel quantization
            let num_channels = (num_parameters as f64).sqrt() as usize;
            num_channels * 6 // 2 bytes scale + 4 bytes for overhead
        } else {
            // Group quantization
            let num_groups = num_parameters / group_size as usize;
            let scale_bytes = num_groups * 2; // FP16 scales
            let zero_bytes = if quant_method == "gptq" {
                (num_groups * 4) / 8 // 4-bit zeros, packed
            } else {
                0 // AWQ doesn't use zero-points
            };
            scale_bytes + zero_bytes
        }
    }
}
```

**Example estimates:**
- Llama-7B FP16: `7B × 2 bytes = 14GB`
- Llama-7B AWQ (4-bit, group=128): `7B × 0.5 bytes + 7B/128 × 2 bytes = 3.5GB + 109MB ≈ 3.6GB`
- Llama-1B AWQ (4-bit, group=128): `1B × 0.5 bytes + 1B/128 × 2 bytes = 0.5GB + 15.6MB ≈ 0.52GB`

### 2. KV Cache

```rust
pub struct KvCacheMemory {
    /// Batch size (number of concurrent sequences)
    pub batch_size: usize,
    
    /// Maximum sequence length
    pub max_seq_len: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Number of KV heads
    pub num_kv_heads: usize,
    
    /// Head dimension
    pub head_dim: usize,
    
    /// Data type (F16, BF16, F32)
    pub dtype: DType,
}

impl KvCacheMemory {
    /// Calculate KV cache memory in bytes
    pub fn bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 2, // Default to FP16
        };
        
        // Cache shape: [num_layers, 2 (K+V), batch_size, num_kv_heads, max_seq_len, head_dim]
        self.num_layers
            * 2  // K and V
            * self.batch_size
            * self.num_kv_heads
            * self.max_seq_len
            * self.head_dim
            * bytes_per_element
    }
    
    /// Create from model config
    pub fn from_config(
        config: &BatchedTransformerConfig,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            batch_size,
            max_seq_len,
            num_layers: config.num_hidden_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim(),
            dtype: config.dtype.unwrap_or(DType::F16),
        }
    }
}
```

**Example estimates:**
- Llama-7B (32 layers, 32 KV heads, 128 head_dim, batch=1, seq=2048, FP16):
  - `32 × 2 × 1 × 32 × 2048 × 128 × 2 bytes = 1GB`
- Llama-7B (batch=8, seq=2048):
  - `32 × 2 × 8 × 32 × 2048 × 128 × 2 bytes = 8GB`

### 3. Activation Memory

```rust
pub struct ActivationMemory {
    /// Batch size
    pub batch_size: usize,
    
    /// Sequence length
    pub seq_len: usize,
    
    /// Hidden size
    pub hidden_size: usize,
    
    /// Intermediate size (MLP)
    pub intermediate_size: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Data type
    pub dtype: DType,
}

impl ActivationMemory {
    /// Estimate activation memory (peak usage during forward pass)
    pub fn bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 2,
        };
        
        // Peak activations (conservative estimate):
        // - Input embeddings: batch × seq × hidden
        // - Attention QKV: batch × seq × hidden × 3
        // - Attention output: batch × seq × hidden
        // - MLP intermediate: batch × seq × intermediate × 2
        // - MLP output: batch × seq × hidden
        // Only one layer active at a time (no layer parallelism)
        
        let input_embed = self.batch_size * self.seq_len * self.hidden_size;
        let attn_qkv = self.batch_size * self.seq_len * self.hidden_size * 3;
        let attn_out = self.batch_size * self.seq_len * self.hidden_size;
        let mlp_intermediate = self.batch_size * self.seq_len * self.intermediate_size * 2;
        let mlp_out = self.batch_size * self.seq_len * self.hidden_size;
        
        let peak_elements = input_embed + attn_qkv + attn_out + mlp_intermediate + mlp_out;
        
        peak_elements * bytes_per_element
    }
    
    /// Create from model config
    pub fn from_config(
        config: &BatchedTransformerConfig,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            batch_size,
            seq_len,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_layers: config.num_hidden_layers,
            dtype: config.dtype.unwrap_or(DType::F16),
        }
    }
}
```

**Example estimates:**
- Llama-7B (batch=1, seq=128, hidden=4096, intermediate=11008):
  - Input: `1 × 128 × 4096 = 524K elements`
  - QKV: `1 × 128 × 4096 × 3 = 1.57M elements`
  - Attn out: `1 × 128 × 4096 = 524K elements`
  - MLP inter: `1 × 128 × 11008 × 2 = 2.82M elements`
  - MLP out: `1 × 128 × 4096 = 524K elements`
  - **Total: 5.96M elements × 2 bytes ≈ 12MB**

---

## Unified Memory Estimate Structure

```rust
/// Complete memory estimate for a model or model pair
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Model weight memory
    pub weights: WeightMemory,
    
    /// KV cache memory
    pub kv_cache: KvCacheMemory,
    
    /// Activation memory (peak during inference)
    pub activations: ActivationMemory,
    
    /// Additional overhead (buffers, workspace tensors, etc.)
    pub overhead_bytes: usize,
}

impl MemoryEstimate {
    /// Total memory required in bytes
    pub fn total_bytes(&self) -> usize {
        self.weights.bytes()
            + self.kv_cache.bytes()
            + self.activations.bytes()
            + self.overhead_bytes
    }
    
    /// Format as human-readable string
    pub fn display(&self) -> String {
        format!(
            "Weights: {}, KV Cache: {}, Activations: {}, Overhead: {}, Total: {}",
            format_bytes(self.weights.bytes()),
            format_bytes(self.kv_cache.bytes()),
            format_bytes(self.activations.bytes()),
            format_bytes(self.overhead_bytes),
            format_bytes(self.total_bytes()),
        )
    }
    
    /// Check if estimate fits in available memory
    pub fn fits_in(&self, available_bytes: usize) -> bool {
        self.total_bytes() < available_bytes
    }
    
    /// Calculate safety margin (percentage of available memory used)
    pub fn utilization(&self, available_bytes: usize) -> f64 {
        self.total_bytes() as f64 / available_bytes as f64
    }
}

fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
```

---

## Speculative Decoding Extensions

```rust
/// Memory estimate for speculative decoding (dual-model)
#[derive(Debug, Clone)]
pub struct SpeculativeMemoryEstimate {
    /// Target model estimate
    pub target: MemoryEstimate,
    
    /// Draft model estimate
    pub draft: MemoryEstimate,
    
    /// Shared resources (if any)
    pub shared_bytes: usize,
}

impl SpeculativeMemoryEstimate {
    /// Total memory for both models
    pub fn total_bytes(&self) -> usize {
        self.target.total_bytes() + self.draft.total_bytes() - self.shared_bytes
    }
    
    /// Create from target and draft specs
    pub fn from_specs(
        target_config: &BatchedTransformerConfig,
        target_quant: Option<&QuantConfig>,
        draft_config: &BatchedTransformerConfig,
        draft_quant: Option<&QuantConfig>,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Self {
        // Target model estimate
        let target_weights = if let Some(qc) = target_quant {
            WeightMemory::Quantized {
                original_dtype: DType::F16,
                bits: qc.bits as usize,
                num_parameters: estimate_parameters(target_config),
                metadata_bytes: WeightMemory::estimate_metadata(
                    estimate_parameters(target_config),
                    qc.group_size,
                    &qc.quant_method,
                ),
            }
        } else {
            WeightMemory::Unquantized {
                dtype: target_config.dtype.unwrap_or(DType::F16),
                num_parameters: estimate_parameters(target_config),
            }
        };
        
        let target = MemoryEstimate {
            weights: target_weights,
            kv_cache: KvCacheMemory::from_config(target_config, batch_size, max_seq_len),
            activations: ActivationMemory::from_config(target_config, batch_size, 256),
            overhead_bytes: 100 * 1024 * 1024, // 100MB overhead
        };
        
        // Draft model estimate (similar logic)
        let draft_weights = if let Some(qc) = draft_quant {
            WeightMemory::Quantized {
                original_dtype: DType::F16,
                bits: qc.bits as usize,
                num_parameters: estimate_parameters(draft_config),
                metadata_bytes: WeightMemory::estimate_metadata(
                    estimate_parameters(draft_config),
                    qc.group_size,
                    &qc.quant_method,
                ),
            }
        } else {
            WeightMemory::Unquantized {
                dtype: draft_config.dtype.unwrap_or(DType::F16),
                num_parameters: estimate_parameters(draft_config),
            }
        };
        
        let draft = MemoryEstimate {
            weights: draft_weights,
            kv_cache: KvCacheMemory::from_config(draft_config, 1, max_seq_len), // Single sequence for draft
            activations: ActivationMemory::from_config(draft_config, 1, 256),
            overhead_bytes: 50 * 1024 * 1024, // 50MB overhead
        };
        
        // Check for shared resources (e.g., same embedding layer)
        let shared_bytes = if can_share_embeddings(target_config, draft_config) {
            estimate_embedding_size(target_config)
        } else {
            0
        };
        
        Self { target, draft, shared_bytes }
    }
    
    /// Display breakdown
    pub fn display(&self) -> String {
        format!(
            "Target: {}\nDraft: {}\nShared: {}\nTotal: {}",
            self.target.display(),
            self.draft.display(),
            format_bytes(self.shared_bytes),
            format_bytes(self.total_bytes()),
        )
    }
}

fn estimate_parameters(config: &BatchedTransformerConfig) -> usize {
    // Rough estimate: embeddings + layers + output head
    let embedding_params = config.vocab_size * config.hidden_size;
    let layer_params = estimate_layer_parameters(config) * config.num_hidden_layers;
    let output_params = config.hidden_size * config.vocab_size;
    
    embedding_params + layer_params + output_params
}

fn estimate_layer_parameters(config: &BatchedTransformerConfig) -> usize {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    
    // Attention: Q, K, V, O projections
    let attn_params = hidden * hidden * 4;
    
    // MLP: gate, up, down
    let mlp_params = hidden * intermediate * 2 + intermediate * hidden;
    
    // Layer norms
    let ln_params = hidden * 2;
    
    attn_params + mlp_params + ln_params
}

fn can_share_embeddings(
    target: &BatchedTransformerConfig,
    draft: &BatchedTransformerConfig,
) -> bool {
    target.vocab_size == draft.vocab_size && target.hidden_size == draft.hidden_size
}

fn estimate_embedding_size(config: &BatchedTransformerConfig) -> usize {
    let dtype_bytes = match config.dtype.unwrap_or(DType::F16) {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => 2,
    };
    config.vocab_size * config.hidden_size * dtype_bytes
}
```

---

## Usage Examples

### Single Model (FP16)

```rust
let config = load_llama_7b_config()?;

let weights = WeightMemory::Unquantized {
    dtype: DType::F16,
    num_parameters: 7_000_000_000,
};

let kv_cache = KvCacheMemory::from_config(&config, batch_size=1, max_seq_len=2048);
let activations = ActivationMemory::from_config(&config, batch_size=1, seq_len=128);

let estimate = MemoryEstimate {
    weights,
    kv_cache,
    activations,
    overhead_bytes: 100 * 1024 * 1024, // 100MB
};

println!("{}", estimate.display());
// Output: Weights: 14.00 GB, KV Cache: 1.00 GB, Activations: 12.00 MB, Overhead: 100.00 MB, Total: 15.11 GB

if !estimate.fits_in(available_memory) {
    return Err("Insufficient memory for model");
}
```

### Single Model (AWQ Quantized)

```rust
let config = load_llama_7b_config()?;

let weights = WeightMemory::Quantized {
    original_dtype: DType::F16,
    bits: 4,
    num_parameters: 7_000_000_000,
    metadata_bytes: WeightMemory::estimate_metadata(7_000_000_000, 128, "awq"),
};

let estimate = MemoryEstimate {
    weights,
    kv_cache: KvCacheMemory::from_config(&config, 1, 2048),
    activations: ActivationMemory::from_config(&config, 1, 128),
    overhead_bytes: 150 * 1024 * 1024, // 150MB (extra for Marlin workspace)
};

println!("{}", estimate.display());
// Output: Weights: 3.61 GB, KV Cache: 1.00 GB, Activations: 12.00 MB, Overhead: 150.00 MB, Total: 4.77 GB
```

### Speculative Decoding (Target FP16 + Draft AWQ)

```rust
let target_config = load_llama_7b_config()?;
let draft_config = load_llama_1b_config()?;

let target_quant = None; // FP16
let draft_quant = Some(QuantConfig {
    bits: 4,
    group_size: 128,
    quant_method: "awq".to_string(),
    // ...
});

let estimate = SpeculativeMemoryEstimate::from_specs(
    &target_config,
    target_quant.as_ref(),
    &draft_config,
    draft_quant.as_ref(),
    batch_size=1,
    max_seq_len=2048,
);

println!("{}", estimate.display());
// Output:
// Target: Weights: 14.00 GB, KV Cache: 1.00 GB, Activations: 12.00 MB, Overhead: 100.00 MB, Total: 15.11 GB
// Draft: Weights: 0.52 GB, KV Cache: 0.13 GB, Activations: 1.50 MB, Overhead: 50.00 MB, Total: 0.70 GB
// Shared: 0.00 B
// Total: 15.81 GB

if estimate.total_bytes() < 24 * 1024 * 1024 * 1024 {
    println!("✅ Fits on 24GB GPU with {:.1}% utilization", 
        estimate.utilization(24 * 1024 * 1024 * 1024) * 100.0);
}
```

---

## Integration Points

### AWQ Implementation (M3_AWQ_IMPLEMENTATION_PLAN.md)

**Phase 3: Model Loader Integration**
- Use `WeightMemory::Quantized` for AWQ models
- Update `load_local_llama()` to return `MemoryEstimate`
- Validate memory before loading

```rust
pub fn load_local_llama(
    model_dir: &str,
    dtype: DType,
    device: Device,
    quant_config: Option<&QuantConfig>,
) -> Result<(BatchedTransformer, MemoryEstimate)> {
    let config = load_config(model_dir)?;
    
    // Estimate memory before loading
    let estimate = if let Some(qc) = quant_config {
        MemoryEstimate {
            weights: WeightMemory::Quantized {
                original_dtype: dtype,
                bits: qc.bits as usize,
                num_parameters: estimate_parameters(&config),
                metadata_bytes: WeightMemory::estimate_metadata(
                    estimate_parameters(&config),
                    qc.group_size,
                    &qc.quant_method,
                ),
            },
            kv_cache: KvCacheMemory::from_config(&config, 1, 2048),
            activations: ActivationMemory::from_config(&config, 1, 256),
            overhead_bytes: 150 * 1024 * 1024, // Marlin workspace
        }
    } else {
        // Standard FP16/BF16 estimate
        // ...
    };
    
    // Check available memory
    let available = device.memory_available()?;
    if !estimate.fits_in(available) {
        anyhow::bail!(
            "Insufficient memory: need {}, have {}",
            format_bytes(estimate.total_bytes()),
            format_bytes(available),
        );
    }
    
    // Load model
    let model = load_model_weights(model_dir, &config, device, quant_config)?;
    
    Ok((model, estimate))
}
```

### Speculative Decoding (M3_SPECULATIVE_DECODING_PLAN.md)

**Phase 1: Production Model Management**
- Use `SpeculativeMemoryEstimate` for dual-model loading
- Validate total memory before loading both models

```rust
impl SpeculativeModelPair {
    pub fn load(
        target_spec: ModelSpec,
        draft_spec: ModelSpec,
        device: Device,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Get configs
        let target_config = load_config(&target_spec.path)?;
        let draft_config = load_config(&draft_spec.path)?;
        
        // Estimate memory
        let estimate = SpeculativeMemoryEstimate::from_specs(
            &target_config,
            target_spec.quant_config.as_ref(),
            &draft_config,
            draft_spec.quant_config.as_ref(),
            1, // batch_size
            max_seq_len,
        );
        
        // Validate
        let available = device.memory_available()?;
        if !estimate.fits_in(available) {
            anyhow::bail!(
                "Insufficient memory for speculative pair: need {}, have {}",
                format_bytes(estimate.total_bytes()),
                format_bytes(available),
            );
        }
        
        tracing::info!("Memory estimate: {}", estimate.display());
        
        // Load both models
        let (target_model, _) = load_local_llama(/* ... */)?;
        let (draft_model, _) = load_local_llama(/* ... */)?;
        
        Ok(Self { target, draft, /* ... */ })
    }
}
```

---

## File Structure

```
src/
├── memory/
│   ├── mod.rs                  # Re-exports
│   ├── estimate.rs             # MemoryEstimate, WeightMemory, etc.
│   ├── speculative.rs          # SpeculativeMemoryEstimate
│   └── utils.rs                # format_bytes, estimate_parameters
└── loaders/
    ├── mod.rs
    ├── llama.rs                # Updated to return MemoryEstimate
    └── speculative.rs          # SpeculativeModelPair with memory checks
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_weight_memory_fp16() {
    let weights = WeightMemory::Unquantized {
        dtype: DType::F16,
        num_parameters: 7_000_000_000,
    };
    assert_eq!(weights.bytes(), 14_000_000_000); // 14GB
}

#[test]
fn test_weight_memory_awq() {
    let weights = WeightMemory::Quantized {
        original_dtype: DType::F16,
        bits: 4,
        num_parameters: 7_000_000_000,
        metadata_bytes: 109_000_000, // ~109MB scales
    };
    // 7B × 0.5 bytes + 109MB ≈ 3.609GB
    assert!((weights.bytes() as f64 - 3.609e9).abs() < 1e8);
}

#[test]
fn test_kv_cache_memory() {
    let cache = KvCacheMemory {
        batch_size: 1,
        max_seq_len: 2048,
        num_layers: 32,
        num_kv_heads: 32,
        head_dim: 128,
        dtype: DType::F16,
    };
    // 32 × 2 × 1 × 32 × 2048 × 128 × 2 = 1,073,741,824 bytes ≈ 1GB
    assert_eq!(cache.bytes(), 1_073_741_824);
}

#[test]
fn test_speculative_estimate() {
    let target_config = mock_llama_7b_config();
    let draft_config = mock_llama_1b_config();
    
    let estimate = SpeculativeMemoryEstimate::from_specs(
        &target_config, None,
        &draft_config, Some(&awq_config()),
        1, 2048,
    );
    
    // Target: ~15GB, Draft: ~0.7GB, Total: ~15.7GB
    assert!(estimate.total_bytes() > 15_000_000_000);
    assert!(estimate.total_bytes() < 16_000_000_000);
}
```

### Integration Tests

```rust
#[test]
fn test_memory_estimate_matches_actual() {
    let device = Device::cuda_if_available(0)?;
    let initial_mem = device.memory_allocated()?;
    
    // Load model and get estimate
    let (model, estimate) = load_local_llama("/models/llama-1b", DType::F16, device.clone(), None)?;
    
    let actual_mem = device.memory_allocated()? - initial_mem;
    
    // Estimate should be within 10% of actual
    let error = (estimate.total_bytes() as f64 - actual_mem as f64).abs() / actual_mem as f64;
    assert!(error < 0.1, "Estimate error: {:.1}%", error * 100.0);
}
```

---

## Success Criteria

- ✅ Estimates accurate within 10% of actual memory usage
- ✅ Works for FP16, BF16, and quantized models (AWQ, GPTQ)
- ✅ Supports single-model and speculative (dual-model) scenarios
- ✅ Prevents OOM errors by pre-validating memory
- ✅ Used by both AWQ and speculative decoding implementations
- ✅ Clear error messages when insufficient memory

---

## References

- AWQ Paper: https://arxiv.org/abs/2306.00978 (quantization overhead)
- Speculative Decoding Paper: https://arxiv.org/abs/2211.17192 (dual-model memory)
- PyTorch Memory Profiling: https://pytorch.org/docs/stable/torch_cuda_memory.html
- Candle Device API: `candle_core::Device::memory_*()` methods

---

## Next Steps

1. ✅ **Complete this design** (done!)
2. 📋 Implement `src/memory/estimate.rs` (core structs and methods)
3. 📋 Integrate into `load_local_llama()` (AWQ Phase 3)
4. 📋 Integrate into `SpeculativeModelPair::load()` (Spec Decoding Phase 1)
5. 📋 Add tests and validate against real model loads
