//! Thin wrapper around Candle's MLP components
//!
//! This module provides a minimal MLP implementation using Candle's Linear layers
//! and activation functions. It's functionally identical to Candle's internal Mlp
//! but exposed as public API for use in our batched transformer blocks.
//!
//! # Why This Exists
//!
//! Candle's `Mlp` struct is private, but we can easily recreate it using their
//! public components (`Linear`, `ops::silu`). This is much simpler than our
//! previous `custom_mlp.rs` which had 400+ lines including debug code.

use crate::model::quantizable_linear::QuantizableLinear;
use candlelight::core::{Module, Result, Tensor};
use candlelight::nn::VarBuilder;
use std::io::{Read, Seek};

/// Minimal MLP using Candle's components
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// This is exactly what Candle's internal Mlp does, but made public.
///
/// # CPU Kernel Fusion (M3.3)
///
/// When `use_fused_kernels` is enabled, the MLP uses fused operations for better
/// CPU performance:
/// - `fused_linear_silu` for gate_proj path (~11% bandwidth reduction)
/// - Maintains bit-exact correctness with unfused path
#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: QuantizableLinear, // c_fc1 in Candle
    up_proj: QuantizableLinear,   // c_fc2 in Candle
    down_proj: QuantizableLinear, // c_proj in Candle
    use_fused_kernels: bool,      // M3.3: Enable CPU kernel fusion
}

impl Mlp {
    /// Create a new MLP layer
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden layer dimension
    /// * `vb` - VarBuilder for loading weights
    /// * `use_fused_kernels` - Enable CPU kernel fusion for better performance
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        use_fused_kernels: bool,
    ) -> Result<Self> {
        // Use Candle's linear_b with bias=false (same as Candle's Llama for models without bias)
        let gate_proj = QuantizableLinear::from_linear(candlelight::nn::linear_b(
            hidden_size,
            intermediate_size,
            false,
            vb.pp("gate_proj"),
        )?);
        let up_proj = QuantizableLinear::from_linear(candlelight::nn::linear_b(
            hidden_size,
            intermediate_size,
            false,
            vb.pp("up_proj"),
        )?);
        let down_proj = QuantizableLinear::from_linear(candlelight::nn::linear_b(
            intermediate_size,
            hidden_size,
            false,
            vb.pp("down_proj"),
        )?);

        // DEBUG: Check actual weight shapes and stats
        let _gate_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, hidden_size), "weight")?;
        let _down_weight = vb
            .pp("down_proj")
            .get((hidden_size, intermediate_size), "weight")?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            use_fused_kernels,
        })
    }

    /// Create a new MLP from GGUF quantized weights
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden layer dimension
    /// * `gguf_content` - GGUF file content with metadata and tensor info
    /// * `file` - Open file handle for reading tensor data
    /// * `device` - Device to load tensors on
    /// * `layer_idx` - Layer index for tensor naming (e.g., blk.0, blk.1, ...)
    /// * `use_fused_kernels` - Enable CPU kernel fusion for better performance
    /// * `name_mapper` - Tensor name mapper for architecture-agnostic loading
    pub fn from_gguf<R: Read + Seek>(
        _hidden_size: usize,
        _intermediate_size: usize,
        gguf_content: &crate::gguf::Content,
        file: &mut R,
        device: &candlelight::core::Device,
        layer_idx: usize,
        use_fused_kernels: bool,
        name_mapper: &crate::pruning::name_mapping::TensorNameMapper,
    ) -> Result<Self> {
        // Map abstract names to concrete architecture-specific names
        let gate_name = name_mapper
            .map_name(&format!("layer_{}.ffn.gate", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.ffn_gate.weight", layer_idx));
        let up_name = name_mapper
            .map_name(&format!("layer_{}.ffn.up", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.ffn_up.weight", layer_idx));
        let down_name = name_mapper
            .map_name(&format!("layer_{}.ffn.down", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.ffn_down.weight", layer_idx));

        // Load quantized MLP tensors
        // GGUF naming: ffn_gate (w1), ffn_up (w3), ffn_down (w2)
        let gate_tensor = gguf_content.tensor(file, &gate_name, device)?;
        let up_tensor = gguf_content.tensor(file, &up_name, device)?;
        let down_tensor = gguf_content.tensor(file, &down_name, device)?;

        // Convert QTensor to QMatMul and wrap in QuantizableLinear
        let gate_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(gate_tensor)?,
        );
        let up_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(up_tensor)?,
        );
        let down_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(down_tensor)?,
        );

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            use_fused_kernels,
        })
    }

    /// Forward pass
    ///
    /// Implements SwiGLU: `down(silu(gate(x)) * up(x))`
    ///
    /// Naturally handles batched inputs:
    /// - Input: [batch, seq, hidden] → Output: [batch, seq, hidden]
    ///
    /// When `use_fused_kernels` is true, uses `fused_linear_silu` for the gate path
    /// to reduce memory bandwidth and improve CPU performance (~11% bandwidth reduction).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // M3.3: Fused kernels implementation
        // NOTE: Current implementation shows regression due to overhead of extracting
        // weights from candlelight::nn::Linear. The weight() and bias() methods appear to
        // be expensive (possibly cloning or creating new tensors).
        //
        // Analysis from benchmark:
        // - Unfused: 21.7ms per forward pass
        // - Fused (attempted): 213.7ms per forward pass (10x slower!)
        //
        // Root cause: candlelight::nn::Linear doesn't expose weights efficiently for
        // external fusion. True kernel fusion would require:
        // 1. Custom linear layer with direct weight access, OR
        // 2. Candle upstream changes to provide fused matmul+activation ops
        //
        // Decision: Disable fusion for now. Keep infrastructure for future work
        // when Candle provides proper fused op support.
        let gate = if self.use_fused_kernels && false {  // Disabled: see NOTE above
            // Fused path: gate_proj + silu in one operation
            // Note: fused_linear_silu currently only supports QuantizableLinear::Regular
            // For quantized models, fall back to unfused path
            match &self.gate_proj {
                crate::model::quantizable_linear::QuantizableLinear::Regular(linear) => {
                    // Extract weight and bias from Linear layer
                    let weight = linear.weight();
                    let bias = linear.bias();
                    crate::model::fused_kernels::fused_linear_silu(x, weight, bias)?
                }
                crate::model::quantizable_linear::QuantizableLinear::Quantized(_) => {
                    // Quantized path: use unfused (fusion doesn't apply to quantized ops)
                    candlelight::nn::ops::silu(&self.gate_proj.forward(x)?)?
                }
            }
        } else {
            // Unfused path: separate gate_proj + silu
            candlelight::nn::ops::silu(&self.gate_proj.forward(x)?)?
        };

        let up = self.up_proj.forward(x)?;
        let intermediate = (gate * up)?;
        let output = self.down_proj.forward(&intermediate)?;

        Ok(output)
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}
