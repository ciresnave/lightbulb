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
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::io::{Read, Seek};

/// Minimal MLP using Candle's components
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// This is exactly what Candle's internal Mlp does, but made public.
#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: QuantizableLinear, // c_fc1 in Candle
    up_proj: QuantizableLinear,   // c_fc2 in Candle
    down_proj: QuantizableLinear, // c_proj in Candle
}

impl Mlp {
    /// Create a new MLP layer
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden layer dimension
    /// * `vb` - VarBuilder for loading weights
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // Use Candle's linear_b with bias=false (same as Candle's Llama for models without bias)
        let gate_proj = QuantizableLinear::from_linear(candle_nn::linear_b(
            hidden_size,
            intermediate_size,
            false,
            vb.pp("gate_proj"),
        )?);
        let up_proj = QuantizableLinear::from_linear(candle_nn::linear_b(
            hidden_size,
            intermediate_size,
            false,
            vb.pp("up_proj"),
        )?);
        let down_proj = QuantizableLinear::from_linear(candle_nn::linear_b(
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
    pub fn from_gguf<R: Read + Seek>(
        _hidden_size: usize,
        _intermediate_size: usize,
        gguf_content: &crate::gguf::Content,
        file: &mut R,
        device: &candle_core::Device,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);

        // Load quantized MLP tensors
        // GGUF naming: ffn_gate (w1), ffn_up (w3), ffn_down (w2)
        let gate_tensor =
            gguf_content.tensor(file, &format!("{}.ffn_gate.weight", prefix), device)?;
        let up_tensor = gguf_content.tensor(file, &format!("{}.ffn_up.weight", prefix), device)?;
        let down_tensor =
            gguf_content.tensor(file, &format!("{}.ffn_down.weight", prefix), device)?;

        // Convert QTensor to QMatMul and wrap in QuantizableLinear
        let gate_proj = QuantizableLinear::from_qmatmul(
            candle_core::quantized::QMatMul::from_qtensor(gate_tensor)?,
        );
        let up_proj = QuantizableLinear::from_qmatmul(
            candle_core::quantized::QMatMul::from_qtensor(up_tensor)?,
        );
        let down_proj = QuantizableLinear::from_qmatmul(
            candle_core::quantized::QMatMul::from_qtensor(down_tensor)?,
        );

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass
    ///
    /// Implements SwiGLU: `down(silu(gate(x)) * up(x))`
    ///
    /// Naturally handles batched inputs:
    /// - Input: [batch, seq, hidden] → Output: [batch, seq, hidden]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // DEBUG: Check input stats
        let input_vec = x.flatten_all()?.to_vec1::<f32>()?;
        let _input_max = input_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let _input_mean: f32 = input_vec.iter().sum::<f32>() / input_vec.len() as f32;

        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
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
