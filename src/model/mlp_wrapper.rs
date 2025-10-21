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

use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Minimal MLP using Candle's components
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// This is exactly what Candle's internal Mlp does, but made public.
#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: Linear, // c_fc1 in Candle
    up_proj: Linear,   // c_fc2 in Candle
    down_proj: Linear, // c_proj in Candle
}

impl Mlp {
    /// Create a new MLP layer
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden layer dimension
    /// * `vb` - VarBuilder for loading weights
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // DEBUG: Print the path prefix
        eprintln!("DEBUG MLP loading from path prefix: {:?}", vb.prefix());

        // Use Candle's linear_b with bias=false (same as Candle's Llama for models without bias)
        let gate_proj =
            candle_nn::linear_b(hidden_size, intermediate_size, false, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_b(hidden_size, intermediate_size, false, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_b(intermediate_size, hidden_size, false, vb.pp("down_proj"))?;

        // DEBUG: Check actual weight shapes and stats
        let gate_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, hidden_size), "weight")?;
        let down_weight = vb
            .pp("down_proj")
            .get((hidden_size, intermediate_size), "weight")?;

        let gate_vec = gate_weight.flatten_all()?.to_vec1::<f32>()?;
        let gate_mean: f32 = gate_vec.iter().sum::<f32>() / gate_vec.len() as f32;
        let gate_max = gate_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let gate_min = gate_vec.iter().cloned().fold(f32::INFINITY, f32::min);

        let down_vec = down_weight.flatten_all()?.to_vec1::<f32>()?;
        let down_mean: f32 = down_vec.iter().sum::<f32>() / down_vec.len() as f32;
        let down_max = down_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let down_min = down_vec.iter().cloned().fold(f32::INFINITY, f32::min);

        eprintln!(
            "DEBUG MLP weights: gate [{:?}] mean={:.6}, range=[{:.6}, {:.6}], down [{:?}] mean={:.6}, range=[{:.6}, {:.6}]",
            gate_weight.shape().dims(),
            gate_mean,
            gate_min,
            gate_max,
            down_weight.shape().dims(),
            down_mean,
            down_min,
            down_max
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
        let input_max = input_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let input_mean: f32 = input_vec.iter().sum::<f32>() / input_vec.len() as f32;

        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;

        // DEBUG: Check intermediate stats
        let gate_vec = gate.flatten_all()?.to_vec1::<f32>()?;
        let gate_max = gate_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let up_vec = up.flatten_all()?.to_vec1::<f32>()?;
        let up_max = up_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let intermediate = (gate * up)?;

        // DEBUG: Check intermediate product shape and stats
        let inter_vec = intermediate.flatten_all()?.to_vec1::<f32>()?;
        let inter_max = inter_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let inter_mean: f32 = inter_vec.iter().sum::<f32>() / inter_vec.len() as f32;
        eprintln!(
            "DEBUG MLP intermediate: shape={:?}, mean={:.6}, max={:.6}",
            intermediate.shape(),
            inter_mean,
            inter_max
        );

        let output = self.down_proj.forward(&intermediate)?;

        // DEBUG: Always print for comparison
        let out_vec = output.flatten_all()?.to_vec1::<f32>()?;
        let out_max = out_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_min = out_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "DEBUG MLP FORWARD: input mean={:.6}, max={:.6} → output min={:.6}, max={:.6}",
            input_mean, input_max, out_min, out_max
        );

        Ok(output)
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}
