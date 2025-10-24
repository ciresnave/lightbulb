//! Quantizable Linear Layer
//!
//! A unified linear layer type that supports both regular (fp32/fp16) and quantized (Q4/Q8) weights.
//! This allows our custom batched transformer to work with both safetensors and GGUF models
//! without changing the inference code.

use candle_core::quantized::QMatMul;
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

/// A linear layer that can use either regular or quantized weights
///
/// This wrapper allows seamless switching between:
/// - **Regular**: Standard fp32/fp16 weights from safetensors
/// - **Quantized**: Q4_0, Q4_K, Q8_0, etc. from GGUF files
///
/// Both variants implement the same `Module` trait, so inference code
/// doesn't need to know which type it's using.
///
/// # Example
///
/// ```rust,ignore
/// // From safetensors (regular)
/// let linear = QuantizableLinear::from_linear(
///     candle_nn::linear(in_dim, out_dim, vb)?
/// );
///
/// // From GGUF (quantized)
/// let linear = QuantizableLinear::from_qmatmul(
///     QMatMul::from_qtensor(qtensor)?
/// );
///
/// // Both work the same way
/// let output = linear.forward(&input)?;
/// ```
#[derive(Clone, Debug)]
pub enum QuantizableLinear {
    /// Regular fp32/fp16/bf16 weights
    Regular(Linear),

    /// Quantized weights (Q4_0, Q4_K, Q8_0, etc.)
    Quantized(QMatMul),
}

impl QuantizableLinear {
    /// Create from a regular Linear layer (for safetensors models)
    pub fn from_linear(linear: Linear) -> Self {
        Self::Regular(linear)
    }

    /// Create from a quantized QMatMul layer (for GGUF models)
    pub fn from_qmatmul(qmatmul: QMatMul) -> Self {
        Self::Quantized(qmatmul)
    }

    /// Check if this layer uses quantized weights
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

/// Implement Module trait so this can be used anywhere a Linear layer is used
impl Module for QuantizableLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Regular(linear) => linear.forward(xs),
            Self::Quantized(qmatmul) => qmatmul.forward(xs),
        }
    }
}
