//! KVTensor: Canonical representation for Key/Value cache tensors
//!
//! Maintains tensors in the shape expected by ScatteredKvCache:
//! `[max_batch_size, num_heads, seq_len, head_dim]`
//!
//! This wrapper ensures type safety and prevents shape mismatches when
//! interacting with Candle's kv_cache system, which requires tensors
//! dimensioned by max_batch_size (not actual batch size).

use anyhow::Result;
use candle_core::Tensor;

/// Wrapper for Key/Value tensors with canonical max_batch_size representation
///
/// # Invariants
/// - Internal tensor always has shape `[max_batch_size, num_heads, seq_len, head_dim]`
/// - Unused batch slots (beyond actual_batch_size) are zero-filled
/// - Views returned by methods are zero-cost (use `narrow()`, no copying)
///
/// # Example
/// ```ignore
/// // After computing K tensor with shape [2, 8, 128, 64]
/// let kv = KVTensor::from_compact(&k_tensor, max_batch_size)?;
///
/// // Pass to cache (needs full [max_batch_size, ...] shape)
/// executor.append_kv(layer_idx, kv.as_full(), v.as_full(), &iam)?;
///
/// // Use in attention computation (extract only active slots)
/// let k_active = kv.as_compact()?;
/// ```
#[derive(Debug, Clone)]
pub struct KVTensor {
    /// Canonical representation: [max_batch_size, num_heads, seq_len, head_dim]
    full: Tensor,
    /// Maximum batch size (tensor dimension 0)
    max_batch_size: usize,
    /// Actual number of active batch slots (for compact views)
    actual_batch_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Sequence length (may be 1 for decode, larger for prefill)
    seq_len: usize,
    /// Head dimension
    head_dim: usize,
}

impl KVTensor {
    /// Create KVTensor from compact tensor by expanding to max_batch_size
    ///
    /// Takes a tensor with shape `[actual_batch_size, num_heads, seq_len, head_dim]`
    /// and expands it to `[max_batch_size, num_heads, seq_len, head_dim]` by
    /// zero-padding the batch dimension.
    ///
    /// # Arguments
    /// * `compact` - Input tensor with shape `[batch, heads, seq, dim]`
    /// * `max_batch_size` - Target batch dimension size (must be >= compact.dim(0))
    ///
    /// # Returns
    /// KVTensor with canonical representation suitable for ScatteredKvCache
    ///
    /// # Errors
    /// - If `compact` doesn't have rank 4
    /// - If `max_batch_size < compact.dim(0)`
    /// - If tensor operations fail
    pub fn from_compact(compact: &Tensor, max_batch_size: usize) -> Result<Self> {
        let dims = compact.dims();
        if dims.len() != 4 {
            anyhow::bail!(
                "Expected 4D tensor [batch, heads, seq, dim], got shape {:?}",
                dims
            );
        }

        let actual_batch_size = dims[0];
        let num_heads = dims[1];
        let seq_len = dims[2];
        let head_dim = dims[3];

        if max_batch_size < actual_batch_size {
            anyhow::bail!(
                "max_batch_size ({}) must be >= actual_batch_size ({})",
                max_batch_size,
                actual_batch_size
            );
        }

        // Fast path: already correct size
        if max_batch_size == actual_batch_size {
            return Ok(Self {
                full: compact.clone(),
                max_batch_size,
                actual_batch_size,
                num_heads,
                seq_len,
                head_dim,
            });
        }

        // Expand batch dimension by padding with zeros
        let padding_shape = (
            max_batch_size - actual_batch_size,
            num_heads,
            seq_len,
            head_dim,
        );
        let padding = Tensor::zeros(padding_shape, compact.dtype(), compact.device())?;

        // Concatenate along batch dimension (dim 0)
        let full = Tensor::cat(&[compact, &padding], 0)?;

        Ok(Self {
            full,
            max_batch_size,
            actual_batch_size,
            num_heads,
            seq_len,
            head_dim,
        })
    }

    /// Get reference to full tensor (for cache operations)
    ///
    /// Returns tensor with shape `[max_batch_size, num_heads, seq_len, head_dim]`
    /// suitable for passing to `ScatteredKvCache.append()`.
    ///
    /// This is a zero-cost operation (just returns a reference).
    #[inline]
    pub fn as_full(&self) -> &Tensor {
        &self.full
    }

    /// Extract compact view containing only active batch slots
    ///
    /// Returns tensor with shape `[actual_batch_size, num_heads, seq_len, head_dim]`
    /// suitable for attention computation.
    ///
    /// This uses `narrow()` internally, so it's a zero-cost view (no copying).
    ///
    /// # Errors
    /// If narrow operation fails (shouldn't happen with valid construction)
    pub fn as_compact(&self) -> Result<Tensor> {
        if self.actual_batch_size == self.max_batch_size {
            // Fast path: no narrowing needed
            Ok(self.full.clone())
        } else {
            // Extract first actual_batch_size rows along dimension 0
            Ok(self.full.narrow(0, 0, self.actual_batch_size)?)
        }
    }

    /// Get tensor view for a specific batch slot
    ///
    /// Returns tensor with shape `[1, num_heads, seq_len, head_dim]`
    ///
    /// # Arguments
    /// * `slot_idx` - Batch slot index (must be < max_batch_size)
    ///
    /// # Errors
    /// - If slot_idx >= max_batch_size
    /// - If narrow operation fails
    pub fn get_slot(&self, slot_idx: usize) -> Result<Tensor> {
        if slot_idx >= self.max_batch_size {
            anyhow::bail!(
                "Slot index {} out of bounds (max_batch_size={})",
                slot_idx,
                self.max_batch_size
            );
        }

        Ok(self.full.narrow(0, slot_idx, 1)?)
    }

    /// Get shape information
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (
            self.max_batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
        )
    }

    /// Get actual batch size (number of active slots)
    #[inline]
    pub fn actual_batch_size(&self) -> usize {
        self.actual_batch_size
    }

    /// Get max batch size (tensor dimension)
    #[inline]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Get number of heads
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get sequence length
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get head dimension
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_from_compact_expansion() {
        let device = Device::Cpu;

        // Create compact tensor: [2, 4, 8, 32]
        let compact = Tensor::zeros((2, 4, 8, 32), DType::F32, &device).unwrap();

        // Expand to max_batch_size=4
        let kv = KVTensor::from_compact(&compact, 4).unwrap();

        assert_eq!(kv.max_batch_size(), 4);
        assert_eq!(kv.actual_batch_size(), 2);
        assert_eq!(kv.num_heads(), 4);
        assert_eq!(kv.seq_len(), 8);
        assert_eq!(kv.head_dim(), 32);

        // Full tensor should be [4, 4, 8, 32]
        assert_eq!(kv.as_full().dims(), &[4, 4, 8, 32]);
    }

    #[test]
    fn test_from_compact_already_correct_size() {
        let device = Device::Cpu;

        // Create tensor already at max size: [4, 4, 8, 32]
        let tensor = Tensor::zeros((4, 4, 8, 32), DType::F32, &device).unwrap();

        // Should take fast path (no expansion needed)
        let kv = KVTensor::from_compact(&tensor, 4).unwrap();

        assert_eq!(kv.max_batch_size(), 4);
        assert_eq!(kv.actual_batch_size(), 4);
        assert_eq!(kv.as_full().dims(), &[4, 4, 8, 32]);
    }

    #[test]
    fn test_as_compact_view() {
        let device = Device::Cpu;

        let compact = Tensor::zeros((2, 4, 8, 32), DType::F32, &device).unwrap();
        let kv = KVTensor::from_compact(&compact, 4).unwrap();

        // Compact view should return [2, 4, 8, 32]
        let compact_view = kv.as_compact().unwrap();
        assert_eq!(compact_view.dims(), &[2, 4, 8, 32]);
    }

    #[test]
    fn test_get_slot() {
        let device = Device::Cpu;

        let compact = Tensor::zeros((2, 4, 8, 32), DType::F32, &device).unwrap();
        let kv = KVTensor::from_compact(&compact, 4).unwrap();

        // Get slot 0
        let slot0 = kv.get_slot(0).unwrap();
        assert_eq!(slot0.dims(), &[1, 4, 8, 32]);

        // Get slot 1
        let slot1 = kv.get_slot(1).unwrap();
        assert_eq!(slot1.dims(), &[1, 4, 8, 32]);

        // Get slot 3 (in padding region)
        let slot3 = kv.get_slot(3).unwrap();
        assert_eq!(slot3.dims(), &[1, 4, 8, 32]);

        // Slot 4 should fail
        let result = kv.get_slot(4);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_rank() {
        let device = Device::Cpu;

        // Create 3D tensor (invalid)
        let invalid = Tensor::zeros((2, 4, 8), DType::F32, &device).unwrap();

        let result = KVTensor::from_compact(&invalid, 4);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected 4D tensor")
        );
    }

    #[test]
    fn test_max_batch_size_too_small() {
        let device = Device::Cpu;

        let compact = Tensor::zeros((4, 4, 8, 32), DType::F32, &device).unwrap();

        // Try to "expand" to smaller size (invalid)
        let result = KVTensor::from_compact(&compact, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be >="));
    }

    #[test]
    fn test_shape_info() {
        let device = Device::Cpu;

        let compact = Tensor::zeros((2, 8, 16, 64), DType::F32, &device).unwrap();
        let kv = KVTensor::from_compact(&compact, 4).unwrap();

        let shape = kv.shape();
        assert_eq!(shape, (4, 8, 16, 64));

        assert_eq!(kv.max_batch_size(), 4);
        assert_eq!(kv.actual_batch_size(), 2);
        assert_eq!(kv.num_heads(), 8);
        assert_eq!(kv.seq_len(), 16);
        assert_eq!(kv.head_dim(), 64);
    }

    #[test]
    fn test_zero_padding_correctness() {
        let device = Device::Cpu;

        // Create tensor with known values in first batch slot
        let mut data = vec![0.0f32; 2 * 4 * 8 * 32];
        // Set first slot to 1.0
        for i in 0..(4 * 8 * 32) {
            data[i] = 1.0;
        }

        let compact = Tensor::from_vec(data, (2, 4, 8, 32), &device).unwrap();
        let kv = KVTensor::from_compact(&compact, 4).unwrap();

        // Check full tensor has correct shape
        assert_eq!(kv.as_full().dims(), &[4, 4, 8, 32]);

        // First slot should have 1.0 values
        let slot0 = kv.get_slot(0).unwrap();
        let slot0_sum = slot0.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((slot0_sum - (4.0 * 8.0 * 32.0)).abs() < 0.01);

        // Third slot (padding) should be all zeros
        let slot2 = kv.get_slot(2).unwrap();
        let slot2_sum = slot2.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(slot2_sum.abs() < 0.01);
    }
}
