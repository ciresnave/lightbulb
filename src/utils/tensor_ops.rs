//! Tensor utility operations vendored from candle-extensions/candle-ext
//!
//! These implementations are based on candle-ext but adapted for lightbulb's needs.
//! Original source: https://github.com/EricLBuehler/candle-extensions
//!
//! M4.2: Selectively vendored utilities for tensor manipulation

use candlelight::core::{Result, Tensor};

/// Creates an upper triangular matrix from a square matrix.
///
/// All elements below the main diagonal (and optionally the diagonal itself) are set to zero.
/// The diagonal parameter specifies the offset: 0 for main diagonal, 1 for one above, -1 for one below.
///
/// # Arguments
/// * `tensor` - Input square matrix tensor
/// * `diagonal` - Diagonal offset (0 = main, >0 = above main, <0 = below main)
///
/// # Example
/// ```ignore
/// let matrix = Tensor::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], (3, 3), &Device::Cpu)?;
/// let upper = triu(&matrix, 0)?;  // Keep main diagonal and above
/// // Result: [[1, 2, 3],
/// //          [0, 5, 6],
/// //          [0, 0, 9]]
/// ```
pub fn triu(tensor: &Tensor, diagonal: i64) -> Result<Tensor> {
    let shape = tensor.shape();
    if shape.rank() < 2 {
        candlelight::core::bail!(
            "triu requires at least 2D tensor, got rank {}",
            shape.rank()
        );
    }

    let dims = shape.dims();
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];

    // Create a mask tensor: 1.0 for elements to keep, 0.0 for elements to zero
    let mut mask_data = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let i_signed = i as i64;
            let j_signed = j as i64;
            // Keep elements where column >= row + diagonal
            if j_signed >= i_signed + diagonal {
                mask_data[i * cols + j] = 1.0;
            }
        }
    }

    let device = tensor.device();
    let mask = Tensor::from_slice(&mask_data, (rows, cols), device)?;

    // Broadcast mask to match tensor shape if needed
    let mask = if shape.rank() > 2 {
        let broadcast_dims = shape.dims().len() - 2;
        let mut broadcasted_mask = mask;
        for _ in 0..broadcast_dims {
            broadcasted_mask = broadcasted_mask.unsqueeze(0)?;
        }
        broadcasted_mask.broadcast_as(shape.dims())?
    } else {
        mask
    };

    // Apply mask
    tensor.mul(&mask)
}

/// Creates a lower triangular matrix from a square matrix.
///
/// All elements above the main diagonal (and optionally the diagonal itself) are set to zero.
/// The diagonal parameter specifies the offset: 0 for main diagonal, 1 for one above, -1 for one below.
///
/// # Arguments
/// * `tensor` - Input square matrix tensor
/// * `diagonal` - Diagonal offset (0 = main, >0 = above main, <0 = below main)
///
/// # Example
/// ```ignore
/// let matrix = Tensor::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], (3, 3), &Device::Cpu)?;
/// let lower = tril(&matrix, 0)?;  // Keep main diagonal and below
/// // Result: [[1, 0, 0],
/// //          [4, 5, 0],
/// //          [7, 8, 9]]
/// ```
pub fn tril(tensor: &Tensor, diagonal: i64) -> Result<Tensor> {
    let shape = tensor.shape();
    if shape.rank() < 2 {
        candlelight::core::bail!(
            "tril requires at least 2D tensor, got rank {}",
            shape.rank()
        );
    }

    let dims = shape.dims();
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];

    // Create a mask tensor: 1.0 for elements to keep, 0.0 for elements to zero
    let mut mask_data = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let i_signed = i as i64;
            let j_signed = j as i64;
            // Keep elements where column <= row + diagonal
            if j_signed <= i_signed + diagonal {
                mask_data[i * cols + j] = 1.0;
            }
        }
    }

    let device = tensor.device();
    let mask = Tensor::from_slice(&mask_data, (rows, cols), device)?;

    // Broadcast mask to match tensor shape if needed
    let mask = if shape.rank() > 2 {
        let broadcast_dims = shape.dims().len() - 2;
        let mut broadcasted_mask = mask;
        for _ in 0..broadcast_dims {
            broadcasted_mask = broadcasted_mask.unsqueeze(0)?;
        }
        broadcasted_mask.broadcast_as(shape.dims())?
    } else {
        mask
    };

    // Apply mask
    tensor.mul(&mask)
}

/// Fills elements of a tensor with a value where mask is true (non-zero).
///
/// This operation is commonly used for applying attention masks, padding masks, etc.
///
/// # Arguments
/// * `tensor` - Input tensor to modify
/// * `mask` - Boolean-like mask tensor (non-zero = fill, zero = keep original)
/// * `value` - Scalar value to fill where mask is true
///
/// # Example
/// ```ignore
/// let data = Tensor::from_slice(&[1., 2., 3., 4.], (2, 2), &Device::Cpu)?;
/// let mask = Tensor::from_slice(&[1., 0., 0., 1.], (2, 2), &Device::Cpu)?;
/// let result = masked_fill(&data, &mask, -1e9)?;
/// // Result: [[-1e9, 2.0],
/// //          [3.0, -1e9]]
/// ```
pub fn masked_fill(tensor: &Tensor, mask: &Tensor, value: f64) -> Result<Tensor> {
    // Broadcast mask to match tensor shape if needed
    let mask_broadcasted = if mask.shape() != tensor.shape() {
        mask.broadcast_as(tensor.shape().dims())?
    } else {
        mask.clone()
    };

    // Create a tensor filled with the target value
    let value_tensor = Tensor::full(value as f32, tensor.shape(), tensor.device())?;

    // Use where: where mask != 0, use value_tensor, else use original tensor
    // mask_broadcasted != 0 gives us a boolean tensor
    let zero = Tensor::zeros(
        mask_broadcasted.shape(),
        mask_broadcasted.dtype(),
        tensor.device(),
    )?;
    let condition = mask_broadcasted.ne(&zero)?;

    // where(condition, value_tensor, tensor)
    condition.where_cond(&value_tensor, tensor)
}

/// Creates a causal attention mask (lower triangular with -inf above diagonal)
///
/// This is a convenience function for creating standard causal masks used in decoder-only transformers.
///
/// # Arguments
/// * `seq_len` - Sequence length (creates seq_len x seq_len mask)
/// * `device` - Device to create the mask on
///
/// # Returns
/// A tensor of shape (seq_len, seq_len) with 0.0 on and below the diagonal, -inf above
///
/// # Example
/// ```ignore
/// let mask = causal_mask(4, &Device::Cpu)?;
/// // Result: [[  0, -inf, -inf, -inf],
/// //          [  0,   0, -inf, -inf],
/// //          [  0,   0,   0, -inf],
/// //          [  0,   0,   0,   0]]
/// ```
pub fn causal_mask(seq_len: usize, device: &candlelight::core::Device) -> Result<Tensor> {
    // Create a matrix of ones
    let ones = Tensor::ones((seq_len, seq_len), candlelight::core::DType::F32, device)?;

    // Get upper triangular part (excluding diagonal)
    let upper = triu(&ones, 1)?;

    // Fill upper triangular with -inf
    masked_fill(
        &Tensor::zeros((seq_len, seq_len), candlelight::core::DType::F32, device)?,
        &upper,
        f64::NEG_INFINITY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candlelight::core::{DType, Device};

    #[test]
    fn test_triu_basic() -> Result<()> {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_slice(&data, (3, 3), &device)?;

        let upper = triu(&tensor, 0)?;
        let upper_data = upper.to_vec2::<f32>()?;

        // Expected: [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
        assert_eq!(upper_data[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(upper_data[1], vec![0.0, 5.0, 6.0]);
        assert_eq!(upper_data[2], vec![0.0, 0.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_triu_diagonal_offset() -> Result<()> {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_slice(&data, (3, 3), &device)?;

        // Diagonal = 1: exclude main diagonal
        let upper = triu(&tensor, 1)?;
        let upper_data = upper.to_vec2::<f32>()?;

        // Expected: [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
        assert_eq!(upper_data[0], vec![0.0, 2.0, 3.0]);
        assert_eq!(upper_data[1], vec![0.0, 0.0, 6.0]);
        assert_eq!(upper_data[2], vec![0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_tril_basic() -> Result<()> {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_slice(&data, (3, 3), &device)?;

        let lower = tril(&tensor, 0)?;
        let lower_data = lower.to_vec2::<f32>()?;

        // Expected: [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
        assert_eq!(lower_data[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(lower_data[1], vec![4.0, 5.0, 0.0]);
        assert_eq!(lower_data[2], vec![7.0, 8.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_masked_fill_basic() -> Result<()> {
        let device = Device::Cpu;
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let mask = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (2, 2), &device)?;

        let result = masked_fill(&data, &mask, -999.0)?;
        let result_data = result.to_vec2::<f32>()?;

        // Expected: [[-999, 2], [3, -999]]
        assert_eq!(result_data[0][0], -999.0);
        assert_eq!(result_data[0][1], 2.0);
        assert_eq!(result_data[1][0], 3.0);
        assert_eq!(result_data[1][1], -999.0);

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = causal_mask(4, &device)?;
        let mask_data = mask.to_vec2::<f32>()?;

        // Check diagonal and below are 0.0
        assert_eq!(mask_data[0][0], 0.0);
        assert_eq!(mask_data[1][0], 0.0);
        assert_eq!(mask_data[1][1], 0.0);
        assert_eq!(mask_data[2][2], 0.0);
        assert_eq!(mask_data[3][3], 0.0);

        // Check above diagonal is -inf
        assert_eq!(mask_data[0][1], f32::NEG_INFINITY);
        assert_eq!(mask_data[0][2], f32::NEG_INFINITY);
        assert_eq!(mask_data[1][2], f32::NEG_INFINITY);
        assert_eq!(mask_data[2][3], f32::NEG_INFINITY);

        Ok(())
    }

    #[test]
    fn test_triu_3d_tensor() -> Result<()> {
        let device = Device::Cpu;
        // Create a 2x3x3 tensor (batch of 2 matrices)
        let data: Vec<f32> = (1..=18).map(|x| x as f32).collect();
        let tensor = Tensor::from_slice(&data, (2, 3, 3), &device)?;

        let upper = triu(&tensor, 0)?;
        let upper_data = upper.to_vec3::<f32>()?;

        // First matrix: [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
        assert_eq!(upper_data[0][0], vec![1.0, 2.0, 3.0]);
        assert_eq!(upper_data[0][1], vec![0.0, 5.0, 6.0]);
        assert_eq!(upper_data[0][2], vec![0.0, 0.0, 9.0]);

        // Second matrix: [[10, 11, 12], [0, 14, 15], [0, 0, 18]]
        assert_eq!(upper_data[1][0], vec![10.0, 11.0, 12.0]);
        assert_eq!(upper_data[1][1], vec![0.0, 14.0, 15.0]);
        assert_eq!(upper_data[1][2], vec![0.0, 0.0, 18.0]);

        Ok(())
    }
}
