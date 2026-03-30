//! GGUF weight extraction and manipulation utilities
//!
//! This module provides low-level operations for reading and writing tensor weights
//! in GGUF format, with support for all quantization types.

use super::{GgmlDType, dequantize_tensor};
use crate::gguf::Content;
use anyhow::{Context, Result};

/// Extract a tensor's raw bytes from GGUF
///
/// # Arguments
///
/// * `content` - GGUF content (memory-mapped)
/// * `tensor_name` - Name of tensor to extract
///
/// # Returns
///
/// Tuple of (quantized bytes, dtype, shape)
pub fn extract_tensor_bytes<'a>(
    content: &'a Content,
    tensor_name: &str,
) -> Result<(&'a [u8], GgmlDType, Vec<u64>)> {
    // Find tensor info
    let tensor_info = content
        .lightning_tensor_infos()
        .iter()
        .find(|t| t.name == tensor_name)
        .with_context(|| format!("Tensor not found: {}", tensor_name))?;

    // Get dtype
    let dtype = GgmlDType::from_u32(tensor_info.tensor_type)
        .with_context(|| format!("Unknown quantization type: {}", tensor_info.tensor_type))?;

    // Calculate tensor size in bytes
    let elem_count: usize = tensor_info.dimensions.iter().product::<u64>() as usize;
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let num_blocks = (elem_count + block_size - 1) / block_size;
    let tensor_size_bytes = num_blocks * type_size;

    // Extract bytes from mmap
    let mmap = content.raw_mmap();
    let start = content.tensor_data_offset() as usize + tensor_info.offset as usize;
    let end = start + tensor_size_bytes;

    if end > mmap.len() {
        anyhow::bail!(
            "Tensor {} extends beyond file: start={}, end={}, file_size={}",
            tensor_name,
            start,
            end,
            mmap.len()
        );
    }

    let bytes = &mmap[start..end];

    Ok((bytes, dtype, tensor_info.dimensions.clone()))
}

/// Extract and dequantize a tensor to F32
///
/// # Arguments
///
/// * `content` - GGUF content (memory-mapped)
/// * `tensor_name` - Name of tensor to extract
///
/// # Returns
///
/// Tuple of (F32 weights, shape)
pub fn extract_tensor_f32(content: &Content, tensor_name: &str) -> Result<(Vec<f32>, Vec<u64>)> {
    let (bytes, dtype, shape) = extract_tensor_bytes(content, tensor_name)?;
    let elem_count: usize = shape.iter().product::<u64>() as usize;

    let weights = dequantize_tensor(bytes, dtype, elem_count)
        .with_context(|| format!("Failed to dequantize tensor: {}", tensor_name))?;

    Ok((weights, shape))
}

/// Apply a mask to F32 weights (zero out pruned weights)
///
/// # Arguments
///
/// * `weights` - F32 weights (will be modified in-place)
/// * `mask` - Boolean mask (true = keep, false = prune)
///
/// # Returns
///
/// Number of weights pruned
pub fn apply_mask_inplace(weights: &mut [f32], mask: &[bool]) -> Result<usize> {
    if weights.len() != mask.len() {
        anyhow::bail!(
            "Weight and mask size mismatch: weights={}, mask={}",
            weights.len(),
            mask.len()
        );
    }

    let mut pruned_count = 0;
    for (weight, &keep) in weights.iter_mut().zip(mask.iter()) {
        if !keep {
            *weight = 0.0;
            pruned_count += 1;
        }
    }

    Ok(pruned_count)
}

/// Get list of all tensor names in GGUF
pub fn list_tensor_names(content: &Content) -> Vec<String> {
    content
        .lightning_tensor_infos()
        .iter()
        .map(|t| t.name.clone())
        .collect()
}

/// Get total model size in bytes
pub fn calculate_model_size(content: &Content) -> usize {
    let mut total_bytes = 0;

    for tensor_info in content.lightning_tensor_infos() {
        if let Some(dtype) = GgmlDType::from_u32(tensor_info.tensor_type) {
            let elem_count: usize = tensor_info.dimensions.iter().product::<u64>() as usize;
            let block_size = dtype.block_size();
            let type_size = dtype.type_size();
            let num_blocks = (elem_count + block_size - 1) / block_size;
            total_bytes += num_blocks * type_size;
        }
    }

    total_bytes
}

/// Tensor metadata for inspection
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: GgmlDType,
    pub shape: Vec<u64>,
    pub elem_count: usize,
    pub size_bytes: usize,
}

/// Get metadata for all tensors in GGUF
pub fn get_all_tensor_metadata(content: &Content) -> Vec<TensorMetadata> {
    content
        .lightning_tensor_infos()
        .iter()
        .filter_map(|tensor_info| {
            let dtype = GgmlDType::from_u32(tensor_info.tensor_type)?;
            let elem_count: usize = tensor_info.dimensions.iter().product::<u64>() as usize;
            let block_size = dtype.block_size();
            let type_size = dtype.type_size();
            let num_blocks = (elem_count + block_size - 1) / block_size;
            let size_bytes = num_blocks * type_size;

            Some(TensorMetadata {
                name: tensor_info.name.clone(),
                dtype,
                shape: tensor_info.dimensions.clone(),
                elem_count,
                size_bytes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_application() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, false, true, false, true];

        let pruned = apply_mask_inplace(&mut weights, &mask).unwrap();

        assert_eq!(pruned, 2);
        assert_eq!(weights, vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    }

    #[test]
    fn test_mask_mismatch() {
        let mut weights = vec![1.0, 2.0, 3.0];
        let mask = vec![true, false];

        let result = apply_mask_inplace(&mut weights, &mask);
        assert!(result.is_err());
    }
}
