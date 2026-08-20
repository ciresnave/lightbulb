//! Apply pruning manifests to GGUF model files
//!
//! This module provides the core functionality to apply pruning masks to
//! quantized GGUF model files by dequantizing, masking, and re-quantizing weights.

use crate::gguf::Content;
use crate::pruning::PruningManifest;
use crate::pruning::name_mapping::TensorNameMapper;
use crate::quantization::{GgmlDType, dequantize_tensor, gguf_ops, quantize_tensor};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Apply a pruning manifest to a GGUF model file
///
/// **IMPORTANT**: This is a prototype implementation that currently stores pruned weights
/// in F32 format (not re-quantized). This means the output file will be 4-8× larger than
/// the input. Full quantization support (dequantize → mask → requantize) is TODO.
///
/// This function loads a GGUF model, applies pruning masks to specified layers,
/// and writes the pruned model to a new file. Weights are dequantized and masked,
/// but not yet re-quantized (stored as F32).
///
/// # Arguments
///
/// * `input_path` - Path to input GGUF model file
/// * `manifest` - Pruning manifest containing masks for layers
/// * `output_path` - Path to output pruned GGUF model file (will be F32 format)
///
/// # Returns
///
/// Statistics about the pruning operation
///
/// # Example
///
/// ```no_run
/// use lightbulb::pruning::{PruningManifest, apply_manifest_to_gguf};
/// use std::path::Path;
///
/// let manifest = PruningManifest::load(Path::new("pruning_manifest.json"))?;
/// let stats = apply_manifest_to_gguf(
///     Path::new("model.gguf"),
///     &manifest,
///     Path::new("model_pruned_f32.gguf"),  // Note: F32 format (larger)
/// )?;
/// println!("Pruned {} weights ({:.1}% sparsity)", stats.weights_pruned, stats.sparsity_percent);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn apply_manifest_to_gguf(
    input_path: &Path,
    manifest: &PruningManifest,
    output_path: &Path,
) -> Result<PruningStats> {
    println!("🔧 Applying pruning manifest to GGUF model...");
    println!("   Input:  {}", input_path.display());
    println!("   Output: {}", output_path.display());

    // Load GGUF model
    let content = Content::read(input_path)?;
    let tensor_infos = content.lightning_tensor_infos();

    // Build dynamic name mapper
    let tensor_names: Vec<String> = tensor_infos.iter().map(|t| t.name.clone()).collect();
    let name_mapper = TensorNameMapper::from_tensor_names(&tensor_names)?;

    println!(
        "   🔍 Detected architecture: {:?}",
        name_mapper.architecture
    );
    println!("   📊 Found {} layers", name_mapper.layer_indices.len());

    let mut total_params = 0usize;
    let mut pruned_params = 0usize;
    let mut tensors_modified = 0usize;

    // Create output file
    let mut output_file = std::fs::File::create(output_path)?;

    // Write GGUF header (copy from input)
    write_gguf_header(&content, &mut output_file)?;

    // Process each tensor
    for (tensor_idx, tensor_info) in tensor_infos.iter().enumerate() {
        let tensor_name = &tensor_info.name;
        let shape = &tensor_info.dimensions;
        let dtype = GgmlDType::from_u32(tensor_info.tensor_type).context("Unknown tensor type")?;

        println!(
            "   Processing tensor {}: {} ({:?}, shape={:?})",
            tensor_idx, tensor_name, dtype, shape
        );

        // Extract raw quantized bytes
        let (raw_bytes, _, _) = gguf_ops::extract_tensor_bytes(&content, tensor_name)?;

        // Try to find mask - first check direct name, then try mapping
        let mask = manifest.masks.get(tensor_name).or_else(|| {
            // Try to map from abstract names in manifest
            for (abstract_name, mask) in &manifest.masks {
                if let Some(mapped_name) = name_mapper.map_name(abstract_name) {
                    if mapped_name == *tensor_name {
                        return Some(mask);
                    }
                }
            }
            None
        });

        if let Some(mask) = mask {
            // Convert mask Tensor to Vec<bool>
            let mask_data = mask
                .mask
                .to_vec1::<f32>()
                .context("Failed to convert mask tensor")?;
            let mask_bool: Vec<bool> = mask_data.iter().map(|&x| x > 0.5).collect();

            println!(
                "      ✓ Applying pruning mask ({} elements)",
                mask_bool.len()
            );

            // Dequantize to F32
            let elem_count: usize = shape.iter().map(|&x| x as usize).product();
            let mut weights = dequantize_tensor(raw_bytes, dtype, elem_count)?;

            // Apply mask
            let pruned = gguf_ops::apply_mask_inplace(&mut weights, &mask_bool)?;

            // Re-quantize
            let quantized_bytes = quantize_tensor(&weights, dtype)?;

            // Write quantized bytes
            output_file.write_all(&quantized_bytes)?;

            total_params += elem_count;
            pruned_params += pruned;
            tensors_modified += 1;

            println!(
                "      → Pruned {}/{} weights ({:.2}%)",
                pruned,
                elem_count,
                (pruned as f32 / elem_count as f32) * 100.0
            );
        } else {
            // No mask - copy tensor as-is
            output_file.write_all(raw_bytes)?;

            let elem_count: usize = shape.iter().map(|&x| x as usize).product();
            total_params += elem_count;
        }
    }

    let achieved_sparsity = if total_params > 0 {
        (pruned_params as f32) / (total_params as f32)
    } else {
        0.0
    };

    println!("✓ Pruning complete!");
    println!("   Tensors modified: {}", tensors_modified);
    println!("   Total parameters: {}", total_params);
    println!("   Pruned parameters: {}", pruned_params);
    println!("   Achieved sparsity: {:.2}%", achieved_sparsity * 100.0);

    Ok(PruningStats {
        total_params,
        pruned_params,
        achieved_sparsity,
        tensors_modified,
    })
}

/// Write GGUF file header
///
/// Copies the header and metadata from the input GGUF file to the output.
/// This preserves model architecture, metadata, and tensor info.
/// Write GGUF file header
///
/// Copies the header and metadata from the input GGUF file to the output.
/// This preserves model architecture, metadata, and tensor info.
fn write_gguf_header(content: &Content, output: &mut File) -> Result<()> {
    // Copy the entire header section from input to output
    let mmap = content.raw_mmap();
    let header_size = content.tensor_data_offset() as usize;
    let header_bytes = &mmap[0..header_size];
    output
        .write_all(header_bytes)
        .context("Failed to write GGUF header")?;
    Ok(())
}

/// Statistics from pruning operation
#[derive(Debug, Clone)]
pub struct PruningStats {
    /// Total number of parameters in model
    pub total_params: usize,

    /// Number of parameters pruned (set to zero)
    pub pruned_params: usize,

    /// Overall sparsity achieved (0.0 to 1.0)
    pub achieved_sparsity: f32,

    /// Number of tensors modified
    pub tensors_modified: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: This test is disabled pending refactor of apply_mask_to_weights
    // The function signature has changed and needs updating.
    /*
    #[test]
    fn test_mask_application() {
        use candlelight::core::{Device, Tensor as CandleTensor};
        use crate::pruning::PruningLoader;

        let mut weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask_tensor =
            CandleTensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0, 1.0], (5,), &Device::Cpu).unwrap();

        let mask = PruningMask {
            mask: mask_tensor,
            sparsity: 0.4,
            pattern: crate::pruning::StructuredPattern::Unstructured,
            layer_id: "test_layer".to_string(),
        };

        let pruned = PruningLoader::apply_mask_to_weights(&mut weights, &mask).unwrap();

        assert_eq!(pruned, 2);
        assert_eq!(weights, vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    }
    */
}
