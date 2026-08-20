//! Tensor serialization codec for KV cache tiered storage
//!
//! Serializes and deserializes Candle tensors to/from raw bytes for
//! disk-tier storage via `infra-storage` backends.
//!
//! # Wire Format
//!
//! ```text
//! [dtype: u8] [ndim: u32] [dim0: u64 ... dimN: u64] [raw data bytes]
//! ```
//!
//! DType encoding:
//! - 0: F32
//! - 1: F16
//! - 2: BF16
//! - 3: U32
//! - 4: U8
//!
//! Tensors are moved to CPU before serialization if on GPU.

use candlelight::core::{DType, Device, Result, Tensor};

/// Encode a DType as a single byte.
fn dtype_to_byte(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::U32 => 3,
        DType::U8 => 4,
        _ => 255, // Unknown
    }
}

/// Decode a byte back to a DType.
fn byte_to_dtype(byte: u8) -> Result<DType> {
    match byte {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::U32),
        4 => Ok(DType::U8),
        _ => Err(candlelight::core::Error::Msg(format!(
            "Unknown dtype byte: {}",
            byte
        ))),
    }
}

/// Serialize a tensor to raw bytes.
///
/// The tensor is moved to CPU if it's on a GPU device.
/// Format: `[dtype:u8][ndim:u32][dims:u64*ndim][raw_data]`
///
/// # Errors
///
/// Returns error if tensor data cannot be extracted.
pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    // Move to CPU if needed
    let cpu_tensor = if tensor.device().is_cpu() {
        tensor.clone()
    } else {
        tensor.to_device(&Device::Cpu)?
    };

    let dtype = cpu_tensor.dtype();
    let shape = cpu_tensor.dims();
    let ndim = shape.len();

    // Convert tensor data to raw bytes based on dtype
    let flat = cpu_tensor.flatten_all()?;
    let raw_data: Vec<u8> = match dtype {
        DType::F32 => {
            let vals: Vec<f32> = flat.to_vec1()?;
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        DType::U32 => {
            let vals: Vec<u32> = flat.to_vec1()?;
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        DType::U8 => {
            let vals: Vec<u8> = flat.to_vec1()?;
            vals
        }
        DType::F16 | DType::BF16 => {
            // Store half-precision as F32 bytes for portability
            // (avoids requiring `half` crate as direct dependency)
            // The dtype header preserves the original type for restoration
            let vals: Vec<f32> = flat.to_dtype(DType::F32)?.to_vec1()?;
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        _ => {
            return Err(candlelight::core::Error::Msg(format!(
                "Unsupported dtype for serialization: {:?}",
                dtype
            )));
        }
    };

    // Calculate total size: 1 (dtype) + 4 (ndim) + 8*ndim (dims) + data_len
    let header_size = 1 + 4 + 8 * ndim;
    let mut bytes = Vec::with_capacity(header_size + raw_data.len());

    // Write dtype
    bytes.push(dtype_to_byte(dtype));

    // Write ndim as u32 little-endian
    bytes.extend_from_slice(&(ndim as u32).to_le_bytes());

    // Write each dimension as u64 little-endian
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }

    // Write raw data
    bytes.extend_from_slice(&raw_data);

    Ok(bytes)
}

/// Deserialize a tensor from raw bytes.
///
/// The tensor is created on the specified device.
///
/// # Arguments
///
/// * `bytes` - Raw bytes produced by `tensor_to_bytes`
/// * `device` - Target device (CPU or GPU)
///
/// # Errors
///
/// Returns error if bytes are malformed or insufficient.
pub fn tensor_from_bytes(bytes: &[u8], device: &Device) -> Result<Tensor> {
    if bytes.len() < 5 {
        return Err(candlelight::core::Error::Msg(
            "Tensor bytes too short for header".to_string(),
        ));
    }

    // Read dtype
    let dtype = byte_to_dtype(bytes[0])?;

    // Read ndim
    let ndim = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;

    // Validate header length
    let dims_start = 5;
    let dims_end = dims_start + 8 * ndim;
    if bytes.len() < dims_end {
        return Err(candlelight::core::Error::Msg(format!(
            "Tensor bytes too short for {} dimensions",
            ndim
        )));
    }

    // Read dimensions
    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let offset = dims_start + 8 * i;
        let dim = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        shape.push(dim);
    }

    // Raw data starts after header
    let data = &bytes[dims_end..];

    // Create tensor on CPU first, then move to target device
    let numel: usize = shape.iter().product();
    // For F16/BF16, data is stored as F32 (4 bytes per element)
    let bytes_per_element = match dtype {
        DType::F16 | DType::BF16 => 4, // Stored as F32
        _ => dtype.size_in_bytes(),
    };
    let expected_bytes = numel * bytes_per_element;
    if data.len() < expected_bytes {
        return Err(candlelight::core::Error::Msg(format!(
            "Tensor data too short: expected {} bytes for {} elements of {:?}, got {}",
            expected_bytes,
            numel,
            dtype,
            data.len()
        )));
    }

    // Reconstruct tensor from raw bytes based on dtype
    let tensor = match dtype {
        DType::F32 => {
            let values: Vec<f32> = data[..expected_bytes]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec(values, shape.as_slice(), &Device::Cpu)?
        }
        DType::U32 => {
            let values: Vec<u32> = data[..expected_bytes]
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec(values, shape.as_slice(), &Device::Cpu)?
        }
        DType::U8 => {
            let values: Vec<u8> = data[..expected_bytes].to_vec();
            Tensor::from_vec(values, shape.as_slice(), &Device::Cpu)?
        }
        DType::F16 | DType::BF16 => {
            // Half-precision was stored as F32 bytes (4 bytes per element)
            let f32_bytes = numel * 4;
            if data.len() < f32_bytes {
                return Err(candlelight::core::Error::Msg(format!(
                    "Tensor data too short for half-as-f32: expected {} bytes, got {}",
                    f32_bytes,
                    data.len()
                )));
            }
            let f32_values: Vec<f32> = data[..f32_bytes]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let t = Tensor::from_vec(f32_values, shape.as_slice(), &Device::Cpu)?;
            t.to_dtype(dtype)?
        }
        _ => {
            return Err(candlelight::core::Error::Msg(format!(
                "Unsupported dtype for deserialization: {:?}",
                dtype
            )));
        }
    };

    // Move to target device if not CPU
    if device.is_cpu() {
        Ok(tensor)
    } else {
        tensor.to_device(device)
    }
}

/// Serialize a vector of per-layer (K, V) tensor pairs.
///
/// Format: `[num_layers: u32] [k_len: u64] [k_bytes] [v_len: u64] [v_bytes] ...`
pub fn kv_layers_to_bytes(layers: &[(Tensor, Tensor)]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();

    // Write number of layers
    bytes.extend_from_slice(&(layers.len() as u32).to_le_bytes());

    for (k, v) in layers {
        let k_bytes = tensor_to_bytes(k)?;
        let v_bytes = tensor_to_bytes(v)?;

        // Write K length and data
        bytes.extend_from_slice(&(k_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&k_bytes);

        // Write V length and data
        bytes.extend_from_slice(&(v_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&v_bytes);
    }

    Ok(bytes)
}

/// Deserialize per-layer (K, V) tensor pairs from raw bytes.
pub fn kv_layers_from_bytes(bytes: &[u8], device: &Device) -> Result<Vec<(Tensor, Tensor)>> {
    if bytes.len() < 4 {
        return Err(candlelight::core::Error::Msg(
            "KV layers bytes too short".to_string(),
        ));
    }

    let num_layers = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut offset = 4;
    let mut layers = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        // Read K
        if offset + 8 > bytes.len() {
            return Err(candlelight::core::Error::Msg(format!(
                "KV bytes truncated at layer {} K length",
                layer_idx
            )));
        }
        let k_len = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;

        if offset + k_len > bytes.len() {
            return Err(candlelight::core::Error::Msg(format!(
                "KV bytes truncated at layer {} K data",
                layer_idx
            )));
        }
        let k = tensor_from_bytes(&bytes[offset..offset + k_len], device)?;
        offset += k_len;

        // Read V
        if offset + 8 > bytes.len() {
            return Err(candlelight::core::Error::Msg(format!(
                "KV bytes truncated at layer {} V length",
                layer_idx
            )));
        }
        let v_len = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;

        if offset + v_len > bytes.len() {
            return Err(candlelight::core::Error::Msg(format!(
                "KV bytes truncated at layer {} V data",
                layer_idx
            )));
        }
        let v = tensor_from_bytes(&bytes[offset..offset + v_len], device)?;
        offset += v_len;

        layers.push((k, v));
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_roundtrip() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();

        let bytes = tensor_to_bytes(&tensor).unwrap();
        let restored = tensor_from_bytes(&bytes, &Device::Cpu).unwrap();

        assert_eq!(restored.dims(), &[2, 3]);
        let restored_data: Vec<f32> = restored.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(restored_data, data);
    }

    #[test]
    fn test_u32_roundtrip() {
        let data = vec![10_u32, 20, 30];
        let tensor = Tensor::from_vec(data.clone(), &[3], &Device::Cpu).unwrap();

        let bytes = tensor_to_bytes(&tensor).unwrap();
        let restored = tensor_from_bytes(&bytes, &Device::Cpu).unwrap();

        assert_eq!(restored.dims(), &[3]);
        let restored_data: Vec<u32> = restored.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(restored_data, data);
    }

    #[test]
    fn test_4d_tensor_roundtrip() {
        // Typical KV cache shape: [batch, heads, seq, head_dim]
        let shape = [1, 4, 8, 16];
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.01).collect();
        let tensor = Tensor::from_vec(data.clone(), &shape, &Device::Cpu).unwrap();

        let bytes = tensor_to_bytes(&tensor).unwrap();
        let restored = tensor_from_bytes(&bytes, &Device::Cpu).unwrap();

        assert_eq!(restored.dims(), &shape);
        let restored_data: Vec<f32> = restored.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in data.iter().zip(restored_data.iter()) {
            assert!((a - b).abs() < 1e-6, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_kv_layers_roundtrip() {
        let k1 = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2], &Device::Cpu).unwrap();
        let v1 = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[1, 2, 2], &Device::Cpu).unwrap();
        let k2 =
            Tensor::from_vec(vec![9.0_f32, 10.0, 11.0, 12.0], &[1, 2, 2], &Device::Cpu).unwrap();
        let v2 =
            Tensor::from_vec(vec![13.0_f32, 14.0, 15.0, 16.0], &[1, 2, 2], &Device::Cpu).unwrap();

        let layers = vec![(k1, v1), (k2, v2)];

        let bytes = kv_layers_to_bytes(&layers).unwrap();
        let restored = kv_layers_from_bytes(&bytes, &Device::Cpu).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].0.dims(), &[1, 2, 2]);
        assert_eq!(restored[1].1.dims(), &[1, 2, 2]);

        let k1_data: Vec<f32> = restored[0].0.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k1_data, vec![1.0, 2.0, 3.0, 4.0]);

        let v2_data: Vec<f32> = restored[1].1.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v2_data, vec![13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn test_empty_tensor_error() {
        let result = tensor_from_bytes(&[], &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_data_error() {
        let tensor = Tensor::from_vec(vec![1.0_f32, 2.0], &[2], &Device::Cpu).unwrap();
        let mut bytes = tensor_to_bytes(&tensor).unwrap();

        // Truncate the data portion
        bytes.truncate(bytes.len() - 4);

        let result = tensor_from_bytes(&bytes, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtype_byte_roundtrip() {
        for dtype in &[DType::F32, DType::F16, DType::BF16, DType::U32, DType::U8] {
            let byte = dtype_to_byte(*dtype);
            let restored = byte_to_dtype(byte).unwrap();
            assert_eq!(*dtype, restored, "DType roundtrip failed for {:?}", dtype);
        }
    }
}
