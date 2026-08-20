use anyhow::Result;
use candlelight::core::{Device, Tensor};

/// Weight sharding strategy for tensor parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Column-wise sharding (split along output dimension)
    ColumnWise,

    /// Row-wise sharding (split along input dimension)
    RowWise,

    /// Hybrid (column for some layers, row for others)
    Hybrid,
}

/// Sharded weight tensor distributed across GPUs
#[derive(Debug, Clone)]
pub struct TensorShard {
    /// Local shard on this device
    pub local_shard: Tensor,

    /// Device this shard resides on
    pub device: Device,

    /// Rank of this GPU (0 to world_size-1)
    pub rank: usize,

    /// Total number of GPUs
    pub world_size: usize,

    /// Sharding dimension (0 = row-wise, 1 = column-wise)
    pub shard_dim: usize,

    /// Original full shape (before sharding)
    pub full_shape: Vec<usize>,
}

impl TensorShard {
    /// Create sharded weights from full tensor
    ///
    /// # Arguments
    /// * `full_tensor` - The full weight tensor to shard
    /// * `devices` - Target devices for each shard
    /// * `shard_dim` - Dimension along which to shard (0 or 1)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Shard a [4096, 4096] weight matrix column-wise across 2 GPUs
    /// let full_weights = Tensor::randn(0.0, 1.0, (4096, 4096), &Device::Cpu)?;
    /// let devices = vec![Device::cuda_if_available(0)?, Device::cuda_if_available(1)?];
    /// let shards = TensorShard::from_full_tensor(&full_weights, &devices, 0)?;
    /// // Each shard: [2048, 4096]
    /// ```
    pub fn from_full_tensor(
        full_tensor: &Tensor,
        devices: &[Device],
        shard_dim: usize,
    ) -> Result<Vec<Self>> {
        let world_size = devices.len();
        let full_shape = full_tensor.dims().to_vec();

        if shard_dim >= full_shape.len() {
            anyhow::bail!(
                "Shard dimension {} out of bounds for tensor shape {:?}",
                shard_dim,
                full_shape
            );
        }

        let dim_size = full_shape[shard_dim];
        if dim_size % world_size != 0 {
            anyhow::bail!(
                "Cannot evenly shard dimension {} (size {}) across {} GPUs",
                shard_dim,
                dim_size,
                world_size
            );
        }

        let shard_size = dim_size / world_size;

        let mut shards = Vec::new();
        for (rank, device) in devices.iter().enumerate() {
            let start = rank * shard_size;
            let end = (rank + 1) * shard_size;

            // Narrow the tensor along shard dimension
            let local_shard = full_tensor.narrow(shard_dim, start, end - start)?;

            // Copy to target device
            let local_shard = local_shard.to_device(device)?;

            shards.push(Self {
                local_shard,
                device: device.clone(),
                rank,
                world_size,
                shard_dim,
                full_shape: full_shape.clone(),
            });
        }

        Ok(shards)
    }

    /// All-reduce across GPUs (sum shards and replicate result)
    ///
    /// This is the key communication primitive for tensor parallelism.
    /// Each GPU contributes its local partial result, and all GPUs receive
    /// the summed result.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Each GPU has computed partial output: [batch, hidden_shard]
    /// // After all-reduce, each GPU has full output: [batch, hidden]
    /// let full_output = TensorShard::all_reduce(&partial_outputs)?;
    /// ```
    pub fn all_reduce(shards: &[Tensor]) -> Result<Tensor> {
        if shards.is_empty() {
            anyhow::bail!("Cannot all-reduce empty shard list");
        }

        // Sum all shards
        let mut result = shards[0].clone();
        for shard in &shards[1..] {
            // Move to same device if needed
            let shard_same_device = shard.to_device(result.device())?;
            result = (result + shard_same_device)?;
        }

        Ok(result)
    }

    /// Gather shards along dimension (concatenate)
    ///
    /// Concatenate shards from all GPUs along the specified dimension.
    /// Unlike all-reduce (which sums), this preserves each shard's contribution.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Column-wise sharding: each GPU has [batch, hidden_shard]
    /// // Gather along dim=1 produces [batch, hidden_full]
    /// let full_tensor = TensorShard::gather(&shards, 1)?;
    /// ```
    pub fn gather(shards: &[Tensor], dim: usize) -> Result<Tensor> {
        if shards.is_empty() {
            anyhow::bail!("Cannot gather empty shard list");
        }

        // Move all shards to same device (first shard's device)
        let target_device = shards[0].device();
        let mut shards_same_device = Vec::new();
        for shard in shards {
            shards_same_device.push(shard.to_device(target_device)?);
        }

        Ok(Tensor::cat(&shards_same_device, dim)?)
    }

    /// Scatter tensor across GPUs along dimension
    ///
    /// Inverse of gather: split a tensor into shards and distribute to GPUs.
    pub fn scatter(tensor: &Tensor, devices: &[Device], dim: usize) -> Result<Vec<Tensor>> {
        let dim_size = tensor.dim(dim)?;
        let world_size = devices.len();

        if dim_size % world_size != 0 {
            anyhow::bail!(
                "Cannot evenly scatter dimension {} (size {}) across {} GPUs",
                dim,
                dim_size,
                world_size
            );
        }

        let shard_size = dim_size / world_size;
        let mut shards = Vec::new();

        for (rank, device) in devices.iter().enumerate() {
            let start = rank * shard_size;
            let shard = tensor.narrow(dim, start, shard_size)?;
            let shard_on_device = shard.to_device(device)?;
            shards.push(shard_on_device);
        }

        Ok(shards)
    }
}

/// Linear layer with tensor parallelism
///
/// Distributes weight matrix across multiple GPUs and uses collective
/// communication (all-reduce) to combine results.
///
/// Supports two sharding strategies:
/// - **Column-wise**: Split output features across GPUs (requires gather)
/// - **Row-wise**: Split input features across GPUs (requires all-reduce)
pub struct ShardedLinear {
    /// Weight shards across GPUs
    pub weight_shards: Vec<TensorShard>,

    /// Bias (replicated on all GPUs, or None)
    pub bias: Option<Tensor>,

    /// Sharding strategy
    pub strategy: ShardingStrategy,
}

impl ShardedLinear {
    /// Create sharded linear layer from full weights
    pub fn from_full_weights(
        weights: &Tensor,
        bias: Option<&Tensor>,
        devices: &[Device],
        strategy: ShardingStrategy,
    ) -> Result<Self> {
        let shard_dim = match strategy {
            ShardingStrategy::ColumnWise => 0, // Split output features (rows)
            ShardingStrategy::RowWise => 1,    // Split input features (cols)
            ShardingStrategy::Hybrid => {
                anyhow::bail!("Hybrid sharding strategy not yet implemented")
            }
        };

        let weight_shards = TensorShard::from_full_tensor(weights, devices, shard_dim)?;

        // Replicate bias on all GPUs
        let bias = if let Some(b) = bias {
            Some(b.clone())
        } else {
            None
        };

        Ok(Self {
            weight_shards,
            bias,
            strategy,
        })
    }

    /// Forward pass with appropriate sharding strategy
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.strategy {
            ShardingStrategy::ColumnWise => self.forward_column_wise(input),
            ShardingStrategy::RowWise => self.forward_row_wise(input),
            ShardingStrategy::Hybrid => {
                anyhow::bail!("Hybrid sharding strategy not yet implemented")
            }
        }
    }

    /// Forward pass with column-wise sharding
    ///
    /// # Algorithm
    /// Input: `[batch, in_features]`  
    /// Weight: `[out_features, in_features]` → split along `out_features`  
    /// Output: `[batch, out_features]` (after gather)
    ///
    /// Each GPU computes:
    /// ```text
    /// local_output[i] = input @ weight_shard[i]^T  (shape: [batch, out_features/N])
    /// ```
    ///
    /// Then concatenate along output dimension:
    /// ```text
    /// output = concat(local_output[0], local_output[1], ..., local_output[N-1])
    /// ```
    pub fn forward_column_wise(&self, input: &Tensor) -> Result<Tensor> {
        let mut local_outputs = Vec::new();

        // Each GPU computes its local output
        for shard in &self.weight_shards {
            // Move input to GPU
            let local_input = input.to_device(&shard.device)?;

            // Local matmul: [batch, in_features] @ [out_features_shard, in_features]^T
            // Result: [batch, out_features_shard]
            let local_output = local_input.matmul(&shard.local_shard.t()?)?;

            local_outputs.push(local_output);
        }

        // Concatenate along output dimension (dim=1)
        let mut output = TensorShard::gather(&local_outputs, 1)?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            let bias_on_device = bias.to_device(output.device())?;
            output = (output + bias_on_device)?;
        }

        Ok(output)
    }

    /// Forward pass with row-wise sharding
    ///
    /// # Algorithm
    /// Input: `[batch, in_features]` → split along `in_features`  
    /// Weight: `[out_features, in_features]` → split along `in_features`  
    /// Output: `[batch, out_features]` (after all-reduce)
    ///
    /// Each GPU computes:
    /// ```text
    /// local_output[i] = input_shard[i] @ weight_shard[i]^T  (shape: [batch, out_features])
    /// ```
    ///
    /// Then all-reduce (sum) across GPUs:
    /// ```text
    /// output = sum(local_output[0], local_output[1], ..., local_output[N-1])
    /// ```
    pub fn forward_row_wise(&self, input: &Tensor) -> Result<Tensor> {
        let world_size = self.weight_shards.len();
        let in_features = input.dim(1)?;
        let shard_size = in_features / world_size;

        let mut local_outputs = Vec::new();

        // Each GPU computes partial output with its input/weight shard
        for (rank, shard) in self.weight_shards.iter().enumerate() {
            let start = rank * shard_size;
            let end = (rank + 1) * shard_size;

            // Narrow input along feature dimension
            let input_shard = input.narrow(1, start, end - start)?;
            let input_shard = input_shard.to_device(&shard.device)?;

            // Local matmul: [batch, in_features_shard] @ [out_features, in_features_shard]^T
            // Result: [batch, out_features]
            let local_output = input_shard.matmul(&shard.local_shard.t()?)?;

            local_outputs.push(local_output);
        }

        // All-reduce (sum partial outputs)
        let mut output = TensorShard::all_reduce(&local_outputs)?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            let bias_on_device = bias.to_device(output.device())?;
            output = (output + bias_on_device)?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires multi-GPU setup
    fn test_tensor_shard_column_wise() -> Result<()> {
        let devices = vec![Device::cuda_if_available(0)?, Device::cuda_if_available(1)?];

        // Create [4, 8] tensor, shard along dim=0 (rows)
        let full_tensor = Tensor::randn(0.0f32, 1.0, (4, 8), &Device::Cpu)?;
        let shards = TensorShard::from_full_tensor(&full_tensor, &devices, 0)?;

        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].local_shard.dims(), &[2, 8]); // Each shard: [2, 8]
        assert_eq!(shards[1].local_shard.dims(), &[2, 8]);

        // Gather and verify
        let gathered = TensorShard::gather(
            &shards
                .iter()
                .map(|s| s.local_shard.clone())
                .collect::<Vec<_>>(),
            0,
        )?;
        assert_eq!(gathered.dims(), &[4, 8]);

        Ok(())
    }

    #[test]
    #[ignore] // Requires multi-GPU setup
    fn test_sharded_linear_column_wise() -> Result<()> {
        let devices = vec![Device::cuda_if_available(0)?, Device::cuda_if_available(1)?];

        // Create linear layer: input=8, output=4
        let weights = Tensor::randn(0.0f32, 1.0, (4, 8), &Device::Cpu)?;
        let bias = Some(Tensor::zeros(
            (4,),
            candlelight::core::DType::F32,
            &Device::Cpu,
        )?);

        let sharded_linear = ShardedLinear::from_full_weights(
            &weights,
            bias.as_ref(),
            &devices,
            ShardingStrategy::ColumnWise,
        )?;

        // Input: [2, 8] (batch=2, features=8)
        let input = Tensor::randn(0.0f32, 1.0, (2, 8), &Device::Cpu)?;

        // Forward pass
        let output = sharded_linear.forward(&input)?;
        assert_eq!(output.dims(), &[2, 4]); // [batch, output_features]

        Ok(())
    }
}
