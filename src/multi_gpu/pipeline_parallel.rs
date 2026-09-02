use anyhow::Result;
use candlelight::core::{Device, Tensor};
use std::collections::VecDeque;

// Forward declare types we'll use
use crate::cache::ParallelCacheBuilder;
use crate::cache::ParallelKvCache;
use crate::model::batch_metadata::BatchMetadata;
use crate::model::custom_transformer::BatchedTransformer;
use crate::pruning::name_mapping::TensorNameMapper;

/// Single stage in pipeline parallelism
///
/// A pipeline stage holds a subset of model layers and processes
/// micro-batches sequentially, forwarding results to the next stage.
pub struct PipelineStage {
    /// GPU device for this stage
    pub device: Device,

    /// Stage ID (0 to num_stages-1)
    pub stage_id: usize,

    /// Transformer layers in this stage (layer indices)
    pub layers: Vec<usize>,

    /// Input buffer for micro-batches
    pub input_buffer: VecDeque<Tensor>,

    /// Output buffer for next stage
    pub output_buffer: VecDeque<Tensor>,
}

impl PipelineStage {
    /// Create a new pipeline stage
    pub fn new(device: Device, stage_id: usize, layers: Vec<usize>) -> Self {
        Self {
            device,
            stage_id,
            layers,
            input_buffer: VecDeque::new(),
            output_buffer: VecDeque::new(),
        }
    }

    /// Process one micro-batch through this stage's layers
    ///
    /// Forwards the micro-batch through layers `[layers[0], layers[-1]+1)` using
    /// the provided model's forward_layers method.
    ///
    /// # Arguments
    /// * `input` - Input hidden states [batch, seq, hidden_size]
    /// * `model` - The transformer model (must contain layers for this stage)
    /// * `index_pos` - RoPE position index
    /// * `cache_builder` - Cache builder for position tracking
    /// * `caches` - KV caches (one per model layer)
    /// * `metadata` - Batch metadata
    pub fn process_micro_batch_with_model(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
        index_pos: usize,
        cache_builder: &mut ParallelCacheBuilder,
        caches: &mut [ParallelKvCache],
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        // Move input to this stage's device
        let input = input.to_device(&self.device)?;

        if self.layers.is_empty() {
            return Ok(input);
        }

        // Get layer range for this stage
        let layer_start = *self.layers.first().unwrap();
        let layer_end = self.layers.last().unwrap() + 1;

        // Forward through assigned layers
        Ok(model.forward_layers(
            &input,
            layer_start,
            layer_end,
            index_pos,
            cache_builder,
            caches,
            metadata,
        )?)
    }

    /// Process one micro-batch through this stage (placeholder for standalone use)
    ///
    /// This is a simplified version for when model integration is not available.
    /// For actual pipeline parallelism, use process_micro_batch_with_model instead.
    pub fn process_micro_batch(&mut self, input: Tensor) -> Result<Tensor> {
        // Move input to this stage's device
        let input = input.to_device(&self.device)?;

        // Placeholder: just pass through
        // Real implementation requires model reference (see process_micro_batch_with_model)
        Ok(input)
    }

    /// Number of layers in this stage
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Pipeline scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStrategy {
    /// GPipe: strict forward/backward separation (simpler, more bubbles)
    GPipe,

    /// PipeDream: interleaved forward/backward (less bubbles, more complex)
    #[allow(dead_code)]
    PipeDream,

    /// Interleaved 1F1B: one forward, one backward per stage (balance)
    #[allow(dead_code)]
    Interleaved1F1B,
}

/// Pipeline parallel execution scheduler
///
/// Manages micro-batch scheduling across pipeline stages to minimize
/// pipeline bubbles and maximize throughput.
pub struct PipelineScheduler {
    /// All pipeline stages
    pub stages: Vec<PipelineStage>,

    /// Micro-batch size
    pub micro_batch_size: usize,

    /// Scheduling strategy (GPipe, PipeDream, interleaved)
    pub strategy: PipelineStrategy,

    /// Optional name mapper for architecture-aware layer detection (M3.6.1)
    pub name_mapper: Option<TensorNameMapper>,
}

impl PipelineScheduler {
    /// Create a new pipeline scheduler
    ///
    /// # Arguments
    /// * `num_stages` - Number of pipeline stages (GPUs)
    /// * `num_layers` - Total number of model layers
    /// * `devices` - GPU devices for each stage
    /// * `micro_batch_size` - Micro-batch size for scheduling
    /// * `strategy` - Scheduling strategy
    ///
    /// # Example
    /// ```rust,ignore
    /// let devices = vec![
    ///     Device::cuda_if_available(0)?,
    ///     Device::cuda_if_available(1)?,
    ///     Device::cuda_if_available(2)?,
    ///     Device::cuda_if_available(3)?,
    /// ];
    /// let scheduler = PipelineScheduler::new(
    ///     4, // 4 stages
    ///     80, // 80 layers total
    ///     devices,
    ///     4, // micro-batch size
    ///     PipelineStrategy::GPipe,
    /// )?;
    /// // Each stage gets 20 layers
    /// ```
    pub fn new(
        num_stages: usize,
        num_layers: usize,
        devices: Vec<Device>,
        micro_batch_size: usize,
        strategy: PipelineStrategy,
    ) -> Result<Self> {
        if devices.len() != num_stages {
            anyhow::bail!(
                "Number of devices ({}) must match number of stages ({})",
                devices.len(),
                num_stages
            );
        }

        if num_layers < num_stages {
            anyhow::bail!(
                "Number of layers ({}) must be >= number of stages ({})",
                num_layers,
                num_stages
            );
        }

        // Distribute layers evenly across stages
        let layers_per_stage = num_layers / num_stages;
        let mut stages = Vec::new();

        for (stage_id, device) in devices.into_iter().enumerate() {
            let start_layer = stage_id * layers_per_stage;
            let end_layer = if stage_id == num_stages - 1 {
                num_layers // Last stage gets remaining layers
            } else {
                (stage_id + 1) * layers_per_stage
            };

            let layers = (start_layer..end_layer).collect();
            stages.push(PipelineStage::new(device, stage_id, layers));
        }

        Ok(Self {
            stages,
            micro_batch_size,
            strategy,
            name_mapper: None,
        })
    }

    /// Create pipeline scheduler with architecture-aware layer detection (M3.6.1)
    ///
    /// This constructor automatically detects the number and structure of layers
    /// from the model's tensor names, eliminating the need to hardcode layer counts.
    ///
    /// # Arguments
    /// * `num_stages` - Number of pipeline stages (GPUs)
    /// * `tensor_names` - Model tensor names for architecture detection
    /// * `devices` - GPU devices for each stage
    /// * `micro_batch_size` - Micro-batch size for scheduling
    /// * `strategy` - Scheduling strategy
    ///
    /// # Example
    /// ```rust,ignore
    /// let devices = vec![
    ///     Device::cuda_if_available(0)?,
    ///     Device::cuda_if_available(1)?,
    /// ];
    /// let scheduler = PipelineScheduler::from_tensor_names(
    ///     2, // 2 stages
    ///     &model.tensor_names(), // Auto-detect layers
    ///     devices,
    ///     4, // micro-batch size
    ///     PipelineStrategy::GPipe,
    /// )?;
    /// // Automatically distributes detected layers across stages
    /// ```
    pub fn from_tensor_names(
        num_stages: usize,
        tensor_names: &[String],
        devices: Vec<Device>,
        micro_batch_size: usize,
        strategy: PipelineStrategy,
    ) -> Result<Self> {
        if devices.len() != num_stages {
            anyhow::bail!(
                "Number of devices ({}) must match number of stages ({})",
                devices.len(),
                num_stages
            );
        }

        // Create name mapper to detect architecture and layers
        let name_mapper = TensorNameMapper::from_tensor_names(tensor_names)?;

        // Get detected layer indices
        let layer_indices = &name_mapper.layer_indices;
        let num_layers = layer_indices.len();

        if num_layers < num_stages {
            anyhow::bail!(
                "Number of detected layers ({}) must be >= number of stages ({})",
                num_layers,
                num_stages
            );
        }

        // Distribute layers evenly across stages
        let layers_per_stage = num_layers / num_stages;
        let mut stages = Vec::new();

        for (stage_id, device) in devices.into_iter().enumerate() {
            let start_idx = stage_id * layers_per_stage;
            let end_idx = if stage_id == num_stages - 1 {
                num_layers // Last stage gets remaining layers
            } else {
                (stage_id + 1) * layers_per_stage
            };

            // Use actual layer indices from name mapper
            let layers = layer_indices[start_idx..end_idx].to_vec();
            stages.push(PipelineStage::new(device, stage_id, layers));
        }

        Ok(Self {
            stages,
            micro_batch_size,
            strategy,
            name_mapper: Some(name_mapper),
        })
    }

    /// Get GPU device ID for a given tensor name
    ///
    /// Uses the name mapper to determine which layer the tensor belongs to,
    /// then returns the GPU assignment for that layer.
    ///
    /// Returns None if tensor doesn't belong to a layer or name mapper not available.
    pub fn get_gpu_for_tensor(&self, tensor_name: &str) -> Option<usize> {
        let mapper = self.name_mapper.as_ref()?;

        // Try to parse layer index from tensor name
        for (layer_idx, component) in mapper.mappings.keys() {
            if let Some(concrete_name) = mapper.mappings.get(&(*layer_idx, *component)) {
                if concrete_name == tensor_name {
                    // Found the layer, now find which stage it's in
                    return self.get_stage_for_layer(*layer_idx);
                }
            }
        }

        None
    }

    /// Get stage ID (GPU) for a given layer index
    ///
    /// Returns None if layer index not found in any stage.
    pub fn get_stage_for_layer(&self, layer_idx: usize) -> Option<usize> {
        for stage in &self.stages {
            if stage.layers.contains(&layer_idx) {
                return Some(stage.stage_id);
            }
        }
        None
    }

    /// Get detected architecture (if name mapper available)
    pub fn detected_architecture(
        &self,
    ) -> Option<&crate::pruning::name_mapping::ModelArchitecture> {
        self.name_mapper.as_ref().map(|m| &m.architecture)
    }

    /// Get total number of detected layers (if name mapper available)
    pub fn detected_layer_count(&self) -> Option<usize> {
        self.name_mapper.as_ref().map(|m| m.layer_indices.len())
    }

    /// Execute pipeline with micro-batching (placeholder version)
    ///
    /// This is a simplified version without model integration.
    /// For actual pipeline parallelism with model inference, use execute_with_model.
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size, seq_len, hidden_size]
    ///
    /// # Returns
    /// Output tensor after processing through all pipeline stages
    pub fn execute(&mut self, input: Tensor) -> Result<Tensor> {
        let batch_size = input.dim(0)?;
        let num_micro_batches = (batch_size + self.micro_batch_size - 1) / self.micro_batch_size;

        match self.strategy {
            PipelineStrategy::GPipe => self.execute_gpipe(input, num_micro_batches),
            PipelineStrategy::PipeDream => {
                // TODO: Implement PipeDream scheduling
                anyhow::bail!("PipeDream scheduling not yet implemented")
            }
            PipelineStrategy::Interleaved1F1B => {
                // TODO: Implement 1F1B scheduling
                anyhow::bail!("Interleaved 1F1B scheduling not yet implemented")
            }
        }
    }

    /// Execute pipeline with model integration
    ///
    /// Processes input through pipeline stages using the model's forward_layers method.
    /// Each stage forwards through its assigned layers.
    ///
    /// # Arguments
    /// * `input` - Input hidden states [batch_size, seq_len, hidden_size]
    /// * `model` - The transformer model
    /// * `index_pos` - RoPE position index
    /// * `cache_builder` - Cache builder for position tracking
    /// * `caches` - KV caches (one per model layer)
    /// * `metadata` - Batch metadata
    ///
    /// # Returns
    /// Output hidden states after processing through all pipeline stages
    ///
    /// # Example
    /// ```rust,ignore
    /// let scheduler = PipelineScheduler::new(2, 40, devices, 4, PipelineStrategy::GPipe)?;
    /// let output = scheduler.execute_with_model(
    ///     hidden_states,
    ///     &model,
    ///     0, // index_pos
    ///     &mut cache_builder,
    ///     &mut caches,
    ///     &metadata,
    /// )?;
    /// ```
    pub fn execute_with_model(
        &mut self,
        input: Tensor,
        model: &BatchedTransformer,
        index_pos: usize,
        cache_builder: &mut ParallelCacheBuilder,
        caches: &mut [ParallelKvCache],
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        let batch_size = input.dim(0)?;
        let num_micro_batches = (batch_size + self.micro_batch_size - 1) / self.micro_batch_size;

        match self.strategy {
            PipelineStrategy::GPipe => self.execute_gpipe_with_model(
                input,
                num_micro_batches,
                model,
                index_pos,
                cache_builder,
                caches,
                metadata,
            ),
            PipelineStrategy::PipeDream => {
                // TODO: Implement PipeDream scheduling with model
                anyhow::bail!("PipeDream scheduling not yet implemented")
            }
            PipelineStrategy::Interleaved1F1B => {
                // TODO: Implement 1F1B scheduling with model
                anyhow::bail!("Interleaved 1F1B scheduling not yet implemented")
            }
        }
    }

    /// GPipe: strict separation of forward passes (placeholder version)
    ///
    /// All micro-batches flow through the pipeline sequentially.
    /// Simple but has pipeline bubbles at start and end.
    ///
    /// This is a placeholder without model integration.
    /// Use execute_gpipe_with_model for actual inference.
    fn execute_gpipe(&mut self, input: Tensor, _num_micro_batches: usize) -> Result<Tensor> {
        // Split input into micro-batches
        let micro_batches = self.split_into_micro_batches(input)?;
        let mut outputs = Vec::new();

        // Forward pass: process all micro-batches through pipeline
        for micro_batch in micro_batches {
            let mut intermediate = micro_batch;

            // Pass through each stage sequentially (placeholder)
            for stage in &mut self.stages {
                intermediate = stage.process_micro_batch(intermediate)?;
            }

            outputs.push(intermediate);
        }

        // Concatenate outputs
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// GPipe scheduling with model integration
    ///
    /// All micro-batches flow through the pipeline sequentially, with each
    /// stage processing through its assigned model layers.
    ///
    /// # Pipeline Schedule (4 stages, 4 micro-batches)
    /// ```text
    /// Time →
    /// Stage 0: [M0] [M1] [M2] [M3]
    /// Stage 1:      [M0] [M1] [M2] [M3]
    /// Stage 2:           [M0] [M1] [M2] [M3]
    /// Stage 3:                [M0] [M1] [M2] [M3]
    /// ```
    /// Bubbles at start (stages idle) and end (stages idle).
    ///
    /// # Arguments
    /// * `input` - Input hidden states
    /// * `num_micro_batches` - Number of micro-batches to split input into
    /// * `model` - The transformer model
    /// * `index_pos` - RoPE position index
    /// * `cache_builder` - Cache builder
    /// * `caches` - KV caches
    /// * `metadata` - Batch metadata
    fn execute_gpipe_with_model(
        &mut self,
        input: Tensor,
        _num_micro_batches: usize,
        model: &BatchedTransformer,
        index_pos: usize,
        cache_builder: &mut ParallelCacheBuilder,
        caches: &mut [ParallelKvCache],
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        // Split input into micro-batches
        let micro_batches = self.split_into_micro_batches(input)?;
        let mut outputs = Vec::new();

        // Forward pass: process all micro-batches through pipeline
        for micro_batch in micro_batches {
            let mut intermediate = micro_batch;

            // Pass through each stage sequentially
            for stage in &mut self.stages {
                intermediate = stage.process_micro_batch_with_model(
                    intermediate,
                    model,
                    index_pos,
                    cache_builder,
                    caches,
                    metadata,
                )?;
            }

            outputs.push(intermediate);
        }

        // Concatenate outputs
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Split input tensor into micro-batches
    fn split_into_micro_batches(&self, input: Tensor) -> Result<Vec<Tensor>> {
        let batch_size = input.dim(0)?;
        let mut micro_batches = Vec::new();

        for start in (0..batch_size).step_by(self.micro_batch_size) {
            let end = (start + self.micro_batch_size).min(batch_size);
            let micro_batch = input.narrow(0, start, end - start)?;
            micro_batches.push(micro_batch);
        }

        Ok(micro_batches)
    }

    /// Get number of pipeline stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Get stage by index
    pub fn stage(&self, idx: usize) -> Option<&PipelineStage> {
        self.stages.get(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain CPU "devices".
    ///
    /// The two tests these replace were `#[ignore]`d behind "Requires multi-GPU
    /// setup" and called `Device::cuda_if_available`, which FALLS BACK to CPU.
    /// Neither needed a GPU, so neither ever ran — the same mis-gating that hid
    /// a live defect in `tensor_parallel`.
    fn cpus(n: usize) -> Vec<Device> {
        vec![Device::Cpu; n]
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Layers are distributed across stages.
    #[test]
    fn a_pipeline_scheduler_distributes_layers_across_stages() -> Result<()> {
        let scheduler = PipelineScheduler::new(4, 80, cpus(4), 4, PipelineStrategy::GPipe)?;
        assert_eq!(scheduler.num_stages(), 4);
        for stage in 0..4 {
            assert_eq!(
                scheduler.stage(stage).unwrap().num_layers(),
                20,
                "stage {stage} did not get an equal share of 80 layers"
            );
        }
        Ok(())
    }

    /// **The GPipe path must return its input unchanged, VALUE FOR VALUE.**
    ///
    /// `PipelineStage::process_micro_batch` is a documented pass-through
    /// placeholder, so `execute` is split -> identity -> concatenate. That
    /// composition is the identity, which makes the whole pipeline checkable
    /// without a model: anything that drops, duplicates or reorders a
    /// micro-batch shows up here.
    ///
    /// The test this replaces asserted `output.dims()` only. Shape survives a
    /// reordering — two micro-batches concatenated in the wrong order have
    /// exactly the right shape — so it could not see the failure it was
    /// nearest to.
    #[test]
    fn a_gpipe_pipeline_returns_its_input_unchanged() -> Result<()> {
        let mut scheduler = PipelineScheduler::new(2, 4, cpus(2), 2, PipelineStrategy::GPipe)?;
        let input = Tensor::randn(0.0f32, 1.0, (4, 8, 128), &Device::Cpu)?;
        let output = scheduler.execute(input.clone())?;
        assert_eq!(output.dims(), input.dims());
        assert_eq!(
            flat(&output),
            flat(&input),
            "the placeholder pipeline is split -> identity -> concat, so it must round-trip"
        );
        Ok(())
    }

    /// And when the batch does NOT divide evenly by the micro-batch size.
    ///
    /// The necessary pair. `split_into_micro_batches` clamps the final chunk
    /// with `.min(batch_size)`; an even split exercises neither the clamp nor
    /// the short tail, and every fixture that existed before used an even one.
    #[test]
    fn a_gpipe_pipeline_round_trips_an_uneven_batch() -> Result<()> {
        let mut scheduler = PipelineScheduler::new(2, 4, cpus(2), 2, PipelineStrategy::GPipe)?;
        // batch 5 over micro-batches of 2 -> chunks of 2, 2, 1.
        let input = Tensor::randn(0.0f32, 1.0, (5, 3, 6), &Device::Cpu)?;
        let output = scheduler.execute(input.clone())?;
        assert_eq!(
            output.dims(),
            input.dims(),
            "the short tail changed the shape"
        );
        assert_eq!(
            flat(&output),
            flat(&input),
            "an uneven split must still round-trip"
        );
        Ok(())
    }

    /// The two unimplemented strategies are selectable public variants and must
    /// say so rather than produce something.
    #[test]
    fn unimplemented_pipeline_strategies_report_themselves() -> Result<()> {
        for (strategy, needle) in [
            (PipelineStrategy::PipeDream, "PipeDream"),
            (PipelineStrategy::Interleaved1F1B, "1F1B"),
        ] {
            let mut scheduler = PipelineScheduler::new(2, 4, cpus(2), 2, strategy)?;
            let input = Tensor::randn(0.0f32, 1.0, (4, 8, 16), &Device::Cpu)?;
            let err = match scheduler.execute(input) {
                Ok(_) => panic!("{strategy:?} is unimplemented and must not return a tensor"),
                Err(e) => e,
            };
            assert!(
                format!("{err:#}").contains(needle),
                "the error must name the strategy the caller selected: {err:#}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_architecture_aware_scheduler() -> Result<()> {
        // Create LLaMA-like tensor names for 32 layers
        let mut tensor_names = Vec::new();
        for layer_idx in 0..32 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
            tensor_names.push(format!("blk.{}.attn_k.weight", layer_idx));
            tensor_names.push(format!("blk.{}.attn_v.weight", layer_idx));
            tensor_names.push(format!("blk.{}.ffn_gate.weight", layer_idx));
        }

        let devices = vec![
            Device::Cpu, // Use CPU for testing
            Device::Cpu,
            Device::Cpu,
            Device::Cpu,
        ];

        let scheduler = PipelineScheduler::from_tensor_names(
            4,             // 4 stages
            &tensor_names, // Auto-detect 32 layers
            devices,
            4, // micro-batch size
            PipelineStrategy::GPipe,
        )?;

        // Verify automatic detection
        assert_eq!(scheduler.num_stages(), 4);
        assert_eq!(scheduler.detected_layer_count(), Some(32));
        assert_eq!(
            scheduler.detected_architecture(),
            Some(&crate::pruning::name_mapping::ModelArchitecture::LLaMA)
        );

        // Verify even distribution: 8 layers per stage
        assert_eq!(scheduler.stage(0).unwrap().num_layers(), 8);
        assert_eq!(scheduler.stage(1).unwrap().num_layers(), 8);
        assert_eq!(scheduler.stage(2).unwrap().num_layers(), 8);
        assert_eq!(scheduler.stage(3).unwrap().num_layers(), 8);

        Ok(())
    }

    #[test]
    fn test_variable_layer_counts() -> Result<()> {
        // Test with 40 layers (e.g., larger model)
        let mut tensor_names = Vec::new();
        for layer_idx in 0..40 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
            tensor_names.push(format!("blk.{}.ffn_gate.weight", layer_idx));
        }

        let devices = vec![Device::Cpu, Device::Cpu];

        let scheduler = PipelineScheduler::from_tensor_names(
            2,
            &tensor_names,
            devices,
            4,
            PipelineStrategy::GPipe,
        )?;

        assert_eq!(scheduler.detected_layer_count(), Some(40));
        assert_eq!(scheduler.stage(0).unwrap().num_layers(), 20);
        assert_eq!(scheduler.stage(1).unwrap().num_layers(), 20);

        Ok(())
    }

    #[test]
    fn test_get_gpu_for_layer() -> Result<()> {
        let mut tensor_names = Vec::new();
        for layer_idx in 0..24 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
        }

        let devices = vec![Device::Cpu, Device::Cpu, Device::Cpu];

        let scheduler = PipelineScheduler::from_tensor_names(
            3,
            &tensor_names,
            devices,
            4,
            PipelineStrategy::GPipe,
        )?;

        // Each stage gets 8 layers
        // Stage 0: layers 0-7
        // Stage 1: layers 8-15
        // Stage 2: layers 16-23
        assert_eq!(scheduler.get_stage_for_layer(0), Some(0));
        assert_eq!(scheduler.get_stage_for_layer(7), Some(0));
        assert_eq!(scheduler.get_stage_for_layer(8), Some(1));
        assert_eq!(scheduler.get_stage_for_layer(15), Some(1));
        assert_eq!(scheduler.get_stage_for_layer(16), Some(2));
        assert_eq!(scheduler.get_stage_for_layer(23), Some(2));
        assert_eq!(scheduler.get_stage_for_layer(100), None); // Out of range

        Ok(())
    }

    #[test]
    fn test_gpu_for_tensor_mapping() -> Result<()> {
        let mut tensor_names = Vec::new();
        for layer_idx in 0..16 {
            tensor_names.push(format!("blk.{}.attn_q.weight", layer_idx));
            tensor_names.push(format!("blk.{}.attn_k.weight", layer_idx));
        }

        let devices = vec![Device::Cpu, Device::Cpu];

        let scheduler = PipelineScheduler::from_tensor_names(
            2,
            &tensor_names,
            devices,
            4,
            PipelineStrategy::GPipe,
        )?;

        // Layers 0-7 on GPU 0, layers 8-15 on GPU 1
        assert_eq!(scheduler.get_gpu_for_tensor("blk.0.attn_q.weight"), Some(0));
        assert_eq!(scheduler.get_gpu_for_tensor("blk.7.attn_k.weight"), Some(0));
        assert_eq!(scheduler.get_gpu_for_tensor("blk.8.attn_q.weight"), Some(1));
        assert_eq!(
            scheduler.get_gpu_for_tensor("blk.15.attn_k.weight"),
            Some(1)
        );
        assert_eq!(scheduler.get_gpu_for_tensor("nonexistent.weight"), None);

        Ok(())
    }
}
