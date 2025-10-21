//! Engine: minimal single-request scheduler and request/state management

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::{IndicesAndMask, ScatteredCacheBuilder, ScatteredKvCache};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub prompt: String,
    pub max_new_tokens: usize,
}

/// Request state machine for continuous batching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Request submitted, waiting to start
    Pending,
    /// Currently generating tokens
    Decoding,
    /// Finished generation
    Completed,
}

/// Enhanced request with state tracking for continuous batching
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request: Request,
    pub state: RequestState,
    pub tokens_generated: usize,
    pub position: usize,            // Current token position in sequence
    pub cache_index: Option<usize>, // Index in ScatteredKvCache batch (None until assigned)
    pub generated_tokens: Vec<u32>, // Tokens generated so far (for continuation)
}

impl RequestContext {
    pub fn new(request: Request) -> Self {
        Self {
            request,
            state: RequestState::Pending,
            tokens_generated: 0,
            position: 0,
            cache_index: None,
            generated_tokens: Vec::new(),
        }
    }

    /// Assign a cache index for batched inference
    pub fn assign_cache_index(&mut self, index: usize) {
        self.cache_index = Some(index);
    }

    /// Advance to decoding state
    pub fn start_decoding(&mut self) {
        self.state = RequestState::Decoding;
    }

    /// Record a generated token
    pub fn record_token(&mut self) {
        self.tokens_generated += 1;
        self.position += 1;
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.state = RequestState::Completed;
    }

    /// Check if should continue generating
    pub fn should_continue(&self) -> bool {
        self.state == RequestState::Decoding && self.tokens_generated < self.request.max_new_tokens
    }
}

/// Thread-safe queue for pending requests
#[derive(Clone)]
pub struct RequestQueue {
    pending: Arc<Mutex<VecDeque<RequestContext>>>,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Submit a new request to the queue
    pub fn submit(&self, req: Request) -> Result<String> {
        let id = req.id.clone();
        let ctx = RequestContext::new(req);
        self.pending.lock().push_back(ctx);
        Ok(id)
    }

    /// Pop the next pending request
    pub fn pop(&self) -> Option<RequestContext> {
        self.pending.lock().pop_front()
    }

    /// Get current queue length
    pub fn len(&self) -> usize {
        self.pending.lock().len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.pending.lock().is_empty()
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for batch assembly
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Maximum number of requests in a batch
    pub max_batch_size: usize,
    /// Maximum total tokens across all requests in a batch (for memory management)
    pub max_batch_tokens: usize,
}

impl BatchConfig {
    pub fn new(max_batch_size: usize, max_batch_tokens: usize) -> Self {
        Self {
            max_batch_size,
            max_batch_tokens,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,      // Conservative default for CPU
            max_batch_tokens: 2048, // Reasonable context window
        }
    }
}

/// Assembles batches of requests from the queue
pub struct BatchAssembler {
    config: BatchConfig,
}

impl BatchAssembler {
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Assemble a batch from the queue
    /// Returns requests that fit within batch size and token limits
    /// Uses greedy algorithm: takes requests in FIFO order until limits reached
    /// Note: Requests that don't fit are temporarily removed but should be resubmitted
    pub fn assemble_batch(&self, queue: &RequestQueue) -> Vec<RequestContext> {
        let mut batch = Vec::new();
        let mut total_tokens = 0;
        let mut overflow = Vec::new(); // Temporarily store requests that don't fit

        while batch.len() < self.config.max_batch_size {
            // Try to pop next request
            let ctx = match queue.pop() {
                Some(ctx) => ctx,
                None => break, // Queue empty
            };

            // Calculate tokens needed for this request
            // For decoding: 1 token per step
            // For prefill: could be many tokens (prompt length)
            // Start conservative: assume each request needs at least 1 token
            let request_tokens = ctx.request.max_new_tokens;

            // Check if adding this request would exceed token limit
            if total_tokens + request_tokens > self.config.max_batch_tokens {
                // Would exceed limit - save for resubmission
                overflow.push(ctx);
                break; // Stop assembling this batch
            }

            // Add to batch
            total_tokens += request_tokens;
            batch.push(ctx);
        }

        // Put overflow requests back in queue
        for ctx in overflow {
            // Reconstruct as Request and resubmit
            // Note: This is a bit inefficient, but maintains queue semantics
            let req = ctx.request;
            let _ = queue.submit(req); // Ignore error (shouldn't fail)
        }

        batch
    }

    /// Assemble batch but return requests that didn't fit back to queue
    pub fn assemble_batch_with_overflow(
        &self,
        queue: &RequestQueue,
    ) -> (Vec<RequestContext>, Vec<RequestContext>) {
        let mut batch = Vec::new();
        let mut overflow = Vec::new();
        let mut total_tokens = 0;

        while batch.len() < self.config.max_batch_size {
            let ctx = match queue.pop() {
                Some(ctx) => ctx,
                None => break,
            };

            let request_tokens = ctx.request.max_new_tokens;

            if total_tokens + request_tokens > self.config.max_batch_tokens {
                // Exceeds limit - add to overflow
                overflow.push(ctx);
                // Continue trying to find smaller requests that fit
                // This allows better packing in some cases
                continue;
            }

            total_tokens += request_tokens;
            batch.push(ctx);
        }

        (batch, overflow)
    }
}

impl Default for BatchAssembler {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

/// Batch manager that assigns cache indices to requests
/// Coordinates with Candle's ScatteredKvCache system
pub struct BatchManager {
    max_batch_size: usize,
    /// Track which cache indices are in use
    cache_index_pool: Vec<bool>, // true = in use, false = available
}

impl BatchManager {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            cache_index_pool: vec![false; max_batch_size],
        }
    }

    /// Assign cache indices to a batch of requests
    /// Returns number of assignments made
    /// Errors if insufficient cache slots available for all unassigned requests
    pub fn assign_cache_indices(&mut self, batch: &mut [RequestContext]) -> Result<usize> {
        // Count how many need assignment
        let need_assignment = batch.iter().filter(|ctx| ctx.cache_index.is_none()).count();

        // Check if we have enough slots
        if need_assignment > self.available_slots() {
            return Err(anyhow::anyhow!(
                "Insufficient cache slots: need {}, have {}",
                need_assignment,
                self.available_slots()
            ));
        }

        let mut assigned = 0;

        for ctx in batch.iter_mut() {
            // Skip if already has an index
            if ctx.cache_index.is_some() {
                continue;
            }

            // Find first available cache index
            if let Some(idx) = self.cache_index_pool.iter().position(|&used| !used) {
                ctx.assign_cache_index(idx);
                self.cache_index_pool[idx] = true;
                assigned += 1;
            }
        }

        Ok(assigned)
    }

    /// Release cache index when request completes
    pub fn release_cache_index(&mut self, index: usize) {
        if index < self.cache_index_pool.len() {
            self.cache_index_pool[index] = false;
        }
    }

    /// Get number of available cache slots
    pub fn available_slots(&self) -> usize {
        self.cache_index_pool.iter().filter(|&&used| !used).count()
    }

    /// Reset all cache indices (for testing or restart)
    pub fn reset(&mut self) {
        self.cache_index_pool.fill(false);
    }
}

impl Default for BatchManager {
    fn default() -> Self {
        Self::new(8) // Match BatchConfig default
    }
}

/// Batch executor that coordinates with Candle's ScatteredKvCache
/// Manages per-layer caches and executes batched forward passes
pub struct BatchExecutor {
    cache_builder: ScatteredCacheBuilder, // Persistent: tracks positions across forward passes
    cached_iam: Option<IndicesAndMask>,   // Cached within a single forward pass
    caches: Vec<ScatteredKvCache>,
    batch_manager: BatchManager,
    max_batch_size: usize,
    batch_size: usize,
    context: usize,
    dtype: DType,
    device: Device,
}

impl BatchExecutor {
    /// Create new batch executor
    ///
    /// # Arguments
    /// * `batch_size` - Maximum number of requests in a batch
    /// * `context` - Context window size (max sequence length)
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads per layer
    /// * `head_dim` - Dimension of each attention head
    /// * `dtype` - Data type for cache tensors
    /// * `device` - Device to use (CPU/CUDA/Metal)
    pub fn new(
        batch_size: usize,
        context: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Create one persistent cache builder for tracking positions across steps
        let cache_builder = ScatteredCacheBuilder::new(batch_size, context, dtype, device)?;

        // Create one cache per layer, each with independent builder to avoid position offset bug
        let mut caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let layer_cache_builder =
                ScatteredCacheBuilder::new(batch_size, context, dtype, device)?;
            caches.push(layer_cache_builder.make_cache(num_heads, head_dim)?);
        }

        Ok(Self {
            cache_builder,
            cached_iam: None,
            caches,
            batch_manager: BatchManager::new(batch_size),
            max_batch_size: batch_size,
            batch_size,
            context,
            dtype,
            device: device.clone(),
        })
    }

    /// Assign cache indices to a batch of requests
    pub fn prepare_batch(&mut self, batch: &mut [RequestContext]) -> Result<()> {
        self.batch_manager.assign_cache_indices(batch)?;
        Ok(())
    }

    /// Get indices and mask for current batch
    /// Returns IndicesAndMask for use in forward pass
    pub fn get_indices_and_mask(
        &mut self,
        batch: &[RequestContext],
        seq_len: usize,
    ) -> Result<IndicesAndMask> {
        // Build batch mask: true for requests that should be processed
        // Must be padded to max_batch_size to match ScatteredCacheBuilder
        let mut batch_mask: Vec<bool> = batch
            .iter()
            .map(|ctx| ctx.state == RequestState::Decoding)
            .collect();

        // Pad with false for unused batch slots
        while batch_mask.len() < self.max_batch_size {
            batch_mask.push(false);
        }

        // Use the persistent cache builder which tracks positions across steps
        Ok(self.cache_builder.indices_and_mask(seq_len, &batch_mask)?)
    }

    /// Simplified interface for getting indices and mask when all requests in batch are active
    /// Used by custom batched layers that process pre-filtered active batches
    ///
    /// # Arguments
    /// * `batch_size` - Number of active requests in the batch
    /// * `seq_len` - Sequence length for this batch
    ///
    /// # Returns
    /// IndicesAndMask for the active batch
    ///
    /// # Implementation Note
    /// This caches the result on first call within a forward pass, so all layers
    /// get identical indices. The cache is cleared at the start of each forward pass.
    pub fn get_indices_and_mask_simple(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<IndicesAndMask> {
        // If we already generated IAM for this forward pass, reuse it
        // This ensures all layers write to the same position for the same token
        if let Some(ref iam) = self.cached_iam {
            return Ok(iam.clone());
        }

        // First layer in this forward pass: generate new IAM
        // The persistent cache_builder tracks position across forward passes
        let mut batch_mask: Vec<bool> = vec![true; batch_size];
        while batch_mask.len() < self.max_batch_size {
            batch_mask.push(false);
        }

        let iam = self.cache_builder.indices_and_mask(seq_len, &batch_mask)?;
        self.cached_iam = Some(iam.clone());
        Ok(iam)
    }

    /// Clear the cached IndicesAndMask at the start of a new forward pass
    /// This should be called before processing a new token
    pub fn clear_iam_cache(&mut self) {
        eprintln!("DEBUG: Clearing IAM cache for new forward pass");
        self.cached_iam = None;
    }
    /// Append key-value tensors to cache for a specific layer
    ///
    /// # Arguments
    /// * `layer_idx` - Which transformer layer (0..num_layers)
    /// * `k` - Key tensor from attention
    /// * `v` - Value tensor from attention
    /// * `iam` - Indices and mask from get_indices_and_mask()
    ///
    /// # Returns
    /// Full key and value cache tensors for attention
    pub fn append_kv(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        iam: &IndicesAndMask,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.caches.len() {
            anyhow::bail!(
                "Layer index {} out of bounds (have {} layers)",
                layer_idx,
                self.caches.len()
            );
        }

        Ok(self.caches[layer_idx].append(k, v, iam)?)
    }

    /// Release cache index when request completes
    pub fn release_request(&mut self, cache_index: usize) {
        self.batch_manager.release_cache_index(cache_index);
        // Reset the persistent cache builder's state for this batch index
        self.cache_builder.reset_batch_index(cache_index);
    }

    /// Get number of available batch slots
    pub fn available_slots(&self) -> usize {
        self.batch_manager.available_slots()
    }

    /// Reset all caches and indices (for testing or restart)
    pub fn reset(&mut self) {
        self.batch_manager.reset();
        // Reset the persistent cache builder's state
        self.cache_builder.reset();
    }

    /// Get reference to device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct KvPageRef {
    pub layer: usize,
    pub start_pos: usize,
    pub len: usize,
}

/// Ultra-minimal paged-KV facade for future expansion
#[derive(Debug, Default)]
pub struct KvPager {
    pub use_kv_cache: bool,
    // Number of layers and simple per-layer page counters
    layers: usize,
    pages_per_layer: Vec<usize>,
}

impl KvPager {
    pub fn new(use_kv_cache: bool) -> Self {
        Self {
            use_kv_cache,
            layers: 0,
            pages_per_layer: Vec::new(),
        }
    }
    pub fn attach(&mut self, layers: usize) {
        self.layers = layers;
        self.pages_per_layer = vec![0; layers];
    }
    pub fn alloc_page(&mut self, layer: usize) -> KvPageRef {
        let idx = self
            .pages_per_layer
            .get_mut(layer)
            .map(|c| {
                *c += 1;
                *c - 1
            })
            .unwrap_or(0);
        KvPageRef {
            layer,
            start_pos: idx,
            len: 0,
        }
    }
}

/// Minimal scheduler: runs a single request synchronously using a provided generate function.
pub struct Scheduler;

impl Scheduler {
    pub fn new() -> Self {
        Self
    }

    /// Run a single request by delegating to a closure that performs model-specific generation.
    /// The closure should take (prompt, max_new_tokens) and return the generated string suffix.
    pub fn run_single<F>(&self, req: &Request, mut generate_fn: F) -> Result<String>
    where
        F: FnMut(&str, usize) -> Result<String>,
    {
        let out = generate_fn(&req.prompt, req.max_new_tokens)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_context_creation() {
        let req = Request {
            id: "test-1".to_string(),
            prompt: "Hello, world!".to_string(),
            max_new_tokens: 10,
        };

        let ctx = RequestContext::new(req.clone());

        assert_eq!(ctx.state, RequestState::Pending);
        assert_eq!(ctx.tokens_generated, 0);
        assert_eq!(ctx.position, 0);
        assert_eq!(ctx.cache_index, None);
        assert_eq!(ctx.request.id, "test-1");
        assert_eq!(ctx.request.prompt, "Hello, world!");
        assert_eq!(ctx.request.max_new_tokens, 10);
    }

    #[test]
    fn test_request_context_cache_index_assignment() {
        let req = Request {
            id: "test-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 5,
        };

        let mut ctx = RequestContext::new(req);
        assert_eq!(ctx.cache_index, None);

        ctx.assign_cache_index(3);
        assert_eq!(ctx.cache_index, Some(3));

        // Can reassign
        ctx.assign_cache_index(7);
        assert_eq!(ctx.cache_index, Some(7));
    }

    #[test]
    fn test_batch_manager_creation() {
        let manager = BatchManager::new(4);
        assert_eq!(manager.available_slots(), 4);
        assert_eq!(manager.max_batch_size, 4);
    }

    #[test]
    fn test_batch_manager_assign_indices() {
        let mut manager = BatchManager::new(3);

        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "A".to_string(),
            max_new_tokens: 5,
        };
        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "B".to_string(),
            max_new_tokens: 5,
        };

        let mut batch = vec![RequestContext::new(req1), RequestContext::new(req2)];

        let count = manager.assign_cache_indices(&mut batch).unwrap();
        assert_eq!(count, 2);
        assert_eq!(manager.available_slots(), 1);

        // Check indices assigned
        assert_eq!(batch[0].cache_index, Some(0));
        assert_eq!(batch[1].cache_index, Some(1));
    }

    #[test]
    fn test_batch_manager_release_and_reuse() {
        let mut manager = BatchManager::new(3);

        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "A".to_string(),
            max_new_tokens: 5,
        };
        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "B".to_string(),
            max_new_tokens: 5,
        };

        let mut batch1 = vec![
            RequestContext::new(req1.clone()),
            RequestContext::new(req2.clone()),
        ];

        manager.assign_cache_indices(&mut batch1).unwrap();
        assert_eq!(manager.available_slots(), 1);
        assert_eq!(batch1[0].cache_index, Some(0));
        assert_eq!(batch1[1].cache_index, Some(1));

        // Release index 0
        manager.release_cache_index(0);
        assert_eq!(manager.available_slots(), 2);

        // Assign to new request - should reuse index 0
        let mut batch2 = vec![RequestContext::new(req1)];
        manager.assign_cache_indices(&mut batch2).unwrap();
        assert_eq!(batch2[0].cache_index, Some(0));
        assert_eq!(manager.available_slots(), 1);
    }

    #[test]
    fn test_batch_manager_pool_exhaustion() {
        let mut manager = BatchManager::new(2);

        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "A".to_string(),
            max_new_tokens: 5,
        };
        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "B".to_string(),
            max_new_tokens: 5,
        };
        let req3 = Request {
            id: "req-3".to_string(),
            prompt: "C".to_string(),
            max_new_tokens: 5,
        };

        let mut batch = vec![
            RequestContext::new(req1),
            RequestContext::new(req2),
            RequestContext::new(req3),
        ];

        let result = manager.assign_cache_indices(&mut batch);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Insufficient cache slots")
        );
    }

    #[test]
    fn test_batch_manager_available_slots() {
        let mut manager = BatchManager::new(5);
        assert_eq!(manager.available_slots(), 5);

        let req = Request {
            id: "req".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 5,
        };

        let mut batch = vec![
            RequestContext::new(req.clone()),
            RequestContext::new(req.clone()),
        ];

        manager.assign_cache_indices(&mut batch).unwrap();
        assert_eq!(manager.available_slots(), 3);

        manager.release_cache_index(0);
        assert_eq!(manager.available_slots(), 4);

        manager.release_cache_index(1);
        assert_eq!(manager.available_slots(), 5);
    }

    #[test]
    fn test_batch_manager_reset() {
        let mut manager = BatchManager::new(3);

        let req = Request {
            id: "req".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 5,
        };

        let mut batch = vec![RequestContext::new(req.clone()), RequestContext::new(req)];

        manager.assign_cache_indices(&mut batch).unwrap();
        assert_eq!(manager.available_slots(), 1);

        manager.reset();
        assert_eq!(manager.available_slots(), 3);

        // After reset, should be able to assign from index 0 again
        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "Test2".to_string(),
            max_new_tokens: 5,
        };
        let mut batch2 = vec![RequestContext::new(req2)];
        manager.assign_cache_indices(&mut batch2).unwrap();
        assert_eq!(batch2[0].cache_index, Some(0));
    }

    #[test]
    fn test_state_transitions() {
        let req = Request {
            id: "test-2".to_string(),
            prompt: "Test prompt".to_string(),
            max_new_tokens: 5,
        };

        let mut ctx = RequestContext::new(req);

        // Start: Pending
        assert_eq!(ctx.state, RequestState::Pending);
        assert!(!ctx.should_continue());

        // Transition to Decoding
        ctx.start_decoding();
        assert_eq!(ctx.state, RequestState::Decoding);
        assert!(ctx.should_continue());

        // Generate tokens
        ctx.record_token();
        assert_eq!(ctx.tokens_generated, 1);
        assert_eq!(ctx.position, 1);
        assert!(ctx.should_continue());

        ctx.record_token();
        assert_eq!(ctx.tokens_generated, 2);
        assert_eq!(ctx.position, 2);

        // Complete
        ctx.complete();
        assert_eq!(ctx.state, RequestState::Completed);
        assert!(!ctx.should_continue());
    }

    #[test]
    fn test_should_continue_respects_max_tokens() {
        let req = Request {
            id: "test-3".to_string(),
            prompt: "Limited".to_string(),
            max_new_tokens: 3,
        };

        let mut ctx = RequestContext::new(req);
        ctx.start_decoding();

        // Should continue for first 3 tokens
        assert!(ctx.should_continue());
        ctx.record_token();

        assert!(ctx.should_continue());
        ctx.record_token();

        assert!(ctx.should_continue());
        ctx.record_token();

        // Should stop after max_new_tokens reached
        assert!(!ctx.should_continue());
        assert_eq!(ctx.tokens_generated, 3);
    }

    #[test]
    fn test_multiple_contexts_independent() {
        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "First".to_string(),
            max_new_tokens: 10,
        };

        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "Second".to_string(),
            max_new_tokens: 20,
        };

        let mut ctx1 = RequestContext::new(req1);
        let mut ctx2 = RequestContext::new(req2);

        // Advance ctx1
        ctx1.start_decoding();
        ctx1.record_token();
        ctx1.record_token();

        // ctx2 should be unaffected
        assert_eq!(ctx1.tokens_generated, 2);
        assert_eq!(ctx2.tokens_generated, 0);
        assert_eq!(ctx1.state, RequestState::Decoding);
        assert_eq!(ctx2.state, RequestState::Pending);

        // Advance ctx2
        ctx2.start_decoding();
        ctx2.record_token();

        assert_eq!(ctx1.tokens_generated, 2);
        assert_eq!(ctx2.tokens_generated, 1);
    }

    #[test]
    fn test_queue_creation() {
        let queue = RequestQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_submit_and_pop() {
        let queue = RequestQueue::new();

        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "First request".to_string(),
            max_new_tokens: 10,
        };

        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "Second request".to_string(),
            max_new_tokens: 20,
        };

        // Submit requests
        let id1 = queue.submit(req1).unwrap();
        assert_eq!(id1, "req-1");
        assert_eq!(queue.len(), 1);

        let id2 = queue.submit(req2).unwrap();
        assert_eq!(id2, "req-2");
        assert_eq!(queue.len(), 2);

        // Pop in FIFO order
        let ctx1 = queue.pop().unwrap();
        assert_eq!(ctx1.request.id, "req-1");
        assert_eq!(ctx1.state, RequestState::Pending);
        assert_eq!(queue.len(), 1);

        let ctx2 = queue.pop().unwrap();
        assert_eq!(ctx2.request.id, "req-2");
        assert_eq!(queue.len(), 0);

        // Queue should be empty
        assert!(queue.is_empty());
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_queue_thread_safety() {
        use std::thread;

        let queue = RequestQueue::new();
        let mut handles = vec![];

        // Spawn 10 threads, each submitting a request
        for i in 0..10 {
            let q = queue.clone();
            let handle = thread::spawn(move || {
                let req = Request {
                    id: format!("thread-req-{}", i),
                    prompt: format!("Prompt from thread {}", i),
                    max_new_tokens: 5,
                };
                q.submit(req).unwrap()
            });
            handles.push(handle);
        }

        // Wait for all threads
        for h in handles {
            h.join().unwrap();
        }

        // All 10 requests should be in queue
        assert_eq!(queue.len(), 10);

        // Verify we can pop all requests
        let mut ids = vec![];
        while let Some(ctx) = queue.pop() {
            ids.push(ctx.request.id);
        }

        assert_eq!(ids.len(), 10);
        assert!(queue.is_empty());

        // Check all IDs are unique and match expected pattern
        ids.sort();
        for id in ids.iter() {
            assert!(id.starts_with("thread-req-"));
        }
    }

    #[test]
    fn test_queue_concurrent_producers_consumers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let queue = RequestQueue::new();
        let produced = Arc::new(AtomicUsize::new(0));
        let consumed = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        // Producer threads
        for i in 0..5 {
            let q = queue.clone();
            let p = produced.clone();
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let req = Request {
                        id: format!("producer-{}-req-{}", i, j),
                        prompt: "test".to_string(),
                        max_new_tokens: 1,
                    };
                    q.submit(req).unwrap();
                    p.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(std::time::Duration::from_micros(10));
                }
            });
            handles.push(handle);
        }

        // Consumer threads
        for _ in 0..3 {
            let q = queue.clone();
            let c = consumed.clone();
            let handle = thread::spawn(move || {
                loop {
                    if let Some(_ctx) = q.pop() {
                        c.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(std::time::Duration::from_micros(20));
                    } else {
                        thread::sleep(std::time::Duration::from_micros(50));
                    }

                    // Exit when we've consumed enough
                    if c.load(Ordering::SeqCst) >= 50 {
                        break;
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for h in handles {
            h.join().unwrap();
        }

        // Verify counts
        assert_eq!(produced.load(Ordering::SeqCst), 50);
        assert_eq!(consumed.load(Ordering::SeqCst), 50);
        assert!(queue.is_empty());
    }

    // ===== Batch Assembly Tests =====

    #[test]
    fn test_batch_config_creation() {
        let config = BatchConfig::new(16, 4096);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.max_batch_tokens, 4096);

        let default_config = BatchConfig::default();
        assert_eq!(default_config.max_batch_size, 8);
        assert_eq!(default_config.max_batch_tokens, 2048);
    }

    #[test]
    fn test_batch_assembler_empty_queue() {
        let queue = RequestQueue::new();
        let assembler = BatchAssembler::default();

        let batch = assembler.assemble_batch(&queue);
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_batch_assembler_single_request() {
        let queue = RequestQueue::new();
        let req = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        queue.submit(req).unwrap();

        let assembler = BatchAssembler::default();
        let batch = assembler.assemble_batch(&queue);

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].request.id, "req-1");
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_assembler_respects_max_batch_size() {
        let queue = RequestQueue::new();

        // Submit 10 small requests
        for i in 0..10 {
            let req = Request {
                id: format!("req-{}", i),
                prompt: "Test".to_string(),
                max_new_tokens: 10, // Small token count
            };
            queue.submit(req).unwrap();
        }

        // Configure batch size limit of 5
        let config = BatchConfig::new(5, 10000); // High token limit
        let assembler = BatchAssembler::new(config);

        let batch = assembler.assemble_batch(&queue);

        // Should only take 5 requests due to batch size limit
        assert_eq!(batch.len(), 5);
        assert_eq!(queue.len(), 5); // 5 remaining

        // Verify FIFO order
        for i in 0..5 {
            assert_eq!(batch[i].request.id, format!("req-{}", i));
        }
    }

    #[test]
    fn test_batch_assembler_respects_token_limit() {
        let queue = RequestQueue::new();

        // Submit 5 requests with 100 tokens each
        for i in 0..5 {
            let req = Request {
                id: format!("req-{}", i),
                prompt: "Test".to_string(),
                max_new_tokens: 100,
            };
            queue.submit(req).unwrap();
        }

        // Configure token limit of 250 (can fit 2 requests)
        let config = BatchConfig::new(10, 250);
        let assembler = BatchAssembler::new(config);

        let batch = assembler.assemble_batch(&queue);

        // Should only take 2 requests (200 tokens) due to token limit
        assert_eq!(batch.len(), 2);
        assert_eq!(queue.len(), 3); // 3 remaining

        // Verify correct requests taken
        assert_eq!(batch[0].request.id, "req-0");
        assert_eq!(batch[1].request.id, "req-1");
    }

    #[test]
    fn test_batch_assembler_mixed_sizes() {
        let queue = RequestQueue::new();

        // Submit requests with varying token counts
        let sizes = vec![10, 50, 200, 30, 100, 20];
        for (i, size) in sizes.iter().enumerate() {
            let req = Request {
                id: format!("req-{}", i),
                prompt: "Test".to_string(),
                max_new_tokens: *size,
            };
            queue.submit(req).unwrap();
        }

        // Token limit: 300
        // Should fit: 10 + 50 + 200 = 260, then 30 = 290 (still fits!), then 100 would exceed
        let config = BatchConfig::new(10, 300);
        let assembler = BatchAssembler::new(config);

        let batch = assembler.assemble_batch(&queue);

        // Should take first 4 requests (total: 290 tokens)
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[0].request.max_new_tokens, 10);
        assert_eq!(batch[1].request.max_new_tokens, 50);
        assert_eq!(batch[2].request.max_new_tokens, 200);
        assert_eq!(batch[3].request.max_new_tokens, 30);
    }

    #[test]
    fn test_batch_assembler_with_overflow() {
        let queue = RequestQueue::new();

        // Submit requests: 100, 200, 50, 300, 40
        let sizes = vec![100, 200, 50, 300, 40];
        for (i, size) in sizes.iter().enumerate() {
            let req = Request {
                id: format!("req-{}", i),
                prompt: "Test".to_string(),
                max_new_tokens: *size,
            };
            queue.submit(req).unwrap();
        }

        // Token limit: 400
        // Can fit: 100 + 200 = 300, then 50 = 350, then skip 300 (too large), add 40 = 390
        let config = BatchConfig::new(10, 400);
        let assembler = BatchAssembler::new(config);

        let (batch, overflow) = assembler.assemble_batch_with_overflow(&queue);

        // Should have 4 requests in batch (100, 200, 50, 40)
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[0].request.max_new_tokens, 100);
        assert_eq!(batch[1].request.max_new_tokens, 200);
        assert_eq!(batch[2].request.max_new_tokens, 50);
        assert_eq!(batch[3].request.max_new_tokens, 40);

        // Should have 1 request in overflow (300)
        assert_eq!(overflow.len(), 1);
        assert_eq!(overflow[0].request.max_new_tokens, 300);

        // Queue should be empty after assembly
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_assembler_all_requests_too_large() {
        let queue = RequestQueue::new();

        // Submit 3 requests that each exceed token limit
        for i in 0..3 {
            let req = Request {
                id: format!("req-{}", i),
                prompt: "Test".to_string(),
                max_new_tokens: 500, // Each exceeds limit
            };
            queue.submit(req).unwrap();
        }

        let config = BatchConfig::new(10, 300); // Token limit too small
        let assembler = BatchAssembler::new(config);

        let batch = assembler.assemble_batch(&queue);

        // Should have empty batch - no requests fit
        assert_eq!(batch.len(), 0);
        // All requests consumed but not included
        // (In real implementation, these would need special handling)
    }

    #[test]
    fn test_batch_assembler_exact_fit() {
        let queue = RequestQueue::new();

        // Submit requests that exactly fit the token limit
        let sizes = vec![100, 100, 100, 100]; // Total = 400
        for (i, size) in sizes.iter().enumerate() {
            let req = Request {
                id: format!("req-{}", i),
                prompt: "Test".to_string(),
                max_new_tokens: *size,
            };
            queue.submit(req).unwrap();
        }

        let config = BatchConfig::new(10, 400); // Exact fit
        let assembler = BatchAssembler::new(config);

        let batch = assembler.assemble_batch(&queue);

        // Should take all 4 requests (exactly 400 tokens)
        assert_eq!(batch.len(), 4);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_assembler_preserves_request_state() {
        let queue = RequestQueue::new();

        let req = Request {
            id: "req-1".to_string(),
            prompt: "Test prompt".to_string(),
            max_new_tokens: 20,
        };
        queue.submit(req).unwrap();

        let assembler = BatchAssembler::default();
        let batch = assembler.assemble_batch(&queue);

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].request.id, "req-1");
        assert_eq!(batch[0].request.prompt, "Test prompt");
        assert_eq!(batch[0].request.max_new_tokens, 20);
        assert_eq!(batch[0].state, RequestState::Pending);
        assert_eq!(batch[0].tokens_generated, 0);
        assert_eq!(batch[0].position, 0);
    }

    #[test]
    fn test_batch_executor_creation() {
        let device = Device::Cpu;
        let batch_size = 4;
        let context = 128;
        let num_layers = 2;
        let num_heads = 8;
        let head_dim = 64;
        let dtype = DType::F32;

        let executor = BatchExecutor::new(
            batch_size, context, num_layers, num_heads, head_dim, dtype, &device,
        );

        assert!(executor.is_ok());
        let executor = executor.unwrap();
        assert_eq!(executor.available_slots(), batch_size);
    }

    #[test]
    fn test_batch_executor_prepare_batch() {
        let device = Device::Cpu;
        let executor = BatchExecutor::new(2, 128, 2, 8, 64, DType::F32, &device).unwrap();

        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "Test2".to_string(),
            max_new_tokens: 10,
        };

        let mut batch = vec![RequestContext::new(req1), RequestContext::new(req2)];

        let mut executor = executor;
        let result = executor.prepare_batch(&mut batch);
        assert!(result.is_ok());

        // Check that cache indices were assigned
        assert_eq!(batch[0].cache_index, Some(0));
        assert_eq!(batch[1].cache_index, Some(1));
        assert_eq!(executor.available_slots(), 0);
    }

    #[test]
    fn test_batch_executor_get_indices_and_mask() {
        let device = Device::Cpu;
        let mut executor = BatchExecutor::new(2, 128, 2, 8, 64, DType::F32, &device).unwrap();

        let req1 = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        let mut ctx1 = RequestContext::new(req1);
        ctx1.start_decoding();

        let batch = vec![ctx1];

        let result = executor.get_indices_and_mask(&batch, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_executor_append_kv() {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let head_dim = 32;
        let seq_len = 1;

        let mut executor =
            BatchExecutor::new(batch_size, 128, 2, num_heads, head_dim, DType::F32, &device)
                .unwrap();

        let req = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        let mut ctx = RequestContext::new(req);
        ctx.start_decoding();

        let mut batch = vec![ctx];
        executor.prepare_batch(&mut batch).unwrap();

        // Get indices and mask
        let iam = executor.get_indices_and_mask(&batch, seq_len).unwrap();

        // Create dummy key and value tensors
        let k_shape = (batch_size, num_heads, seq_len, head_dim);
        let k = Tensor::zeros(k_shape, DType::F32, &device).unwrap();
        let v = Tensor::zeros(k_shape, DType::F32, &device).unwrap();

        // Append to layer 0
        let result = executor.append_kv(0, &k, &v, &iam);
        assert!(result.is_ok());

        let (k_out, v_out) = result.unwrap();
        // Output should have full cache shape (batch, heads, context, head_dim)
        assert_eq!(k_out.dims(), &[batch_size, num_heads, 128, head_dim]);
        assert_eq!(v_out.dims(), &[batch_size, num_heads, 128, head_dim]);
    }

    #[test]
    fn test_batch_executor_release_request() {
        let device = Device::Cpu;
        let mut executor = BatchExecutor::new(2, 128, 2, 8, 64, DType::F32, &device).unwrap();

        let req = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        let mut batch = vec![RequestContext::new(req)];
        executor.prepare_batch(&mut batch).unwrap();

        assert_eq!(executor.available_slots(), 1);
        assert_eq!(batch[0].cache_index, Some(0));

        // Release the request
        executor.release_request(0);
        assert_eq!(executor.available_slots(), 2);

        // Can assign again
        let req2 = Request {
            id: "req-2".to_string(),
            prompt: "Test2".to_string(),
            max_new_tokens: 10,
        };
        let mut batch2 = vec![RequestContext::new(req2)];
        executor.prepare_batch(&mut batch2).unwrap();
        assert_eq!(batch2[0].cache_index, Some(0)); // Reused index 0
    }

    #[test]
    fn test_batch_executor_multiple_layers() {
        let device = Device::Cpu;
        let num_layers = 3;
        let mut executor =
            BatchExecutor::new(2, 128, num_layers, 4, 32, DType::F32, &device).unwrap();

        let req = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        let mut ctx = RequestContext::new(req);
        ctx.start_decoding();

        let mut batch = vec![ctx];
        executor.prepare_batch(&mut batch).unwrap();

        let iam = executor.get_indices_and_mask(&batch, 1).unwrap();

        // Create dummy tensors
        let k = Tensor::zeros((2, 4, 1, 32), DType::F32, &device).unwrap();
        let v = Tensor::zeros((2, 4, 1, 32), DType::F32, &device).unwrap();

        // Should work for all layers
        for layer_idx in 0..num_layers {
            let result = executor.append_kv(layer_idx, &k, &v, &iam);
            assert!(result.is_ok(), "Failed for layer {}", layer_idx);
        }

        // Should fail for invalid layer
        let result = executor.append_kv(num_layers, &k, &v, &iam);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_batch_executor_reset() {
        let device = Device::Cpu;
        let mut executor = BatchExecutor::new(2, 128, 2, 8, 64, DType::F32, &device).unwrap();

        let req = Request {
            id: "req-1".to_string(),
            prompt: "Test".to_string(),
            max_new_tokens: 10,
        };
        let mut batch = vec![RequestContext::new(req)];
        executor.prepare_batch(&mut batch).unwrap();

        assert_eq!(executor.available_slots(), 1);

        executor.reset();
        assert_eq!(executor.available_slots(), 2);
    }
}
