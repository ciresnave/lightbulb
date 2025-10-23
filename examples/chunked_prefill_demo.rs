//! Chunked Prefill with Padding - Hybrid Parallelization Strategy
//!
//! This example demonstrates an advanced prefill strategy that combines:
//! 1. **Chunking**: Break long prompts into smaller chunks
//! 2. **Padding**: Pad the last chunk to enable batching
//! 3. **Parallel Processing**: Process multiple requests together
//!
//! Benefits:
//! - Reduced memory usage (chunking)
//! - Better time-to-first-token (can start decoding earlier)
//! - Parallel efficiency (padding enables batching)
//! - Interleaving opportunities (can decode between chunks)
//!
//! Example scenario:
//! - Request 1: 2000 token prompt
//! - Request 2: 500 token prompt  
//! - Request 3: 150 token prompt
//!
//! Strategy with chunk_size=512:
//! ```
//! Chunk 1: [Req1: 512 tokens, Req2: 500+12 pad, Req3: 150+362 pad] → Batch forward
//! Chunk 2: [Req1: 512 tokens]                                       → Single forward
//! Chunk 3: [Req1: 512 tokens]                                       → Single forward
//! Chunk 4: [Req1: 464+48 pad]                                       → Single forward
//! ```
//!
//! This enables starting decode for Req2 and Req3 while Req1 is still in prefill!

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Configuration for chunked prefill strategy
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Maximum tokens to process in a single forward pass
    pub chunk_size: usize,

    /// Maximum number of requests to batch together
    pub max_batch_size: usize,

    /// Whether to pad the last chunk to enable batching
    pub pad_last_chunk: bool,

    /// Padding token ID (usually 0 or a special token)
    pub pad_token_id: u32,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_batch_size: 4,
            pad_last_chunk: true,
            pad_token_id: 0,
        }
    }
}

/// A prefill request with token stream
#[derive(Debug, Clone)]
pub struct PrefillRequest {
    pub id: String,
    pub tokens: Vec<u32>,
    pub processed: usize, // How many tokens have been processed
}

impl PrefillRequest {
    pub fn new(id: String, tokens: Vec<u32>) -> Self {
        Self {
            id,
            tokens,
            processed: 0,
        }
    }

    pub fn remaining(&self) -> usize {
        self.tokens.len() - self.processed
    }

    pub fn is_complete(&self) -> bool {
        self.processed >= self.tokens.len()
    }

    pub fn next_chunk(&mut self, chunk_size: usize) -> Vec<u32> {
        let start = self.processed;
        let end = (start + chunk_size).min(self.tokens.len());
        let chunk = self.tokens[start..end].to_vec();
        self.processed = end;
        chunk
    }
}

/// Represents a batched chunk ready for processing
#[derive(Debug)]
pub struct ChunkedBatch {
    pub requests: Vec<String>,      // Request IDs in this batch
    pub tokens: Vec<Vec<u32>>,      // Tokens per request (may be padded)
    pub actual_lengths: Vec<usize>, // Actual token counts (before padding)
    pub is_padded: bool,
}

impl ChunkedBatch {
    /// Create concatenated tensor for batch processing
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let total_tokens: usize = self.tokens.iter().map(|t| t.len()).sum();
        let flat_tokens: Vec<u32> = self.tokens.iter().flatten().copied().collect();
        Ok(Tensor::new(flat_tokens.as_slice(), device)?)
    }

    /// Get sequence boundaries for BatchMetadata
    pub fn prompt_lengths(&self) -> Vec<usize> {
        self.actual_lengths.clone()
    }
}

/// Chunked prefill scheduler
pub struct ChunkedPrefillScheduler {
    config: ChunkedPrefillConfig,
    pending_requests: Vec<PrefillRequest>,
}

impl ChunkedPrefillScheduler {
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        Self {
            config,
            pending_requests: Vec::new(),
        }
    }

    pub fn add_request(&mut self, request: PrefillRequest) {
        self.pending_requests.push(request);
    }

    /// Assemble the next batch of chunks to process
    ///
    /// Strategy:
    /// 1. Find all requests that have remaining tokens
    /// 2. Take up to max_batch_size requests
    /// 3. Extract next chunk from each
    /// 4. If pad_last_chunk enabled, pad shorter chunks to max length in batch
    pub fn next_batch(&mut self) -> Option<ChunkedBatch> {
        // Filter requests that still have tokens to process
        let mut active_requests: Vec<&mut PrefillRequest> = self
            .pending_requests
            .iter_mut()
            .filter(|req| !req.is_complete())
            .take(self.config.max_batch_size)
            .collect();

        if active_requests.is_empty() {
            return None;
        }

        // Extract next chunk from each request
        let mut requests = Vec::new();
        let mut tokens = Vec::new();
        let mut actual_lengths = Vec::new();

        for req in &mut active_requests {
            let chunk = req.next_chunk(self.config.chunk_size);
            actual_lengths.push(chunk.len());
            requests.push(req.id.clone());
            tokens.push(chunk);
        }

        // Apply padding if enabled and batch size > 1
        let is_padded = if self.config.pad_last_chunk && tokens.len() > 1 {
            let max_len = tokens.iter().map(|t| t.len()).max().unwrap();

            for token_seq in tokens.iter_mut() {
                let pad_count = max_len - token_seq.len();
                if pad_count > 0 {
                    token_seq.extend(vec![self.config.pad_token_id; pad_count]);
                }
            }
            true
        } else {
            false
        };

        Some(ChunkedBatch {
            requests,
            tokens,
            actual_lengths,
            is_padded,
        })
    }

    /// Check if all requests are complete
    pub fn all_complete(&self) -> bool {
        self.pending_requests.iter().all(|req| req.is_complete())
    }

    /// Get statistics about pending work
    pub fn stats(&self) -> PrefillStats {
        let active = self
            .pending_requests
            .iter()
            .filter(|r| !r.is_complete())
            .count();
        let total_remaining: usize = self.pending_requests.iter().map(|r| r.remaining()).sum();
        let max_remaining = self
            .pending_requests
            .iter()
            .map(|r| r.remaining())
            .max()
            .unwrap_or(0);

        PrefillStats {
            active_requests: active,
            total_remaining_tokens: total_remaining,
            max_remaining_tokens: max_remaining,
        }
    }
}

#[derive(Debug)]
pub struct PrefillStats {
    pub active_requests: usize,
    pub total_remaining_tokens: usize,
    pub max_remaining_tokens: usize,
}

/// Simulate the chunked prefill process
fn simulate_chunked_prefill() {
    println!("🔄 Chunked Prefill with Padding - Simulation\n");

    let config = ChunkedPrefillConfig {
        chunk_size: 512,
        max_batch_size: 4,
        pad_last_chunk: true,
        pad_token_id: 0,
    };

    println!("Configuration:");
    println!("  Chunk size: {} tokens", config.chunk_size);
    println!("  Max batch size: {} requests", config.max_batch_size);
    println!("  Pad last chunk: {}\n", config.pad_last_chunk);

    // Create test requests with different prompt lengths
    let requests = vec![
        PrefillRequest::new("long_context".into(), vec![1u32; 2000]),
        PrefillRequest::new("medium_prompt".into(), vec![2u32; 500]),
        PrefillRequest::new("short_query".into(), vec![3u32; 150]),
    ];

    println!("Requests:");
    for req in &requests {
        println!("  {} - {} tokens", req.id, req.tokens.len());
    }
    println!();

    let mut scheduler = ChunkedPrefillScheduler::new(config);
    for req in requests {
        scheduler.add_request(req);
    }

    let mut batch_num = 0;
    let mut total_forward_passes = 0;
    let mut total_tokens_processed = 0;

    println!("{}", "=".repeat(80));
    println!("Processing Batches\n");

    while let Some(batch) = scheduler.next_batch() {
        batch_num += 1;
        total_forward_passes += 1;

        let batch_tokens: usize = batch.actual_lengths.iter().sum();
        total_tokens_processed += batch_tokens;

        println!("Batch #{}", batch_num);
        println!("  Requests: {}", batch.requests.len());
        println!("  Padded: {}", batch.is_padded);

        for (i, req_id) in batch.requests.iter().enumerate() {
            let actual = batch.actual_lengths[i];
            let with_padding = batch.tokens[i].len();

            if actual == with_padding {
                println!("    {} - {} tokens", req_id, actual);
            } else {
                println!(
                    "    {} - {} tokens (+{} padding)",
                    req_id,
                    actual,
                    with_padding - actual
                );
            }
        }

        let stats = scheduler.stats();
        println!(
            "  Remaining: {} requests, {} tokens\n",
            stats.active_requests, stats.total_remaining_tokens
        );
    }

    println!("{}", "=".repeat(80));
    println!("\n📊 Summary:");
    println!("  Total batches: {}", batch_num);
    println!("  Total forward passes: {}", total_forward_passes);
    println!("  Total tokens processed: {}", total_tokens_processed);
    println!(
        "  Avg tokens per forward pass: {:.1}",
        total_tokens_processed as f64 / total_forward_passes as f64
    );

    // Compare with naive sequential
    println!("\n📈 Comparison with naive sequential:");
    println!("  Naive: 3 forward passes (one per request)");
    println!("  Chunked+Padded: {} forward passes", total_forward_passes);

    // Calculate efficiency
    let max_possible_batching = 2650.0 / 512.0; // total tokens / chunk size
    let efficiency = total_forward_passes as f64 / max_possible_batching;
    println!(
        "  Batching efficiency: {:.1}% (lower is better)",
        efficiency * 100.0
    );

    println!("\n💡 Benefits:");
    println!("  ✓ Requests 2 & 3 can start decoding after batch 1");
    println!("  ✓ Memory usage bounded by chunk_size");
    println!("  ✓ Parallel processing of multiple requests");
    println!("  ✓ Better time-to-first-token for shorter prompts");
}

fn main() {
    simulate_chunked_prefill();

    println!("\n{}", "=".repeat(80));
    println!("🎯 Real-World Application\n");

    println!("This strategy is ideal for:");
    println!("  • RAG systems (long context + short query)");
    println!("  • Document Q&A (large document + questions)");
    println!("  • Mixed workloads (various prompt lengths)");
    println!("  • Streaming responses (start generating ASAP)");

    println!("\nImplementation with BatchedTransformer:");
    println!(
        r#"
  while let Some(batch) = scheduler.next_batch() {{
      let tensor = batch.to_tensor(&device)?;
      let metadata = BatchMetadata::from_prefill_batch(
          (0..batch.requests.len()).collect(),
          batch.prompt_lengths(),
      );
      
      // Single parallel forward pass for this chunk batch
      let logits = batched_transformer.forward(
          &tensor,
          &mut batch_executor,
          &metadata,
      )?;
      
      // Requests that completed prefill can start decoding
      for (i, req_id) in batch.requests.iter().enumerate() {{
          if request_complete(req_id) {{
              start_decoding(req_id);
          }}
      }}
  }}
"#
    );

    println!("\n✨ Chunked + Padded prefill strategy enables truly responsive inference!\n");
}
