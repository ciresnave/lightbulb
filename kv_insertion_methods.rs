// ================================
// KV Cache Insertion APIs (Phase 2.5)
// Add these methods to ParallelCacheBuilder impl block before make_cache()
// ================================

/// Evict a range of cache positions to make room for insertion
///
/// This clears cache slots in the range [start_pos, end_pos) for the given slot,
/// marking any affected spans as evicted and returning information about what was
/// removed. This is typically used before inserting new context mid-conversation.
///
/// # Arguments
///
/// * `slot` - Batch slot index to evict from
/// * `start_pos` - Starting position (inclusive) to evict
/// * `end_pos` - Ending position (exclusive) to evict
///
/// # Returns
///
/// EvictionResult containing:
/// - `spans_affected`: List of spans that were fully or partially evicted
/// - `positions_evicted`: List of (slot, range) tuples for evicted positions
///
/// # Example
///
/// ```ignore
/// // Evict positions 50-100 in slot 0 to make room for RAG context
/// let result = cache_builder.evict_range(0, 50, 100)?;
/// println!("Evicted {} spans", result.spans_affected.len());
/// ```
pub fn evict_range(
    &mut self,
    slot: usize,
    start_pos: usize,
    end_pos: usize,
) -> Result<crate::cache::EvictionResult> {
    use crate::cache::{EvictionImpact, EvictionResult, SpanState};

    let mut result = EvictionResult::empty();

    // Find all spans that overlap with this range
    let overlapping = self
        .span_registry
        .spans_overlapping_range(slot, start_pos, end_pos);

    for span in overlapping {
        let span_id = span.id;
        let span_range = span.start_pos..span.end_pos;

        // Calculate overlap
        let overlap_start = start_pos.max(span.start_pos);
        let overlap_end = end_pos.min(span.end_pos);

        if overlap_start >= overlap_end {
            continue; // No actual overlap
        }

        // Get mutable reference to update span state
        if let Some(span_mut) = self.span_registry.get_mut(span_id) {
            if overlap_start == span_range.start && overlap_end == span_range.end {
                // Fully evicted
                span_mut.state = SpanState::FullyEvicted;
                result
                    .spans_affected
                    .push((span_id, EvictionImpact::FullyEvicted));
            } else {
                // Partially evicted - track remaining ranges
                let mut remaining = Vec::new();
                if overlap_start > span_range.start {
                    remaining.push(span_range.start..overlap_start);
                }
                if overlap_end < span_range.end {
                    remaining.push(overlap_end..span_range.end);
                }
                span_mut.state = SpanState::PartiallyEvicted {
                    remaining_ranges: remaining,
                };
                result
                    .spans_affected
                    .push((span_id, EvictionImpact::PartiallyEvicted));
            }

            result
                .positions_evicted
                .push((slot, overlap_start..overlap_end));
        }
    }

    // Note: Actual KV tensor zeroing would happen in the model's forward pass
    // This method only tracks the metadata for what should be evicted

    Ok(result)
}

/// Insert new context at a specific position, evicting everything after it
///
/// This is the main API for mid-conversation context injection (e.g., RAG retrieval).
/// It evicts all cache content after the insertion point and returns information
/// about what was evicted so it can be re-processed.
///
/// # Process
///
/// 1. Evict everything from `insertion_pos` onwards
/// 2. Caller inserts new context tokens at `insertion_pos`
/// 3. Caller re-processes evicted content to rebuild KV cache
///
/// # Arguments
///
/// * `slot` - Batch slot index for insertion
/// * `insertion_pos` - Position where new context will be inserted
///
/// # Returns
///
/// EvictionResult with information about evicted spans and positions.
/// The caller should use `positions_evicted` to determine what content
/// needs to be re-processed after insertion.
///
/// # Example
///
/// ```ignore
/// // Insert RAG context at position 50
/// let evicted = cache_builder.insert_context_at(slot_id, 50)?;
///
/// // Process new context
/// model.forward(&new_context_tokens, &metadata)?;
///
/// // Re-process evicted content to rebuild cache
/// if !evicted.positions_evicted.is_empty() {
///     let evicted_tokens = get_tokens_for_positions(&evicted);
///     model.forward_kv_only(&evicted_tokens, &metadata)?;
/// }
/// ```
pub fn insert_context_at(
    &mut self,
    slot: usize,
    insertion_pos: usize,
) -> Result<crate::cache::EvictionResult> {
    // Evict from insertion_pos to current position
    let current_pos = self.positions.get(slot).copied().unwrap_or(0);

    if insertion_pos >= current_pos {
        // Nothing to evict - insertion is at or beyond current position
        return Ok(crate::cache::EvictionResult::empty());
    }

    // Evict everything from insertion_pos onwards
    let result = self.evict_range(slot, insertion_pos, current_pos)?;

    // Reset the position to insertion_pos so new content can be written there
    if let Some(pos) = self.positions.get_mut(slot) {
        *pos = insertion_pos;
    }
    if let Some(idx) = self.indices.get_mut(slot) {
        *idx = insertion_pos % self.context;
    }

    Ok(result)
}
