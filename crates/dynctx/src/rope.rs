// ============================================================================
// rope.rs — edge‑aware Rotary Position Encoding helpers
// ============================================================================
//! Minimal utilities for computing and updating rotary positional
//! embeddings when the sequence is represented as a doubly‑linked list with
//! per‑edge `rel_pos` deltas.  The goal is to avoid `O(n)` recompute after
//! every splice; only positions on the *splice boundaries* need patching.
//!
//! We expose a single public routine:
//!
//! ```rust
//! # use dynctx::rope::cumulative_positions;
//! let rel_deltas = vec![1, 2, 4, 3];
//! let absolute_positions = cumulative_positions(0, &rel_deltas).unwrap();
//! assert_eq!(absolute_positions, vec![1, 3, 7, 10]);
//! ```
//!
//! which converts a slice of `rel_pos` values into *absolute* rotary angles
//! (actually integer tick counts; conversion to FP32 angles happens inside
//! the matmul kernel).  The first element of `rel` is interpreted as the
//! delta from `start_angle`.
//!
//! Additionally, we provide a helper `build_dense_view()` that walks the
//! live linked list in the `SlotArena` and returns the `DenseSlice` struct
//! expected by the kernel.

use crate::arena::{ArenaError, DenseSlice, SlotArena, TokenNode};

/// Compute absolute rotary ticks from relative deltas, with overflow checking.  
/// Pure CPU; `rel.len() <= 32k` so this is negligible cost.
#[inline]
pub fn cumulative_positions(start: u32, rel: &[u32]) -> Result<Vec<u32>, ArenaError> {
    let mut out = Vec::with_capacity(rel.len());
    let mut acc = start;
    for &d in rel {
        acc = acc.checked_add(d).ok_or(ArenaError::PositionOverflow)?;
        out.push(acc);
    }
    Ok(out)
}

/// Build a `DenseSlice` covering the entire live list.  Returns token ids,
/// absolute rotary positions, and back‑mapping to NodeKey for gradient
/// writes.
pub fn build_dense_view(arena: &SlotArena) -> Result<DenseSlice<'_>, ArenaError> {
    use crate::node_arena::NodeKey;

    // First pass: collect all keys by following the link chain
    let mut mapping: Vec<NodeKey> = Vec::new();
    let mut node_opt = arena.head;
    while let Some(key) = node_opt {
        mapping.push(key);
        node_opt = arena.get_links(key).next;
    }

    // Second pass: batch access hot data for all keys
    let mut ids = Vec::with_capacity(mapping.len());
    let mut rels = Vec::with_capacity(mapping.len());
    for key in &mapping {
        let node: &TokenNode = arena.get(*key);
        ids.push(node.tok_id);
        rels.push(node.rel_pos);
    }
    // Convert rel -> absolute (first token's rel_pos is distance from a
    // virtual start token at angle 0).
    let abs_pos = cumulative_positions(0, &rels)?;

    Ok(DenseSlice {
        ids,
        abs_pos,
        mapping,
        _marker: std::marker::PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::SlotArena;

    #[test]
    fn test_build_dense_view_performance() {
        let mut arena = SlotArena::new();
        let mut keys = Vec::new(); // Create a large linked list (simulate typical inference sequence)
        for i in 0..5000 {
            if let Some(key) = arena.alloc(i + 1000, i as u32) {
                keys.push(key);
            }
        } // Link them together in sequence
        for window in keys.windows(2) {
            arena.get_links_mut(window[0]).next = Some(window[1]);
            arena.get_links_mut(window[1]).prev = Some(window[0]);
        }
        if !keys.is_empty() {
            arena.head = Some(keys[0]);
            arena.tail = Some(*keys.last().unwrap());
        }

        println!("=== build_dense_view Performance Test ===");
        println!("Testing with {} linked nodes", keys.len());

        // Measure multiple runs to get stable timings
        let mut total_time = std::time::Duration::ZERO;
        let runs = 100;
        for _ in 0..runs {
            let start = std::time::Instant::now();
            let dense_slice = build_dense_view(&arena).unwrap();
            let elapsed = start.elapsed();
            total_time += elapsed;

            // Verify correctness
            assert_eq!(dense_slice.ids.len(), keys.len());
            assert_eq!(dense_slice.abs_pos.len(), keys.len());
            assert_eq!(dense_slice.mapping.len(), keys.len());
        }

        let avg_time = total_time / runs;
        println!(
            "Average build_dense_view time: {:?} (over {} runs)",
            avg_time, runs
        );
        println!("Per-node time: {:?}", avg_time / keys.len() as u32);

        // Performance expectations
        #[cfg(debug_assertions)]
        assert!(
            avg_time.as_micros() < 10000,
            "Average time too slow: {:?}",
            avg_time
        ); // < 10ms in debug

        #[cfg(not(debug_assertions))]
        assert!(
            avg_time.as_micros() < 1000,
            "Average time too slow: {:?}",
            avg_time
        ); // < 1ms in release
    }
}
