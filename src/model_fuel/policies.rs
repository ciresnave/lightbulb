//! Policy → allocator adapter: Lightbulb **decides**, Fuel **executes**.
//!
//! This module is the seam between Lightbulb's cache *policies* (which span is
//! cold, when to shed, whether a prompt prefix is worth reusing) and Fuel's KV
//! block *allocator* (`fuel::kv_block_pool::KvBlockPool` /
//! `fuel::kv_block_pool_device::DeviceKvPool`), which owns block identity,
//! refcounts, the free list, and byte movement.
//!
//! The split is load-bearing and is visible in every signature here:
//!
//! | Lightbulb decides | Fuel executes |
//! |---|---|
//! | [`PagedAdmission::decide`] — admit / reject, incl. a reserve headroom | `blocks_required`, `capacity`, `append` |
//! | [`BlockPrefixIndex::lookup`] — which donor, how many blocks | `splice` (refcount bump, COW) |
//! | [`SpanBlockMap::plan`] — which spans die, which blocks that means | `evict_blocks` → [`EvictReport`] |
//! | [`BlockTierMover`] — which blocks go to disk, and when they come back | `read_block` / `write_block` / `restore` |
//!
//! Nothing in this file allocates, frees, or moves a byte itself. Every mutation
//! of pool state is a call into Fuel with an argument Lightbulb computed.
//!
//! # Policies EXCLUDED from this adapter, and why
//!
//! * **`crate::cache::kv_compression`** (`LowRankCompressor`, grouped
//!   quantization) — **gate-zero failure**. `kv_compression.rs:446` is a live
//!   `todo!("Grouped quantization not yet implemented")`, and
//!   `LowRankCompressor::compute_low_rank` (`:1070`) computes `reshaped` from its
//!   input at `:1085`, never uses it, and returns `Tensor::randn` projections
//!   (`:1092`, `:1099`) — outside `#[cfg(test)]`. It has a rich API and returns
//!   noise. Not wired, not stubbed here.
//! * **`crate::cache::eviction_policy::VotingAggregator`** — real code, but
//!   defective for exactly this use. `compute_aggregated_scores`
//!   (`eviction_policy.rs:132-140`) `continue`s past every non-finite score after
//!   handling `NEG_INFINITY`, so a slot scored `f32::INFINITY` gets **no entry**
//!   in the aggregate and drops out of the ranking. `+INFINITY` is precisely the
//!   "never attended / not in any span — evict me first" signal that
//!   `H2OPolicy` (`h2o_policy.rs:176,180`) and `SegmentedEvictionPolicy`
//!   (`segmented_eviction_policy.rs:343,348,352`) emit. The aggregator would
//!   silently discard the best eviction candidates. This adapter therefore calls
//!   the policies **directly** and does its own aggregation
//!   ([`h2o_block_scores`]), which preserves `+INFINITY` — see the unit test
//!   `h2o_block_scores_preserve_positive_infinity`.
//! * **`crate::cache::tiered_storage::TieredStorageManager`** and
//!   **`crate::cache::tensor_codec`** — sound and tested, but candle-typed
//!   (`DemotedSegment.cpu_kv_layers: Option<Vec<(candlelight Tensor, Tensor)>>`).
//!   Fuel KV never becomes a candle `Tensor`; converting through one would be
//!   both slow and a rule-1 graph-affinity hazard. Only the byte-typed subset —
//!   [`DiskStore`] / `FileDiskStore` — is reused, verbatim.
//! * **`crate::cache::prefix_cache::PrefixKvCache`** — candle-typed *and*
//!   carrying a live hazard: `record_prompt` (`prefix_cache.rs:336`) inserts an
//!   entry with `kv_by_layer: Vec::new()`, and `get`/`get_best_prefix` validate
//!   only `entry.length == tokens.len()`, so that empty placeholder is returned
//!   as a genuine hit. A caller that skips prefill on a hit would skip it with
//!   zero KV. Its *concepts* (SHA-256 keying, LRU, hit stats) are reimplemented
//!   in [`BlockPrefixIndex`], which cannot express a zero-block hit by
//!   construction ([`PrefixMatch::blocks`] is always ≥ 1).
//! * **`PerTokenTracker::select_evictions` / `remaining_ranges`** as a *driver* —
//!   they return individual token positions and arbitrary non-contiguous token
//!   ranges. `evict_blocks` takes whole logical block indices; there is no
//!   sub-block eviction anywhere in Fuel. Per-token output is usable only as an
//!   *input* to a whole-block decision (which blocks became entirely dead), never
//!   as a direct driver, so no such path is offered here.
//!
//! # Granularity mismatches, and how they are aligned
//!
//! **1. Token-granular prefix matching vs whole-block splice.**
//! `PrefixKvCache::get_best_prefix` matches at token granularity. `splice` shares
//! *whole blocks* and sets `dst.filled_tokens += ((to-from)*bs).min(...)`
//! (`kv_block_pool.rs:628`). If a spliced prefix ends mid-block, the next decode
//! step computes `tok_pos = filled_tokens`, `append(1)` allocates **no** new
//! block, and the write lands in the **shared** block — mutating the donor's KV.
//! `cow_break` exists but is called **nowhere** in `fuel-core/src` outside its own
//! definition and unit tests, so nothing in Fuel guards this.
//! **Alignment:** [`BlockPrefixIndex`] only ever indexes and returns *floor-
//! aligned* block counts, and [`splice_prefix`] re-checks the invariant
//! (`filled_tokens % block_size == 0`) after the call and errors if it does not
//! hold. **Documented loss: up to `block_size - 1` tokens of matched prefix at
//! the tail are NOT reused and must be re-prefilled** — reported as
//! [`PrefixMatch::remainder_tokens`].
//!
//! **2. Token-range spans vs logical block indices.**
//! `CacheSpan` is `start_pos..end_pos` in tokens; `evict_blocks` wants block
//! indices. A block at a span boundary holds tokens from *two* spans, so the
//! conversion must be conservative: [`blocks_fully_covered`] (`ceil(start/bs) ..
//! floor(end/bs)`) for what may be evicted, [`blocks_touched`] (`floor(start/bs)
//! .. ceil(end/bs)`) for what must be protected, and the plan subtracts every
//! block touched by a surviving span. **Documented loss: a cold span shorter than
//! `block_size`, or one that does not cover a whole block, frees nothing.**
//!
//! **3. Per-token H2O scores vs whole-block eviction.**
//! [`h2o_block_scores`] aggregates by **min** over the block's tokens (an
//! eviction score, so min = most protected wins). **Documented loss: one hot
//! token retains its entire block — up to `block_size - 1` cold tokens are kept.**
//!
//! # The `still_shared` rule (get this wrong and bookkeeping silently diverges)
//!
//! `EvictReport` returns `freed` **and** `still_shared`, both as `Vec<usize>` of
//! *logical block indices*. A block in `still_shared` was **not detached** — its
//! refcount was > 1 (a spliced prefix), so it is **still resident** and its tokens
//! are **still valid**. Only `freed` blocks lost their contents. Therefore
//! [`SpanBlockMap::apply_report`] computes a span's surviving token ranges as
//! `span_range \ ⋃ token_range(b) for b in report.freed` — `still_shared` is
//! deliberately *not* subtracted — and [`BlockTierMover::demote`] **discards** the
//! byte payloads it read for `still_shared` blocks rather than writing them to
//! disk (a stale disk copy of a block another session is still writing would
//! clobber the sharer on promote).
//!
//! # What Fuel cannot do today (so this adapter does not pretend to)
//!
//! `session_block_table` returns `Err(SessionNotResident)` if **any** logical slot
//! is externalized (`kv_block_pool.rs:430`), and `materialize_block_table` →
//! `forward_paged_step` both go through it. So **evict-then-decode is impossible**:
//! the moment a middle block is evicted, the next decode step fails outright.
//! Everything here is shaped for **evict → restore → resume**, never
//! **evict → decode**. Sub-session tiering *while decoding* needs a Fuel-side
//! change (per-key masking or block-table compaction with RoPE re-injection).

use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

use sha2::{Digest, Sha256};

use fuel::kv_block_pool::{
    EvictReport, Externalized, KvAllocError, KvBlockPool, KvGeometry, PhysBlockId, PoolCapacity,
    SessionHandle,
};
use fuel::kv_block_pool_device::{BlockKind, DeviceKvPool};

use crate::cache::cache_span::{EvictionImpact, SpanId, SpanRegistry, SpanState};
use crate::cache::h2o_policy::H2OPolicy;
use crate::cache::segmented_eviction_policy::SegmentedEvictionPolicy;
use crate::cache::tiered_storage::DiskStore;

// ===========================================================================
// Errors
// ===========================================================================

/// Failures at the policy→allocator boundary.
///
/// `KvAllocError` implements neither `Display` nor `Error`, so it is carried by
/// value and rendered with `Debug` — losing it to a `String` at the boundary
/// would make "was this OutOfBlocks or a mis-sequenced call?" unanswerable.
#[derive(Debug)]
pub enum PolicyError {
    /// Fuel's allocator rejected the call.
    Alloc(KvAllocError),
    /// Fuel's device layer (byte movement through the executor) failed.
    Device(String),
    /// The backing `DiskStore` failed.
    Store(String),
    /// A Lightbulb-side invariant this adapter is responsible for was violated.
    /// These are the guards that keep the allocator from being driven into a
    /// silently-corrupting state (e.g. a non-block-aligned splice).
    Invariant(String),
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyError::Alloc(e) => write!(f, "KV allocator rejected the call: {e:?}"),
            PolicyError::Device(s) => write!(f, "KV device layer: {s}"),
            PolicyError::Store(s) => write!(f, "tier store: {s}"),
            PolicyError::Invariant(s) => write!(f, "adapter invariant violated: {s}"),
        }
    }
}

impl std::error::Error for PolicyError {}

impl From<KvAllocError> for PolicyError {
    fn from(e: KvAllocError) -> Self {
        PolicyError::Alloc(e)
    }
}

// ===========================================================================
// Geometry — the one place `block_size` lives
// ===========================================================================

/// The KV pool geometry, owned by Lightbulb.
///
/// `KvGeometry` has **no `Default`** in Fuel — the consumer constructs it — so
/// `block_size` is a Lightbulb decision available *before any pool exists*. This
/// wrapper is the single place it is chosen, and the only place token↔block
/// arithmetic is done.
///
/// `elem_size` is fixed at 4 (f32): `DeviceKvPool::new` rejects
/// `elem_size != dtype.size_in_bytes()`, and `read_block`/`write_block`/
/// `forward_paged_step` are all f32-only. That is consistent with rule 2
/// (all-f32 weights via `loader_f32`), at the cost of 2× the KV bytes of a bf16
/// pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedGeometry {
    kv: KvGeometry,
}

impl PagedGeometry {
    /// Build from a loaded model's config plus the two Lightbulb-chosen knobs.
    ///
    /// `block_size` trades prefix-sharing granularity (bigger blocks ⇒ more of a
    /// matched prefix falls in the unusable remainder) against eviction
    /// granularity (bigger blocks ⇒ more short spans free nothing). There is no
    /// measurement informing the choice yet.
    pub fn for_llama(cfg: &fuel::lazy::LlamaConfig, block_size: usize, num_blocks: usize) -> Self {
        Self {
            kv: KvGeometry {
                n_layers: cfg.n_layers,
                num_blocks,
                block_size,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
                elem_size: 4, // f32 — see the type doc
            },
        }
    }

    /// Wrap a geometry built elsewhere (e.g. read back from an existing pool via
    /// `KvBlockPool::geometry()`).
    pub fn from_kv(kv: KvGeometry) -> Self {
        Self { kv }
    }

    /// The geometry to hand to `KvBlockPool::new` / `DeviceKvPool::new`.
    pub fn kv(&self) -> KvGeometry {
        self.kv
    }

    pub fn block_size(&self) -> usize {
        self.kv.block_size
    }

    pub fn n_layers(&self) -> usize {
        self.kv.n_layers
    }

    /// f32 elements in one physical block for one layer, one of {K, V} —
    /// the length `read_block` returns and `write_block` demands.
    pub fn block_elems(&self) -> usize {
        self.kv.block_size * self.kv.n_kv_heads * self.kv.head_dim
    }

    /// Token positions covered by logical block `b`.
    pub fn token_range_of_block(&self, b: usize) -> Range<usize> {
        let bs = self.kv.block_size;
        (b * bs)..((b + 1) * bs)
    }

    /// The block holding the newest token — never evictable while the session is
    /// live. When `filled % block_size != 0` this is also the block
    /// `forward_paged_step` writes the *next* token into (it computes
    /// `tok_pos = filled_tokens` and indexes `tok_pos / block_size`), so this one
    /// method covers both hazards. `None` for an empty session.
    pub fn protected_tail_block(&self, filled_tokens: usize) -> Option<usize> {
        if filled_tokens == 0 {
            None
        } else {
            Some((filled_tokens - 1) / self.kv.block_size)
        }
    }
}

/// Blocks **entirely inside** `range` — the CONSERVATIVE mapping, the only one
/// safe to hand to `evict_blocks`.
///
/// `ceil(start/bs) .. floor(end/bs)`. A block straddling `range`'s start or end
/// also holds a neighbour's tokens, so it is excluded. An empty result is always
/// the canonical `0..0`, so callers may compare ranges for equality rather than
/// having to remember that `2..2` and `0..0` mean the same thing.
pub fn blocks_fully_covered(range: &Range<usize>, block_size: usize) -> Range<usize> {
    if block_size == 0 || range.start >= range.end {
        return 0..0;
    }
    let first = range.start.div_ceil(block_size);
    let last = range.end / block_size;
    if first >= last { 0..0 } else { first..last }
}

/// Blocks **overlapping** `range` — the LIBERAL mapping, used only to compute
/// what a *surviving* span protects. Handing this to `evict_blocks` would evict
/// a neighbour's live tokens.
///
/// `floor(start/bs) .. ceil(end/bs)`.
pub fn blocks_touched(range: &Range<usize>, block_size: usize) -> Range<usize> {
    if block_size == 0 || range.start >= range.end {
        return 0..0;
    }
    (range.start / block_size)..range.end.div_ceil(block_size)
}

/// Floor-align a token-granular prefix match down to whole blocks.
///
/// Returns `(blocks, remainder_tokens)`. The remainder is the documented loss:
/// up to `block_size - 1` matched tokens that must be re-prefilled because
/// sharing them would leave `dst.filled_tokens` mid-block.
pub fn floor_align_prefix(match_len: usize, block_size: usize) -> (usize, usize) {
    if block_size == 0 {
        return (0, match_len);
    }
    (match_len / block_size, match_len % block_size)
}

// ===========================================================================
// Admission (C-1)
// ===========================================================================

/// Lightbulb's admission verdict. Fuel never decides this — it only reports
/// `blocks_required` and `capacity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit { blocks_needed: usize },
    Reject { blocks_needed: usize, blocks_available: usize },
}

impl AdmissionDecision {
    pub fn is_admit(&self) -> bool {
        matches!(self, AdmissionDecision::Admit { .. })
    }
}

/// Admission policy over a pool's advertised capacity.
///
/// The *policy* is `reserve_blocks`: blocks Lightbulb refuses to spend on new
/// work, held back so a live session can always take its next `append`. Fuel has
/// no opinion about that — `free_blocks` would happily hand out the last block.
#[derive(Clone, Copy, Debug)]
pub struct PagedAdmission {
    reserve_blocks: usize,
}

impl PagedAdmission {
    pub fn new(reserve_blocks: usize) -> Self {
        Self { reserve_blocks }
    }

    pub fn reserve_blocks(&self) -> usize {
        self.reserve_blocks
    }

    /// Decide whether `add_tokens` more may be admitted into a session currently
    /// holding `cur_filled`.
    ///
    /// The block arithmetic is **Fuel's** (`blocks_required`), deliberately never
    /// reimplemented here — the whole point of that method existing is that the
    /// consumer's math cannot drift from the allocator's.
    pub fn decide(
        &self,
        pool: &KvBlockPool,
        cur_filled: usize,
        add_tokens: usize,
    ) -> AdmissionDecision {
        let need = pool.blocks_required(cur_filled, add_tokens);
        self.verdict(need, pool.capacity())
    }

    /// Batch form: one `(cur_filled, add_tokens)` per sequence. Summation stays
    /// in Fuel (`blocks_required_batch`) so it stays correct if geometry ever
    /// makes the per-sequence cost non-additive.
    pub fn decide_batch(&self, pool: &KvBlockPool, seqs: &[(usize, usize)]) -> AdmissionDecision {
        let need = pool.blocks_required_batch(seqs);
        self.verdict(need, pool.capacity())
    }

    fn verdict(&self, need: usize, cap: PoolCapacity) -> AdmissionDecision {
        let available = cap.free_blocks.saturating_sub(self.reserve_blocks);
        if need <= available {
            AdmissionDecision::Admit { blocks_needed: need }
        } else {
            AdmissionDecision::Reject { blocks_needed: need, blocks_available: available }
        }
    }
}

// ===========================================================================
// Prefix sharing (C-3 splice)
// ===========================================================================

/// Hit/miss accounting for [`BlockPrefixIndex`]. Same shape as
/// `PrefixCacheStats`, but counted in **blocks and tokens actually shareable**,
/// not in tokens merely matched — the alignment remainder is never counted as
/// saved work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockPrefixStats {
    pub hits: usize,
    pub misses: usize,
    pub blocks_indexed: usize,
    /// Tokens genuinely served from a donor's blocks (always a multiple of
    /// `block_size`).
    pub tokens_shared: usize,
    /// Tokens that matched but fell in the sub-block remainder and had to be
    /// re-prefilled. This is the alignment loss, measured.
    pub tokens_lost_to_alignment: usize,
}

/// A block-aligned prefix hit. Constructed only by [`BlockPrefixIndex::lookup`],
/// which never produces `blocks == 0` — the structural fix for the
/// `record_prompt` placeholder hazard that made an empty entry look like a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Session whose blocks `[0, blocks)` are to be spliced. Must still be open.
    pub donor: SessionHandle,
    /// Whole blocks to share. Always ≥ 1.
    pub blocks: usize,
    /// `blocks * block_size` — tokens the splice actually carries.
    pub shared_tokens: usize,
    /// Tokens of the *query* left to prefill normally. Includes both the
    /// alignment remainder and everything past the match.
    pub remainder_tokens: usize,
}

#[derive(Clone, Copy, Debug)]
struct PrefixEntry {
    donor: SessionHandle,
    blocks: usize,
}

/// Chain-hashed, block-aligned prefix index: `chain_hash(blocks[0..=i]) →
/// (donor session, i+1 blocks)`.
///
/// No tensors are stored. With a refcounted block pool the KV *is* the donor's
/// resident blocks, and reuse is a `splice` (refcount bump) — which is why
/// `PrefixKvEntry.kv_by_layer` is not merely candle-typed but the wrong model
/// entirely.
///
/// Chain hashing (`h_i = SHA256(h_{i-1} ‖ block_i)`, the vLLM shape) makes
/// longest-prefix lookup `O(n_blocks)` exact probes instead of
/// `get_best_prefix`'s `O(n_tokens)` trial re-hashes, and it can only ever
/// express block-aligned lengths — the alignment rule is structural, not a
/// checked convention.
///
/// **Donor lifetime is this index's responsibility.** A donor must stay open
/// (never `discard`ed) while it is referenced; [`BlockPrefixIndex::is_donor`]
/// and [`BlockPrefixIndex::forget_donor`] are the accounting.
pub struct BlockPrefixIndex {
    block_size: usize,
    by_hash: HashMap<[u8; 32], PrefixEntry>,
    donor_refs: HashMap<SessionHandle, usize>,
    stats: BlockPrefixStats,
}

impl BlockPrefixIndex {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            block_size,
            by_hash: HashMap::new(),
            donor_refs: HashMap::new(),
            stats: BlockPrefixStats::default(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn stats(&self) -> BlockPrefixStats {
        self.stats
    }

    /// Chain hash of `tokens`' whole blocks, one entry per prefix length.
    fn chain_hashes(&self, tokens: &[u32]) -> Vec<[u8; 32]> {
        let n_full = tokens.len() / self.block_size;
        let mut out = Vec::with_capacity(n_full);
        let mut prev = [0u8; 32];
        for b in 0..n_full {
            let mut h = Sha256::new();
            h.update(b"lightbulb.block-prefix.v1");
            h.update(prev);
            for &t in &tokens[b * self.block_size..(b + 1) * self.block_size] {
                h.update(t.to_le_bytes());
            }
            let digest: [u8; 32] = h.finalize().into();
            out.push(digest);
            prev = digest;
        }
        out
    }

    /// Index `donor`'s prompt so later prompts can splice from it.
    ///
    /// Only whole blocks are indexed; a prompt shorter than one block indexes
    /// nothing and returns 0. Returns the number of block-prefix lengths added.
    ///
    /// The caller is asserting that `donor`'s logical blocks `[0, n_full)` hold
    /// exactly the KV for `tokens`' first `n_full * block_size` tokens.
    pub fn record(&mut self, tokens: &[u32], donor: SessionHandle) -> usize {
        let hashes = self.chain_hashes(tokens);
        let mut added = 0;
        for (i, h) in hashes.into_iter().enumerate() {
            let entry = PrefixEntry { donor, blocks: i + 1 };
            if let Some(prev) = self.by_hash.insert(h, entry) {
                // Re-pointing an existing chain at a new donor: release the old.
                self.release_donor(prev.donor);
            } else {
                added += 1;
            }
            *self.donor_refs.entry(donor).or_insert(0) += 1;
        }
        self.stats.blocks_indexed += added;
        added
    }

    /// Longest block-aligned prefix of `tokens` held by some donor.
    ///
    /// Returns `None` on a miss. On a hit, `blocks >= 1` is guaranteed:
    /// a zero-block "hit" is unrepresentable, so a caller cannot skip prefill
    /// on an entry that carries no KV.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<PrefixMatch> {
        let hashes = self.chain_hashes(tokens);
        let mut best: Option<PrefixEntry> = None;
        for h in &hashes {
            match self.by_hash.get(h) {
                Some(e) => best = Some(*e),
                // Chain hashes: a miss at depth i means every deeper prefix
                // differs too, so stop.
                None => break,
            }
        }
        match best {
            None => {
                self.stats.misses += 1;
                None
            }
            Some(e) => {
                debug_assert!(e.blocks >= 1, "zero-block prefix entry is unrepresentable");
                let shared = e.blocks * self.block_size;
                let remainder = tokens.len().saturating_sub(shared);
                self.stats.hits += 1;
                self.stats.tokens_shared += shared;
                // The alignment loss for THIS lookup: matched tokens that the
                // block-aligned index could not express. Bounded by
                // block_size - 1 when the whole prompt matched.
                self.stats.tokens_lost_to_alignment += remainder.min(self.block_size - 1);
                Some(PrefixMatch {
                    donor: e.donor,
                    blocks: e.blocks,
                    shared_tokens: shared,
                    remainder_tokens: remainder,
                })
            }
        }
    }

    /// Is this session still referenced as a splice source? While true, the
    /// caller must NOT `KvBlockPool::discard` it.
    pub fn is_donor(&self, s: SessionHandle) -> bool {
        self.donor_refs.get(&s).copied().unwrap_or(0) > 0
    }

    pub fn donors(&self) -> Vec<SessionHandle> {
        let mut v: Vec<SessionHandle> =
            self.donor_refs.iter().filter(|&(_, &n)| n > 0).map(|(&s, _)| s).collect();
        v.sort_by_key(|s| format!("{s:?}"));
        v
    }

    /// Drop every entry pointing at `donor`, making it safe to `discard`.
    /// Returns the number of entries removed.
    pub fn forget_donor(&mut self, donor: SessionHandle) -> usize {
        let before = self.by_hash.len();
        self.by_hash.retain(|_, e| e.donor != donor);
        self.donor_refs.remove(&donor);
        before - self.by_hash.len()
    }

    fn release_donor(&mut self, donor: SessionHandle) {
        if let Some(n) = self.donor_refs.get_mut(&donor) {
            *n = n.saturating_sub(1);
        }
    }
}

/// Execute a [`PrefixMatch`]: Fuel splices the donor's blocks into `dst`.
///
/// Lightbulb decided *which* donor and *how many* blocks; this function is the
/// call, plus the two guards that keep the call from being silently corrupting:
///
/// 1. `dst` must be **empty**. `splice` appends, so splicing into a non-empty
///    session would put the shared prefix after existing KV — a different
///    operation with different alignment.
/// 2. After the call, `filled_tokens(dst) % block_size` must be **0**. If it is
///    not, the next `forward_paged_step` writes into the donor's block
///    (`append(1)` allocates nothing, `resident_block(tok_pos / bs)` returns the
///    shared block, `write_slice` mutates it) and `cow_break` is never called
///    anywhere in Fuel. This check is the only thing standing between a
///    mis-computed match and silent cross-session KV corruption.
///
/// Returns the number of tokens now resident in `dst`.
pub fn splice_prefix(
    pool: &mut KvBlockPool,
    m: &PrefixMatch,
    dst: SessionHandle,
) -> Result<usize, PolicyError> {
    if m.blocks == 0 {
        return Err(PolicyError::Invariant(
            "PrefixMatch with 0 blocks carries no KV; refusing to skip prefill".into(),
        ));
    }
    let bs = pool.geometry().block_size;
    match pool.filled_tokens(dst) {
        None => return Err(PolicyError::Alloc(KvAllocError::UnknownSession)),
        Some(0) => {}
        Some(n) => {
            return Err(PolicyError::Invariant(format!(
                "splice_prefix requires an empty destination; dst holds {n} tokens"
            )));
        }
    }
    let donor_blocks = pool
        .session_blocks(m.donor)
        .ok_or(PolicyError::Alloc(KvAllocError::UnknownSession))?;
    if m.blocks > donor_blocks {
        return Err(PolicyError::Invariant(format!(
            "donor holds {donor_blocks} blocks but the match claims {}; the donor was \
             discarded or shrank while indexed",
            m.blocks
        )));
    }

    // **Decide the alignment invariant BEFORE mutating anything.** Fuel's
    // `splice` sets `dst.filled_tokens += ((to-from)*bs).min(src_filled -
    // from*bs)` (`kv_block_pool.rs:641`). With `from = 0` and `dst` verified
    // empty above, the post-splice fill is *exactly* that expression, so it is
    // predictable. Checking after the call would be too late: `splice` has by
    // then pushed shared slots into `dst` and bumped the donor's refcounts, and
    // there is no unsplice. The old post-check returned a correct error over an
    // already-corrupting state — a caller that logged and continued, or retried,
    // decoded into the donor's block regardless.
    let donor_filled = pool
        .filled_tokens(m.donor)
        .ok_or(PolicyError::Alloc(KvAllocError::UnknownSession))?;
    let projected = (m.blocks * bs).min(donor_filled);
    if projected % bs != 0 {
        return Err(PolicyError::Invariant(format!(
            "post-splice fill would be {projected}, not block-aligned (block_size={bs}); \
             the next decode step would write into the donor's shared block. Refused \
             before splicing — dst is untouched and the donor's refcounts are unchanged"
        )));
    }

    pool.splice(m.donor, dst, 0, m.blocks)?;

    // Belt and braces: if this ever fires, the projection above disagrees with
    // Fuel and `dst` IS in the corrupting state. Distinguish it loudly from the
    // pre-check, because the remedy is different — fix the projection, not the
    // caller.
    let filled = pool.filled_tokens(dst).unwrap_or(0);
    if filled % bs != 0 {
        return Err(PolicyError::Invariant(format!(
            "post-splice fill {filled} is not block-aligned (block_size={bs}) but the \
             pre-check projected {projected}: Fuel's splice fill semantics have changed, \
             and dst is now in the state this guard exists to prevent. Re-derive the \
             projection in splice_prefix against kv_block_pool.rs"
        )));
    }
    Ok(filled)
}

// ===========================================================================
// Segmented eviction: spans → block indices → EvictReport → span state
// ===========================================================================

/// What Lightbulb decided to evict, in Fuel's coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvictionPlan {
    /// Logical block indices to hand to `evict_blocks`. Sorted, deduplicated,
    /// every one fully inside a victim span and touched by no survivor.
    pub blocks: Vec<usize>,
    /// The spans whose eviction this plan implements, most-evictable first.
    pub victim_spans: Vec<SpanId>,
    /// The block excluded because it holds the newest token / is the next write
    /// target. Kept for reporting, never in `blocks`.
    pub protected_tail: Option<usize>,
}

impl EvictionPlan {
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// What actually happened, reconciled against `EvictReport`.
#[derive(Clone, Debug, PartialEq)]
pub struct EvictionOutcome {
    /// Per victim span, how it was affected.
    pub impacts: Vec<(SpanId, EvictionImpact)>,
    /// Blocks Fuel actually detached (contents gone).
    pub blocks_freed: Vec<usize>,
    /// Blocks Fuel left in place because another session still references them.
    /// **These are still resident and their tokens are still valid.**
    pub blocks_still_resident: Vec<usize>,
}

/// Maps Lightbulb's token-coordinate spans onto Fuel's logical block indices,
/// and reconciles the resulting `EvictReport` back onto span state.
#[derive(Clone, Copy, Debug)]
pub struct SpanBlockMap {
    geom: PagedGeometry,
}

impl SpanBlockMap {
    pub fn new(geom: PagedGeometry) -> Self {
        Self { geom }
    }

    /// Turn a ranked span list into a block set.
    ///
    /// `ranked` is `SegmentedEvictionPolicy::rank_spans_for_eviction`'s output
    /// (ascending relevance ⇒ most-evictable first). Spans are taken greedily
    /// until the yielded block set reaches `want_blocks`.
    ///
    /// The set is the union of each victim's **fully covered** blocks, minus every
    /// block **touched** by a surviving span in the same slot, minus the protected
    /// tail block, minus anything at or past `session_blocks`.
    pub fn plan(
        &self,
        registry: &SpanRegistry,
        slot: usize,
        ranked: &[(SpanId, f32)],
        want_blocks: usize,
        session_blocks: usize,
        filled_tokens: usize,
    ) -> EvictionPlan {
        let protected_tail = self.geom.protected_tail_block(filled_tokens);

        // Every span living in this slot that could still hold live tokens.
        let live_spans: Vec<(SpanId, Range<usize>)> = registry
            .spans_for_slot(slot)
            .into_iter()
            .filter(|s| !s.is_fully_evicted())
            .map(|s| (s.id, s.start_pos..s.end_pos))
            .collect();

        let mut victims: Vec<SpanId> = Vec::new();
        let mut chosen: Vec<usize> = Vec::new();

        for &(span_id, _score) in ranked {
            if want_blocks > 0 && chosen.len() >= want_blocks {
                break;
            }
            if !live_spans.iter().any(|(id, _)| *id == span_id) {
                continue; // ranked a span this slot does not own (or already dead)
            }
            victims.push(span_id);
            chosen = self.blocks_for_victims(&live_spans, &victims, session_blocks, protected_tail);
        }

        // Greedy admission may overshoot on the last span; that is correct —
        // blocks come in whole-span units here, never partial.
        EvictionPlan { blocks: chosen, victim_spans: victims, protected_tail }
    }

    /// Convenience: rank with the real policy, then plan. Proves the two compose
    /// without `VotingAggregator` in the path.
    pub fn plan_with_policy(
        &self,
        policy: &SegmentedEvictionPolicy,
        registry: &SpanRegistry,
        slot: usize,
        want_blocks: usize,
        session_blocks: usize,
        filled_tokens: usize,
    ) -> EvictionPlan {
        let ranked = policy.rank_spans_for_eviction(registry);
        self.plan(registry, slot, &ranked, want_blocks, session_blocks, filled_tokens)
    }

    /// Turn H2O's per-block scores into a block set, for callers whose eviction
    /// signal is attention rather than spans.
    ///
    /// `scores` is [`h2o_block_scores`]'s output — descending, evict-first.
    /// Blocks are taken in that order until `want_blocks` is reached, skipping:
    ///
    /// * **`NEG_INFINITY`** — H2O's protected/recent marker. Since
    ///   [`h2o_block_scores`] aggregates by *minimum*, one protected token makes
    ///   its whole block unevictable; ignoring that here would discard the
    ///   aggregation's entire purpose.
    /// * **`NaN`** — an unknown score. Evicting on "don't know" is not a
    ///   decision, and `NaN` also has no defined position in the sort order, so
    ///   its rank is meaningless.
    /// * the protected tail block (the next write target), and anything at or
    ///   past `session_blocks`.
    ///
    /// **`victim_spans` is empty**, because H2O ranks blocks and knows nothing
    /// about spans. [`Self::apply_report`] will therefore record no span impacts
    /// for this plan — correct for a caller that does not track spans, wrong for
    /// one that does. A caller holding a `SpanRegistry` should use [`Self::plan`]
    /// or [`Self::plan_with_policy`], or intersect the two block sets itself.
    /// This adapter will not guess which.
    pub fn plan_from_block_scores(
        &self,
        scores: &[(usize, f32)],
        want_blocks: usize,
        session_blocks: usize,
        filled_tokens: usize,
    ) -> EvictionPlan {
        let protected_tail = self.geom.protected_tail_block(filled_tokens);
        let mut blocks: Vec<usize> = Vec::new();
        for &(b, score) in scores {
            if want_blocks > 0 && blocks.len() >= want_blocks {
                break;
            }
            if score.is_nan() || (score.is_infinite() && score.is_sign_negative()) {
                continue;
            }
            if Some(b) == protected_tail || b >= session_blocks {
                continue;
            }
            blocks.push(b);
        }
        blocks.sort_unstable();
        blocks.dedup();
        EvictionPlan { blocks, victim_spans: Vec::new(), protected_tail }
    }

    /// End-to-end H2O eviction: score, plan, detach.
    ///
    /// This is the wiring that makes [`h2o_block_scores`] reachable from
    /// ordinary code rather than only from tests. Lightbulb scores the tokens
    /// and decides which blocks die; Fuel performs the detach. Session geometry
    /// (`session_blocks`, `filled_tokens`) is read from the pool rather than
    /// taken as arguments, because those are Fuel's facts and a caller passing
    /// its own stale copies is exactly how the tail block gets evicted.
    ///
    /// Returns the plan alongside Fuel's report so the caller can reconcile.
    ///
    /// **Bytes are not preserved** — `evict_blocks` is the pure detach. A caller
    /// that needs this KV back later should take the plan from
    /// [`Self::plan_from_block_scores`] and hand its `blocks` to
    /// [`BlockTierMover::demote`], which captures the payloads first.
    pub fn evict_with_h2o(
        &self,
        pool: &mut KvBlockPool,
        policy: &H2OPolicy,
        positions: &HashMap<usize, usize>,
        s: SessionHandle,
        want_blocks: usize,
    ) -> Result<(EvictionPlan, EvictReport), PolicyError> {
        let session_blocks =
            pool.session_blocks(s).ok_or(PolicyError::Alloc(KvAllocError::UnknownSession))?;
        let filled_tokens =
            pool.filled_tokens(s).ok_or(PolicyError::Alloc(KvAllocError::UnknownSession))?;
        let scores = h2o_block_scores(policy, positions, &self.geom);
        let plan = self.plan_from_block_scores(&scores, want_blocks, session_blocks, filled_tokens);
        let report = pool.evict_blocks(s, &plan.blocks)?;
        Ok((plan, report))
    }

    fn blocks_for_victims(
        &self,
        live_spans: &[(SpanId, Range<usize>)],
        victims: &[SpanId],
        session_blocks: usize,
        protected_tail: Option<usize>,
    ) -> Vec<usize> {
        let bs = self.geom.block_size();

        let mut candidate: BTreeSet<usize> = BTreeSet::new();
        for (id, range) in live_spans {
            if victims.contains(id) {
                candidate.extend(blocks_fully_covered(range, bs));
            }
        }
        // A block touched by ANY survivor holds live tokens — the conservative
        // subtraction. Without it, a boundary block would take a neighbour's
        // tokens with it.
        for (id, range) in live_spans {
            if !victims.contains(id) {
                for b in blocks_touched(range, bs) {
                    candidate.remove(&b);
                }
            }
        }
        if let Some(t) = protected_tail {
            candidate.remove(&t);
        }
        candidate.into_iter().filter(|&b| b < session_blocks).collect()
    }

    /// Reconcile Fuel's `EvictReport` back onto span state.
    ///
    /// **The `still_shared` rule.** Only `report.freed` blocks lost their
    /// contents. A block in `report.still_shared` was left in place because
    /// another session references it — it is still resident and its tokens are
    /// still valid — so it is deliberately NOT subtracted from a span's surviving
    /// range. A span all of whose blocks came back `still_shared` is `Unchanged`,
    /// not `FullyEvicted`; recording it as fully evicted would make Lightbulb
    /// believe memory was reclaimed that Fuel never released, and would make it
    /// re-prefill tokens that are still sitting in the pool.
    pub fn apply_report(
        &self,
        registry: &mut SpanRegistry,
        plan: &EvictionPlan,
        report: &EvictReport,
    ) -> EvictionOutcome {
        debug_assert!(
            report.freed.iter().all(|b| !report.still_shared.contains(b)),
            "a block cannot be both freed and still_shared",
        );

        let cuts: Vec<Range<usize>> =
            report.freed.iter().map(|&b| self.geom.token_range_of_block(b)).collect();

        let mut impacts = Vec::with_capacity(plan.victim_spans.len());
        for &span_id in &plan.victim_spans {
            let Some(span) = registry.get(span_id) else { continue };
            // **Start from what the span still holds, not from what it once
            // held.** A span is evicted over as many rounds as the scheduler
            // needs; recomputing from `start_pos..end_pos` every time re-adds
            // every range an earlier round already freed. Evict block 0 then
            // block 1 of a [0,32) span at bs=8 and the second call would report
            // [0,8) ∪ [16,32) — resurrecting [0,8), which round one detached.
            // Lightbulb would then believe tokens are resident that Fuel has
            // already handed back to the free list.
            let full = span.start_pos..span.end_pos;
            let base: Vec<Range<usize>> = match &span.state {
                SpanState::Active => vec![full.clone()],
                SpanState::PartiallyEvicted { remaining_ranges } => remaining_ranges.clone(),
                SpanState::FullyEvicted => Vec::new(),
            };
            let remaining: Vec<Range<usize>> =
                base.iter().flat_map(|p| subtract_ranges(p.clone(), &cuts)).collect();

            let (state, impact) = if remaining.is_empty() {
                (SpanState::FullyEvicted, EvictionImpact::FullyEvicted)
            } else if remaining == base {
                // Nothing of this span was detached THIS round — e.g. every
                // block it owned came back in `still_shared`. Preserve the
                // existing state: forcing `Active` here would resurrect an
                // earlier round's evictions, which is the same bug one level up.
                (span.state.clone(), EvictionImpact::Unchanged)
            } else {
                (
                    SpanState::PartiallyEvicted { remaining_ranges: remaining.clone() },
                    EvictionImpact::PartiallyEvicted { remaining },
                )
            };

            if let Some(s) = registry.get_mut(span_id) {
                s.state = state;
            }
            impacts.push((span_id, impact));
        }

        EvictionOutcome {
            impacts,
            blocks_freed: report.freed.clone(),
            blocks_still_resident: report.still_shared.clone(),
        }
    }
}

/// `base` minus every range in `cuts`, as sorted, coalesced, non-empty pieces.
fn subtract_ranges(base: Range<usize>, cuts: &[Range<usize>]) -> Vec<Range<usize>> {
    if base.start >= base.end {
        return Vec::new();
    }
    let mut pieces = vec![base];
    for c in cuts {
        if c.start >= c.end {
            continue;
        }
        let mut next = Vec::with_capacity(pieces.len() + 1);
        for p in pieces {
            if c.end <= p.start || c.start >= p.end {
                next.push(p);
                continue;
            }
            if p.start < c.start {
                next.push(p.start..c.start);
            }
            if c.end < p.end {
                next.push(c.end..p.end);
            }
        }
        pieces = next;
    }
    pieces.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(pieces.len());
    for p in pieces {
        match merged.last_mut() {
            Some(last) if last.end == p.start => last.end = p.end,
            _ => merged.push(p),
        }
    }
    merged
}

// ===========================================================================
// H2O → block scores
// ===========================================================================

/// The slot→position map `H2OPolicy` expects, built explicitly as the identity.
///
/// `H2OPolicy::slot_metadata` is keyed by the map's KEY ("cache slot"), while
/// `SegmentedEvictionPolicy::update_scores` looks that metadata up by absolute
/// POSITION. The two only agree because the one live caller
/// (`ParallelCacheBuilder::update_attention_scores`) passes an identity map.
/// This constructor is that invariant made into code rather than a comment:
/// single-sequence paged decode has exactly one coordinate system, and the only
/// map this adapter builds is the identity. A real batch-slot map would
/// mis-attribute attention, so none is offered.
pub fn identity_positions(n_tokens: usize) -> HashMap<usize, usize> {
    (0..n_tokens).map(|p| (p, p)).collect()
}

/// Aggregate H2O's per-token eviction scores into per-block scores.
///
/// Higher score = evict sooner (H2O's convention). A block can only be evicted
/// whole, so the block's score is the **minimum** over its tokens: the most
/// protected token dominates. `NEG_INFINITY` (recent / protected) therefore makes
/// the whole block unevictable, and `+INFINITY` survives only when *every* token
/// in the block was never attended to.
///
/// **Documented loss:** one hot token retains its entire block — up to
/// `block_size - 1` cold tokens are kept.
///
/// **`+INFINITY` is preserved.** `VotingAggregator::compute_aggregated_scores`
/// drops those slots entirely (`eviction_policy.rs:132-140` `continue`s past
/// every non-finite score after handling `NEG_INFINITY`), which silently discards
/// H2O's highest-priority candidates. That is why this aggregation exists rather
/// than reusing the aggregator.
///
/// Returns `(block_index, score)` sorted by score descending (evict first).
pub fn h2o_block_scores(
    policy: &H2OPolicy,
    positions: &HashMap<usize, usize>,
    geom: &PagedGeometry,
) -> Vec<(usize, f32)> {
    let bs = geom.block_size();
    let per_slot = policy.compute_eviction_scores(positions.len(), positions.len(), positions);

    let mut per_block: HashMap<usize, f32> = HashMap::new();
    for (slot, score) in per_slot {
        let Some(&pos) = positions.get(&slot) else { continue };
        let b = pos / bs;
        let e = per_block.entry(b).or_insert(f32::INFINITY);
        if score < *e {
            *e = score;
        }
    }

    let mut out: Vec<(usize, f32)> = per_block.into_iter().collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
    });
    out
}

// ===========================================================================
// Tiering: block bytes ↔ DiskStore
// ===========================================================================

/// The two planes of a paged KV pool, as one object.
///
/// * the **block plane** — block identity, refcounts, the free list
///   (`KvBlockPool`, pure, byte-free);
/// * the **byte plane** — `read_block` / `write_block`.
///
/// `DeviceKvPool` owns both, and demote/promote need to alternate between them.
/// Exposing them as one trait keeps the alternation *sequential* (read, then
/// evict, then store) instead of forcing a caller to hold `&DeviceKvPool` and
/// `&mut KvBlockPool` simultaneously — which cannot be obtained safely.
///
/// This is deliberately **not** `DeviceKvPool::evict_blocks` / `restore`: those
/// return a `DeviceEvicted` whose bytes are **inaccessible** (`core_handle` and
/// `saved` are private, `SavedBlock` is private, and the only accessors are
/// `covers()` / `fidelity()` / `saved_block_count()`), so there is no CPU→disk
/// path through them. The usable seam is one level down and fully public:
/// `read_block` / `write_block` plus the pure `core_mut().evict_blocks` /
/// `core_mut().restore`.
pub trait BlockPlane {
    fn plane_geometry(&self) -> KvGeometry;
    /// The pure block allocator — read view.
    fn blocks(&self) -> &KvBlockPool;
    /// The pure block allocator — the view Fuel mutates through.
    fn blocks_mut(&mut self) -> &mut KvBlockPool;
    fn read_block_f32(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
    ) -> Result<Vec<f32>, PolicyError>;
    fn write_block_f32(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
        data: &[f32],
    ) -> Result<(), PolicyError>;
}

impl BlockPlane for DeviceKvPool {
    fn plane_geometry(&self) -> KvGeometry {
        self.geometry()
    }

    fn blocks(&self) -> &KvBlockPool {
        self.core()
    }

    fn blocks_mut(&mut self) -> &mut KvBlockPool {
        self.core_mut()
    }

    fn read_block_f32(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
    ) -> Result<Vec<f32>, PolicyError> {
        DeviceKvPool::read_block(self, layer, kind, phys)
            .map_err(|e| PolicyError::Device(e.to_string()))
    }

    fn write_block_f32(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
        data: &[f32],
    ) -> Result<(), PolicyError> {
        DeviceKvPool::write_block(self, layer, kind, phys, data)
            .map_err(|e| PolicyError::Device(e.to_string()))
    }
}

/// The reversible receipt for a demotion. Carries Fuel's `Externalized` handle
/// (consumed by `restore`) plus the disk keys Lightbulb wrote.
///
/// `still_shared` blocks appear here for reporting but have **no key**: they were
/// never detached, so their bytes are still live in the pool and writing a stale
/// disk copy back over them on promote would clobber the sharer.
#[derive(Debug)]
pub struct DemotionTicket {
    handle: Externalized,
    pub freed: Vec<usize>,
    pub still_shared: Vec<usize>,
    /// `(logical block index, store key)` — one entry per **freed** block only.
    pub keys: Vec<(usize, String)>,
}

impl DemotionTicket {
    pub fn bytes_stored_for(&self) -> Vec<usize> {
        self.keys.iter().map(|(b, _)| *b).collect()
    }
}

/// Moves whole blocks between the pool and a `DiskStore`.
///
/// Lightbulb owns the bytes and the ordering; Fuel owns block identity. The
/// ordering is not stylistic — `restore` allocates **fresh** physical blocks, so
/// a physical id captured before the restore points at whatever the pool handed
/// to someone else in the meantime.
#[derive(Clone, Debug)]
pub struct BlockTierMover {
    geom: PagedGeometry,
    key_prefix: String,
}

impl BlockTierMover {
    /// `key_prefix` must be unique per session — keys are
    /// `"{key_prefix}.b{logical:06}"`.
    pub fn new(geom: PagedGeometry, key_prefix: impl Into<String>) -> Self {
        Self { geom, key_prefix: key_prefix.into() }
    }

    fn key_for(&self, logical: usize) -> String {
        format!("{}.b{:06}", self.key_prefix, logical)
    }

    /// Demote `blocks` of session `s` to `store`.
    ///
    /// Sequence, which is mandatory:
    /// 1. **read first** — bytes must be captured while the blocks are still
    ///    resident (`evict_blocks` returns them to the free list);
    /// 2. `core.evict_blocks` — the pure, byte-free detach;
    /// 3. store payloads for `report.freed` **only**, dropping every
    ///    `still_shared` payload.
    pub fn demote<P: BlockPlane + ?Sized>(
        &self,
        plane: &mut P,
        store: &mut dyn DiskStore,
        s: SessionHandle,
        blocks: &[usize],
    ) -> Result<DemotionTicket, PolicyError> {
        let n_layers = self.geom.n_layers();
        let block_elems = self.geom.block_elems();

        // 1. Snapshot bytes BEFORE the detach.
        let mut payloads: HashMap<usize, Vec<f32>> = HashMap::new();
        for &logical in blocks {
            let Some(phys) = plane.blocks().resident_block(s, logical) else {
                continue; // already externalized — its bytes live in an earlier ticket
            };
            let mut buf = Vec::with_capacity(n_layers * 2 * block_elems);
            for layer in 0..n_layers {
                buf.extend_from_slice(&plane.read_block_f32(layer, BlockKind::K, phys)?);
                buf.extend_from_slice(&plane.read_block_f32(layer, BlockKind::V, phys)?);
            }
            payloads.insert(logical, buf);
        }

        // 2. Fuel executes the detach (pure core — no bytes involved).
        let report = plane.blocks_mut().evict_blocks(s, blocks)?;

        // 3. Persist FREED blocks only. `still_shared` blocks were never
        //    detached: they are still resident, still being written by the
        //    sharer, and a stale disk copy would clobber it on promote.
        //
        // **Everything below runs over an already-detached session, so nothing
        // here may use `?`.** `evict_blocks` has returned the blocks to the free
        // list and `report.handle` is the *only* thing that can bring them back.
        // Propagating an error would drop the handle with the rest of `report`,
        // leaving the KV unrecoverable: blocks gone, bytes possibly not on disk,
        // and no second handle anywhere. Failures therefore roll the eviction
        // back instead of escaping over it.
        let mut keys: Vec<(usize, String)> = Vec::with_capacity(report.freed.len());
        let mut failure: Option<PolicyError> = None;
        for &logical in &report.freed {
            let Some(data) = payloads.get(&logical) else {
                failure = Some(PolicyError::Invariant(format!(
                    "Fuel freed block {logical} but no payload was captured for it"
                )));
                break;
            };
            let key = self.key_for(logical);
            match store.store(&key, &f32s_to_le_bytes(data)) {
                Ok(()) => keys.push((logical, key)),
                Err(e) => {
                    failure = Some(PolicyError::Store(e));
                    break;
                }
            }
        }

        if let Some(cause) = failure {
            return Err(self.roll_back_demotion(
                plane,
                store,
                s,
                report.handle,
                &report.freed,
                &payloads,
                &keys,
                cause,
            ));
        }

        Ok(DemotionTicket {
            handle: report.handle,
            freed: report.freed,
            still_shared: report.still_shared,
            keys,
        })
    }

    /// Undo a demotion that failed after the detach: put the blocks back and
    /// rewrite their bytes from the **in-memory** payloads captured in step 1.
    ///
    /// Deliberately not from disk — an unreliable store is the usual reason to
    /// be here, so its copy is exactly what cannot be trusted. Any partial disk
    /// state is dropped first so a retry cannot find a half-written key.
    ///
    /// Returns the error the caller should see: the original cause when the
    /// rollback succeeded, or a compound error naming both failures when it did
    /// not — because at that point the KV really is gone, and reporting only
    /// "store failed" would describe a recoverable situation that no longer
    /// exists.
    #[allow(clippy::too_many_arguments)]
    fn roll_back_demotion<P: BlockPlane + ?Sized>(
        &self,
        plane: &mut P,
        store: &mut dyn DiskStore,
        s: SessionHandle,
        handle: Externalized,
        freed: &[usize],
        payloads: &HashMap<usize, Vec<f32>>,
        stored: &[(usize, String)],
        cause: PolicyError,
    ) -> PolicyError {
        for (_, key) in stored {
            let _ = store.delete(key);
        }
        if let Err(e) = plane.blocks_mut().restore(s, handle) {
            return PolicyError::Invariant(format!(
                "demote failed ({cause}) AND the rollback restore failed ({e:?}); the \
                 detached blocks are unrecoverable"
            ));
        }
        let n_layers = self.geom.n_layers();
        let block_elems = self.geom.block_elems();
        for &logical in freed {
            let Some(data) = payloads.get(&logical) else { continue };
            let Some(phys) = plane.blocks().resident_block(s, logical) else {
                return PolicyError::Invariant(format!(
                    "demote failed ({cause}) AND block {logical} is not resident after the \
                     rollback restore; its KV is lost"
                ));
            };
            for layer in 0..n_layers {
                let k0 = layer * 2 * block_elems;
                let v0 = k0 + block_elems;
                if let Err(e) =
                    plane.write_block_f32(layer, BlockKind::K, phys, &data[k0..k0 + block_elems])
                {
                    return PolicyError::Invariant(format!(
                        "demote failed ({cause}) AND rewriting block {logical} K during \
                         rollback failed ({e}); its KV is lost"
                    ));
                }
                if let Err(e) =
                    plane.write_block_f32(layer, BlockKind::V, phys, &data[v0..v0 + block_elems])
                {
                    return PolicyError::Invariant(format!(
                        "demote failed ({cause}) AND rewriting block {logical} V during \
                         rollback failed ({e}); its KV is lost"
                    ));
                }
            }
        }
        cause
    }

    /// Promote a demoted set back into the pool.
    ///
    /// Sequence, which is mandatory:
    /// 1. `core.restore(handle)` — allocates **fresh** physical blocks;
    /// 2. `resident_block` for the **new** physical ids;
    /// 3. `write_block` the bytes back;
    /// 4. drop the disk keys.
    ///
    /// Restoring first is not an ordering preference: the physical id a block had
    /// before the demotion may now belong to another session.
    pub fn promote<P: BlockPlane + ?Sized>(
        &self,
        plane: &mut P,
        store: &mut dyn DiskStore,
        s: SessionHandle,
        ticket: DemotionTicket,
    ) -> Result<Vec<usize>, PolicyError> {
        let n_layers = self.geom.n_layers();
        let block_elems = self.geom.block_elems();

        // 1. Fuel reallocates.
        plane.blocks_mut().restore(s, ticket.handle)?;

        let mut restored = Vec::with_capacity(ticket.keys.len());
        for (logical, key) in &ticket.keys {
            // 2. NEW physical id, queried AFTER the restore.
            let phys = plane.blocks().resident_block(s, *logical).ok_or_else(|| {
                PolicyError::Invariant(format!(
                    "block {logical} is not resident after restore; the handle did not \
                     cover it"
                ))
            })?;
            let bytes = store.load(key).map_err(PolicyError::Store)?;
            let data = le_bytes_to_f32s(&bytes);
            let want = n_layers * 2 * block_elems;
            if data.len() != want {
                return Err(PolicyError::Invariant(format!(
                    "stored payload for block {logical} has {} f32, expected {want}",
                    data.len()
                )));
            }
            // 3. Bytes back into the NEW block.
            for layer in 0..n_layers {
                let k0 = layer * 2 * block_elems;
                let v0 = k0 + block_elems;
                plane.write_block_f32(layer, BlockKind::K, phys, &data[k0..k0 + block_elems])?;
                plane.write_block_f32(layer, BlockKind::V, phys, &data[v0..v0 + block_elems])?;
            }
            restored.push(*logical);
        }

        // 4.
        for (_, key) in &ticket.keys {
            store.delete(key).map_err(PolicyError::Store)?;
        }
        Ok(restored)
    }
}

fn f32s_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cache_span::{CacheSpan, CacheTag};
    use crate::cache::h2o_policy::H2OConfig;

    fn geom(num_blocks: usize, block_size: usize) -> PagedGeometry {
        PagedGeometry::from_kv(KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: 1,
            head_dim: 2,
            elem_size: 4,
        })
    }

    fn pool(num_blocks: usize, block_size: usize) -> KvBlockPool {
        KvBlockPool::new(geom(num_blocks, block_size).kv())
    }

    /// In-memory `DiskStore`, so a test can corrupt a stored payload.
    #[derive(Default)]
    struct MemStore {
        map: HashMap<String, Vec<u8>>,
    }
    impl DiskStore for MemStore {
        fn store(&mut self, key: &str, data: &[u8]) -> Result<(), String> {
            self.map.insert(key.to_string(), data.to_vec());
            Ok(())
        }
        fn load(&self, key: &str) -> Result<Vec<u8>, String> {
            self.map.get(key).cloned().ok_or_else(|| format!("missing {key}"))
        }
        fn delete(&mut self, key: &str) -> Result<(), String> {
            self.map.remove(key);
            Ok(())
        }
    }

    /// Host-side stand-in for `DeviceKvPool`: a real `KvBlockPool` (the block
    /// plane is genuine Fuel) plus a `HashMap` byte plane. Exercises the mover's
    /// ORDERING and `still_shared` handling without a device or the checkpoint.
    /// The real `DeviceKvPool` path is covered separately by
    /// `device_pool_round_trip_survives_a_reallocating_restore`.
    struct MemPlane {
        g: KvGeometry,
        pool: KvBlockPool,
        buf: std::cell::RefCell<HashMap<(usize, bool, PhysBlockId), Vec<f32>>>,
    }
    impl MemPlane {
        fn new(g: PagedGeometry) -> Self {
            Self {
                g: g.kv(),
                pool: KvBlockPool::new(g.kv()),
                buf: std::cell::RefCell::new(HashMap::new()),
            }
        }
        fn elems(&self) -> usize {
            self.g.block_size * self.g.n_kv_heads * self.g.head_dim
        }
    }
    impl BlockPlane for MemPlane {
        fn plane_geometry(&self) -> KvGeometry {
            self.g
        }
        fn blocks(&self) -> &KvBlockPool {
            &self.pool
        }
        fn blocks_mut(&mut self) -> &mut KvBlockPool {
            &mut self.pool
        }
        fn read_block_f32(
            &self,
            layer: usize,
            kind: BlockKind,
            phys: PhysBlockId,
        ) -> Result<Vec<f32>, PolicyError> {
            Ok(self
                .buf
                .borrow()
                .get(&(layer, kind == BlockKind::K, phys))
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.elems()]))
        }
        fn write_block_f32(
            &self,
            layer: usize,
            kind: BlockKind,
            phys: PhysBlockId,
            data: &[f32],
        ) -> Result<(), PolicyError> {
            self.buf.borrow_mut().insert((layer, kind == BlockKind::K, phys), data.to_vec());
            Ok(())
        }
    }

    // -------------------------------------------------------------------
    // Coordinate conversion + CONTROL
    // -------------------------------------------------------------------

    #[test]
    fn blocks_fully_covered_excludes_straddling_blocks() {
        // bs = 4. Span [6, 14) straddles block 1 (4..8) and block 3 (12..16),
        // fully covers only block 2 (8..12).
        assert_eq!(blocks_fully_covered(&(6..14), 4), 2..3);
        // Exactly aligned span covers everything it touches.
        assert_eq!(blocks_fully_covered(&(8..16), 4), 2..4);
        // Shorter than a block: frees nothing. This is the documented loss.
        assert_eq!(blocks_fully_covered(&(5..7), 4), 0..0);
        assert_eq!(blocks_fully_covered(&(0..0), 4), 0..0);
    }

    #[test]
    fn blocks_touched_includes_straddling_blocks() {
        assert_eq!(blocks_touched(&(6..14), 4), 1..4);
        assert_eq!(blocks_touched(&(8..16), 4), 2..4);
        assert_eq!(blocks_touched(&(5..7), 4), 1..2);
    }

    /// CONTROL. Two adjacent spans share block 1 (tokens 4..8): span A is 0..6,
    /// span B is 6..12. The conservative mapping must EXCLUDE the boundary block
    /// from A's evictable set; the liberal mapping must INCLUDE it. If the two
    /// mappings agreed, this test could not tell a correct implementation from
    /// one that evicts a neighbour's tokens — so it asserts they differ, and
    /// asserts the boundary block really does hold B's tokens.
    #[test]
    fn control_boundary_block_separates_conservative_from_liberal() {
        let bs = 4;
        let a = 0..6;
        let b = 6..12;

        let conservative: BTreeSet<usize> = blocks_fully_covered(&a, bs).collect();
        let liberal: BTreeSet<usize> = blocks_touched(&a, bs).collect();

        assert_eq!(conservative, BTreeSet::from([0]), "only block 0 is entirely inside A");
        assert_eq!(liberal, BTreeSet::from([0, 1]), "A touches block 1 as well");
        assert_ne!(
            conservative, liberal,
            "if these agreed the test would not distinguish the two mappings"
        );

        // The block the liberal mapping would wrongly evict really does hold
        // survivor B's tokens (6 and 7 live in block 1 = tokens 4..8).
        let boundary = 1usize;
        let g = geom(8, bs);
        let tr = g.token_range_of_block(boundary);
        assert!(
            b.start < tr.end && b.end > tr.start,
            "block {boundary} ({tr:?}) must overlap survivor B {b:?} for this control to bite"
        );
        assert!(!conservative.contains(&boundary));
        assert!(liberal.contains(&boundary));
    }

    #[test]
    fn plan_never_evicts_a_block_a_survivor_touches() {
        let g = geom(16, 4);
        let map = SpanBlockMap::new(g);
        let mut reg = SpanRegistry::new();
        // A: tokens 0..6 (victim). B: tokens 6..12 (survivor). Block 1 straddles.
        reg.register(CacheSpan::new(1, 0, 0, 6, CacheTag::UserInput, Some("a".into()))).unwrap();
        reg.register(CacheSpan::new(2, 0, 6, 12, CacheTag::UserInput, Some("b".into()))).unwrap();

        let plan = map.plan(&reg, 0, &[(1, 0.1)], 4, 3, 12);
        assert_eq!(plan.victim_spans, vec![1]);
        assert_eq!(plan.blocks, vec![0], "block 1 is shared with survivor B, block 2 is B's");
        assert_eq!(plan.protected_tail, Some(2), "token 11 lives in block 2");
    }

    /// **B-4.** Real coverage for the survivor-protection subtraction, which
    /// had none.
    ///
    /// `plan_never_evicts_a_block_a_survivor_touches` never reaches it. With
    /// victim `0..6` and survivor `6..12` at bs=4, `blocks_fully_covered` yields
    /// `{0}` and `blocks_touched` yields `{1,2}` — **disjoint**, so
    /// `candidate.remove` removes nothing and that test passes identically with
    /// the loop deleted. Confirmed by deleting it: all 30 tests stayed green.
    /// The name asserts the property; the body exercises the case where
    /// `blocks_fully_covered`'s own conservatism already handles it.
    ///
    /// The loop only bites when a block is **both** fully covered by a victim
    /// and touched by a survivor, which requires the spans to *overlap* rather
    /// than merely abut. Victim `0..8` and survivor `6..12` overlap on tokens
    /// 6 and 7, which live in block 1 — a block lying entirely inside the
    /// victim.
    #[test]
    fn survivor_protection_actually_removes_a_fully_covered_block() {
        let bs = 4;
        let g = geom(16, bs);
        let map = SpanBlockMap::new(g);
        let mut reg = SpanRegistry::new();
        // A (victim) 0..8 fully covers blocks 0 and 1.
        // B (survivor) 6..12 overlaps A on tokens 6,7 — inside block 1.
        reg.register(CacheSpan::new(1, 0, 0, 8, CacheTag::UserInput, Some("a".into()))).unwrap();
        reg.register(CacheSpan::new(2, 0, 6, 12, CacheTag::UserInput, Some("b".into()))).unwrap();

        // The setup must genuinely create the conflict, or this proves nothing.
        let contested = 1usize;
        assert!(
            blocks_fully_covered(&(0..8), bs).contains(&contested),
            "block {contested} must be FULLY COVERED by the victim, or the survivor \
             subtraction is not what excludes it"
        );
        assert!(
            blocks_touched(&(6..12), bs).contains(&contested),
            "block {contested} must be TOUCHED by the survivor for this test to bite"
        );

        let plan = map.plan(&reg, 0, &[(1, 0.1)], 4, 3, 12);
        assert_eq!(plan.victim_spans, vec![1]);
        assert_eq!(
            plan.blocks,
            vec![0],
            "block 1 lies entirely inside victim A but holds survivor B's tokens 6..8; \
             evicting it would silently take live tokens from a span nobody asked to \
             evict. Without the survivor subtraction the plan is [0, 1]"
        );
    }

    #[test]
    fn plan_excludes_the_protected_tail_block() {
        let g = geom(16, 4);
        let map = SpanBlockMap::new(g);
        let mut reg = SpanRegistry::new();
        // One span covering everything; blocks 0,1,2 fully covered by 0..12.
        reg.register(CacheSpan::new(1, 0, 0, 12, CacheTag::UserInput, None)).unwrap();
        let plan = map.plan(&reg, 0, &[(1, 0.0)], 8, 3, 12);
        assert_eq!(plan.protected_tail, Some(2));
        assert_eq!(plan.blocks, vec![0, 1], "block 2 holds the newest token");
    }

    #[test]
    fn plan_with_the_real_segmented_policy_type_checks_and_runs() {
        use crate::cache::segmented_cache_config::SegmentedCacheConfig;
        let g = geom(16, 4);
        let map = SpanBlockMap::new(g);
        let policy = SegmentedEvictionPolicy::new(SegmentedCacheConfig::default());
        let mut reg = SpanRegistry::new();
        reg.register(CacheSpan::new(1, 0, 0, 8, CacheTag::UserInput, None)).unwrap();
        // No scores recorded yet ⇒ nothing ranked ⇒ empty plan. The point is that
        // rank_spans_for_eviction composes with plan() without VotingAggregator.
        let plan = map.plan_with_policy(&policy, &reg, 0, 4, 2, 8);
        assert!(plan.is_empty());
        assert!(plan.victim_spans.is_empty());
    }

    // -------------------------------------------------------------------
    // Admission + CONTROL
    // -------------------------------------------------------------------

    #[test]
    fn admission_agrees_with_the_allocator() {
        let mut p = pool(4, 4);
        let s = p.open();
        let adm = PagedAdmission::new(0);

        // 4 free blocks, ask for 16 tokens = 4 blocks: admit, and append succeeds.
        let d = adm.decide(&p, 0, 16);
        assert_eq!(d, AdmissionDecision::Admit { blocks_needed: 4 });
        assert!(p.append(s, 16).is_ok());

        // Now 0 free. Asking for one more token needs 1 block: reject, and the
        // allocator agrees by returning OutOfBlocks.
        let d = adm.decide(&p, 16, 1);
        assert_eq!(d, AdmissionDecision::Reject { blocks_needed: 1, blocks_available: 0 });
        assert_eq!(p.append(s, 1), Err(KvAllocError::OutOfBlocks { need: 1, have: 0 }));
    }

    #[test]
    fn admission_reserve_is_a_lightbulb_policy_not_a_fuel_fact() {
        let p = pool(4, 4);
        let s0 = PagedAdmission::new(0);
        let s2 = PagedAdmission::new(2);
        // Same pool, same request, different verdict: the reserve is policy.
        assert!(s0.decide(&p, 0, 16).is_admit());
        assert!(!s2.decide(&p, 0, 16).is_admit());
        assert_eq!(
            s2.decide(&p, 0, 16),
            AdmissionDecision::Reject { blocks_needed: 4, blocks_available: 2 }
        );
    }

    /// CONTROL. A block-math error of ±1 in the admission path must be caught by
    /// the comparison against the allocator. Two paths agreeing is only evidence
    /// if a deliberately wrong path disagrees.
    #[test]
    fn control_off_by_one_block_math_is_caught() {
        let p = pool(4, 4);
        // Fuel's answer.
        let correct = p.blocks_required(0, 16);
        // A plausible consumer-side reimplementation that is wrong by one.
        let wrong = (16 / 4) + 1;
        assert_eq!(correct, 4);
        assert_ne!(correct, wrong, "the control must actually differ from the truth");

        // And the difference is verdict-changing at the boundary.
        let adm = PagedAdmission::new(0);
        let good = adm.decide(&p, 0, 16);
        let bad = adm.verdict(wrong, p.capacity());
        assert!(good.is_admit());
        assert!(!bad.is_admit(), "the off-by-one must flip a legal admit into a reject");
    }

    // -------------------------------------------------------------------
    // Prefix sharing + the mandatory donor-corruption CONTROL
    // -------------------------------------------------------------------

    #[test]
    fn prefix_index_returns_only_block_aligned_matches() {
        let mut idx = BlockPrefixIndex::new(4);
        let donor_tokens: Vec<u32> = (0..10).collect(); // 2 full blocks + 2 spare
        let mut p = pool(16, 4);
        let donor = p.open();
        p.append(donor, donor_tokens.len()).unwrap();
        assert_eq!(idx.record(&donor_tokens, donor), 2, "only whole blocks are indexed");

        // Query shares 9 tokens then diverges.
        let mut q: Vec<u32> = (0..9).collect();
        q.push(999);
        let m = idx.lookup(&q).expect("2 full blocks match");
        assert_eq!(m.blocks, 2);
        assert_eq!(m.shared_tokens, 8);
        assert_eq!(m.remainder_tokens, 2, "tokens 8 and 9 are re-prefilled");
        assert!(
            m.remainder_tokens >= 1,
            "the alignment loss is real: matched-but-unshared tokens exist"
        );
        assert_eq!(idx.stats().hits, 1);
        assert_eq!(idx.stats().tokens_shared, 8);
    }

    #[test]
    fn prefix_index_cannot_express_a_zero_block_hit() {
        let mut idx = BlockPrefixIndex::new(4);
        let mut p = pool(16, 4);
        let donor = p.open();
        // A prompt shorter than one block indexes nothing — the structural fix
        // for prefix_cache's empty-placeholder-as-a-hit hazard.
        assert_eq!(idx.record(&[1, 2, 3], donor), 0);
        assert!(idx.lookup(&[1, 2, 3]).is_none());
        assert_eq!(idx.stats().misses, 1);
        assert_eq!(idx.stats().hits, 0);
    }

    #[test]
    fn aligned_splice_leaves_the_next_write_block_exclusive() {
        let bs = 4;
        let mut p = pool(16, bs);
        let donor = p.open();
        p.append(donor, 12).unwrap(); // 3 blocks

        let dst = p.open();
        let m = PrefixMatch { donor, blocks: 2, shared_tokens: 8, remainder_tokens: 0 };
        let filled = splice_prefix(&mut p, &m, dst).unwrap();
        assert_eq!(filled, 8);
        assert_eq!(filled % bs, 0, "the invariant splice_prefix enforces");

        // The next token starts a FRESH block: append allocates, refcount 1.
        p.append(dst, 1).unwrap();
        let write_idx = 8 / bs; // forward_paged_step: tok_pos / block_size
        let phys = p.resident_block(dst, write_idx).unwrap();
        assert_eq!(p.block_refcount(phys), 1, "the write target must be exclusive");
        assert_ne!(
            Some(phys),
            p.resident_block(donor, write_idx),
            "and must not be the donor's block"
        );
    }

    /// CONTROL — the mandatory one. Proves the corruption the floor-alignment
    /// rule prevents is REAL, at the allocator level, without needing the model.
    ///
    /// With a NON-aligned prefix (6 tokens, bs=4), `splice` leaves
    /// `dst.filled_tokens = 6`. `forward_paged_step` then computes
    /// `tok_pos = 6`, `append(1)` allocates NO new block (6→7 stays inside block
    /// 1), and takes `resident_block(dst, 6/4 = 1)` — which is the DONOR's block,
    /// refcount 2 — and `write_slice`s the new token into it. `cow_break` is
    /// never called anywhere in fuel-core. So the donor is silently corrupted.
    ///
    /// The test asserts exactly that shared-write-target condition, and then
    /// asserts `splice_prefix` REFUSES the same situation.
    #[test]
    fn control_unaligned_splice_targets_the_donors_shared_block() {
        let bs = 4;
        let mut p = pool(16, bs);
        let donor = p.open();
        p.append(donor, 12).unwrap(); // 3 blocks, 12 tokens

        // Raw Fuel call, deliberately NON-aligned: share 2 blocks of a donor that
        // only holds 6 tokens, so dst.filled_tokens lands at 6.
        let donor6 = p.open();
        p.append(donor6, 6).unwrap(); // 2 blocks, 6 tokens (block 1 half full)
        let dst = p.open();
        p.splice(donor6, dst, 0, 2).unwrap();
        assert_eq!(p.filled_tokens(dst), Some(6));
        assert_ne!(6 % bs, 0, "the setup must actually be non-aligned");

        // Reproduce forward_paged_step's sequence.
        let tok_pos = p.filled_tokens(dst).unwrap();
        let blocks_before = p.session_blocks(dst).unwrap();
        p.append(dst, 1).unwrap();
        assert_eq!(
            p.session_blocks(dst).unwrap(),
            blocks_before,
            "append allocated nothing — the write must land in an existing block"
        );
        let target = p.resident_block(dst, tok_pos / bs).unwrap();
        assert_eq!(
            Some(target),
            p.resident_block(donor6, tok_pos / bs),
            "THE CORRUPTION: the write target IS the donor's block"
        );
        assert!(p.block_refcount(target) > 1, "and it is still shared");

        // Now the guard: splice_prefix refuses to create that state.
        let dst2 = p.open();
        let bad = PrefixMatch { donor: donor6, blocks: 2, shared_tokens: 8, remainder_tokens: 0 };
        let err = splice_prefix(&mut p, &bad, dst2).unwrap_err();
        assert!(
            matches!(err, PolicyError::Invariant(ref m) if m.contains("not block-aligned")),
            "expected the alignment invariant to fire, got {err}"
        );

        // And the aligned form of the same donor is accepted.
        let dst3 = p.open();
        let good = PrefixMatch { donor, blocks: 1, shared_tokens: 4, remainder_tokens: 0 };
        assert_eq!(splice_prefix(&mut p, &good, dst3).unwrap(), 4);
    }

    /// **B-1.** A refusal must leave nothing behind. The control test above
    /// asserts the alignment guard *fires*; this asserts the guard is
    /// **transactional**, which is a different claim and the one that was false.
    ///
    /// The guard used to run after `pool.splice`, so on the error path `dst` had
    /// already received the shared slots and the donor's refcounts had already
    /// been bumped. The error was correct and the damage was done: a caller that
    /// logged it and carried on — or retried into the same `dst` — decoded into
    /// the donor's block anyway, which is exactly the corruption the guard
    /// exists to prevent. There is no unsplice, so the check has to precede the
    /// mutation rather than report on it.
    ///
    /// Fails on the pre-fix code at the very first assertion (`dst` holds 6
    /// tokens, not 0).
    #[test]
    fn refused_splice_leaves_the_pool_completely_untouched() {
        let bs = 4;
        let mut p = pool(16, bs);
        let donor = p.open();
        p.append(donor, 6).unwrap(); // 2 blocks, 6 tokens — block 1 half full

        // Snapshot every piece of state the splice would have moved.
        let donor_blocks = p.session_blocks(donor).unwrap();
        let donor_filled = p.filled_tokens(donor).unwrap();
        let free_before = p.free_blocks();
        let refs_before: Vec<u32> = (0..donor_blocks)
            .map(|i| p.block_refcount(p.resident_block(donor, i).unwrap()))
            .collect();

        let dst = p.open();
        let bad = PrefixMatch { donor, blocks: 2, shared_tokens: 8, remainder_tokens: 0 };
        let err = splice_prefix(&mut p, &bad, dst).unwrap_err();
        assert!(
            matches!(err, PolicyError::Invariant(ref m) if m.contains("not block-aligned")),
            "expected the alignment invariant, got {err}"
        );

        assert_eq!(
            p.filled_tokens(dst),
            Some(0),
            "dst must hold NOTHING after a refusal — the pre-fix guard spliced first \
             and reported afterwards, leaving dst holding the donor's tokens"
        );
        assert_eq!(p.session_blocks(dst), Some(0), "dst must own no blocks after a refusal");
        for (i, before) in refs_before.iter().enumerate() {
            let phys = p.resident_block(donor, i).unwrap();
            assert_eq!(
                p.block_refcount(phys), *before,
                "donor block {i} refcount changed on a refused splice — the donor is \
                 now pinned by a session that holds nothing"
            );
        }
        assert_eq!(p.free_blocks(), free_before, "free list moved on a refused splice");
        assert_eq!(p.session_blocks(donor), Some(donor_blocks), "donor block count changed");
        assert_eq!(p.filled_tokens(donor), Some(donor_filled), "donor fill changed");

        // And the pool is still usable for the aligned case afterwards — a
        // refusal must not poison the donor for a later legitimate splice.
        let dst2 = p.open();
        let good = PrefixMatch { donor, blocks: 1, shared_tokens: 4, remainder_tokens: 0 };
        assert_eq!(splice_prefix(&mut p, &good, dst2).unwrap(), 4);
    }

    #[test]
    fn splice_prefix_refuses_a_non_empty_destination() {
        let mut p = pool(16, 4);
        let donor = p.open();
        p.append(donor, 8).unwrap();
        let dst = p.open();
        p.append(dst, 4).unwrap();
        let m = PrefixMatch { donor, blocks: 2, shared_tokens: 8, remainder_tokens: 0 };
        assert!(matches!(splice_prefix(&mut p, &m, dst), Err(PolicyError::Invariant(_))));
    }

    #[test]
    fn donor_lifetime_is_tracked() {
        let mut idx = BlockPrefixIndex::new(4);
        let mut p = pool(16, 4);
        let donor = p.open();
        p.append(donor, 8).unwrap();
        let toks: Vec<u32> = (0..8).collect();
        idx.record(&toks, donor);
        assert!(idx.is_donor(donor), "must not be discarded while indexed");
        assert_eq!(idx.donors().len(), 1);
        assert_eq!(idx.forget_donor(donor), 2);
        assert!(!idx.is_donor(donor));
        assert!(idx.lookup(&toks).is_none());
    }

    // -------------------------------------------------------------------
    // EvictReport reconciliation + CONTROL
    // -------------------------------------------------------------------

    fn wrong_apply_treating_shared_as_freed(
        g: &PagedGeometry,
        reg: &SpanRegistry,
        plan: &EvictionPlan,
        report: &EvictReport,
    ) -> Vec<(SpanId, EvictionImpact)> {
        // The deliberately-wrong implementation: subtract still_shared too.
        let mut cuts: Vec<Range<usize>> =
            report.freed.iter().map(|&b| g.token_range_of_block(b)).collect();
        cuts.extend(report.still_shared.iter().map(|&b| g.token_range_of_block(b)));
        plan.victim_spans
            .iter()
            .filter_map(|&id| {
                let s = reg.get(id)?;
                let rem = subtract_ranges(s.start_pos..s.end_pos, &cuts);
                Some((
                    id,
                    if rem.is_empty() {
                        EvictionImpact::FullyEvicted
                    } else {
                        EvictionImpact::PartiallyEvicted { remaining: rem }
                    },
                ))
            })
            .collect()
    }

    /// CONTROL. A span whose blocks are all shared with a spliced sitter must come
    /// back `still_shared` and be recorded as UNCHANGED — the blocks are still
    /// resident. The wrong implementation (treating `still_shared` as gone) is run
    /// side by side and asserted to produce `FullyEvicted`, proving the assertion
    /// distinguishes the two.
    #[test]
    fn control_still_shared_blocks_must_not_be_recorded_as_evicted() {
        let bs = 4;
        let g = geom(16, bs);
        let map = SpanBlockMap::new(g);
        let mut p = pool(16, bs);

        let donor = p.open();
        p.append(donor, 8).unwrap(); // blocks 0,1
        let sitter = p.open();
        p.splice(donor, sitter, 0, 2).unwrap(); // both blocks now refcount 2

        let mut reg = SpanRegistry::new();
        reg.register(CacheSpan::new(1, 0, 0, 8, CacheTag::UserInput, None)).unwrap();
        let plan = EvictionPlan {
            blocks: vec![0, 1],
            victim_spans: vec![1],
            protected_tail: None,
        };

        let report = p.evict_blocks(donor, &plan.blocks).unwrap();
        assert!(report.freed.is_empty(), "both blocks are shared, so nothing is detachable");
        assert_eq!(report.still_shared, vec![0, 1]);

        // Wrong implementation first, so we know the assertion can fail.
        let wrong = wrong_apply_treating_shared_as_freed(&g, &reg, &plan, &report);
        assert_eq!(
            wrong,
            vec![(1, EvictionImpact::FullyEvicted)],
            "the control must actually produce the wrong answer"
        );

        let outcome = map.apply_report(&mut reg, &plan, &report);
        assert_eq!(
            outcome.impacts,
            vec![(1, EvictionImpact::Unchanged)],
            "still_shared blocks are STILL RESIDENT — the span is not demoted"
        );
        assert_ne!(outcome.impacts, wrong, "correct and wrong must differ, or this proves nothing");
        assert_eq!(outcome.blocks_still_resident, vec![0, 1]);
        assert!(outcome.blocks_freed.is_empty());
        assert_eq!(reg.get(1).unwrap().state, SpanState::Active);

        // And the pool agrees the blocks were never released.
        assert_eq!(p.free_blocks(), 16 - 2);
    }

    #[test]
    fn mixed_report_yields_partially_evicted_with_the_surviving_ranges() {
        let bs = 4;
        let g = geom(16, bs);
        let map = SpanBlockMap::new(g);
        let mut p = pool(16, bs);

        let donor = p.open();
        p.append(donor, 12).unwrap(); // blocks 0,1,2
        let sitter = p.open();
        p.splice(donor, sitter, 0, 1).unwrap(); // block 0 shared

        let mut reg = SpanRegistry::new();
        reg.register(CacheSpan::new(1, 0, 0, 12, CacheTag::UserInput, None)).unwrap();
        let plan =
            EvictionPlan { blocks: vec![0, 1], victim_spans: vec![1], protected_tail: Some(2) };

        let report = p.evict_blocks(donor, &plan.blocks).unwrap();
        assert_eq!(report.freed, vec![1]);
        assert_eq!(report.still_shared, vec![0]);

        let outcome = map.apply_report(&mut reg, &plan, &report);
        // Only block 1 (tokens 4..8) is gone; 0..4 (shared, resident) and 8..12
        // (never asked for) survive.
        assert_eq!(
            outcome.impacts,
            vec![(
                1,
                EvictionImpact::PartiallyEvicted { remaining: vec![0..4, 8..12] }
            )]
        );
        assert_eq!(
            reg.get(1).unwrap().state,
            SpanState::PartiallyEvicted { remaining_ranges: vec![0..4, 8..12] }
        );
    }

    /// **B-2.** Eviction runs over as many rounds as the scheduler needs, so
    /// `apply_report` has to *fold* onto prior state rather than recompute from
    /// the span's original extent.
    ///
    /// It used to take `start_pos..end_pos` and subtract only the current
    /// report's cuts, discarding everything earlier rounds had already taken.
    /// Two rounds on one span therefore resurrected the first round's losses —
    /// the span was reported as holding tokens Fuel had already returned to the
    /// free list, so Lightbulb would skip re-prefilling them and decode against
    /// KV that is no longer there.
    ///
    /// On the pre-fix code the round-2 assertion fails with `[0..4, 8..12]` —
    /// `0..4` back from the dead.
    #[test]
    fn apply_report_folds_across_eviction_rounds() {
        let bs = 4;
        let g = geom(16, bs);
        let map = SpanBlockMap::new(g);
        let mut p = pool(16, bs);
        let s = p.open();
        p.append(s, 12).unwrap(); // blocks 0,1,2 → tokens 0..12

        let mut reg = SpanRegistry::new();
        reg.register(CacheSpan::new(1, 0, 0, 12, CacheTag::UserInput, None)).unwrap();

        // Round 1 — block 0 (tokens 0..4).
        let plan1 =
            EvictionPlan { blocks: vec![0], victim_spans: vec![1], protected_tail: Some(2) };
        let r1 = p.evict_blocks(s, &plan1.blocks).unwrap();
        assert_eq!(r1.freed, vec![0], "setup: block 0 must actually be freed");
        let o1 = map.apply_report(&mut reg, &plan1, &r1);
        assert_eq!(
            o1.impacts,
            vec![(1, EvictionImpact::PartiallyEvicted { remaining: vec![4..12] })]
        );

        // Round 2 — block 1 (tokens 4..8). Only 8..12 can still be live.
        let plan2 =
            EvictionPlan { blocks: vec![1], victim_spans: vec![1], protected_tail: Some(2) };
        let r2 = p.evict_blocks(s, &plan2.blocks).unwrap();
        assert_eq!(r2.freed, vec![1], "setup: block 1 must actually be freed");
        let o2 = map.apply_report(&mut reg, &plan2, &r2);
        assert_eq!(
            o2.impacts,
            vec![(1, EvictionImpact::PartiallyEvicted { remaining: vec![8..12] })],
            "round 2 must fold onto round 1's survivors — tokens 0..4 were freed in \
             round 1 and must not reappear"
        );
        assert_eq!(
            reg.get(1).unwrap().state,
            SpanState::PartiallyEvicted { remaining_ranges: vec![8..12] }
        );

        // Round 3 — the last block. Now the span really is gone.
        let plan3 = EvictionPlan { blocks: vec![2], victim_spans: vec![1], protected_tail: None };
        let r3 = p.evict_blocks(s, &plan3.blocks).unwrap();
        assert_eq!(r3.freed, vec![2], "setup: block 2 must actually be freed");
        let o3 = map.apply_report(&mut reg, &plan3, &r3);
        assert_eq!(o3.impacts, vec![(1, EvictionImpact::FullyEvicted)]);
        assert_eq!(reg.get(1).unwrap().state, SpanState::FullyEvicted);

        // Replaying a report is a no-op, not a resurrection — the scheduler may
        // retry, and a duplicate delivery must not revive a dead span.
        let o4 = map.apply_report(&mut reg, &plan3, &r3);
        assert_eq!(o4.impacts, vec![(1, EvictionImpact::FullyEvicted)]);
        assert_eq!(reg.get(1).unwrap().state, SpanState::FullyEvicted);
    }

    #[test]
    fn fully_covered_span_becomes_fully_evicted() {
        let bs = 4;
        let g = geom(16, bs);
        let map = SpanBlockMap::new(g);
        let mut p = pool(16, bs);
        let s = p.open();
        p.append(s, 12).unwrap();

        let mut reg = SpanRegistry::new();
        reg.register(CacheSpan::new(1, 0, 0, 8, CacheTag::UserInput, None)).unwrap();
        let plan =
            EvictionPlan { blocks: vec![0, 1], victim_spans: vec![1], protected_tail: Some(2) };
        let report = p.evict_blocks(s, &plan.blocks).unwrap();
        assert_eq!(report.freed, vec![0, 1]);
        let outcome = map.apply_report(&mut reg, &plan, &report);
        assert_eq!(outcome.impacts, vec![(1, EvictionImpact::FullyEvicted)]);
        assert_eq!(reg.get(1).unwrap().state, SpanState::FullyEvicted);
    }

    /// Pins the constraint this whole module is shaped around (the "F2 blocker").
    ///
    /// Fuel's `evict_blocks` doc says "the rest of the session stays resident and
    /// keeps decoding", and its own test is named
    /// `evict_range_sheds_a_span_and_leaves_the_live_session_decoding` — but that
    /// test only asserts `append` still succeeds. It does not. Attention needs
    /// `session_block_table`, which `materialize_block_table` and therefore
    /// `forward_paged_step` both go through, and it rejects a session with ANY
    /// externalized slot. So evict → decode is impossible; only
    /// evict → restore → resume works, which is why nothing here offers the
    /// former.
    #[test]
    fn evicting_a_middle_block_makes_the_session_undecodable_until_restore() {
        let mut p = pool(8, 4);
        let s = p.open();
        p.append(s, 12).unwrap(); // blocks 0, 1, 2
        assert!(p.session_block_table(s).is_ok());

        let report = p.evict_blocks(s, &[1]).unwrap();
        assert_eq!(report.freed, vec![1]);

        // `append` still works — which is all Fuel's own test checks.
        assert!(p.append(s, 1).is_ok(), "the allocator half really does keep working");

        // But the block table cannot be materialized, so the next
        // forward_paged_step would fail outright.
        assert_eq!(
            p.session_block_table(s),
            Err(KvAllocError::SessionNotResident { slot: 1 }),
            "evict-then-decode is unreachable in Fuel today"
        );

        p.restore(s, report.handle).unwrap();
        assert!(p.session_block_table(s).is_ok(), "restore is what re-enables decode");
    }

    #[test]
    fn subtract_ranges_coalesces_and_handles_full_cover() {
        assert_eq!(subtract_ranges(0..12, &[4..8]), vec![0..4, 8..12]);
        assert_eq!(subtract_ranges(0..12, &[0..4, 4..8, 8..12]), Vec::<Range<usize>>::new());
        assert_eq!(subtract_ranges(0..12, &[]), vec![0..12]);
        // Adjacent cuts leave coalesced survivors, not fragments.
        assert_eq!(subtract_ranges(0..16, &[4..8, 8..12]), vec![0..4, 12..16]);
    }

    // -------------------------------------------------------------------
    // H2O aggregation (the case VotingAggregator drops)
    // -------------------------------------------------------------------

    /// `VotingAggregator::compute_aggregated_scores` `continue`s past every
    /// `+INFINITY`, silently dropping the never-attended slots that H2O flags as
    /// evict-first. This adapter must NOT.
    #[test]
    fn h2o_block_scores_preserve_positive_infinity() {
        let g = geom(16, 4);
        let policy = H2OPolicy::new(H2OConfig {
            enabled: true,
            num_recent_to_keep: 0,
            decay_factor: 1.0,
        });
        let positions = identity_positions(8);
        // No attention was ever recorded ⇒ every token scores +INFINITY.
        let scores = h2o_block_scores(&policy, &positions, &g);
        assert_eq!(scores.len(), 2, "8 tokens at bs=4 ⇒ 2 blocks");
        for (b, s) in &scores {
            assert!(s.is_infinite() && s.is_sign_positive(), "block {b} lost its +INF score");
        }
    }

    #[test]
    fn h2o_block_score_is_the_minimum_over_the_blocks_tokens() {
        let g = geom(16, 4);
        let mut policy = H2OPolicy::new(H2OConfig {
            enabled: true,
            num_recent_to_keep: 0,
            decay_factor: 1.0,
        });
        let positions = identity_positions(8);
        // One heavily-attended token (position 5) inside block 1.
        let mut row = vec![0.0f32; 8];
        row[5] = 1.0;
        policy.update_attention_scores(&[row], &positions);

        let scores = h2o_block_scores(&policy, &positions, &g);
        let by_block: HashMap<usize, f32> = scores.into_iter().collect();
        // Block 0: every token unattended ⇒ +INF (evict first).
        assert!(by_block[&0].is_infinite());
        // Block 1: one hot token drags the whole block's score down to finite.
        assert!(
            by_block[&1].is_finite(),
            "one hot token must protect its whole block (block 1 scored {})",
            by_block[&1]
        );
        assert!(by_block[&1] < by_block[&0], "block 1 must be less evictable than block 0");
    }

    /// Each guard in `plan_from_block_scores` gets its own ineligible block, so
    /// a regression that drops one is visible rather than masked by the others.
    #[test]
    fn h2o_plan_skips_protected_tail_nan_and_out_of_range_blocks() {
        let g = geom(16, 4);
        let map = SpanBlockMap::new(g);
        // Descending, as `h2o_block_scores` returns. 6 blocks, 12 filled tokens
        // ⇒ protected tail is block 2.
        let scores = vec![
            (0, f32::INFINITY),     // never attended ⇒ evict first
            (2, 5.0),               // the protected tail — next write target
            (7, 4.0),               // at/past session_blocks
            (1, f32::NAN),          // unknown ⇒ not a decision
            (3, f32::NEG_INFINITY), // H2O-protected ⇒ never evictable
            (4, 2.0),               // eligible
        ];
        let plan = map.plan_from_block_scores(&scores, 8, 6, 12);
        assert_eq!(plan.protected_tail, Some(2));
        assert_eq!(
            plan.blocks,
            vec![0, 4],
            "block 2 is the tail, 7 is out of range, 1 is NaN, 3 is H2O-protected"
        );
        assert!(plan.victim_spans.is_empty(), "H2O ranks blocks, not spans");
    }

    /// **H2O reaches the allocator.** `h2o_block_scores` had no caller outside
    /// tests: it computed per-block scores that nothing turned into an
    /// `EvictionPlan`, and nothing handed to Fuel. This drives the whole path —
    /// attention in, detached blocks out — against a real `KvBlockPool`.
    #[test]
    fn evict_with_h2o_drives_the_allocator_end_to_end() {
        let bs = 4;
        let g = geom(16, bs);
        let map = SpanBlockMap::new(g);
        let mut p = pool(16, bs);
        let s = p.open();
        p.append(s, 12).unwrap(); // blocks 0,1,2

        let mut policy =
            H2OPolicy::new(H2OConfig { enabled: true, num_recent_to_keep: 0, decay_factor: 1.0 });
        let positions = identity_positions(12);
        // One heavily-attended token inside block 1.
        let mut row = vec![0.0f32; 12];
        row[5] = 1.0;
        policy.update_attention_scores(&[row], &positions);

        let free_before = p.free_blocks();
        let (plan, report) = map.evict_with_h2o(&mut p, &policy, &positions, s, 1).unwrap();

        assert_eq!(plan.protected_tail, Some(2), "token 11 lives in block 2");
        assert_eq!(
            plan.blocks,
            vec![0],
            "block 0 is unattended, block 1 is protected by its hot token, block 2 is \
             the tail"
        );

        // Fuel actually executed it — not just a plan that was computed and dropped.
        assert_eq!(report.freed, vec![0]);
        assert_eq!(p.free_blocks(), free_before + 1, "the block returned to the free list");
        assert!(p.resident_block(s, 0).is_none(), "block 0 is detached");
        assert!(p.resident_block(s, 1).is_some(), "the hot block survives");
        assert!(p.resident_block(s, 2).is_some(), "the tail survives");
    }

    // -------------------------------------------------------------------
    // Tiering + CONTROL
    // -------------------------------------------------------------------

    fn fill_io(plane: &MemPlane, g: &PagedGeometry, phys: PhysBlockId, seed: f32) {
        for layer in 0..g.n_layers() {
            let k: Vec<f32> = (0..g.block_elems()).map(|i| seed + i as f32).collect();
            let v: Vec<f32> = (0..g.block_elems()).map(|i| seed + 100.0 + i as f32).collect();
            plane.write_block_f32(layer, BlockKind::K, phys, &k).unwrap();
            plane.write_block_f32(layer, BlockKind::V, phys, &v).unwrap();
        }
    }

    /// A `DiskStore` that accepts `ok_before` writes and then fails, so a
    /// demotion can be interrupted with real bytes already persisted.
    struct FlakyStore {
        ok_before: usize,
        writes: usize,
        map: HashMap<String, Vec<u8>>,
        deleted: Vec<String>,
    }
    impl FlakyStore {
        fn new(ok_before: usize) -> Self {
            Self { ok_before, writes: 0, map: HashMap::new(), deleted: Vec::new() }
        }
    }
    impl DiskStore for FlakyStore {
        fn store(&mut self, key: &str, data: &[u8]) -> Result<(), String> {
            if self.writes >= self.ok_before {
                return Err("disk full".to_string());
            }
            self.writes += 1;
            self.map.insert(key.to_string(), data.to_vec());
            Ok(())
        }
        fn load(&self, key: &str) -> Result<Vec<u8>, String> {
            self.map.get(key).cloned().ok_or_else(|| format!("missing {key}"))
        }
        fn delete(&mut self, key: &str) -> Result<(), String> {
            self.map.remove(key);
            self.deleted.push(key.to_string());
            Ok(())
        }
    }

    /// **B-3.** A demotion that fails after the detach must not take the restore
    /// handle with it.
    ///
    /// `evict_blocks` returns the blocks to the free list and hands back the one
    /// `Externalized` handle that can undo that. The store step runs afterwards
    /// and used to propagate with `?`, dropping `report` — and the handle inside
    /// it. The blocks were then off the session, their bytes were *not* on disk
    /// (that is precisely what had just failed), and nothing anywhere could
    /// bring them back. An unrecoverable session, reported as a routine store
    /// error.
    ///
    /// Two blocks with a store that fails on the second, so the rollback has to
    /// undo a genuinely partial demotion: one payload already written, one not.
    ///
    /// On the pre-fix code this fails at the residency assertion.
    #[test]
    fn a_failed_store_rolls_the_demotion_back_instead_of_orphaning_the_handle() {
        let bs = 2;
        let g = geom(4, bs);
        let mut plane = MemPlane::new(g);
        let mut store = FlakyStore::new(1); // block 0 persists, block 1 fails
        let mover = BlockTierMover::new(g, "sess-rollback");

        let s = plane.blocks_mut().open();
        plane.blocks_mut().append(s, 4).unwrap(); // blocks 0 and 1
        let phys0 = plane.blocks().resident_block(s, 0).unwrap();
        let phys1 = plane.blocks().resident_block(s, 1).unwrap();
        fill_io(&plane, &g, phys0, 3.0);
        fill_io(&plane, &g, phys1, 7.0);
        let want0_k = plane.read_block_f32(0, BlockKind::K, phys0).unwrap();
        let want1_k = plane.read_block_f32(0, BlockKind::K, phys1).unwrap();
        let want1_v = plane.read_block_f32(0, BlockKind::V, phys1).unwrap();
        let free_before = plane.blocks().free_blocks();
        let blocks_before = plane.blocks().session_blocks(s).unwrap();

        let err = mover.demote(&mut plane, &mut store, s, &[0, 1]).unwrap_err();
        assert!(
            matches!(err, PolicyError::Store(ref m) if m.contains("disk full")),
            "the ORIGINAL cause must survive the rollback rather than being replaced \
             by a rollback message, got {err}"
        );

        // Both blocks are back on the session.
        let after0 = plane.blocks().resident_block(s, 0).expect(
            "block 0 must be resident after a rolled-back demotion — the pre-fix code \
             dropped the Externalized handle, detaching it permanently",
        );
        let after1 = plane.blocks().resident_block(s, 1).expect("block 1 must be resident");
        assert_eq!(plane.blocks().session_blocks(s), Some(blocks_before));
        assert_eq!(plane.blocks().free_blocks(), free_before, "free list must be restored");

        // And their bytes are back, at whatever physical ids the restore chose.
        assert_eq!(plane.read_block_f32(0, BlockKind::K, after0).unwrap(), want0_k);
        assert_eq!(plane.read_block_f32(0, BlockKind::K, after1).unwrap(), want1_k);
        assert_eq!(plane.read_block_f32(0, BlockKind::V, after1).unwrap(), want1_v);

        // The half-written disk state is gone — a retry must not find a stale key
        // for block 0 while block 1 has none.
        assert!(
            store.map.is_empty(),
            "the partial payload for block 0 survived the rollback: {:?}",
            store.map.keys().collect::<Vec<_>>()
        );
        assert_eq!(store.deleted.len(), 1, "exactly the one persisted key is deleted");

        // The pool is not poisoned: a real demote/promote cycle still works.
        let mut good = MemStore::default();
        let ticket = mover.demote(&mut plane, &mut good, s, &[1]).unwrap();
        assert_eq!(ticket.freed, vec![1]);
        assert_eq!(mover.promote(&mut plane, &mut good, s, ticket).unwrap(), vec![1]);
        let final1 = plane.blocks().resident_block(s, 1).unwrap();
        assert_eq!(plane.read_block_f32(0, BlockKind::K, final1).unwrap(), want1_k);
    }

    #[test]
    fn tier_round_trip_is_byte_exact_across_a_reallocating_restore() {
        let bs = 2;
        let g = geom(4, bs);
        let mut plane = MemPlane::new(g);
        let mut store = MemStore::default();
        let mover = BlockTierMover::new(g, "sess-a");

        let s = plane.blocks_mut().open();
        plane.blocks_mut().append(s, 4).unwrap(); // 2 blocks
        let phys_before = plane.blocks().resident_block(s, 1).unwrap();
        fill_io(&plane, &g, phys_before, 7.0);
        let expected_k = plane.read_block_f32(0, BlockKind::K, phys_before).unwrap();
        let expected_v = plane.read_block_f32(0, BlockKind::V, phys_before).unwrap();

        let ticket = mover.demote(&mut plane, &mut store, s, &[1]).unwrap();
        assert_eq!(ticket.freed, vec![1]);
        assert!(ticket.still_shared.is_empty());
        assert_eq!(ticket.bytes_stored_for(), vec![1]);

        // Steal the freed physical block, and poison it, so a promote that reused
        // the OLD physical id would visibly read the wrong bytes.
        let decoy = plane.blocks_mut().open();
        plane.blocks_mut().append(decoy, 2).unwrap();
        let stolen = plane.blocks().resident_block(decoy, 0).unwrap();
        assert_eq!(stolen, phys_before, "the decoy must take the block we just freed");
        fill_io(&plane, &g, stolen, -999.0);

        mover.promote(&mut plane, &mut store, s, ticket).unwrap();
        let phys_after = plane.blocks().resident_block(s, 1).unwrap();
        assert_ne!(
            phys_after, phys_before,
            "restore must have allocated a FRESH block — this is why promote queries \
             resident_block AFTER restore"
        );
        assert_eq!(plane.read_block_f32(0, BlockKind::K, phys_after).unwrap(), expected_k);
        assert_eq!(plane.read_block_f32(0, BlockKind::V, phys_after).unwrap(), expected_v);
    }

    /// CONTROL. Flip one f32 in the stored payload and assert the round-trip
    /// comparison FAILS. Without this, "the bytes matched" only proves the
    /// comparison ran, not that it discriminates.
    #[test]
    fn control_corrupted_payload_fails_the_round_trip_comparison() {
        let bs = 2;
        let g = geom(4, bs);
        let mut plane = MemPlane::new(g);
        let mut store = MemStore::default();
        let mover = BlockTierMover::new(g, "sess-b");

        let s = plane.blocks_mut().open();
        plane.blocks_mut().append(s, 4).unwrap();
        let phys_before = plane.blocks().resident_block(s, 1).unwrap();
        fill_io(&plane, &g, phys_before, 7.0);
        let expected_k = plane.read_block_f32(0, BlockKind::K, phys_before).unwrap();

        let ticket = mover.demote(&mut plane, &mut store, s, &[1]).unwrap();
        let key = ticket.keys[0].1.clone();

        // Corrupt exactly one f32 in the stored bytes.
        let mut bytes = store.load(&key).unwrap();
        let mut vals = le_bytes_to_f32s(&bytes);
        let original = vals[0];
        vals[0] += 1.0;
        bytes = f32s_to_le_bytes(&vals);
        store.store(&key, &bytes).unwrap();

        mover.promote(&mut plane, &mut store, s, ticket).unwrap();
        let phys_after = plane.blocks().resident_block(s, 1).unwrap();
        let got_k = plane.read_block_f32(0, BlockKind::K, phys_after).unwrap();

        assert_ne!(got_k, expected_k, "a corrupted payload MUST fail the comparison");
        assert_eq!(got_k[0], original + 1.0, "and it must fail in exactly the corrupted slot");
        assert_eq!(got_k[1..], expected_k[1..], "the rest must still round-trip");
    }

    /// `still_shared` blocks must never reach the store: they were never
    /// detached, so a stale disk copy would clobber the sharer on promote.
    #[test]
    fn demote_stores_freed_blocks_only_never_still_shared() {
        let bs = 2;
        let g = geom(8, bs);
        let mut plane = MemPlane::new(g);
        let mut store = MemStore::default();
        let mover = BlockTierMover::new(g, "sess-c");

        let s = plane.blocks_mut().open();
        plane.blocks_mut().append(s, 6).unwrap(); // 3 blocks
        let sitter = plane.blocks_mut().open();
        plane.blocks_mut().splice(s, sitter, 0, 1).unwrap(); // block 0 shared

        let ticket = mover.demote(&mut plane, &mut store, s, &[0, 1]).unwrap();
        assert_eq!(ticket.freed, vec![1]);
        assert_eq!(ticket.still_shared, vec![0]);
        assert_eq!(ticket.bytes_stored_for(), vec![1], "block 0 must NOT be persisted");
        assert_eq!(store.map.len(), 1);
        assert!(!store.map.contains_key(&mover.key_for(0)));

        // Block 0 is still resident and still backing both sessions.
        let phys0 = plane.blocks().resident_block(s, 0).expect("block 0 was never detached");
        assert_eq!(plane.blocks().resident_block(sitter, 0), Some(phys0));
        assert!(plane.blocks().block_refcount(phys0) > 1);
    }

    #[test]
    fn file_disk_store_composes_with_the_mover() {
        use crate::cache::tiered_storage::FileDiskStore;
        let dir = tempfile::tempdir().unwrap();
        let bs = 2;
        let g = geom(4, bs);
        let mut plane = MemPlane::new(g);
        let mut store = FileDiskStore::new(dir.path()).unwrap();
        let mover = BlockTierMover::new(g, "sess-file");

        let s = plane.blocks_mut().open();
        plane.blocks_mut().append(s, 4).unwrap();
        let phys_before = plane.blocks().resident_block(s, 1).unwrap();
        fill_io(&plane, &g, phys_before, 3.5);
        let expected = plane.read_block_f32(0, BlockKind::V, phys_before).unwrap();

        let ticket = mover.demote(&mut plane, &mut store, s, &[1]).unwrap();
        assert!(dir.path().join(&ticket.keys[0].1).exists());
        mover.promote(&mut plane, &mut store, s, ticket).unwrap();
        let phys_after = plane.blocks().resident_block(s, 1).unwrap();
        assert_eq!(plane.read_block_f32(0, BlockKind::V, phys_after).unwrap(), expected);
        // promote deletes its keys.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    // -------------------------------------------------------------------
    // The real Fuel device pool. No checkpoint needed — a 4-block toy pool.
    // -------------------------------------------------------------------

    #[test]
    fn device_pool_round_trip_survives_a_reallocating_restore() {
        use fuel::{DType, Device};

        let bs = 2;
        let g = geom(4, bs);
        let dev = Device::cpu();
        let mut dpool = DeviceKvPool::new(g.kv(), DType::F32, &dev).unwrap();
        let mut store = MemStore::default();
        let mover = BlockTierMover::new(g, "dev-sess");

        let s = dpool.core_mut().open();
        dpool.core_mut().append(s, 4).unwrap(); // 2 blocks
        let phys_before = dpool.core().resident_block(s, 1).unwrap();

        let k: Vec<f32> = (0..g.block_elems()).map(|i| 10.0 + i as f32).collect();
        let v: Vec<f32> = (0..g.block_elems()).map(|i| 20.0 + i as f32).collect();
        dpool.write_block(0, BlockKind::K, phys_before, &k).unwrap();
        dpool.write_block(0, BlockKind::V, phys_before, &v).unwrap();

        // Demote through the PURE core + read_block/write_block seam (NOT
        // DeviceKvPool::evict_blocks, whose bytes are inaccessible).
        let ticket = mover.demote(&mut dpool, &mut store, s, &[1]).unwrap();
        assert_eq!(ticket.freed, vec![1]);

        // Steal and poison the freed physical block.
        let decoy = dpool.core_mut().open();
        dpool.core_mut().append(decoy, 2).unwrap();
        let stolen = dpool.core().resident_block(decoy, 0).unwrap();
        assert_eq!(stolen, phys_before);
        let poison: Vec<f32> = vec![-999.0; g.block_elems()];
        dpool.write_block(0, BlockKind::K, stolen, &poison).unwrap();
        dpool.write_block(0, BlockKind::V, stolen, &poison).unwrap();

        mover.promote(&mut dpool, &mut store, s, ticket).unwrap();
        let phys_after = dpool.core().resident_block(s, 1).unwrap();
        assert_ne!(phys_after, phys_before, "restore reallocated, as designed");
        assert_eq!(dpool.read_block(0, BlockKind::K, phys_after).unwrap(), k);
        assert_eq!(dpool.read_block(0, BlockKind::V, phys_after).unwrap(), v);
        // And the poisoned old location is untouched by the promote.
        assert_eq!(dpool.read_block(0, BlockKind::K, phys_before).unwrap(), poison);
    }
}
