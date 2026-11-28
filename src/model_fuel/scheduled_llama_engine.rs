// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wires `FuelPagedSchedulerDriver` (`super::scheduler_driver_paged`) into
//! `ModelRunner::start`'s `fuel-engine` arm (board item 97's actual remaining
//! deliverable: everything before this file only unblocked it). All three
//! arms were ported from the block-reservation `FuelSchedulerDriver` to the
//! paged one once Fuel 0.22.4 published `PagedDecodeModel` for all three
//! concrete types this file loads (paged-scheduler slice 2, 2026-10-23).
//!
//! `ModelRunner::start`'s `fuel-engine` arm can load an eager SafeTensors
//! checkpoint (`LlamaModel`), a GGUF-quantized Llama one
//! (`QuantizedLlama3Model`), or — since `serve_gguf` reads the file's own
//! `general.architecture` via `crate::gguf::detect_architecture` — a
//! GGUF-quantized Qwen3 one (`QuantizedQwen3Model`). Three different
//! concrete `DecodeModel` types, so one `FuelSchedulerDriver<'m, M>` cannot
//! hold any two interchangeably. `ServingDriver` (below, `pub(crate)` — not
//! an intra-doc link, same reason as `scheduler_driver.rs`'s own module doc)
//! is the small enum-dispatch wrapper that lets `run_scheduled_jobs` stay
//! generic over ONE `SchedulerDriver` regardless of which checkpoint format
//! or architecture was loaded.
//!
//! The SafeTensors-directory path (`serve_f32`) stays Llama-only — there is
//! no eager Qwen3 loader in this codebase, and no architecture metadata to
//! read the way GGUF's `general.architecture` provides it; widening that
//! path is out of scope here, not an oversight.
//!
//! # Multi-token EOS checkpoints: no longer a special case
//!
//! Earlier revisions of this file routed `LlamaEosToks::Multiple` checkpoints
//! to the serial `FuelEngineModel`/`run_jobs` path instead of the batched
//! one, because fuel's `SessionState` tracked only ONE `eos_id: Option<u32>`
//! — a checkpoint with several stop tokens would have stopped on whichever
//! one was chosen and run PAST the others to the budget ceiling on the
//! batched path. **fuel#307 widened that to `eos_ids: Option<Vec<u32>>`**
//! (stop-check is "any configured id"), so every checkpoint — `None`,
//! `Single`, or `Multiple` — now takes the batched path uniformly via
//! `eos_ids_from` (private, not an intra-doc link here since this module
//! is `pub`).
#![allow(dead_code)]

use std::path::Path;
use std::sync::mpsc::Receiver;

use anyhow::Result;

use fuel_model_llama::LlamaModel;
use fuel_transformers::models::lazy_llama_full::LlamaEosToks;
use fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model;
use fuel_transformers::models::lazy_quantized_qwen3::QuantizedQwen3Model;

use fuel_inference::multi_session::{KvBudget, PagedSchedulePolicy};

use super::engine_model::effective_generation_budget;
use super::policies::BlockPrefixIndexFor;
use super::prefix_retention::{PrefixSource, RetainedId, RetainedStore};

mod serve_loaders;
use super::scheduler_driver_paged::{FuelPagedSchedulerDriver, SERVING_SCHEDULE_POLICY};
use serve_loaders::*;

use crate::engine::InferenceJob;
use crate::engine::SlotPoolRequest as SlotRequest;
use crate::engine::model_runner::{
    ControlMsg, SamplingExtras, SharedCacheStats, SharedScheduler, drain_with_error,
};
use crate::engine::scheduled_runner::{
    AdmitError, FinishOutcome, RawLogprob, SchedulerDriver, run_scheduled_jobs,
};
use crate::gguf::{GgufArchitecture, detect_architecture};

/// The checkpoint's full EOS set, in fuel's `eos_ids: Option<Vec<u32>>`
/// shape. `LlamaEosToks::Single`/`Multiple`/absent map onto it directly —
/// see this module's doc comment for why there is no longer a case this
/// cannot represent.
fn eos_ids_from(eos: Option<&LlamaEosToks>) -> Option<Vec<u32>> {
    match eos {
        None => None,
        Some(LlamaEosToks::Single(id)) => Some(vec![*id]),
        Some(LlamaEosToks::Multiple(ids)) => Some(ids.clone()),
    }
}

enum ServingDriverInner<'m> {
    F32(FuelPagedSchedulerDriver<'m, LlamaModel>),
    QuantizedGguf(FuelPagedSchedulerDriver<'m, QuantizedLlama3Model>),
    QuantizedGgufQwen3(FuelPagedSchedulerDriver<'m, QuantizedQwen3Model>),
}

/// Enum-dispatch over the concrete `DecodeModel` types `ModelRunner::start`'s
/// fuel-engine arm can load (Llama eager/GGUF, and — since
/// `detect_architecture` lets `serve_gguf` pick a loader — Qwen3 GGUF), so
/// `run_scheduled_jobs` sees one `SchedulerDriver` regardless of checkpoint
/// format or architecture.
///
/// Owns `context_length` so `try_admit` can apply
/// [`effective_generation_budget`] itself — the one piece of policy that is
/// NOT `FuelPagedSchedulerDriver`'s concern (it has no notion of a
/// server-wide context ceiling), but IS this integration layer's, same role
/// `FuelEngineModel::step_one` played in the old serial path.
///
/// `scheduler`/`sessions` mirror every real admission/departure into the
/// shared `MemoryAwareScheduler` (board PM ruling, 2026-10-22, now that
/// Fuel 0.20.71 is published) — the same role `run_jobs`'s `admit_pending`/
/// `release_scheduler_slot` played on the deleted candlelight path (M4
/// slice 5.4a), rebuilt here against the live Fuel path instead. A passive
/// mirror, not a second authority: Fuel's own `try_admit` above already
/// decided admission — optimistically, as of the paged-scheduler port (see
/// `FuelPagedSchedulerDriver`'s own doc comment: there is no up-front
/// KV-block capacity check any more, only a later per-session failure if it
/// doesn't fit); this only records that decision so `scheduler_stats`/
/// `scheduler_queue` have real numbers to report. `sessions` maps Fuel's
/// `SessionId` to the synthetic request id (`"session-{id}"`) submitted to
/// the mirror, needed because `reap_finished` only gets the `SessionId`
/// back, not the original request's own metadata.
///
/// `prefix_index`/`cache_stats` give `cache_stats` real DETECTION numbers
/// (board PM task, 2026-10-22, ROADMAP research-track item 1 / FUEL-GAPS
/// entry #1) — `BlockPrefixIndexFor<SessionId>` (the policy-level chain-hash
/// index `src/model_fuel/policies.rs` already built and proved correct, now
/// generalized off `fuel::kv_block_pool::SessionHandle` so a caller with no
/// live `KvBlockPool` session can still use it) records every admitted
/// prompt's token-level prefix and looks up each new one against it. **This
/// is detection only, not exploitation**: a hit is NOT currently spliced
/// into the donor's KV blocks or used to skip prefill — this file now runs
/// on `PagedSessionScheduler` (via `FuelPagedSchedulerDriver`), which DOES
/// expose `add_session_sharing_prefix`/`register_prefix` for that, but
/// wiring a detected hit into an actual splice is a separate, not-yet-done
/// step. `hits`/`misses`/`insertions`/`entries` are therefore real; `evictions`
/// counts real `forget_donor` calls (on departure, below); nothing here
/// claims a `total_saved_tokens` figure — `CacheStatsResponse` never reads
/// that field, and claiming it without an actual skipped prefill would be
/// exactly the fabricated-number defect this project's conventions forbid.
/// What a prefix-index entry names: a running session (gone when it finishes) or a retained registration that
/// outlives it (`docs/PREFIX-CACHE-DESIGN.md`).
type PrefixSrc = PrefixSource<fuel_inference::multi_session::SessionId, RetainedId>;

/// One retained depth: the pool registration and the index key it serves (so eviction can drop both).
#[derive(Clone, Copy, Debug)]
struct RetainedEntry {
    prefix: fuel::kv_block_pool::PrefixId,
    key: [u8; 32],
}

/// Deepest chain registered per prompt (in blocks). Every depth up to here is registered, so the index stays
/// prefix-closed; a longer prompt shares its first `RETENTION_MAX_DEPTH` blocks.
const RETENTION_MAX_DEPTH: usize = 64;

struct ServingDriver<'m> {
    inner: ServingDriverInner<'m>,
    context_length: usize,
    scheduler: SharedScheduler,
    sessions: std::collections::HashMap<fuel_inference::multi_session::SessionId, String>,
    prefix_index: BlockPrefixIndexFor<PrefixSrc>,
    /// Prompt prefixes kept past their donor (budget 0 = off, the default). Bounded and LRU; see `prefix_retention`.
    retained: RetainedStore<RetainedEntry>,
    /// Sessions whose prompt blocks are not all filled yet: their chain keys, retried after every step.
    pending_retention:
        std::collections::HashMap<fuel_inference::multi_session::SessionId, Vec<[u8; 32]>>,
    cache_stats: SharedCacheStats,
    /// ROADMAP research-track item 1, slice 3: off-by-default switch for
    /// actually exploiting a `prefix_index` hit (splice + skip prefill),
    /// rather than only detecting it (`hits`/`misses` in `cache_stats`,
    /// which run regardless of this flag). `false` for every construction
    /// site today -- no operator-facing toggle exists yet; exposing one
    /// (env var/CLI) is a follow-up once this is trusted, not this slice's
    /// scope. Exists so `try_admit`'s off path and on path share one
    /// function instead of diverging into two.
    prefix_sharing_enabled: bool,
    /// Donors already registered as a shared [`fuel::kv_block_pool::PrefixId`]
    /// with the live `PagedSessionScheduler`, and how many blocks each
    /// registration covers -- a SEPARATE lifetime mechanism from
    /// `prefix_index`'s own `donor_refs` (that one is lightbulb's own
    /// bookkeeping for the old, never-live `KvBlockPool`/`splice_prefix`
    /// path; this one is the pool's real refcount on the paged path).
    /// Released when the donor departs (`mirror_departure`), which is safe
    /// even if a live sharer still references the blocks independently
    /// (`PagedSessionScheduler::release_prefix`'s own doc comment).
    ///
    /// Keyed by `(donor, blocks)` EXACTLY: a `PrefixId` splices every block
    /// it was registered over, so reusing a registration that covers MORE
    /// blocks than the current lookup matched would graft KV of tokens the
    /// new prompt does not contain.
    registered_prefixes: std::collections::HashMap<
        (fuel_inference::multi_session::SessionId, usize),
        fuel::kv_block_pool::PrefixId,
    >,
    /// Admissions that actually went through `add_session_sharing_prefix`
    /// (not the cold fallback). Lets a test tell a real splice from a
    /// silent fallback, which `registered_prefixes` alone cannot.
    prefix_shared_admissions: u64,
}

impl<'m> ServingDriver<'m> {
    /// Mirror a just-admitted session into the shared scheduler: submit a
    /// synthetic [`SlotRequest`] and immediately try to allocate it a slot.
    /// `top_p` has no real value here -- `SchedulerDriver::try_admit` never
    /// receives one -- so `1.0` (no nucleus truncation) is used as a
    /// placeholder; nothing downstream of the mirror reads it back.
    fn mirror_admission(
        &mut self,
        id: fuel_inference::multi_session::SessionId,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) {
        let request_id = format!("session-{}", id.0);
        let request = SlotRequest {
            id: request_id.clone(),
            prompt_tokens: prompt.to_vec(),
            max_new_tokens: max_new,
            temperature,
            top_p: 1.0,
        };
        let mut scheduler = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
        scheduler.submit_request(request);
        scheduler.allocate_pending_requests();
        drop(scheduler);
        self.sessions.insert(id, request_id);
    }

    /// Mirror a just-reaped session's departure: free its slot in the
    /// shared scheduler, if the earlier mirrored admission ever got one
    /// (it may not have, if the mirror's own smaller/misconfigured budget
    /// queued it rather than allocating it -- freeing a slot that was
    /// never allocated would be a bug, not a no-op, so this checks first).
    fn mirror_departure(&mut self, id: fuel_inference::multi_session::SessionId) {
        // Bounds the index's growth and gives `evictions` a real count — not
        // a correctness requirement the way it is for `splice_prefix`'s
        // actual pool callers (this index never gates a `KvBlockPool::
        // discard`), just honest bookkeeping for a finished session.
        self.pending_retention.remove(&id);
        if self.prefix_index.forget_donor(PrefixSource::Live(id)) > 0 {
            let entries = self.prefix_index.len();
            let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.evictions += 1;
            stats.entries = entries;
        }
        // Release this donor's pool-level prefix registration, if it had
        // one -- safe even while a live sharer still references the blocks
        // (see `registered_prefixes`'s own doc comment). A no-op whenever
        // `prefix_sharing_enabled` is false: nothing is ever inserted into
        // `registered_prefixes` on that path.
        let owned: Vec<(fuel_inference::multi_session::SessionId, usize)> = self
            .registered_prefixes
            .keys()
            .filter(|(donor, _)| *donor == id)
            .copied()
            .collect();
        for key in owned {
            if let Some(prefix) = self.registered_prefixes.remove(&key) {
                let _ = self.release_prefix_inner(prefix);
            }
        }
        let Some(request_id) = self.sessions.remove(&id) else {
            return;
        };
        let scheduler = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
        let slot_id = scheduler.slot_pool().get_request_slot(&request_id).copied();
        drop(scheduler);
        if let Some(slot_id) = slot_id {
            let mut scheduler = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
            let _ = scheduler.free_slot(slot_id);
        }
    }

    /// Index `prompt` itself under `id` (a just-admitted session) so a
    /// LATER admission can be detected against it, then mirror the real
    /// `BlockPrefixStats` into `SharedCacheStats` — the same immediate-write
    /// pattern `mirror_admission`/`mirror_departure` already use for the
    /// scheduler. The matching *lookup* for `prompt` itself happens earlier,
    /// in `try_admit`, before admission is even decided (needed there to
    /// pick the admission path when `prefix_sharing_enabled`); this is only
    /// the record-and-mirror tail, run unconditionally and identically on
    /// both the exploited and detection-only paths.
    fn record_prefix(&mut self, id: fuel_inference::multi_session::SessionId, prompt: &[u32]) {
        self.prefix_index.record(prompt, PrefixSource::Live(id));
        let block_stats = self.prefix_index.stats();
        let entries = self.prefix_index.len();
        let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.hits = block_stats.hits;
        stats.misses = block_stats.misses;
        stats.insertions = block_stats.blocks_indexed;
        stats.entries = entries;
    }

    /// The `prefix_sharing_enabled == false` admission path — unchanged
    /// from before slice 3: every arm's own cold `try_admit`, nothing else.
    fn plain_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
        seed: Option<u64>,
        extras: SamplingExtras,
    ) -> Result<fuel_inference::multi_session::SessionId, AdmitError> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.try_admit(prompt, max_new, temperature, seed, extras),
            ServingDriverInner::QuantizedGguf(d) => {
                d.try_admit(prompt, max_new, temperature, seed, extras)
            }
            ServingDriverInner::QuantizedGgufQwen3(d) => {
                d.try_admit(prompt, max_new, temperature, seed, extras)
            }
        }
    }

    /// On a lookup `hit`, try to actually splice the shared prefix instead
    /// of a cold prefill (`prefix_sharing_enabled`'s real exploitation).
    /// Falls back to [`Self::plain_admit`] -- never fails the request just
    /// because sharing didn't pan out -- on a miss, on a donor whose blocks
    /// are not (or no longer) fully resident up to the matched length (e.g.
    /// still mid-prefill, or reaped since the match was computed), or on any
    /// other Fuel-side refusal of the splice.
    fn try_admit_sharing_or_fallback(
        &mut self,
        hit: Option<super::policies::PrefixMatchFor<PrefixSrc>>,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
        seed: Option<u64>,
        extras: SamplingExtras,
    ) -> Result<fuel_inference::multi_session::SessionId, AdmitError> {
        let Some(hit) = hit else {
            return self.plain_admit(prompt, max_new, temperature, seed, extras);
        };
        // Fuel rejects a share that leaves no suffix (`prompt.len() <=
        // shared_tokens`): the first generated token needs logits from at
        // least one prefilled prompt token. A full-prompt match (the common
        // repeat-request case) therefore shares one block fewer.
        let blocks = hit
            .blocks
            .min(prompt.len().saturating_sub(1) / self.prefix_index.block_size());
        if blocks == 0 {
            return self.plain_admit(prompt, max_new, temperature, seed, extras);
        }
        let Some(prefix) = self.prefix_for(hit.donor, blocks) else {
            return self.plain_admit(prompt, max_new, temperature, seed, extras);
        };
        let shared = match &mut self.inner {
            ServingDriverInner::F32(d) => {
                d.try_admit_sharing_prefix(prefix, prompt, max_new, temperature, seed, extras)
            }
            ServingDriverInner::QuantizedGguf(d) => {
                d.try_admit_sharing_prefix(prefix, prompt, max_new, temperature, seed, extras)
            }
            ServingDriverInner::QuantizedGgufQwen3(d) => {
                d.try_admit_sharing_prefix(prefix, prompt, max_new, temperature, seed, extras)
            }
        };
        match shared {
            Ok(id) => {
                self.prefix_shared_admissions += 1;
                Ok(id)
            }
            Err(_) => self.plain_admit(prompt, max_new, temperature, seed, extras),
        }
    }

    /// The pool registration that serves a `blocks`-deep hit on `source`: a live donor is registered on demand, a
    /// retained chain already holds one per depth (and counts as used).
    fn prefix_for(
        &mut self,
        source: PrefixSrc,
        blocks: usize,
    ) -> Option<fuel::kv_block_pool::PrefixId> {
        match source {
            PrefixSource::Live(donor) => self.ensure_prefix_registered(donor, blocks),
            PrefixSource::Retained(id) => {
                self.retained.touch(id);
                self.retained.payload_at(id, blocks).map(|e| e.prefix)
            }
        }
    }

    /// Register `donor`'s first `blocks` filled blocks as a shared prefix
    /// with the live `PagedSessionScheduler`, reusing a registration only
    /// if it covers EXACTLY this many blocks (see `registered_prefixes`).
    /// `None` on any Fuel-side refusal (e.g. the donor's blocks are not all
    /// filled yet).
    fn ensure_prefix_registered(
        &mut self,
        donor: fuel_inference::multi_session::SessionId,
        blocks: usize,
    ) -> Option<fuel::kv_block_pool::PrefixId> {
        if let Some(&prefix) = self.registered_prefixes.get(&(donor, blocks)) {
            return Some(prefix);
        }
        match self.register_inner(donor, blocks) {
            Ok(prefix) => {
                self.registered_prefixes.insert((donor, blocks), prefix);
                Some(prefix)
            }
            Err(_) => None,
        }
    }

    fn register_inner(
        &mut self,
        donor: fuel_inference::multi_session::SessionId,
        blocks: usize,
    ) -> fuel::Result<fuel::kv_block_pool::PrefixId> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.register_prefix(donor, blocks),
            ServingDriverInner::QuantizedGguf(d) => d.register_prefix(donor, blocks),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.register_prefix(donor, blocks),
        }
    }

    /// Queue `id`'s prompt chain for retention (no-op when retention is off, or the prompt's whole chain is
    /// already indexed). The registrations happen in [`Self::retain_ready_prefixes`], once the blocks are filled.
    fn queue_retention(
        &mut self,
        id: fuel_inference::multi_session::SessionId,
        prompt: &[u32],
        hit_blocks: usize,
    ) {
        if self.retained.budget() == 0 {
            return;
        }
        let keys = self.prefix_index.chain_keys(prompt);
        let depth = keys.len().min(RETENTION_MAX_DEPTH);
        if depth > hit_blocks {
            self.pending_retention.insert(id, keys[..depth].to_vec());
        }
    }

    /// Retry every queued chain. Fuel refuses `register_prefix` until the donor holds that many fully filled
    /// blocks and a refusal leaves the pool untouched, so the DEEPEST registration is the readiness probe: it
    /// succeeds exactly when the whole prompt prefix is resident.
    fn retain_ready_prefixes(&mut self) {
        let ids: Vec<_> = self.pending_retention.keys().copied().collect();
        for id in ids {
            let depth = self.pending_retention[&id].len();
            let Ok(deepest) = self.register_inner(id, depth) else {
                continue;
            };
            if let Some(keys) = self.pending_retention.remove(&id) {
                self.retain_chain(id, &keys, deepest);
            }
        }
    }

    /// Register the depths `id` still owns in the index, hand them to the store (evicting to fit), and move the
    /// index entries from `Live(id)` to the new retained chain, so they survive `forget_donor` at departure.
    fn retain_chain(
        &mut self,
        id: fuel_inference::multi_session::SessionId,
        keys: &[[u8; 32]],
        deepest: fuel::kv_block_pool::PrefixId,
    ) {
        let regs = self.register_owned_depths(id, keys, deepest);
        self.store_chain(id, regs);
    }

    /// Register every depth of `keys` that `id` still owns in the index (the deepest one is already registered
    /// as `deepest`, which is released instead when `id` no longer owns that depth).
    fn register_owned_depths(
        &mut self,
        id: fuel_inference::multi_session::SessionId,
        keys: &[[u8; 32]],
        deepest: fuel::kv_block_pool::PrefixId,
    ) -> Vec<(usize, RetainedEntry)> {
        let live = PrefixSource::Live(id);
        let depth = keys.len();
        let mut regs: Vec<(usize, RetainedEntry)> = Vec::new();
        for d in 1..depth {
            if self.prefix_index.owned_by(&keys[d - 1], live) {
                match self.register_inner(id, d) {
                    Ok(prefix) => regs.push((
                        d,
                        RetainedEntry {
                            prefix,
                            key: keys[d - 1],
                        },
                    )),
                    Err(_) => break,
                }
            }
        }
        if self.prefix_index.owned_by(&keys[depth - 1], live) {
            regs.push((
                depth,
                RetainedEntry {
                    prefix: deepest,
                    key: keys[depth - 1],
                },
            ));
        } else {
            let _ = self.release_prefix_inner(deepest);
        }
        regs
    }

    /// Hand registered depths to the store (evicting to fit; releasing them all if it refuses) and move the
    /// index entries from `Live(id)` to the new retained chain.
    fn store_chain(
        &mut self,
        id: fuel_inference::multi_session::SessionId,
        regs: Vec<(usize, RetainedEntry)>,
    ) {
        let live = PrefixSource::Live(id);
        let held: Vec<_> = regs.iter().map(|r| r.1.prefix).collect();
        match self.retained.insert(regs.clone()) {
            Err(_) => held.into_iter().for_each(|p| {
                let _ = self.release_prefix_inner(p);
            }),
            Ok((rid, evicted)) => {
                for e in evicted {
                    self.drop_evicted(&e);
                }
                for (_, entry) in regs {
                    self.prefix_index
                        .adopt(&entry.key, live, PrefixSource::Retained(rid));
                }
            }
        }
        let entries = self.prefix_index.len();
        self.cache_stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries = entries;
    }

    /// Undo one eviction the store reported: drop its index entry and release its pool registration.
    fn drop_evicted(&mut self, e: &super::prefix_retention::Evicted<RetainedEntry>) {
        self.prefix_index
            .remove_key_owned(&e.payload.key, PrefixSource::Retained(e.chain));
        let _ = self.release_prefix_inner(e.payload.prefix);
        let entries = self.prefix_index.len();
        let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.evictions += 1;
        stats.entries = entries;
    }

    fn free_blocks(&self) -> usize {
        match &self.inner {
            ServingDriverInner::F32(d) => d.kv_free_blocks(),
            ServingDriverInner::QuantizedGguf(d) => d.kv_free_blocks(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.kv_free_blocks(),
        }
    }

    /// Make room for a request of `tokens` total tokens. The chain the request is about to splice from is
    /// touched first so it is not the first one evicted for it.
    fn make_room_for(&mut self, hit_source: Option<PrefixSrc>, tokens: usize) {
        if self.retained.chains() == 0 {
            return;
        }
        if let Some(PrefixSource::Retained(id)) = hit_source {
            self.retained.touch(id);
        }
        self.make_room(tokens.div_ceil(self.prefix_index.block_size()));
    }

    /// Pressure rule (docs/PREFIX-CACHE-DESIGN.md 3.3): a retained prefix must never be the reason a live
    /// request fails. Fuel's paged admission is optimistic (pool exhaustion surfaces later as that session's
    /// `Failed`), so there is no refusal to retry on; the check is up front instead: evict retained chains,
    /// least recently used first, until the pool has `need` free blocks or nothing retained is left.
    fn make_room(&mut self, need: usize) {
        while self.free_blocks() < need {
            let Some(e) = self.retained.evict_one() else {
                break;
            };
            self.drop_evicted(&e);
        }
    }

    fn release_prefix_inner(&mut self, prefix: fuel::kv_block_pool::PrefixId) -> fuel::Result<()> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.release_prefix(prefix),
            ServingDriverInner::QuantizedGguf(d) => d.release_prefix(prefix),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.release_prefix(prefix),
        }
    }

    /// The real `cache_clear` admin endpoint's implementation (board PM
    /// task, 2026-10-22). Rebuilds `prefix_index` from scratch -- the
    /// simplest correct "clear": every future lookup is a fresh miss until
    /// new prompts are admitted and recorded again -- and zeroes the
    /// `SharedCacheStats` mirror to match. `sessions`/`scheduler`
    /// (admission bookkeeping) are untouched: clearing the prefix cache
    /// must never affect in-flight requests' admission state.
    fn clear_prefix_cache(&mut self) {
        let block_size = self.prefix_index.block_size();
        self.prefix_index = BlockPrefixIndexFor::new(block_size);
        self.pending_retention.clear();
        for e in self.retained.drain() {
            let _ = self.release_prefix_inner(e.payload.prefix);
        }
        let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = crate::cache::PrefixCacheStats::default();
    }
}

impl<'m> SchedulerDriver for ServingDriver<'m> {
    type Id = fuel_inference::multi_session::SessionId;

    fn try_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
        seed: Option<u64>,
        extras: SamplingExtras,
    ) -> Result<Self::Id, AdmitError> {
        let (effective_max_new, truncated) =
            effective_generation_budget(prompt.len(), max_new, self.context_length);
        if truncated {
            tracing::warn!(
                requested_max_new_tokens = max_new,
                truncated_max_new_tokens = effective_max_new,
                context_length = self.context_length,
                prompt_tokens = prompt.len(),
                "requested max_new_tokens does not fit in context_length; truncating \
                 the generation budget so this request returns the tokens it can \
                 generate instead of erroring mid-decode"
            );
        }
        // Always run, regardless of `prefix_sharing_enabled` -- same
        // lookup `detect_and_record_prefix` used to do here directly,
        // giving `hits`/`misses` byte-identical behavior on both paths.
        // Doing it before deciding the admission path (rather than after,
        // as before) is required so the `true` path can act on the result;
        // the `false` path simply never looks at `hit`, which is the whole
        // reason the off path stays byte-identical to pre-slice-3 behavior.
        let hit = self.prefix_index.lookup(prompt);
        let hit_blocks = hit.map_or(0, |h| h.blocks);
        self.make_room_for(hit.map(|h| h.donor), prompt.len() + effective_max_new);

        let admitted = if self.prefix_sharing_enabled {
            self.try_admit_sharing_or_fallback(
                hit,
                prompt,
                effective_max_new,
                temperature,
                seed,
                extras,
            )
        } else {
            self.plain_admit(prompt, effective_max_new, temperature, seed, extras)
        };
        if let Ok(id) = admitted {
            self.mirror_admission(id, prompt, effective_max_new, temperature);
            self.record_prefix(id, prompt);
            self.queue_retention(id, prompt, hit_blocks);
        }
        admitted
    }

    fn step(&mut self) -> Result<Vec<(Self::Id, u32)>, String> {
        let out = match &mut self.inner {
            ServingDriverInner::F32(d) => d.step(),
            ServingDriverInner::QuantizedGguf(d) => d.step(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.step(),
        };
        self.retain_ready_prefixes();
        out
    }

    fn take_step_logprobs(&mut self) -> Vec<(Self::Id, RawLogprob)> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.take_step_logprobs(),
            ServingDriverInner::QuantizedGguf(d) => d.take_step_logprobs(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.take_step_logprobs(),
        }
    }

    fn is_eos(&self, token: u32) -> bool {
        match &self.inner {
            ServingDriverInner::F32(d) => d.is_eos(token),
            ServingDriverInner::QuantizedGguf(d) => d.is_eos(token),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.is_eos(token),
        }
    }

    fn reap_finished(&mut self) -> Vec<(Self::Id, FinishOutcome)> {
        let reaped = match &mut self.inner {
            ServingDriverInner::F32(d) => d.reap_finished(),
            ServingDriverInner::QuantizedGguf(d) => d.reap_finished(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.reap_finished(),
        };
        for (id, _) in &reaped {
            self.mirror_departure(*id);
        }
        reaped
    }

    fn has_active_sessions(&self) -> bool {
        match &self.inner {
            ServingDriverInner::F32(d) => d.has_active_sessions(),
            ServingDriverInner::QuantizedGguf(d) => d.has_active_sessions(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.has_active_sessions(),
        }
    }

    fn handle_control(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::ClearPrefixCache => self.clear_prefix_cache(),
        }
    }

    fn set_priority(&mut self, id: Self::Id, priority: i32) -> Result<(), String> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.set_priority(id, priority),
            ServingDriverInner::QuantizedGguf(d) => d.set_priority(id, priority),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.set_priority(id, priority),
        }
    }
}

/// A generous block size: large enough that `kv_blocks_required`'s ceiling
/// division wastes little, small enough that a short prompt does not reserve
/// a whole extra block it will never use. Matches the block size fuel's own
/// `multi_session.rs` test fixtures use.
pub(super) const KV_BLOCK_SIZE: usize = 16;

/// The server-wide (not per-request) pieces of the batched path, bundled so
/// every function between [`run_fuel_engine`] and [`run_batched`] takes one
/// parameter instead of two or more — `device`/`policy`/`budget` are all
/// `Copy`; `scheduler` is an `Arc` clone away from being too, so bundling it
/// here too keeps every one of those functions at the same parameter count
/// it had before the 2026-10-22 scheduler-mirroring wiring added a new
/// cross-cutting dependency (`clippy::too_many_arguments`, not just style).
struct BatchedServeConfig {
    device: fuel::Device,
    /// Passed to `FuelPagedSchedulerDriver::step`'s `step_batched` every
    /// tick -- `PagedSchedulePolicy` (below) only controls ORDER (admission
    /// order vs. priority); batching itself is a separate knob on
    /// `PagedSessionScheduler::step_batched`, not a policy variant the way
    /// `SchedulePolicy::Batched` was on the non-paged scheduler.
    max_batch: usize,
    policy: PagedSchedulePolicy,
    budget: KvBudget,
    scheduler: SharedScheduler,
    cache_stats: SharedCacheStats,
    ctrl_rx: Receiver<ControlMsg>,
    /// ROADMAP research-track item 1, slice 3's off-by-default switch --
    /// see `ServingDriver::prefix_sharing_enabled`'s own doc comment.
    /// `false` from `new`; `run_fuel_engine` sets it from
    /// `EngineOptions::prefix_sharing`.
    prefix_sharing_enabled: bool,
    /// Retention budget in blocks; `run_fuel_engine` sets it only where sharing is actually on (0 otherwise).
    prefix_retention_blocks: usize,
}

impl BatchedServeConfig {
    fn new(
        max_batch_size: usize,
        context_length: usize,
        scheduler: SharedScheduler,
        cache_stats: SharedCacheStats,
        ctrl_rx: Receiver<ControlMsg>,
    ) -> anyhow::Result<Self> {
        let max_batch_size = max_batch_size.max(1);
        Ok(Self {
            device: super::device::try_select()?,
            max_batch: max_batch_size,
            policy: SERVING_SCHEDULE_POLICY,
            budget: KvBudget {
                block_size: KV_BLOCK_SIZE,
                num_blocks: max_batch_size * context_length.div_ceil(KV_BLOCK_SIZE),
            },
            scheduler,
            cache_stats,
            ctrl_rx,
            prefix_sharing_enabled: false,
            prefix_retention_blocks: 0,
        })
    }
}

/// Build the batched driver for one already-loaded model and drive it —
/// shared by both the `.gguf` and SafeTensors-directory branches of
/// [`run_fuel_engine`], which differ only in the CONCRETE `M` and which
/// `ServingDriverInner` variant wraps it (`wrap`, a tuple-variant
/// constructor used as a plain `fn`).
fn run_batched<'m, M: fuel_inference::multi_session::PagedDecodeModel>(
    model: &'m M,
    config: BatchedServeConfig,
    eos_ids: Option<Vec<u32>>,
    context_length: usize,
    tokenizer: tokenizers::Tokenizer,
    rx: Receiver<InferenceJob>,
    wrap: fn(FuelPagedSchedulerDriver<'m, M>) -> ServingDriverInner<'m>,
) {
    let fuel_scheduler = match FuelPagedSchedulerDriver::new(
        model,
        config.budget,
        fuel::DType::F32,
        &config.device,
        config.policy,
        config.max_batch,
        eos_ids,
    ) {
        Ok(s) => s,
        Err(e) => return drain_with_error(rx, &format!("building the scheduler: {e}")),
    };
    let ctrl_rx = config.ctrl_rx;
    let driver = ServingDriver {
        inner: wrap(fuel_scheduler),
        context_length,
        scheduler: config.scheduler,
        sessions: std::collections::HashMap::new(),
        prefix_index: BlockPrefixIndexFor::new(KV_BLOCK_SIZE),
        retained: RetainedStore::new(config.prefix_retention_blocks),
        pending_retention: std::collections::HashMap::new(),
        cache_stats: config.cache_stats,
        prefix_sharing_enabled: config.prefix_sharing_enabled,
        registered_prefixes: std::collections::HashMap::new(),
        prefix_shared_admissions: 0,
    };
    // `tokenizer.encode`/`.decode` each need their own `move` closure below,
    // and `Tokenizer` is not `Copy` — an `Arc` is what lets both closures
    // share it rather than one of them moving the only copy.
    let tokenizer = std::sync::Arc::new(tokenizer);
    let encode_tokenizer = tokenizer.clone();
    run_scheduled_jobs(
        driver,
        rx,
        ctrl_rx,
        move |text, add_special| {
            encode_tokenizer
                .encode(text, add_special)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))
                .map(|enc| enc.get_ids().to_vec())
        },
        move |tokens, skip_special| {
            tokenizer
                .decode(tokens, skip_special)
                .map_err(|e| anyhow::anyhow!("detokenizing: {e}"))
        },
    );
}

/// Load a checkpoint (dispatching on `.gguf` vs a SafeTensors directory) and
/// drive it through the real `FuelSchedulerDriver`/`run_scheduled_jobs`
/// batched path — uniformly, regardless of its EOS shape (see this module's
/// doc comment).
///
/// `max_batch_size` sizes the KV pool (`max_batch_size` concurrent sessions
/// of up to `context_length` tokens each) AND selects
/// `SchedulePolicy::Batched` on the batched path — a model that doesn't
/// implement the batched arm degrades to serial automatically
/// (`SessionScheduler`'s own documented fallback), so this is never a
/// correctness risk, only a speed one.
///
/// Never returns an `Err` to the caller: a load or construction failure is
/// handled HERE, by answering every already-queued and future job with an
/// error (`drain_with_error`, the same function both backends' `ModelRunner::
/// start` already use) — matching the old call site's own two-level match
/// exactly, since `rx` is consumed by whichever path runs and cannot be
/// handed back to a caller to drain a second time.
#[allow(clippy::too_many_arguments)] // `ModelRunner::start`'s seven plus `options`
pub(crate) fn run_fuel_engine(
    model_path: &Path,
    max_batch_size: usize,
    context_length: usize,
    rx: Receiver<InferenceJob>,
    ctrl_rx: Receiver<ControlMsg>,
    scheduler: SharedScheduler,
    cache_stats: SharedCacheStats,
    options: crate::engine::model_runner::EngineOptions,
) {
    let mut config = match BatchedServeConfig::new(
        max_batch_size,
        context_length,
        scheduler,
        cache_stats,
        ctrl_rx,
    ) {
        Ok(c) => c,
        Err(e) => return load_failed(model_path, e, rx),
    };
    config.prefix_sharing_enabled =
        super::memory_shape::resolve_prefix_sharing(options.prefix_sharing, model_path);
    config.prefix_retention_blocks = if config.prefix_sharing_enabled {
        options.prefix_retention_blocks
    } else {
        0
    };
    if let Err(e) = super::preflight::enforce(
        model_path,
        &config.device,
        options.allow_oversize_model,
        max_batch_size,
        context_length,
    ) {
        return load_failed(model_path, e, rx);
    }
    let is_gguf = model_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
    if is_gguf {
        serve_gguf(model_path, context_length, config, rx);
    } else {
        serve_f32(model_path, context_length, config, rx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
    use fuel::lazy::LayerWeights;
    use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};
    use std::sync::{Arc, Mutex};

    #[test]
    fn eos_ids_from_reads_through_none_and_single() {
        assert_eq!(eos_ids_from(None), None);
        assert_eq!(eos_ids_from(Some(&LlamaEosToks::Single(2))), Some(vec![2]));
    }

    #[test]
    fn eos_ids_from_carries_the_full_multiple_set() {
        // fuel#307's whole point: a checkpoint with several stop tokens no
        // longer needs a different code path, it needs its full set carried
        // through unchanged.
        assert_eq!(
            eos_ids_from(Some(&LlamaEosToks::Multiple(vec![7, 8, 9]))),
            Some(vec![7, 8, 9])
        );
    }

    // Mirrors `scheduler_driver.rs`'s own `tiny_cfg`/`tiny_weights`/
    // `tiny_model` byte-for-byte (same seeded PRNG) -- kept local rather
    // than shared, same reasoning that module gives for its own copy of
    // fuel's `#[cfg(test)]`-private fixture: this file needs a real
    // `ServingDriver` constructed with direct (private-field) access, which
    // only a `#[cfg(test)] mod tests` inside THIS module can do.
    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of =
            |n: usize| -> Arc<[f32]> { Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
        let kv = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }

    fn tiny_model(seed: u32) -> LlamaModel {
        let cfg = tiny_cfg();
        LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, seed),
        }
    }

    /// A real `ServingDriver` wrapping a real tiny `LlamaModel`, generous
    /// enough on KV capacity that admission never fails for these tests'
    /// short prompts -- the point is `try_admit`'s prefix-detection wiring,
    /// not capacity edge cases (those belong to `scheduler_driver_paged.rs`'s
    /// own tests).
    fn serving_driver(model: &LlamaModel, prefix_sharing_enabled: bool) -> ServingDriver<'_> {
        serving_driver_with_blocks(model, prefix_sharing_enabled, 4096)
    }

    fn serving_driver_with_blocks(
        model: &LlamaModel,
        prefix_sharing_enabled: bool,
        num_blocks: usize,
    ) -> ServingDriver<'_> {
        let fuel_scheduler = FuelPagedSchedulerDriver::new(
            model,
            KvBudget {
                block_size: 2,
                num_blocks,
            },
            fuel::DType::F32,
            &fuel::Device::cpu(),
            PagedSchedulePolicy::AdmissionOrder,
            1,
            None,
        )
        .expect("tiny model's uniform per-head KV must satisfy ModelDims::from_model");
        ServingDriver {
            inner: ServingDriverInner::F32(fuel_scheduler),
            context_length: 4096,
            scheduler: Arc::new(Mutex::new(MemoryAwareScheduler::new(
                MemoryAwareConfig::default(),
            ))),
            sessions: std::collections::HashMap::new(),
            prefix_index: BlockPrefixIndexFor::new(2),
            retained: RetainedStore::new(0),
            pending_retention: std::collections::HashMap::new(),
            cache_stats: Arc::new(Mutex::new(crate::cache::PrefixCacheStats::default())),
            prefix_sharing_enabled,
            registered_prefixes: std::collections::HashMap::new(),
            prefix_shared_admissions: 0,
        }
    }

    /// The test `detect_and_record_prefix` exists for: admitting the SAME
    /// prompt twice must detect the second admission as a repeat (a real
    /// `BlockPrefixIndexFor::lookup` hit), while a genuinely different
    /// prompt must miss. If `try_admit`'s call into
    /// `detect_and_record_prefix` were ever deleted, `hits` would stay `0`
    /// and this test would fail -- the mutation this slice's board task
    /// asked to be provable against.
    #[test]
    fn repeated_prefix_is_detected_as_a_real_hit_distinct_prompt_is_a_real_miss() {
        let model = tiny_model(1);
        let mut d = serving_driver(&model, false);

        // Two whole blocks (block_size 2): long enough to be indexed.
        let shared_prompt = [1u32, 2, 3, 4];
        d.try_admit(&shared_prompt, 1, 0.0, None, Default::default())
            .expect("first admission must fit in the generous budget");
        assert_eq!(
            d.cache_stats.lock().unwrap().hits,
            0,
            "the FIRST admission of a prompt can never be a hit -- nothing was \
             indexed yet when its own lookup ran"
        );

        // Same tokens again: must hit against the first admission's record.
        d.try_admit(&shared_prompt, 1, 0.0, None, Default::default())
            .expect("second admission must also fit");
        let stats_after_repeat = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_after_repeat.hits, 1,
            "repeating an already-admitted prompt must register a real hit"
        );
        assert!(
            stats_after_repeat.entries >= 1,
            "a real record must have been indexed: {stats_after_repeat:?}"
        );

        // A genuinely different prompt: must miss, not hit.
        let distinct_prompt = [9u32, 10, 11, 12];
        d.try_admit(&distinct_prompt, 1, 0.0, None, Default::default())
            .expect("third admission must also fit");
        let stats_after_distinct = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_after_distinct.hits, stats_after_repeat.hits,
            "a distinct prompt must not be counted as a hit"
        );
        assert!(
            stats_after_distinct.misses >= 1,
            "a distinct prompt must register a real miss: {stats_after_distinct:?}"
        );
    }

    /// The test `handle_control`/`clear_prefix_cache` exist for (board PM
    /// task, 2026-10-22): fill the index with a real hit, clear it via a
    /// real `ControlMsg`, then prove the clear actually took -- the SAME
    /// prefix that was a hit a moment ago must now be a miss, and the
    /// shared stats must read zero. If `handle_control`'s match arm (or the
    /// `ClearPrefixCache` message itself) were ever skipped, the repeated
    /// prompt below would still hit and this test would fail.
    #[test]
    fn handle_control_clear_prefix_cache_makes_the_next_identical_prefix_a_real_miss() {
        let model = tiny_model(1);
        let mut d = serving_driver(&model, false);

        let prompt = [1u32, 2, 3, 4];
        d.try_admit(&prompt, 1, 0.0, None, Default::default())
            .expect("first admission must fit");
        d.try_admit(&prompt, 1, 0.0, None, Default::default())
            .expect("second admission must also fit");
        let stats_before_clear = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_before_clear.hits, 1,
            "sanity check: the index must actually be primed before testing the clear"
        );

        d.handle_control(ControlMsg::ClearPrefixCache);

        let stats_after_clear = d.cache_stats.lock().unwrap().clone();
        assert_eq!(stats_after_clear.hits, 0, "{stats_after_clear:?}");
        assert_eq!(stats_after_clear.misses, 0, "{stats_after_clear:?}");
        assert_eq!(stats_after_clear.entries, 0, "{stats_after_clear:?}");

        // The SAME prompt that was a hit before the clear must now miss --
        // proving the live index was actually rebuilt, not just the stats
        // mirror zeroed out from underneath it (which would desync the two
        // and make a later real hit under-report).
        d.try_admit(&prompt, 1, 0.0, None, Default::default())
            .expect("third admission must also fit");
        let stats_after_repeat = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_after_repeat.hits, 0,
            "the cleared index must not still recognize the pre-clear prefix: \
             {stats_after_repeat:?}"
        );
        assert_eq!(
            stats_after_repeat.misses, 1,
            "the post-clear admission must register as a genuine miss: \
             {stats_after_repeat:?}"
        );
    }

    /// Drives `id` to completion, returning its generated tokens in order.
    /// `step`/`reap_finished` are called on `d` directly -- the same
    /// `SchedulerDriver` trait methods `run_scheduled_jobs` itself uses --
    /// so this exercises the real admission-to-completion path, not a
    /// shortcut around it.
    fn drive_to_completion(
        d: &mut ServingDriver<'_>,
        id: fuel_inference::multi_session::SessionId,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();
        loop {
            for (tok_id, tok) in d.step().expect("step must not error") {
                if tok_id == id {
                    tokens.push(tok);
                }
            }
            if d.reap_finished()
                .into_iter()
                .any(|(fin_id, _)| fin_id == id)
            {
                return tokens;
            }
        }
    }

    /// Greedy completion of `prompt` on a driver that never shared anything:
    /// the cold reference every shared completion below is compared to.
    fn cold_completion(model: &LlamaModel, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut d = serving_driver(model, false);
        let id = d
            .try_admit(prompt, max_new, 0.0, None, Default::default())
            .expect("cold admission must succeed");
        drive_to_completion(&mut d, id)
    }

    /// A donor (`donor_prompt`) admitted and prefilled on a sharing-enabled
    /// driver, kept alive (large `max_new`) so later admissions can splice
    /// from it. Block size is 2 (`serving_driver`).
    fn driver_with_live_donor<'m>(
        model: &'m LlamaModel,
        donor_prompt: &[u32],
    ) -> (ServingDriver<'m>, fuel_inference::multi_session::SessionId) {
        let mut d = serving_driver(model, true);
        let donor = d
            .try_admit(donor_prompt, 32, 0.0, None, Default::default())
            .expect("donor must admit");
        // `prefill_pass` feeds a session's whole prompt in one `step()`.
        let _ = d.step().expect("donor's prefill tick must not error");
        (d, donor)
    }

    /// ROADMAP research-track item 1, slice 3's required test: a request
    /// sharing a prefix with a live donor produces the SAME completion as a
    /// cold admission AND really went through `add_session_sharing_prefix`.
    ///
    /// The second half is `prefix_shared_admissions`, not
    /// `registered_prefixes`: registration happens BEFORE Fuel decides
    /// whether to accept the share, so a registration alone cannot tell a
    /// splice from a silent cold fallback (which would also match the cold
    /// completion trivially). Prefill skipping itself is Fuel's contract
    /// (`add_session_sharing_prefix` prefills only `prompt[shared..]`);
    /// timing it would be flaky on CI.
    #[test]
    fn a_prefix_sharing_admission_really_splices_and_matches_the_cold_completion() {
        let model = tiny_model(1);
        // 5 tokens, block size 2: two whole blocks (4 tokens) shared, one
        // suffix token left to prefill.
        let prompt = [1u32, 2, 3, 4, 5];
        let (mut d, donor) = driver_with_live_donor(&model, &prompt);
        assert!(d.registered_prefixes.is_empty() && d.prefix_shared_admissions == 0);

        let sharer = d
            .try_admit(&prompt, 3, 0.0, None, Default::default())
            .expect("sharer must admit");
        assert_eq!(d.prefix_shared_admissions, 1, "must be a real splice");
        assert!(d.registered_prefixes.contains_key(&(donor, 2)));
        assert_eq!(
            drive_to_completion(&mut d, sharer),
            cold_completion(&model, &prompt, 3),
            "shared completion must be bit-identical to a cold admission"
        );
    }

    /// Admit `prompt`, run it to completion and let it depart, on a sharing driver whose retention budget is
    /// `budget` blocks. Returns the driver for a follow-up admission.
    fn driver_after_donor_finished<'m>(
        model: &'m LlamaModel,
        prompt: &[u32],
        budget: usize,
    ) -> ServingDriver<'m> {
        let mut d = serving_driver(model, true);
        d.retained = RetainedStore::new(budget);
        let donor = d
            .try_admit(prompt, 3, 0.0, None, Default::default())
            .expect("donor must admit");
        drive_to_completion(&mut d, donor);
        assert!(
            !d.sessions.contains_key(&donor),
            "the donor must have departed"
        );
        d
    }

    /// Slice 2 (docs/PREFIX-CACHE-DESIGN.md 3.2): with retention on, the donor's prompt prefix outlives it, so
    /// the next identical prompt takes a real splice and its completion is bit-identical to a cold one.
    #[test]
    fn a_retained_prefix_serves_a_second_request_after_its_donor_finished() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4, 5];
        let mut d = driver_after_donor_finished(&model, &prompt, 64);
        assert!(
            d.retained.pinned_blocks() > 0,
            "the prompt prefix must be retained"
        );
        let second = d
            .try_admit(&prompt, 3, 0.0, None, Default::default())
            .expect("second must admit");
        assert_eq!(
            d.prefix_shared_admissions, 1,
            "must be a real splice from the retained prefix"
        );
        assert_eq!(
            drive_to_completion(&mut d, second),
            cold_completion(&model, &prompt, 3),
            "completion through a retained prefix must be bit-identical to a cold admission"
        );
    }

    /// The control: the same sequence with retention 0 (today's behaviour) is a cold admit, so the test above
    /// cannot pass by the splice path being taken for some other reason.
    #[test]
    fn without_retention_the_second_request_after_the_donor_is_cold() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4, 5];
        let mut d = driver_after_donor_finished(&model, &prompt, 0);
        assert_eq!(d.retained.pinned_blocks(), 0);
        d.try_admit(&prompt, 3, 0.0, None, Default::default())
            .expect("second must admit");
        assert_eq!(d.prefix_shared_admissions, 0, "no retention: a cold admit");
    }

    /// A 6-block pool holding one retained 2-block prefix (4 free): a live request that needs 5 blocks only fits
    /// if that retained prefix is evicted. The donor generates 3 tokens: one that finishes IN its prefill step
    /// (`max_new` 1) has released its blocks before retention can register them, so it is never retained (a
    /// known limit, see the design note).
    fn driver_with_a_retained_prefix_in_a_tight_pool(model: &LlamaModel) -> ServingDriver<'_> {
        let mut d = serving_driver_with_blocks(model, true, 6);
        d.retained = RetainedStore::new(64);
        let id = d
            .try_admit(&[1, 2, 3, 4, 5], 3, 0.0, None, Default::default())
            .expect("donor admits");
        drive_to_completion(&mut d, id);
        assert_eq!(
            d.retained.chains(),
            1,
            "setup: the donor's prefix is retained"
        );
        assert!(
            d.free_blocks() < 5,
            "setup: the pool cannot take the live request as it is"
        );
        d
    }

    /// Tokens stay under the tiny model's 16-entry vocabulary: an out-of-range id fails the request on its own,
    /// which would make the control below fail for the wrong reason.
    const LIVE: [u32; 7] = [9, 8, 7, 6, 5, 4, 3];

    fn outcome_of(d: &mut ServingDriver<'_>, id: fuel_inference::multi_session::SessionId) -> bool {
        loop {
            let _ = d.step();
            if let Some((_, o)) = d.reap_finished().into_iter().find(|(i, _)| *i == id) {
                return matches!(o, FinishOutcome::Completed { .. });
            }
        }
    }

    /// Slice 3: a request that needs the space a retained prefix pins is admitted and completes, and the retained
    /// chain was released for it. A retained prefix is never the reason a live request fails.
    #[test]
    fn a_live_request_evicts_retained_prefixes_instead_of_failing() {
        let model = tiny_model(1);
        let mut d = driver_with_a_retained_prefix_in_a_tight_pool(&model);
        let id = d
            .try_admit(&LIVE, 3, 0.0, None, Default::default())
            .expect("live request admits");
        assert!(
            d.retained.pinned_blocks() < 2,
            "eviction is deepest-first and stops once the request fits: {} blocks still pinned",
            d.retained.pinned_blocks()
        );
        assert!(outcome_of(&mut d, id), "the live request must complete");
    }

    /// The control: the SAME tight pool without the pressure rule (straight to the inner driver) fails the live
    /// request. So the test above cannot pass for a reason other than the eviction.
    #[test]
    fn without_the_pressure_rule_the_same_request_fails_in_the_tight_pool() {
        let model = tiny_model(1);
        let mut d = driver_with_a_retained_prefix_in_a_tight_pool(&model);
        let id = d
            .plain_admit(&LIVE, 3, 0.0, None, Default::default())
            .expect("optimistic admission accepts it");
        assert!(
            !outcome_of(&mut d, id),
            "no eviction: the pool is exhausted mid-request"
        );
        assert_eq!(
            d.retained.pinned_blocks(),
            2,
            "the retained chain was untouched"
        );
    }

    /// A request that fits does not touch the retained prefixes.
    #[test]
    fn a_request_that_fits_leaves_the_retained_prefixes_alone() {
        let model = tiny_model(1);
        let mut d = driver_with_a_retained_prefix_in_a_tight_pool(&model);
        let id = d
            .try_admit(&[12, 13, 14], 1, 0.0, None, Default::default())
            .expect("admits");
        assert_eq!(
            d.retained.pinned_blocks(),
            2,
            "2 blocks needed, 4 free: nothing evicted"
        );
        assert!(outcome_of(&mut d, id));
    }

    /// `cache_clear` releases the retained prefixes and the next identical prompt is cold again.
    #[test]
    fn clearing_the_prefix_cache_drops_retained_prefixes() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4, 5];
        let mut d = driver_after_donor_finished(&model, &prompt, 64);
        d.clear_prefix_cache();
        assert_eq!(d.retained.chains(), 0);
        d.try_admit(&prompt, 3, 0.0, None, Default::default())
            .expect("must admit");
        assert_eq!(d.prefix_shared_admissions, 0, "cleared: cold");
    }

    /// A repeat of a prompt that is exactly block-aligned would match ALL
    /// its blocks, which Fuel refuses (no suffix left to sample from). The
    /// driver must share one block fewer rather than fall back cold.
    #[test]
    fn a_fully_block_aligned_repeat_shares_one_block_fewer_instead_of_falling_back() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4];
        let (mut d, donor) = driver_with_live_donor(&model, &prompt);

        let sharer = d
            .try_admit(&prompt, 3, 0.0, None, Default::default())
            .expect("sharer must admit");
        assert_eq!(d.prefix_shared_admissions, 1, "must be a real splice");
        assert!(d.registered_prefixes.contains_key(&(donor, 1)));
        assert!(!d.registered_prefixes.contains_key(&(donor, 2)));
        assert_eq!(
            drive_to_completion(&mut d, sharer),
            cold_completion(&model, &prompt, 3)
        );
    }

    /// A donor's longer registration must never be reused for a request that
    /// matched fewer blocks: the splice would graft KV for tokens that
    /// request does not contain. `BlockPrefixIndexFor::record` is
    /// last-writer-wins, so a live index rarely leaves ONE donor holding both
    /// lengths -- the registration table is driven directly here to put the
    /// donor in exactly that state, then the real fallback path is run with
    /// a 2-block hit.
    #[test]
    fn a_shorter_match_never_reuses_the_donors_longer_registration() {
        let model = tiny_model(1);
        let donor_prompt = [1u32, 2, 3, 4, 5, 6];
        let (mut d, donor) = driver_with_live_donor(&model, &donor_prompt);

        let long = d
            .ensure_prefix_registered(donor, 3)
            .expect("donor holds 3 whole blocks");
        let hit = super::super::policies::PrefixMatchFor {
            donor: PrefixSource::Live(donor),
            blocks: 2,
            shared_tokens: 4,
            remainder_tokens: 3,
        };
        let short_prompt = [1u32, 2, 3, 4, 9, 9, 9];
        let id = d
            .try_admit_sharing_or_fallback(
                Some(hit),
                &short_prompt,
                3,
                0.0,
                None,
                Default::default(),
            )
            .expect("short sharer must admit");

        assert_eq!(d.prefix_shared_admissions, 1, "must be a real splice");
        assert_eq!(d.registered_prefixes.len(), 2);
        assert_ne!(
            d.registered_prefixes[&(donor, 2)],
            long,
            "the 2-block match needs its own registration, not the 3-block one"
        );
        assert_eq!(
            drive_to_completion(&mut d, id),
            cold_completion(&model, &short_prompt, 3),
            "the short sharer must see only its own 2 matched blocks"
        );
    }

    /// F10 L5 at the engine: a hybrid model's effective setting is OFF, so the
    /// donor-then-identical-prompt scenario that splices on a non-hybrid must
    /// not splice. The control arm runs the same scenario with the non-hybrid
    /// setting and must splice, so a driver that cannot share would not pass
    /// this test vacuously.
    #[test]
    fn a_hybrid_model_never_takes_the_sharing_admission_path() {
        use super::super::memory_shape::effective_prefix_sharing;
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4, 5];
        let shared_admissions = |hybrid: bool| {
            let mut d = serving_driver(&model, effective_prefix_sharing(true, hybrid));
            d.try_admit(&prompt, 3, 0.0, None, Default::default())
                .expect("donor must admit");
            let _ = d.step().expect("donor's prefill tick must not error");
            d.try_admit(&prompt, 3, 0.0, None, Default::default())
                .expect("repeat must admit");
            d.prefix_shared_admissions
        };
        assert_eq!(shared_admissions(false), 1, "control: non-hybrid splices");
        assert_eq!(shared_admissions(true), 0, "hybrid must not splice");
    }

    /// The companion half of the slice 3 requirement: with the switch OFF
    /// (the default for every construction site today), behavior must be
    /// byte-identical to before slice 3 existed. Runs the EXACT scenario
    /// that would trigger sharing if the flag were on (a donor, then an
    /// identical-prompt second admission) and asserts two things the
    /// on-path test above does NOT need to: `registered_prefixes` never
    /// gains an entry (the new pool-level machinery is never touched at
    /// all when the flag is off, not merely "touched but declines to
    /// share"), and the second admission's completion is identical to an
    /// admission of that prompt on a driver that never saw the donor at
    /// all -- proving the mere PRESENCE of a detected hit changes nothing
    /// observable when `prefix_sharing_enabled` is false.
    #[test]
    fn prefix_sharing_disabled_is_byte_identical_to_detection_only_behavior() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4];
        let max_new = 3;

        let mut off_driver = serving_driver(&model, false);
        let _donor_id = off_driver
            .try_admit(&prompt, max_new, 0.0, None, Default::default())
            .expect("donor must admit");
        let _ = off_driver
            .step()
            .expect("donor's prefill tick must not error");

        let repeat_id = off_driver
            .try_admit(&prompt, max_new, 0.0, None, Default::default())
            .expect("identical-prompt admission must still succeed");
        assert!(
            off_driver.registered_prefixes.is_empty(),
            "the off path must never call register_prefix at all, not even once -- the \
             pool-level machinery this slice added must stay completely untouched"
        );
        let repeat_tokens = drive_to_completion(&mut off_driver, repeat_id);

        let mut fresh_driver = serving_driver(&model, false);
        let fresh_id = fresh_driver
            .try_admit(&prompt, max_new, 0.0, None, Default::default())
            .expect("a driver with no donor at all must admit the same prompt identically");
        let fresh_tokens = drive_to_completion(&mut fresh_driver, fresh_id);

        assert_eq!(
            repeat_tokens, fresh_tokens,
            "a detected-but-unexploited hit must not change the completion at all when \
             prefix sharing is disabled"
        );
    }
}
