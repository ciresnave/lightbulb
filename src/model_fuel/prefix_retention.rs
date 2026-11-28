// SPDX-License-Identifier: MIT OR Apache-2.0
//! Prefix reuse after the donor finishes, slice 1 (design: `docs/PREFIX-CACHE-DESIGN.md`, sections 3.1 and 3.3):
//! the pure bookkeeping, with no Fuel types, so every rule can be tested without a model.
//!
//! - [`PrefixSource`]: what an index entry names. A `Live` donor goes away when its session does
//!   (`forget_donor`); a `Retained` one stays until it is evicted.
//! - [`RetainedStore`]: bounded, LRU over whole chains. A chain is the set of depths registered from one prompt
//!   (`PrefixId` per depth, the payload `P`). Eviction takes the least recently hit chain and removes its DEEPEST
//!   depth first, so what remains of a chain is always a prefix of it and the index's stop-at-first-miss
//!   `lookup` stays correct.
//!
//! Accounting here is `sum over chains of the deepest depth`: an UPPER bound on pinned pool blocks (chains that
//! share a leading prefix share its blocks). The real figure is read back from the pool in a later slice; this
//! number is what the budget is checked against until then, and it errs toward evicting early, never late.

use std::collections::HashMap;

/// What an index entry points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefixSource<S, R> {
    /// A running session's blocks; dropped when the session ends.
    Live(S),
    /// A retained registration; dropped only by eviction.
    Retained(R),
}

/// Never reused, so an evicted chain's id cannot name a later chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RetainedId(pub u64);

/// Why a chain was not retained.
#[derive(Debug, PartialEq, Eq)]
pub enum RetainError {
    /// Budget 0: retention is off.
    Disabled,
    /// The chain alone is larger than the budget. Nothing was evicted for it.
    TooLarge { blocks: usize, budget: usize },
    /// No depths given.
    Empty,
}

struct Chain<P> {
    /// `(depth in blocks, payload)`, ascending by depth.
    depths: Vec<(usize, P)>,
    last_used: u64,
}

impl<P> Chain<P> {
    fn blocks(&self) -> usize {
        self.depths.last().map_or(0, |d| d.0)
    }
}

/// An eviction the caller must carry out (`release_prefix` on `payload`).
#[derive(Debug, PartialEq, Eq)]
pub struct Evicted<P> {
    pub chain: RetainedId,
    pub depth: usize,
    pub payload: P,
}

pub struct RetainedStore<P> {
    budget: usize,
    clock: u64,
    next: u64,
    chains: HashMap<RetainedId, Chain<P>>,
}

impl<P> RetainedStore<P> {
    /// `budget_blocks == 0` disables retention.
    pub fn new(budget_blocks: usize) -> Self {
        Self {
            budget: budget_blocks,
            clock: 0,
            next: 0,
            chains: HashMap::new(),
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn chains(&self) -> usize {
        self.chains.len()
    }

    /// Upper bound on pool blocks pinned (see the module doc).
    pub fn pinned_blocks(&self) -> usize {
        self.chains.values().map(Chain::blocks).sum()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Retain a chain of registered depths (any order). Returns its id and the evictions made to fit it; the
    /// caller releases each. An impossible chain evicts nothing.
    pub fn insert(
        &mut self,
        mut depths: Vec<(usize, P)>,
    ) -> Result<(RetainedId, Vec<Evicted<P>>), RetainError> {
        if self.budget == 0 {
            return Err(RetainError::Disabled);
        }
        depths.sort_by_key(|d| d.0);
        let blocks = depths.last().ok_or(RetainError::Empty)?.0;
        if blocks > self.budget {
            return Err(RetainError::TooLarge {
                blocks,
                budget: self.budget,
            });
        }
        let mut evicted = Vec::new();
        while self.pinned_blocks() + blocks > self.budget {
            match self.evict_one() {
                Some(e) => evicted.push(e),
                None => break,
            }
        }
        let id = RetainedId(self.next);
        self.next += 1;
        let last_used = self.tick();
        self.chains.insert(id, Chain { depths, last_used });
        Ok((id, evicted))
    }

    /// A lookup hit this chain: it becomes the most recently used. Unknown ids are ignored.
    pub fn touch(&mut self, id: RetainedId) {
        let t = self.tick();
        if let Some(c) = self.chains.get_mut(&id) {
            c.last_used = t;
        }
    }

    /// Evict the deepest depth of the least recently used chain; the chain goes when its last depth does.
    pub fn evict_one(&mut self) -> Option<Evicted<P>> {
        let id = *self
            .chains
            .iter()
            .min_by_key(|(id, c)| (c.last_used, **id))
            .map(|(id, _)| id)?;
        let chain = self.chains.get_mut(&id)?;
        let (depth, payload) = chain.depths.pop()?;
        if chain.depths.is_empty() {
            self.chains.remove(&id);
        }
        Some(Evicted {
            chain: id,
            depth,
            payload,
        })
    }

    /// Everything, for `cache_clear` and model unload. LRU order, deepest first within a chain.
    pub fn drain(&mut self) -> Vec<Evicted<P>> {
        let mut out = Vec::new();
        while let Some(e) = self.evict_one() {
            out.push(e);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_fuel::policies::BlockPrefixIndexFor;

    fn chain(depths: &[usize]) -> Vec<(usize, String)> {
        depths.iter().map(|d| (*d, format!("p{d}"))).collect()
    }

    #[test]
    fn budget_is_enforced_by_evicting_the_least_recently_used_chain() {
        let mut s = RetainedStore::new(10);
        let (a, _) = s.insert(chain(&[2, 4])).unwrap();
        let (b, _) = s.insert(chain(&[3, 6])).unwrap();
        assert_eq!(s.pinned_blocks(), 10);
        s.touch(a); // b is now the least recently used
        let (_c, ev) = s.insert(chain(&[5])).unwrap();
        assert!(
            s.pinned_blocks() <= 10,
            "a store that never evicts fails here"
        );
        assert!(
            ev.iter().all(|e| e.chain == b),
            "evicted the LRU chain, not {ev:?}"
        );
        assert!(s.chains.contains_key(&a), "the touched chain survived");
    }

    #[test]
    fn eviction_is_deepest_first_so_a_chain_stays_prefix_closed() {
        let mut s = RetainedStore::new(100);
        s.insert(chain(&[2, 4, 8])).unwrap();
        let depths: Vec<usize> = s.drain().into_iter().map(|e| e.depth).collect();
        assert_eq!(
            depths,
            vec![8, 4, 2],
            "shallow-first would leave a hole lookup cannot cross"
        );
    }

    #[test]
    fn partial_eviction_leaves_the_shallow_depths_and_the_payloads_to_release() {
        let mut s = RetainedStore::new(8);
        s.insert(chain(&[2, 8])).unwrap();
        let e = s.evict_one().unwrap();
        assert_eq!((e.depth, e.payload.as_str()), (8, "p8"));
        assert_eq!(s.pinned_blocks(), 2);
        assert_eq!(s.chains(), 1);
    }

    #[test]
    fn disabled_oversize_and_empty_are_typed_and_evict_nothing() {
        assert_eq!(
            RetainedStore::<String>::new(0)
                .insert(chain(&[1]))
                .unwrap_err(),
            RetainError::Disabled
        );
        let mut s = RetainedStore::new(4);
        s.insert(chain(&[4])).unwrap();
        assert_eq!(
            s.insert(chain(&[5])).unwrap_err(),
            RetainError::TooLarge {
                blocks: 5,
                budget: 4
            }
        );
        assert_eq!(s.insert(Vec::new()).unwrap_err(), RetainError::Empty);
        assert_eq!(
            s.pinned_blocks(),
            4,
            "an impossible chain must not evict the ones already held"
        );
    }

    #[test]
    fn ids_are_never_reused() {
        let mut s = RetainedStore::new(2);
        let (a, _) = s.insert(chain(&[2])).unwrap();
        let (b, ev) = s.insert(chain(&[2])).unwrap();
        assert_eq!(ev.len(), 1);
        assert_ne!(a, b);
    }

    #[test]
    fn a_retained_entry_survives_forget_donor_and_a_live_one_does_not() {
        type Src = PrefixSource<u32, RetainedId>;
        let toks: Vec<u32> = (0..8).collect();
        let mut idx = BlockPrefixIndexFor::<Src>::new(4);
        let live = PrefixSource::Live(7);
        let kept = PrefixSource::Retained(RetainedId(1));
        idx.record(&toks, live);
        assert_eq!(idx.lookup(&toks).unwrap().donor, live);
        // Finish the donor: its Live entries go.
        assert_eq!(idx.forget_donor(live), 2);
        assert!(
            idx.lookup(&toks).is_none(),
            "a live-only index is cold after the donor leaves"
        );
        // The same prompt recorded against a retained source is still found.
        idx.record(&toks, kept);
        idx.forget_donor(live);
        let m = idx
            .lookup(&toks)
            .expect("retained entries outlive the donor");
        assert_eq!((m.donor, m.blocks), (kept, 2));
    }

    #[test]
    fn the_key_carries_the_seed_so_other_adapters_and_models_never_hit() {
        let toks: Vec<u32> = (0..8).collect();
        let keys = |seed: &[u8]| BlockPrefixIndexFor::<u32>::with_seed(4, seed).chain_keys(&toks);
        let base = keys(b"model=m1;adapter=none");
        assert_eq!(base.len(), 2);
        assert_eq!(
            base,
            keys(b"model=m1;adapter=none"),
            "control: the same seed gives the same keys"
        );
        for other in [&b"model=m1;adapter=lora-x"[..], b"model=m2;adapter=none"] {
            let k = keys(other);
            assert!(
                k.iter().all(|h| !base.contains(h)),
                "no key may be shared across seeds"
            );
        }
        // The empty seed is the historical key, so nothing recorded before this change is orphaned.
        assert_eq!(
            keys(b""),
            BlockPrefixIndexFor::<u32>::new(4).chain_keys(&toks)
        );
    }
}
