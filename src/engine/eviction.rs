// SPDX-License-Identifier: MIT OR Apache-2.0
//! Which resident models to unload to make room for another (ROADMAP F2, P2).
//!
//! Pure logic, no GPU and no threads, so the order can be tested exhaustively
//! before any model is unloaded. The rule is the one the design note fixes
//! (`docs/superpowers/specs/2026-10-30-f2-multi-model-design.md`):
//!
//! * if the new model fits next to what is resident, evict nothing (co-residency);
//! * otherwise unload the **least recently used** models, oldest first, only as many
//!   as it takes;
//! * **never** one with a request in flight, whatever its age;
//! * if the idle ones together still do not make room, say so (the caller answers
//!   503 with `Retry-After`) rather than evict part of the way and fail anyway.
//!
//! "Used" is a caller-supplied logical tick (a counter bumped per admitted request),
//! not a clock, so ordering is deterministic and testable.

/// One loaded model, as the planner sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resident {
    pub name: String,
    /// Device bytes unloading it would free (the B9 estimate for that model).
    pub bytes: u64,
    /// Logical tick of its last admitted request; smaller is older.
    pub last_used: u64,
    /// Requests currently running on it. Non-zero means it cannot be evicted.
    pub in_flight: usize,
}

/// What to do before loading the new model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// It fits as things are: load it alongside.
    Fits,
    /// Unload these, in this order, then load it.
    Evict(Vec<String>),
}

/// The idle models cannot free enough.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CannotMakeRoom {
    pub needed: u64,
    pub free: u64,
    /// Bytes held by models that could be evicted (no request in flight).
    pub evictable: u64,
}

/// Decide what to evict so that `needed` bytes are free, given `free` now.
pub fn plan_eviction(
    resident: &[Resident],
    needed: u64,
    free: u64,
) -> Result<Plan, CannotMakeRoom> {
    if free >= needed {
        return Ok(Plan::Fits);
    }
    let mut idle: Vec<&Resident> = resident.iter().filter(|r| r.in_flight == 0).collect();
    // Oldest first; the name breaks a tie so the result never depends on input order.
    idle.sort_by(|a, b| {
        a.last_used
            .cmp(&b.last_used)
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut available = free;
    let mut evict = Vec::new();
    for r in &idle {
        if available >= needed {
            break;
        }
        available = available.saturating_add(r.bytes);
        evict.push(r.name.clone());
    }
    if available >= needed {
        return Ok(Plan::Evict(evict));
    }
    Err(CannotMakeRoom {
        needed,
        free,
        evictable: idle.iter().fold(0u64, |s, r| s.saturating_add(r.bytes)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(name: &str, bytes: u64, last_used: u64, in_flight: usize) -> Resident {
        Resident {
            name: name.into(),
            bytes,
            last_used,
            in_flight,
        }
    }

    #[test]
    fn a_model_that_fits_evicts_nothing() {
        let r = [m("a", 4, 1, 0)];
        assert_eq!(plan_eviction(&r, 3, 3), Ok(Plan::Fits));
        assert_eq!(plan_eviction(&[], 3, 8), Ok(Plan::Fits));
        assert_eq!(
            plan_eviction(&r, 0, 0),
            Ok(Plan::Fits),
            "needing nothing fits"
        );
    }

    #[test]
    fn the_least_recently_used_goes_first_whatever_the_input_order() {
        let newest_first = [m("new", 4, 9, 0), m("mid", 4, 5, 0), m("old", 4, 1, 0)];
        assert_eq!(
            plan_eviction(&newest_first, 4, 0),
            Ok(Plan::Evict(vec!["old".into()]))
        );
        let oldest_first = [m("old", 4, 1, 0), m("mid", 4, 5, 0), m("new", 4, 9, 0)];
        assert_eq!(
            plan_eviction(&oldest_first, 4, 0),
            Ok(Plan::Evict(vec!["old".into()]))
        );
    }

    #[test]
    fn it_stops_as_soon_as_enough_is_free_and_not_before() {
        let r = [m("a", 3, 1, 0), m("b", 3, 2, 0), m("c", 3, 3, 0)];
        // 4 needed, 0 free: a (3) is not enough, a+b (6) is.
        assert_eq!(
            plan_eviction(&r, 4, 0),
            Ok(Plan::Evict(vec!["a".into(), "b".into()]))
        );
        // Free memory counts: 2 free + a (3) reaches 4, so b stays.
        assert_eq!(plan_eviction(&r, 4, 2), Ok(Plan::Evict(vec!["a".into()])));
    }

    #[test]
    fn a_model_with_a_request_in_flight_is_never_evicted_however_old() {
        let r = [m("busy_old", 8, 1, 2), m("idle_new", 8, 9, 0)];
        assert_eq!(
            plan_eviction(&r, 8, 0),
            Ok(Plan::Evict(vec!["idle_new".into()])),
            "the older model is busy, so the newer idle one goes"
        );
    }

    #[test]
    fn when_the_idle_ones_are_not_enough_nothing_is_evicted_and_the_shortfall_is_reported() {
        let r = [m("busy", 8, 1, 1), m("idle", 3, 2, 0)];
        assert_eq!(
            plan_eviction(&r, 8, 1),
            Err(CannotMakeRoom {
                needed: 8,
                free: 1,
                evictable: 3
            })
        );
        // Control: the same models with the busy one idle can make room.
        let all_idle = [m("busy", 8, 1, 0), m("idle", 3, 2, 0)];
        assert!(matches!(plan_eviction(&all_idle, 8, 1), Ok(Plan::Evict(_))));
    }

    #[test]
    fn equal_ages_break_on_the_name_so_the_plan_is_deterministic() {
        let a = [m("b", 4, 5, 0), m("a", 4, 5, 0)];
        let b = [m("a", 4, 5, 0), m("b", 4, 5, 0)];
        assert_eq!(plan_eviction(&a, 4, 0), plan_eviction(&b, 4, 0));
        assert_eq!(plan_eviction(&a, 4, 0), Ok(Plan::Evict(vec!["a".into()])));
    }

    #[test]
    fn huge_sizes_do_not_overflow() {
        let r = [m("a", u64::MAX, 1, 0), m("b", u64::MAX, 2, 0)];
        assert_eq!(
            plan_eviction(&r, u64::MAX, 1),
            Ok(Plan::Evict(vec!["a".into()]))
        );
        let err = plan_eviction(&[m("a", u64::MAX, 1, 1), m("b", 1, 2, 0)], u64::MAX, 0);
        assert_eq!(
            err.unwrap_err().evictable,
            1,
            "the busy giant is not counted"
        );
    }
}
