// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use crate::api::registry::ModelEntry;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A loader whose "runner" is a thread that ends when its sender is dropped, with sizes
/// from a table; `loads` counts how many times a model was actually started.
struct Fake {
    sizes: BTreeMap<String, u64>,
    loads: AtomicUsize,
}

impl ModelLoader for Fake {
    fn needed_bytes(&self, name: &str) -> Result<u64, LoadError> {
        self.sizes
            .get(name)
            .copied()
            .ok_or_else(|| LoadError::NotFound(name.into()))
    }

    fn load(&self, registry: &ModelRegistry, name: &str) -> Result<(), LoadError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<crate::engine::model_runner::InferenceJob>(1);
        let join = std::thread::spawn(move || while rx.recv().is_ok() {});
        registry.set(
            name,
            ModelEntry {
                inference_tx: Some(tx),
                join: Some(Arc::new(Mutex::new(Some(join)))),
                ..ModelEntry::default()
            },
        );
        Ok(())
    }
}

/// A device of `capacity` bytes. `ghost` = (model, bytes): after that model is unloaded,
/// this many bytes stay unavailable (the #298 behaviour: an unload frees less than the
/// estimate). Free memory is computed from what the registry holds, so it moves only
/// when a model is really unloaded.
fn rig(
    sizes: &[(&str, u64)],
    capacity: u64,
    ghost: Option<(&'static str, u64)>,
) -> (ModelManager, Arc<Fake>) {
    let sizes: BTreeMap<String, u64> = sizes.iter().map(|(n, b)| (n.to_string(), *b)).collect();
    let fake = Arc::new(Fake {
        sizes: sizes.clone(),
        loads: AtomicUsize::new(0),
    });
    let registry = ModelRegistry::default();
    let reg = registry.clone();
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let free: FreeBytes = Arc::new(move || {
        let now = reg.names();
        let mut seen = seen.lock().unwrap();
        for n in &now {
            if !seen.contains(n) {
                seen.push(n.clone());
            }
        }
        let used: u64 = now.iter().map(|n| sizes[n]).sum();
        let stuck = match ghost {
            Some((g, b)) if seen.iter().any(|s| s == g) && !now.iter().any(|n| n == g) => b,
            _ => 0,
        };
        capacity - used - stuck
    });
    let mgr = ModelManager::new(registry, fake.clone(), free, Duration::from_secs(5));
    (mgr, fake)
}

#[test]
fn an_idle_model_can_be_unloaded_on_request_and_is_loaded_again_when_asked_for() {
    let (m, fake) = rig(&[("a", 4)], 10, None);
    drop(m.acquire("a").unwrap());
    let out = m.unload("a").unwrap();
    assert!(out.joined);
    assert!(names(&m).is_empty());
    drop(m.acquire("a").unwrap());
    assert_eq!(fake.loads.load(Ordering::SeqCst), 2);
}

#[test]
fn unloading_a_model_that_is_not_loaded_or_is_busy_is_refused() {
    let (m, _) = rig(&[("a", 4), ("b", 4)], 10, None);
    assert_eq!(m.unload("a").unwrap_err(), UnloadError::NotLoaded);
    let lease = m.acquire("a").unwrap();
    assert_eq!(m.unload("a").unwrap_err(), UnloadError::Busy);
    assert_eq!(
        names(&m),
        vec!["a".to_string()],
        "a busy model stays loaded"
    );
    drop(lease);
    assert!(m.unload("a").is_ok());
}

fn names(m: &ModelManager) -> Vec<String> {
    m.registry.names()
}

#[test]
fn a_loaded_model_is_leased_without_loading_again() {
    let (m, fake) = rig(&[("a", 4)], 10, None);
    drop(m.acquire("a").unwrap());
    drop(m.acquire("a").unwrap());
    assert_eq!(fake.loads.load(Ordering::SeqCst), 1);
    assert_eq!(names(&m), vec!["a"]);
}

#[test]
fn a_model_that_fits_is_loaded_next_to_the_others() {
    let (m, _) = rig(&[("a", 4), ("b", 4)], 10, None);
    drop(m.acquire("a").unwrap());
    drop(m.acquire("b").unwrap());
    assert_eq!(names(&m), vec!["a", "b"], "co-residency: nothing evicted");
}

#[test]
fn an_unknown_model_is_not_found_and_nothing_is_loaded() {
    let (m, fake) = rig(&[("a", 4)], 10, None);
    assert_eq!(
        m.acquire("nope").err(),
        Some(LoadError::NotFound("nope".into()))
    );
    assert_eq!(fake.loads.load(Ordering::SeqCst), 0);
}

#[test]
fn the_least_recently_used_idle_model_is_evicted_first() {
    let (m, _) = rig(&[("a", 4), ("b", 4), ("c", 6)], 10, None);
    drop(m.acquire("a").unwrap());
    drop(m.acquire("b").unwrap());
    drop(m.acquire("a").unwrap()); // a is now newer than b
    drop(m.acquire("c").unwrap());
    assert_eq!(names(&m), vec!["a", "c"], "b was the oldest idle one");
}

#[test]
fn free_memory_is_read_again_after_each_unload() {
    // c needs 6 of 10; a and b hold 4 each (2 free). Unloading a leaves 3 bytes stuck
    // (the #298 shape), so free is 10-4-3 = 3, not the 6 the sizes promise: b must go too.
    let (m, _) = rig(&[("a", 4), ("b", 4), ("c", 6)], 10, Some(("a", 3)));
    drop(m.acquire("a").unwrap());
    drop(m.acquire("b").unwrap());
    drop(m.acquire("c").unwrap());
    assert_eq!(names(&m), vec!["c"]);
}

#[test]
fn control_without_the_stuck_bytes_one_eviction_is_enough() {
    let (m, _) = rig(&[("a", 4), ("b", 4), ("c", 6)], 10, None);
    drop(m.acquire("a").unwrap());
    drop(m.acquire("b").unwrap());
    drop(m.acquire("c").unwrap());
    assert_eq!(
        names(&m),
        vec!["b", "c"],
        "with no stuck bytes the estimate holds, so only a goes; the test above needs b too"
    );
}

#[test]
fn a_model_with_a_request_in_flight_is_never_evicted() {
    let (m, _) = rig(&[("a", 4), ("b", 4), ("c", 6)], 10, None);
    let busy = m.acquire("a").unwrap(); // oldest, but a request holds it
    drop(m.acquire("b").unwrap());
    drop(m.acquire("c").unwrap());
    assert_eq!(names(&m), vec!["a", "c"], "b went, the busy older a stayed");
    drop(busy);
}

#[test]
fn when_only_busy_models_stand_in_the_way_the_shortfall_is_reported() {
    let (m, fake) = rig(&[("a", 8), ("c", 6)], 10, None);
    let _busy = m.acquire("a").unwrap();
    let err = m.acquire("c").err().expect("no room");
    assert!(matches!(err, LoadError::NoRoom(s) if s.needed == 6 && s.evictable == 0));
    assert_eq!(names(&m), vec!["a"], "nothing was evicted");
    assert_eq!(fake.loads.load(Ordering::SeqCst), 1, "c was not started");
}

#[test]
fn dropping_a_lease_makes_the_model_evictable_again() {
    let (m, _) = rig(&[("a", 8), ("c", 6)], 10, None);
    let lease = m.acquire("a").unwrap();
    assert!(m.acquire("c").is_err());
    drop(lease);
    drop(m.acquire("c").unwrap());
    assert_eq!(names(&m), vec!["c"]);
}

#[test]
fn an_adopted_model_is_tracked_like_a_loaded_one() {
    let (m, fake) = rig(&[("a", 8), ("c", 6)], 10, None);
    fake.load(&m.registry, "a").unwrap(); // started outside the manager
    m.adopt("a", 8);
    drop(m.acquire("c").unwrap());
    assert_eq!(names(&m), vec!["c"], "the adopted idle model was evicted");
}
