// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use crate::api::manager::{FreeBytes, ModelLoader, ModelManager};
use crate::api::registry::{ModelEntry, ModelRegistry};
use crate::api::tests::bare_state;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

/// Models "a" (8 bytes) and "c" (6) on a 10-byte device: they cannot both be resident.
struct Fake;

fn size(name: &str) -> Option<u64> {
    BTreeMap::from([("a", 8u64), ("c", 6), ("d", 1)])
        .get(name)
        .copied()
}

impl ModelLoader for Fake {
    fn needed_bytes(&self, name: &str) -> Result<u64, LoadError> {
        size(name).ok_or_else(|| LoadError::NotFound(name.into()))
    }
    fn load(&self, registry: &ModelRegistry, name: &str) -> Result<(), LoadError> {
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<crate::engine::model_runner::InferenceJob>(1);
        let join = std::thread::spawn(move || while rx.recv().is_ok() {});
        registry.set(
            name,
            ModelEntry {
                inference_tx: Some(tx),
                join: Some(Arc::new(std::sync::Mutex::new(Some(join)))),
                ..ModelEntry::default()
            },
        );
        Ok(())
    }
}

fn managed_state() -> AppState {
    let mut state = bare_state(Some("."), false);
    state.models = ModelRegistry::default();
    let reg = state.models.clone();
    let free: FreeBytes =
        Arc::new(move || 10 - reg.names().iter().map(|n| size(n).unwrap()).sum::<u64>());
    state.manager = Some(Arc::new(ModelManager::new(
        state.models.clone(),
        Arc::new(Fake),
        free,
        Duration::from_secs(5),
    )));
    state
}

fn loaded(r: Result<Scoped, Rejection>) -> Scoped {
    match r {
        Ok(s) => s,
        Err(e) => panic!("expected the model to resolve, got {}", e.status()),
    }
}

fn status(r: Result<Scoped, Rejection>) -> StatusCode {
    r.err().expect("expected a rejection").status()
}

#[tokio::test]
async fn without_a_manager_only_the_loaded_model_is_accepted() {
    let state = bare_state(None, true);
    let ok = scope_to_model(&state, &state.config.default_model).await;
    assert!(matches!(ok, Ok(Scoped { lease: None, .. })));
    assert_eq!(
        status(scope_to_model(&state, "something-else").await),
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn without_a_manager_and_nothing_loaded_the_handler_decides() {
    let state = bare_state(None, false);
    assert!(scope_to_model(&state, "anything").await.is_ok());
}

#[tokio::test]
async fn a_known_model_is_loaded_on_demand_and_the_state_is_scoped_to_it() {
    let state = managed_state();
    let scoped = loaded(scope_to_model(&state, "d").await);
    assert!(scoped.lease.is_some());
    assert_eq!(scoped.state.models.names(), vec!["d"]);
    assert!(scoped.state.model().inference_tx.is_some());
}

#[tokio::test]
async fn an_unknown_model_is_a_400_once_something_is_loaded() {
    let state = managed_state();
    drop(scope_to_model(&state, "d").await);
    assert_eq!(
        status(scope_to_model(&state, "nope").await),
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn no_room_is_a_503_with_retry_after_and_the_busy_model_survives() {
    let state = managed_state();
    let busy = loaded(scope_to_model(&state, "a").await);
    let refused = scope_to_model(&state, "c").await.err().expect("no room");
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused.headers().get(header::RETRY_AFTER).unwrap(), "5");
    assert_eq!(
        state.models.names(),
        vec!["a"],
        "the busy model was not evicted"
    );
    drop(busy);
    // Control: once the lease is gone the same request succeeds by evicting a.
    assert!(scope_to_model(&state, "c").await.is_ok());
    assert_eq!(state.models.names(), vec!["c"]);
}

#[tokio::test]
async fn a_held_response_pins_its_model_until_the_body_is_dropped() {
    let state = managed_state();
    let scoped = loaded(scope_to_model(&state, "a").await);
    let response = hold((StatusCode::OK, "streamed").into_response(), scoped.lease);
    assert_eq!(
        status(scope_to_model(&state, "c").await),
        StatusCode::SERVICE_UNAVAILABLE,
        "the response is alive, so a is pinned"
    );
    drop(response);
    assert!(
        scope_to_model(&state, "c").await.is_ok(),
        "released with the body"
    );
}
