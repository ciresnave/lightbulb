// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{InFlight, pull_response};
use crate::api::admin;
use crate::api::auth_middleware::ApiKeyInfo;
use crate::api::tests::bare_state;
use crate::model_fuel::pull::parse_spec;
use crate::model_fuel::pull::tests::{hub_for, serve_hub};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;

async fn post(
    state: crate::api::AppState,
    role: Option<&str>,
    body: &str,
) -> (StatusCode, serde_json::Value) {
    let mut req = Request::builder()
        .method("POST")
        .uri("/v1/lightbulb/admin/models/pull")
        .header("content-type", "application/json");
    if let Some(role) = role {
        req = req.extension(ApiKeyInfo {
            api_key_id: uuid::Uuid::new_v4(),
            role: role.to_string(),
        });
    }
    let resp = admin::routes()
        .with_state(state)
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

#[test]
fn a_pull_reports_each_file_and_a_second_pull_finds_it_present() {
    let (url, _srv) = serve_hub(b"GGUFdata", b"GGUFdata");
    let dir = tempfile::tempdir().unwrap();
    let hub = hub_for(&url, dir.path());
    let spec = parse_spec("o/n:Q8_0").unwrap();
    let (status, body) = pull_response(&hub, dir.path(), &spec, &|_| None);
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["files"][0]["status"], "downloaded");
    let (_, again) = pull_response(&hub, dir.path(), &spec, &|_| None);
    assert_eq!(again["files"][0]["status"], "present");
}

#[test]
fn wrong_quant_small_disk_and_dead_hub_are_404_507_502() {
    let (url, _srv) = serve_hub(b"GGUFdata", b"GGUFdata");
    let dir = tempfile::tempdir().unwrap();
    let hub = hub_for(&url, dir.path());
    let (s, _) = pull_response(&hub, dir.path(), &parse_spec("o/n:Q2_K").unwrap(), &|_| {
        None
    });
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _) = pull_response(&hub, dir.path(), &parse_spec("o/n:Q8_0").unwrap(), &|_| {
        Some(1)
    });
    assert_eq!(s, StatusCode::INSUFFICIENT_STORAGE);
    assert!(
        !dir.path().join("m-Q8_0.gguf").exists(),
        "a refused pull downloads nothing"
    );

    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let dead_hub = hub_for(&dead, dir.path());
    let (s, _) = pull_response(&dead_hub, dir.path(), &parse_spec("o/n").unwrap(), &|_| {
        None
    });
    assert_eq!(s, StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn bad_spec_no_models_dir_and_a_repo_already_in_flight_are_400_501_409() {
    let state = bare_state(Some("."), false);
    let (s, _) = post(state.clone(), Some("admin"), r#"{"model":"nonsense"}"#).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);

    let none = bare_state(None, true);
    let (s, _) = post(none, Some("admin"), r#"{"model":"a/b"}"#).await;
    assert_eq!(s, StatusCode::NOT_IMPLEMENTED);

    let held = InFlight::take("inflight/test").expect("free");
    assert!(
        InFlight::take("inflight/test").is_none(),
        "second is refused"
    );
    let (s, _) = post(state.clone(), Some("admin"), r#"{"model":"inflight/test"}"#).await;
    assert_eq!(s, StatusCode::CONFLICT);
    drop(held);
    assert!(InFlight::take("inflight/test").is_some(), "freed on drop");
}

#[tokio::test]
async fn a_non_admin_key_and_no_credential_are_refused_and_download_nothing() {
    let state = bare_state(Some("."), false);
    let (s, _) = post(state.clone(), Some("user"), r#"{"model":"a/b"}"#).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, _) = post(state, None, r#"{"model":"a/b"}"#).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}
