// SPDX-License-Identifier: MIT OR Apache-2.0
//! Operator model download (ROADMAP F1): `POST /v1/lightbulb/admin/models/pull` with `{"model": "repo[:quant]"}` fetches
//! one GGUF, or a complete split set, into the models directory with a hash check (`model_fuel::pull`). Merged into
//! `admin::routes`, so it sits behind the same admin-role check as every other `/v1/lightbulb/admin/*` route.
//!
//! The request is answered when the download ends: it holds the connection for as long as the download takes and reports
//! no progress (the CLI, `lightbulb-cli pull`, prints it). Two pulls of one repo at once are refused with 409.

use super::AppState;
use crate::model_fuel::pull::{
    FileOutcome, InsufficientSpace, PullSpec, Selection, available_space, hub_in, parse_spec, pull,
};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use mlmf_source_hub::HubSource;
use std::path::Path;
use std::sync::Mutex;

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/lightbulb/admin/models/pull", post(pull_model))
}

#[derive(serde::Deserialize)]
struct PullRequest {
    model: String,
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

/// Repos being pulled right now: two pulls of one repo would race on its partial files.
static PULLING: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct InFlight(String);

impl InFlight {
    fn take(repo: &str) -> Option<Self> {
        let mut held = PULLING.lock().unwrap_or_else(|e| e.into_inner());
        if held.iter().any(|r| r == repo) {
            return None;
        }
        held.push(repo.to_string());
        Some(Self(repo.to_string()))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        let mut held = PULLING.lock().unwrap_or_else(|e| e.into_inner());
        held.retain(|r| r != &self.0);
    }
}

/// Run one pull and say what happened as a status and a body: 200 with each file's outcome; 404 when the repo has no `.gguf`
/// the spec names; 507 when the disk is too small; 502 for anything the Hub or the download did.
pub(crate) fn pull_response(
    hub: &HubSource,
    dir: &Path,
    spec: &PullSpec,
    free_space: &dyn Fn(&Path) -> Option<u64>,
) -> (StatusCode, serde_json::Value) {
    match pull(hub, spec, dir, free_space, &mut |_, _| {}) {
        Ok(outcomes) => {
            let files: Vec<_> = outcomes
                .iter()
                .map(|o| match o {
                    FileOutcome::Present(p) => serde_json::json!({"path": p, "status": "present"}),
                    FileOutcome::Downloaded(p) => {
                        serde_json::json!({"path": p, "status": "downloaded"})
                    }
                })
                .collect();
            (
                StatusCode::OK,
                serde_json::json!({ "repo": spec.repo, "files": files }),
            )
        }
        Err(e) => {
            let status = if e.downcast_ref::<Selection>().is_some() {
                StatusCode::NOT_FOUND
            } else if e.downcast_ref::<InsufficientSpace>().is_some() {
                StatusCode::INSUFFICIENT_STORAGE
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, serde_json::json!({ "error": e.to_string() }))
        }
    }
}

/// `POST /v1/lightbulb/admin/models/pull`: 400 for a malformed spec, 501 when the server has no models directory, 409 when
/// that repo is already being pulled; otherwise see [`pull_response`].
async fn pull_model(State(state): State<AppState>, Json(req): Json<PullRequest>) -> Response {
    let Some(dir) = state.config.models_dir.clone() else {
        return error(
            StatusCode::NOT_IMPLEMENTED,
            "this server has no models directory, so there is nowhere to download to".into(),
        );
    };
    let spec = match parse_spec(&req.model) {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let Some(guard) = InFlight::take(&spec.repo) else {
        return error(
            StatusCode::CONFLICT,
            format!("{} is already being pulled", spec.repo),
        );
    };
    let joined = tokio::task::spawn_blocking(move || {
        let _held = guard;
        let dir = std::path::PathBuf::from(dir);
        pull_response(&hub_in(&dir), &dir, &spec, &available_space)
    })
    .await;
    match joined {
        Ok((status, body)) => (status, Json(body)).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, format!("pull task: {e}")),
    }
}

#[cfg(test)]
mod tests;
