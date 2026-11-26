// SPDX-License-Identifier: MIT OR Apache-2.0
//! Operator model download (ROADMAP F1): `POST /v1/lightbulb/admin/models/pull` with `{"model": "repo[:quant]"}` fetches
//! one GGUF, or a complete split set, into the models directory with a hash check (`model_fuel::pull`). Merged into
//! `admin::routes`, so it sits behind the same admin-role check as every other `/v1/lightbulb/admin/*` route.
//!
//! By default the request is answered when the download ends: it holds the connection for as long as the download takes
//! and reports no progress (the CLI, `lightbulb-cli pull`, prints it). `{"model": ..., "wait": false}` answers 202 with a
//! job id at once and the download runs in the background; `GET /v1/lightbulb/admin/models/pull/{id}` then reports the job:
//! `state` (`running`, `done`, `failed`), `progress` (bytes `downloaded` and `total` per file, `total` null when the Hub
//! declared none), and on the end `result` (the blocking route's 200 body) or `status` and `error`. Jobs live in memory
//! until the process ends. Two pulls of one repo at once are refused with 409 either way.

use super::AppState;
use crate::model_fuel::pull::{
    FileOutcome, InsufficientSpace, PullSpec, Selection, available_space, hub_in, parse_spec, pull,
};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use mlmf_source_hub::HubSource;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/lightbulb/admin/models/pull", post(pull_model))
        .route("/v1/lightbulb/admin/models/pull/{id}", get(pull_job))
}

#[derive(serde::Deserialize)]
struct PullRequest {
    model: String,
    /// `false` answers 202 with a job id and downloads in the background; absent or `true` holds the connection.
    #[serde(default = "wait_by_default")]
    wait: bool,
}

fn wait_by_default() -> bool {
    true
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

/// Repos being pulled right now: two pulls of one repo would race on its partial files.
static PULLING: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub(crate) struct InFlight(String);

impl InFlight {
    pub(crate) fn take(repo: &str) -> Option<Self> {
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

/// One background pull, as `GET .../pull/{id}` reports it.
struct Job {
    value: serde_json::Value,
}

fn jobs() -> &'static Mutex<HashMap<String, Job>> {
    static JOBS: OnceLock<Mutex<HashMap<String, Job>>> = OnceLock::new();
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with_job(id: &str, f: impl FnOnce(&mut serde_json::Value)) {
    let mut all = jobs().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(job) = all.get_mut(id) {
        f(&mut job.value);
    }
}

/// Start `spec` downloading on its own thread and return the job id. `guard` is the repo's in-flight claim: it is
/// released when the job ends. `make_hub` builds the Hub client on the job's thread.
pub(crate) fn start_job(
    guard: InFlight,
    spec: PullSpec,
    dir: std::path::PathBuf,
    make_hub: impl FnOnce() -> HubSource + Send + 'static,
    free_space: impl Fn(&Path) -> Option<u64> + Send + 'static,
) -> String {
    let id = uuid::Uuid::new_v4().to_string();
    jobs().lock().unwrap_or_else(|e| e.into_inner()).insert(
        id.clone(),
        Job {
            value: serde_json::json!({
                "id": id, "repo": spec.repo, "state": "running", "progress": [],
            }),
        },
    );
    let job_id = id.clone();
    std::thread::spawn(move || {
        let _held = guard;
        let hub = make_hub();
        let (status, body) = run_pull(&hub, &dir, &spec, &free_space, &mut |path, p| {
            with_job(&job_id, |v| set_progress(v, path, p.downloaded, p.total));
        });
        with_job(&job_id, |v| {
            if status == StatusCode::OK {
                v["state"] = "done".into();
                v["result"] = body;
            } else {
                v["state"] = "failed".into();
                v["status"] = status.as_u16().into();
                v["error"] = body["error"].clone();
            }
        });
    });
    id
}

fn set_progress(job: &mut serde_json::Value, path: &str, downloaded: u64, total: Option<u64>) {
    let rows = job["progress"]
        .as_array_mut()
        .expect("progress is an array");
    let row = serde_json::json!({"path": path, "downloaded": downloaded, "total": total});
    match rows.iter_mut().find(|r| r["path"] == path) {
        Some(r) => *r = row,
        None => rows.push(row),
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
    run_pull(hub, dir, spec, free_space, &mut |_, _| {})
}

fn run_pull(
    hub: &HubSource,
    dir: &Path,
    spec: &PullSpec,
    free_space: &dyn Fn(&Path) -> Option<u64>,
    on_progress: &mut dyn FnMut(&str, mlmf_source_hub::Progress),
) -> (StatusCode, serde_json::Value) {
    match pull(hub, spec, dir, free_space, on_progress) {
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
    if !req.wait {
        let hub_dir = std::path::PathBuf::from(&dir);
        let id = start_job(
            guard,
            spec,
            std::path::PathBuf::from(dir),
            move || hub_in(&hub_dir),
            available_space,
        );
        return (StatusCode::ACCEPTED, Json(serde_json::json!({ "job": id }))).into_response();
    }
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

/// `GET /v1/lightbulb/admin/models/pull/{id}`: the job's state, 404 for an id this process never issued.
async fn pull_job(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    let all = jobs().lock().unwrap_or_else(|e| e.into_inner());
    match all.get(&id) {
        Some(job) => Json(job.value.clone()).into_response(),
        None => error(StatusCode::NOT_FOUND, format!("no pull job {id}")),
    }
}

#[cfg(test)]
mod tests;
