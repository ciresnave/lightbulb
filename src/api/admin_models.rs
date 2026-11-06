// SPDX-License-Identifier: MIT OR Apache-2.0
//! Operator control over loaded models (ROADMAP F2): unload one now instead of waiting for
//! memory pressure to evict it. Merged into `admin::routes`, so it sits behind the same
//! admin-role check as every other `/v1/lightbulb/admin/*` route.

use super::AppState;
use super::manager::UnloadError;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};

pub fn routes() -> Router<AppState> {
    Router::new().route(
        "/v1/lightbulb/admin/models/{name}/unload",
        post(unload_model),
    )
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

/// `POST /v1/lightbulb/admin/models/{name}/unload`: 200 with what the unload did, 404 if the
/// model is not loaded, 409 if a request is running on it, 501 when the server has no model
/// manager (no models directory).
async fn unload_model(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some(manager) = state.manager.clone() else {
        return error(
            StatusCode::NOT_IMPLEMENTED,
            "this server has no model manager (no models directory), so models cannot be unloaded"
                .into(),
        );
    };
    let target = name.clone();
    let result = tokio::task::spawn_blocking(move || manager.unload(&target)).await;
    match result {
        Ok(Ok(out)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "unloaded": name,
                "runner_joined": out.joined,
                "device_memory_returned": out.released.is_some(),
            })),
        )
            .into_response(),
        Ok(Err(UnloadError::NotLoaded)) => error(
            StatusCode::NOT_FOUND,
            format!("model '{name}' is not loaded"),
        ),
        Ok(Err(UnloadError::Busy)) => error(
            StatusCode::CONFLICT,
            format!("model '{name}' has a request running; retry when it ends"),
        ),
        Ok(Err(UnloadError::Failed(m))) => error(StatusCode::INTERNAL_SERVER_ERROR, m),
        Err(e) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unload task: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests;
