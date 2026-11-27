// SPDX-License-Identifier: MIT OR Apache-2.0
//! `POST /v1/embeddings` (ROADMAP F3): the HTTP wrapper over [`super::embeddings`]. The core knows nothing of
//! HTTP; this maps its [`ErrorKind`] to a status (`Invalid` = 400, `Internal` = 500) and runs the encoder off the
//! async runtime, because a forward pass is blocking CPU/GPU work.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::embeddings::{EmbeddingError, EmbeddingRequest, ErrorKind, build_response, validate};
use crate::api::AppState;

fn error(status: StatusCode, message: String, param: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": message, "param": param})),
    )
        .into_response()
}

fn from_embedding_error(e: EmbeddingError) -> Response {
    let status = match e.kind {
        ErrorKind::Invalid => StatusCode::BAD_REQUEST,
        ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error(status, e.message, e.param)
}

pub async fn create_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let Some(entry) = state.models.get(&req.model) else {
        return error(
            StatusCode::BAD_REQUEST,
            format!("model '{}' is not loaded", req.model),
            "model",
        );
    };
    let Some(loaded) = entry.embedding else {
        return error(
            StatusCode::BAD_REQUEST,
            format!("model '{}' is not an embedding model", req.model),
            "model",
        );
    };
    let name = req.model.clone();
    let done = tokio::task::spawn_blocking(move || {
        let v = validate(&req, &loaded.spec)?;
        build_response(&name, &v, &loaded.spec, loaded.embedder.as_ref())
    })
    .await;
    match done {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err(e)) => from_embedding_error(e),
        Err(join) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("the embedding task failed: {join}"),
            "model",
        ),
    }
}
