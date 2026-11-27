// SPDX-License-Identifier: MIT OR Apache-2.0
//! `POST /v1/embeddings` (ROADMAP F3): the HTTP wrapper over [`super::embeddings`]. The core knows nothing of
//! HTTP; this maps its [`ErrorKind`] to a status (`Invalid` = 400, `Internal` = 500) and runs the encoder off the
//! async runtime, because a forward pass is blocking CPU/GPU work.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::embeddings::{
    EmbeddingError, EmbeddingRequest, ErrorKind, LoadedEmbedder, build_response, validate,
};
use crate::api::AppState;
use crate::api::manager::Lease;
use crate::api::model_select::scope_to_model;

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

/// The encoder for `name` and the lease that pins it. With a manager, `scope_to_model` loads the model on
/// demand (evicting idle ones) and leases it so it cannot be evicted mid-request; without one the registry is
/// read directly. A name that is unknown, or a chat model, is a 400 naming which.
async fn find_embedder(
    state: &AppState,
    name: &str,
) -> Result<(Arc<LoadedEmbedder>, Option<Lease>), Box<Response>> {
    let (lookup, lease) = if state.manager.is_some() {
        match scope_to_model(state, name).await {
            Ok(s) => (s.state, s.lease),
            Err(rejection) => return Err(rejection),
        }
    } else {
        (state.clone(), None)
    };
    let entry = lookup.models.get(name).ok_or_else(|| {
        Box::new(error(
            StatusCode::BAD_REQUEST,
            format!("model '{name}' is not loaded"),
            "model",
        ))
    })?;
    let loaded = entry.embedding.ok_or_else(|| {
        Box::new(error(
            StatusCode::BAD_REQUEST,
            format!("model '{name}' is not an embedding model"),
            "model",
        ))
    })?;
    Ok((loaded, lease))
}

pub async fn create_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let (loaded, _lease) = match find_embedder(&state, &req.model).await {
        Ok(found) => found,
        Err(response) => return *response,
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
