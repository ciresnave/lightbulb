// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pick the model a request names (ROADMAP F2, P2b-2).
//!
//! [`scope_to_model`] resolves `request.model` and returns a **request-scoped**
//! [`AppState`] whose registry holds just that model, so every `state.model()` read
//! deeper in a handler sees the model the request asked for, with no change to that
//! code. With no [`ModelManager`](super::manager::ModelManager) (tests, a server with no
//! models directory) it is the old rule: the name must be the one loaded model.
//!
//! The [`Lease`] that keeps the model from being evicted is returned with the state and
//! must live as long as the response does, including a streamed body: [`hold`] moves it
//! into the body.

use super::AppState;
use super::manager::{Lease, LoadError};
use super::registry::ModelRegistry;
use axum::{
    Json,
    body::Body,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;

/// A response to send instead of serving (boxed: `Response` is large).
pub type Rejection = Box<Response>;

/// The state to serve one request with, and the lease that pins its model.
pub struct Scoped {
    pub state: AppState,
    pub lease: Option<Lease>,
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

fn not_loaded(state: &AppState, requested: &str) -> Response {
    error(
        StatusCode::BAD_REQUEST,
        format!(
            "model '{requested}' is not loaded on this server; the loaded model is '{}'",
            state.config.default_model
        ),
    )
}

/// The old single-model rule, used when there is no manager.
fn legacy(state: &AppState, requested: &str) -> Result<Scoped, Rejection> {
    if state.model().inference_tx.is_some() && requested != state.config.default_model {
        return Err(Box::new(not_loaded(state, requested)));
    }
    Ok(Scoped {
        state: state.clone(),
        lease: None,
    })
}

fn scoped_to(state: &AppState, name: &str, lease: Lease) -> Scoped {
    let mut scoped = state.clone();
    scoped.models = ModelRegistry::single(name, state.models.get(name).unwrap_or_default());
    Scoped {
        state: scoped,
        lease: Some(lease),
    }
}

fn busy(message: String) -> Response {
    let mut r = error(StatusCode::SERVICE_UNAVAILABLE, message);
    r.headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
    r
}

/// Resolve `requested` to a loaded model (loading it, and evicting idle ones, if needed).
///
/// Answers: 400 for a name that is not a model here (as before), 503 + `Retry-After` when
/// the idle models cannot free enough room, 500 when the load fails.
pub async fn scope_to_model(state: &AppState, requested: &str) -> Result<Scoped, Rejection> {
    let Some(manager) = state.manager.clone() else {
        return legacy(state, requested);
    };
    let name = requested.to_string();
    let result = tokio::task::spawn_blocking(move || manager.acquire(&name))
        .await
        .map_err(|e| {
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("model load task: {e}"),
            )
        })?;
    match result {
        Ok(lease) => Ok(scoped_to(state, requested, lease)),
        // Nothing is loaded at all: let the handler's own "no model available" answer it.
        Err(LoadError::NotFound(_)) if state.model().inference_tx.is_none() => {
            legacy(state, requested)
        }
        Err(LoadError::NotFound(_)) => Err(Box::new(not_loaded(state, requested))),
        Err(LoadError::NoRoom(s)) => Err(Box::new(busy(format!(
            "not enough device memory for model '{requested}': needs {} bytes, {} free, {} evictable",
            s.needed, s.free, s.evictable
        )))),
        Err(LoadError::Failed(m)) => Err(Box::new(error(StatusCode::INTERNAL_SERVER_ERROR, m))),
    }
}

/// Tie `lease` to `response`: it is released when the body is finished or dropped, so a
/// streaming response keeps its model pinned until the last chunk.
pub fn hold(response: Response, lease: Option<Lease>) -> Response {
    let Some(lease) = lease else {
        return response;
    };
    let (parts, body) = response.into_parts();
    let held = body.into_data_stream().map(move |chunk| {
        let _pin = &lease;
        chunk
    });
    Response::from_parts(parts, Body::from_stream(held))
}

#[cfg(test)]
mod tests;
