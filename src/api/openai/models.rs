//! Models Endpoint
//!
//! OpenAI-compatible `/v1/models` endpoint for listing available models.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Serialize;
use std::time::SystemTime;

use crate::api::AppState;

/// Model list response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// List models endpoint handler
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    match get_available_models(state).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Get available models
async fn get_available_models(_state: AppState) -> anyhow::Result<ModelListResponse> {
    // TODO: Query actual model manager for loaded models
    // For now, return a mock list

    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    Ok(ModelListResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "lightbulb-default".to_string(),
                object: "model".to_string(),
                created,
                owned_by: "lightbulb".to_string(),
            },
            ModelInfo {
                id: "lightbulb-7b".to_string(),
                object: "model".to_string(),
                created,
                owned_by: "lightbulb".to_string(),
            },
            ModelInfo {
                id: "lightbulb-13b".to_string(),
                object: "model".to_string(),
                created,
                owned_by: "lightbulb".to_string(),
            },
        ],
    })
}
