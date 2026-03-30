//! OpenAI-Compatible API Endpoints
//!
//! Implements OpenAI API v1 specification for drop-in compatibility
//! with existing tools (LangChain, OpenAI SDK, etc.)

pub mod chat;
pub mod completions;
pub mod models;

use axum::{Router, routing::post};

use super::AppState;

/// Create OpenAI-compatible routes
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/models", post(models::list_models))
}
