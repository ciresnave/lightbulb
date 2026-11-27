// SPDX-License-Identifier: MIT OR Apache-2.0
//! OpenAI-Compatible API Endpoints
//!
//! Implements OpenAI API v1 specification for drop-in compatibility
//! with existing tools (LangChain, OpenAI SDK, etc.)

pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod embeddings_route;
pub(crate) mod logprobs;
pub mod models;
pub mod nomic_embedder;
mod tool_parsers;

use axum::{
    Router,
    routing::{get, post},
};

use super::AppState;

/// Create OpenAI-compatible routes
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/embeddings", post(embeddings_route::create_embeddings))
        .route(
            "/v1/models",
            get(models::list_models).post(models::list_models),
        )
        .route("/v1/models/{model}", get(models::retrieve_model))
}
