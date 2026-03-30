//! Shared type definitions for API

pub use crate::api::openai::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, LightbulbExtensions,
    ReasoningBudget, RequestMetadata,
};

pub use crate::api::openai::completions::{CompletionRequest, CompletionResponse, PromptInput};

pub use crate::api::openai::models::{ModelInfo, ModelListResponse};

pub use crate::api::auth_middleware::{ApiKeyInfo, ErrorDetail, ErrorResponse};
