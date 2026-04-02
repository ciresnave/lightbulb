//! Error types for DynCtx Core

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ArenaError>;

/// Arena operation errors
#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("Position is out of bounds")]
    OutOfBounds,

    #[error("Invalid position: {0}")]
    InvalidPosition(u32),

    #[error("Arena is full")]
    Full,

    #[error("Type mismatch")]
    TypeMismatch,
}
