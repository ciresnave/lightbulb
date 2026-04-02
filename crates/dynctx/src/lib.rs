// dynctx — Dynamic Context Management for LLMs

pub mod arena;
pub mod constants;
pub mod error;
pub mod log;
pub mod node_arena;
pub mod op_queue;
pub mod relationship_manager;
pub mod relationship_types;
pub mod rope;
pub mod snapshot;

// Re-export common types for easier access
pub use arena::{ArenaError, SlotArena, TokenNode};
pub use node_arena::NodeKey;
pub use rope::*;

// For compatibility with examples, provide type aliases
pub type Arena = SlotArena;
pub type Position = NodeKey;
