// ============================================================================
// dynctx_core/src/constants.rs — System-wide constants and configuration
// ============================================================================
// MIT licence — © 2025 Eric Evans & contributors

//! System-wide constants used across multiple modules.
//! This module centralizes configuration values to prevent circular imports
//! and make system-wide changes easier to manage.

/// Maximum number of slots in the arena.
/// This determines the upper bound on how many tokens can be stored simultaneously.
pub const MAX_SLOTS: u32 = 32 * 1024;

/// Tombstone value used to mark deleted nodes in the arena.
/// Using u32::MAX as it's unlikely to be a legitimate token ID.
/// This value is used by both arena.rs and snapshot.rs to identify freed nodes.
pub const TOMBSTONE_VALUE: u32 = u32::MAX;
