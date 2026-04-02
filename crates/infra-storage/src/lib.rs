//! # Infrastructure Storage
//!
//! Multi-backend distributed storage system for the DynAniML ecosystem.
//!
//! This crate provides:
//! - Multi-backend storage abstraction (RocksDB, SQLite, Memory)
//! - Distributed storage coordination
//! - Data replication and sharding
//! - Backup and recovery mechanisms
//! - Performance optimization and caching

#![deny(missing_docs)]
#![warn(clippy::all)]

mod backend;
pub use backend::*;

mod memory;
mod rocksdb;
mod sled; // Add back sled support
mod sqlite;
mod unified;

pub use memory::MemoryBackend;
pub use rocksdb::RocksDBBackend;
pub use sled::SledBackend;
pub use sqlite::SQLiteBackend;
pub use unified::UnifiedBackend;

#[cfg(test)]
mod tests {
    mod rocksdb_tests;
    mod sled_tests;
    mod sqlite_tests;
}
