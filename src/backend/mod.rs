// SPDX-License-Identifier: MIT OR Apache-2.0
// Backend module for hardware-specific implementations
//
// This module contains FFI bindings and custom operations for
// hardware-accelerated backends (CUDA, ROCm, etc.)

// `marlin.rs` (the candlelight-typed `MarlinMatMul` wrapper) removed 2026-10-22
// (CireSnave: "Remove all dependencies on Candle and Candlelight..."). This
// FFI layer itself was already candlelight-free and had no caller outside
// `marlin.rs`'s own re-export -- orphaned now, not newly broken. See
// `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
#[cfg(feature = "cuda")]
pub mod marlin_ffi;

// Re-export commonly used items
#[cfg(feature = "cuda")]
pub use marlin_ffi::*;
