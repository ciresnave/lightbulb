// Backend module for hardware-specific implementations
//
// This module contains FFI bindings and custom operations for
// hardware-accelerated backends (CUDA, ROCm, etc.)

#[cfg(feature = "cuda")]
pub mod marlin_ffi;

#[cfg(feature = "cuda")]
pub mod marlin;

// Re-export commonly used items
#[cfg(feature = "cuda")]
pub use marlin_ffi::*;

#[cfg(feature = "cuda")]
pub use marlin::{MarlinMatMul, Precision, QuantFormat};
