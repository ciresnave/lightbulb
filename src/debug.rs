//! Conditional debug output system
//!
//! Enable specific debug categories via feature flags:
//! - debug-prefill: Prefill phase debugging
//! - debug-decode: Decode phase debugging  
//! - debug-attention: Attention layer debugging
//! - debug-cache: Cache operations debugging
//! - debug-rope: RoPE computation debugging
//! - debug-mlp: MLP layer debugging
//! - debug-chunking: Chunk scheduling debugging
//!
//! Or enable all: debug-all

/// Debug output for prefill operations
#[macro_export]
macro_rules! debug_prefill {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-prefill", feature = "debug-all"))]
        eprintln!("[PREFILL] {}", format!($($arg)*));
    };
}

/// Debug output for decode operations
#[macro_export]
macro_rules! debug_decode {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-decode", feature = "debug-all"))]
        eprintln!("[DECODE] {}", format!($($arg)*));
    };
}

/// Debug output for attention computations
#[macro_export]
macro_rules! debug_attention {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-attention", feature = "debug-all"))]
        eprintln!("[ATTENTION] {}", format!($($arg)*));
    };
}

/// Debug output for cache operations
#[macro_export]
macro_rules! debug_cache {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-cache", feature = "debug-all"))]
        eprintln!("[CACHE] {}", format!($($arg)*));
    };
}

/// Debug output for RoPE computations
#[macro_export]
macro_rules! debug_rope {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-rope", feature = "debug-all"))]
        eprintln!("[ROPE] {}", format!($($arg)*));
    };
}

/// Debug output for MLP operations
#[macro_export]
macro_rules! debug_mlp {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-mlp", feature = "debug-all"))]
        eprintln!("[MLP] {}", format!($($arg)*));
    };
}

/// Debug output for chunk scheduling
#[macro_export]
macro_rules! debug_chunking {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-chunking", feature = "debug-all"))]
        eprintln!("[CHUNKING] {}", format!($($arg)*));
    };
}

/// Debug output for general engine operations
#[macro_export]
macro_rules! debug_engine {
    ($($arg:tt)*) => {
        #[cfg(any(feature = "debug-engine", feature = "debug-all"))]
        eprintln!("[ENGINE] {}", format!($($arg)*));
    };
}
