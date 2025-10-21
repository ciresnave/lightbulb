//! Feature-gated kernels backends (CUDA/Metal/Flash-Attn)

#[cfg(feature = "cuda")]
pub mod cuda {
    pub fn available() -> bool {
        true
    }
}

#[cfg(feature = "metal")]
pub mod metal {
    pub fn available() -> bool {
        true
    }
}

#[cfg(feature = "flash-attn")]
pub mod flash_attn {
    pub fn available() -> bool {
        true
    }
}
