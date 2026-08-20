// Build script for Lightbulb
//
// This build script handles CUDA kernel compilation when the 'cuda' feature is enabled.
// Currently, Marlin AWQ kernels require manual setup due to CUDA toolkit dependencies.

fn main() {
    // Rerun if CUDA feature changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Add cuDNN library path (required for FlashAttention)
    #[cfg(target_os = "windows")]
    {
        println!(
            "cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA\\CUDNN\\v9.16\\lib\\12.9\\x64"
        );
    }

    #[cfg(feature = "cuda")]
    {
        compile_cuda_kernels();
    }
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    // NOT IMPLEMENTED, and the kernels it referred to have been REMOVED.
    //
    // This stub used to list `kernels/src/marlin_matmul_*.cu`,
    // `kernels/src/marlin_repack.cu` and `kernels/marlin/*.cuh` as the files it
    // would compile "when properly implemented". Those eight files were deleted
    // on 2026-08-20: they were verbatim third-party CUDA, byte-identical to
    // guoqingbao/attention.rs (MIT-only), carrying no attribution, in a crate
    // declaring `MIT OR Apache-2.0`. They also could never have compiled — the
    // `.cu` files `#include "marlin_gptq_cuda_kernel.cuh"` and
    // "marlin_cuda_kernel.cuh", neither of which was ever in this repository.
    // A partial vendor of someone else's kernels, wired to nothing.
    //
    // If the AWQ/Marlin GPU path is ever revived, DO NOT restore them from git.
    // Re-vendor deliberately with attribution, or use Fuel's
    // `fuel-cuda-backend/src/baracuda/quant_w4a16.rs`, which covers Marlin and
    // AWQ W4A16 on the backend this project is porting to.

    println!("cargo:warning=CUDA kernel compilation not yet implemented");
    println!("cargo:warning=AWQ quantization requires manual CUDA setup");
    println!("cargo:warning=See docs/AWQ_SETUP.md for instructions");
}
