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
    // TODO: Implement CUDA kernel compilation
    //
    // This function will compile Marlin CUDA kernels when properly implemented.
    // Currently requires:
    // 1. CUDA Toolkit 12.0+ installed
    // 2. NVCC compiler in PATH
    // 3. Proper compute capability flags (sm_80, sm_89, etc.)
    //
    // Kernel files to compile:
    // - kernels/src/marlin_matmul_f16.cu
    // - kernels/src/marlin_matmul_bf16.cu
    // - kernels/src/marlin_matmul_awq_f16.cu
    // - kernels/src/marlin_matmul_awq_bf16.cu
    // - kernels/src/marlin_repack.cu
    //
    // Headers:
    // - kernels/marlin/marlin.cuh
    // - kernels/marlin/marlin_dtypes.cuh
    //
    // Example compilation (reference from candle-vllm):
    // ```
    // use cc::Build;
    //
    // let mut build = Build::new();
    // build
    //     .cuda(true)
    //     .flag("-std=c++17")
    //     .flag("-O3")
    //     .flag("--expt-relaxed-constexpr")
    //     .flag("-gencode")
    //     .flag("arch=compute_80,code=sm_80")  // Ampere
    //     .flag("-gencode")
    //     .flag("arch=compute_89,code=sm_89")  // Ada Lovelace
    //     .include("kernels/marlin")
    //     .file("kernels/src/marlin_matmul_f16.cu")
    //     .file("kernels/src/marlin_matmul_bf16.cu")
    //     .file("kernels/src/marlin_matmul_awq_f16.cu")
    //     .file("kernels/src/marlin_matmul_awq_bf16.cu")
    //     .file("kernels/src/marlin_repack.cu")
    //     .compile("marlin_kernels");
    //
    // println!("cargo:rustc-link-lib=static=marlin_kernels");
    // println!("cargo:rustc-link-lib=dylib=cudart");
    // ```

    println!("cargo:warning=CUDA kernel compilation not yet implemented");
    println!("cargo:warning=AWQ quantization requires manual CUDA setup");
    println!("cargo:warning=See docs/AWQ_SETUP.md for instructions");
}
