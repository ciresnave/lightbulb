# Building Lightbulb Without CUDA (CPU-Only Mode)

## Quick Start

Build Lightbulb for CPU-only mode:

```powershell
cargo build --release --no-default-features
```

## Why Use CPU-Only Mode?

- Don't have an NVIDIA GPU
- Don't have cuDNN installed
- Testing on non-GPU systems
- CI/CD pipelines without GPU access

## Features Available in CPU-Only Mode

- ✅ Model loading (GGUF, SafeTensors)
- ✅ Text generation
- ✅ Quantized model support
- ✅ Batched inference
- ✅ HTTP API server
- ✅ All core functionality

## Features NOT Available (Require CUDA)

- ❌ Flash Attention (FlashAttention-2)
- ❌ CUDA-accelerated layer normalization
- ❌ GPU-accelerated inference
- ❌ Multi-GPU distribution

## Performance Expectations

CPU-only mode will be significantly slower than GPU mode:
- Small models (1-7B params): Usable on modern CPUs
- Medium models (7-13B params): Slow but functional
- Large models (13B+ params): Impractical without GPU

## Running with CPU-Only Build

```powershell
# Build
cargo build --release --no-default-features

# Run
.\target\release\lightbulb-cli.exe --help
```

## Enabling CUDA Later

To switch to GPU mode later:

1. Install cuDNN (see CUDNN_INSTALL.md)
2. Rebuild with CUDA features:
   ```powershell
   cargo build --release --features cuda-full
   ```

## Feature Flags Explained

- `default = []` - No features by default (CPU-only)
- `cuda` - Basic CUDA support
- `flash-attn` - FlashAttention-2 (requires CUDA)
- `layer-norm` - CUDA layer normalization (requires CUDA)
- `cuda-full` - All CUDA optimizations (cuda + flash-attn + layer-norm)
