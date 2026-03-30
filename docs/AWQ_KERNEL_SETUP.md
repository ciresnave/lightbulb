# AWQ/Marlin Kernel Setup Script

This script helps you set up the Marlin CUDA kernels from candle-vllm for AWQ quantization support.

## Prerequisites

1. Clone candle-vllm repository:
```bash
cd C:\Users\cires\OneDrive\Documents\projects\lightbulb
git clone https://github.com/EricLBuehler/candle-vllm.git
```

## Files to Copy

### Step 1: Create directory structure
```powershell
# From lightbulb/lightbulb directory
New-Item -ItemType Directory -Path "kernels\src" -Force
New-Item -ItemType Directory -Path "kernels\marlin" -Force
```

### Step 2: Copy CUDA kernels (.cu files)
```powershell
# Source: candle-vllm/kernels/src/
# Destination: lightbulb/lightbulb/kernels/src/

Copy-Item "..\candle-vllm\kernels\src\marlin_matmul_f16.cu" "kernels\src\"
Copy-Item "..\candle-vllm\kernels\src\marlin_matmul_bf16.cu" "kernels\src\"
Copy-Item "..\candle-vllm\kernels\src\marlin_matmul_awq_f16.cu" "kernels\src\"
Copy-Item "..\candle-vllm\kernels\src\marlin_matmul_awq_bf16.cu" "kernels\src\"
Copy-Item "..\candle-vllm\kernels\src\marlin_repack.cu" "kernels\src\"
```

### Step 3: Copy header files (.cuh files)
```powershell
# Source: candle-vllm/kernels/marlin/
# Destination: lightbulb/lightbulb/kernels/marlin/

Copy-Item "..\candle-vllm\kernels\marlin\marlin.cuh" "kernels\marlin\"
Copy-Item "..\candle-vllm\kernels\marlin\marlin_dtypes.cuh" "kernels\marlin\"
Copy-Item "..\candle-vllm\kernels\marlin\marlin_cuda_kernel.cuh" "kernels\marlin\"
```

### Step 4: Add copyright notices to copied files

Add this header to each .cu and .cuh file:

```c
// Portions adapted from candle-vllm (MIT License)
// Copyright (c) Eric Buehler and contributors
// https://github.com/EricLBuehler/candle-vllm
//
// Original Marlin kernel (Apache-2.0 License)
// Copyright (c) IST Austria DASL Lab
// https://github.com/IST-DASLab/marlin
```

## Verification

After copying, verify the structure:

```powershell
tree /F kernels
```

Expected output:
```
kernels
├── marlin
│   ├── marlin.cuh
│   ├── marlin_dtypes.cuh
│   └── marlin_cuda_kernel.cuh
└── src
    ├── marlin_matmul_f16.cu
    ├── marlin_matmul_bf16.cu
    ├── marlin_matmul_awq_f16.cu
    ├── marlin_matmul_awq_bf16.cu
    └── marlin_repack.cu
```

## Next Steps

After copying files, run:
```bash
# This will create FFI bindings and build.rs
cargo run -- --help
```

Then proceed to Phase 1 tasks 4-7 in the todo list.
