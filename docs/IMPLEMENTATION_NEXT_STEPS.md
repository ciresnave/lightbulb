# Quick Start: Implementing AWQ and Speculative Decoding

## Current Status

✅ **Task 1 Complete**: Unified memory estimation design created
- File: `docs/MEMORY_ESTIMATION_DESIGN.md`
- Defines `MemoryEstimate`, `WeightMemory`, `KvCacheMemory` structs
- Supports both single-model and dual-model (speculative) scenarios

🔄 **Task 2 In Progress**: AWQ Phase 1 (CUDA Kernel Integration)
🔄 **Task 3 Pending**: Speculative Decoding Phase 1 (Dual-model Loading)

---

## Manual Steps Required (You Need To Do This)

### 1. Clone candle-vllm Repository

```bash
cd C:\Users\cires\OneDrive\Documents\projects\lightbulb
git clone https://github.com/EricLBuehler/candle-vllm.git
```

### 2. Copy Marlin Kernels to Lightbulb

Run these PowerShell commands from `lightbulb/lightbulb/` directory:

```powershell
# Create directories
New-Item -ItemType Directory -Path "kernels\src" -Force
New-Item -ItemType Directory -Path "kernels\marlin" -Force

# Copy CUDA kernels (.cu files)
Copy-Item "..\..\candle-vllm\kernels\src\marlin_matmul_f16.cu" "kernels\src\"
Copy-Item "..\..\candle-vllm\kernels\src\marlin_matmul_bf16.cu" "kernels\src\"
Copy-Item "..\..\candle-vllm\kernels\src\marlin_matmul_awq_f16.cu" "kernels\src\"
Copy-Item "..\..\candle-vllm\kernels\src\marlin_matmul_awq_bf16.cu" "kernels\src\"
Copy-Item "..\..\candle-vllm\kernels\src\marlin_repack.cu" "kernels\src\"

# Copy header files (.cuh files)
Copy-Item "..\..\candle-vllm\kernels\marlin\marlin.cuh" "kernels\marlin\"
Copy-Item "..\..\candle-vllm\kernels\marlin\marlin_dtypes.cuh" "kernels\marlin\"
Copy-Item "..\..\candle-vllm\kernels\marlin\marlin_cuda_kernel.cuh" "kernels\marlin\"
```

### 3. After Manual Steps Complete

Let me know and I'll:
- Create Rust FFI bindings (`kernels/src/marlin_ffi.rs`)
- Update `kernels/build.rs` for CUDA compilation
- Add license attributions to `docs/THIRD_PARTY_NOTICES.md`
- Create test harness to verify kernel compilation

---

## What I Can Do Now (Automated)

While you clone and copy files, I can prepare:

1. ✅ Memory estimation module (`src/memory/estimate.rs`)
2. ✅ Speculative model loader structure (`src/loaders/speculative.rs`)
3. ✅ Configuration updates for both features
4. ✅ Test scaffolding

Should I proceed with creating these files now?

---

## Implementation Timeline

**Today (November 5, 2025)**:
- ✅ Memory estimation design
- 🔄 AWQ Phase 1 setup (waiting for candle-vllm clone)
- ⏳ Create supporting infrastructure (memory module, loaders)

**Next Steps**:
- Once kernels copied → Create FFI bindings + build.rs
- Test kernel compilation
- Proceed to AWQ Phase 2 (Rust backend)

**Week 1 Goal**: AWQ kernels compiling + memory estimation working
