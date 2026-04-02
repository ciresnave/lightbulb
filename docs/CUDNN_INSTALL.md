# Installing cuDNN for Lightbulb

## Problem
Build fails with: `LINK : fatal error LNK1181: cannot open input file 'cudnn.lib'`

## Solution: Install cuDNN

### 1. Download cuDNN
- Visit: https://developer.nvidia.com/cudnn
- Login with NVIDIA Developer account (free)
- Download cuDNN for CUDA 13.x (compatible with your CUDA 13.0 installation)
- Choose Windows version

### 2. Install cuDNN
After downloading the zip file (e.g., `cudnn-windows-x86_64-9.x.x.x_cuda13-archive.zip`):

```powershell
# Extract to a temporary location
# Copy files to CUDA installation directory:

# Copy headers
Copy-Item -Path "cuda\include\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\" -Recurse

# Copy libraries
Copy-Item -Path "cuda\lib\x64\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64\" -Recurse

# Copy DLLs
Copy-Item -Path "cuda\bin\*" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\" -Recurse
```

### 3. Verify Installation
```powershell
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64\cudnn.lib"
```

Should return `True`

### 4. Rebuild Lightbulb
```powershell
cd C:\Users\cires\OneDrive\Documents\projects\lightbulb
cargo clean
cargo build --release
```

## Alternative: Build Without CUDA

If you don't need CUDA acceleration, you can build with CPU-only mode (see CPU_BUILD.md).

## Notes
- cuDNN requires NVIDIA account for download
- Make sure to match cuDNN version with CUDA 13.x
- After installing cuDNN, flash attention and other GPU optimizations will work
