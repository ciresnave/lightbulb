kind: baracuda 0.1.0-alpha.148
---
Repins `baracuda-driver` (optional, `fuel-cuda`) to `0.1.0-alpha.148`, the newest on crates.io, and moves the 17 baracuda crates in `Cargo.lock` from alpha.95 to alpha.148. The `fuel-cuda` feature now also enables `fuel-model-llama/cuda`: without it `tests/gpu_paged_vs_contiguous.rs` did not compile (`forward_with_kv_context_captured` is gated on that feature). `cargo check --locked --features fuel-cuda --all-targets` passes.
