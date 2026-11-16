kind: baracuda 0.1.0-alpha.148
---
Repins `baracuda-driver` (optional, `fuel-cuda`) to `0.1.0-alpha.148` and moves the 17 baracuda crates in `Cargo.lock` from alpha.95 to alpha.148, per the baracuda-published notice. `cargo metadata --locked` passes; the `fuel-cuda` `cargo check` was NOT completed (interrupted by a reboot).
