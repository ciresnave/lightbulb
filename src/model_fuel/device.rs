//! Where the Fuel path's device is chosen — and the only place it is.
//!
//! `generate.rs` originally hardcoded `Device::cpu()`. Spreading that choice
//! across the loader, the session and the runner is how a model and its KV
//! cache end up on different devices, which fails deep inside `realize` as a
//! confusing byte-count error rather than at the point of the mistake.

use fuel::Device;

/// The device this build serves on.
///
/// CUDA when `fuel-cuda` is enabled and a device is actually present; CPU
/// otherwise.
///
/// **Falling back to CPU is deliberate.** A server that refuses to start on a
/// CPU-only host is worse than one that starts slow, and the log line names
/// which was chosen so a surprising benchmark has somewhere to look.
///
/// The measurement harness (`tests/gpu_paged_vs_contiguous.rs`) makes the
/// OPPOSITE choice and fails outright, because a benchmark that silently
/// measures CPU is an artifact rather than a result.
pub fn select() -> Device {
    #[cfg(feature = "fuel-cuda")]
    {
        match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => {
                tracing::info!("model_fuel: CUDA device 0 selected");
                return Device::from(d);
            }
            Err(e) => {
                tracing::warn!(
                    "model_fuel: `fuel-cuda` is enabled but CUDA device 0 is unavailable \
                     ({e:?}); falling back to CPU"
                );
            }
        }
    }
    tracing::info!("model_fuel: CPU selected");
    Device::cpu()
}

#[cfg(test)]
mod tests {
    /// `select()` returns a usable device and does not panic.
    ///
    /// Deliberately weak on WHICH device — that depends on features and
    /// hardware. The claim under test is that selection is total: there is no
    /// build configuration in which it fails to produce one.
    #[test]
    fn select_returns_a_usable_device() {
        let dev = super::select();
        // Realizing a trivial graph proves the device is not merely
        // constructed but actually drivable.
        let a = fuel::lazy::LazyTensor::from_f32(vec![2.0, 3.0], (1usize, 2usize), &dev);
        let w = a.const_f32_like(vec![1.0, 0.0, 0.0, 1.0], (2usize, 2usize));
        let y = a.matmul(&w).expect("matmul on the selected device");
        assert_eq!(y.realize_f32(), vec![2.0, 3.0]);
    }
}
