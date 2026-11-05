// SPDX-License-Identifier: MIT OR Apache-2.0
//! `LIGHTBULB_DEVICE` set explicitly must never quietly become the CPU.
//!
//! `device::select()` (the infallible form the loaders and `main` call) used to
//! log an error and return the CPU when the selector could not be honoured, so
//! "you asked for cuda:7 and it is missing" produced a slow CPU run that looked
//! like a result. It must fail loudly. This file holds ONE test because it
//! sets a process environment variable, which would race any sibling test.

use lightbulb::model_fuel::device;

#[test]
fn an_explicit_device_that_cannot_be_used_is_a_loud_failure_not_the_cpu() {
    // SAFETY: the only test in this process, so nothing reads the environment
    // concurrently.
    unsafe { std::env::set_var("LIGHTBULB_DEVICE", "cuda:7") };
    // The fallible form already refuses (that part was always true) ...
    assert!(
        device::try_select().is_err(),
        "control: try_select must refuse an absent cuda:7"
    );
    // ... and the infallible form must not turn the refusal into a CPU.
    let outcome = std::panic::catch_unwind(device::select);
    assert!(
        outcome.is_err(),
        "select() returned a device ({outcome:?}) for an explicit, absent cuda:7"
    );

    // With no variable at all, `auto` still works (CPU on a build with no
    // accelerator), and the choice is recorded for /v1/models.
    unsafe { std::env::remove_var("LIGHTBULB_DEVICE") };
    let _ = device::select();
    assert!(
        device::chosen().is_some(),
        "a successful selection must be recorded so a run can prove its device"
    );
}
