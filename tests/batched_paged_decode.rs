//! Harness for `src/model_fuel/batched.rs` **before it is wired into
//! `src/model_fuel/mod.rs`**.
//!
//! `batched.rs` was written by an agent that was not permitted to edit
//! `mod.rs` (three agents share it; the orchestrator adds the `pub mod`
//! declarations). Without a `mod` declaration the file is not compiled at all,
//! so `cargo check --lib` would say nothing about it. This shim gives it a
//! compilation unit: `#[path]`-including it into an integration-test crate
//! compiles the module **and** its `#[cfg(test)] mod tests` (integration test
//! crates are built with `--test`, so `cfg(test)` is on).
//!
//! `batched.rs` deliberately references nothing inside Lightbulb — only `fuel`,
//! `anyhow`, and `std` — which is exactly what makes this possible.
//!
//! **Once the orchestrator adds `pub mod batched;` to `src/model_fuel/mod.rs`,
//! delete this file.** The same tests then run under
//! `cargo test --lib model_fuel::batched`, and keeping both would compile the
//! module twice.
//!
//! Run: `cargo test --test batched_paged_decode`
//! Plus the checkpoint test: `cargo test --release --test batched_paged_decode -- --ignored --nocapture`

#[path = "../src/model_fuel/batched.rs"]
#[allow(dead_code)]
mod batched;
