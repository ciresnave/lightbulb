// ============================================================================
// build.rs — Cap'n Proto code‑gen & rerun hints
// ============================================================================
//! Build‑script run by Cargo.  Compiles `src/log.capnp` into Rust and places
//! the generated file in `$OUT_DIR/log_capnp.rs`, then tells Cargo when to
//! re‑run the build (if the schema changes).
//!
//! Requires the `capnp` and `capnpc` crates in `[build-dependencies]`.

fn main() {
    println!("cargo:rerun-if-changed=src/log.capnp");

    capnpc::CompilerCommand::new()
        .src_prefix("src")
        .file("src/log.capnp")
        .run()
        .expect("schema compiler command");
}
