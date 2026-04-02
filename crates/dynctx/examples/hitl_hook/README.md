hitl_hook example
=================

This small Rust example simulates uncertainty during a short loop and optionally invokes an external Clarify CLI when uncertainty is high.

How it works

- The program prints an uncertainty value for steps 1..5.

If an uncertainty value > 0.8 is observed it prepares a prompt and, when the environment variable `INVOKE_CLI` is set to `1`, attempts to run `dynaniml clarify --prompt "..."` on PATH.

-- Build and run (standalone):

```powershell
cargo run --manifest-path .\Cargo.toml
```

```

- To enable external CLI invocation, put a compatible `dynaniml` executable on your PATH (for example build `dynaniml-cli-verify` and add its built binary to PATH), then run:

```powershell
$env:INVOKE_CLI = "1"
cargo run --manifest-path .\Cargo.toml
```

Note: This example is intentionally lightweight and standalone so it can be built without the full workspace.
