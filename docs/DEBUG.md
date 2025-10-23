# Conditional Debug Output

The codebase uses feature-gated debug macros to allow selective enabling of debug output by component.

## Available Debug Categories

- `debug-prefill` - Prefill phase operations (chunk scheduling, position tracking)
- `debug-decode` - Decode phase operations (token generation)
- `debug-attention` - Attention layer computations
- `debug-cache` - KV cache operations (position updates, index management)
- `debug-rope` - RoPE (Rotary Position Embedding) computations
- `debug-mlp` - MLP layer operations
- `debug-chunking` - Chunk size calculations and scheduling
- `debug-engine` - General engine operations (IAM caching, batch assembly)
- `debug-all` - Enable all debug categories

## Usage

### Run with no debug output (default)
```bash
cargo run --example test_multi_request --release
```

### Run with specific debug category
```bash
# Only prefill debug output
cargo run --example test_multi_request --release --features debug-prefill

# Only decode debug output
cargo run --example test_multi_request --release --features debug-decode

# Prefill and cache debug output
cargo run --example test_multi_request --release --features debug-prefill,debug-cache
```

### Run with all debug output
```bash
cargo run --example test_multi_request --release --features debug-all
```

### Build with debug features
```bash
# Build library with specific debug features
cargo build --release --features debug-prefill,debug-decode

# Build with all debug features
cargo build --release --features debug-all
```

## Adding Debug Output

To add debug output in your code, use the appropriate macro:

```rust
// In prefill code
crate::debug_prefill!("Processing chunk with {} tokens", chunk_size);

// In decode code
crate::debug_decode!("Generated token: {}", token);

// In cache operations
crate::debug_cache!("Updating slot {} to position {}", slot, pos);
```

## Macro Definitions

All macros are defined in `src/debug.rs`. Each macro:
- Only emits output when its corresponding feature is enabled
- Prefixes output with a category label (e.g., `[PREFILL]`, `[DECODE]`)
- Uses standard Rust formatting syntax

## Performance

Debug macros have **zero runtime cost** when their features are disabled. The compiler completely removes the code during optimization.

## Examples

```bash
# Debug only the chunked prefill scheduling
cargo run --example test_parallel_quality --release --features debug-chunking

# Debug prefill and cache to troubleshoot position tracking
cargo run --example test_multi_request --release --features debug-prefill,debug-cache

# Full debug output for investigation
cargo run --example test_multi_request --release --features debug-all
```
