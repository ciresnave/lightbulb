# Conditional Debug Output Implementation

## Summary

Implemented a flexible, zero-cost debug output system using Rust feature flags and macros. This allows you to selectively enable debug output for specific components without the noise of seeing everything at once.

## What Changed

### New Files
- **`src/debug.rs`** - Defines 8 debug macros for different components
- **`docs/DEBUG.md`** - User guide for the debug system

### Modified Files
- **`Cargo.toml`** - Added 9 feature flags (8 categories + debug-all)
- **`src/lib.rs`** - Added debug module
- **`src/model/parallel_model_manager.rs`** - Converted eprintln! to crate::debug_prefill! and crate::debug_decode!
- **`src/cache/parallel_cache_builder.rs`** - Converted to crate::debug_cache!
- **`src/engine.rs`** - Converted to crate::debug_engine!

## Debug Categories

| Feature Flag      | Macro              | Use Case                                   |
| ----------------- | ------------------ | ------------------------------------------ |
| `debug-prefill`   | `debug_prefill!`   | Prefill phase, chunking, position tracking |
| `debug-decode`    | `debug_decode!`    | Decode phase, token generation             |
| `debug-attention` | `debug_attention!` | Attention computations (not yet converted) |
| `debug-cache`     | `debug_cache!`     | KV cache operations                        |
| `debug-rope`      | `debug_rope!`      | RoPE computations (not yet converted)      |
| `debug-mlp`       | `debug_mlp!`       | MLP layer ops (not yet converted)          |
| `debug-chunking`  | `debug_chunking!`  | Chunk scheduling (not yet converted)       |
| `debug-engine`    | `debug_engine!`    | General engine (IAM, batch assembly)       |
| `debug-all`       | (enables all)      | Everything                                 |

## Usage Examples

```bash
# No debug output (clean!)
cargo run --example test_multi_request --release

# Only prefill debug
cargo run --example test_multi_request --release --features debug-prefill

# Prefill + cache debug (troubleshoot position tracking)
cargo run --example test_multi_request --release --features debug-prefill,debug-cache

# Everything
cargo run --example test_multi_request --release --features debug-all
```

## Performance

**Zero runtime cost when disabled.** The macros use `#[cfg(...)]` so the compiler completely removes the debug code when features aren't enabled. No runtime checks, no function calls, nothing.

## Benefits

1. **See the forest for the trees** - Enable only the component you're debugging
2. **Clean production builds** - No debug output by default
3. **Zero overhead** - Compiler optimizes away disabled macros
4. **Easy to use** - Same syntax as `format!()` macro
5. **Labeled output** - Each message prefixed with `[PREFILL]`, `[DECODE]`, etc.

## Future Work

Still need to convert debug statements in:
- `src/model/custom_attention.rs` → use `debug_attention!` and `debug_rope!`
- `src/model/custom_transformer.rs` → use appropriate macros
- `src/model/mlp_wrapper.rs` → use `debug_mlp!`
- `src/model/chunked_prefill.rs` → use `debug_chunking!`
- Other files with remaining `eprintln!("DEBUG ...")` statements

## Example Output

### Without debug features (clean):
```
🧪 Testing Multi-Request Parallel Generation
✓ Model loaded
🔹 Requests:
  math-1 → "What is 5 + 3? Answer: "
  ...
✓ All requests completed
📝 Results:
  math-1 → "What is 5 + 3? Answer: 10000000"
✨ Test complete
```

### With `--features debug-prefill` (selective):
```
🧪 Testing Multi-Request Parallel Generation
[PREFILL] input_ids shape=[512], batch_size=3, metadata=...
[PREFILL] logits shape=[3, 49152]
[PREFILL] Slot 0 position: 0 -> 11 (advanced by 11 actual tokens)
[PREFILL] Slot 1 position: 0 -> 6 (advanced by 6 actual tokens)
[PREFILL] Slot 2 position: 0 -> 4 (advanced by 4 actual tokens)
...
```

## Testing

Tested with:
- ✅ No features: Clean output, all tests pass
- ✅ `debug-prefill`: Only prefill output shown
- ✅ `debug-decode`: Only decode output shown  
- ✅ Multiple features: Combined output works
- ✅ Compile time: Zero overhead verified
