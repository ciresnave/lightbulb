# Hugging Face Hub Integration

Lightbulb now supports automatic model downloading from the Hugging Face Hub! You can use model IDs instead of managing local file paths.

## Quick Start

### Loading Models from Hub

Instead of downloading models manually, just provide a model ID:

```rust
use lightbulb::loaders::load_llama;

// Load from Hub - automatically downloads to cache
let (model, cache, config, device, _) = load_llama(
    "meta-llama/Llama-2-7b-hf",  // Model ID
    None,                         // revision (optional)
    Some("f16"),                  // dtype
    true,                         // use_kv_cache
    false,                        // use_flash_attn
)?;
```

### Loading Quantized Models

For GGUF quantized models, specify the filename:

```rust
use lightbulb::loaders::load_gguf;

let (model, config, tokenizer, device, _) = load_gguf(
    "TheBloke/Llama-2-7B-GGUF",           // Model ID
    None,                                  // revision
    Some("llama-2-7b.Q4_K_M.gguf"),       // GGUF filename
)?;
```

## Model Source Detection

Lightbulb automatically detects whether you're providing a local path or Hub model ID:

- **Hub Model ID**: Contains `/` but doesn't start with `.` or `/`
  - Examples: `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`
- **Local Path**: Anything else
  - Examples: `./models/llama`, `/absolute/path/to/model`, `relative/path`

## Caching

Models downloaded from the Hub are cached locally by the `hf-hub` library:
- **Location**: `~/.cache/huggingface/hub/` (Linux/Mac) or `%USERPROFILE%\.cache\huggingface\hub\` (Windows)
- **Behavior**: Downloaded once, reused on subsequent loads
- **Management**: Use `huggingface-cli` to manage cache or delete manually

## Revisions and Branches

Load specific model versions using Git revisions:

```rust
load_llama(
    "meta-llama/Llama-2-7b-hf",
    Some("main"),                    // or "refs/pr/123", commit SHA, etc.
    Some("f16"),
    true,
    false,
)?;
```

## API Functions

### `load_llama()` - SafeTensors Models

Load standard LLaMA-family models from Hub or local path:

```rust
pub fn load_llama(
    model_path_or_id: &str,     // Local path or Hub ID
    revision: Option<&str>,      // Git revision (for Hub)
    dtype: Option<&str>,         // "f32", "f16", "bf16"
    use_kv_cache: bool,          // Enable KV caching
    use_flash_attn: bool,        // Enable flash attention (CUDA only)
) -> Result<(Model, Cache, Config, Device, NameMapper)>
```

### `load_gguf()` - Quantized Models

Load quantized GGUF models from Hub or local file:

```rust
pub fn load_gguf(
    path_or_id: &str,            // Local .gguf path or Hub ID
    revision: Option<&str>,      // Git revision (for Hub)
    gguf_filename: Option<&str>, // GGUF filename (required for Hub)
) -> Result<(QuantModel, Config, Tokenizer, Device, NameMapper)>
```

### `ModelSource` - Explicit Control

For advanced use cases, use `ModelSource` enum directly:

```rust
use lightbulb::hub::{ModelSource, resolve_model_path};

// Parse from string
let source = ModelSource::parse("meta-llama/Llama-2-7b-hf");

// Or create explicitly
let source = ModelSource::Hub {
    model_id: "meta-llama/Llama-2-7b-hf".to_string(),
    revision: Some("main".to_string()),
};

// Resolve to local path (downloads if needed)
let path = resolve_model_path(&source)?;
```

## Examples

### Example 1: Load Popular Model

```rust
// Mistral 7B Instruct
let (model, cache, config, device, _) = load_llama(
    "mistralai/Mistral-7B-Instruct-v0.1",
    None,
    Some("f16"),
    true,
    false,
)?;
```

### Example 2: Quantized Model from TheBloke

```rust
// Llama 2 7B with Q4_K_M quantization
let (model, config, tokenizer, device, _) = load_gguf(
    "TheBloke/Llama-2-7B-GGUF",
    None,
    Some("llama-2-7b.Q4_K_M.gguf"),
)?;
```

### Example 3: Load Specific Version

```rust
// Load from a pull request or commit
let (model, cache, config, device, _) = load_llama(
    "meta-llama/Llama-2-7b-hf",
    Some("refs/pr/42"),  // or a commit SHA
    Some("f16"),
    true,
    false,
)?;
```

### Example 4: Fallback to Local

```rust
// Try Hub first, fallback to local if offline
let model_source = match load_llama("meta-llama/Llama-2-7b-hf", None, Some("f16"), true, false) {
    Ok(loaded) => loaded,
    Err(_) => {
        println!("Hub download failed, using local model");
        load_llama("./models/llama-2-7b", None, Some("f16"), true, false)?
    }
};
```

## Hub Module Reference

The `lightbulb::hub` module provides low-level utilities:

### `download_model(model_id, revision)`
Download a complete model directory from Hub.

### `get_repo(model_id, revision)`
Get a repository handle for manual file downloads.

### `download_safetensors(repo)`
Download all safetensors files (handles sharding automatically).

### `download_tokenizer(repo)`
Download tokenizer.json.

### `download_config(repo)`
Download config.json.

### `download_gguf(repo, filename)`
Download a specific GGUF file.

## Compatibility Notes

- **Backward Compatible**: Existing code using local paths continues to work unchanged
- **Deprecated Functions**: `load_local_llama()` and `load_gguf_llama()` are deprecated in favor of `load_llama()` and `load_gguf()`
- **MLMF Integration**: Hub downloads work seamlessly with MLMF v0.2.1 for optimized loading

## Troubleshooting

### Authentication Required

Some models require authentication. Set your Hugging Face token:

```bash
# Linux/Mac
export HF_TOKEN=your_token_here

# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"
```

Or use `huggingface-cli login`.

### Download Failures

If downloads fail:
1. Check internet connection
2. Verify model ID is correct (case-sensitive)
3. Ensure you have access to gated models
4. Check cache disk space (~4-13GB per 7B model)

### Clearing Cache

```bash
# Linux/Mac
rm -rf ~/.cache/huggingface/hub/

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\"
```

Or use `huggingface-cli delete-cache`.

## Performance Tips

1. **Use Quantized Models**: GGUF models (Q4_K_M) offer 4x smaller size with minimal quality loss
2. **Cache Locally**: First load downloads to cache; subsequent loads are fast
3. **Use WiFi for Initial Download**: 7B models are 13GB+ for fp16, 3.5GB for Q4_K_M
4. **Specify Revision**: Pin to specific commits for reproducibility

## Common Model IDs

### Llama Family
- `meta-llama/Llama-2-7b-hf` (13GB fp16)
- `meta-llama/Llama-2-13b-hf` (25GB fp16)
- `meta-llama/Llama-2-7b-chat-hf` (chat-tuned)
- `meta-llama/Meta-Llama-3-8B` (16GB fp16)

### Mistral Family
- `mistralai/Mistral-7B-v0.1` (13GB fp16)
- `mistralai/Mistral-7B-Instruct-v0.1` (instruction-tuned)
- `mistralai/Mistral-7B-Instruct-v0.2` (newer version)

### Quantized Collections (TheBloke)
- `TheBloke/Llama-2-7B-GGUF`
- `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- `TheBloke/Llama-2-13B-GGUF`

### SmolLM (Small Models)
- `HuggingFaceTB/SmolLM-135M` (270MB)
- `HuggingFaceTB/SmolLM-360M` (720MB)
- `HuggingFaceTB/SmolLM-1.7B` (3.4GB)

## See Also

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [hf-hub Rust Crate](https://docs.rs/hf-hub)
- [Lightbulb Model Loading](./MODEL_LOADING.md)
