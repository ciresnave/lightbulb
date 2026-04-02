# Local model setup (CPU-only, offline)

This guide explains how to run a small, local LLaMA-family model offline using Lightbulb (no GPU, no network).

## Folder layout

Place your files under a directory, for example:

models/llama-3b/

- config.json
- tokenizer.json
- model.safetensors (one or more, sharded files are also supported)

Then run the local generator:

- Windows cmd:
  - cargo run -p lightbulb -- local-llama-gen --model-dir models\\llama-3b --prompt "Machine learning is " --sample-len 32

## Obtaining files from Hugging Face Hub

You can download these files with the Hugging Face CLI or by opening the model page and fetching:

- config.json
- tokenizer.json
- model.safetensors (single file or shards)

For tiny models suitable for CPU demo:

- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- HuggingFaceTB/SmolLM2-135M or -360M

Download the files and place them into your models/llama-3b folder (you may rename the folder as you prefer).

## Notes

- Lightbulb loads the model and runs on CPU by default.
- No network calls are made during generation.
- KV cache is enabled to speed up decode even on CPU.
- DType defaults to f32 in this path.
