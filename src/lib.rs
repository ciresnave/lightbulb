pub mod api;
pub mod backend; // Hardware-specific backends (CUDA, ROCm, etc.)
pub mod cache;
pub mod contracts; // Output Contracts — structured responses from small/weak LLMs
pub mod debug;
pub mod engine;
pub mod gguf;
pub mod hardware; // Hardware detection and adaptive configuration
pub mod init; // Hardware-aware system initialization
pub mod kernels;
pub mod loaders;
pub mod lora; // LoRA (Low-Rank Adaptation) support with name mapping (M5.4)
pub mod memory; // Unified memory estimation for models and caches
pub mod model;
pub mod model_fuel; // The Fuel-backed tensor core (D1: parallel path; `model` stays frozen on candlelight until parity)
pub mod multi_gpu; // Multi-GPU inference support (M3.6)
pub mod pruning; // Model pruning utilities (M5)
pub mod quantization; // GGUF quantization utilities
pub mod sampling;
pub mod server;
pub mod tls; // TLS certificate management
pub mod tools; // Tool registry with capability detection (M5.6)

pub async fn hello_generate(prompt: &str) -> anyhow::Result<()> {
    // Trivial Candle call to verify dependency linkage; replace with real text-gen soon
    let a = candlelight::core::Tensor::arange(0f32, 4f32, &candlelight::core::Device::Cpu)?;
    let s = a.sum_all()?.to_scalar::<f32>()?;
    println!("[hello-generate] prompt='{prompt}', candle_sum={s}");
    Ok(())
}

/// Generate a few tokens locally using Candle's LLaMA model and tokenizer.json in the same folder.
/// CPU-only, offline. Expects `config.json`, `tokenizer.json`, and `*.safetensors` in `model_dir`.
pub fn local_llama_generate(
    model_dir: &str,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
) -> anyhow::Result<String> {
    use candlelight::transformers::generation::{LogitsProcessor, Sampling};
    use tokenizers::Tokenizer;

    let (model, mut cache, cfg, device, _name_mapper) =
        crate::loaders::load_local_llama(model_dir, Some("f32"), true, false)?;

    // Load tokenizer.json from model_dir
    let tokenizer_path = std::path::Path::new(model_dir).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!(format!("tokenizer load error: {e}")))?;
    let eos_token_id = cfg.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id("</s>")
            .map(candlelight::transformers::models::llama::LlamaEosToks::Single)
    });
    let mut toks = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    let tokenizer = tokenizer;

    let mut logits_proc = {
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    };

    use candlelight::core::Tensor;
    let mut index_pos = 0usize;
    let mut out = String::new();
    for _ in 0..sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && !toks.is_empty() {
            (1, index_pos)
        } else {
            (toks.len(), 0)
        };
        let ctxt = &toks[toks.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_proc.sample(&logits)?;
        toks.push(next_token);

        match eos_token_id {
            Some(candlelight::transformers::models::llama::LlamaEosToks::Single(eos))
                if next_token == eos =>
            {
                break;
            }
            Some(candlelight::transformers::models::llama::LlamaEosToks::Multiple(ref ids))
                if ids.contains(&next_token) =>
            {
                break;
            }
            _ => {}
        }

        let piece = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        out.push_str(&piece);
    }
    Ok(out)
}

/// Encode and decode using a local tokenizer.json (CPU-only, offline)
pub fn encode_decode_with_tokenizer(
    tokenizer_json: &str,
    text: &str,
) -> anyhow::Result<(Vec<u32>, String)> {
    use tokenizers::Tokenizer;
    let tok = Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let enc = tok
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let decoded = tok
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok((ids, decoded))
}
