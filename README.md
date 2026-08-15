# lightbulb (WIP)

An ML inference and training library and server, written in Rust, exposing an
OpenAI-compatible HTTP API.

## Status

Working and under active development. The server builds, starts, and serves
completions; the library carries ~650 unit tests plus integration suites that
drive the real HTTP surface against a local checkpoint.

It is **not** released and the API is **not** stable.

## Backends

Lightbulb runs its models through one of two engines, chosen at compile time.

| backend | feature | state |
| --- | --- | --- |
| **candlelight** (a fork of Hugging Face Candle) | default | the shipping path |
| **Fuel** (`src/model_fuel/`) | `fuel-engine` | opt-in, **not yet at parity** |

**`default = []`, so an ordinary `cargo build` gives you candlelight.** The port
to Fuel is in progress and is the project's main architectural thread, but
"being ported to Fuel" does not mean Fuel is live — both backends compile today
and candlelight is what serves.

`fuel-core` is an unconditional dependency regardless of which engine is
selected, so Fuel's format readers are available in every build. It is pinned by
git revision, not floating; see the comments around the dependency in
`Cargo.toml` for why.

Other features: `cuda`, `fuel-cuda`, `vulkan`. Op placement on the Fuel path is
per-op and decided by Fuel's planner — enabling a GPU feature does not mean
every op runs there.

## Binaries

- `lightbulb-cli` — client for a running server
- `lightbulb-probe` — measures which chat template a checkpoint actually
  responds to, for checkpoints that do not declare one

## Building

Use `-j 4`.

```sh
cargo build -j 4
cargo test  -j 4 --lib
```

Full build parallelism ICEs rustc on some machines here and surfaces as
misleading `rlib format` or `E0786 paging file` errors that look like unrelated
bugs. This was diagnosed by elimination; `-j 4` is the standing workaround.

## Models

Model directories resolve in this order, a convention shared across these
projects:

1. `$LIGHTBULB_MODELS_DIR`
2. `$MODELS_DIR`
3. `C:\Models`

Pin checkpoint fixtures by revision. A plain `git clone` of a Hugging Face repo
tracks its default branch and moves under you on `git pull`, which silently
invalidates recorded measurements.

## Documentation

Current design and planning work lives in:

- `docs/superpowers/specs/` — design documents
- `docs/superpowers/plans/` — implementation plans
- `docs/API.md` — the HTTP API

**Much of the rest of `docs/` is historical.** Roughly forty `M*_`-prefixed
milestone and completion notes date from the candlelight era and describe an
earlier architecture. They are kept for provenance, not as a description of the
present. `V1_ROADMAP.md` in particular was written against a state that no
longer exists — its Phase 1 is complete despite being labelled as blocking.

> **Removed 2026-08-15:** a ~250-entry "Literature index" and a matching link
> list used to sit here. Every entry pointed into `docs/summaries/`, which has
> never existed in this repository's history — the summaries were local files
> that were never committed. The index was removed rather than left as several
> hundred dead links. Recover it from git history if those files resurface.

## Links

- [Roadmap](./ROADMAP.md)
- [Candle](https://github.com/huggingface/candle) — upstream of the candlelight fork
- [candle-vllm](https://github.com/EricLBuehler/candle-vllm) — design influence (batching, scheduler, paged KV cache)
- [atoma-infer](https://github.com/atoma-network/atoma-infer) — design influence (kernels, sampling, loaders, server)

## License

This project is dual-licensed under either of:

- MIT License
- Apache License, Version 2.0

The dual license is declared in `Cargo.toml` (`license = "MIT OR Apache-2.0"`).

> ⚠️ **The license files are missing.** This section referenced `LICENSE-MIT`
> and `LICENSE-APACHE` at the repository root and `docs/THIRD_PARTY_NOTICES.md`;
> none of the three exists (checked 2026-08-15). The dual-license *intent* is
> corroborated by `Cargo.toml`, but the texts themselves need to be added before
> any release, and third-party attributions have not been collected.

## Offline, CPU-only quickstart

Run a local LLaMA-family model entirely offline on CPU. Prepare a folder with `config.json`, `tokenizer.json`, `tokenizer_config.json`, and one or more `model.safetensors` files. Then run:

<!-- Previously pointed at `docs/Local model setup.md`, which does not exist
     (checked 2026-08-15). `tokenizer_config.json` added to the list above: it
     carries the chat template, and a checkpoint without it falls back to a
     family guess rather than the template the model was trained on. -->


- Windows cmd:
  - cargo run -p lightbulb -- local-llama-gen --model-dir models\llama-3b --prompt "Hello" --sample-len 16

This path avoids any network calls and doesn't require GPU drivers, which is ideal for CI and deterministic tests.

### Why this path exists

- Deterministic dev/CI without GPU or drivers
- Works in air-gapped or rate-limited environments (no network)
- Easier debugging and tracing on CPU

## TLS/SSL Configuration

Lightbulb's API server supports comprehensive TLS/SSL configuration for secure HTTPS deployments. The implementation includes automatic certificate management, HTTP-to-HTTPS redirects, and support for multiple deployment scenarios.

### Quick Start: Self-Signed Certificates (Development)

For development and testing, enable self-signed certificates:

```rust
use lightbulb::api::{ApiConfig, TlsConfig, CertificateSource};

let config = ApiConfig {
    bind_address: "127.0.0.1:8080".to_string(),
    tls: TlsConfig {
        enabled: true,
        cert_source: CertificateSource::SelfSigned {
            cache_dir: "./certs".to_string(),
        },
        https_bind_address: Some("127.0.0.1:8443".to_string()),
        force_https: true,
    },
    // ... other config
};
```

### Certificate Management Strategies

Lightbulb supports four deployment scenarios:

#### 1. HTTP-Only (Development/Testing)

```rust
let tls = TlsConfig {
    enabled: false,
    ..Default::default()
};
```

**Use case:** Local development, CI/CD environments, internal testing

#### 2. Self-Signed Certificates (Internal/Development)

```rust
let tls = TlsConfig {
    enabled: true,
    cert_source: CertificateSource::SelfSigned {
        cache_dir: "./certs".to_string(),
    },
    https_bind_address: Some("0.0.0.0:8443".to_string()),
    force_https: true,
};
```

**Features:**
- Automatic certificate generation with Subject Alternative Names
- Certificate caching (30-day validity with auto-renewal)
- Suitable for internal networks and development environments
- No external dependencies

**Use case:** Development servers, internal tools, testing environments

#### 3. Existing Certificates (Production)

```rust
let tls = TlsConfig {
    enabled: true,
    cert_source: CertificateSource::Existing {
        cert_path: "/etc/letsencrypt/live/example.com/fullchain.pem".to_string(),
        key_path: "/etc/letsencrypt/live/example.com/privkey.pem".to_string(),
    },
    https_bind_address: Some("0.0.0.0:443".to_string()),
    force_https: true,
};
```

**Features:**
- Use certificates from Let's Encrypt, commercial CAs, or internal PKI
- Automatic certificate validation with 30-day renewal buffer
- Supports PEM format certificates

**Use case:** Production deployments with reverse proxies (nginx, Caddy, Traefik)

#### 4. ACME/Let's Encrypt (Production - Framework Ready)

```rust
let tls = TlsConfig {
    enabled: true,
    cert_source: CertificateSource::Acme {
        domain: "api.example.com".to_string(),
        email: "admin@example.com".to_string(),
        cache_dir: "/var/lib/lightbulb/certs".to_string(),
        production: true,
    },
    https_bind_address: Some("0.0.0.0:443".to_string()),
    force_https: true,
};
```

**Features:**
- HTTP-01 challenge handler infrastructure in place
- ACME account and order management framework
- Falls back to self-signed certificates if ACME acquisition fails
- Certificate renewal logic with 30-day buffer

**Status:** Framework implemented, full ACME integration pending. Currently falls back to self-signed certificates while ACME implementation is completed.

**Use case:** Production deployments without reverse proxy, direct internet-facing servers

### HTTP-to-HTTPS Redirect

When `force_https: true`, Lightbulb automatically redirects HTTP requests to HTTPS:

```rust
let tls = TlsConfig {
    enabled: true,
    force_https: true,
    https_bind_address: Some("0.0.0.0:8443".to_string()),
    cert_source: CertificateSource::SelfSigned {
        cache_dir: "./certs".to_string(),
    },
};
```

**Features:**
- Preserves original request path and query parameters
- Supports `X-Forwarded-Proto` header for reverse proxy setups
- Returns 301 Permanent Redirect for SEO compliance

### Dual HTTP/HTTPS Servers

When TLS is enabled, Lightbulb runs both HTTP and HTTPS servers simultaneously:

```rust
// HTTP server on :8080 (redirects to HTTPS if force_https=true)
// HTTPS server on :8443 (serves API requests securely)
```

This allows:
- HTTP health checks while serving HTTPS traffic
- ACME HTTP-01 challenge responses on port 80
- Gradual migration from HTTP to HTTPS

### Configuration Reference

```rust
pub struct TlsConfig {
    /// Enable TLS/SSL support
    pub enabled: bool,
    
    /// Certificate source strategy
    pub cert_source: CertificateSource,
    
    /// HTTPS bind address (None = derive from HTTP address with port 8443)
    pub https_bind_address: Option<String>,
    
    /// Force HTTP-to-HTTPS redirects
    pub force_https: bool,
}

pub enum CertificateSource {
    /// Use existing certificate files
    Existing {
        cert_path: String,
        key_path: String,
    },
    
    /// Generate self-signed certificates
    SelfSigned {
        cache_dir: String,
    },
    
    /// Acquire certificates via ACME protocol (Let's Encrypt)
    Acme {
        domain: String,
        email: String,
        cache_dir: String,
        production: bool,
    },
}
```

### Certificate Validation and Renewal

Lightbulb automatically validates certificates and triggers renewal when:
- Certificate is expired
- Certificate expires within 30 days
- Certificate file is missing or invalid

For self-signed certificates, renewal regenerates the certificate automatically. For existing certificates, check your certificate provider's renewal process.

### Security Considerations

1. **File Permissions:** Ensure certificate and key files have restrictive permissions:
   ```bash
   chmod 600 /path/to/privkey.pem
   chmod 644 /path/to/fullchain.pem
   ```

2. **Self-Signed Certificates:** Not suitable for public-facing production servers. Browsers will show security warnings. Use only for:
   - Development environments
   - Internal networks with trusted clients
   - Testing scenarios

3. **Reverse Proxy Setup:** For production, consider using a reverse proxy (nginx, Caddy) to handle TLS termination:
   - Reverse proxy manages certificates (Let's Encrypt, commercial CA)
   - Lightbulb runs with TLS disabled behind the proxy
   - Proxy forwards traffic to Lightbulb's HTTP endpoint

4. **Port Requirements:** 
   - Port 80 required for ACME HTTP-01 challenges
   - Port 443 standard for HTTPS
   - Custom ports supported (e.g., 8443 for development)

### Testing TLS Configuration

```bash
# Test HTTPS endpoint
curl -k https://localhost:8443/health

# Test HTTP-to-HTTPS redirect
curl -v http://localhost:8080/health
# Should return 301 redirect to https://localhost:8443/health

# Verify certificate details
openssl s_client -connect localhost:8443 -showcerts
```

### Troubleshooting

**Issue:** "Certificate validation failed"
- **Solution:** Check certificate file paths and permissions. Ensure certificate is valid and not expired.

**Issue:** "Address already in use"
- **Solution:** Another service is using the HTTPS port. Change `https_bind_address` or stop the conflicting service.

**Issue:** "ACME challenge failed"
- **Solution:** Ensure port 80 is accessible from the internet. Check firewall rules and DNS configuration.

**Issue:** Browser security warning with self-signed certificates
- **Solution:** This is expected behavior. For development, add exception in browser. For production, use certificates from a trusted CA.

## Multi-GPU Inference (M3.6)

Lightbulb supports distributed inference across multiple GPUs using tensor parallelism, pipeline parallelism, or hybrid strategies. All model architectures (`BatchedTransformer`) benefit from multi-GPU support transparently.

### Quick Start: 2-GPU Tensor Parallelism

```rust
use lightbulb::model::{BatchedTransformer, BatchedTransformerConfig};
use lightbulb::multi_gpu::config::{MultiGPUConfig, ParallelismMode};

// Create config with multi-GPU enabled
let mut config = BatchedTransformerConfig::llama_7b();

// Enable 2-GPU tensor parallelism
let multi_gpu = MultiGPUConfig::manual(
    ParallelismMode::TensorParallel { world_size: 2 },
    2,
)?;
config.multi_gpu = Some(multi_gpu);

// Load model (weights sharded automatically)
let mut model = BatchedTransformer::new(config, vb)?;

// Initialize distributed cache
model.enable_distributed_cache(4, 2048)?;

// Use normally - multi-GPU is transparent!
let logits = model.forward(&input_ids, &mut cache_builder, &mut caches, &metadata)?;
```

### Supported Strategies

| Strategy          | GPUs | Use Case                                 | Target Speedup |
| ----------------- | ---- | ---------------------------------------- | -------------- |
| Tensor Parallel   | 2-4  | High throughput, model fits when sharded | 1.7-3.2×       |
| Pipeline Parallel | 2-8  | Very large models, memory-bound          | 3.5-6.5×       |
| Hybrid (2×4)      | 8    | Maximum scalability                      | ~6×            |

### Pipeline Parallelism Example (4 GPUs)

```rust
let multi_gpu = MultiGPUConfig::manual(
    ParallelismMode::PipelineParallel {
        num_stages: 4,
        micro_batch_size: 2,
    },
    4,
)?;
```

Pipeline parallelism distributes transformer layers across GPUs (e.g., 40 layers → 4 GPUs = 10 layers/GPU). The `forward_layers()` method enables explicit layer-range processing:

```rust
// GPU 0: Process layers 0-9
let hidden = model.forward_layers(&hidden_states, 0, 10, index_pos, 
                                   &mut cache_builder, &mut caches, &metadata)?;

// Transfer to GPU 1 and continue...
```

### Hybrid Parallelism (8+ GPUs)

Combine tensor and pipeline parallelism for maximum scalability:

```rust
let multi_gpu = MultiGPUConfig::manual(
    ParallelismMode::Hybrid {
        tensor_world_size: 2,  // 2-way tensor parallel per stage
        pipeline_stages: 4,     // 4 pipeline stages
        micro_batch_size: 2,
    },
    8, // total: 2 × 4 = 8 GPUs
)?;
```

### Architecture Support

Multi-GPU works with **all** `BatchedTransformer` architectures:
- Llama, Llama2, Llama3 (`BatchedLlama`)
- Mistral (`BatchedMistral`)
- Gemma (`BatchedGemma`)
- Phi, Qwen (via `BatchedTransformerConfig`)

Configuration is architecture-agnostic - just set `config.multi_gpu` before model creation.

### Testing

Multi-GPU requires hardware. Tests are gated with `#[ignore]`:

```bash
# Requires 2+ GPUs
cargo test --test multi_gpu_validation -- --ignored --test-threads=1
```

### Documentation

- **Integration Guide:** `docs/MULTI_GPU_INTEGRATION.md` - Complete API reference and examples
- **Testing Guide:** `tests/MULTI_GPU_TESTING.md` - Hardware requirements and test categories
- **Architecture:** `docs/M3_6_MULTI_GPU_ARCHITECTURE.md` - Design and implementation details
- **ROADMAP:** See M3.6 milestone for performance targets and future work
