# lightbulb (WIP)

An ML inference and training library and server, written in Rust, exposing an
OpenAI-compatible HTTP API.

## Status

Working and under active development. The server builds, starts, and serves
completions; the library carries ~650 unit tests plus integration suites that
drive the real HTTP surface against a local checkpoint.

It is **not** released and the API is **not** stable.

## Backends

**Lightbulb runs its models entirely through Fuel** (`src/model_fuel/`,
crate `fuel` on crates.io). The candlelight backend (a fork of Hugging Face
Candle) that used to be the default was removed entirely on 2026-10-22
(CireSnave: "Remove all dependencies on Candle and Candlelight..."; PR #195).
There is no compile-time choice between backends any more — `fuel-engine` is
now a no-op feature flag kept only so existing callers don't break, and an
ordinary `cargo build` gives you Fuel regardless of whether it's passed.

`fuel` and its sibling crates (`fuel-cpu-backend`, `fuel-transformers`,
`fuel-loaders`, `fuel-ir`, `fuel-inference`, `fuel-cuda-backend`) are
unconditional dependencies, pinned by version on crates.io (`0.25.10` at
this writing, see `Cargo.toml`; first moved off the git source 2026-10-22, PR #196) — see the comments
around the dependency in `Cargo.toml` for the history.

Removing candlelight opened two real, tracked capability gaps (the
`cache_stats`/`scheduler_stats`/`scheduler_queue` admin endpoints have no
live data source on Fuel's path; AWQ quantized-inference parity is
unverified against the portfolio's other quantization code) — see
`docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` for the full accounting and
`ROADMAP.md` for routing.

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

## Experimental switches

| Variable | Default | Effect |
|---|---|---|
| `LIGHTBULB_PREFIX_SHARING` | off | `1`/`true`/`yes`/`on` lets a request splice a live donor's KV blocks for a shared block-aligned prompt prefix instead of prefilling it again. Experimental: no benchmark yet shows a win (Fuel's paged attention has no fused CUDA kernel), so it stays off by default. Prefix-hit *detection* (the `cache_stats` counters) runs either way. Library users set `ApiConfig::model_prefix_sharing` or pass `EngineOptions` to `ModelRunner::start_with_options`; only the binary reads the variable. |
| `LIGHTBULB_ALLOW_OVERSIZE_MODEL` | off | `1`/`true`/`yes`/`on` loads a model even when the serving GPU reports its weights cannot fit. Without it a GPU load whose on-disk weight size exceeds the available VRAM is refused; on the CPU the same finding only logs a warning and loads (free RAM is soft). Library users set `ApiConfig::model_allow_oversize` or `EngineOptions::allow_oversize_model`. |

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

The license texts are `LICENSE-MIT` and `LICENSE-APACHE` at the repository root. Third-party
attributions are in [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) (provenance method and
evidence: [`PROVENANCE.md`](PROVENANCE.md)).

## Offline, CPU-only quickstart

Run a local model entirely offline on CPU through the API server. Put the model in a subdirectory of a models folder (GGUF is the primary format), then start the server bound to loopback without authentication (`--no-auth` is only accepted on loopback when no database is configured):

```bash
LIGHTBULB_MODELS_DIR=./models LIGHTBULB_DEFAULT_MODEL=llama-3b LIGHTBULB_BIND_ADDRESS=127.0.0.1:8080 cargo run --release -- --no-auth
curl http://127.0.0.1:8080/v1/models
```

The earlier `local-llama-gen` subcommand was removed with the candlelight path; `QUICKSTART.md` has the full request examples.

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

## Multi-GPU Inference

**Not available.** The tensor/pipeline/hybrid parallel code (`distributed_cache`, `pipeline_parallel`, `tensor_parallel`) was deleted with the candlelight removal (PR #195) and has not been rebuilt on Fuel. Only the plain `multi_gpu::{config, topology}` types remain (configuration and device-topology discovery); nothing serves a request across several GPUs. The history is in `ROADMAP.md` (M3.6, marked superseded) and `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`; `docs/MULTI_GPU_INTEGRATION.md` and `docs/M3_6_MULTI_GPU_ARCHITECTURE.md` describe the removed design.
