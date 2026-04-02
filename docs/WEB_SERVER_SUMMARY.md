# web-server-abstraction: Quick Reference

**Status**: 📋 Integrating  
**Version**: 1.0.2  
**License**: MIT OR Apache-2.0  
**Documentation**: <https://docs.rs/web-server-abstraction>

---

## What It Is

Production-ready web server abstraction providing unified API across multiple Rust frameworks (Axum, Actix-Web, Warp, Rocket, Salvo, Poem) with enterprise security, monitoring, and configuration.

---

## Why Lightbulb Needs It

- **REST API**: Standard inference interface (`/v1/completions`)
- **WebSocket**: Real-time streaming, MCP transport
- **MCP Hosting**: Provides HTTP/WebSocket for mocopr-server
- **Production**: Security (CSRF, TLS), monitoring, scalability
- **Dashboards**: Web UI for metrics, training, sentience

---

## Key Features

### Framework Agnostic

- Axum (high-performance)
- Actix-Web (actor-based)
- Warp, Rocket, Salvo, Poem
- Switch backends without code changes

### Security

- CSRF protection
- XSS prevention
- TLS/SSL encryption
- Rate limiting
- Input sanitization
- Security event monitoring

### Monitoring

- Prometheus metrics
- Distributed tracing
- Health checks
- Performance stats
- Alerting system

### Configuration

- Compatible with distributed-config
- YAML/JSON/TOML support
- Environment variables
- Hot-reload capability

---

## Architecture

```text
┌─────────────────────────────────────────┐
│    web-server-abstraction              │
│  ┌──────────┐      ┌──────────┐        │
│  │ REST API │      │ WebSocket│        │
│  │          │      │          │        │
│  │ /v1/*    │      │ /ws/*    │        │
│  └────┬─────┘      └────┬─────┘        │
│       │                 │              │
│  ┌────┴─────────────────┴─────┐        │
│  │  Security + Monitoring      │        │
│  └─────────────────────────────┘        │
└─────────────────┬───────────────────────┘
                  │
         ┌────────▼────────┐
         │  Lightbulb Core │
         │  + MCP Server   │
         └─────────────────┘
```

---

## Core API

### Server Creation

```rust
use web_server_abstraction::WebServer;

let server = WebServer::with_axum_adapter()
    .get("/health", health_handler)
    .post("/v1/completions", inference_handler)
    .websocket("/ws/mcp")
    .middleware(CsrfMiddleware::new())
    .bind("0.0.0.0:8080")
    .await?;

server.run().await?;
```

### Routing Methods

- `route(path, method, handler)` - Generic
- `get/post/put/delete/patch/head/options` - HTTP methods
- `param_route("/users/:id", ...)` - Path parameters
- `wildcard_route("/static/*file", ...)` - Wildcards
- `websocket(path)` - WebSocket support

### Middleware

- CSRF protection
- Rate limiting
- CORS
- Compression (gzip, brotli)
- Security headers
- Authentication

---

## Lightbulb API Design

### M4 - Scheduling & Inference

```text
POST   /v1/completions      - Submit inference request
GET    /v1/models           - List available models
GET    /v1/queue/status     - Queue metrics
PUT    /v1/requests/:id/priority - Adjust priority
DELETE /v1/requests/:id     - Cancel request
```

### M7 - Sentience

```text
GET  /v1/sentience/identity      - Identity graph
GET  /v1/sentience/motivations   - Motivational hierarchy
POST /v1/sentience/introspect    - Introspection query
GET  /v1/sentience/partnerships  - Partnership metrics
POST /v1/sentience/capabilities/:cap/unlock - Capability request
```

### M8 - Training

```text
POST   /v1/training/jobs            - Start training
GET    /v1/training/jobs/:id        - Job status
POST   /v1/training/jobs/:id/pause  - Pause job
POST   /v1/training/jobs/:id/resume - Resume job
DELETE /v1/training/jobs/:id        - Cancel job
GET    /v1/training/jobs/:id/checkpoints - List checkpoints
GET    /v1/training/patterns        - Pattern library
```

### Monitoring

```text
GET /health          - Basic health check
GET /health/detailed - Detailed health
GET /metrics         - Prometheus metrics
```

### WebSocket Endpoints

```text
/ws/mcp      - MCP protocol (mocopr integration)
/ws/stream   - Streaming inference responses
/ws/training - Real-time training progress
/ws/metrics  - Live metrics updates
```

---

## Integration with Infrastructure

| Crate                  | Integration                        |
| ---------------------- | ---------------------------------- |
| **distributed-config** | WebServerConfig uses ConfigManager |
| **mocopr**             | WebSocket/HTTP transports for MCP  |
| **coalescent**         | Multi-agent API endpoints          |
| **system-analysis**    | Hardware capabilities API          |
| **auto-discovery**     | Service registration               |
| **infra-network**      | Network topology visualization     |

---

## Configuration Example

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

security:
  csrf_protection: true
  tls:
    enabled: true
    cert_path: "/path/to/cert.pem"
    key_path: "/path/to/key.pem"

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  health_checks_enabled: true

middleware:
  cors:
    enabled: true
    allow_origins: ["https://dashboard.example.com"]
  rate_limiting:
    enabled: true
    requests_per_minute: 100
  compression:
    enabled: true
    gzip: true
    brotli: true
```

---

## Monitoring & Metrics

### Health Checks

```rust
monitoring.add_health_check(HealthCheck::new("scheduler", || async {
    scheduler.is_healthy().await
}));
```

### Metrics

- **Counters**: requests_total, tokens_generated_total
- **Gauges**: active_requests, queue_depth
- **Histograms**: ttft_seconds, total_latency_seconds

### Tracing

```rust
let trace_ctx = TraceContext::from_request(&req);
let span = trace_ctx.start_span("inference_request");
// ... process request ...
span.end();
```

---

## Security Features

- **CSRF Protection**: Secure token generation
- **XSS Prevention**: Input sanitization
- **SQL Injection**: Parameterized queries
- **TLS/SSL**: Modern cipher suites
- **Rate Limiting**: Token bucket algorithm
- **Content Security Policy**: Header-based protection
- **Security Monitoring**: Event logging

---

## Time Savings

| Approach                        | Time            |
| ------------------------------- | --------------- |
| **Build from Scratch**          | 17-24 weeks     |
| **With web-server-abstraction** | 6 weeks         |
| **Net Savings**                 | **11-18 weeks** |

---

## Integration Phases

1. **Foundation** (Week 1): Basic server, health checks
2. **Core API** (Week 2): Inference endpoints, WebSocket
3. **Security** (Week 3): TLS, auth, rate limiting
4. **MCP Integration** (Week 4): WebSocket MCP, remote clients
5. **Dashboard** (Week 5): Web UI, metrics visualization
6. **Advanced** (Week 6): Tracing, profiling, optimization

---

## Strategic Value

✅ **Framework Agnostic** - Not locked into specific framework  
✅ **Production Security** - CSRF, XSS, TLS out-of-box  
✅ **Configuration Synergy** - Integrates with distributed-config  
✅ **Monitoring Excellence** - Prometheus, tracing, health checks  
✅ **Developer Experience** - Standard REST, WebSocket, OpenAPI  
✅ **Scalability** - Async-first, load balancing ready

---

## ROADMAP Impact

- **M4 (Scheduling)**: REST API, metrics dashboard
- **M6 (Tool Registry)**: OpenAPI serving
- **M7 (Sentience)**: Partnership interfaces, introspection API
- **M8 (Training)**: Training dashboard, remote control

---

## Critical Synergies

### With mocopr

```rust
// MCP server hosted on web server
server.websocket("/ws/mcp")
    .route("/ws/mcp", HttpMethod::WEBSOCKET, |ws| async {
        mcp_server.handle_websocket(ws).await
    });
```

**Result**: Complete communication stack (MCP + HTTP/WebSocket)

### With distributed-config

```rust
let config_manager = ConfigManager::new();
let web_config: WebServerConfig = config_manager.get("server")?;
let server = WebServer::with_config(web_config);
```

**Result**: Unified configuration across all components

---

## Use Cases

### 1. Remote Inference Request

```bash
curl -X POST https://lightbulb.ai/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```

### 2. WebSocket Streaming

```javascript
const ws = new WebSocket('wss://lightbulb.ai/ws/stream');
ws.onmessage = (event) => console.log(event.data);
```

### 3. Metrics Monitoring

```bash
curl https://lightbulb.ai/metrics
```

### 4. Training Dashboard

```javascript
const ws = new WebSocket('wss://lightbulb.ai/ws/training');
ws.onmessage = (event) => updateChart(JSON.parse(event.data));
```

---

## Next Steps

### Immediate

1. ✅ Add web-server-abstraction to Cargo.toml
2. 📋 Create lightbulb-web crate
3. 📋 Implement basic server
4. 📋 Health endpoint

### Short-Term

1. 📋 Inference API
2. 📋 WebSocket streaming
3. 📋 Security middleware
4. 📋 Metrics endpoints

### Medium-Term

1. 📋 MCP integration
2. 📋 Web dashboard
3. 📋 Training API
4. 📋 Sentience API

---

## Summary

**web-server-abstraction completes Lightbulb's communication infrastructure:**

- 🌐 **Standard Web Interface**: REST API for inference
- 🔌 **Real-Time Streaming**: WebSocket for live updates
- 🔒 **Production Security**: CSRF, TLS, rate limiting
- 📊 **Complete Observability**: Metrics, tracing, health
- ⚙️ **Unified Configuration**: distributed-config integration
- 🚀 **MCP Foundation**: HTTP/WebSocket transports for mocopr

**Status**: Essential for production deployment  
**Priority**: HIGH  
**Timeline**: 6 weeks  
**Savings**: 11-18 weeks

---

**See**: docs/WEB_SERVER_INTEGRATION.md for detailed integration guide
