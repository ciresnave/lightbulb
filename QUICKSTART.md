# Quick Start Guide - Lightbulb API & CLI

## Prerequisites

- Rust toolchain (1.70+)
- PostgreSQL database
- (Optional) CUDA toolkit for GPU acceleration

## Step 1: Database Setup

Create a PostgreSQL database and set the connection URL:

```bash
# Create database
createdb lightbulb

# Set connection URL
export DATABASE_URL="postgresql://user:password@localhost/lightbulb"
```

## Step 2: Build Lightbulb

```bash
# CPU-only build
cargo build --release

# GPU build (with CUDA)
cargo build --release --features cuda
```

## Step 3: Initialize Database

The migrations run automatically on server startup, but you can also run them manually:

```bash
sqlx migrate run
```

## Step 4: Create Bootstrap Admin Key

Since you need an admin key to create more keys, manually insert the first one:

```bash
# Generate a key (save this!)
echo "lb-$(openssl rand -hex 32)"
# Example output: lb-a1b2c3d4e5f6...

# Compute SHA-256 hash
echo -n "lb-a1b2c3d4e5f6..." | sha256sum
# Example output: abc123def456...

# Insert into database
psql $DATABASE_URL -c "
INSERT INTO api_keys (key_hash, role) 
VALUES ('abc123def456...', 'admin');
"
```

## Step 5: Start the API Server

```bash
# With default config
cargo run --release

# With custom config (create config.toml)
cargo run --release -- --config config.toml
```

The server will start on `http://localhost:8080` by default.

## Step 6: Create a User API Key

Use your bootstrap admin key to create a user key:

```bash
curl -X POST http://localhost:8080/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer lb-a1b2c3d4e5f6..." \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "expires_in_seconds": 2592000,
    "description": "My user key"
  }'
```

Response:
```json
{
  "api_key": "lb-xyz789...",
  "key_id": "uuid-here",
  "role": "user",
  "expires_at": "2024-12-01T00:00:00Z"
}
```

**Save the `api_key` value!** It's only shown once.

## Step 7: Test with CLI

### Non-streaming mode

```bash
cargo run --bin lightbulb-cli -- --api-key lb-xyz789...
```

### Streaming mode

```bash
cargo run --bin lightbulb-cli -- --api-key lb-xyz789... --stream
```

### With environment variable

```bash
# Set once
export LIGHTBULB_API_KEY="lb-xyz789..."

# Use anywhere
cargo run --bin lightbulb-cli
cargo run --bin lightbulb-cli -- --stream
```

## Step 8: Test with curl

### Non-streaming request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer lb-xyz789..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Streaming request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer lb-xyz789..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Configuration

Create a `config.toml` file:

```toml
[api]
host = "0.0.0.0"
port = 8080
rate_limit_per_minute = 60

models_dir = "/path/to/models"
default_model = "llama-2-7b"
model_max_batch_size = 8
model_context_length = 4096

[database]
url = "postgresql://user:password@localhost/lightbulb"
max_connections = 10

[logging]
level = "info"
```

## Troubleshooting

### Database Connection Errors

```bash
# Check PostgreSQL is running
psql $DATABASE_URL -c "SELECT 1;"

# Check migrations
sqlx migrate info
```

### Model Loading Errors

```bash
# Check model path exists
ls /path/to/models/default

# Check model format is supported
file /path/to/models/default/model.safetensors
```

### API Key Issues

```bash
# List all keys (admin only)
psql $DATABASE_URL -c "SELECT id, role, created_at, expires_at FROM api_keys;"

# Delete expired keys
psql $DATABASE_URL -c "DELETE FROM api_keys WHERE expires_at < NOW();"
```

### CLI Connection Issues

```bash
# Test server is reachable
curl http://localhost:8080/health

# Check API key is valid
curl -H "Authorization: Bearer lb-your-key" \
  http://localhost:8080/v1/models
```

## Next Steps

- Read `CLI.md` for detailed CLI usage
- Read `IMPLEMENTATION_SUMMARY.md` for technical details
- Explore admin endpoints at `/v1/lightbulb/admin/*`
- Check OpenAPI spec (if available) for full API documentation

## Security Notes

- **Never commit API keys** to version control
- Store admin keys securely (password manager, secrets manager)
- Use short expiration times for user keys
- Rotate keys regularly
- Use HTTPS in production (reverse proxy like nginx)
- Configure firewall rules to limit database access
