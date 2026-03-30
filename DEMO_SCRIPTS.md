# Demo Scripts - Complete Guide

## 📚 Overview

This directory contains automated demo scripts that set up everything needed to test Lightbulb:

- ✅ PostgreSQL database in Docker
- ✅ Database migrations
- ✅ Secure secret generation
- ✅ Bootstrap admin API key
- ✅ Project compilation
- ✅ Environment configuration

## 🎯 Available Scripts

### 1. `demo-setup.ps1` / `demo-setup.sh`

**Purpose**: One-time setup of all infrastructure and dependencies

**What it does**:
- Starts PostgreSQL 15 in Docker container
- Generates secure random passwords
- Creates database and runs migrations
- Generates bootstrap admin API key (with SHA-256 hash)
- Inserts admin key into database
- Builds Lightbulb (release mode)
- Builds CLI binary
- Saves all secrets to `.demo-secrets.env`

**Usage**:
```powershell
# Windows
.\demo-setup.ps1

# Linux/Mac
./demo-setup.sh
```

**Options**:
- `-CleanStart` / `--clean-start` - Remove existing setup and start fresh
- `-PostgresPort` / `--postgres-port` - Custom PostgreSQL port (default: 5432)
- `-ApiPort` / `--api-port` - Custom API server port (default: 8080)

### 2. `demo-run.ps1` (PowerShell only)

**Purpose**: Complete automated demo including server startup

**What it does**:
- Runs setup if not already done
- Loads secrets from `.demo-secrets.env`
- Starts API server in background PowerShell job
- Waits for server to become ready
- Creates a user API key automatically
- Displays test commands
- Shows live server logs

**Usage**:
```powershell
.\demo-run.ps1
```

**Options**:
- `-StopAfterSetup` - Run setup only, don't start server
- `-WaitSeconds <n>` - Seconds to wait before checking server (default: 10)

## 🚀 Quick Start

### Simplest Path (PowerShell)

```powershell
# Run everything
.\demo-run.ps1

# Start chatting
cargo run --release --bin lightbulb-cli -- --api-key $env:LIGHTBULB_USER_KEY --stream
```

### Standard Path (All platforms)

```bash
# 1. Setup
./demo-setup.sh  # or .\demo-setup.ps1

# 2. Load secrets
source .demo-secrets.env  # or: . ./.demo-secrets.env (PowerShell)

# 3. Start server (in one terminal)
cargo run --release

# 4. Create user key (in another terminal)
source .demo-secrets.env
curl -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"role":"user"}' | jq -r '.api_key'

# 5. Test with CLI
export LIGHTBULB_USER_KEY="lb-your-key"
cargo run --release --bin lightbulb-cli -- --api-key $LIGHTBULB_USER_KEY
```

## 📁 Generated Files

### `.demo-secrets.env`

**CRITICAL**: This file contains all secrets. Never commit it!

Contents:
```env
DATABASE_URL=postgresql://lightbulb:randompass@localhost:5432/lightbulb
LIGHTBULB_ADMIN_KEY=lb-64-hex-chars...
LIGHTBULB_ADMIN_KEY_HASH=sha256-hash...
LIGHTBULB_API_URL=http://localhost:8080
LIGHTBULB_USER_KEY=lb-64-hex-chars...  # Added after first user creation
```

**Security**:
- ✅ Added to `.gitignore`
- ✅ File permissions: 600 (Unix)
- ❌ Never commit
- ❌ Never share publicly

## 🐳 Docker Container

**Name**: `lightbulb-postgres-demo`

**Details**:
- Image: `postgres:15-alpine`
- Port: 5432 (configurable)
- Database: `lightbulb`
- User: `lightbulb`
- Password: Random 20-character string

**Management**:
```bash
# Stop
docker stop lightbulb-postgres-demo

# Start
docker start lightbulb-postgres-demo

# Logs
docker logs -f lightbulb-postgres-demo

# Connect
docker exec -it lightbulb-postgres-demo psql -U lightbulb -d lightbulb

# Remove (deletes all data!)
docker rm lightbulb-postgres-demo
```

## 🔑 API Keys

### Bootstrap Admin Key

- Created during setup
- Role: `admin`
- No expiration
- Stored in `.demo-secrets.env`
- Hash stored in database

### User Keys

Created via admin endpoint:
```bash
curl -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "expires_in_seconds": 86400,
    "description": "24-hour test key"
  }'
```

## 🧪 Testing Scenarios

### Scenario 1: CLI Chat (Non-streaming)

```bash
source .demo-secrets.env
cargo run --release --bin lightbulb-cli -- \
  --api-key $LIGHTBULB_USER_KEY
```

### Scenario 2: CLI Chat (Streaming)

```bash
cargo run --release --bin lightbulb-cli -- \
  --api-key $LIGHTBULB_USER_KEY \
  --stream
```

### Scenario 3: Direct HTTP (curl)

**Non-streaming**:
```bash
curl -X POST $LIGHTBULB_API_URL/v1/chat/completions \
  -H "Authorization: Bearer $LIGHTBULB_USER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"Hello!"}],
    "stream": false
  }'
```

**Streaming**:
```bash
curl -N -X POST $LIGHTBULB_API_URL/v1/chat/completions \
  -H "Authorization: Bearer $LIGHTBULB_USER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"Count to 10"}],
    "stream": true
  }'
```

### Scenario 4: Admin Operations

**List keys**:
```sql
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT id, role, created_at, expires_at FROM api_keys;"
```

**Check rate limits**:
```sql
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT * FROM api_key_usage ORDER BY window_start DESC LIMIT 5;"
```

**View audit log**:
```sql
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 5;"
```

## 🔧 Troubleshooting

### "Docker not found"

Install Docker Desktop:
- Windows: https://www.docker.com/products/docker-desktop
- Mac: https://www.docker.com/products/docker-desktop
- Linux: https://docs.docker.com/engine/install/

### "Port 5432 already in use"

Use custom port:
```bash
./demo-setup.sh --postgres-port 5433
```

Then update connection strings to use port 5433.

### "Build failed"

Clean and retry:
```bash
cargo clean
./demo-setup.sh
```

### "Server won't start"

Check logs:
```bash
RUST_LOG=debug cargo run --release
```

Check port availability:
```bash
# Windows
netstat -ano | findstr :8080

# Linux/Mac
lsof -i :8080
```

### "Migrations failed"

Run manually:
```bash
source .demo-secrets.env
cargo install sqlx-cli --no-default-features --features postgres
sqlx migrate run
```

### "API key doesn't work"

Verify in database:
```bash
# Get your key hash
echo -n "lb-your-key" | sha256sum

# Check database
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT id, role, key_hash FROM api_keys;"
```

## 🧹 Cleanup

### Keep data, restart

```bash
# Just restart container
docker restart lightbulb-postgres-demo

# Reload secrets
source .demo-secrets.env
```

### Full cleanup

```bash
# PowerShell
.\demo-setup.ps1 -CleanStart

# Bash
./demo-setup.sh --clean-start
```

### Manual cleanup

```bash
# Stop and remove container
docker stop lightbulb-postgres-demo
docker rm lightbulb-postgres-demo

# Remove secrets
rm .demo-secrets.env

# Clean build
cargo clean
```

## 📖 Related Documentation

- **Full Demo Guide**: `DEMO.md` - Comprehensive documentation
- **CLI Usage**: `CLI.md` - CLI-specific instructions
- **Quick Start**: `QUICKSTART.md` - Manual setup guide
- **Implementation**: `IMPLEMENTATION_SUMMARY.md` - Technical details

## 💡 Tips

1. **Multiple environments**: Use different ports for parallel testing
   ```bash
   ./demo-setup.sh --postgres-port 5433 --api-port 8081
   ```

2. **Persistent data**: Container persists between restarts
   ```bash
   docker stop lightbulb-postgres-demo  # Data preserved
   docker start lightbulb-postgres-demo  # Same data
   ```

3. **Quick key creation**: Save admin key for reuse
   ```bash
   source .demo-secrets.env
   alias create-key='curl -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" -H "Content-Type: application/json" -d'
   
   create-key '{"role":"user"}'
   ```

4. **CLI shortcuts**: Create shell aliases
   ```bash
   alias lb-cli='cargo run --release --bin lightbulb-cli --'
   lb-cli --api-key $LIGHTBULB_USER_KEY --stream
   ```

## ⚡ One-Liners

**Complete setup and test**:
```bash
# Bash
./demo-setup.sh && \
source .demo-secrets.env && \
(cargo run --release &) && \
sleep 10 && \
export LIGHTBULB_USER_KEY=$(curl -s -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" -H "Content-Type: application/json" -d '{"role":"user"}' | jq -r '.api_key') && \
cargo run --release --bin lightbulb-cli -- --api-key $LIGHTBULB_USER_KEY
```

**Quick restart**:
```bash
docker restart lightbulb-postgres-demo && source .demo-secrets.env
```

**Quick cleanup**:
```bash
docker stop lightbulb-postgres-demo && docker rm lightbulb-postgres-demo && rm .demo-secrets.env
```
