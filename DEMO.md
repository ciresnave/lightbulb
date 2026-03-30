# Lightbulb Demo Guide

Complete automated demo setup for testing Lightbulb with PostgreSQL, API keys, and CLI.

## 🚀 Quick Start

### Windows (PowerShell)

```powershell
# One-command setup and run
.\demo-setup.ps1
.\demo-run.ps1
```

### Linux/Mac (Bash)

```bash
# Make scripts executable
chmod +x demo-setup.sh

# One-command setup
./demo-setup.sh

# Start server manually
source .demo-secrets.env
cargo run --release
```

## 📋 What the Demo Does

The automated demo setup:

1. ✅ **PostgreSQL in Docker** - Starts PostgreSQL 15 in a container
2. ✅ **Database Creation** - Creates `lightbulb` database
3. ✅ **Migrations** - Runs all database migrations automatically
4. ✅ **Secret Generation** - Creates secure random passwords and API keys
5. ✅ **Bootstrap Admin Key** - Generates and stores admin API key with SHA-256 hash
6. ✅ **Project Build** - Compiles Lightbulb and CLI in release mode
7. ✅ **Environment Setup** - Saves all configuration to `.demo-secrets.env`

## 📦 Prerequisites

- **Docker** - For PostgreSQL container
- **Rust** - 1.70+ with Cargo
- **PowerShell** 5.1+ (Windows) or Bash (Linux/Mac)
- (Optional) **sqlx-cli** - For manual migrations: `cargo install sqlx-cli`

## 🔧 Setup Scripts

### `demo-setup.ps1` / `demo-setup.sh`

**Automated setup script that configures everything needed.**

#### Options

**PowerShell:**
```powershell
.\demo-setup.ps1 [-CleanStart] [-PostgresPort <port>] [-ApiPort <port>]
```

**Bash:**
```bash
./demo-setup.sh [--clean-start] [--postgres-port <port>] [--api-port <port>]
```

#### Examples

**Fresh setup:**
```powershell
.\demo-setup.ps1
```

**Clean existing and restart:**
```powershell
.\demo-setup.ps1 -CleanStart
```

**Custom ports:**
```bash
./demo-setup.sh --postgres-port 5433 --api-port 8081
```

#### What Gets Created

- **Docker Container**: `lightbulb-postgres-demo`
  - PostgreSQL 15 Alpine
  - Database: `lightbulb`
  - User: `lightbulb`
  - Random secure password
  - Port: 5432 (default)

- **Secrets File**: `.demo-secrets.env`
  ```env
  DATABASE_URL=postgresql://lightbulb:password@localhost:5432/lightbulb
  LIGHTBULB_ADMIN_KEY=lb-a1b2c3d4e5f6...
  LIGHTBULB_ADMIN_KEY_HASH=sha256hash...
  LIGHTBULB_API_URL=http://localhost:8080
  ```

- **Database Tables**:
  - `api_keys` - API key hashes and metadata
  - `audit_logs` - Request audit trail
  - `sessions` - Session management
  - `api_key_usage` - Rate limiting counters

- **Bootstrap Admin Key**:
  - Inserted into `api_keys` table
  - Role: `admin`
  - No expiration

### `demo-run.ps1` (PowerShell only)

**Complete demo runner that starts the server and creates a user key.**

```powershell
.\demo-run.ps1
```

This script:
1. Runs setup if not already done
2. Starts API server in background job
3. Waits for server to be ready
4. Creates a user API key automatically
5. Displays test commands
6. Shows live server logs

#### Options

```powershell
.\demo-run.ps1 [-StopAfterSetup] [-WaitSeconds <seconds>]
```

## 🧪 Testing the Demo

After running setup, you have several testing options:

### Option 1: Interactive CLI

**Non-streaming:**
```bash
# Load secrets first
source .demo-secrets.env  # or: . ./.demo-secrets.env (PowerShell)

# Run CLI
cargo run --release --bin lightbulb-cli -- --api-key $LIGHTBULB_USER_KEY
```

**Streaming:**
```bash
cargo run --release --bin lightbulb-cli -- --api-key $LIGHTBULB_USER_KEY --stream
```

### Option 2: curl Commands

**Create user key:**
```bash
source .demo-secrets.env

curl -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"role":"user"}'
```

**Test chat completion:**
```bash
USER_KEY="lb-your-user-key"

curl -X POST $LIGHTBULB_API_URL/v1/chat/completions \
  -H "Authorization: Bearer $USER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Test streaming:**
```bash
curl -X POST $LIGHTBULB_API_URL/v1/chat/completions \
  -H "Authorization: Bearer $USER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Option 3: Using demo-run.ps1 (Windows)

```powershell
# Starts everything automatically
.\demo-run.ps1

# Then use the displayed commands or CLI
```

## 🔐 Security

### Secrets Management

All secrets are stored in `.demo-secrets.env`:

```env
DATABASE_URL=postgresql://...
LIGHTBULB_ADMIN_KEY=lb-abc123...
LIGHTBULB_ADMIN_KEY_HASH=sha256...
LIGHTBULB_API_URL=http://localhost:8080
LIGHTBULB_USER_KEY=lb-xyz789...  # Added after first user key creation
```

**⚠️ IMPORTANT:**
- ✅ `.demo-secrets.env` is in `.gitignore`
- ✅ File permissions are restricted (600 on Unix)
- ❌ **NEVER commit this file to version control**
- ❌ **NEVER share admin keys publicly**

### Key Management

**View all keys:**
```bash
docker exec -it lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT id, role, created_at, expires_at FROM api_keys;"
```

**Delete a key:**
```bash
docker exec -it lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "DELETE FROM api_keys WHERE id = 'uuid-here';"
```

**Create expiring key:**
```bash
curl -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "expires_in_seconds": 3600,
    "description": "1-hour test key"
  }'
```

## 🛠️ Management Commands

### PostgreSQL Container

**Stop:**
```bash
docker stop lightbulb-postgres-demo
```

**Start:**
```bash
docker start lightbulb-postgres-demo
```

**Remove (deletes all data):**
```bash
docker stop lightbulb-postgres-demo
docker rm lightbulb-postgres-demo
```

**View logs:**
```bash
docker logs lightbulb-postgres-demo
docker logs -f lightbulb-postgres-demo  # Follow
```

**Connect to database:**
```bash
docker exec -it lightbulb-postgres-demo psql -U lightbulb -d lightbulb
```

### Environment Loading

**PowerShell:**
```powershell
# Load secrets
. ./.demo-secrets.env

# Verify
echo $env:DATABASE_URL
echo $env:LIGHTBULB_ADMIN_KEY
```

**Bash:**
```bash
# Load secrets
source .demo-secrets.env

# Verify
echo $DATABASE_URL
echo $LIGHTBULB_ADMIN_KEY
```

### Clean Up Everything

**Complete cleanup:**
```powershell
# Stop and remove container
docker stop lightbulb-postgres-demo
docker rm lightbulb-postgres-demo

# Remove secrets
rm .demo-secrets.env

# Or use clean start flag
.\demo-setup.ps1 -CleanStart
```

## 🐛 Troubleshooting

### PostgreSQL Won't Start

**Check if port is in use:**
```bash
# Windows
netstat -ano | findstr :5432

# Linux/Mac
lsof -i :5432
```

**Use different port:**
```bash
./demo-setup.sh --postgres-port 5433
```

### Database Connection Fails

**Test connection:**
```bash
docker exec lightbulb-postgres-demo pg_isready -U lightbulb
```

**Check logs:**
```bash
docker logs lightbulb-postgres-demo
```

### Migrations Fail

**Run manually:**
```bash
source .demo-secrets.env
cargo install sqlx-cli --no-default-features --features postgres
sqlx migrate run
```

**Check migration status:**
```bash
sqlx migrate info
```

### API Server Won't Start

**Check if port 8080 is in use:**
```bash
# Windows
netstat -ano | findstr :8080

# Linux/Mac
lsof -i :8080
```

**Run with verbose logging:**
```bash
RUST_LOG=debug cargo run --release
```

### API Key Not Working

**Verify key exists:**
```bash
source .demo-secrets.env
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT id, role, expires_at FROM api_keys;"
```

**Check key hash:**
```bash
# Compute hash of your key
echo -n "lb-your-key" | sha256sum

# Compare with database
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "SELECT key_hash FROM api_keys WHERE role='admin';"
```

### Build Fails

**Update Rust:**
```bash
rustup update stable
```

**Clean and rebuild:**
```bash
cargo clean
cargo build --release
```

**Check dependencies:**
```bash
cargo tree
```

## 📊 Monitoring

### Database Activity

```sql
-- Connect to database
docker exec -it lightbulb-postgres-demo psql -U lightbulb -d lightbulb

-- Check API key usage
SELECT * FROM api_key_usage ORDER BY window_start DESC LIMIT 10;

-- Check audit logs
SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10;

-- Key statistics
SELECT role, COUNT(*), 
       SUM(CASE WHEN expires_at IS NULL OR expires_at > NOW() THEN 1 ELSE 0 END) as active
FROM api_keys 
GROUP BY role;
```

### Server Logs

**PowerShell (with demo-run.ps1):**
```powershell
Receive-Job <job-id>
```

**Manual run:**
```bash
RUST_LOG=info cargo run --release
```

## 🔄 Reset Demo

**Full reset:**
```bash
# Stop everything
docker stop lightbulb-postgres-demo
docker rm lightbulb-postgres-demo
rm .demo-secrets.env

# Start fresh
./demo-setup.sh
```

**Keep container, reset data:**
```bash
docker exec lightbulb-postgres-demo psql -U lightbulb -d lightbulb \
  -c "TRUNCATE api_keys, audit_logs, sessions, api_key_usage CASCADE;"

# Re-run migrations
sqlx migrate run
```

## 📚 Additional Resources

- **CLI Documentation**: `CLI.md`
- **Quick Start Guide**: `QUICKSTART.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Main README**: `README.md`

## ⚡ Quick Reference

```bash
# Setup
./demo-setup.sh                           # First time setup

# Start server
source .demo-secrets.env
cargo run --release

# Create user key
curl -X POST $LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys \
  -H "Authorization: Bearer $LIGHTBULB_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"role":"user"}'

# Test CLI
cargo run --release --bin lightbulb-cli -- \
  --api-key $LIGHTBULB_USER_KEY

# Clean up
docker stop lightbulb-postgres-demo
docker rm lightbulb-postgres-demo
rm .demo-secrets.env
```
