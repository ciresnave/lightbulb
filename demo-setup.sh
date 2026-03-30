#!/bin/bash
# Lightbulb Demo Setup Script (Bash)
# Automatically sets up PostgreSQL, generates secrets, and prepares for testing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="lightbulb-postgres-demo"
DB_NAME="lightbulb"
DB_USER="lightbulb"
DB_PASSWORD=$(openssl rand -base64 20 | tr -d "=+/" | cut -c1-20)
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
API_PORT="${API_PORT:-8080}"
DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${POSTGRES_PORT}/${DB_NAME}"
SECRETS_FILE=".demo-secrets.env"

# Parse arguments
CLEAN_START=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean-start)
            CLEAN_START=true
            shift
            ;;
        --postgres-port)
            POSTGRES_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}🔦 Lightbulb Demo Setup${NC}"
echo -e "${CYAN}======================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    echo -e "${RED}Please install Docker: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Rust/Cargo is not installed or not in PATH${NC}"
    echo -e "${RED}Please install Rust: https://rustup.rs/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker found: $(docker --version)${NC}"
echo -e "${GREEN}✅ Cargo found: $(cargo --version)${NC}"
echo ""

# Clean up if requested
if [ "$CLEAN_START" = true ]; then
    echo -e "${YELLOW}🧹 Cleaning up existing demo setup...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    rm -f $SECRETS_FILE
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    echo ""
fi

# Check if PostgreSQL container already exists
if docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}📦 Found existing PostgreSQL container: $CONTAINER_NAME${NC}"
    
    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}✅ Container is already running${NC}"
    else
        echo -e "${YELLOW}▶️  Starting existing container...${NC}"
        docker start $CONTAINER_NAME > /dev/null
        sleep 3
        echo -e "${GREEN}✅ Container started${NC}"
    fi
    
    # Load existing secrets
    if [ -f "$SECRETS_FILE" ]; then
        echo -e "${YELLOW}📄 Loading existing secrets from $SECRETS_FILE${NC}"
        source "$SECRETS_FILE"
        DATABASE_URL="$DATABASE_URL"
    fi
else
    # Start PostgreSQL in Docker
    echo -e "${YELLOW}🐳 Starting PostgreSQL in Docker...${NC}"

    docker run -d \
        --name $CONTAINER_NAME \
        -e POSTGRES_DB=$DB_NAME \
        -e POSTGRES_USER=$DB_USER \
        -e POSTGRES_PASSWORD=$DB_PASSWORD \
        -p "${POSTGRES_PORT}:5432" \
        postgres:15-alpine

    echo -e "${GREEN}✅ PostgreSQL container started${NC}"
    echo -e "${YELLOW}⏳ Waiting for PostgreSQL to be ready...${NC}"

    for i in {1..30}; do
        if docker exec $CONTAINER_NAME pg_isready -U $DB_USER >/dev/null 2>&1; then
            echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ PostgreSQL failed to become ready after 30 seconds${NC}"
            docker logs $CONTAINER_NAME
            exit 1
        fi
        sleep 1
    done
    echo ""
fi

# Generate secrets
echo -e "${YELLOW}🔐 Generating API keys...${NC}"

# Generate bootstrap admin key
ADMIN_KEY="lb-$(openssl rand -hex 32)"

# Compute SHA-256 hash
ADMIN_KEY_HASH=$(echo -n "$ADMIN_KEY" | sha256sum | cut -d' ' -f1)

echo -e "${GREEN}✅ Generated bootstrap admin key${NC}"

# Save secrets
echo -e "${YELLOW}💾 Saving secrets to $SECRETS_FILE...${NC}"

cat > "$SECRETS_FILE" << EOF
# Lightbulb Demo Secrets - DO NOT COMMIT
# Generated: $(date "+%Y-%m-%d %H:%M:%S")

export DATABASE_URL="$DATABASE_URL"
export LIGHTBULB_ADMIN_KEY="$ADMIN_KEY"
export LIGHTBULB_ADMIN_KEY_HASH="$ADMIN_KEY_HASH"
export LIGHTBULB_API_URL="http://localhost:$API_PORT"
EOF

chmod 600 "$SECRETS_FILE"

echo -e "${GREEN}✅ Secrets saved to $SECRETS_FILE${NC}"
echo ""

# Set environment variables for this session
export DATABASE_URL="$DATABASE_URL"
export LIGHTBULB_ADMIN_KEY="$ADMIN_KEY"
export LIGHTBULB_API_URL="http://localhost:$API_PORT"

# Run database migrations
echo -e "${YELLOW}📊 Running database migrations...${NC}"

if command -v sqlx &> /dev/null; then
    if sqlx migrate run 2>/dev/null; then
        echo -e "${GREEN}✅ Migrations completed${NC}"
    else
        echo -e "${YELLOW}⚠️  Migration with sqlx failed, will run migrations on server startup${NC}"
    fi
else
    echo -e "${CYAN}ℹ️  sqlx-cli not found, migrations will run on server startup${NC}"
fi

echo ""

# Insert bootstrap admin key into database
echo -e "${YELLOW}🔑 Inserting bootstrap admin key into database...${NC}"

SQL_COMMAND="INSERT INTO api_keys (key_hash, role, created_at) VALUES ('$ADMIN_KEY_HASH', 'admin', NOW()) ON CONFLICT DO NOTHING;"

if docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c "$SQL_COMMAND" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Bootstrap admin key inserted${NC}"
else
    echo -e "${YELLOW}⚠️  Failed to insert admin key (may already exist)${NC}"
fi

echo ""

# Build the project
echo -e "${YELLOW}🔨 Building Lightbulb...${NC}"
echo -e "${GRAY}   (This may take a few minutes on first run)${NC}"

if cargo build --release >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Build complete${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    echo -e "${RED}Run 'cargo build --release' manually to see errors${NC}"
    exit 1
fi

echo ""

# Build CLI
echo -e "${YELLOW}🔨 Building Lightbulb CLI...${NC}"

if cargo build --release --bin lightbulb-cli >/dev/null 2>&1; then
    echo -e "${GREEN}✅ CLI build complete${NC}"
else
    echo -e "${RED}❌ CLI build failed${NC}"
    exit 1
fi

echo ""

# Summary
echo -e "${GREEN}🎉 Demo setup complete!${NC}"
echo ""
echo -e "${CYAN}📋 Setup Summary:${NC}"
echo -e "${CYAN}=================${NC}"
echo -e "${WHITE}PostgreSQL:     Running in Docker container '$CONTAINER_NAME'${NC}"
echo -e "${WHITE}Database:       $DB_NAME${NC}"
echo -e "${WHITE}Port:           $POSTGRES_PORT${NC}"
echo -e "${WHITE}Admin Key:      Saved in $SECRETS_FILE${NC}"
echo ""
echo -e "${CYAN}🚀 Next Steps:${NC}"
echo -e "${CYAN}==============${NC}"
echo ""
echo -e "${YELLOW}1. Start the API server (in a new terminal):${NC}"
echo -e "${GRAY}   cd $(pwd)${NC}"
echo -e "${GRAY}   source ./$SECRETS_FILE${NC}"
echo -e "${GRAY}   cargo run --release${NC}"
echo ""
echo -e "${YELLOW}2. Create a user API key (in another terminal):${NC}"
echo -e "${GRAY}   source ./$SECRETS_FILE${NC}"
echo -e "${GRAY}   USER_KEY=\$(curl -s -X POST \\${NC}"
echo -e "${GRAY}     \"\$LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys\" \\${NC}"
echo -e "${GRAY}     -H \"Authorization: Bearer \$LIGHTBULB_ADMIN_KEY\" \\${NC}"
echo -e "${GRAY}     -H \"Content-Type: application/json\" \\${NC}"
echo -e "${GRAY}     -d '{\"role\":\"user\"}' | jq -r '.api_key')${NC}"
echo -e "${GRAY}   export LIGHTBULB_USER_KEY=\$USER_KEY${NC}"
echo -e "${GRAY}   echo \"User key: \$LIGHTBULB_USER_KEY\"${NC}"
echo ""
echo -e "${YELLOW}3. Test with CLI:${NC}"
echo -e "${GRAY}   cargo run --release --bin lightbulb-cli -- --api-key \$LIGHTBULB_USER_KEY${NC}"
echo ""
echo -e "${YELLOW}4. Or test streaming:${NC}"
echo -e "${GRAY}   cargo run --release --bin lightbulb-cli -- --api-key \$LIGHTBULB_USER_KEY --stream${NC}"
echo ""
echo -e "${CYAN}📝 Quick Commands:${NC}"
echo -e "${CYAN}==================${NC}"
echo -e "${WHITE}Load secrets:       source ./$SECRETS_FILE${NC}"
echo -e "${WHITE}Stop PostgreSQL:    docker stop $CONTAINER_NAME${NC}"
echo -e "${WHITE}Start PostgreSQL:   docker start $CONTAINER_NAME${NC}"
echo -e "${WHITE}View logs:          docker logs $CONTAINER_NAME${NC}"
echo -e "${WHITE}Connect to DB:      docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME${NC}"
echo -e "${WHITE}Clean up:           ./demo-setup.sh --clean-start${NC}"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: The admin key is saved in $SECRETS_FILE${NC}"
echo -e "${YELLOW}   Keep this file secure and do NOT commit it to version control!${NC}"
echo ""
