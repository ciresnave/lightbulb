# Lightbulb Demo Setup Script (PowerShell)
# Automatically sets up PostgreSQL, generates secrets, and prepares for testing

param(
    [switch]$SkipDockerCheck,
    [switch]$CleanStart,
    [string]$PostgresPort = '5432',
    [string]$ApiPort = '8080'
)

$ErrorActionPreference = 'Stop'

Write-Host 'Lightbulb Demo Setup' -ForegroundColor Cyan
Write-Host '====================' -ForegroundColor Cyan
Write-Host ''

# Function to check if command exists
function Test-Command {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check prerequisites
Write-Host '[*] Checking prerequisites...' -ForegroundColor Yellow

if (-not (Test-Command 'docker')) {
    Write-Host '[ERROR] Docker is not installed or not in PATH' -ForegroundColor Red
    Write-Host 'Please install Docker Desktop: https://www.docker.com/products/docker-desktop' -ForegroundColor Red
    exit 1
}

if (-not (Test-Command 'cargo')) {
    Write-Host '[ERROR] Rust/Cargo is not installed or not in PATH' -ForegroundColor Red
    Write-Host 'Please install Rust: https://rustup.rs/' -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Docker found: $(docker --version)" -ForegroundColor Green
Write-Host "[OK] Cargo found: $(cargo --version)" -ForegroundColor Green
Write-Host ''

# Configuration
$CONTAINER_NAME = 'lightbulb-postgres-demo'
$DB_NAME = 'lightbulb'
$DB_USER = 'lightbulb'
$DB_PASSWORD = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 20 | ForEach-Object { [char]$_ })
$DATABASE_URL = "postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${PostgresPort}/${DB_NAME}"
$SECRETS_FILE = '.demo-secrets.env'

# Clean up existing container if requested
if ($CleanStart) {
    Write-Host '[*] Cleaning up existing demo setup...' -ForegroundColor Yellow
    docker stop $CONTAINER_NAME 2>$null
    docker rm $CONTAINER_NAME 2>$null
    if (Test-Path $SECRETS_FILE) {
        Remove-Item $SECRETS_FILE
    }
    Write-Host '[OK] Cleanup complete' -ForegroundColor Green
    Write-Host ''
}

# Check if PostgreSQL container already exists
$existingContainer = docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' 2>$null

if ($existingContainer) {
    Write-Host "[*] Found existing PostgreSQL container: $CONTAINER_NAME" -ForegroundColor Yellow
    
    $containerState = docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>$null
    
    if ($containerState -eq 'true') {
        Write-Host '[OK] Container is already running' -ForegroundColor Green
    }
    else {
        Write-Host '[*] Starting existing container...' -ForegroundColor Yellow
        docker start $CONTAINER_NAME | Out-Null
        Start-Sleep -Seconds 3
        Write-Host '[OK] Container started' -ForegroundColor Green
    }
    
    # Load existing secrets
    if (Test-Path $SECRETS_FILE) {
        Write-Host "[*] Loading existing secrets from $SECRETS_FILE" -ForegroundColor Yellow
        Get-Content $SECRETS_FILE | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                Set-Item -Path "env:$($matches[1])" -Value $matches[2]
            }
        }
        $DATABASE_URL = $env:DATABASE_URL
    }
}
else {
    # Start PostgreSQL in Docker
    Write-Host '[*] Starting PostgreSQL in Docker...' -ForegroundColor Yellow

    docker run -d `
        --name $CONTAINER_NAME `
        -e POSTGRES_DB=$DB_NAME `
        -e POSTGRES_USER=$DB_USER `
        -e POSTGRES_PASSWORD=$DB_PASSWORD `
        -p "${PostgresPort}:5432" `
        postgres:15-alpine

    if ($LASTEXITCODE -ne 0) {
        Write-Host '[ERROR] Failed to start PostgreSQL container' -ForegroundColor Red
        exit 1
    }

    Write-Host '[OK] PostgreSQL container started' -ForegroundColor Green
    Write-Host '[*] Waiting for PostgreSQL to be ready...' -ForegroundColor Yellow

    $maxRetries = 30
    $retries = 0
    $ready = $false

    while (-not $ready -and $retries -lt $maxRetries) {
        $retries++
        try {
            docker exec $CONTAINER_NAME pg_isready -U $DB_USER 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $ready = $true
            }
        }
        catch {
            # Ignore errors during connection attempts
        }
        
        if (-not $ready) {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ready) {
        Write-Host '[ERROR] PostgreSQL failed to become ready after 30 seconds' -ForegroundColor Red
        docker logs $CONTAINER_NAME
        exit 1
    }

    Write-Host '[OK] PostgreSQL is ready' -ForegroundColor Green
    Write-Host ''
}

# Generate secrets
Write-Host '[*] Generating API keys...' -ForegroundColor Yellow

# Generate bootstrap admin key
$adminKeyBytes = New-Object byte[] 32
[Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($adminKeyBytes)
$adminKey = 'lb-' + ($adminKeyBytes | ForEach-Object { $_.ToString('x2') }) -join ''

# Compute SHA-256 hash
$sha256 = [Security.Cryptography.SHA256]::Create()
$adminKeyHash = ($sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($adminKey)) | ForEach-Object { $_.ToString('x2') }) -join ''

Write-Host '[OK] Generated bootstrap admin key' -ForegroundColor Green

# Save secrets
Write-Host "[*] Saving secrets to $SECRETS_FILE..." -ForegroundColor Yellow

@"
# Lightbulb Demo Secrets - DO NOT COMMIT
# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

DATABASE_URL=$DATABASE_URL
LIGHTBULB_ADMIN_KEY=$adminKey
LIGHTBULB_ADMIN_KEY_HASH=$adminKeyHash
LIGHTBULB_API_URL=http://localhost:$ApiPort
"@ | Out-File -FilePath $SECRETS_FILE -Encoding UTF8

Write-Host "[OK] Secrets saved to $SECRETS_FILE" -ForegroundColor Green
Write-Host ''

# Set environment variables for this session
$env:DATABASE_URL = $DATABASE_URL
$env:LIGHTBULB_ADMIN_KEY = $adminKey
$env:LIGHTBULB_API_URL = "http://localhost:$ApiPort"

# Run database migrations
Write-Host '[*] Running database migrations...' -ForegroundColor Yellow

$env:DATABASE_URL = $DATABASE_URL

# Check if sqlx-cli is available
if (Test-Command 'sqlx') {
    sqlx migrate run
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[WARN] Migration with sqlx failed, will try manual execution' -ForegroundColor Yellow
        $sqlxFailed = $true
    }
    else {
        Write-Host '[OK] Migrations completed with sqlx' -ForegroundColor Green
        $sqlxFailed = $false
    }
}
else {
    Write-Host '[INFO] sqlx-cli not found, running migrations manually' -ForegroundColor Cyan
    $sqlxFailed = $true
}

# If sqlx is not available or failed, run migrations manually
if ($sqlxFailed) {
    $migrationFiles = Get-ChildItem -Path 'migrations' -Filter '*.sql' | Sort-Object Name
    
    foreach ($file in $migrationFiles) {
        Write-Host "[*] Running migration: $($file.Name)" -ForegroundColor Yellow
        
        $migrationSql = Get-Content $file.FullName -Raw
        
        # Run migration, suppress NOTICE messages
        $ErrorActionPreference = 'Continue'
        $migrationOutput = docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c $migrationSql 2>&1 | Where-Object { $_ -notmatch '^NOTICE:' }
        $migrationExitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        
        # Check if migration succeeded
        if ($migrationExitCode -eq 0) {
            Write-Host "[OK] Migration $($file.Name) completed" -ForegroundColor Green
        }
        else {
            # Check if it failed due to something other than "already exists"
            $outputStr = $migrationOutput | Out-String
            if ($outputStr -match 'already exists') {
                Write-Host "[OK] Migration $($file.Name) already applied" -ForegroundColor Green
            }
            else {
                Write-Host "[ERROR] Migration $($file.Name) failed" -ForegroundColor Red
                Write-Host $outputStr -ForegroundColor Red
                exit 1
            }
        }
    }
    
    Write-Host '[OK] All migrations completed' -ForegroundColor Green
}

Write-Host ''

# Insert bootstrap admin key into database
Write-Host '[*] Inserting bootstrap admin key into database...' -ForegroundColor Yellow

$sqlCommand = @"
INSERT INTO api_keys (key_hash, role, created_at) 
VALUES ('$adminKeyHash', 'admin', NOW())
ON CONFLICT DO NOTHING;
"@

docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c $sqlCommand 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host '[OK] Bootstrap admin key inserted' -ForegroundColor Green
}
else {
    Write-Host '[WARN] Failed to insert admin key (may already exist)' -ForegroundColor Yellow
}

Write-Host ''

# Build the project
Write-Host '[*] Building Lightbulb...' -ForegroundColor Yellow
Write-Host '    (This may take a few minutes on first run)' -ForegroundColor Gray

$ErrorActionPreference = 'Continue'
$buildOutput = cargo build --release 2>&1 | Where-Object { $_ -notmatch '^warning:' }
$buildExitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'

if ($buildExitCode -ne 0) {
    Write-Host '[ERROR] Build failed' -ForegroundColor Red
    Write-Host 'Build output:' -ForegroundColor Yellow
    $buildOutput | ForEach-Object { Write-Host $_ }
    exit 1
}

Write-Host '[OK] Build complete' -ForegroundColor Green
Write-Host ''

# Build CLI
Write-Host '[*] Building Lightbulb CLI...' -ForegroundColor Yellow

$ErrorActionPreference = 'Continue'
$cliBuildOutput = cargo build --release --bin lightbulb-cli 2>&1 | Where-Object { $_ -notmatch '^warning:' }
$cliBuildExitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'

if ($cliBuildExitCode -ne 0) {
    Write-Host '[ERROR] CLI build failed' -ForegroundColor Red
    Write-Host 'Build output:' -ForegroundColor Yellow
    $cliBuildOutput | ForEach-Object { Write-Host $_ }
    exit 1
}

Write-Host '[OK] CLI build complete' -ForegroundColor Green
Write-Host ''

# Summary
Write-Host '[SUCCESS] Demo setup complete!' -ForegroundColor Green
Write-Host ''
Write-Host 'Setup Summary:' -ForegroundColor Cyan
Write-Host '==============' -ForegroundColor Cyan
Write-Host "PostgreSQL:     Running in Docker container '$CONTAINER_NAME'" -ForegroundColor White
Write-Host "Database:       $DB_NAME" -ForegroundColor White
Write-Host "Port:           $PostgresPort" -ForegroundColor White
Write-Host "Admin Key:      Saved in $SECRETS_FILE" -ForegroundColor White
Write-Host ''
Write-Host 'Next Steps:' -ForegroundColor Cyan
Write-Host '===========' -ForegroundColor Cyan
Write-Host ''
Write-Host '1. Start the API server (in a new terminal):' -ForegroundColor Yellow
Write-Host "   cd $(Get-Location)" -ForegroundColor Gray
Write-Host "   . ./$SECRETS_FILE" -ForegroundColor Gray
Write-Host '   cargo run --release' -ForegroundColor Gray
Write-Host ''
Write-Host '2. Create a user API key (in another terminal):' -ForegroundColor Yellow
Write-Host "   . ./$SECRETS_FILE" -ForegroundColor Gray
Write-Host '   $response = Invoke-RestMethod -Method POST `' -ForegroundColor Gray
Write-Host '     -Uri "$env:LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys" `' -ForegroundColor Gray
Write-Host '     -Headers @{"Authorization"="Bearer $env:LIGHTBULB_ADMIN_KEY"} `' -ForegroundColor Gray
Write-Host '     -ContentType "application/json" `' -ForegroundColor Gray
Write-Host '     -Body ''{"role":"user"}''' -ForegroundColor Gray
Write-Host '   $env:LIGHTBULB_USER_KEY = $response.api_key' -ForegroundColor Gray
Write-Host '   Write-Host "User key: $env:LIGHTBULB_USER_KEY"' -ForegroundColor Gray
Write-Host ''
Write-Host '3. Test with CLI:' -ForegroundColor Yellow
Write-Host '   cargo run --release --bin lightbulb-cli -- --api-key $env:LIGHTBULB_USER_KEY' -ForegroundColor Gray
Write-Host ''
Write-Host '4. Or test streaming:' -ForegroundColor Yellow
Write-Host '   cargo run --release --bin lightbulb-cli -- --api-key $env:LIGHTBULB_USER_KEY --stream' -ForegroundColor Gray
Write-Host ''
Write-Host 'Quick Commands:' -ForegroundColor Cyan
Write-Host '===============' -ForegroundColor Cyan
Write-Host "Load secrets:       . ./$SECRETS_FILE" -ForegroundColor White
Write-Host "Stop PostgreSQL:    docker stop $CONTAINER_NAME" -ForegroundColor White
Write-Host "Start PostgreSQL:   docker start $CONTAINER_NAME" -ForegroundColor White
Write-Host "View logs:          docker logs $CONTAINER_NAME" -ForegroundColor White
Write-Host "Connect to DB:      docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME" -ForegroundColor White
Write-Host 'Clean up:           ./demo-setup.ps1 -CleanStart' -ForegroundColor White
Write-Host ''
Write-Host "[IMPORTANT] The admin key is saved in $SECRETS_FILE" -ForegroundColor Yellow
Write-Host '            Keep this file secure and do NOT commit it to version control!' -ForegroundColor Yellow
Write-Host ''
