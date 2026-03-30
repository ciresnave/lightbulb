# Lightbulb Complete Demo Runner (PowerShell)
# Runs the full demo including server startup and user key creation

param(
    [switch]$SkipBuild,
    [switch]$StopAfterSetup,
    [int]$WaitSeconds = 10
)

$ErrorActionPreference = 'Stop'

Write-Host 'Lightbulb Complete Demo Runner' -ForegroundColor Cyan
Write-Host '==============================' -ForegroundColor Cyan
Write-Host ''

# Check if setup has been run
if (-not (Test-Path '.demo-secrets.env')) {
    Write-Host '[WARN] Demo setup has not been run yet' -ForegroundColor Yellow
    Write-Host 'Running setup first...' -ForegroundColor Yellow
    Write-Host ''
    
    & .\demo-setup.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[ERROR] Setup failed' -ForegroundColor Red
        exit 1
    }
    Write-Host ''
}

# Load secrets
Write-Host '[*] Loading secrets...' -ForegroundColor Yellow
Get-Content '.demo-secrets.env' | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}
Write-Host '[OK] Secrets loaded' -ForegroundColor Green
Write-Host ''

if ($StopAfterSetup) {
    Write-Host '[STOP] Stopping after setup as requested' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'To start the server manually:' -ForegroundColor Cyan
    Write-Host '  . ./.demo-secrets.env' -ForegroundColor Gray
    Write-Host '  cargo run --release' -ForegroundColor Gray
    exit 0
}

# Start the API server in a background job
Write-Host '[*] Starting Lightbulb API server...' -ForegroundColor Yellow

$serverJob = Start-Job -ScriptBlock {
    param($WorkDir, $DatabaseUrl)
    Set-Location $WorkDir
    $env:DATABASE_URL = $DatabaseUrl
    $env:RUST_LOG = 'info'
    cargo run --release --bin lightbulb 2>&1
} -ArgumentList (Get-Location), $env:DATABASE_URL

Write-Host "[OK] Server started (Job ID: $($serverJob.Id))" -ForegroundColor Green
Write-Host "[*] Waiting ${WaitSeconds} seconds for server to initialize..." -ForegroundColor Yellow

# Monitor server startup
$startTime = Get-Date
$timeout = 60
$serverReady = $false

while (((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
    # Check if job is still running
    if ($serverJob.State -ne 'Running') {
        Write-Host '[ERROR] Server job stopped unexpectedly' -ForegroundColor Red
        Receive-Job $serverJob
        Remove-Job $serverJob
        exit 1
    }
    
    # Try to connect to server
    try {
        $response = Invoke-WebRequest -Uri "$env:LIGHTBULB_API_URL/v1/models" `
            -Method GET `
            -TimeoutSec 2 `
            -ErrorAction SilentlyContinue 2>$null
        
        if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 401) {
            $serverReady = $true
            break
        }
    }
    catch {
        # Server not ready yet
    }
    
    Start-Sleep -Seconds 1
}

if (-not $serverReady) {
    Write-Host "[ERROR] Server failed to become ready within $timeout seconds" -ForegroundColor Red
    Write-Host 'Server output:' -ForegroundColor Yellow
    Receive-Job $serverJob
    Stop-Job $serverJob
    Remove-Job $serverJob
    exit 1
}

Write-Host '[OK] Server is ready!' -ForegroundColor Green
Write-Host ''

# Create a user API key
Write-Host '[*] Creating user API key...' -ForegroundColor Yellow

$headers = @{
    'Authorization' = "Bearer $env:LIGHTBULB_ADMIN_KEY"
    'Content-Type'  = 'application/json'
}

$body = @{
    role        = 'user'
    description = 'Demo user key'
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod `
        -Uri "$env:LIGHTBULB_API_URL/v1/lightbulb/admin/api-keys" `
        -Method POST `
        -Headers $headers `
        -Body $body `
        -TimeoutSec 10

    $userKey = $response.api_key
    $env:LIGHTBULB_USER_KEY = $userKey
    
    Write-Host "[OK] User key created: $userKey" -ForegroundColor Green
    
    # Save to secrets file
    Add-Content -Path '.demo-secrets.env' -Value "LIGHTBULB_USER_KEY=$userKey"
    
}
catch {
    Write-Host "[ERROR] Failed to create user key: $_" -ForegroundColor Red
    Stop-Job $serverJob
    Remove-Job $serverJob
    exit 1
}

Write-Host ''

# Display demo information
Write-Host '[SUCCESS] Demo is ready!' -ForegroundColor Green
Write-Host ''
Write-Host 'Demo Status:' -ForegroundColor Cyan
Write-Host '============' -ForegroundColor Cyan
Write-Host "API Server:     Running (Job ID: $($serverJob.Id))" -ForegroundColor White
Write-Host "Server URL:     $env:LIGHTBULB_API_URL" -ForegroundColor White
Write-Host "Admin Key:      $env:LIGHTBULB_ADMIN_KEY" -ForegroundColor White
Write-Host "User Key:       $env:LIGHTBULB_USER_KEY" -ForegroundColor White
Write-Host ''
Write-Host 'Try these commands:' -ForegroundColor Cyan
Write-Host '===================' -ForegroundColor Cyan
Write-Host ''
Write-Host '1. Test with curl (non-streaming):' -ForegroundColor Yellow
Write-Host @"
   curl -X POST $env:LIGHTBULB_API_URL/v1/chat/completions ``
     -H "Authorization: Bearer $env:LIGHTBULB_USER_KEY" ``
     -H "Content-Type: application/json" ``
     -d '{
       "model": "default",
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": false
     }'
"@ -ForegroundColor Gray
Write-Host ''
Write-Host '2. Test with CLI (interactive):' -ForegroundColor Yellow
Write-Host "   cargo run --release --bin lightbulb-cli -- --api-key $env:LIGHTBULB_USER_KEY" -ForegroundColor Gray
Write-Host ''
Write-Host '3. Test with CLI (streaming):' -ForegroundColor Yellow
Write-Host "   cargo run --release --bin lightbulb-cli -- --api-key $env:LIGHTBULB_USER_KEY --stream" -ForegroundColor Gray
Write-Host ''
Write-Host '4. Check server logs:' -ForegroundColor Yellow
Write-Host "   Receive-Job $($serverJob.Id)" -ForegroundColor Gray
Write-Host ''
Write-Host '5. Stop the demo:' -ForegroundColor Yellow
Write-Host "   Stop-Job $($serverJob.Id); Remove-Job $($serverJob.Id)" -ForegroundColor Gray
Write-Host '   docker stop lightbulb-postgres-demo' -ForegroundColor Gray
Write-Host ''
Write-Host 'Notes:' -ForegroundColor Cyan
Write-Host '======' -ForegroundColor Cyan
Write-Host '* Server is running in background (PowerShell job)' -ForegroundColor White
Write-Host '* All secrets are saved in .demo-secrets.env' -ForegroundColor White
Write-Host '* Press Ctrl+C to stop this script (server will keep running)' -ForegroundColor White
Write-Host '* Or run the stop command above to cleanly shut down' -ForegroundColor White
Write-Host ''

# Keep script running and show live logs
Write-Host 'Live server logs (Ctrl+C to exit):' -ForegroundColor Cyan
Write-Host '===================================' -ForegroundColor Cyan
Write-Host ''

try {
    while ($true) {
        $output = Receive-Job $serverJob
        if ($output) {
            Write-Host $output
        }
        
        # Check if job is still running
        if ($serverJob.State -ne 'Running') {
            Write-Host ''
            Write-Host '[WARN] Server job stopped' -ForegroundColor Yellow
            break
        }
        
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host ''
    Write-Host '[*] Stopping demo...' -ForegroundColor Yellow
    
    if ($serverJob.State -eq 'Running') {
        Stop-Job $serverJob
    }
    Remove-Job $serverJob
    
    Write-Host '[OK] Demo stopped' -ForegroundColor Green
    Write-Host ''
    Write-Host 'To restart: ./demo-run.ps1' -ForegroundColor Cyan
}
