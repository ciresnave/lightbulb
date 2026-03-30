<#
activate_vsdev_env.ps1

Import Visual Studio developer environment variables into the current PowerShell session.

Usage (interactive PowerShell session):
  .\scripts\activate_vsdev_env.ps1

Usage (non-interactive, in a single command):
  pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\activate_vsdev_env.ps1

The script attempts to find a Visual Studio installation (via vswhere if present, or common
installation paths), runs the appropriate vcvars/vsdevcmd batch file for x64, captures the
resulting environment, and imports it into the current process.
#>

param(
    [switch]$VerboseOutput
)

$ErrorActionPreference = 'Stop'

function Find-VsInstall {
    # Try vswhere if available
    $vswhereCandidates = @(
        "$env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    ) | Where-Object { Test-Path $_ }

    if ($vswhereCandidates) {
        $vswhere = $vswhereCandidates[0]
        try {
            $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
            if ($inst) { return $inst.Trim() }
        } catch {
            # ignore and fallback
        }
    }

    # Common candidate install roots
    $candidates = @(
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools',
        'C:\Program Files\Microsoft Visual Studio\2022\Community',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise'
    )

    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }

    return $null
}

$inst = Find-VsInstall
if (-not $inst) {
    Write-Error "Visual Studio installation not found. Please open Developer PowerShell for Visual Studio or install Build Tools and try again."
    exit 1
}

$vcvars = Join-Path $inst 'VC\Auxiliary\Build\vcvarsall.bat'
$vsdevcmd = Join-Path $inst 'Common7\Tools\VsDevCmd.bat'

if (Test-Path $vcvars) {
    # prefer vcvarsall for explicit arch selection
    $setupCmd = "`"$vcvars`" x64"
} elseif (Test-Path $vsdevcmd) {
    # fallback to VsDevCmd
    $setupCmd = "`"$vsdevcmd`" -arch=amd64"
} else {
    Write-Error "Neither vcvarsall.bat nor VsDevCmd.bat were found under $inst"
    exit 1
}

Write-Output "Running: $setupCmd"

# Run the batch setup under cmd.exe and capture the environment output
# We run: cmd.exe /c "<setupCmd> && set"
$batCmd = "$setupCmd && set"
try {
    $raw = & cmd.exe /c $batCmd 2>$null
} catch {
    Write-Error "Failed to run Visual Studio setup command: $_"
    exit 1
}

if (-not $raw) {
    Write-Error "Visual Studio setup produced no environment output"
    exit 1
}

# Parse lines of the form NAME=VALUE and set in current PowerShell env
foreach ($line in $raw) {
    if ($line -match '^(.*?)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item -Path Env:\$name -Value $value -Force
    }
}

Write-Output "Imported Visual Studio environment into current session."
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($cl) { Write-Output "cl.exe found at: $($cl.Source)" } else { Write-Output "cl.exe not found after importing environment" }

if ($VerboseOutput) {
    Write-Output "PATH head (first 6 entries):"
    $env:PATH -split ';' | Select-Object -First 6 | ForEach-Object { Write-Output " - $_" }
}

Write-Output "Done. You can now run 'cargo build' in this terminal."