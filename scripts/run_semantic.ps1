#Requires -Version 5.1
<#
.SYNOPSIS
  Start the CUCUMBER backend in semantic mode (submission path).
.DESCRIPTION
  Sets CUCUMBER_QA_MODE=semantic for this process, warns when the
  frontend bundle is missing, starts uvicorn, then polls /api/health
  until qa_mode=semantic (or the timeout expires).
#>
param(
  [string]$ListenAddress = "127.0.0.1",
  [int]$Port = 8000,
  [int]$HealthTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$env:CUCUMBER_QA_MODE = "semantic"

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  throw "Python venv not found at $python. Create it per README section 3.1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "frontend\dist\index.html"))) {
  Write-Warning "frontend/dist/index.html not found. Run 'npm install' and 'npm run build' in frontend/ first (README 3.3)."
}

$healthUrl = "http://${ListenAddress}:${Port}/api/health"
Write-Host "Starting semantic backend: $healthUrl (CUCUMBER_QA_MODE=$env:CUCUMBER_QA_MODE)"

$server = Start-Process -FilePath $python -ArgumentList @(
  "-m", "uvicorn", "backend.main:app",
  "--host", $ListenAddress,
  "--port", "$Port"
) -PassThru

try {
  $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    if ($server.HasExited) {
      throw "Backend process exited early with code $($server.ExitCode)."
    }
    try {
      $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
      if ($health.qa_mode -eq "semantic" -and $health.status -eq "ok" -and $health.database_ready -eq $true) {
        Write-Host "Health OK: status=ok database_ready=True qa_mode=semantic"
        Write-Host "Open: http://${ListenAddress}:${Port}/chat"
        return
      }
      Write-Host "Waiting for semantic health (qa_mode=$($health.qa_mode) status=$($health.status))..."
    } catch {
      Write-Host "Waiting for backend to listen..."
    }
    Start-Sleep -Seconds 3
  }
  throw "Timed out waiting for semantic health at $healthUrl."
} catch {
  if (-not $server.HasExited) {
    Stop-Process -Id $server.Id -Force
  }
  throw
}
