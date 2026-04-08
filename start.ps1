# Lemonade Eval Dashboard - Quick Start Script (PowerShell)
# Starts both backend and frontend servers for development

param(
    [switch]$NoOpen,
    [switch]$Verbose
)

# Configuration
$ScriptDir = $PSScriptRoot
$DashboardDir = Join-Path $ScriptDir "dashboard"
$BackendDir = Join-Path $DashboardDir "backend"
$FrontendDir = Join-Path $DashboardDir "frontend"

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error-Exit {
    Write-Host "[ERROR] $args" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║       LEMONADE EVAL DASHBOARD - QUICK START            ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Check if backend is set up
if (-not (Test-Path "$BackendDir\venv")) {
    Write-Error-Exit "Backend not set up. Run .\install.ps1 first"
}

# Check if frontend is set up
if (-not (Test-Path "$FrontendDir\node_modules")) {
    Write-Error-Exit "Frontend not set up. Run .\install.ps1 first"
}

# Start backend
Write-Info "Starting backend server..."
Set-Location $BackendDir
& ".\venv\Scripts\Activate.ps1"

# Check if port 8000 is already in use
$backendPort = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($backendPort) {
    Write-Warn "Port 8000 is already in use"
}
else {
    # Start backend in new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$BackendDir'
& '.\venv\Scripts\Activate.ps1'
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"@
    Write-Success "Backend server started in new window"
}

# Start frontend
Write-Info "Starting frontend server..."
Set-Location $FrontendDir

# Check if port 3000 is already in use
$frontendPort = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($frontendPort) {
    Write-Warn "Port 3000 is already in use"
}
else {
    # Start frontend in new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$FrontendDir'
npm run dev
"@
    Write-Success "Frontend server started in new window"
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Servers starting in separate windows..." -ForegroundColor Green
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Wait for servers to start
Start-Sleep -Seconds 5

# Check backend health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Success "Backend is healthy"
    }
}
catch {
    Write-Warn "Backend may still be starting..."
}

Write-Host ""
Write-Host "To stop servers, close the PowerShell windows or run:" -ForegroundColor Yellow
Write-Host "  Get-Process | Where-Object {$_.MainWindowTitle -like '*uvicorn*'} | Stop-Process" -ForegroundColor Yellow
Write-Host "  Get-Process | Where-Object {$_.MainWindowTitle -like '*node*'} | Stop-Process" -ForegroundColor Yellow
Write-Host ""

# Open browser if not disabled
if (-not $NoOpen) {
    Start-Process "http://localhost:3000"
    Write-Info "Opened browser to http://localhost:3000"
}
