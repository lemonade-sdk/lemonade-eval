# Lemonade Eval Dashboard - Production Installation Script (Windows PowerShell)
# This script automates the complete setup and installation process

param(
    [switch]$SkipDatabase,
    [switch]$SkipFrontend,
    [switch]$Verbose
)

# Configuration
$ScriptDir = $PSScriptRoot
$DashboardDir = Join-Path $ScriptDir "dashboard"
$BackendDir = Join-Path $DashboardDir "backend"
$FrontendDir = Join-Path $DashboardDir "frontend"
$VenvDir = Join-Path $BackendDir "venv"
$LogFile = Join-Path $ScriptDir "install.log"

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error-Exit {
    Write-Host "[ERROR] $args" -ForegroundColor Red
    exit 1
}

# Logging
function Log {
    param($Message)
    Write-Info $Message
    Add-Content -Path $LogFile -Value "$(Get-Date): $Message"
}

function Log-Success {
    param($Message)
    Write-Success $Message
    Add-Content -Path $LogFile -Value "$(Get-Date): SUCCESS - $Message"
}

function Log-Warn {
    param($Message)
    Write-Warn $Message
    Add-Content -Path $LogFile -Value "$(Get-Date): WARNING - $Message"
}

function Log-Error {
    param($Message)
    Write-Error-Exit $Message
}

# Check Python installation
function Check-PythonVersion {
    Log "Checking Python version..."

    try {
        $pythonVersion = python --version 2>&1
        Write-Info "Python version: $pythonVersion"

        # Extract version number
        $versionMatch = $pythonVersion -match '(\d+)\.(\d+)\.(\d+)'
        if ($versionMatch) {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]

            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
                Log-Error "Python 3.11+ is required. Found: $pythonVersion"
            }
        }

        Log-Success "Python version check passed"
    }
    catch {
        Log-Error "Python is not installed or not in PATH. Please install Python 3.11+ from python.org"
    }
}

# Check Node.js installation
function Check-NodeVersion {
    Log "Checking Node.js version..."

    try {
        $nodeVersion = node --version
        Write-Info "Node.js version: $nodeVersion"

        # Extract major version
        $versionMatch = $nodeVersion -match 'v(\d+)\.'
        if ($versionMatch) {
            $major = [int]$Matches[1]

            if ($major -lt 18) {
                Log-Error "Node.js 18+ is required. Found: $nodeVersion"
            }
        }

        Log-Success "Node.js version check passed"
    }
    catch {
        Log-Error "Node.js is not installed or not in PATH. Please install Node.js 18+ from nodejs.org"
    }
}

# Check PostgreSQL installation
function Check-PostgreSQL {
    Log "Checking PostgreSQL installation..."

    $pgPath = Get-Command psql -ErrorAction SilentlyContinue

    if (-not $pgPath) {
        Log-Warn "PostgreSQL is not installed or not in PATH"
        Log "Download and install PostgreSQL from: https://www.postgresql.org/download/windows/"
        Log "Or use Docker (see docker-compose.yml)"

        if (-not $SkipDatabase) {
            Log-Error "PostgreSQL is required. Please install it and re-run this script."
        }
    }
    else {
        Log-Success "PostgreSQL is installed"

        # Check if PostgreSQL is running
        try {
            $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
            if ($pgService -and $pgService.Status -eq "Running") {
                Log-Success "PostgreSQL service is running"
            }
            elseif ($pgService) {
                Log-Warn "PostgreSQL service is stopped. Starting..."
                Start-Service -Name "postgresql*"
                Start-Sleep -Seconds 3
            }
        }
        catch {
            Log-Warn "Could not check PostgreSQL service status"
        }
    }
}

# Create database
function Create-Database {
    if ($SkipDatabase) {
        Log-Warn "Skipping database creation (--SkipDatabase)"
        return
    }

    Log "Creating database..."

    $dbName = "lemonade_dashboard"
    $dbUser = "lemonade_user"
    $dbPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 24 | ForEach-Object {[char]$_})

    try {
        # Create database
        Write-Info "Creating database: $dbName"
        psql -U postgres -c "CREATE DATABASE $dbName;" 2>$null

        # Create user
        Write-Info "Creating user: $dbUser"
        psql -U postgres -c "CREATE USER $dbUser WITH PASSWORD '$dbPassword';" 2>$null

        # Grant privileges
        Write-Info "Granting privileges"
        psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $dbName TO $dbUser;" 2>$null

        Log-Success "Database created: $dbName"

        # Store for later use
        $script:DbName = $dbName
        $script:DbUser = $dbUser
        $script:DbPassword = $dbPassword
        $script:DatabaseUrl = "postgresql://${dbUser}:${dbPassword}@localhost:5432/${dbName}"
    }
    catch {
        Log-Warn "Database creation may have partially succeeded. Check manually."
    }
}

# Setup backend
function Setup-Backend {
    Log "Setting up backend..."
    Set-Location $BackendDir

    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Log "Creating Python virtual environment..."
        python -m venv venv
    }
    Log-Success "Virtual environment ready"

    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    & ".\venv\Scripts\Activate.ps1"

    # Upgrade pip
    Log "Upgrading pip..."
    python -m pip install --upgrade pip

    # Install dependencies
    Log "Installing Python dependencies..."
    pip install -r requirements.txt

    Log-Success "Backend dependencies installed"

    # Create .env file
    if (-not (Test-Path ".env")) {
        Log "Creating .env file..."
        Copy-Item ".env.example" ".env"

        # Generate secret key
        $secretKey = python -c "import secrets; print(secrets.token_urlsafe(32))"

        # Update .env with actual values
        $envContent = Get-Content ".env" -Raw
        $envContent = $envContent -replace 'DATABASE_URL=.*', "DATABASE_URL=$script:DatabaseUrl"
        $envContent = $envContent -replace 'SECRET_KEY=.*', "SECRET_KEY=$secretKey"
        Set-Content ".env" -Value $envContent

        Log-Success "Backend configuration created"
    }

    # Run migrations
    Log "Running database migrations..."
    alembic upgrade head
    Log-Success "Database migrations completed"

    # Deactivate virtual environment
    deactivate
}

# Setup frontend
function Setup-Frontend {
    if ($SkipFrontend) {
        Log-Warn "Skipping frontend setup (--SkipFrontend)"
        return
    }

    Log "Setting up frontend..."
    Set-Location $FrontendDir

    # Install dependencies
    Log "Installing Node.js dependencies..."
    npm install

    Log-Success "Frontend dependencies installed"

    # Create .env file
    if (-not (Test-Path ".env")) {
        Log "Creating frontend .env file..."
        Copy-Item "..\.env.example" ".env"
        Log-Success "Frontend configuration created"
    }
}

# Create admin user
function Create-AdminUser {
    Log "Creating admin user..."
    Set-Location $BackendDir

    & ".\venv\Scripts\Activate.ps1"

    $pythonScript = @'
import sys
sys.path.insert(0, '.')

from app.database import SyncSessionLocal, init_db
from app.models import User
import bcrypt

# Initialize database
init_db()

db = SyncSessionLocal()

try:
    # Check if admin already exists
    existing_admin = db.query(User).filter(User.role == "admin").first()
    if existing_admin:
        print("Admin user already exists")
        sys.exit(0)

    # Create admin user
    admin_email = "admin@example.com"
    admin_password = "ChangeMe123!"

    hashed_password = bcrypt.hashpw(
        admin_password.encode(),
        bcrypt.gensalt()
    ).decode()

    admin_user = User(
        email=admin_email,
        name="System Administrator",
        hashed_password=hashed_password,
        role="admin",
        is_active=True,
    )

    db.add(admin_user)
    db.commit()

    print("\n" + "="*50)
    print("ADMIN USER CREATED")
    print("="*50)
    print(f"Email:    {admin_email}")
    print(f"Password: {admin_password}")
    print("\n IMPORTANT: Change password after first login!")
    print("="*50 + "\n")

except Exception as e:
    db.rollback()
    print(f"Error: {e}")
    sys.exit(1)
finally:
    db.close()
'@

    python -c $pythonScript

    deactivate
}

# Verify installation
function Verify-Installation {
    Log "Verifying installation..."

    # Check backend
    Set-Location $BackendDir
    & ".\venv\Scripts\Activate.ps1"

    try {
        python -c "import app.main" 2>$null
        Log-Success "Backend import check passed"
    }
    catch {
        Log-Error "Backend import check failed"
    }

    deactivate

    # Check frontend build
    Set-Location $FrontendDir
    try {
        npm run build 2>$null
        Log-Success "Frontend build check passed"
    }
    catch {
        Log-Warn "Frontend build check failed - may need manual intervention"
    }

    Log-Success "Installation verification completed"
}

# Print summary
function Print-Summary {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║     LEMONADE EVAL DASHBOARD - INSTALLATION COMPLETE    ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Backend Directory:  $BackendDir"
    Write-Host "Frontend Directory: $FrontendDir"
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "STARTING THE SERVERS"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host ""
    Write-Host "Terminal 1 (Backend):" -ForegroundColor Cyan
    Write-Host "  cd $BackendDir"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    Write-Host ""
    Write-Host "Terminal 2 (Frontend):" -ForegroundColor Cyan
    Write-Host "  cd $FrontendDir"
    Write-Host "  npm run dev"
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "ACCESS URLS"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host ""
    Write-Host "  Frontend:    http://localhost:3000"
    Write-Host "  Backend API: http://localhost:8000"
    Write-Host "  API Docs:    http://localhost:8000/docs"
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "DEFAULT LOGIN"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host ""
    Write-Host "  Email:    admin@example.com" -ForegroundColor Yellow
    Write-Host "  Password: ChangeMe123!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   WARNING: CHANGE THESE CREDENTIALS IMMEDIATELY!" -ForegroundColor Red
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host ""
    Log-Success "Installation completed successfully!"
    Write-Host ""
}

# Main script
function Main {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║     LEMONADE EVAL DASHBOARD - INSTALLATION SCRIPT      ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""

    # Check prerequisites
    Log "Checking prerequisites..."
    Check-PythonVersion
    Check-NodeVersion
    Check-PostgreSQL

    # Create database
    Create-Database

    # Setup backend
    Setup-Backend

    # Setup frontend
    Setup-Frontend

    # Create admin user
    Create-AdminUser

    # Verify installation
    Verify-Installation

    # Print summary
    Print-Summary
}

# Run main script
Main
