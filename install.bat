@echo off
REM Lemonade Eval Dashboard - Installation Script (Windows Batch)
REM This script automates the complete setup and installation process

echo.
echo ===============================================
echo    LEMONADE EVAL DASHBOARD - INSTALLATION
echo ===============================================
echo.

REM Set paths
set SCRIPT_DIR=%~dp0
set DASHBOARD_DIR=%SCRIPT_DIR%dashboard
set BACKEND_DIR=%DASHBOARD_DIR%\backend
set FRONTEND_DIR=%DASHBOARD_DIR%\frontend
set LOG_FILE=%SCRIPT_DIR%install.log

REM Logging function
:log
echo [INFO] %1 | tee -a "%LOG_FILE%"
goto :eof

:success
echo [SUCCESS] %1 | tee -a "%LOG_FILE%"
goto :eof

:warn
echo [WARNING] %1 | tee -a "%LOG_FILE%"
goto :eof

:error
echo [ERROR] %1 | tee -a "%LOG_FILE%"
exit /b 1

:check_python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    call :error "Python is not installed or not in PATH"
)
python --version
call :success "Python found"
goto :check_node

:check_node
echo.
echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    call :error "Node.js is not installed or not in PATH"
)
node --version
call :success "Node.js found"
goto :check_npm

:check_npm
echo.
echo Checking npm installation...
npm --version >nul 2>&1
if errorlevel 1 (
    call :error "npm is not installed or not in PATH"
)
npm --version
call :success "npm found"
goto :check_postgresql

:check_postgresql
echo.
echo Checking PostgreSQL installation...
where psql >nul 2>&1
if errorlevel 1 (
    call :warn "PostgreSQL is not installed or not in PATH"
    echo Download from: https://www.postgresql.org/download/windows/
    echo Or use Docker (see docker-compose.yml)
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
) else (
    call :success "PostgreSQL found"
)
goto :create_database

:create_database
echo.
echo Creating database...
set DB_NAME=lemonade_dashboard
set DB_USER=lemonade_user
set DB_PASSWORD=%RANDOM%%RANDOM%%RANDOM%

REM Create database using psql
psql -U postgres -c "CREATE DATABASE %DB_NAME%;" >nul 2>&1 || echo Database may already exist
psql -U postgres -c "CREATE USER %DB_USER% WITH PASSWORD '%DB_PASSWORD%';" >nul 2>&1 || echo User may already exist
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE %DB_NAME% TO %DB_USER%;" >nul 2>&1

call :success "Database configured"
set DATABASE_URL=postgresql://%DB_USER%:%DB_PASSWORD%@localhost:5432/%DB_NAME%
goto :setup_backend

:setup_backend
echo.
echo Setting up backend...
cd /d "%BACKEND_DIR%"

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call :success "Virtual environment ready"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

call :success "Backend dependencies installed"

REM Create .env file
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
)

REM Run migrations
echo Running database migrations...
alembic upgrade head

call :success "Database migrations completed"

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

goto :setup_frontend

:setup_frontend
echo.
echo Setting up frontend...
cd /d "%FRONTEND_DIR%"

REM Install dependencies
echo Installing Node.js dependencies...
call npm install

call :success "Frontend dependencies installed"

REM Create .env file
if not exist ".env" (
    echo Creating frontend .env file...
    copy ..\.env.example .env
)

goto :create_admin

:create_admin
echo.
echo Creating admin user...
cd /d "%BACKEND_DIR%"
call venv\Scripts\activate.bat

python -c "import sys; sys.path.insert(0, '.'); from app.database import SyncSessionLocal, init_db; from app.models import User; import bcrypt; init_db(); db = SyncSessionLocal(); admin = db.query(User).filter(User.role == 'admin').first(); db.close(); sys.exit(0) if admin else (lambda: (db.add(User(email='admin@example.com', name='System Administrator', hashed_password=bcrypt.hashpw('ChangeMe123!'.encode(), bcrypt.gensalt()).decode(), role='admin', is_active=True)), db.commit(), db.close(), print('\n==================================================\nADMIN USER CREATED\n==================================================\nEmail:    admin@example.com\nPassword: ChangeMe123!\n\nIMPORTANT: Change password after first login!\n==================================================\n')))()"

call venv\Scripts\deactivate.bat

goto :verify

:verify
echo.
echo Verifying installation...
cd /d "%BACKEND_DIR%"
call venv\Scripts\activate.bat

python -c "import app.main" >nul 2>&1
if errorlevel 1 (
    call :error "Backend import check failed"
)
call :success "Backend check passed"

call venv\Scripts\deactivate.bat

cd /d "%FRONTEND_DIR%"
call npm run build >nul 2>&1
if errorlevel 1 (
    call :warn "Frontend build check failed"
) else (
    call :success "Frontend build check passed"
)

goto :summary

:summary
echo.
echo ===============================================
echo    INSTALLATION COMPLETE
echo ===============================================
echo.
echo Backend Directory:  %BACKEND_DIR%
echo Frontend Directory: %FRONTEND_DIR%
echo.
echo ------------------------------------------------
echo STARTING THE SERVERS
echo ------------------------------------------------
echo.
echo Terminal 1 (Backend):
echo   cd %BACKEND_DIR%
echo   venv\Scripts\activate.bat
echo   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo Terminal 2 (Frontend):
echo   cd %FRONTEND_DIR%
echo   npm run dev
echo.
echo ------------------------------------------------
echo ACCESS URLS
echo ------------------------------------------------
echo.
echo   Frontend:    http://localhost:3000
echo   Backend API: http://localhost:8000
echo   API Docs:    http://localhost:8000/docs
echo.
echo ------------------------------------------------
echo DEFAULT LOGIN
echo ------------------------------------------------
echo.
echo   Email:    admin@example.com
echo   Password: ChangeMe123!
echo.
echo   WARNING: CHANGE THESE CREDENTIALS IMMEDIATELY!
echo.
echo ===============================================
echo Installation completed successfully!
echo ===============================================
echo.

exit /b 0
