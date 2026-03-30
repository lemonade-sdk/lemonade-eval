@echo off
REM Startup script for Lemonade Eval Dashboard Backend (Windows)

echo ========================================
echo Lemonade Eval Dashboard - Backend Setup
echo ========================================

REM Check Python version
echo.
echo Checking Python version...
python --version
py -3.12 --version 2>nul
if %errorlevel% neq 0 (
    echo Error: Python 3.12 or higher is required
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    py -3.12 -m venv venv
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo.
    echo Creating .env file from template...
    copy .env.example .env
    echo Please update .env with your configuration
)

REM Run migrations
echo.
echo Running database migrations...
alembic upgrade head

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the server, run:
echo   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo Then visit:
echo   - API Docs: http://localhost:8000/docs
echo   - Health:   http://localhost:8000/api/v1/health
echo.
