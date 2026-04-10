@echo off
REM Lemonade Eval Dashboard - Quick Start Script (Windows Batch)
REM Starts both backend and frontend servers

echo.
echo ===============================================
echo    LEMONADE EVAL DASHBOARD - QUICK START
echo ===============================================
echo.

REM Set paths
set SCRIPT_DIR=%~dp0
set DASHBOARD_DIR=%SCRIPT_DIR%dashboard
set BACKEND_DIR=%DASHBOARD_DIR%\backend
set FRONTEND_DIR=%DASHBOARD_DIR%\frontend

REM Check if backend is set up
if not exist "%BACKEND_DIR%\venv" (
    echo [ERROR] Backend not set up. Run install.bat first
    exit /b 1
)

REM Check if frontend is set up
if not exist "%FRONTEND_DIR%\node_modules" (
    echo [ERROR] Frontend not set up. Run install.bat first
    exit /b 1
)

REM Start backend in new window
echo Starting backend server...
start "Lemonade Backend" cmd /k "cd /d %BACKEND_DIR% && venv\Scripts\activate.bat && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start frontend in new window
echo Starting frontend server...
start "Lemonade Frontend" cmd /k "cd /d %FRONTEND_DIR% && npm run dev"

echo.
echo ===============================================
echo    SERVERS STARTED
echo ===============================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo To stop servers, close the command windows.
echo ===============================================
echo.

REM Wait for servers to start
timeout /t 5 /nobreak >nul

REM Check backend health
curl -s http://localhost:8000/api/v1/health >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Backend may still be starting...
) else (
    echo [SUCCESS] Backend is healthy
)

echo.

REM Open browser
start http://localhost:3000
echo [INFO] Opened browser to http://localhost:3000
echo.

exit /b 0
