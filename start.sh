#!/usr/bin/env bash
# Lemonade Eval Dashboard - Quick Start Script
# Starts both backend and frontend servers for development

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="${SCRIPT_DIR}/dashboard"
BACKEND_DIR="${DASHBOARD_DIR}/backend"
FRONTEND_DIR="${DASHBOARD_DIR}/frontend"

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║       LEMONADE EVAL DASHBOARD - QUICK START            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if backend is set up
if [[ ! -d "${BACKEND_DIR}/venv" ]]; then
    echo -e "${RED}Error: Backend not set up. Run ./install.sh first${NC}"
    exit 1
fi

# Check if frontend is set up
if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    echo -e "${RED}Error: Frontend not set up. Run ./install.sh first${NC}"
    exit 1
fi

# Start backend in background
echo -e "${BLUE}Starting backend server...${NC}"
cd "$BACKEND_DIR"
source venv/bin/activate

# Check if port 8000 is already in use
if lsof -i :8000 &> /dev/null; then
    echo -e "${YELLOW}Warning: Port 8000 is already in use${NC}"
else
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
    BACKEND_PID=$!
    echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
fi

# Start frontend in background
echo -e "${BLUE}Starting frontend server...${NC}"
cd "$FRONTEND_DIR"

# Check if port 3000 is already in use
if lsof -i :3000 &> /dev/null; then
    echo -e "${YELLOW}Warning: Port 3000 is already in use${NC}"
else
    npm run dev > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Servers starting...${NC}"
echo ""
echo -e "${BLUE}Backend:${NC}  http://localhost:8000"
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo ""
echo -e "${YELLOW}To stop servers:${NC}"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo -e "${YELLOW}Or press Ctrl+C to stop${NC}"
echo ""

# Wait for servers to start
sleep 5

# Check if servers are running
if curl -s http://localhost:8000/api/v1/health &> /dev/null; then
    echo -e "${GREEN}✓ Backend is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Backend may still be starting...${NC}"
fi

echo ""
echo "View logs:"
echo "  Backend:  tail -f /tmp/backend.log"
echo "  Frontend: tail -f /tmp/frontend.log"
echo ""

# Keep script running
wait
