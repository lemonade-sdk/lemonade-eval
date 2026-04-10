#!/bin/bash
# Startup script for Lemonade Eval Dashboard Backend

set -e

echo "========================================"
echo "Lemonade Eval Dashboard - Backend Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if running Python 3.12+
required_version="3.12"
if [ "$(printf '%s\n' "$required_version" "$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.12 or higher is required${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please update .env with your configuration${NC}"
fi

# Check if PostgreSQL is running
echo -e "\n${YELLOW}Checking database connection...${NC}"
if command -v psql &> /dev/null; then
    if psql -h localhost -U postgres -d lemonade_dashboard -c '\q' 2>/dev/null; then
        echo -e "${GREEN}Database connection successful${NC}"
    else
        echo -e "${RED}Warning: Could not connect to PostgreSQL${NC}"
        echo "Make sure PostgreSQL is running and the database exists"
        echo "Create with: createdb -U postgres lemonade_dashboard"
    fi
else
    echo -e "${YELLOW}psql not found, skipping database check${NC}"
fi

# Run migrations
echo -e "\n${YELLOW}Running database migrations...${NC}"
if command -v alembic &> /dev/null; then
    alembic upgrade head
    echo -e "${GREEN}Migrations completed${NC}"
else
    echo -e "${RED}Alembic not found. Install with: pip install alembic${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start the server, run:"
echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then visit:"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health:   http://localhost:8000/api/v1/health"
echo ""
