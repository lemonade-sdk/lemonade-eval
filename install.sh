#!/usr/bin/env bash
# Lemonade Eval Dashboard - Production Installation Script (Linux/macOS)
# This script automates the complete setup and installation process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="${SCRIPT_DIR}/dashboard"
BACKEND_DIR="${DASHBOARD_DIR}/backend"
FRONTEND_DIR="${DASHBOARD_DIR}/frontend"
VENV_DIR="${BACKEND_DIR}/venv"
LOG_FILE="${SCRIPT_DIR}/install.log"

# Functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "$1 is not installed. Please install $1 first."
    fi
}

check_python_version() {
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    local major_version=$(echo "$python_version" | cut -d. -f1)
    local minor_version=$(echo "$python_version" | cut -d. -f2)

    if [[ "$major_version" -lt 3 ]] || [[ "$major_version" -eq 3 && "$minor_version" -lt 11 ]]; then
        error "Python 3.11+ is required. Found: $python_version"
    fi
    success "Python version check passed: $python_version"
}

check_node_version() {
    local node_version=$(node --version 2>&1 | cut -d'v' -f2)
    local major_version=$(echo "$node_version" | cut -d. -f1)

    if [[ "$major_version" -lt 18 ]]; then
        error "Node.js 18+ is required. Found: $node_version"
    fi
    success "Node.js version check passed: $node_version"
}

setup_postgresql() {
    log "Checking PostgreSQL installation..."

    if ! command -v psql &> /dev/null; then
        warn "PostgreSQL is not installed."
        log "Installing PostgreSQL..."

        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install postgresql@15
            brew services start postgresql@15
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if command -v apt &> /dev/null; then
                sudo apt update
                sudo apt install -y postgresql postgresql-contrib
                sudo systemctl start postgresql
                sudo systemctl enable postgresql
            elif command -v yum &> /dev/null; then
                sudo yum install -y postgresql postgresql-server
                sudo systemctl start postgresql
                sudo systemctl enable postgresql
            fi
        fi
    else
        success "PostgreSQL is already installed"
    fi

    # Check if PostgreSQL is running
    if ! pg_isready &> /dev/null; then
        warn "PostgreSQL is not running. Starting..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew services start postgresql@15
        else
            sudo systemctl start postgresql
        fi
        sleep 2
    fi
    success "PostgreSQL is running"
}

create_database() {
    log "Creating database..."

    local db_name="lemonade_dashboard"
    local db_user="lemonade_user"
    local db_password=$(openssl rand -base64 24)

    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE ${db_name};" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE USER ${db_user} WITH PASSWORD '${db_password}';" 2>/dev/null || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${db_name} TO ${db_user};" 2>/dev/null || true

    success "Database created: ${db_name}"

    # Store credentials for later use
    export DB_NAME="$db_name"
    export DB_USER="$db_user"
    export DB_PASSWORD="$db_password"
    export DATABASE_URL="postgresql://${db_user}:${db_password}@localhost:5432/${db_name}"
}

setup_backend() {
    log "Setting up backend..."
    cd "$BACKEND_DIR"

    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    success "Virtual environment ready"

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip

    # Install dependencies
    log "Installing Python dependencies..."
    pip install -r requirements.txt

    success "Backend dependencies installed"

    # Create .env file
    if [[ ! -f ".env" ]]; then
        log "Creating .env file..."
        cp .env.example .env

        # Generate secret key
        local secret_key=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

        # Update .env with actual values
        sed -i.bak "s|DATABASE_URL=.*|DATABASE_URL=${DATABASE_URL}|" .env
        sed -i.bak "s|SECRET_KEY=.*|SECRET_KEY=${secret_key}|" .env
        rm -f .env.bak

        success "Backend configuration created"
    fi

    # Run migrations
    log "Running database migrations..."
    alembic upgrade head
    success "Database migrations completed"

    # Deactivate virtual environment
    deactivate
}

setup_frontend() {
    log "Setting up frontend..."
    cd "$FRONTEND_DIR"

    # Install dependencies
    log "Installing Node.js dependencies..."
    npm install

    success "Frontend dependencies installed"

    # Create .env file
    if [[ ! -f ".env" ]]; then
        log "Creating frontend .env file..."
        cp ../.env.example .env
        success "Frontend configuration created"
    fi
}

create_admin_user() {
    log "Creating admin user..."
    cd "$BACKEND_DIR"

    source venv/bin/activate

    python3 << 'PYTHON_SCRIPT'
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
    print("\n⚠️  IMPORTANT: Change password after first login!")
    print("="*50 + "\n")

except Exception as e:
    db.rollback()
    print(f"Error: {e}")
    sys.exit(1)
finally:
    db.close()
PYTHON_SCRIPT

    deactivate
}

verify_installation() {
    log "Verifying installation..."

    # Check backend
    cd "$BACKEND_DIR"
    source venv/bin/activate

    if python3 -c "import app.main" 2>/dev/null; then
        success "Backend import check passed"
    else
        error "Backend import check failed"
    fi

    deactivate

    # Check frontend build
    cd "$FRONTEND_DIR"
    if npm run build 2>/dev/null; then
        success "Frontend build check passed"
    else
        warn "Frontend build check failed - may need manual intervention"
    fi

    success "Installation verification completed"
}

print_summary() {
    echo ""
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║     LEMONADE EVAL DASHBOARD - INSTALLATION COMPLETE    ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "Backend Directory:  $BACKEND_DIR"
    echo "Frontend Directory: $FRONTEND_DIR"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STARTING THE SERVERS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Terminal 1 (Backend):"
    echo "  cd $BACKEND_DIR"
    echo "  source venv/bin/activate"
    echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "Terminal 2 (Frontend):"
    echo "  cd $FRONTEND_DIR"
    echo "  npm run dev"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "ACCESS URLS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Frontend:    http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Docs:    http://localhost:8000/docs"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "DEFAULT LOGIN"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Email:    admin@example.com"
    echo "  Password: ChangeMe123!"
    echo ""
    echo "  ⚠️  CHANGE THESE CREDENTIALS IMMEDIATELY!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    success "Installation completed successfully!"
    echo ""
}

# Main script
main() {
    echo ""
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║     LEMONADE EVAL DASHBOARD - INSTALLATION SCRIPT      ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""

    # Check prerequisites
    log "Checking prerequisites..."
    check_command python3
    check_command node
    check_command npm
    check_python_version
    check_node_version

    # Setup PostgreSQL
    setup_postgresql

    # Create database
    create_database

    # Setup backend
    setup_backend

    # Setup frontend
    setup_frontend

    # Create admin user
    create_admin_user

    # Verify installation
    verify_installation

    # Print summary
    print_summary
}

# Run main script
main "$@"
