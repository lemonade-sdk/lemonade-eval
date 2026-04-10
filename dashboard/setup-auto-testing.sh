#!/bin/bash
# Auto-Integration Testing Setup Script
# This script sets up the comprehensive auto-integration testing system

set -e

echo "=============================================="
echo "Auto-Integration Testing Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo ""
echo "Setting up auto-integration testing for UI-UX Eval Dashboard..."
echo ""

# Check if we're in the right directory
if [ ! -d "$BACKEND_DIR/tests" ]; then
    echo -e "${RED}Error: backend/tests directory not found${NC}"
    echo "Please run this script from the dashboard directory"
    exit 1
fi

# Step 1: Install git hooks
echo -e "${YELLOW}Step 1: Installing git hooks...${NC}"
cd "$BACKEND_DIR"
python -c "
from tests.auto_activate_hooks import GitHookInstaller
installer = GitHookInstaller()
hooks = installer.install_hooks()
print(f'Installed hooks: {list(hooks.keys())}')
"
echo -e "${GREEN}Git hooks installed successfully${NC}"
echo ""

# Step 2: Verify test files
echo -e "${YELLOW}Step 2: Verifying test files...${NC}"
python -m py_compile tests/stress/test_rate_limiting_load.py && echo "  - test_rate_limiting_load.py: OK"
python -m py_compile tests/stress/test_cache_stampede.py && echo "  - test_cache_stampede.py: OK"
python -m py_compile tests/stress/test_websocket_stress.py && echo "  - test_websocket_stress.py: OK"
python -m py_compile tests/integration/test_import_pipeline.py && echo "  - test_import_pipeline.py: OK"
python -m py_compile tests/auto_activate_hooks.py && echo "  - auto_activate_hooks.py: OK"
python -m py_compile tests/test_reporter.py && echo "  - test_reporter.py: OK"
echo -e "${GREEN}All test files verified${NC}"
echo ""

# Step 3: Create test reports directory
echo -e "${YELLOW}Step 3: Creating test reports directory...${NC}"
mkdir -p "$BACKEND_DIR/test-reports"
echo -e "${GREEN}Test reports directory created${NC}"
echo ""

# Step 4: Display test commands
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Available test commands:"
echo ""
echo "  Run all tests:"
echo "    cd dashboard/backend && pytest --cov=app -v"
echo ""
echo "  Run unit tests only:"
echo "    pytest tests/ -m 'unit' -v"
echo ""
echo "  Run integration tests only:"
echo "    pytest tests/integration/ -m 'integration' -v"
echo ""
echo "  Run stress tests only:"
echo "    pytest tests/stress/ -v"
echo ""
echo "  Generate test reports:"
echo "    python tests/test_reporter.py tests/ --output-dir test-reports --notify"
echo ""
echo "  Run tests with coverage:"
echo "    pytest --cov=app --cov-report=html --cov-report=term-missing -v"
echo ""
echo "Documentation:"
echo "  See dashboard/backend/tests/AUTO_INTEGRATION_TESTING.md"
echo ""
echo "=============================================="
