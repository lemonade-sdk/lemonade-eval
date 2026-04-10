# Lemonade Eval Dashboard - Setup Guide

This guide provides step-by-step instructions for setting up the Lemonade Eval Dashboard development environment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Database Configuration](#database-configuration)
- [Creating First Admin User](#creating-first-admin-user)
- [Running in Development Mode](#running-in-development-mode)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Node.js**: Version 18.x or higher
  - Download from [nodejs.org](https://nodejs.org/)
  - Verify installation: `node --version`

- **npm**: Version 9.x or higher (comes with Node.js)
  - Verify installation: `npm --version`

- **Python**: Version 3.11 or higher
  - Download from [python.org](https://www.python.org/)
  - Verify installation: `python --version`

- **PostgreSQL**: Version 14 or higher
  - Download from [postgresql.org](https://www.postgresql.org/)
  - Verify installation: `psql --version`

### Optional Tools

- **git**: For version control
- **Docker**: For containerized development (optional)
- **Lemonade SDK**: For importing evaluation results

## Backend Setup

### 1. Navigate to Backend Directory

```bash
cd dashboard/backend
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install the core dependencies:

```bash
pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings \
    python-jose[cryptography] bcrypt python-multipart \
    asyncpg pytest pytest-asyncio httpx
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp ../.env.example .env
```

Edit `.env` and configure:

- `SECRET_KEY`: Generate a secure key for production
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
- `DATABASE_URL`: Your PostgreSQL connection string
- `DEBUG=true`: For development mode

### 5. Initialize Database

```bash
# The database will be auto-initialized on first run in debug mode
# Or manually run:
python -m app.database
```

### 6. Create First Admin User

**IMPORTANT:** The application does not have a `/register` endpoint for security reasons.
Users must be created manually via script or SQL. Here's how to create the first admin user:

#### Option A: Python Script (Recommended)

Create a file `create_admin.py` in the `dashboard/backend` directory:

```python
#!/usr/bin/env python
"""Script to create the first admin user."""

import sys
sys.path.insert(0, '.')

from app.database import get_db, init_db
from app.models import User
import bcrypt

# Initialize database first
init_db()

# Get database session
db = next(get_db())

try:
    # Check if admin already exists
    existing_admin = db.query(User).filter(User.role == "admin").first()
    if existing_admin:
        print(f"Admin user already exists: {existing_admin.email}")
        sys.exit(0)

    # Create admin user with properly hashed password
    admin_email = "admin@example.com"
    admin_password = "ChangeMe123!"  # CHANGE THIS IMMEDIATELY!

    # Hash password using bcrypt (same method as auth module)
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

    print(f"✓ Admin user created successfully!")
    print(f"  Email: {admin_email}")
    print(f"  Password: {admin_password}")
    print("\n⚠️  IMPORTANT: Change the password after first login!")

except Exception as e:
    db.rollback()
    print(f"Error creating admin user: {e}")
    sys.exit(1)
finally:
    db.close()
```

Run the script:

```bash
python create_admin.py
```

#### Option B: Manual SQL

If you need to create a user directly in the database:

```sql
-- First, generate a bcrypt hash using Python:
-- python -c "import bcrypt; print(bcrypt.hashpw(b'YourPassword123!', bcrypt.gensalt()).decode())"

-- Then insert into database:
INSERT INTO users (id, email, name, hashed_password, role, is_active, created_at, updated_at)
VALUES (
    gen_random_uuid(),  -- or use uuid_generate_v4() depending on PostgreSQL setup
    'admin@example.com',
    'System Administrator',
    '$2b$12$...paste-bcrypt-hash-here...',  -- Replace with actual bcrypt hash
    'admin',
    true,
    NOW(),
    NOW()
);
```

#### Password Requirements

When creating users, passwords must meet these requirements:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number

#### Default Credentials (Development Only)

For development, you can use:
- **Email:** `admin@example.com`
- **Password:** `ChangeMe123!`

**⚠️  CRITICAL:** Change these credentials immediately in production!

## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd dashboard/frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

```bash
# The frontend uses Vite for environment variables
# Copy from the dashboard root
cp ../.env.example .env
```

Edit `.env` and configure:

- `VITE_API_BASE_URL=http://localhost:8000`: Backend API URL

### 4. Verify Installation

```bash
# Type check
npm run type-check

# Lint
npm run lint
```

## Database Configuration

### PostgreSQL Setup

1. **Install PostgreSQL** (if not already installed)

2. **Create Database**

   ```sql
   CREATE DATABASE lemonade_dashboard;
   ```

3. **Create User** (optional, for dedicated access)

   ```sql
   CREATE USER dashboard_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE lemonade_dashboard TO dashboard_user;
   ```

4. **Update Connection String**

   In your `.env` file:
   ```
   DATABASE_URL="postgresql://dashboard_user:your_password@localhost:5432/lemonade_dashboard"
   ```

### SQLite (For Testing Only)

For quick testing without PostgreSQL:

```
TEST_DATABASE_URL="sqlite:///./test.db"
DEBUG=true
```

## Running in Development Mode

### Start Backend Server

```bash
cd dashboard/backend

# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Start server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

### Start Frontend Development Server

```bash
cd dashboard/frontend

# Start Vite dev server
npm run dev
```

The frontend will be available at:
- App: http://localhost:3000

### Access the Application

1. Open your browser and navigate to http://localhost:3000
2. Navigate to the login page
3. Use any email and password for demo purposes (or create a user via API)

## Running Tests

### Backend Tests

```bash
cd dashboard/backend

# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_auth.py -v

# Run with coverage
pytest --cov=app --cov-report=html
```

### Frontend Tests

```bash
cd dashboard/frontend

# Run unit tests
npm run test

# Run tests in UI mode
npm run test:ui

# Run tests with coverage
npm run test:coverage

# Run type checking
npm run type-check

# Run linter
npm run lint
```

### End-to-End Tests (Frontend)

```bash
cd dashboard/frontend

# Install Playwright browsers (first time only)
npx playwright install

# Run E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e -- --ui
```

## Troubleshooting

### Backend Issues

**Database Connection Error**
```
Error: could not connect to database
```
- Ensure PostgreSQL is running: `pg_ctl status`
- Check DATABASE_URL in `.env`
- Verify database exists: `psql -U postgres -l`

**Port Already in Use**
```
Error: Address already in use
```
- Change port in uvicorn command: `--port 8001`
- Or kill the process using port 8000

**Import Errors**
```
ModuleNotFoundError: No module named 'xxx'
```
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Frontend Issues

**Node Module Errors**
```
Error: Cannot find module 'xxx'
```
- Delete `node_modules` and reinstall:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

**API Connection Errors**
```
Network Error / CORS Error
```
- Ensure backend is running on http://localhost:8000
- Check VITE_API_BASE_URL in frontend `.env`
- Verify CORS_ORIGINS in backend `.env`

**Port Already in Use**
```
Error: Port 3000 is already in use
```
- Change port in `vite.config.ts` or use different port:
  ```bash
  npm run dev -- --port 3001
  ```

### Common Issues

**Authentication Not Working**
- Clear browser localStorage and sessionStorage
- Ensure SECRET_KEY is consistent
- Check token expiration settings

**TypeScript Errors**
- Run `npm run type-check` to see all errors
- Ensure all dependencies are installed
- Check for missing type definitions

**Build Fails**
- Clear cache: `rm -rf node_modules/.vite`
- Reinstall dependencies
- Check Node.js version compatibility

## Next Steps

After successful setup:
1. Review the [README.md](./README.md) for project overview
2. Check the API documentation at http://localhost:8000/docs
3. Start developing features or running evaluations

For production deployment, see the deployment guide (coming soon).
