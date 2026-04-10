# Lemonade Eval Dashboard - Developer Guide

Complete guide for developers contributing to the Lemonade Eval Dashboard.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Local Development Setup](#local-development-setup)
3. [Testing Guide](#testing-guide)
4. [Contributing Guidelines](#contributing-guidelines)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Lemonade Eval Dashboard                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │     Frontend     │    │      Backend     │                  │
│  │   React 18 + TS  │◄──►│   FastAPI + UV   │                  │
│  │   Mantine v7     │    │   SQLAlchemy     │                  │
│  │   Zustand        │    │   WebSocket      │                  │
│  │   React Query    │    │                  │                  │
│  └──────────────────┘    └──────────────────┘                  │
│           │                      │                              │
│           │                      │                              │
│           └──────────────────────┼──────────────────────────┐   │
│                                  │                          │   │
│  ┌───────────────────────────────▼──────────────────────────▼─┐ │
│  │                      Data Layer                            │ │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐   │ │
│  │  │   PostgreSQL    │    │        Redis                │   │ │
│  │  │   (Primary DB)  │    │   (Cache + Rate Limiting)   │   │ │
│  │  │   + Alembic     │    │                             │   │ │
│  │  └─────────────────┘    └─────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              External Integrations                         │ │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐   │ │
│  │  │  lemonade-eval  │    │    Prometheus/Grafana       │   │ │
│  │  │  CLI            │    │    (Monitoring)             │   │ │
│  │  └─────────────────┘    └─────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Backend Architecture

#### Layer Structure

```
backend/
├── main.py              # FastAPI application entry
├── config.py            # Configuration management
├── database.py          # Database connection
├── websocket.py         # WebSocket manager
│
├── api/
│   ├── v1/
│   │   ├── auth.py      # Authentication endpoints
│   │   ├── health.py    # Health check endpoints
│   │   ├── models.py    # Model CRUD endpoints
│   │   ├── runs.py      # Run CRUD endpoints
│   │   ├── metrics.py   # Metrics endpoints
│   │   ├── import_routes.py  # Import endpoints
│   │   └── cli_integration.py # CLI integration
│   └── deps.py          # Dependency injection
│
├── models/              # SQLAlchemy ORM models
│   ├── __init__.py
│   └── ...
│
├── schemas/             # Pydantic validation schemas
│   ├── __init__.py
│   └── ...
│
├── services/            # Business logic layer
│   ├── models.py
│   ├── runs.py
│   ├── metrics.py
│   └── import_service.py
│
├── db/
│   └── migrations/      # Alembic migrations
│
├── middleware/
│   ├── rate_limiter.py  # Rate limiting middleware
│   └── __init__.py
│
├── cache/
│   ├── cache_manager.py # Redis cache management
│   └── __init__.py
│
├── integration/
│   ├── cli_client.py    # CLI HTTP client
│   └── import_pipeline.py # Import pipeline
│
└── monitoring/
    ├── metrics.py       # Prometheus metrics
    └── __init__.py
```

#### Database Schema

```
┌─────────────────┐     ┌─────────────────┐
│     users       │     │     tags        │
├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ id (PK)         │
│ email           │     │ name            │
│ name            │     │ color           │
│ role            │     │ created_at      │
│ hashed_password │     └─────────────────┘
│ is_active       │
│ created_at      │     ┌─────────────────┐
│ updated_at      │     │   run_tags      │
└─────────────────┘     ├─────────────────┤
        │               │ run_id (FK)     │
        │               │ tag_id (FK)     │
        ▼               └─────────────────┘
┌─────────────────┐
│     models      │
├─────────────────┤
│ id (PK)         │     ┌─────────────────┐
│ name            │     │ model_versions  │
│ checkpoint      │     ├─────────────────┤
│ family          │     │ id (PK)         │
│ model_type      │     │ model_id (FK)   │
│ parameters      │     │ version_name    │
│ metadata (JSON) │     │ checkpoint      │
│ created_by (FK) │     │ created_at      │
│ created_at      │     └─────────────────┘
│ updated_at      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│      runs       │
├─────────────────┤
│ id (PK)         │
│ model_id (FK)   │
│ user_id (FK)    │
│ build_name      │
│ run_type        │
│ status          │
│ device          │
│ backend         │
│ dtype           │
│ config (JSON)   │
│ duration_seconds│
│ started_at      │
│ completed_at    │
│ created_at      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│     metrics     │
├─────────────────┤
│ id (PK)         │
│ run_id (FK)     │
│ model_id (FK)   │
│ category        │
│ name            │
│ display_name    │
│ value_numeric   │
│ value_text      │
│ unit            │
│ metadata (JSON) │
│ created_at      │
└─────────────────┘
```

### Frontend Architecture

#### Component Structure

```
frontend/src/
├── api/                 # API client layer
│   ├── client.ts        # Axios configuration
│   ├── auth.ts          # Auth API methods
│   ├── models.ts        # Models API methods
│   ├── runs.ts          # Runs API methods
│   ├── metrics.ts       # Metrics API methods
│   └── import.ts        # Import API methods
│
├── components/
│   ├── charts/          # Chart components
│   │   ├── BarChart.tsx
│   │   ├── LineChart.tsx
│   │   └── RadarChart.tsx
│   │
│   ├── common/          # Reusable UI components
│   │   ├── DataTable.tsx
│   │   ├── StatusBadge.tsx
│   │   ├── MetricCard.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ErrorDisplay.tsx
│   │
│   └── metrics/         # Metric display components
│
├── hooks/               # Custom React hooks
│   ├── useModels.ts
│   ├── useRuns.ts
│   ├── useMetrics.ts
│   ├── useImport.ts
│   └── useWebSocket.ts
│
├── pages/               # Page components
│   ├── auth/
│   │   └── LoginPage.tsx
│   ├── dashboard/
│   │   └── DashboardPage.tsx
│   ├── models/
│   │   ├── ModelsPage.tsx
│   │   └── ModelDetailPage.tsx
│   ├── runs/
│   │   ├── RunsPage.tsx
│   │   └── RunDetailPage.tsx
│   ├── compare/
│   │   └── ComparePage.tsx
│   ├── import/
│   │   └── ImportPage.tsx
│   └── settings/
│       └── SettingsPage.tsx
│
├── stores/              # Zustand state management
│   ├── authStore.ts
│   ├── uiStore.ts
│   └── notificationStore.ts
│
├── types/               # TypeScript type definitions
│
├── utils/               # Utility functions
│   └── formatters.ts
│
├── App.tsx              # Main application component
└── main.tsx             # Entry point
```

#### State Management Flow

```
┌─────────────────┐     ┌─────────────────┐
│   React Query   │     │   Zustand       │
│   (Server State)│     │   (Client State)│
├─────────────────┤     ├─────────────────┤
│ - API data      │     │ - Auth token    │
│ - Caching       │     │ - UI state      │
│ - Refetching    │     │ - Notifications │
│ - Mutations     │     │ - Preferences   │
└─────────────────┘     └─────────────────┘
         │                      │
         │                      │
         └──────────────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │   Components    │
         │   (Pages)       │
         └─────────────────┘
```

### Data Flow

#### Import Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI/CLI   │────►│   Import    │────►│  Database   │
│   Client    │     │  Service    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  WebSocket  │
                    │  Updates    │
                    └─────────────┘
```

#### Real-time Updates

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Backend   │────►│  WebSocket  │────►│  Frontend   │
│   Service   │     │   Manager   │     │  Components │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Local Development Setup

### Prerequisites

- **Python 3.12+**
- **Node.js 18+**
- **PostgreSQL 16+** (or SQLite for development)
- **Redis** (optional, for caching and rate limiting)
- **Git**

### Backend Setup

#### 1. Clone and Configure

```bash
cd dashboard/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Environment Configuration

Create `.env` file:

```bash
# Copy example
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS

# Edit .env
```

**Development Environment (.env):**

```env
# Application
APP_NAME=Lemonade Eval Dashboard
APP_VERSION=1.0.0
DEBUG=true

# Database (SQLite for development)
DATABASE_URL=sqlite:///./lemonade_dev.db
DATABASE_ASYNC_URL=sqlite+aiosqlite:///./lemonade_dev.db

# Security (development only)
SECRET_KEY=dev-secret-key-change-in-production
CLI_SECRET=dev-cli-secret

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# API Configuration
API_V1_PREFIX=/api/v1
WS_V1_PREFIX=/ws/v1

# Redis (optional for development)
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_ENABLED=false

# Pagination
DEFAULT_PAGE_SIZE=20
MAX_PAGE_SIZE=100
```

#### 3. Database Setup

```bash
# Run migrations
alembic upgrade head

# Create admin user (optional)
python scripts/create_admin.py
```

#### 4. Start Development Server

```bash
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 3001

# Or using the Makefile/command
make dev
```

#### 5. Verify Setup

```bash
# Health check
curl http://localhost:3001/api/v1/health

# Access API docs
# Open http://localhost:3001/docs in browser
```

### Frontend Setup

#### 1. Install Dependencies

```bash
cd dashboard/frontend

# Install packages
npm install
```

#### 2. Environment Configuration

Create `.env` file:

```bash
# Copy example
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS
```

**Development Environment (.env):**

```env
VITE_API_BASE_URL=http://localhost:3001
VITE_WS_BASE_URL=ws://localhost:3001
VITE_APP_NAME=Lemonade Eval Dashboard
VITE_APP_VERSION=1.0.0

# Polling intervals (seconds)
VITE_POLLING_INTERVAL_FAST=30
VITE_POLLING_INTERVAL_SLOW=15
VITE_POLLING_INTERVAL_IMPORT=2
```

#### 3. Start Development Server

```bash
npm run dev
```

#### 4. Verify Setup

Open `http://localhost:5173` in your browser.

### Docker Development

#### Using Docker Compose

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    ports:
      - "3001:3001"
    environment:
      - DEBUG=true
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/lemonade_dev
      - SECRET_KEY=dev-secret-key
    volumes:
      - ./backend:/app
      - backend_logs:/app/logs
    depends_on:
      - db
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 3001

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: development
    ports:
      - "3000:80"
    volumes:
      - ./frontend:/app
    depends_on:
      - backend

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=lemonade_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  backend_logs:
```

Start development environment:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

### IDE Configuration

#### VS Code Extensions

Recommended extensions:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "bradlc.vscode-tailwindcss",
    "dsznajer.es7-react-js-snippets"
  ]
}
```

#### VS Code Settings

`.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./dashboard/backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

---

## Testing Guide

### Backend Testing

#### Test Structure

```
backend/tests/
├── conftest.py          # Pytest configuration and fixtures
├── test_api.py          # API endpoint tests
├── test_auth.py         # Authentication tests
├── test_cli_integration.py  # CLI integration tests
├── test_import.py       # Import functionality tests
├── test_metrics.py      # Metrics API tests
├── test_models.py       # Models API tests
├── test_runs.py         # Runs API tests
├── test_services_import.py  # Import service tests
├── test_services_metrics.py # Metrics service tests
├── test_services_models.py  # Models service tests
├── test_services_runs.py    # Runs service tests
└── test_websocket.py    # WebSocket tests
```

#### Running Tests

```bash
cd dashboard/backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::test_create_model -v

# Run with HTML coverage report
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

#### Test Fixtures

The `conftest.py` provides these fixtures:

- `test_engine`: Test database engine
- `db_session`: Database session for each test
- `client`: TestClient with mocked dependencies
- `test_user`: Test user fixture
- `test_model`: Test model fixture
- `test_run`: Test run fixture
- `test_metric`: Test metric fixture

#### Writing Tests

```python
# tests/test_example.py

import pytest
from fastapi.testclient import TestClient
from app.models import Model

def test_create_model(client: TestClient, db_session, test_user):
    """Test creating a new model."""
    response = client.post(
        "/api/v1/models",
        json={
            "name": "Test Model",
            "checkpoint": "test/checkpoint",
            "model_type": "llm",
            "family": "Test",
        },
        headers={"Authorization": "Bearer token"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["data"]["name"] == "Test Model"

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await some_async_function()
    assert result == expected
```

#### Mocking External Services

```python
from unittest.mock import patch, MagicMock

@patch('app.services.external_api.fetch_data')
def test_with_mock(mock_fetch):
    mock_fetch.return_value = {"data": "mocked"}
    # Test logic here
```

### Frontend Testing

#### Test Structure

```
frontend/src/tests/
├── setup.ts             # Test configuration
├── utils.tsx            # Test utilities
├── api/
│   └── client.test.ts   # API client tests
├── components/
│   ├── DataTable.test.tsx
│   ├── LoadingSpinner.test.tsx
│   ├── MetricCard.test.tsx
│   └── StatusBadge.test.tsx
├── hooks/
│   ├── useModels.test.ts
│   └── useRuns.test.ts
├── stores/
│   └── authStore.test.ts
└── utils/
    └── formatters.test.ts
```

#### Running Tests

```bash
cd dashboard/frontend

# Run all tests
npm run test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run specific test file
npm run test -- src/tests/components/DataTable.test.tsx
```

#### Writing Component Tests

```typescript
// src/tests/components/MetricCard.test.tsx

import { render, screen } from '@testing-library/react';
import { MantineProvider } from '@mantine/core';
import { MetricCard } from '@/components/common';

describe('MetricCard', () => {
  it('renders metric value correctly', () => {
    render(
      <MantineProvider>
        <MetricCard
          title="Test Metric"
          value={42.5}
          unit="tokens/s"
        />
      </MantineProvider>
    );

    expect(screen.getByText('Test Metric')).toBeInTheDocument();
    expect(screen.getByText('42.5 tokens/s')).toBeInTheDocument();
  });

  it('handles loading state', () => {
    render(
      <MantineProvider>
        <MetricCard
          title="Test Metric"
          value={null}
          loading
        />
      </MantineProvider>
    );

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
});
```

#### Writing Hook Tests

```typescript
// src/tests/hooks/useModels.test.ts

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useModels } from '@/hooks/useModels';

const createWrapper = () => {
  const queryClient = new QueryClient();
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('useModels', () => {
  it('fetches models successfully', async () => {
    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toBeDefined();
  });
});
```

#### Writing Store Tests

```typescript
// src/tests/stores/authStore.test.ts

import { authStore } from '@/stores/authStore';

describe('authStore', () => {
  beforeEach(() => {
    authStore.setState({
      token: null,
      user: null,
      isAuthenticated: false,
    });
  });

  it('sets authentication state', () => {
    const mockUser = { id: '1', email: 'test@example.com' };
    const mockToken = 'test-token';

    authStore.getState().setAuth(mockUser, mockToken);

    expect(authStore.getState().isAuthenticated).toBe(true);
    expect(authStore.getState().token).toBe(mockToken);
  });

  it('clears authentication state', () => {
    authStore.getState().setAuth({ id: '1' }, 'token');
    authStore.getState().clearAuth();

    expect(authStore.getState().isAuthenticated).toBe(false);
    expect(authStore.getState().token).toBeNull();
  });
});
```

### E2E Testing

#### Playwright Configuration

```typescript
// e2e/auth.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('login with valid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'SecurePassword123');
    await page.click('button[type="submit"]');

    await expect(page).toHaveURL('/dashboard');
    await expect(page.getByText('Dashboard')).toBeVisible();
  });

  test('login with invalid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.fill('[name="email"]', 'invalid@example.com');
    await page.fill('[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');

    await expect(page.getByText('Invalid credentials')).toBeVisible();
  });
});
```

#### Running E2E Tests

```bash
cd dashboard/frontend

# Run all E2E tests
npm run test:e2e

# Run with UI
npx playwright test --ui

# Run specific test
npx playwright test e2e/auth.spec.ts
```

### Test Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| Backend | 75% |
| Frontend | 60% |
| Critical Services | 85% |
| API Endpoints | 80% |

---

## Contributing Guidelines

### Code Style

#### Python

Follow PEP 8 guidelines:

```python
# Use type hints
def create_model(name: str, checkpoint: str) -> Model:
    """Create a new model."""
    return Model(name=name, checkpoint=checkpoint)

# Use docstrings
class ModelService:
    """Service layer for model operations."""

    def get_model(self, model_id: str) -> Model | None:
        """
        Retrieve a model by ID.

        Args:
            model_id: The model UUID

        Returns:
            Model if found, None otherwise
        """
        return self.db.query(Model).filter(Model.id == model_id).first()
```

#### TypeScript/React

```typescript
// Use TypeScript interfaces
interface Model {
  id: string;
  name: string;
  checkpoint: string;
  modelType: 'llm' | 'vlm' | 'embedding';
}

// Use functional components with hooks
const ModelCard: React.FC<{ model: Model }> = ({ model }) => {
  return (
    <Card>
      <Text>{model.name}</Text>
    </Card>
  );
};

// Use proper typing for hooks
const useModels = (params: UseModelsParams): UseModelsReturn => {
  // Implementation
};
```

### Git Workflow

#### Branch Naming

```
feature/add-model-search
fix/authentication-timeout
docs/update-api-reference
test/add-metrics-tests
refactor/optimize-queries
```

#### Commit Messages

Follow conventional commits:

```
feat: Add model search functionality

- Implement search by name and checkpoint
- Add family filter dropdown
- Update ModelsPage with search bar

Fixes: #123

feat(api): Add pagination to models endpoint

- Add page and per_page query parameters
- Update response format with meta object
- Add max_page_size validation

BREAKING CHANGE: Response format changed for /api/v1/models

fix(auth): Resolve token expiration issue

- Increase token expiration to 30 minutes
- Add token refresh mechanism
- Clear token on 401 response

Closes: #456

docs: Update API documentation

- Add examples for all endpoints
- Document error codes
- Update rate limiting section
```

### Pull Request Process

1. **Create Branch**: `git checkout -b feature/your-feature`

2. **Make Changes**: Implement your feature or fix

3. **Run Tests**: Ensure all tests pass
   ```bash
   # Backend
   cd dashboard/backend && pytest

   # Frontend
   cd dashboard/frontend && npm run test
   ```

4. **Update Documentation**: Add/update relevant docs

5. **Create PR**: Submit pull request with:
   - Clear title
   - Description of changes
   - Screenshots (if UI changes)
   - Test evidence

### PR Checklist

```markdown
## Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No sensitive data committed
- [ ] Breaking changes documented
```

### Code Review Guidelines

#### Reviewers Should Check

1. **Functionality**: Does it work as expected?
2. **Security**: Any security concerns?
3. **Performance**: Any performance issues?
4. **Tests**: Adequate test coverage?
5. **Documentation**: Updated and accurate?

#### Review Response Time

- Acknowledge PR within 24 hours
- Complete review within 48 hours
- Provide constructive feedback

### Release Process

#### Version Numbering

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

#### Release Steps

1. Update version in `pyproject.toml` and `package.json`
2. Update `CHANGELOG.md`
3. Create release tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release

### Reporting Issues

When reporting issues, include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: How to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, browser, versions
- **Logs**: Relevant error logs
- **Screenshots**: If applicable

### Security Guidelines

1. **Never commit secrets**: Use environment variables
2. **Input validation**: Always validate user input
3. **SQL injection**: Use parameterized queries
4. **XSS prevention**: Sanitize user content
5. **Authentication**: Verify authentication on all protected endpoints
6. **Rate limiting**: Implement rate limiting on public endpoints

### Accessibility

Ensure accessibility compliance:

- Use semantic HTML
- Add ARIA labels where needed
- Ensure keyboard navigation
- Maintain color contrast ratios
- Test with screen readers

---

## Additional Resources

- **API Documentation**: `http://localhost:3001/docs`
- **Database Migrations**: `dashboard/backend/alembic/versions/`
- **Component Library**: Mantine v7 docs at https://mantine.dev/
- **Testing Library**: https://testing-library.com/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
