# Lemonade Eval Dashboard - Backend

FastAPI-based backend service for the Lemonade Eval Dashboard, providing REST API and WebSocket support for managing LLM/VLM evaluation results.

## Features

- **REST API**: CRUD operations for models, runs, and metrics
- **Real-time Updates**: WebSocket support for live evaluation progress
- **YAML Import**: Migration tool for importing existing lemonade-eval results
- **PostgreSQL**: Robust data storage with SQLAlchemy ORM
- **Alembic Migrations**: Database schema version control

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- pip

### Installation

1. **Install dependencies:**

```bash
cd dashboard/backend
pip install -r requirements.txt
```

2. **Set up environment variables:**

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lemonade_dashboard
DATABASE_ASYNC_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/lemonade_dashboard

# Security
SECRET_KEY=your-secret-key-change-in-production

# Application
DEBUG=true
```

3. **Run migrations:**

```bash
alembic upgrade head
```

4. **Start the server:**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API:**

- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/api/v1/health

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database connection
│   ├── websocket.py         # WebSocket handler
│   ├── models/              # SQLAlchemy models
│   │   ├── __init__.py
│   │   └── ...
│   ├── schemas/             # Pydantic schemas
│   │   ├── __init__.py
│   │   └── ...
│   ├── api/
│   │   ├── v1/
│   │   │   ├── models.py    # Model endpoints
│   │   │   ├── runs.py      # Run endpoints
│   │   │   ├── metrics.py   # Metric endpoints
│   │   │   ├── import_routes.py  # Import endpoints
│   │   │   └── health.py    # Health endpoints
│   │   └── deps.py          # Dependencies
│   ├── services/            # Business logic
│   │   ├── models.py
│   │   ├── runs.py
│   │   ├── metrics.py
│   │   └── import_service.py
│   └── db/
│       └── migrations/      # Alembic migrations
├── tests/
│   └── test_api.py
├── requirements.txt
└── alembic.ini
```

## API Endpoints

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/health/ready` | Readiness check |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models` | List models |
| POST | `/api/v1/models` | Create model |
| GET | `/api/v1/models/{id}` | Get model details |
| PUT | `/api/v1/models/{id}` | Update model |
| DELETE | `/api/v1/models/{id}` | Delete model |
| GET | `/api/v1/models/{id}/versions` | List model versions |
| GET | `/api/v1/models/{id}/runs` | Get model runs |

### Runs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/runs` | List runs |
| POST | `/api/v1/runs` | Create run |
| GET | `/api/v1/runs/{id}` | Get run details |
| PUT | `/api/v1/runs/{id}` | Update run |
| DELETE | `/api/v1/runs/{id}` | Delete run |
| POST | `/api/v1/runs/{id}/status` | Update run status |
| GET | `/api/v1/runs/{id}/metrics` | Get run metrics |

### Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/metrics` | List metrics |
| POST | `/api/v1/metrics` | Create metric |
| POST | `/api/v1/metrics/bulk` | Bulk create metrics |
| GET | `/api/v1/metrics/aggregate` | Aggregate metrics |
| GET | `/api/v1/metrics/trends` | Get metric trends |
| GET | `/api/v1/metrics/compare` | Compare metrics |

### Import

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/import/yaml` | Import YAML files |
| GET | `/api/v1/import/status/{id}` | Get import status |
| GET | `/api/v1/import/scan` | Scan cache directory |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/v1/evaluations` | Real-time evaluation updates |

## Database Schema

The backend uses the following main tables:

- **users**: Dashboard users with authentication
- **models**: LLM/VLM models being evaluated
- **model_versions**: Different variants of a model
- **runs**: Evaluation runs
- **metrics**: Performance and accuracy metrics
- **tags**: Tags for organization
- **run_tags**: Many-to-many relationship

## YAML Import

The import service can migrate existing lemonade-eval YAML files:

```python
# Using the API
curl -X POST "http://localhost:8000/api/v1/import/yaml" \
  -H "Content-Type: application/json" \
  -d '{"cache_dir": "~/.cache/lemonade", "skip_duplicates": true}'
```

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

## Development

### Running Migrations

Create a new migration:

```bash
alembic revision -m "Description of changes"
```

Apply migrations:

```bash
alembic upgrade head
```

Rollback:

```bash
alembic downgrade -1
```

### Code Style

The codebase follows PEP 8 guidelines. Run linting:

```bash
flake8 app/
pylint app/
```

## License

Part of the lemonade-eval project. See main project for license details.
