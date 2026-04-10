# Lemonade Eval Dashboard

A modern, full-featured dashboard for visualizing and comparing LLM/VLM evaluation results from the Lemonade SDK.

![Dashboard Preview](./docs/dashboard-preview.png)

## Overview

The Lemonade Eval Dashboard provides a centralized platform for:
- **Storing** evaluation results from Lemonade SDK runs
- **Visualizing** performance and accuracy metrics
- **Comparing** different model configurations
- **Tracking** evaluation history and trends
- **Automating** evaluation pipelines with scheduled runs
- **Monitoring** real-time evaluation progress via WebSocket

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **PostgreSQL** - Primary database
- **Redis** - Caching and rate limiting
- **JWT** - Authentication with secure token management
- **Pydantic** - Data validation
- **Prometheus** - Metrics and monitoring
- **Celery** - Background task processing

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Mantine UI** - Component library
- **Zustand** - State management
- **React Query** - Data fetching
- **React Hook Form** - Form handling
- **Axios** - HTTP client
- **Vite** - Build tool

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- PostgreSQL 14+
- Redis 7+ (for caching and rate limiting)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd lemonade-eval/dashboard
```

### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp ../.env.example .env

# Start server
uvicorn app.main:app --reload --port 8000
```

### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp ../.env.example .env

# Start dev server
npm run dev
```

### 4. Access the Dashboard

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

## Documentation

- [Setup Guide](./SETUP.md) - Detailed setup instructions
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Environment Variables](./.env.example) - Configuration reference

## CLI Integration

The dashboard integrates directly with the `lemonade-eval` CLI for seamless evaluation result upload.

### Direct Upload from CLI

```bash
# Run evaluation and upload to dashboard
lemonade-eval --input meta-llama/Llama-3.2-1B-Instruct \
    --dashboard-url http://localhost:8000 \
    --dashboard-api-key your-api-key \
    ServerBench
```

### Using the Integration Script

```bash
# Upload existing YAML results
python scripts/lemonade_dashboard_integration.py \
    --dashboard-url http://localhost:8000 \
    --api-key your-api-key \
    --yaml-path ~/.cache/lemonade/builds/my-run/lemonade_stats.yaml
```

### API Upload

```bash
# Upload evaluation via API
curl -X POST http://localhost:8000/api/v1/import/evaluation \
    -H "Authorization: Bearer your-api-key" \
    -H "Content-Type: application/json" \
    -d '{
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "run_type": "benchmark",
        "build_name": "my-evaluation-run",
        "metrics": [
            {"name": "seconds_to_first_token", "value": 0.025, "unit": "seconds"}
        ],
        "status": "completed"
    }'
```

## Production Deployment

### Docker Compose (Production)

```bash
cd docker/production

# Set environment variables
export DB_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(openssl rand -base64 32)

# Start all services
docker-compose up -d

# Access services
# Dashboard: http://localhost
# Grafana: http://localhost:3001
# Prometheus: http://localhost:9090
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `SECRET_KEY` | JWT secret key (32+ chars) | - |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:3000` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |
| `RATE_LIMIT_DEFAULT` | Default rate limit (req/min) | `100` |

### Production Checklist

- [ ] Set strong `SECRET_KEY` (minimum 32 characters)
- [ ] Configure `DATABASE_URL` for PostgreSQL
- [ ] Configure `REDIS_URL` for Redis
- [ ] Set up SSL certificates for HTTPS
- [ ] Configure `CORS_ORIGINS` for production domains
- [ ] Enable rate limiting
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure backup strategy

## Monitoring Setup

### Prometheus Metrics

The dashboard exposes Prometheus metrics at `/metrics`:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency histogram
- `db_connections_active` - Active database connections
- `websocket_connections_total` - Active WebSocket connections
- `import_jobs_total` - Import job count
- `evaluation_runs_total` - Evaluation run count

### Grafana Dashboards

Import the provided Grafana configuration for pre-built dashboards:

1. Access Grafana at http://localhost:3001
2. Login with admin/admin (change password!)
3. Go to Configuration -> Data Sources
4. Add Prometheus data source (URL: http://prometheus:9090)
5. Import dashboards from `grafana/provisioning/dashboards/`

### Alerts

Configure alerts in Prometheus for:
- High error rates (> 1%)
- High response latency (p95 > 500ms)
- Database connection pool exhaustion
- Redis connection failures

## Automation Pipeline

### Scheduled Evaluations

Run evaluations on a schedule using Celery beat:

```bash
# Start Celery worker
celery -A app.services.scheduler worker --loglevel=info

# Start Celery beat
celery -A app.services.scheduler beat --loglevel=info
```

### Trend Analysis

Analyze evaluation trends and detect anomalies:

```bash
python scripts/check_trends.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct \
    --metric token_generation_tokens_per_second \
    --days 30 \
    --dashboard-url http://localhost:8000 \
    --api-key your-api-key
```

### Notifications

Send evaluation completion notifications:

```bash
python scripts/send_notifications.py \
    --event evaluation_complete \
    --recipient user@example.com \
    --data '{"run_id": "123", "model_name": "Llama-3.2-1B", "status": "completed"}' \
    --smtp-host smtp.example.com \
    --slack-webhook https://hooks.slack.com/xxx
```

## Features

### Authentication
- JWT-based authentication
- Secure token storage in sessionStorage
- Automatic token refresh
- Role-based access control (admin, editor, viewer)
- API key support for CLI integration

### Model Management
- Create and manage LLM/VLM models
- Track model versions and configurations
- Organize by family and architecture

### Evaluation Runs
- Start and monitor evaluation runs
- Real-time status updates via WebSocket
- Run history and comparison

### Metrics Visualization
- Performance metrics (latency, throughput)
- Accuracy metrics (MMLU, HumanEval)
- Interactive charts and tables
- Export capabilities

### CLI Integration
- Direct upload from `lemonade-eval` CLI
- Bulk import from YAML files
- Real-time progress reporting
- Offline queue for failed uploads

### Production Features
- Rate limiting (Redis-based)
- Request caching (Redis)
- Prometheus metrics
- Health check endpoints
- Background job processing (Celery)

## Project Structure

```
dashboard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── auth.py          # Authentication endpoints
│   │   │   │   ├── models.py        # Model CRUD endpoints
│   │   │   │   ├── runs.py          # Run management endpoints
│   │   │   │   ├── metrics.py       # Metrics endpoints
│   │   │   │   ├── health.py        # Health check endpoint
│   │   │   │   └── cli_integration.py  # CLI integration endpoints
│   │   │   └── deps.py              # Auth dependencies
│   │   ├── middleware/
│   │   │   └── rate_limiter.py      # Rate limiting middleware
│   │   ├── monitoring/
│   │   │   └── metrics.py           # Prometheus metrics
│   │   ├── cache/
│   │   │   ├── cache_manager.py     # Redis cache manager
│   │   │   └── cache_service.py     # Cache service
│   │   ├── integration/
│   │   │   ├── cli_client.py        # CLI client
│   │   │   └── import_pipeline.py   # Import pipeline
│   │   ├── models/                  # SQLAlchemy models
│   │   ├── schemas/                 # Pydantic schemas
│   │   ├── services/                # Business logic
│   │   ├── main.py                  # FastAPI app entry
│   │   └── config.py                # Configuration
│   ├── tests/
│   │   ├── test_api.py              # API tests
│   │   └── test_cli_integration.py  # CLI integration tests
│   └── requirements.txt
├── docker/production/
│   ├── docker-compose.yml           # Production deployment
│   ├── nginx.conf                   # Nginx configuration
│   └── prometheus.yml               # Prometheus config
├── scripts/
│   ├── lemonade_dashboard_integration.py  # CLI integration script
│   ├── run_scheduled_eval.py        # Scheduled evaluations
│   ├── check_trends.py              # Trend analysis
│   └── send_notifications.py        # Notifications
├── frontend/
│   └── ...                          # React frontend
├── .env.example                      # Environment template
├── SETUP.md                          # Setup guide
└── README.md                         # This file
```

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | User login |
| POST | `/api/v1/auth/logout` | User logout |
| POST | `/api/v1/auth/refresh` | Refresh token |
| GET | `/api/v1/auth/me` | Get current user |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models` | List models |
| POST | `/api/v1/models` | Create model |
| GET | `/api/v1/models/{id}` | Get model |
| PUT | `/api/v1/models/{id}` | Update model |
| DELETE | `/api/v1/models/{id}` | Delete model |

### Runs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/runs` | List runs |
| POST | `/api/v1/runs` | Create run |
| GET | `/api/v1/runs/{id}` | Get run details |
| PUT | `/api/v1/runs/{id}` | Update run |
| GET | `/api/v1/runs/stats` | Get run statistics |

### CLI Integration
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/import/evaluation` | Import evaluation from CLI |
| POST | `/api/v1/import/bulk` | Bulk import evaluations |
| POST | `/api/v1/import/yaml` | Import YAML data |
| GET | `/api/v1/import/status/{run_id}` | Get import status |
| WebSocket | `/ws/v1/evaluation-progress` | Real-time progress |

### Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/metrics` | List metrics |
| POST | `/api/v1/metrics` | Create metric |
| GET | `/api/v1/metrics/{run_id}` | Get run metrics |
| GET | `/api/v1/metrics/trend` | Get metric trends |

### Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | Prometheus metrics |
| GET | `/health/live` | Liveness check |
| GET | `/health/ready` | Readiness check |

## Development

### Running Tests

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm run test
```

### Code Style

```bash
# Backend (Python)
black app/
isort app/

# Frontend (TypeScript)
npm run lint
npm run format
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check the [Setup Guide](./SETUP.md)
- Review API documentation at `/docs`
- Open an issue on the repository

---

Built with React, FastAPI, and Lemonade SDK
