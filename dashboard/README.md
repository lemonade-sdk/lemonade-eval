# Lemonade Eval Dashboard

A modern, full-featured dashboard for visualizing and comparing LLM/VLM evaluation results from the Lemonade SDK.

![Dashboard Preview](./docs/dashboard-preview.png)

## Overview

The Lemonade Eval Dashboard provides a centralized platform for:
- **Storing** evaluation results from Lemonade SDK runs
- **Visualizing** performance and accuracy metrics
- **Comparing** different model configurations
- **Tracking** evaluation history and trends

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **PostgreSQL** - Primary database
- **JWT** - Authentication with secure token management
- **Pydantic** - Data validation

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

## Documentation

- [Setup Guide](./SETUP.md) - Detailed setup instructions
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Environment Variables](./.env.example) - Configuration reference

## Features

### Authentication
- JWT-based authentication
- Secure token storage in sessionStorage
- Automatic token refresh
- Role-based access control (admin, editor, viewer)

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

### Error Handling
- User-friendly error messages
- Error codes for support
- Copy-to-clipboard for error details

## Project Structure

```
dashboard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── auth.py      # Authentication endpoints
│   │   │   │   ├── models.py    # Model CRUD endpoints
│   │   │   │   ├── runs.py      # Run management endpoints
│   │   │   │   ├── metrics.py   # Metrics endpoints
│   │   │   │   └── health.py    # Health check endpoint
│   │   │   └── deps.py          # Auth dependencies
│   │   ├── models/              # SQLAlchemy models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── main.py              # FastAPI app entry
│   │   └── config.py            # Configuration
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── auth.ts          # Auth API methods
│   │   │   ├── client.ts        # Axios configuration
│   │   │   └── index.ts         # API exports
│   │   ├── components/          # React components
│   │   ├── pages/               # Page components
│   │   ├── stores/              # Zustand stores
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Utility functions
│   ├── tests/
│   └── package.json
├── .env.example                  # Environment template
├── SETUP.md                      # Setup guide
└── README.md                     # This file
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

### Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/metrics` | List metrics |
| POST | `/api/v1/metrics` | Create metric |
| GET | `/api/v1/metrics/{run_id}` | Get run metrics |
| GET | `/api/v1/metrics/trend` | Get metric trends |

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
