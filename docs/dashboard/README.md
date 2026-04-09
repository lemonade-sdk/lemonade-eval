# Lemonade Eval Dashboard - Documentation Index

Welcome to the Lemonade Eval Dashboard documentation suite. This documentation covers all aspects of the dashboard for storing, visualizing, and comparing LLM/VLM evaluation results.

## Documentation Overview

| Document | Purpose | Audience |
|----------|---------|----------|
| [User Guide](./USER_GUIDE.md) | How to use the dashboard | End users, analysts |
| [API Documentation](./API_DOCUMENTATION.md) | API reference and examples | Developers, integrators |
| [Developer Guide](./DEVELOPER_GUIDE.md) | Development and contribution | Developers, contributors |
| [Operations Guide](./OPERATIONS_GUIDE.md) | Deployment and maintenance | DevOps, SREs |

## Quick Links

### Getting Started
- [User Guide - Getting Started](./USER_GUIDE.md#getting-started)
- [Backend Setup](./DEVELOPER_GUIDE.md#backend-setup)
- [Frontend Setup](./DEVELOPER_GUIDE.md#frontend-setup)

### API Reference
- [Authentication](./API_DOCUMENTATION.md#authentication)
- [Models API](./API_DOCUMENTATION.md#models-api)
- [Runs API](./API_DOCUMENTATION.md#runs-api)
- [Metrics API](./API_DOCUMENTATION.md#metrics-api)
- [WebSocket API](./API_DOCUMENTATION.md#websocket-api)

### Deployment
- [Production Checklist](./OPERATIONS_GUIDE.md#production-deployment-checklist)
- [Docker Deployment](./OPERATIONS_GUIDE.md#docker-development)
- [Monitoring Setup](./OPERATIONS_GUIDE.md#monitoring-setup)

---

## System Overview

### What is the Lemonade Eval Dashboard?

The Lemonade Eval Dashboard is a web-based platform for:
- Storing LLM/VLM evaluation results in a SQLite (development) / PostgreSQL (production) database
- Visualizing performance and accuracy metrics
- Comparing models and runs side-by-side
- Importing existing lemonade-eval YAML data
- Real-time updates during evaluation runs

### Architecture

```
┌──────────────────┐    ┌──────────────────┐
│   Frontend       │    │   Backend        │
│   React + TS     │◄──►│   FastAPI        │
│   Mantine        │    │   SQLAlchemy     │
└──────────────────┘    └──────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   SQLite /       │
                        │   PostgreSQL     │
                        │   + optional     │
                        │   Redis          │
                        └──────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Models Management** | CRUD operations for LLM/VLM models |
| **Runs Tracking** | Track evaluation runs with status and configuration |
| **Metrics Storage** | Store performance and accuracy metrics |
| **Comparison** | Side-by-side run comparison with charts |
| **Import Pipeline** | Import from lemonade-eval cache |
| **Real-time Updates** | WebSocket-based live updates |
| **CLI Integration** | Automated submission from lemonade-eval CLI |
| **Authentication** | JWT-based auth with API key support |
| **Rate Limiting** | Redis-based rate limiting |
| **Benchmarks Page** | Sweep benchmark visualization with charts |
| **Accuracy Page** | Accuracy metric comparison across models |

---

## Document Summaries

### User Guide

The User Guide provides comprehensive instructions for using the dashboard:

- **Getting Started**: Installation and first login
- **Feature Walkthrough**: Dashboard, Models, Runs, Compare, Import pages
- **CLI Integration**: How to integrate with lemonade-eval CLI
- **Troubleshooting**: Common issues and solutions

**Key Sections:**
- [Getting Started Tutorial](./USER_GUIDE.md#getting-started)
- [Dashboard Overview](./USER_GUIDE.md#dashboard-overview)
- [Compare Page](./USER_GUIDE.md#compare-page)
- [Import from Cache](./USER_GUIDE.md#import-page)

### API Documentation

The API Documentation provides complete API reference:

- **Authentication**: JWT tokens and API keys
- **Rate Limiting**: Limits and handling
- **Error Codes**: All error codes and meanings
- **Endpoints**: Complete endpoint reference with examples

**Key Sections:**
- [Authentication Guide](./API_DOCUMENTATION.md#authentication)
- [Models API](./API_DOCUMENTATION.md#models-api)
- [Runs API](./API_DOCUMENTATION.md#runs-api)
- [WebSocket API](./API_DOCUMENTATION.md#websocket-api)

### Developer Guide

The Developer Guide covers development and contribution:

- **Architecture Overview**: System design and data flow
- **Local Development**: Setup for development
- **Testing Guide**: How to run and write tests
- **Contributing Guidelines**: Code style and PR process

**Key Sections:**
- [System Architecture](./DEVELOPER_GUIDE.md#architecture-overview)
- [Local Development Setup](./DEVELOPER_GUIDE.md#local-development-setup)
- [Testing Guide](./DEVELOPER_GUIDE.md#testing-guide)
- [Contributing Guidelines](./DEVELOPER_GUIDE.md#contributing-guidelines)

### Operations Guide

The Operations Guide covers production deployment:

- **Deployment Checklist**: Pre and post-deployment tasks
- **Monitoring Setup**: Prometheus, Grafana, alerting
- **Backup/Recovery**: Backup procedures and recovery steps
- **Scaling Guidelines**: Vertical and horizontal scaling

**Key Sections:**
- [Production Deployment](./OPERATIONS_GUIDE.md#production-deployment-checklist)
- [Monitoring Setup](./OPERATIONS_GUIDE.md#monitoring-setup)
- [Backup Procedures](./OPERATIONS_GUIDE.md#backup-and-recovery-procedures)
- [Scaling Guidelines](./OPERATIONS_GUIDE.md#scaling-guidelines)

---

## Common Tasks

### For End Users

| Task | Document | Section |
|------|----------|---------|
| First login | User Guide | [Getting Started](./USER_GUIDE.md#getting-started) |
| Import data | User Guide | [Import Page](./USER_GUIDE.md#import-page) |
| Compare runs | User Guide | [Compare Page](./USER_GUIDE.md#compare-page) |
| CLI integration | User Guide | [CLI Integration](./USER_GUIDE.md#cli-integration-guide) |

### For Developers

| Task | Document | Section |
|------|----------|---------|
| Setup dev environment | Developer Guide | [Local Development](./DEVELOPER_GUIDE.md#local-development-setup) |
| Run tests | Developer Guide | [Testing Guide](./DEVELOPER_GUIDE.md#testing-guide) |
| Add new endpoint | Developer Guide | [Backend Architecture](./DEVELOPER_GUIDE.md#backend-architecture) |
| Submit PR | Developer Guide | [Contributing Guidelines](./DEVELOPER_GUIDE.md#contributing-guidelines) |

### For DevOps

| Task | Document | Section |
|------|----------|---------|
| Deploy to production | Operations Guide | [Deployment Checklist](./OPERATIONS_GUIDE.md#production-deployment-checklist) |
| Configure monitoring | Operations Guide | [Monitoring Setup](./OPERATIONS_GUIDE.md#monitoring-setup) |
| Set up backups | Operations Guide | [Backup Procedures](./OPERATIONS_GUIDE.md#backup-and-recovery-procedures) |
| Scale deployment | Operations Guide | [Scaling Guidelines](./OPERATIONS_GUIDE.md#scaling-guidelines) |

---

## API Quick Reference

### Base URLs

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:3001` |
| Production | `https://your-domain.com` |

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/auth/login` | User login |
| GET | `/api/v1/models` | List models |
| POST | `/api/v1/models` | Create model |
| GET | `/api/v1/runs` | List runs |
| POST | `/api/v1/runs` | Create run |
| GET | `/api/v1/metrics` | List metrics |
| POST | `/api/v1/import/evaluation` | Import evaluation |
| WS | `/ws/v1/evaluations` | WebSocket updates |

### Authentication

```bash
# JWT Token
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# API Key
X-API-Key: ledash_your-api-key-here
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Model** | An LLM or VLM being evaluated |
| **Run** | A single evaluation execution |
| **Metric** | A measured value (performance or accuracy) |
| **Build** | A unique identifier for evaluation results |
| **TTFT** | Time to First Token (latency metric) |
| **TPS** | Tokens Per Second (throughput metric) |
| **MMLU** | Massively Multitask Language Understanding benchmark |
| **HumanEval** | Code generation accuracy benchmark |

---

## Support

### Getting Help

- **Documentation**: This documentation suite
- **API Docs**: `http://localhost:3001/docs` (Swagger UI)
- **GitHub Issues**: https://github.com/lemonade/lemonade-eval/issues

### Reporting Issues

When reporting issues, include:
1. Description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, browser, versions)
5. Relevant logs or screenshots

---

## Version Information

| Component | Version |
|-----------|---------|
| Dashboard | 1.0.0 |
| API | v1 |
| Database Schema | v1 (initial) |

## Changelog

See [CHANGELOG.md](../../CHANGELOG.md) for version history.

---

## License

Part of the lemonade-eval project. See main project for license details.
