# UI-UX Eval Dashboard Database - Implementation Plan

**Document Version:** 1.0
**Date:** 2026-03-29
**Author:** Dr. Sarah Kim, Technical Product Strategist & Engineering Lead

---

## Executive Summary

This document provides a comprehensive implementation plan for building a **UI-UX Eval Dashboard Database** for the lemonade-eval CLI tool. The dashboard will enable users to store, visualize, compare, and analyze LLM/VLM model evaluation results through a modern web interface, replacing the current file-based YAML/JSON storage system.

### Current State
- lemonade-eval (v9.1.4) is a Python-based CLI for evaluating LLM/VLM models
- Evaluation results stored as YAML files in `~/.cache/lemonade/`
- No existing database or web UI
- Benchmarking tools: `bench` (TTFT, TPS), MMLU, HumanEval, lm-eval-harness, perplexity

### Proposed Solution
A full-stack web application with:
- PostgreSQL database for structured evaluation data storage
- FastAPI backend REST API for data management and real-time updates
- React/TypeScript frontend dashboard for visualization and comparison
- WebSocket support for real-time evaluation progress tracking
- Migration tools for importing existing YAML data

---

## 1. Recommended Tech Stack

### Backend

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Framework** | FastAPI (Python 3.12+) | High performance, async support, automatic OpenAPI docs, Python ecosystem alignment with lemonade-eval |
| **Database** | PostgreSQL 16+ | ACID compliance, excellent JSON support for flexible metrics, time-series capabilities, robust ORM support |
| **ORM** | SQLAlchemy 2.0 + Alembic | Type-safe queries, migration management, async support |
| **Real-time** | WebSockets (FastAPI native) | Native async support, no additional infrastructure needed |
| **Task Queue** | Celery + Redis | Background tasks for YAML migration, report generation, scheduled analyses |
| **Authentication** | JWT + OAuth2 | Industry standard, stateless, supports SSO integration |
| **Caching** | Redis | Session management, query result caching, real-time pub/sub |

### Frontend

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Framework** | React 18+ with TypeScript | Type safety, large ecosystem, team familiarity |
| **State Management** | Zustand | Lightweight, TypeScript-first, simpler than Redux |
| **UI Components** | Mantine v7 | Modern components, excellent TypeScript support, theming, data tables |
| **Charts** | Recharts + Visx | React-native, customizable, good for time-series and comparisons |
| **Data Tables** | TanStack Table v8 | Virtualization, sorting, filtering, column management |
| **Real-time** | WebSocket hook (custom) | Direct FastAPI WebSocket integration |
| **Forms** | React Hook Form + Zod | Type-safe validation, performant |
| **Build Tool** | Vite | Fast HMR, optimized builds |

### Infrastructure

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Containerization** | Docker + Docker Compose | Consistent environments, easy deployment |
| **API Documentation** | OpenAPI/Swagger (FastAPI auto-generated) | Interactive docs, client generation |
| **Testing** | pytest (backend), Vitest + React Testing Library (frontend) | Python/JS ecosystem standards |
| **CI/CD** | GitHub Actions | Native GitHub integration, existing workflow familiarity |

---

## 2. Database Schema Design

### Entity Relationship Overview

```
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│   Users     │       │    Models    │       │  Runs       │
├─────────────┤       ├──────────────┤       ├─────────────┤
│ id          │       │ id           │       │ id          │
│ email       │◄──────│ created_by   │◄──────│ model_id    │
│ name        │       │ name         │       │ user_id     │
│ role        │       │ checkpoint   │       │ status      │
│ api_keys    │       │ model_type   │       │ started_at  │
│ created_at  │       │ parameters   │       │ completed_at│
└─────────────┘       │ metadata     │       │ config      │
        │             └──────────────┘       └─────────────┘
        │                        │                   │
        │                        │                   │
        ▼                        ▼                   ▼
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│  UserRoles  │       │ModelVersions │       │  Metrics    │
├─────────────┤       ├──────────────┤       ├─────────────┤
│ id          │       │ id           │       │ id          │
│ user_id     │       │ model_id     │       │ run_id      │
│ role        │       │ version      │       │ metric_type │
│ scope       │       │ quantization │       │ name        │
└─────────────┘       │ dtype        │       │ value       │
                      │ backend      │       │ unit        │
                      └──────────────┘       │ metadata    │
                                             └─────────────┘
```

### Table Definitions

#### 2.1 Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer', -- admin, editor, viewer
    hashed_password VARCHAR(255),
    api_key_hash VARCHAR(255),
    api_key_prefix VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_api_key_prefix ON users(api_key_prefix);
```

#### 2.2 Models Table
```sql
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    checkpoint VARCHAR(500) NOT NULL, -- e.g., "Llama-3.2-1B-Instruct-GGUF"
    model_type VARCHAR(50) DEFAULT 'llm', -- llm, vlm, embedding
    family VARCHAR(100), -- e.g., "Llama", "Qwen", "Phi"
    parameters BIGINT, -- parameter count
    max_context_length INTEGER,
    architecture VARCHAR(100), -- e.g., "transformer"
    license_type VARCHAR(100),
    hf_repo VARCHAR(255), -- HuggingFace repository
    metadata JSONB DEFAULT '{}', -- flexible additional model info
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(checkpoint)
);

CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_family ON models(family);
CREATE INDEX idx_models_checkpoint ON models(checkpoint);
```

#### 2.3 Runs Table (Evaluation Runs)
```sql
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),

    -- Run identification
    build_name VARCHAR(255) NOT NULL, -- e.g., "Llama-3.2-1B-Instruct-GGUF_2026y_03m_29d_14h_30m_00s"
    run_type VARCHAR(50) NOT NULL, -- benchmark, accuracy-mmlu, accuracy-humaneval, lm-eval, perplexity

    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed, cancelled
    status_message TEXT,

    -- Configuration
    device VARCHAR(50), -- cpu, igpu, npu, hybrid, gpu
    backend VARCHAR(100), -- llamacpp, ort, flm
    dtype VARCHAR(50), -- float32, float16, int4, int8
    config JSONB DEFAULT '{}', -- store all run parameters

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTERVAL,

    -- System info snapshot
    system_info JSONB DEFAULT '{}', -- OS, CPU, RAM, driver versions

    --柠檬ade version tracking
    lemonade_version VARCHAR(20),
    build_uid VARCHAR(100),

    -- File references
    log_file_path VARCHAR(500),
    error_log TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_runs_model ON runs(model_id);
CREATE INDEX idx_runs_user ON runs(user_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_type ON runs(run_type);
CREATE INDEX idx_runs_created ON runs(created_at);
CREATE INDEX idx_runs_device_dtype ON runs(device, dtype);
```

#### 2.4 Metrics Table
```sql
CREATE TABLE metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,

    -- Metric identification
    category VARCHAR(50) NOT NULL, -- performance, accuracy, efficiency
    name VARCHAR(255) NOT NULL, -- e.g., "seconds_to_first_token", "mmlu_management_accuracy"
    display_name VARCHAR(255),

    -- Value storage (flexible for different metric types)
    value_numeric DECIMAL(20, 6), -- for numeric metrics
    value_text TEXT, -- for categorical/text results
    unit VARCHAR(50), -- tokens/s, %, ms, GB

    -- Statistical data
    mean_value DECIMAL(20, 6),
    std_dev DECIMAL(20, 6),
    min_value DECIMAL(20, 6),
    max_value DECIMAL(20, 6),

    -- Per-iteration raw data (JSON array for detailed analysis)
    iteration_values JSONB, -- e.g., [0.025, 0.023, 0.027]

    -- Metadata
    metadata JSONB DEFAULT '{}', -- prompt_tokens, response_tokens, etc.

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_metrics_run ON metrics(run_id);
CREATE INDEX idx_metrics_category ON metrics(category);
CREATE INDEX idx_metrics_name ON metrics(name);
CREATE INDEX idx_metrics_value ON metrics(value_numeric);
CREATE UNIQUE INDEX idx_unique_metric ON metrics(run_id, category, name);
```

#### 2.5 ModelVersions Table (for tracking model variants)
```sql
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,

    version VARCHAR(50) NOT NULL, -- v1, v2, or quantization label
    quantization VARCHAR(50), -- int4, int8, fp16, awq
    dtype VARCHAR(50),
    backend VARCHAR(100),

    -- Version-specific config
    config JSONB DEFAULT '{}',

    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(model_id, version, quantization, dtype, backend)
);

CREATE INDEX idx_model_versions_model ON model_versions(model_id);
```

#### 2.6 Tags Table (for flexible organization)
```sql
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    color VARCHAR(7) DEFAULT '#6B7280',
    created_by UUID REFERENCES users(id)
);

CREATE TABLE run_tags (
    run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (run_id, tag_id)
);
```

### Metrics Categories and Standard Names

#### Performance Metrics (category: 'performance')
| Metric Name | Unit | Description |
|-------------|------|-------------|
| `seconds_to_first_token` | seconds | Time to first token (TTFT) |
| `std_dev_seconds_to_first_token` | seconds | TTFT standard deviation |
| `prefill_tokens_per_second` | tokens/s | Prefill throughput |
| `token_generation_tokens_per_second` | tokens/s | Token generation throughput |
| `std_dev_tokens_per_second` | tokens/s | Generation TPS std dev |
| `max_memory_used_gbyte` | GB | Peak memory usage |
| `prompt_tokens` | tokens | Input prompt length |
| `response_tokens` | tokens | Generated tokens |

#### Accuracy Metrics (category: 'accuracy')
| Metric Name | Unit | Description |
|-------------|------|-------------|
| `mmlu_{subject}_accuracy` | % | MMLU subject accuracy |
| `average_mmlu_accuracy` | % | Average MMLU across subjects |
| `humaneval_pass@1` | % | HumanEval pass@1 |
| `humaneval_pass@10` | % | HumanEval pass@10 |
| `humaneval_pass@100` | % | HumanEval pass@100 |
| `lm_eval_{task}_{metric}` | % | LM-eval-harness metrics |
| `perplexity` | raw | Perplexity score |

---

## 3. Architecture Diagram Description

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SYSTEMS                                │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  lemonade-eval  │   Lemonade      │   HuggingFace   │      Users            │
│  CLI Tool       │   Server        │   API           │   (Browsers)          │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         │ YAML/JSON       │ OpenAI API      │ Model Metadata     │ HTTPS/WSS
         │ Import          │ Compat API      │ Sync               │
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY LAYER                                  │
│                        (FastAPI + CORS + Auth)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  /api/v1/auth/*          /api/v1/models/*      /api/v1/runs/*              │
│  /api/v1/metrics/*       /api/v1/reports/*     /api/v1/import/*            │
│  /ws/v1/evaluations      /api/v1/tags/*        /api/v1/users/*             │
└─────────────────────────────────────────────────────────────────────────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
├───────────────────┬───────────────────┬─────────────────┬───────────────────┤
│   Auth Service    │   Model Service   │   Run Service   │   Metrics Service │
│   - JWT handling  │   - CRUD          │   - CRUD        │   - Aggregation   │
│   - API keys      │   - Search        │   - Status      │   - Calculations  │
│   - RBAC          │   - Versioning    │   - Comparison  │   - Trends        │
├───────────────────┬───────────────────┬─────────────────┬───────────────────┤
│   Import Service  │   Report Service  │   WebSocket     │   Notification    │
│   - YAML parse    │   - PDF/CSV gen   │   - Progress    │   Service         │
│   - Validation    │   - Export        │   - Real-time   │   - Email         │
│   - Migration     │   - Scheduling    │   - Events      │   - Webhook       │
└───────────────────┴───────────────────┴─────────────────┴───────────────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├───────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   PostgreSQL      │   Redis         │   File Storage  │   Message Queue     │
│   - Primary DB    │   - Cache       │   - Logs        │   - Celery          │
│   - Full-text     │   - Sessions    │   - Exports     │   - Tasks           │
│   - Time-series   │   - Pub/Sub     │   - Backups     │   - Scheduling      │
└───────────────────┴─────────────────┴─────────────────┴─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER                                      │
│                        (React + TypeScript)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Dashboard    Model Library    Run Manager    Comparisons    Reports        │
│  - Overview   - List/Grid      - Queue        - Side-by-side - Export       │
│  - Charts     - Search         - Details      - Trend lines  - Schedule     │
│  - Alerts     - Upload         - Logs         - Statistics   - Templates    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration Points with Existing lemonade-eval

1. **Direct Database Write Mode** (New)
   - Add `--dashboard-url` and `--api-key` flags to lemonade-eval CLI
   - Results written directly to dashboard API after evaluation
   - Real-time progress via WebSocket

2. **YAML Import Mode** (Migration)
   - `lemonade-eval import-to-dashboard --cache-dir ~/.cache/lemonade --dashboard-url ...`
   - Batch import existing YAML files
   - Deduplication based on build_name + checksum

3. **Hybrid Mode** (Backward Compatible)
   - Continue writing YAML files locally
   - Dashboard periodically scans/syncs cache directory
   - User-configurable sync schedule

---

## 4. Phased Implementation Plan

### Phase 1: MVP (Weeks 1-4)
**Goal:** Core database + basic API + minimal UI for viewing results

#### Week 1: Foundation
- [ ] Set up project structure (monorepo with pnpm workspaces)
- [ ] Docker Compose configuration (PostgreSQL, Redis, FastAPI, React)
- [ ] Database schema implementation + Alembic migrations
- [ ] Basic FastAPI setup with health checks

#### Week 2: Backend Core
- [ ] SQLAlchemy models and repositories
- [ ] Authentication (JWT + API keys)
- [ ] Models API (CRUD)
- [ ] Runs API (CRUD)
- [ ] Metrics API (list, aggregate)

#### Week 3: Frontend Core
- [ ] React app setup with Vite + TypeScript
- [ ] Mantine UI integration with theming
- [ ] Authentication flow (login, API key management)
- [ ] Basic routing and layout

#### Week 4: MVP Features
- [ ] Model list view with search
- [ ] Run list view with filtering
- [ ] Run detail view with metrics display
- [ ] YAML import script (CLI tool)
- [ ] Basic dashboard with summary statistics

**MVP Deliverables:**
- Working database with schema
- REST API with authentication
- Web UI to view imported evaluation results
- YAML import capability

---

### Phase 2: Core Features (Weeks 5-8)
**Goal:** Comparison, visualization, and real-time updates

#### Week 5: Data Visualization
- [ ] Metrics chart components (line, bar, scatter)
- [ ] Performance comparison charts
- [ ] Accuracy radar charts
- [ ] Time-series trend views

#### Week 6: Comparison Features
- [ ] Side-by-side run comparison UI
- [ ] Multi-select for runs/models
- [ ] Statistical comparison (mean, std dev, significance)
- [ ] Export comparison as PDF/image

#### Week 7: Real-time Updates
- [ ] WebSocket endpoint for evaluation progress
- [ ] Frontend WebSocket hook
- [ ] Real-time run status updates
- [ ] Live metrics streaming during evaluation

#### Week 8: Integration with lemonade-eval
- [ ] Python SDK/client library
- [ ] CLI integration (`--dashboard-url` flag)
- [ ] Real-time progress from CLI to dashboard
- [ ] Error handling and retry logic

**Phase 2 Deliverables:**
- Rich visualization of evaluation results
- Side-by-side model/run comparison
- Real-time evaluation progress tracking
- Direct CLI-to-dashboard integration

---

### Phase 3: Advanced Features (Weeks 9-12)
**Goal:** Reporting, automation, and team collaboration

#### Week 9: Reporting
- [ ] Report template system
- [ ] PDF report generation
- [ ] Scheduled report emails
- [ ] Custom report builder UI

#### Week 10: Search & Filtering
- [ ] Advanced search (full-text on model names, tags)
- [ ] Saved filters/views
- [ ] Tag management
- [ ] Bulk operations

#### Week 11: Team Features
- [ ] User roles (admin, editor, viewer)
- [ ] Team/workspaces
- [ ] Shared dashboards
- [ ] Activity audit log

#### Week 12: Automation
- [ ] Scheduled evaluation runs
- [ ] CI/CD integration hooks
- [ ] Webhook notifications
- [ ] API rate limiting and quotas

**Phase 3 Deliverables:**
- Automated reporting system
- Team collaboration features
- Advanced search and organization
- CI/CD integration capabilities

---

### Phase 4: Polish & Scale (Weeks 13-16)
**Goal:** Performance optimization, testing, documentation

#### Week 13: Performance
- [ ] Query optimization and indexing
- [ ] Frontend code splitting
- [ ] Caching strategy implementation
- [ ] Database connection pooling

#### Week 14: Testing
- [ ] Backend unit/integration tests (pytest)
- [ ] Frontend unit tests (Vitest)
- [ ] E2E tests (Playwright)
- [ ] Load testing (Locust)

#### Week 15: Documentation
- [ ] API documentation (OpenAPI)
- [ ] User guide
- [ ] Developer setup guide
- [ ] Deployment guide

#### Week 16: Launch Prep
- [ ] Security audit
- [ ] Performance benchmarking
- [ ] Bug fixes
- [ ] Release candidate testing

**Phase 4 Deliverables:**
- Production-ready application
- Comprehensive test coverage
- Complete documentation
- Deployment playbooks

---

## 5. Key API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | User login (email/password) |
| POST | `/api/v1/auth/logout` | User logout |
| POST | `/api/v1/auth/refresh` | Refresh JWT token |
| POST | `/api/v1/auth/api-key` | Generate new API key |
| DELETE | `/api/v1/auth/api-key/{id}` | Revoke API key |
| GET | `/api/v1/auth/me` | Get current user |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models` | List models (paginated, filtered) |
| GET | `/api/v1/models/{id}` | Get model details |
| POST | `/api/v1/models` | Create model |
| PUT | `/api/v1/models/{id}` | Update model |
| DELETE | `/api/v1/models/{id}` | Delete model |
| GET | `/api/v1/models/{id}/versions` | List model versions |
| GET | `/api/v1/models/{id}/runs` | Get runs for model |
| GET | `/api/v1/models/{id}/metrics` | Aggregate metrics for model |

### Runs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/runs` | List runs (paginated, filtered) |
| GET | `/api/v1/runs/{id}` | Get run details |
| POST | `/api/v1/runs` | Create run |
| PUT | `/api/v1/runs/{id}` | Update run |
| DELETE | `/api/v1/runs/{id}` | Delete run |
| GET | `/api/v1/runs/{id}/metrics` | Get run metrics |
| POST | `/api/v1/runs/{id}/metrics` | Batch create metrics |
| POST | `/api/v1/runs/{id}/cancel` | Cancel running evaluation |
| POST | `/api/v1/runs/{id}/retry` | Retry failed run |
| GET | `/api/v1/runs/{id}/logs` | Get run logs |

### Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/metrics` | List metrics (filtered) |
| GET | `/api/v1/metrics/aggregate` | Aggregate metrics across runs |
| GET | `/api/v1/metrics/trends` | Time-series trends |
| GET | `/api/v1/metrics/compare` | Compare metrics across models |
| GET | `/api/v1/metrics/benchmarks` | Industry benchmark data |

### Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/reports` | List saved reports |
| POST | `/api/v1/reports` | Create report |
| GET | `/api/v1/reports/{id}` | Get report |
| DELETE | `/api/v1/reports/{id}` | Delete report |
| GET | `/api/v1/reports/{id}/export` | Export report (PDF/CSV) |
| POST | `/api/v1/reports/schedule` | Schedule report generation |

### Import
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/import/yaml` | Import YAML files |
| POST | `/api/v1/import/batch` | Batch import |
| GET | `/api/v1/import/status/{job_id}` | Import job status |
| POST | `/api/v1/import/scan` | Scan cache directory |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws/v1/evaluations` | Real-time evaluation updates |
| `/ws/v1/notifications` | User notifications |

### API Response Format

```json
{
  "success": true,
  "data": { },
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5
  },
  "errors": []
}
```

### Error Response Format

```json
{
  "success": false,
  "data": null,
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Invalid input",
      "field": "model_name",
      "details": { }
    }
  ]
}
```

---

## 6. Frontend Page/Component Structure

### Page Structure

```
src/
├── pages/
│   ├── auth/
│   │   ├── LoginPage.tsx
│   │   └── RegisterPage.tsx
│   ├── dashboard/
│   │   ├── DashboardPage.tsx          # Main overview
│   │   └── components/
│   │       ├── SummaryCards.tsx
│   │       ├── RecentRunsTable.tsx
│   │       ├── PerformanceChart.tsx
│   │       └── AccuracyOverview.tsx
│   ├── models/
│   │   ├── ModelListPage.tsx
│   │   ├── ModelDetailPage.tsx
│   │   └── components/
│   │       ├── ModelCard.tsx
│   │       ├── ModelTable.tsx
│   │       ├── ModelForm.tsx
│   │       └── VersionSelector.tsx
│   ├── runs/
│   │   ├── RunListPage.tsx
│   │   ├── RunDetailPage.tsx
│   │   ├── RunQueuePage.tsx
│   │   └── components/
│   │       ├── RunTable.tsx
│   │       ├── RunStatusBadge.tsx
│   │       ├── RunFilters.tsx
│   │       └── MetricsPanel.tsx
│   ├── compare/
│   │   ├── ComparePage.tsx
│   │   └── components/
│   │       ├── RunSelector.tsx
│   │       ├── ComparisonTable.tsx
│   │       ├── ComparisonCharts.tsx
│   │       └── StatisticalAnalysis.tsx
│   ├── reports/
│   │   ├── ReportsPage.tsx
│   │   ├── ReportBuilderPage.tsx
│   │   └── components/
│   │       ├── ReportList.tsx
│   │       ├── ReportPreview.tsx
│   │       └── ReportScheduler.tsx
│   ├── import/
│   │   ├── ImportPage.tsx
│   │   └── components/
│   │       ├── FileUploader.tsx
│   │       ├── ImportProgress.tsx
│   │       └── ImportHistory.tsx
│   └── settings/
│       ├── SettingsPage.tsx
│       └── components/
│           ├── ProfileForm.tsx
│           ├── ApiKeyManagement.tsx
│           └── NotificationSettings.tsx
├── components/
│   ├── common/
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   ├── PageHeader.tsx
│   │   ├── DataTable.tsx
│   │   ├── Pagination.tsx
│   │   ├── SearchInput.tsx
│   │   ├── FilterBar.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ErrorBoundary.tsx
│   ├── charts/
│   │   ├── LineChart.tsx
│   │   ├── BarChart.tsx
│   │   ├── ScatterPlot.tsx
│   │   ├── RadarChart.tsx
│   │   ├── Heatmap.tsx
│   │   └── TimeSeriesChart.tsx
│   ├── metrics/
│   │   ├── MetricCard.tsx
│   │   ├── MetricComparison.tsx
│   │   ├── MetricTrend.tsx
│   │   └── MetricDistribution.tsx
│   └── forms/
│       ├── TextField.tsx
│       ├── SelectField.tsx
│       ├── MultiSelectField.tsx
│       ├── DateRangePicker.tsx
│       └── FileUploadField.tsx
├── hooks/
│   ├── useAuth.ts
│   ├── useModels.ts
│   ├── useRuns.ts
│   ├── useMetrics.ts
│   ├── useWebSocket.ts
│   └── usePagination.ts
├── stores/
│   ├── authStore.ts
│   ├── uiStore.ts
│   └── notificationStore.ts
├── api/
│   ├── client.ts              # Axios/fetch wrapper
│   ├── auth.ts
│   ├── models.ts
│   ├── runs.ts
│   ├── metrics.ts
│   └── reports.ts
├── types/
│   ├── models.ts
│   ├── runs.ts
│   ├── metrics.ts
│   └── api.ts
└── utils/
    ├── formatters.ts
    ├── validators.ts
    └── constants.ts
```

### Key Component Specifications

#### DashboardPage Components

```tsx
// SummaryCards.tsx - Display key metrics at a glance
interface SummaryCardsProps {
  totalModels: number;
  totalRuns: number;
  avgTTFT: number;
  avgTPS: number;
  avgMMLU: number;
}

// RecentRunsTable.tsx - Latest evaluation runs
interface RecentRunsTableProps {
  runs: Run[];
  onViewRun: (id: string) => void;
  onCompare: (ids: string[]) => void;
}

// PerformanceChart.tsx - TTFT/TPS trends over time
interface PerformanceChartProps {
  modelId?: string;
  timeRange: TimeRange;
  metrics: ('ttft' | 'tps' | 'memory')[];
}
```

#### ComparePage Components

```tsx
// RunSelector.tsx - Multi-select runs for comparison
interface RunSelectorProps {
  selectedRunIds: string[];
  onChange: (ids: string[]) => void;
  filters?: RunFilters;
}

// ComparisonTable.tsx - Side-by-side metrics
interface ComparisonTableProps {
  runs: Run[];
  metrics: Metric[];
  highlightBest: boolean;
  showStats: boolean;
}

// ComparisonCharts.tsx - Visual comparison
interface ComparisonChartsProps {
  runs: Run[];
  chartType: 'bar' | 'radar' | 'scatter';
}
```

#### RunDetailPage Components

```tsx
// MetricsPanel.tsx - Detailed metrics display
interface MetricsPanelProps {
  runId: string;
  category?: 'all' | 'performance' | 'accuracy';
  showIterations?: boolean;
}

// RunLogs.tsx - View evaluation logs
interface RunLogsProps {
  runId: string;
  autoScroll?: boolean;
}
```

---

## 7. Data Migration Strategy

### Migration Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIGRATION WORKFLOW                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: SCAN                                                   │
│  - Enumerate all YAML files in cache directory                  │
│  - Parse file metadata (size, modified date)                    │
│  - Generate checksums for deduplication                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: PARSE                                                  │
│  - Load each YAML file                                          │
│  - Validate schema compatibility                                │
│  - Extract: model info, run config, metrics                     │
│  - Handle errors gracefully (log and continue)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: DEDUPLICATE                                            │
│  - Check existing runs by build_name + checksum                 │
│  - Skip duplicates or mark for update based on user preference  │
│  - Handle conflicts (newer timestamp wins, or manual review)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: TRANSFORM                                              │
│  - Map YAML structure to database schema                        │
│  - Normalize metric names                                       │
│  - Convert data types (timestamps, numbers)                     │
│  - Enrich with derived data (model family, etc.)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: LOAD                                                   │
│  - Batch insert models (upsert by checkpoint)                   │
│  - Batch insert runs                                            │
│  - Batch insert metrics                                         │
│  - Handle foreign key relationships                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: VERIFY                                                 │
│  - Count records before/after                                   │
│  - Sample validation (random record checks)                     │
│  - Generate migration report                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Migration Script (Python CLI)

```python
# src/lemonade/tools/dashboard_import.py

class DashboardImport(Tool):
    """Import evaluation results to dashboard database"""

    unique_name = "import-dashboard"

    def run(
        self,
        state: State,
        dashboard_url: str,
        api_key: str,
        cache_dir: str = None,
        dry_run: bool = False,
        skip_duplicates: bool = True,
        batch_size: int = 100,
    ) -> State:
        """
        Import YAML evaluation results to dashboard database.

        Args:
            dashboard_url: Dashboard API URL
            api_key: API key for authentication
            cache_dir: Cache directory to scan (default: ~/.cache/lemonade)
            dry_run: If True, only scan and report without importing
            skip_duplicates: If True, skip runs that already exist
            batch_size: Number of records to batch in each API call
        """
```

### YAML to Database Mapping

| YAML Field | Database Table | Column |
|------------|----------------|--------|
| `checkpoint` | models | checkpoint |
| `device` | runs | device |
| `dtype` | runs | dtype |
| `backend` | runs | backend |
| `timestamp` | runs | created_at |
| `build_name` | runs | build_name |
| `iterations` | runs.config | iterations |
| `prompts` | runs.config | prompts |
| `output_tokens` | runs.config | output_tokens |
| `seconds_to_first_token` | metrics | value_numeric (name=seconds_to_first_token) |
| `token_generation_tokens_per_second` | metrics | value_numeric (name=token_generation_tokens_per_second) |
| `max_memory_used_gbyte` | metrics | value_numeric (name=max_memory_used_gbyte) |
| `mmlu_*_accuracy` | metrics | value_numeric (name=mmlu_{subject}_accuracy) |
| `humaneval_*` | metrics | value_numeric (name=humaneval_*) |
| `system_info` | runs | system_info (JSONB) |

### Migration Validation Checklist

- [ ] All YAML files discovered and counted
- [ ] Parsing errors logged and reported
- [ ] Duplicate detection working correctly
- [ ] All metric types mapped correctly
- [ ] Timestamp conversion accurate (timezone handling)
- [ ] Foreign key relationships established
- [ ] Record counts match expected totals
- [ ] Sample data verified manually
- [ ] Rollback procedure tested

---

## 8. Security Considerations

### Authentication & Authorization

#### JWT Token Structure
```json
{
  "sub": "user-uuid",
  "email": "user@example.com",
  "role": "editor",
  "iat": 1234567890,
  "exp": 1234568890,
  "type": "access"
}
```

#### API Key Format
```
ledash_<prefix>_<secret>
Example: ledash_sk_a1b2c3_d4e5f6g7h8i9j0...
```

#### Role-Based Access Control (RBAC)

| Role | Read | Write | Delete | Admin |
|------|------|-------|--------|-------|
| viewer | Own + Shared | None | None | None |
| editor | All | Own + Shared | Own | None |
| admin | All | All | All | All |

### API Security

1. **Rate Limiting**
   - Default: 100 requests/minute per API key
   - Burst: 200 requests in 10 seconds
   - Implement via FastAPI middleware + Redis

2. **Input Validation**
   - Pydantic models for all request/response validation
   - SQL injection prevention via SQLAlchemy ORM
   - XSS prevention via output encoding

3. **API Key Security**
   - Hash keys with bcrypt before storage
   - Store only prefix (first 8 chars) in plain text for identification
   - Rotate keys via UI (revoke old, generate new)

### Data Security

1. **Encryption at Rest**
   - PostgreSQL TDE (Transparent Data Encryption)
   - Encrypted backups

2. **Encryption in Transit**
   - TLS 1.3 for all HTTPS connections
   - WebSocket Secure (WSS) for real-time

3. **Sensitive Data Handling**
   - Never log API keys or passwords
   - Redact sensitive fields in error messages
   - Audit log for all data access

### Infrastructure Security

1. **Network Security**
   - Database not exposed to public internet
   - Internal service communication via private network
   - Firewall rules for port access

2. **Container Security**
   - Non-root user in containers
   - Minimal base images
   - Regular vulnerability scanning

3. **Secret Management**
   - Environment variables via Docker secrets or Vault
   - Never commit secrets to version control
   - Rotate secrets regularly

### Security Checklist

- [ ] JWT implementation with proper expiration
- [ ] API key generation and validation
- [ ] RBAC middleware on all protected endpoints
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention verified
- [ ] XSS prevention implemented
- [ ] HTTPS enforced (HSTS headers)
- [ ] Security headers configured (CSP, X-Frame-Options)
- [ ] Audit logging implemented
- [ ] Penetration testing completed

---

## 9. Testing Strategy

### Testing Pyramid

```
                        ┌─────────┐
                        │   E2E   │  ~10% (Playwright)
                       ┌┴─────────┴┐
                       │Integration│  ~20% (pytest, Supertest)
                      ┌┴───────────┴┐
                      │    Unit     │  ~70% (pytest, Vitest)
                     ┌┴─────────────┴┐
```

### Backend Testing (pytest)

#### Unit Tests
```python
# tests/unit/test_metrics_service.py

def test_calculate_aggregate_metrics():
    """Test metric aggregation calculations"""
    metrics = [
        {"name": "ttft", "value": 0.025},
        {"name": "ttft", "value": 0.023},
        {"name": "ttft", "value": 0.027},
    ]
    result = calculate_aggregate(metrics)
    assert result["mean"] == 0.025
    assert result["std_dev"] == pytest.approx(0.002)

def test_validate_yaml_import():
    """Test YAML file validation"""
    valid_yaml = {"checkpoint": "test-model", "seconds_to_first_token": 0.025}
    assert validate_yaml(valid_yaml) is True

    invalid_yaml = {"checkpoint": None}
    assert validate_yaml(invalid_yaml) is False
```

#### Integration Tests
```python
# tests/integration/test_runs_api.py

@pytest.mark.asyncio
async def test_create_run(api_client, test_user):
    """Test run creation via API"""
    response = await api_client.post(
        "/api/v1/runs",
        json={
            "model_id": "test-model-id",
            "run_type": "benchmark",
            "device": "cpu",
        },
        headers={"Authorization": f"Bearer {test_user.token}"},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "pending"

@pytest.mark.asyncio
async def test_run_comparison(api_client, test_runs):
    """Test run comparison endpoint"""
    run_ids = [r["id"] for r in test_runs[:2]]
    response = await api_client.get(
        f"/api/v1/metrics/compare?run_ids={','.join(run_ids)}"
    )
    assert response.status_code == 200
    assert "comparison" in response.json()
```

#### Database Tests
```python
# tests/integration/test_database.py

def test_model_uniqueness_constraint(db_session):
    """Test that duplicate checkpoints are rejected"""
    model1 = Model(checkpoint="test-model", name="Test")
    model2 = Model(checkpoint="test-model", name="Test Duplicate")

    db_session.add(model1)
    db_session.commit()

    db_session.add(model2)
    with pytest.raises(IntegrityError):
        db_session.commit()
```

### Frontend Testing (Vitest + React Testing Library)

#### Component Tests
```tsx
// tests/components/MetricCard.test.tsx

describe('MetricCard', () => {
  it('renders metric value and unit correctly', () => {
    render(<MetricCard name="TTFT" value={0.025} unit="s" />);

    expect(screen.getByText('TTFT')).toBeInTheDocument();
    expect(screen.getByText('0.025 s')).toBeInTheDocument();
  });

  it('highlights best value when multiple cards present', () => {
    render(
      <div>
        <MetricCard name="TTFT" value={0.025} unit="s" isBest />
        <MetricCard name="TTFT" value={0.030} unit="s" />
      </div>
    );

    expect(screen.getByText('0.025 s')).toHaveClass('highlight');
  });
});
```

#### Hook Tests
```tsx
// tests/hooks/useRuns.test.tsx

describe('useRuns', () => {
  it('fetches runs on mount', async () => {
    const { result } = renderHook(() => useRuns(), {
      wrapper: QueryClientWrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toHaveLength(3);
  });

  it('handles fetch error gracefully', async () => {
    server.use(
      rest.get('/api/v1/runs', (req, res, ctx) => {
        return res(ctx.status(500));
      })
    );

    const { result } = renderHook(() => useRuns(), {
      wrapper: QueryClientWrapper,
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });
});
```

### E2E Testing (Playwright)

```typescript
// tests/e2e/dashboard.spec.ts

test('user can view dashboard and compare runs', async ({ page }) => {
  // Login
  await page.goto('/login');
  await page.fill('[name="email"]', 'test@example.com');
  await page.fill('[name="password"]', 'password123');
  await page.click('button[type="submit"]');

  // Navigate to dashboard
  await expect(page).toHaveURL('/dashboard');
  await expect(page.locator('text=Summary')).toBeVisible();

  // Go to runs page
  await page.click('text=Runs');
  await expect(page).toHaveURL('/runs');

  // Select runs for comparison
  await page.check('[data-testid="run-select-1"]');
  await page.check('[data-testid="run-select-2"]');
  await page.click('text=Compare Selected');

  // Verify comparison page
  await expect(page).toHaveURL(/\/compare/);
  await expect(page.locator('text=Comparison')).toBeVisible();
});
```

### Performance Testing (Locust)

```python
# tests/performance/locustfile.py

from locust import HttpUser, task, between

class DashboardUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def view_dashboard(self):
        self.client.get("/api/v1/runs")
        self.client.get("/api/v1/metrics/aggregate")

    @task(2)
    def view_models(self):
        self.client.get("/api/v1/models")

    @task(1)
    def compare_runs(self):
        self.client.get("/api/v1/metrics/compare?run_ids=1,2,3")
```

### Test Coverage Goals

| Component | Target Coverage | Critical Areas |
|-----------|-----------------|----------------|
| API Endpoints | 90%+ | Auth, data validation, error handling |
| Services | 85%+ | Business logic, calculations |
| Database | 80%+ | Queries, relationships, constraints |
| Frontend Components | 75%+ | User-facing components |
| Frontend Hooks | 85%+ | Data fetching, state management |
| E2E Flows | Key user journeys | Login, view, compare, import |

### CI/CD Integration

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  backend-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
    steps:
      - uses: actions/checkout@v4
      - name: Run pytest
        run: |
          pip install -e ".[test]"
          pytest --cov=src --cov-report=xml

  frontend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Vitest
        run: |
          npm ci
          npm run test:coverage

  e2e-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
    steps:
      - uses: actions/checkout@v4
      - name: Run Playwright
        run: |
          docker compose up -d
          npm run test:e2e
```

---

## Appendix A: File Paths Reference

### Backend Structure
```
dashboard-backend/
├── src/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py
│   │   │   │   ├── models.py
│   │   │   │   ├── runs.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── reports.py
│   │   │   │   └── import.py
│   │   │   └── deps.py          # Dependencies (auth, DB session)
│   │   └── websocket.py
│   ├── models/
│   │   ├── user.py
│   │   ├── model.py
│   │   ├── run.py
│   │   ├── metric.py
│   │   └── tag.py
│   ├── schemas/
│   │   ├── user.py
│   │   ├── model.py
│   │   ├── run.py
│   │   └── metric.py
│   ├── services/
│   │   ├── auth.py
│   │   ├── models.py
│   │   ├── runs.py
│   │   ├── metrics.py
│   │   ├── import.py
│   │   └── reports.py
│   ├── db/
│   │   ├── session.py
│   │   ├── base.py
│   │   └── migrations/
│   └── main.py
├── tests/
├── alembic.ini
├── pyproject.toml
└── Dockerfile
```

### Frontend Structure
```
dashboard-frontend/
├── src/
│   ├── pages/
│   ├── components/
│   ├── hooks/
│   ├── stores/
│   ├── api/
│   ├── types/
│   └── utils/
├── tests/
├── index.html
├── vite.config.ts
├── package.json
└── Dockerfile
```

### Monorepo Structure
```
lemonade-eval-dashboard/
├── apps/
│   ├── backend/
│   └── frontend/
├── packages/
│   ├── shared-types/
│   └── api-client/
├── docker-compose.yml
├── docker-compose.dev.yml
├── Makefile
└── README.md
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| TTFT | Time To First Token - latency before first token generation |
| TPS | Tokens Per Second - generation throughput |
| MMLU | Massive Multitask Language Understanding benchmark |
| HumanEval | Code generation benchmark (pass@k metrics) |
| lm-eval-harness | EleutherAI's evaluation framework |
| GGUF | GGML Universal File Format for quantized models |
| OGA | ONNX Runtime GenAI |
| NPU | Neural Processing Unit |
| Quantization | Model compression technique (int4, int8, etc.) |

---

## Document Approval

| Role | Name | Date |
|------|------|------|
| Technical Lead | | |
| Product Manager | | |
| Engineering Manager | | |

---

*This document is living and should be updated as the implementation progresses and requirements evolve.*
