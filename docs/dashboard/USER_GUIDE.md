# Lemonade Eval Dashboard - User Guide

Complete guide for using the Lemonade Eval Dashboard to store, visualize, and compare LLM/VLM evaluation results.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Feature Walkthrough](#feature-walkthrough)
   - [Dashboard Overview](#dashboard-overview)
   - [Models Management](#models-management)
   - [Runs Management](#runs-management)
   - [Compare Page](#compare-page)
   - [Benchmarks Page](#benchmarks-page)
   - [Accuracy Page](#accuracy-page)
   - [Import Page](#import-page)
   - [Settings Page](#settings-page)
3. [CLI Integration Guide](#cli-integration-guide)
4. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

Before using the dashboard, ensure you have:

- **Backend**: Python 3.12+, SQLite (default) or PostgreSQL 16+
- **Frontend**: Node.js 18+, npm or pnpm
- **Existing Data**: lemonade-eval YAML files in `~/.cache/lemonade/` (optional)

### Quick Start

#### Step 1: Install and Start Backend

```bash
# Navigate to backend directory
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

# Copy environment file and configure
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/macOS

# Edit .env file with your settings (SQLite is the default, no DB setup needed):
# - DATABASE_URL=sqlite:///./eval.db
# - SECRET_KEY=your-secret-key-at-least-32-characters

# Run database migrations
alembic upgrade head

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 3001
```

The backend API will be available at `http://localhost:3001`.

#### Step 2: Install and Start Frontend

```bash
# Navigate to frontend directory
cd dashboard/frontend

# Install dependencies
npm install

# Copy environment file
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/macOS

# Start development server
npm run dev
```

The frontend dashboard will be available at `http://localhost:5173` (Vite dev server default).

#### Step 3: First Login

1. Navigate to `http://localhost:5173`
2. Click "Login" or navigate to `/login`
3. Use your credentials (default admin user must be created via database script)

**Create First Admin User:**

```bash
# Create admin user script
cd dashboard/backend
python -c "
from sqlalchemy import create_engine
from app.database import Base
from app.models import User
import bcrypt
from uuid import uuid4

engine = create_engine('your-database-url')

with engine.connect() as conn:
    user = User(
        id=str(uuid4()),
        email='admin@example.com',
        name='Admin User',
        role='admin',
        is_active=True,
        hashed_password=bcrypt.hashpw('AdminPassword123'.encode(), bcrypt.gensalt()).decode()
    )
    conn.execute(User.__table__.insert().values(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        hashed_password=user.hashed_password
    ))
    conn.commit()
    print('Admin user created: admin@example.com / AdminPassword123')
"
```

---

## Feature Walkthrough

### Dashboard Overview

The main dashboard provides an at-a-glance view of your evaluation ecosystem.

**Key Metrics Cards:**
- **Total Models**: Number of registered models
- **Total Runs**: Total evaluation runs
- **Completed**: Successfully completed runs
- **Failed**: Failed runs requiring attention

**Recent Runs Table:**
- Displays the 10 most recent evaluation runs
- Click any run to view detailed results
- Shows run type, status, duration, and creation date

**Run Status Breakdown:**
- Visual breakdown of runs by status (Running, Pending, Completed, Failed)
- Average duration indicator for completed runs

### Models Management

Access: `/models`

#### Models List

The models page displays all registered LLM/VLM models with:

- **Search**: Filter by model name or checkpoint path
- **Family Filter**: Group by model family (Llama, Qwen, Mistral, etc.)
- **Type Filter**: Filter by type (llm, vlm, embedding)
- **Pagination**: Navigate through large model lists

| Column | Description |
|--------|-------------|
| Name | Model display name |
| Checkpoint | Model checkpoint path |
| Family | Model family |
| Type | Model type (llm/vlm/embedding) |
| Parameters | Parameter count |
| Actions | View, Edit, Delete |

#### Model Detail View

Click any model to see:

- **Model Information**: Full details including metadata
- **Version History**: All variants/versions of the model
- **Associated Runs**: Evaluation runs for this model
- **Performance Trends**: Charts showing metric trends over time

#### Creating a Model

1. Click "New Model" button
2. Fill in required fields:
   - **Name**: Display name (required)
   - **Checkpoint**: Model path/identifier (required)
   - **Family**: Model family (e.g., Llama, Qwen)
   - **Type**: llm, vlm, or embedding
   - **Parameters**: Parameter count
3. Optionally add metadata (license, source, etc.)
4. Click "Create Model"

### Runs Management

Access: `/runs`

#### Runs List

View all evaluation runs with advanced filtering:

- **Status Filter**: pending, running, completed, failed
- **Run Type**: benchmark, accuracy-mmlu, accuracy-humaneval, etc.
- **Device**: cpu, gpu, npu
- **Backend**: llamacpp, ort, flm
- **Model Filter**: Filter by specific model

#### Run Detail View

Click any run to see comprehensive details:

**Run Information:**
- Build name and run type
- Status and status message
- Device, backend, and data type
- Duration and timestamps

**Metrics Breakdown:**

| Category | Metrics |
|----------|---------|
| **Performance** | TTFT, Prefill TPS, Generation TPS, Memory Usage |
| **Accuracy** | MMLU scores, HumanEval pass rate, Perplexity |

**Configuration:**
- Iterations, prompts, and other run parameters
- Environment details

#### Run Status Lifecycle

```
pending -> running -> completed
                     -> failed
                     -> cancelled
```

### Compare Page

Access: `/compare`

The compare feature enables side-by-side analysis of multiple runs.

#### Compare Runs

1. **Select Runs**: Use the multi-select dropdown to choose 2-5 runs
2. **View Comparison Table**: See all metrics side by side
3. **Analyze Charts**:
   - **Bar Chart**: Compare tokens per second across runs
   - **Radar Chart**: Multi-metric visualization

#### Compare Models

Switch to "Compare Models" mode using the segmented control to aggregate results across all runs of each model:

1. **Select Models**: Choose 2+ models from the multi-select dropdown
2. **View Aggregated Table**: Mean performance across all runs, with best values highlighted green
3. **Bar Chart**: Average TPS comparison across selected models

#### Comparison Table Features

- **Best Value Highlighting**:
  - Green highlight for best values
  - Lower-is-better metrics (TTFT, memory) highlight minimum
  - Higher-is-better metrics (TPS) highlight maximum
- **Metric Categories**: Performance and accuracy metrics grouped

### Benchmarks Page

Access: `/benchmarks`

Visualizes benchmark results from the `bench` command across all runs.

**Winner Highlight**: Automatically identifies and prominently displays the fastest model.

**Summary Statistics**: Total runs, best TPS, best TTFT, and number of unique models.

**Charts:**
- **TPS Comparison Bar Chart**: Token generation speed for each run, sorted best-first
- **TPS vs Prompt Length Chart**: Line chart showing how TPS changes across prompt lengths (64 → 128 → 256 tokens) for each run that has per-length metrics

**Detailed Results Table:**

| Column | Description |
|--------|-------------|
| Model | Model identifier |
| TPS (tok/s) | Token generation speed |
| TPS@64 / TPS@128 / TPS@256 | Per-prompt-length TPS |
| TTFT (s) | Time to first token |
| Std Dev | Consistency across iterations |
| Backend | Inference backend (llamacpp, ort, etc.) |
| Device | Hardware (cpu/gpu/npu) |

Per-prompt-length columns (`TPS@64`, `TPS@128`, `TPS@256`) are populated when benchmarks are run with `--prompt-lengths 64,128,256`. See `docs/dashboard/benchmark_guide.md` for the full workflow.

**Model Averages**: Cards showing aggregated TPS and TTFT when multiple runs exist for the same model.

### Accuracy Page

Access: `/accuracy`

Displays accuracy evaluation results across three tabs. Use the model selector in the top-right to filter all tabs to a specific model.

#### MMLU Tab

- Select a specific MMLU run from the dropdown (defaults to most recent)
- **Average MMLU Accuracy** metric card shows overall score
- **Subject Accuracy Chart**: Horizontal bar chart of all 57 MMLU subjects, sorted highest-first
- Use the scroll area to explore all subjects

#### HumanEval Tab

- Lists all HumanEval runs for the selected model
- Shows run name, status, and date

#### Perplexity Tab

- Displays perplexity trend over time for the selected model
- Only fetches data when this tab is active (no unnecessary network calls)
- Select a model first — no data is shown without a model selection

### Import Page

Access: `/import`

Import existing evaluation data from lemonade-eval cache.

#### Import from Cache

1. **Enter Cache Directory**: Default is `~/.cache/lemonade/`
2. **Scan for Files**: Click "Scan" to find YAML files
3. **Review Candidates**: See available evaluations to import
4. **Select Files**: Choose which evaluations to import
5. **Configure Options**:
   - Skip duplicates: Avoid re-importing existing runs
   - Include metadata: Import full configuration details
6. **Start Import**: Click "Import Selected"

#### Import Progress

- Real-time progress indicator
- WebSocket updates for long-running imports
- Summary report on completion:
  - Imported count
  - Skipped count (duplicates)
  - Failed count with error details

#### YAML Format

The dashboard accepts lemonade-eval YAML format:

```yaml
model:
  name: Llama-3-8B-Instruct
  checkpoint: meta-llama/Llama-3-8B-Instruct
  family: Llama
  model_type: llm
  parameters: 8000000000

run:
  build_name: llama-3-8b-benchmark-20240115
  run_type: benchmark
  device: gpu
  backend: ort
  dtype: float16
  status: completed
  duration_seconds: 120.5

metrics:
  - name: seconds_to_first_token
    value: 0.025
    unit: seconds
    category: performance
  - name: token_generation_tokens_per_second
    value: 45.5
    unit: tokens/s
    category: performance

config:
  iterations: 10
  warmup_iterations: 2
  output_tokens: 128
```

### Settings Page

Access: `/settings`

Configure dashboard preferences:

- **Profile**: Update user information
- **API Keys**: Generate and manage API keys for CLI integration
- **Preferences**: Theme, notifications, and display settings

---

## CLI Integration Guide

The dashboard integrates seamlessly with the lemonade-eval CLI for automated result submission.

### Configuration

#### Generate API Key

1. Navigate to Settings > API Keys
2. Click "Generate New Key"
3. Copy the key (it won't be shown again)
4. Store securely

#### Configure CLI

Add dashboard configuration to your lemonade-eval setup:

```bash
# Set environment variables
export LEDASH_API_URL=http://localhost:3001
export LEDASH_API_KEY=your-api-key-here
export LEDASH_SECRET=your-cli-secret-here
```

### Automatic Submission

When running evaluations, results are automatically submitted to the dashboard:

```bash
# Run evaluation with dashboard integration
lemonade-eval -i Qwen3-4B-Instruct-2507-GGUF load bench \
  --dashboard-submit \
  --dashboard-api-key $LEDASH_API_KEY
```

### Manual Submission

Submit existing evaluation results:

```bash
# Submit a single evaluation
curl -X POST http://localhost:3001/api/v1/import/evaluation \
  -H "Content-Type: application/json" \
  -H "X-CLI-Signature: $(echo -n '{"model_id":"..."}' | openssl dgst -sha256 -hmac 'secret' -binary | base64)" \
  -d '{
    "model_id": "meta-llama/Llama-3-8B-Instruct",
    "run_type": "benchmark",
    "build_name": "test-benchmark-001",
    "metrics": [
      {"name": "seconds_to_first_token", "value": 0.025, "unit": "seconds"},
      {"name": "token_generation_tokens_per_second", "value": 45.5, "unit": "tokens/s"}
    ],
    "device": "gpu",
    "backend": "ort",
    "status": "completed",
    "duration_seconds": 120.5
  }'
```

### Bulk Import

For migrating historical data:

```bash
curl -X POST http://localhost:3001/api/v1/import/bulk \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "evaluations": [...],
    "skip_duplicates": true
  }'
```

### WebSocket Progress

Subscribe to real-time progress updates:

```python
import asyncio
import websockets

async def watch_progress(run_id):
    uri = f"ws://localhost:3001/ws/v1/evaluation-progress?run_id={run_id}"
    async with websockets.connect(uri) as ws:
        while True:
            msg = await ws.recv()
            print(f"Progress: {msg}")

asyncio.run(watch_progress("run-xyz-123"))
```

---

## Troubleshooting

### Common Issues

#### Cannot Connect to Backend

**Symptom**: Frontend shows "Connection refused" or API errors

**Solutions:**
1. Verify backend is running: `curl http://localhost:3001/api/v1/health`
2. Check backend logs: `tail -f dashboard/backend/logs/error.log`
3. Verify CORS settings in backend `.env`:
   ```
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   ```
4. Ensure frontend API URL is correct in `.env`:
   ```
   VITE_API_BASE_URL=http://localhost:3001
   ```

#### Database Connection Errors

**Symptom**: Backend fails to start with database errors

**Solutions (SQLite — default):**
1. Verify `eval.db` exists: `ls dashboard/backend/eval.db`
2. Initialize if missing:
   ```bash
   cd dashboard/backend
   python -c "from app.database import Base, sync_engine; Base.metadata.create_all(bind=sync_engine)"
   ```
3. Run migrations: `alembic upgrade head`

**Solutions (PostgreSQL — optional):**
1. Verify PostgreSQL is running: `sudo systemctl status postgresql`
2. Check database credentials in `.env`
3. Test connection: `psql postgresql://user:password@localhost:5432/lemonade_dashboard`

#### Authentication Failures

**Symptom**: "Invalid credentials" or token errors

**Solutions:**
1. Verify password meets requirements:
   - Minimum 8 characters
   - At least one uppercase letter
   - At least one lowercase letter
   - At least one number
2. Clear browser cache and sessionStorage
3. Regenerate API key if using CLI integration
4. Check SECRET_KEY is properly set in backend `.env`

#### Import Fails

**Symptom**: YAML import fails or shows errors

**Solutions:**
1. Validate YAML format matches expected schema
2. Check file encoding is UTF-8
3. Verify required fields in YAML:
   - `model.name` or `model.checkpoint`
   - `run.build_name`
   - At least one metric
4. Check for duplicate handling settings
5. Review import logs for specific errors

#### WebSocket Disconnections

**Symptom**: Real-time updates stop working

**Solutions:**
1. Check WebSocket URL configuration
2. Verify firewall allows WebSocket connections
3. For nginx proxy, ensure upgrade headers are configured:
   ```nginx
   location /ws/ {
       proxy_pass http://localhost:3001;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "Upgrade";
   }
   ```

#### Slow Performance

**Symptom**: Dashboard loads slowly or times out

**Solutions:**
1. Add database indexes:
   ```sql
   CREATE INDEX idx_runs_status ON runs(status);
   CREATE INDEX idx_runs_created_at ON runs(created_at);
   CREATE INDEX idx_metrics_run_id ON metrics(run_id);
   ```
2. Enable Redis caching in production
3. Increase pagination page size limits
4. Optimize frontend bundle with `npm run build`

#### Memory Issues

**Symptom**: Backend runs out of memory

**Solutions:**
1. Reduce number of gunicorn workers
2. Enable database connection pooling
3. Increase pagination limits cautiously
4. Monitor with Prometheus metrics at `/metrics`

### Error Codes Reference

| Code | Meaning | Resolution |
|------|---------|------------|
| 400 | Bad Request | Check request format and required fields |
| 401 | Unauthorized | Verify API key or login token |
| 403 | Forbidden | Check user permissions |
| 404 | Not Found | Verify resource ID exists |
| 409 | Conflict | Resource already exists |
| 429 | Rate Limited | Wait and retry, or increase limits |
| 500 | Server Error | Check backend logs |

### Getting Help

- **Documentation**: `/docs` directory
- **API Reference**: `http://localhost:3001/docs` (Swagger UI)
- **Logs**: `dashboard/backend/logs/`
- **GitHub Issues**: https://github.com/lemonade/lemonade-eval/issues
