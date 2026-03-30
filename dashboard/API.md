# Lemonade Eval Dashboard API Documentation

Complete API reference for the Lemonade Eval Dashboard backend.

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

## Authentication

API endpoints require authentication via Bearer token in the Authorization header:

```bash
Authorization: Bearer your_api_key_here
```

## Common Response Format

All responses follow a consistent format:

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5
  }
}
```

Error responses:

```json
{
  "detail": "Error message here"
}
```

---

## Health Endpoints

### GET /api/v1/health

Health check endpoint to verify API and database status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "database": "connected",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/health
```

### GET /api/v1/health/ready

Readiness check for Kubernetes/load balancer health checks.

**Response:**
```json
{
  "ready": true
}
```

---

## Models API

### GET /api/v1/models

List all models with pagination and filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page (max 100) |
| search | string | - | Search term for name/checkpoint |
| family | string | - | Filter by model family (e.g., Llama, Qwen) |
| model_type | string | - | Filter by type (llm, vlm, embedding) |

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/models?page=1&per_page=10&family=Llama"
```

**Example Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "model_abc123",
      "name": "Llama 3 8B Instruct",
      "checkpoint": "meta-llama/Llama-3-8B-Instruct",
      "family": "Llama",
      "model_type": "llm",
      "parameters": 8000000000,
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T10:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "per_page": 10,
    "total": 25,
    "total_pages": 3
  }
}
```

### POST /api/v1/models

Create a new model.

**Request Body:**
```json
{
  "name": "Llama 3 8B Instruct",
  "checkpoint": "meta-llama/Llama-3-8B-Instruct",
  "family": "Llama",
  "model_type": "llm",
  "parameters": 8000000000,
  "metadata": {
    "license": "Llama 3 Community License",
    "source": "HuggingFace"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "name": "Llama 3 8B Instruct",
    "checkpoint": "meta-llama/Llama-3-8B-Instruct",
    "family": "Llama",
    "model_type": "llm"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "model_abc123",
    "name": "Llama 3 8B Instruct",
    "checkpoint": "meta-llama/Llama-3-8B-Instruct",
    "family": "Llama",
    "model_type": "llm",
    "parameters": 8000000000,
    "created_at": "2024-01-15T10:00:00Z"
  }
}
```

### GET /api/v1/models/{model_id}

Get a specific model by ID.

**Example:**
```bash
curl http://localhost:8000/api/v1/models/model_abc123 \
  -H "Authorization: Bearer your_api_key"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "model_abc123",
    "name": "Llama 3 8B Instruct",
    "checkpoint": "meta-llama/Llama-3-8B-Instruct",
    "family": "Llama",
    "model_type": "llm",
    "parameters": 8000000000,
    "metadata": {
      "license": "Llama 3 Community License"
    },
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z"
  }
}
```

### PUT /api/v1/models/{model_id}

Update an existing model.

**Example:**
```bash
curl -X PUT http://localhost:8000/api/v1/models/model_abc123 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "name": "Llama 3 8B Instruct (Updated)",
    "metadata": {"license": "New License"}
  }'
```

### DELETE /api/v1/models/{model_id}

Delete a model and all associated runs/metrics.

**Example:**
```bash
curl -X DELETE http://localhost:8000/api/v1/models/model_abc123 \
  -H "Authorization: Bearer your_api_key"
```

### GET /api/v1/models/families/list

Get list of unique model families.

**Response:**
```json
{
  "success": true,
  "data": ["Llama", "Qwen", "Mistral", "Gemma"]
}
```

### GET /api/v1/models/{model_id}/versions

Get all versions of a model.

### GET /api/v1/models/{model_id}/runs

Get recent runs for a model.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 100 | Maximum runs to return (max 1000) |

---

## Runs API

### GET /api/v1/runs

List evaluation runs with pagination and filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page (max 100) |
| model_id | string | - | Filter by model ID |
| status | string | - | Filter by status (pending, running, completed, failed) |
| run_type | string | - | Filter by type (benchmark, accuracy-mmlu, etc.) |
| device | string | - | Filter by device (cpu, gpu, npu) |
| backend | string | - | Filter by backend (llamacpp, ort, flm) |

**Example:**
```bash
curl "http://localhost:8000/api/v1/runs?status=completed&run_type=benchmark" \
  -H "Authorization: Bearer your_api_key"
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "run_xyz789",
      "model_id": "model_abc123",
      "build_name": "build_001",
      "run_type": "benchmark",
      "status": "completed",
      "device": "gpu",
      "backend": "llamacpp",
      "started_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:05:00Z",
      "created_at": "2024-01-15T09:59:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 50,
    "total_pages": 3
  }
}
```

### POST /api/v1/runs

Create a new evaluation run.

**Request Body:**
```json
{
  "model_id": "model_abc123",
  "build_name": "build_001",
  "run_type": "benchmark",
  "device": "gpu",
  "backend": "llamacpp",
  "metadata": {
    "quantization": "q4_0",
    "context_size": 4096
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "model_id": "model_abc123",
    "build_name": "build_001",
    "run_type": "benchmark",
    "device": "gpu",
    "backend": "llamacpp"
  }'
```

### GET /api/v1/runs/{run_id}

Get a specific run by ID.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| include_metrics | bool | false | Include metrics in response |

**Example with metrics:**
```bash
curl "http://localhost:8000/api/v1/runs/run_xyz789?include_metrics=true" \
  -H "Authorization: Bearer your_api_key"
```

### PUT /api/v1/runs/{run_id}

Update an existing run.

**Example:**
```bash
curl -X PUT http://localhost:8000/api/v1/runs/run_xyz789 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "status": "completed",
    "status_message": "Evaluation completed successfully"
  }'
```

### POST /api/v1/runs/{run_id}/status

Update run status with automatic timestamp management.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | New status (required) |
| message | string | Status message (optional) |

**Valid statuses:** `pending`, `running`, `completed`, `failed`, `cancelled`

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/runs/run_xyz789/status?status=running" \
  -H "Authorization: Bearer your_api_key"
```

### GET /api/v1/runs/recent/list

Get recent runs.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 10 | Number of runs to return (max 100) |

### GET /api/v1/runs/stats

Get overall run statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_runs": 150,
    "completed": 120,
    "running": 5,
    "failed": 10,
    "pending": 15,
    "avg_duration_seconds": 300
  }
}
```

### DELETE /api/v1/runs/{run_id}

Delete a run (cascades to metrics).

### GET /api/v1/runs/{run_id}/metrics

Get all metrics for a run.

---

## Metrics API

### GET /api/v1/metrics

List metrics with pagination and filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page |
| run_id | string | - | Filter by run ID |
| model_id | string | - | Filter by model ID |
| metric_name | string | - | Filter by metric name |
| category | string | - | Filter by category |

**Example:**
```bash
curl "http://localhost:8000/api/v1/metrics?run_id=run_xyz789&category=performance" \
  -H "Authorization: Bearer your_api_key"
```

### POST /api/v1/metrics

Create a new metric.

**Request Body:**
```json
{
  "run_id": "run_xyz789",
  "model_id": "model_abc123",
  "metric_name": "tokens_per_second",
  "value": 45.5,
  "category": "performance",
  "metadata": {
    "unit": "tokens/s",
    "iteration": 100
  }
}
```

### GET /api/v1/metrics/aggregate

Get aggregated metrics.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| model_id | string | Filter by model |
| run_type | string | Filter by run type |
| category | string | Filter by category |
| metric_name | string | Filter by metric name |

### GET /api/v1/metrics/trends

Get metric trends for a model.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| modelId | string | - | Model ID (required) |
| metricName | string | - | Metric name (required) |
| limit | int | 100 | Maximum results |

### GET /api/v1/metrics/compare

Compare metrics across runs.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| run_ids | string[] | Comma-separated run IDs |
| categories | string[] | Optional categories to include |

### GET /api/v1/metrics/performance/{run_id}

Get performance metrics for a specific run.

---

## Import API

### POST /api/v1/import/yaml

Import evaluation results from YAML.

**Request Body:**
```json
{
  "yaml_content": "model:\n  name: Test Model\n  ...\n",
  "source": "file_upload"
}
```

### GET /api/v1/import/jobs

List all import jobs.

### GET /api/v1/import/jobs/{job_id}

Get status of a specific import job.

### POST /api/v1/import/scan

Scan a cache directory for import candidates.

---

## WebSocket API

### WS /ws/v1/evaluations

Connect for real-time evaluation updates.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| run_id | string | Optional run ID to subscribe to |

**Example:**
```bash
wscat -c "ws://localhost:8000/ws/v1/evaluations?run_id=run_xyz789"
```

**Message Types:**

**Subscribe to a run:**
```json
{
  "type": "subscribe",
  "run_id": "run_xyz789"
}
```

**Response:**
```json
{
  "type": "subscribed",
  "run_id": "run_xyz789"
}
```

**Unsubscribe:**
```json
{
  "type": "unsubscribe"
}
```

**Ping (health check):**
```json
{
  "type": "ping"
}
```

**Response:**
```json
{
  "type": "pong"
}
```

**Server Events:**

Run status update:
```json
{
  "event_type": "run_status",
  "run_id": "run_xyz789",
  "status": "running",
  "message": "Evaluation started",
  "data": { "progress": 0 }
}
```

Metrics stream:
```json
{
  "event_type": "metrics_stream",
  "run_id": "run_xyz789",
  "metrics": [
    {
      "metric_name": "tokens_per_second",
      "value": 45.5,
      "timestamp": "2024-01-15T10:00:00Z"
    }
  ]
}
```

Progress update:
```json
{
  "event_type": "progress",
  "run_id": "run_xyz789",
  "progress": 50.0,
  "message": "Processing batch 5/10"
}
```

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 409 | Conflict - Resource already exists |
| 500 | Internal Server Error |

---

## Rate Limiting

API requests are rate limited to:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated endpoints

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642234567
```

---

## OpenAPI Specification

The complete OpenAPI spec is available at:
- JSON: `/openapi.json`
- Swagger UI: `/docs`
- ReDoc: `/redoc`

**Generate OpenAPI spec:**
```bash
curl http://localhost:8000/openapi.json -o openapi.json
```
