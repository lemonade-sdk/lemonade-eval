# Lemonade Eval Dashboard - API Documentation

Complete API reference for the Lemonade Eval Dashboard backend.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Codes](#error-codes)
5. [Health Endpoints](#health-endpoints)
6. [Authentication Endpoints](#authentication-endpoints)
7. [Models API](#models-api)
8. [Runs API](#runs-api)
9. [Metrics API](#metrics-api)
10. [Import API](#import-api)
11. [WebSocket API](#websocket-api)

---

## Overview

### Base URLs

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:3001` |
| Production | `https://your-domain.com` |

### API Prefixes

| Type | Prefix |
|------|--------|
| REST API | `/api/v1` |
| WebSocket | `/ws/v1` |

### Common Response Format

**Success Response:**
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5
  },
  "message": "Optional success message"
}
```

**Error Response:**
```json
{
  "success": false,
  "detail": "Error message here",
  "error_code": "ERR400"
}
```

**Validation Error Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Pagination

All list endpoints support pagination with the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number (1-indexed) |
| `per_page` | int | 20 | Items per page (max 100) |

### Standard Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `search` | string | Search term for text fields |
| `sort_by` | string | Field to sort by |
| `sort_order` | string | `asc` or `desc` |

---

## Authentication

### Authentication Methods

The dashboard supports two authentication methods:

1. **JWT Bearer Token** (for user authentication)
2. **API Key** (for CLI/service integration)

### Using JWT Tokens

Include the token in the Authorization header:

```bash
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Using API Keys

Include the API key in a custom header:

```bash
X-API-Key: ledash_your-api-key-here
```

### Token Expiration

- Access tokens expire after 30 minutes (configurable)
- Refresh tokens to maintain session
- Handle 401 responses by re-authenticating

### Security Best Practices

1. Store tokens securely (sessionStorage, not localStorage)
2. Use HTTPS in production
3. Rotate API keys periodically
4. Implement token refresh before expiration

---

## Rate Limiting

### Default Limits

| Endpoint Type | Rate Limit | Burst Limit |
|---------------|------------|-------------|
| Default API | 100 req/min | 200 requests |
| Authentication | 10 req/min | 15 requests |
| Import (YAML) | 10 req/min | 20 requests |
| Import (Bulk) | 1000 req/min | 2000 requests |
| Reports Export | 5 req/min | 10 requests |
| Metrics Bulk | 500 req/min | 1000 requests |

### Rate Limit Headers

Responses include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642234567
Retry-After: 60  (on 429 response)
```

### Handling Rate Limits

When rate limited (429 response):

1. Check `Retry-After` header for wait time
2. Implement exponential backoff
3. Cache responses when possible
4. Use bulk endpoints for batch operations

```python
import time
import requests

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

---

## Error Codes

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource does not exist |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Application Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `ERR400` | Bad Request | Check request format |
| `ERR401` | Authentication Required | Provide valid credentials |
| `ERR403` | Access Denied | Check permissions |
| `ERR404` | Not Found | Verify resource ID |
| `ERR409` | Duplicate Resource | Use different identifier |
| `ERR422` | Validation Error | Fix validation issues |
| `ERR429` | Rate Limited | Wait and retry |
| `ERR500` | Internal Error | Contact support |

---

## Health Endpoints

### GET /api/v1/health

Health check endpoint to verify API and database status.

**Request:**
```bash
curl http://localhost:3001/api/v1/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "database": "connected",
  "version": "1.0.0"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "database": "disconnected",
  "error": "Unable to connect to database"
}
```

### GET /api/v1/health/ready

Readiness check for Kubernetes/load balancer health checks.

**Request:**
```bash
curl http://localhost:3001/api/v1/health/ready
```

**Response (200 OK):**
```json
{
  "ready": true
}
```

**Response (503 Service Unavailable):**
```json
{
  "ready": false,
  "reason": "Database not initialized"
}
```

---

## Authentication Endpoints

### POST /api/v1/auth/login

User login endpoint.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePassword123"
  }'
```

**Password Requirements:**
- Minimum 8 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 1800,
    "user": {
      "id": "user-uuid-here",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "admin",
      "is_active": true,
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-15T00:00:00Z"
    }
  }
}
```

**Errors:**
- 401: Invalid email or password
- 422: Password does not meet requirements

### POST /api/v1/auth/logout

User logout endpoint.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/auth/logout \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Logout successful. Please clear your token."
}
```

**Note:** JWT tokens are stateless. The client must clear the token from storage.

### POST /api/v1/auth/refresh

Refresh access token.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/auth/refresh \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

### GET /api/v1/auth/me

Get current user information.

**Request:**
```bash
curl http://localhost:3001/api/v1/auth/me \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": "user-uuid-here",
    "email": "user@example.com",
    "name": "John Doe",
    "role": "admin",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-15T00:00:00Z"
  }
}
```

---

## Models API

### GET /api/v1/models

List all models with pagination and filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 20 | Items per page (max 100) |
| `search` | string | - | Search term for name/checkpoint |
| `family` | string | - | Filter by model family |
| `model_type` | string | - | Filter by type (llm, vlm, embedding) |

**Request:**
```bash
curl "http://localhost:3001/api/v1/models?page=1&per_page=10&family=Llama" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "model-abc123",
      "name": "Llama 3 8B Instruct",
      "checkpoint": "meta-llama/Llama-3-8B-Instruct",
      "family": "Llama",
      "model_type": "llm",
      "parameters": 8000000000,
      "metadata": {
        "license": "Llama 3 Community License",
        "source": "HuggingFace"
      },
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

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "name": "Llama 3 8B Instruct",
    "checkpoint": "meta-llama/Llama-3-8B-Instruct",
    "family": "Llama",
    "model_type": "llm",
    "parameters": 8000000000,
    "metadata": {
      "license": "Llama 3 Community License",
      "source": "HuggingFace"
    }
  }'
```

**Request Body Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name |
| `checkpoint` | string | Yes | Model checkpoint path |
| `family` | string | No | Model family |
| `model_type` | string | Yes | llm, vlm, or embedding |
| `parameters` | integer | No | Parameter count |
| `metadata` | object | No | Additional metadata |

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": "model-abc123",
    "name": "Llama 3 8B Instruct",
    "checkpoint": "meta-llama/Llama-3-8B-Instruct",
    "family": "Llama",
    "model_type": "llm",
    "parameters": 8000000000,
    "created_at": "2024-01-15T10:00:00Z"
  },
  "message": "Model created successfully"
}
```

### GET /api/v1/models/{model_id}

Get a specific model by ID.

**Request:**
```bash
curl http://localhost:3001/api/v1/models/model-abc123 \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": "model-abc123",
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

**Request:**
```bash
curl -X PUT http://localhost:3001/api/v1/models/model-abc123 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "name": "Llama 3 8B Instruct (Updated)",
    "metadata": {"license": "New License"}
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": "model-abc123",
    "name": "Llama 3 8B Instruct (Updated)",
    "checkpoint": "meta-llama/Llama-3-8B-Instruct",
    "family": "Llama",
    "model_type": "llm",
    "parameters": 8000000000,
    "metadata": {
      "license": "New License"
    },
    "updated_at": "2024-01-15T11:00:00Z"
  }
}
```

### DELETE /api/v1/models/{model_id}

Delete a model and all associated runs/metrics.

**Request:**
```bash
curl -X DELETE http://localhost:3001/api/v1/models/model-abc123 \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Model and associated data deleted successfully"
}
```

**Note:** This operation cascades to delete all related runs and metrics.

### GET /api/v1/models/families/list

Get list of unique model families.

**Request:**
```bash
curl http://localhost:3001/api/v1/models/families/list \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": ["Llama", "Qwen", "Mistral", "Gemma"]
}
```

### GET /api/v1/models/{model_id}/versions

Get all versions of a model.

**Request:**
```bash
curl http://localhost:3001/api/v1/models/model-abc123/versions \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "version-1",
      "model_id": "model-abc123",
      "version_name": "v1.0",
      "checkpoint": "meta-llama/Llama-3-8B-Instruct-v1.0",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

### GET /api/v1/models/{model_id}/runs

Get recent runs for a model.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Maximum runs to return (max 1000) |

**Request:**
```bash
curl "http://localhost:3001/api/v1/models/model-abc123/runs?limit=50" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "run-xyz789",
      "build_name": "llama-3-8b-benchmark-001",
      "run_type": "benchmark",
      "status": "completed",
      "device": "gpu",
      "backend": "ort",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

---

## Runs API

### GET /api/v1/runs

List evaluation runs with pagination and filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 20 | Items per page (max 100) |
| `model_id` | string | - | Filter by model ID |
| `status` | string | - | Filter by status |
| `run_type` | string | - | Filter by run type |
| `device` | string | - | Filter by device |
| `backend` | string | - | Filter by backend |
| `dtype` | string | - | Filter by data type |

**Valid Status Values:** `pending`, `running`, `completed`, `failed`, `cancelled`

**Valid Run Types:** `benchmark`, `accuracy-mmlu`, `accuracy-humaneval`, `accuracy-perplexity`, `lm-eval-harness`

**Valid Devices:** `cpu`, `gpu`, `npu`, `hybrid`

**Request:**
```bash
curl "http://localhost:3001/api/v1/runs?status=completed&run_type=benchmark" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "run-xyz789",
      "model_id": "model-abc123",
      "build_name": "llama-3-8b-benchmark-001",
      "run_type": "benchmark",
      "status": "completed",
      "device": "gpu",
      "backend": "ort",
      "dtype": "float16",
      "config": {
        "iterations": 10,
        "warmup_iterations": 2
      },
      "started_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:05:00Z",
      "duration_seconds": 300.5,
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

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/runs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model_id": "model-abc123",
    "build_name": "llama-3-8b-benchmark-001",
    "run_type": "benchmark",
    "device": "gpu",
    "backend": "ort",
    "dtype": "float16",
    "config": {
      "iterations": 10,
      "warmup_iterations": 2,
      "output_tokens": 128
    }
  }'
```

**Request Body Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model ID |
| `build_name` | string | Yes | Unique build identifier |
| `run_type` | string | Yes | Type of evaluation |
| `device` | string | No | Device type |
| `backend` | string | No | Backend runtime |
| `dtype` | string | No | Data type |
| `config` | object | No | Run configuration |
| `started_at` | string | No | Start timestamp (ISO 8601) |

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": "run-xyz789",
    "model_id": "model-abc123",
    "build_name": "llama-3-8b-benchmark-001",
    "run_type": "benchmark",
    "status": "pending",
    "created_at": "2024-01-15T09:59:00Z"
  },
  "message": "Run created successfully"
}
```

### GET /api/v1/runs/{run_id}

Get a specific run by ID.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_metrics` | bool | false | Include metrics in response |

**Request:**
```bash
curl "http://localhost:3001/api/v1/runs/run-xyz789?include_metrics=true" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": "run-xyz789",
    "model_id": "model-abc123",
    "build_name": "llama-3-8b-benchmark-001",
    "run_type": "benchmark",
    "status": "completed",
    "device": "gpu",
    "backend": "ort",
    "dtype": "float16",
    "config": {
      "iterations": 10,
      "warmup_iterations": 2
    },
    "started_at": "2024-01-15T10:00:00Z",
    "completed_at": "2024-01-15T10:05:00Z",
    "duration_seconds": 300.5,
    "status_message": "Evaluation completed successfully",
    "created_at": "2024-01-15T09:59:00Z",
    "metrics": [
      {
        "id": "metric-1",
        "name": "seconds_to_first_token",
        "value_numeric": 0.025,
        "unit": "seconds",
        "category": "performance"
      },
      {
        "id": "metric-2",
        "name": "token_generation_tokens_per_second",
        "value_numeric": 45.5,
        "unit": "tokens/s",
        "category": "performance"
      }
    ]
  }
}
```

### PUT /api/v1/runs/{run_id}

Update an existing run.

**Request:**
```bash
curl -X PUT http://localhost:3001/api/v1/runs/run-xyz789 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "status": "completed",
    "status_message": "Evaluation completed successfully",
    "completed_at": "2024-01-15T10:05:00Z",
    "duration_seconds": 300.5
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": "run-xyz789",
    "status": "completed",
    "status_message": "Evaluation completed successfully",
    "completed_at": "2024-01-15T10:05:00Z",
    "duration_seconds": 300.5,
    "updated_at": "2024-01-15T10:05:00Z"
  }
}
```

### POST /api/v1/runs/{run_id}/status

Update run status with automatic timestamp management.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | Yes | New status |
| `message` | string | No | Status message |

**Valid Statuses:** `pending`, `running`, `completed`, `failed`, `cancelled`

**Request:**
```bash
curl -X POST "http://localhost:3001/api/v1/runs/run-xyz789/status?status=running" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": "run-xyz789",
    "status": "running",
    "started_at": "2024-01-15T10:00:00Z"
  },
  "message": "Run status updated to running"
}
```

### GET /api/v1/runs/recent/list

Get recent runs.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 10 | Number of runs to return (max 100) |

**Request:**
```bash
curl "http://localhost:3001/api/v1/runs/recent/list?limit=5" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "run-xyz789",
      "build_name": "llama-3-8b-benchmark-001",
      "status": "completed",
      "created_at": "2024-01-15T09:59:00Z"
    }
  ]
}
```

### GET /api/v1/runs/stats

Get overall run statistics.

**Request:**
```bash
curl http://localhost:3001/api/v1/runs/stats \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
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

**Request:**
```bash
curl -X DELETE http://localhost:3001/api/v1/runs/run-xyz789 \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Run and associated metrics deleted successfully"
}
```

### GET /api/v1/runs/{run_id}/metrics

Get all metrics for a run.

**Request:**
```bash
curl http://localhost:3001/api/v1/runs/run-xyz789/metrics \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "metric-1",
      "run_id": "run-xyz789",
      "category": "performance",
      "name": "seconds_to_first_token",
      "display_name": "Time to First Token",
      "value_numeric": 0.025,
      "unit": "seconds"
    },
    {
      "id": "metric-2",
      "run_id": "run-xyz789",
      "category": "performance",
      "name": "token_generation_tokens_per_second",
      "display_name": "Tokens per Second",
      "value_numeric": 45.5,
      "unit": "tokens/s"
    }
  ]
}
```

---

## Metrics API

### GET /api/v1/metrics

List metrics with pagination and filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 20 | Items per page |
| `run_id` | string | - | Filter by run ID |
| `model_id` | string | - | Filter by model ID |
| `metric_name` | string | - | Filter by metric name |
| `category` | string | - | Filter by category |

**Valid Categories:** `performance`, `accuracy`, `memory`, `power`

**Request:**
```bash
curl "http://localhost:3001/api/v1/metrics?run_id=run-xyz789&category=performance" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "metric-1",
      "run_id": "run-xyz789",
      "model_id": "model-abc123",
      "category": "performance",
      "name": "seconds_to_first_token",
      "display_name": "Time to First Token",
      "value_numeric": 0.025,
      "unit": "seconds",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 5,
    "total_pages": 1
  }
}
```

### POST /api/v1/metrics

Create a new metric.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/metrics \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "run_id": "run-xyz789",
    "model_id": "model-abc123",
    "metric_name": "seconds_to_first_token",
    "value": 0.025,
    "category": "performance",
    "display_name": "Time to First Token",
    "unit": "seconds",
    "metadata": {
      "iteration": 100
    }
  }'
```

**Request Body Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string | Yes | Run ID |
| `model_id` | string | Yes | Model ID |
| `metric_name` | string | Yes | Metric identifier |
| `value` | number | Yes | Metric value |
| `category` | string | No | Metric category |
| `display_name` | string | No | Human-readable name |
| `unit` | string | No | Unit of measurement |
| `metadata` | object | No | Additional data |

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": "metric-1",
    "run_id": "run-xyz789",
    "metric_name": "seconds_to_first_token",
    "value_numeric": 0.025,
    "unit": "seconds",
    "category": "performance",
    "created_at": "2024-01-15T10:00:00Z"
  }
}
```

### GET /api/v1/metrics/aggregate

Get aggregated metrics.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | string | Filter by model |
| `run_type` | string | Filter by run type |
| `category` | string | Filter by category |
| `metric_name` | string | Filter by metric name |

**Request:**
```bash
curl "http://localhost:3001/api/v1/metrics/aggregate?category=performance&metric_name=token_generation_tokens_per_second" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "count": 50,
    "mean": 42.5,
    "min": 35.2,
    "max": 55.8,
    "std_dev": 5.3,
    "unit": "tokens/s"
  }
}
```

### GET /api/v1/metrics/trends

Get metric trends for a model.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelId` | string | - | Model ID (required) |
| `metricName` | string | - | Metric name (required) |
| `limit` | int | 100 | Maximum results |

**Request:**
```bash
curl "http://localhost:3001/api/v1/metrics/trends?modelId=model-abc123&metricName=token_generation_tokens_per_second&limit=50" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "model_id": "model-abc123",
    "metric_name": "token_generation_tokens_per_second",
    "trends": [
      {
        "run_id": "run-1",
        "value": 40.5,
        "timestamp": "2024-01-10T10:00:00Z",
        "change_percent": null
      },
      {
        "run_id": "run-2",
        "value": 42.5,
        "timestamp": "2024-01-12T10:00:00Z",
        "change_percent": 4.94
      },
      {
        "run_id": "run-3",
        "value": 45.5,
        "timestamp": "2024-01-15T10:00:00Z",
        "change_percent": 7.06
      }
    ]
  }
}
```

### GET /api/v1/metrics/compare

Compare metrics across runs.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_ids` | string[] | Comma-separated run IDs (required) |
| `categories` | string[] | Optional categories to include |

**Request:**
```bash
curl "http://localhost:3001/api/v1/metrics/compare?run_ids=run-1,run-2,run-3&categories=performance,accuracy" \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "runs": [
      {
        "run_id": "run-1",
        "build_name": "benchmark-001",
        "metrics": [
          {
            "name": "seconds_to_first_token",
            "value": 0.025,
            "unit": "seconds",
            "category": "performance"
          },
          {
            "name": "token_generation_tokens_per_second",
            "value": 45.5,
            "unit": "tokens/s",
            "category": "performance"
          }
        ]
      },
      {
        "run_id": "run-2",
        "build_name": "benchmark-002",
        "metrics": [
          {
            "name": "seconds_to_first_token",
            "value": 0.022,
            "unit": "seconds",
            "category": "performance"
          },
          {
            "name": "token_generation_tokens_per_second",
            "value": 48.2,
            "unit": "tokens/s",
            "category": "performance"
          }
        ]
      }
    ]
  }
}
```

### GET /api/v1/metrics/performance/{run_id}

Get performance metrics for a specific run.

**Request:**
```bash
curl http://localhost:3001/api/v1/metrics/performance/run-xyz789 \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "name": "seconds_to_first_token",
      "value": 0.025,
      "unit": "seconds",
      "display_name": "Time to First Token"
    },
    {
      "name": "prefill_tokens_per_second",
      "value": 1500.0,
      "unit": "tokens/s",
      "display_name": "Prefill TPS"
    },
    {
      "name": "token_generation_tokens_per_second",
      "value": 45.5,
      "unit": "tokens/s",
      "display_name": "Generation TPS"
    },
    {
      "name": "max_memory_used_gbyte",
      "value": 4.2,
      "unit": "GB",
      "display_name": "Peak Memory"
    }
  ]
}
```

---

## Import API

### POST /api/v1/import/evaluation

Receive evaluation results from lemonade-eval CLI.

**Security:** Requires `X-CLI-Signature` header with HMAC-SHA256 signature.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/import/evaluation \
  -H "Content-Type: application/json" \
  -H "X-CLI-Signature: <hmac-signature>" \
  -d '{
    "model_id": "meta-llama/Llama-3-8B-Instruct",
    "run_type": "benchmark",
    "build_name": "llama-3-8b-benchmark-001",
    "metrics": [
      {"name": "seconds_to_first_token", "value": 0.025, "unit": "seconds"},
      {"name": "token_generation_tokens_per_second", "value": 45.5, "unit": "tokens/s"}
    ],
    "config": {"iterations": 10},
    "device": "gpu",
    "backend": "ort",
    "status": "completed",
    "duration_seconds": 120.5
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "run_id": "run-xyz789",
    "status": "completed",
    "metrics_imported": 2
  },
  "message": "Evaluation imported successfully"
}
```

### POST /api/v1/import/bulk

Bulk import multiple evaluations.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/import/bulk \
  -H "Content-Type: application/json" \
  -H "X-CLI-Signature: <hmac-signature>" \
  -d '{
    "evaluations": [
      {
        "model_checkpoint": "meta-llama/Llama-3-8B-Instruct",
        "run_type": "benchmark",
        "build_name": "benchmark-001",
        "metrics": [...],
        "status": "completed"
      },
      {
        "model_checkpoint": "meta-llama/Llama-3-8B-Instruct",
        "run_type": "accuracy-mmlu",
        "build_name": "mmlu-001",
        "metrics": [...],
        "status": "completed"
      }
    ],
    "skip_duplicates": true
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "imported": 15,
    "skipped": 3,
    "failed": 0
  },
  "message": "Imported 15 evaluations, skipped 3, failed 0"
}
```

### POST /api/v1/import/yaml

Import evaluation from YAML data.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/import/yaml \
  -H "Content-Type: application/json" \
  -H "X-CLI-Signature: <hmac-signature>" \
  -d '{
    "yaml_data": {
      "model": {"name": "Test Model", "checkpoint": "..."},
      "run": {"build_name": "test-001", "run_type": "benchmark"},
      "metrics": [...]
    },
    "build_name": "test-001",
    "skip_duplicates": true
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "models_created": 1,
    "runs_imported": 1,
    "metrics_imported": 5
  },
  "message": "YAML data imported successfully"
}
```

### GET /api/v1/import/jobs

List all import jobs.

**Request:**
```bash
curl http://localhost:3001/api/v1/import/jobs \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "job_id": "job-123",
      "status": "completed",
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:05:00Z",
      "results": {
        "imported": 10,
        "skipped": 2,
        "failed": 0
      }
    }
  ]
}
```

### GET /api/v1/import/jobs/{job_id}

Get status of a specific import job.

**Request:**
```bash
curl http://localhost:3001/api/v1/import/jobs/job-123 \
  -H "Authorization: Bearer your-token"
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "job_id": "job-123",
    "status": "completed",
    "progress": 100.0,
    "message": "Import completed successfully",
    "results": {
      "imported": 10,
      "skipped": 2,
      "failed": 0
    }
  }
}
```

### POST /api/v1/import/scan

Scan a cache directory for import candidates.

**Request:**
```bash
curl -X POST http://localhost:3001/api/v1/import/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "cache_dir": "~/.cache/lemonade/"
  }'
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "candidates": [
      {
        "path": "~/.cache/lemonade/build-001/results.yaml",
        "model": "Llama-3-8B-Instruct",
        "run_type": "benchmark",
        "build_name": "build-001"
      }
    ],
    "total_found": 5
  }
}
```

---

## WebSocket API

### WS /ws/v1/evaluations

Connect for real-time evaluation updates.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | Optional run ID to subscribe to |

**Connection Example:**
```javascript
const ws = new WebSocket('ws://localhost:3001/ws/v1/evaluations?run_id=run-xyz789');

ws.onopen = () => {
  console.log('Connected to WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

**Client Messages:**

Subscribe to a run:
```json
{
  "type": "subscribe",
  "run_id": "run-xyz789"
}
```

Unsubscribe:
```json
{
  "type": "unsubscribe"
}
```

Ping (health check):
```json
{
  "type": "ping"
}
```

**Server Messages:**

Subscription confirmed:
```json
{
  "type": "subscribed",
  "run_id": "run-xyz789"
}
```

Run status update:
```json
{
  "type": "run_status",
  "run_id": "run-xyz789",
  "status": "running",
  "message": "Evaluation started",
  "data": {"progress": 0}
}
```

Metrics stream:
```json
{
  "type": "metrics_stream",
  "run_id": "run-xyz789",
  "metrics": [
    {
      "metric_name": "seconds_to_first_token",
      "value": 0.025,
      "timestamp": "2024-01-15T10:00:00Z"
    }
  ]
}
```

Progress update:
```json
{
  "type": "progress",
  "run_id": "run-xyz789",
  "progress": 50.0,
  "message": "Processing batch 5/10"
}
```

Pong (ping response):
```json
{
  "type": "pong"
}
```

### WS /ws/v1/evaluation-progress

WebSocket endpoint for CLI progress reporting.

Same interface as `/ws/v1/evaluations` but optimized for CLI client updates.

---

## OpenAPI Specification

The complete OpenAPI specification is available at:

- **Swagger UI**: `http://localhost:3001/docs`
- **ReDoc**: `http://localhost:3001/redoc`
- **JSON Spec**: `http://localhost:3001/openapi.json`

**Download OpenAPI Spec:**
```bash
curl http://localhost:3001/openapi.json -o openapi.json
```
