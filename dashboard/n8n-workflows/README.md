# n8n Automation Workflows for UI-UX Eval Dashboard

This directory contains n8n workflow configurations for automating the Lemonade Eval Dashboard operations.

## Table of Contents

1. [Workflow Overview](#workflow-overview)
2. [Quick Start](#quick-start)
3. [Credential Setup](#credential-setup)
4. [Workflow Configurations](#workflow-configurations)
5. [Testing Procedures](#testing-procedures)
6. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
7. [API Reference](#api-reference)

---

## Workflow Overview

| # | Workflow | File | Purpose | Trigger |
|---|----------|------|---------|---------|
| 01 | Scheduled Evaluations | `01-scheduled-evaluations.json` | Automatically trigger model evaluations on schedule | Cron (hourly) |
| 02 | Evaluation Notifications | `02-evaluation-notifications.json` | Send Email/Slack/Teams notifications when evaluations complete | Webhook |
| 03 | Anomaly Detection | `03-anomaly-detection.json` | Monitor metrics and alert on anomalies | Cron (6-hourly) |
| 04 | Weekly/Monthly Reports | `04-weekly-monthly-reports.json` | Generate and distribute periodic reports | Cron (weekly) |
| 05 | Model Comparison | `05-model-comparison.json` | Compare models and generate comparison reports | Webhook |

---

## Quick Start

### Prerequisites

- n8n instance (self-hosted or cloud)
- Dashboard API access
- Email service (SMTP or SendGrid)
- Slack/Teams webhook URLs (optional)

### Installation Steps

1. **Import Workflows**
   ```bash
   # In n8n UI: Settings > Import > Select JSON files
   # Or use n8n CLI:
   n8n import:workflow --input=01-scheduled-evaluations.json
   ```

2. **Configure Credentials**
   - See [Credential Setup](#credential-setup) section

3. **Set Environment Variables**
   ```bash
   export DASHBOARD_API_URL="http://localhost:8000"
   export DASHBOARD_FRONTEND_URL="http://localhost:3000"
   export EVAL_CACHE_DIR="/path/to/cache"
   ```

4. **Activate Workflows**
   - Toggle each workflow to "Active" in n8n UI

---

## Credential Setup

### Required Credentials

#### 1. Dashboard API Authentication (HTTP Header Auth)

**Configuration:**
- Name: `Dashboard API`
- Header Name: `Authorization`
- Header Value: `Bearer <your-api-key>`

**Obtain API Key:**
```bash
# Generate via dashboard CLI
lemonade-dashboard api-key generate --name "n8n-automation"
```

#### 2. Email Service (SMTP or SendGrid)

**Option A: SMTP**
- Name: `SMTP Credentials`
- Host: `smtp.gmail.com` (or your provider)
- Port: `587`
- User: `your-email@gmail.com`
- Password: `<app-password>`

**Option B: SendGrid API**
- Name: `SendGrid API`
- API Key: `<sendgrid-api-key>`
- Base URL: `https://api.sendgrid.com/v3/mail/send`

#### 3. Slack Webhook

- Name: `Slack Webhook`
- Webhook URL: `https://hooks.slack.com/services/TXXXX/BXXXX/XXXX`

**Create Slack Webhook:**
1. Go to https://my.slack.com/services/new/incoming-webhook/
2. Select channel
3. Copy webhook URL

#### 4. Microsoft Teams Webhook

- Name: `Teams Webhook`
- Webhook URL: `https://outlook.office.com/webhook/XXXX`

**Create Teams Webhook:**
1. In Teams channel: Connectors > Configure
2. Add "Incoming Webhook"
3. Copy webhook URL

### Environment Variables

Add these to n8n settings or `.env` file:

```bash
# Dashboard
DASHBOARD_API_URL=http://localhost:8000
DASHBOARD_FRONTEND_URL=http://localhost:3000
EVAL_CACHE_DIR=~/.cache/lemonade/evals

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=notifications@lemonade.ai
SMTP_PASSWORD=<app-password>
EMAIL_FROM_ADDRESS=notifications@lemonade.ai
ALERT_EMAIL=team@lemonade.ai

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Teams
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...

# CLI Integration
LEMONADE_CLI_PATH=/usr/bin/lemonade-eval
```

---

## Workflow Configurations

### 01. Scheduled Evaluations

**File:** `01-scheduled-evaluations.json`

**Purpose:** Automatically trigger model evaluations based on configured schedules.

**Nodes:**
```
Schedule Trigger (hourly)
    └── Fetch Models
        └── Filter Scheduled Models
            └── Split In Batches
                ├── Create Evaluation Run
                │   └── Update Run Status
                │       └── Execute Evaluation
                │           └── Check Execution Success
                │               ├── Mark Completed → Import Results
                │               └── Mark Failed
```

**Configuration:**
- Trigger: Every hour
- Batch Processing: Yes (1 model at a time)
- Retry Logic: 3 retries with 5-10 second intervals
- Timeout: 1 hour per evaluation

**Customization:**
```json
// Modify trigger frequency
"parameters": {
  "rule": {
    "interval": [{
      "field": "hours",
      "hoursInterval": 1  // Change to 24 for daily
    }]
  }
}
```

---

### 02. Evaluation Notifications

**File:** `02-evaluation-notifications.json`

**Purpose:** Send notifications when evaluations complete via Email, Slack, and Teams.

**Trigger Webhook:**
```json
POST /webhook/evaluation-complete
{
  "run": {
    "id": "uuid",
    "status": "completed",
    "run_type": "benchmark",
    "model_id": "uuid"
  },
  "model": {
    "name": "Llama-3-8B",
    "checkpoint": "meta-llama/Llama-3-8B"
  },
  "recipients": {
    "email": ["team@lemonade.ai"],
    "slack": ["#evaluations"],
    "teams": ["Eval Team"]
  }
}
```

**Features:**
- HTML email templates with styled reports
- Slack message blocks for rich formatting
- Teams actionable cards
- Notification logging

---

### 03. Anomaly Detection

**File:** `03-anomaly-detection.json`

**Purpose:** Monitor performance metrics and detect anomalies using statistical methods.

**Detection Algorithm:**
- Z-Score analysis (threshold: 2.0 std dev)
- Trend degradation detection (10% change)
- Severity classification: Critical (>3σ), High (>2.5σ), Medium (>2σ)

**Metrics Monitored:**
- Seconds to First Token (TTFT)
- Token Generation (TPS)
- Maximum Memory Usage

**Alert Channels:**
- Slack (immediate)
- Teams (immediate)
- Email (for critical only)
- Dashboard alerts (stored in DB)

---

### 04. Weekly/Monthly Reports

**File:** `04-weekly-monthly-reports.json`

**Purpose:** Generate comprehensive evaluation reports.

**Schedule:**
- Weekly: Every Monday at 9 AM
- Monthly: First week of each month

**Report Sections:**
- Executive Summary
- Total Runs & Success Rate
- Runs by Type/Status/Device
- Top Models
- Performance Metrics
- Trend Analysis

**Distribution:**
- Email (HTML report)
- Slack summary
- Dashboard storage

---

### 05. Model Comparison

**File:** `05-model-comparison.json`

**Purpose:** Compare multiple models and generate comparison reports.

**Trigger Webhook:**
```json
POST /webhook/compare-models
{
  "model_ids": ["uuid-1", "uuid-2"],
  "categories": ["performance", "accuracy"],
  "requested_by": "user@example.com"
}
```

**Comparison Logic:**
- TTFT: Lower is better
- TPS: Higher is better
- Memory: Lower is better
- Overall winner determined by metric wins

---

## Testing Procedures

### Test Scheduled Evaluations

1. **Manual Trigger:**
   ```bash
   # In n8n UI, click "Execute Workflow"
   ```

2. **Verify API Calls:**
   ```bash
   curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/runs/recent/list?limit=5
   ```

3. **Check Logs:**
   - n8n execution logs
   - Dashboard API logs

### Test Notifications

1. **Send Test Webhook:**
   ```bash
   curl -X POST http://localhost:5678/webhook/evaluation-complete \
     -H "Content-Type: application/json" \
     -d '{
       "run": {"id": "test-123", "status": "completed"},
       "model": {"name": "Test Model"},
       "recipients": {"email": ["test@example.com"]}
     }'
   ```

2. **Verify Delivery:**
   - Check email inbox
   - Check Slack channel
   - Check Teams channel

### Test Anomaly Detection

1. **Insert Test Data:**
   ```bash
   # Add anomalous metrics via API
   curl -X POST http://localhost:8000/api/v1/metrics \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "run_id": "test-run",
       "category": "performance",
       "name": "seconds_to_first_token",
       "value_numeric": 100.0
     }'
   ```

2. **Trigger Workflow:**
   - Execute "Anomaly Detection" workflow manually

3. **Verify Alerts:**
   - Check Slack for alert message
   - Check dashboard alerts

### Test Reports

1. **Generate Test Report:**
   - Modify schedule trigger to "Manual"
   - Execute workflow

2. **Verify Output:**
   - Check email for HTML report
   - Verify dashboard storage: `/api/v1/reports`

### Test Model Comparison

1. **Send Comparison Request:**
   ```bash
   curl -X POST http://localhost:5678/webhook/compare-models \
     -H "Content-Type: application/json" \
     -d '{
       "model_ids": ["uuid-1", "uuid-2"],
       "requested_by": "test@example.com"
     }'
   ```

2. **Verify Response:**
   ```json
   {
     "comparison_id": "cmp_123456",
     "overall_winner": "Model A",
     "status": "success"
   }
   ```

---

## Monitoring and Troubleshooting

### Monitor Workflow Executions

1. **In n8n UI:**
   - Go to "Executions" tab
   - Filter by workflow name
   - Check success/failure status

2. **Via API:**
   ```bash
   # List recent executions
   curl http://localhost:5678/api/v1/executions?workflowId=<id>
   ```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| HTTP 401 | Invalid API key | Regenerate API key, update credentials |
| HTTP 404 | Wrong API URL | Check `DASHBOARD_API_URL` env var |
| Timeout | Long evaluation | Increase timeout in HTTP Request node |
| Email not sent | SMTP auth fail | Use app password, enable less secure apps |
| Slack not received | Invalid webhook | Regenerate webhook URL |
| No anomalies detected | Insufficient data | Need 3+ data points for z-score |

### Debug Mode

Enable debug logging in n8n:
```bash
export N8N_LOG_LEVEL=debug
```

### Error Handling

Each workflow includes:
- Retry logic (3 attempts)
- Error notifications
- Status tracking in dashboard

---

## API Reference

### Dashboard Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/models` | GET | List models |
| `/api/v1/models/{id}` | GET | Get model details |
| `/api/v1/runs` | GET/POST | List/create runs |
| `/api/v1/runs/{id}` | GET/PUT | Get/update run |
| `/api/v1/runs/{id}/status` | POST | Update status |
| `/api/v1/metrics` | GET/POST | List/create metrics |
| `/api/v1/metrics/trends` | GET | Get metric trends |
| `/api/v1/metrics/aggregate` | GET | Get aggregated metrics |
| `/api/v1/metrics/compare` | GET | Compare metrics |
| `/api/v1/alerts` | POST | Create alert |
| `/api/v1/reports` | POST | Store report |
| `/api/v1/comparisons` | POST | Store comparison |

### Webhook Endpoints

| Workflow | Path | Method |
|----------|------|--------|
| Evaluation Notifications | `/webhook/evaluation-complete` | POST |
| Model Comparison | `/webhook/compare-models` | POST |

---

## Export/Import Workflows

### Export from n8n

```bash
# Export single workflow
curl http://localhost:5678/api/v1/workflows/<id> \
  -H "Authorization: Bearer $N8N_API_KEY" \
  -o workflow-export.json

# Export all workflows
curl http://localhost:5678/api/v1/workflows \
  -H "Authorization: Bearer $N8N_API_KEY"
```

### Import to n8n

```bash
# Import via CLI
n8n import:workflow --input=workflow.json

# Import via API
curl -X POST http://localhost:5678/api/v1/workflows \
  -H "Authorization: Bearer $N8N_API_KEY" \
  -H "Content-Type: application/json" \
  -d @workflow.json
```

---

## Best Practices

1. **Credential Security:**
   - Use n8n credential storage, not environment variables for secrets
   - Rotate API keys regularly
   - Use separate credentials for dev/prod

2. **Error Handling:**
   - Always configure retry logic
   - Log errors to dashboard
   - Set up error notifications

3. **Performance:**
   - Use split batches for large datasets
   - Configure appropriate timeouts
   - Monitor execution times

4. **Testing:**
   - Test with small datasets first
   - Use manual triggers before enabling schedules
   - Verify all integrations after updates

---

## Support

For issues or questions:
- Check n8n documentation: https://docs.n8n.io
- Dashboard API docs: http://localhost:8000/docs
- Contact: team@lemonade.ai
