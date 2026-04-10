# n8n Workflow Architecture

This document describes the architecture and data flow of the n8n automation workflows for the UI-UX Eval Dashboard.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UI-UX Eval Dashboard                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Models     │  │     Runs     │  │   Metrics    │  │   Reports   │ │
│  │     API      │  │     API      │  │     API      │  │    API      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/REST
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            n8n Automation                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Workflow Triggers                             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │  Cron   │  │ Webhook │  │  Cron   │  │  Cron   │  │Webhook │ │   │
│  │  │Schd Eval│  │ Notify  │  │ Anomaly │  │ Reports │  │Compare │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Processing Nodes                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │  HTTP   │  │  Code   │  │  Split  │  │   If    │  │ Merge  │ │   │
│  │  │ Request │  │Transform│  │ Batches │  │Condition│  │        │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   Output Actions                                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │  Email  │  │  Slack  │  │  Teams  │  │   DB    │  │  File  │ │   │
│  │  │  (SMTP) │  │Webhook  │  │Webhook  │  │  Store  │  │ Export │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        External Services                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │    Gmail/    │  │    Slack     │  │  Microsoft   │  │   Lemonade  │ │
│  │   SendGrid   │  │   App        │  │    Teams     │  │     CLI     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Data Flows

### 1. Scheduled Evaluations Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Schedule  │────▶│   Fetch     │────▶│  Filter Models  │
│   Trigger   │     │   Models    │     │  (Scheduled)    │
│  (Hourly)   │     │  (API GET)  │     │  (Code Node)    │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Import    │◀────│  Execute    │◀────│  Create Run     │
│   Results   │     │Evaluation   │     │  (API POST)     │
│  (API POST) │     │  (CLI)      │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              │
                    ┌─────────────────────────┴────────┐
                    ▼                                  ▼
            ┌───────────────┐                 ┌───────────────┐
            │  Success      │                 │   Failure     │
            │  (Completed)  │                 │   (Failed)    │
            └───────────────┘                 └───────────────┘
```

**Data Structure:**
```json
{
  "model_id": "uuid",
  "model_name": "Llama-3-8B",
  "checkpoint": "meta-llama/Llama-3-8B",
  "eval_frequency": "daily",
  "eval_types": ["benchmark", "accuracy-mmlu"],
  "run_id": "uuid",
  "status": "completed|failed",
  "duration_seconds": 1200
}
```

---

### 2. Evaluation Notifications Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Webhook   │────▶│   Parse     │────▶│  Check Status   │
│   Trigger   │     │   Data      │     │  (If Node)      │
│  (POST /    │     │  (Code)     │     │  completed?     │
│   eval-     │     │             │     │                 │
│   complete) │     │             │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              │
                    ┌─────────────────────────┴────────┐
                    ▼ (completed)                      ▼ (failed)
            ┌───────────────┐                 ┌───────────────┐
            │ Format Email  │                 │  Send Slack   │
            │   (HTML)      │                 │  (Failure)    │
            └───────────────┘                 └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  Check Slack  │
            │  Configured?  │
            └───────────────┘
                    │
        ┌───────────┴───────────┐
        ▼ (yes)                 ▼ (no)
┌───────────────┐         ┌───────────────┐
│  Send Slack   │         │ Check Teams   │
│  (Success)    │         │ Configured?   │
└───────────────┘         └───────────────┘
```

**Webhook Payload:**
```json
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
    "email": ["team@example.com"],
    "slack": ["#evaluations"],
    "teams": ["Eval Team"]
  }
}
```

---

### 3. Anomaly Detection Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Schedule  │────▶│   Fetch     │────▶│  Extract Models │
│   Trigger   │     │   Models    │     │  (Code Node)    │
│ (6-hourly)  │     │             │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Fetch TTFT │────▶│  Fetch TPS  │────▶│ Fetch Memory    │
│  Trends     │     │  Trends     │     │ Trends          │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Log Alert  │◀────│  Send Alert │◀────│  Analyze        │
│  (API)      │     │  (Slack/    │     │  Anomalies      │
│             │     │   Email)    │     │  (Z-Score)      │
└─────────────┘     └─────────────┘     └─────────────────┘
```

**Anomaly Detection Algorithm:**
```
For each metric:
  1. Calculate mean and std_dev from historical data
  2. Compute z-score for latest value: z = |x - mean| / std_dev
  3. Classify severity:
     - z > 3.0: CRITICAL
     - z > 2.5: HIGH
     - z > 2.0: MEDIUM
  4. Check trend degradation (last 5 vs previous 5)
  5. Alert if z > threshold OR degradation > 10%
```

---

### 4. Report Generation Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Schedule  │────▶│  Determine  │────▶│  Fetch Run      │
│   Trigger   │     │  Report     │     │  Stats          │
│ (Mon 9AM)   │     │  Type       │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Fetch Runs   │ │  Fetch Models │ │  Fetch        │
│  (Recent)     │ │  (List)       │ │  Aggregates   │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                  ┌───────────────┐
                  │  Aggregate    │
                  │  Report Data  │
                  └───────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Generate     │ │  Send Email   │ │  Send Slack   │
│  HTML Report  │ │  (Report)     │ │  (Summary)    │
└───────────────┘ └───────────────┘ └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Store Report │
                  │  (API)        │
                  └───────────────┘
```

---

### 5. Model Comparison Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Webhook   │────▶│   Parse     │────▶│  Fetch Models   │
│   Trigger   │     │   Request   │     │  (1, 2, 3...)   │
│  (POST /    │     │             │     │                 │
│   compare-  │     │             │     │                 │
│   models)   │     │             │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Respond to │◀────│   Store     │◀────│  Compare        │
│  Webhook    │     │ Comparison  │     │  Metrics        │
│             │     │  (API)      │     │  (Code)         │
└─────────────┘     └─────────────┘     └─────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   Generate    │
                  │  Comparison   │
                  │    Report     │
                  └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Email Report  │
                  │ (Optional)    │
                  └───────────────┘
```

**Comparison Logic:**
```
For each metric:
  - TTFT (Seconds to First Token): Lower is better
  - TPS (Tokens Per Second): Higher is better
  - Memory Usage: Lower is better

Winner determination:
  - Count metric wins for each model
  - Model with most wins = Overall winner
  - Tie if equal wins
```

---

## Node Types Reference

| Node Type | Purpose | Used In |
|-----------|---------|---------|
| Schedule Trigger | Time-based triggers | 01, 03, 04 |
| Webhook | HTTP endpoint triggers | 02, 05 |
| HTTP Request | API calls | All |
| Code | Data transformation | All |
| Split In Batches | Loop processing | 01, 03 |
| If | Conditional branching | 01, 02, 03 |
| Merge | Combine data streams | 04, 05 |
| Respond to Webhook | Send HTTP response | 05 |

---

## Error Handling Strategy

### Retry Configuration

```json
{
  "retryOnFail": {
    "value": true,
    "maxRetries": 3,
    "retryInterval": 5000
  }
}
```

### Error Paths

Each workflow has dedicated error handling:
- **Success path**: Continue normal processing
- **Failure path**: Log error, send failure notification, update status

### Fallback Mechanisms

| Failure | Fallback |
|---------|----------|
| Email failed | Log to dashboard, try Slack |
| Slack failed | Try Teams, log to dashboard |
| API timeout | Retry 3x, then alert |
| CLI execution fail | Mark run as failed, notify |

---

## Performance Considerations

### Batch Processing

- Split large datasets into batches of 1
- Process sequentially to avoid rate limits
- Track progress with split batches node

### Timeout Settings

| Operation | Timeout |
|-----------|---------|
| API calls | 30 seconds |
| Email sending | 30 seconds |
| Webhook calls | 15 seconds |
| CLI execution | 3600 seconds (1 hour) |

### Concurrency

- n8n default: 5 concurrent executions
- Adjust based on API rate limits
- Use queue mode for high volume

---

## Scaling Recommendations

### For High Volume

1. **Enable n8n queue mode**
   ```bash
   export N8N_EXECUTIONS_MODE=queue
   ```

2. **Add worker nodes**
   ```bash
   n8n worker --concurrency=10
   ```

3. **Implement caching**
   - Cache model lists (5 min TTL)
   - Cache aggregate metrics (15 min TTL)

4. **Rate limiting**
   - Dashboard API: 100 req/min
   - Slack: 1 req/sec
   - Email: 100/day (SendGrid free tier)

---

## Security Architecture

### Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   n8n       │────▶│   API       │────▶│   Dashboard │
│   Workflow  │     │   Gateway   │     │     API     │
│             │     │             │     │             │
│  Credentials│     │  Validates  │     │   Processes │
│  (Stored    │     │  Bearer     │     │   Request   │
│  Encrypted) │     │  Token      │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Data Protection

- All credentials encrypted at rest (AES-256)
- TLS 1.3 for all API communications
- API keys rotated every 90 days
- Webhook payloads signed (HMAC-SHA256)

---

## Monitoring Integration

### Execution Metrics

Track these metrics for each workflow:
- Execution count
- Success rate
- Average duration
- Error rate by node type

### Alerting

Configure alerts for:
- Workflow failure rate > 5%
- Execution time > 2x average
- API rate limit approaching
- Credential expiration (30 days warning)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-07 | Initial architecture |

---

## Support

For architecture questions:
- Review workflow JSON files
- Check execution logs in n8n
- Contact: team@lemonade.ai
