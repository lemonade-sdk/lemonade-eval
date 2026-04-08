# n8n Workflows Quickstart Guide

Get up and running with the UI-UX Eval Dashboard n8n automations in 15 minutes.

## Prerequisites

- [ ] n8n instance (self-hosted or cloud)
- [ ] Dashboard API access
- [ ] Email service (Gmail/SendGrid)
- [ ] Slack workspace (optional)
- [ ] Microsoft Teams (optional)

---

## 5-Minute Setup

### Step 1: Copy Environment Template

```bash
cd dashboard/n8n-workflows
cp .env.template .env
```

### Step 2: Edit Environment File

Edit `.env` and set these required values:

```bash
# Required
DASHBOARD_API_URL=http://localhost:8000
DASHBOARD_API_KEY=ledash_your-key-here

# Email (at least one method)
SMTP_HOST=smtp.gmail.com
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Optional but recommended
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
```

### Step 3: Import Workflows

**Option A: n8n UI (Recommended)**
1. Open n8n UI (http://localhost:5678)
2. Go to **Workflows** > **Import from File**
3. Select each JSON file:
   - `01-scheduled-evaluations.json`
   - `02-evaluation-notifications.json`
   - `03-anomaly-detection.json`
   - `04-weekly-monthly-reports.json`
   - `05-model-comparison.json`

**Option B: n8n CLI**
```bash
n8n import:workflow --input=01-scheduled-evaluations.json
n8n import:workflow --input=02-evaluation-notifications.json
n8n import:workflow --input=03-anomaly-detection.json
n8n import:workflow --input=04-weekly-monthly-reports.json
n8n import:workflow --input=05-model-comparison.json
```

### Step 4: Configure Credentials

For each workflow, update credential references:

1. Open workflow in n8n editor
2. Click on each HTTP Request node
3. Select your credentials from dropdown
4. Save workflow

See [CREDENTIALS.md](CREDENTIALS.md) for detailed setup.

### Step 5: Activate Workflows

1. Go to **Workflows** list
2. Toggle each workflow to **Active**
3. Verify green status indicator

---

## Testing (5 minutes)

### Test 1: Trigger Notifications Webhook

```bash
curl -X POST http://localhost:5678/webhook/evaluation-complete \
  -H "Content-Type: application/json" \
  -d '{
    "run": {"id": "test-1", "status": "completed"},
    "model": {"name": "Test Model"},
    "recipients": {"email": ["your-email@example.com"]}
  }'
```

**Expected:** Check your email for notification.

### Test 2: Run Test Script

```bash
pip install requests
python test_workflows.py --api-key your-api-key
```

**Expected:** All tests pass with green checkmarks.

---

## First Run Checklist

After setup, verify:

- [ ] All 5 workflows imported successfully
- [ ] Credentials configured for all nodes
- [ ] Workflows activated (green toggle)
- [ ] Test webhook received email
- [ ] No errors in n8n execution logs

---

## Next Steps

### Configure Scheduled Evaluations

1. Open `01-scheduled-evaluations` workflow
2. Edit **Schedule Trigger** node
3. Set desired frequency:
   - Every hour: `0 * * * *`
   - Every 6 hours: `0 */6 * * *`
   - Daily at 9 AM: `0 9 * * *`

### Configure Report Schedule

1. Open `04-weekly-monthly-reports` workflow
2. Edit **Schedule Trigger** node
3. Set:
   - Weekly: `0 9 * * 1` (Monday 9 AM)
   - Monthly: `0 9 1 * *` (1st of month)

### Set Up Anomaly Alerts

1. Open `03-anomaly-detection` workflow
2. Edit **Analyze Anomalies** code node
3. Adjust thresholds:
   ```javascript
   alert_thresholds: {
     ttsft_std_dev: 2.0,      // Z-score threshold
     tps_drop_percent: 20,    // TPS drop %
     memory_increase_percent: 15  // Memory increase %
   }
   ```

---

## Troubleshooting Quick Fixes

### Workflow Not Running

**Problem:** Workflow doesn't trigger on schedule

**Fix:**
1. Check workflow is **Active** (green toggle)
2. Verify n8n is running
3. Check n8n logs: `docker logs n8n` or journalctl

### Emails Not Sending

**Problem:** Notifications don't arrive

**Fix:**
1. Verify SMTP credentials in n8n
2. For Gmail: Use App Password, not regular password
3. Check spam folder
4. Test: `telnet smtp.gmail.com 587`

### API Errors (401/403)

**Problem:** HTTP Request nodes fail with auth errors

**Fix:**
1. Regenerate API key: `lemonade-dashboard api-key generate`
2. Update credential in n8n
3. Test: `curl -H "Authorization: Bearer KEY" http://localhost:8000/api/v1/health`

### Webhook Not Received

**Problem:** Webhook triggers return 404

**Fix:**
1. Verify n8n URL in `.env` is correct
2. Check webhook path matches workflow setting
3. Ensure workflow is active

---

## Common Customizations

### Change Email Template

1. Open workflow (e.g., `02-evaluation-notifications`)
2. Find **Format Email HTML** code node
3. Edit HTML/CSS as needed
4. Save and test

### Add New Notification Channel

1. Add HTTP Request node after notification trigger
2. Configure webhook/API for new channel
3. Add conditional check (If node)
4. Save workflow

### Modify Anomaly Thresholds

1. Open `03-anomaly-detection`
2. Find **Analyze Anomalies** code node
3. Adjust threshold values
4. Save workflow

---

## File Reference

| File | Purpose |
|------|---------|
| `01-scheduled-evaluations.json` | Auto-trigger evaluations |
| `02-evaluation-notifications.json` | Send completion notifications |
| `03-anomaly-detection.json` | Monitor & alert on anomalies |
| `04-weekly-monthly-reports.json` | Generate periodic reports |
| `05-model-comparison.json` | Compare models |
| `.env.template` | Environment variables template |
| `README.md` | Full documentation |
| `CREDENTIALS.md` | Credential setup guide |
| `ARCHITECTURE.md` | System architecture |
| `test_workflows.py` | Automated test script |

---

## Getting Help

### Documentation

- [README.md](README.md) - Complete workflow documentation
- [CREDENTIALS.md](CREDENTIALS.md) - Credential setup details
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design

### n8n Resources

- n8n Docs: https://docs.n8n.io
- Workflow Templates: https://n8n.io/workflows
- Community Forum: https://community.n8n.io

### Contact

- Email: team@lemonade.ai
- Dashboard API: http://localhost:8000/docs

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│ n8n Workflow Quick Reference                                │
├─────────────────────────────────────────────────────────────┤
│ n8n UI: http://localhost:5678                               │
│ Dashboard: http://localhost:8000                            │
│                                                             │
│ Webhook URLs:                                               │
│   - Notifications: /webhook/evaluation-complete             │
│   - Comparison: /webhook/compare-models                     │
│                                                             │
│ Schedule Defaults:                                          │
│   - Evaluations: Every hour                                 │
│   - Anomaly Check: Every 6 hours                            │
│   - Weekly Report: Monday 9 AM                              │
│                                                             │
│ Test Command:                                               │
│   python test_workflows.py --api-key YOUR_KEY               │
└─────────────────────────────────────────────────────────────┘
```

---

**Setup Time:** 15 minutes
**Test Time:** 5 minutes
**Total:** 20 minutes to full automation! 🚀
