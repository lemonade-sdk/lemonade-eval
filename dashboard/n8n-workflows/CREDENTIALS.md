# n8n Credential Setup Guide

This guide walks you through setting up all required credentials for the UI-UX Eval Dashboard n8n workflows.

## Required Credentials

### 1. Dashboard API Credentials

**Type:** HTTP Header Auth

#### Step-by-Step Setup:

1. **Generate API Key**
   ```bash
   # Using dashboard CLI
   lemonade-dashboard api-key generate \
     --name "n8n-automation" \
     --expires-in 365d
   ```

2. **Add to n8n**
   - Open n8n UI
   - Go to **Credentials** > **Add Credential**
   - Select **HTTP Header Auth**
   - Configure:
     ```
     Name: Dashboard API
     Header Name: Authorization
     Header Value: Bearer ledash_YOUR_API_KEY_HERE
     ```
   - Click **Save**

3. **Test Connection**
   ```bash
   curl -H "Authorization: Bearer ledash_YOUR_API_KEY_HERE" \
     http://localhost:8000/api/v1/health
   ```

---

### 2. SMTP Email Credentials

**Type:** SMTP

#### Option A: Gmail SMTP

1. **Enable App Password (Gmail)**
   - Go to Google Account settings
   - Security > 2-Step Verification > App passwords
   - Generate new app password for "Mail"
   - Copy the 16-character password

2. **Add to n8n**
   - **Credentials** > **Add Credential** > **SMTP**
   - Configure:
     ```
     Name: Gmail SMTP
     Host: smtp.gmail.com
     Port: 587
     Secure: false (uses STARTTLS)
     User: your-email@gmail.com
     Password: YOUR_APP_PASSWORD (16 chars)
     ```

#### Option B: SendGrid

1. **Create SendGrid API Key**
   - Go to https://app.sendgrid.com/settings/api_keys
   - Click **Create API Key**
   - Name: "n8n-workflows"
   - Permissions: Full Access
   - Copy the API key

2. **Add to n8n**
   - **Credentials** > **Add Credential** > **SendGrid API**
   - Configure:
     ```
     Name: SendGrid API
     API Key: SG.xxxxxxxxxxxx
     ```

---

### 3. Slack Webhook Credentials

**Type:** Webhook (Custom)

#### Step-by-Step Setup:

1. **Create Slack App**
   - Go to https://my.slack.com/services/new/incoming-webhook/
   - Select workspace
   - Select channel (e.g., #evaluations)
   - Click **Add Incoming Webhooks integration**
   - Copy the Webhook URL

2. **Add to n8n**
   - **Credentials** > **Add Credential** > **Webhook**
   - Configure:
     ```
     Name: Slack Webhook
     URL: https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXX
     Authentication: None
     HTTP Method: POST
     ```

3. **Test Webhook**
   ```bash
   curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
     -H "Content-Type: application/json" \
     -d '{"text": "Test message from n8n"}'
   ```

---

### 4. Microsoft Teams Webhook Credentials

**Type:** Webhook (Custom)

#### Step-by-Step Setup:

1. **Create Teams Webhook**
   - Open Microsoft Teams
   - Go to the channel for notifications
   - Click **...** (More options) > **Connectors**
   - Find **Incoming Webhook** > **Add**
   - Configure name and image
   - Copy the Webhook URL

2. **Add to n8n**
   - **Credentials** > **Add Credential** > **Webhook**
   - Configure:
     ```
     Name: Teams Webhook
     URL: https://outlook.office.com/webhook/XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXX
     Authentication: None
     HTTP Method: POST
     ```

3. **Test Webhook**
   ```bash
   curl -X POST https://outlook.office.com/webhook/YOUR/WEBHOOK/URL \
     -H "Content-Type: application/json" \
     -d '{
       "@type": "MessageCard",
       "text": "Test message from n8n"
     }'
   ```

---

### 5. Redis Cache Credentials (Optional)

**Type:** Redis

#### Step-by-Step Setup:

1. **Install Redis** (if not already installed)
   ```bash
   # Linux/Mac
   brew install redis  # Mac
   sudo apt install redis-server  # Ubuntu

   # Windows (WSL)
   sudo apt install redis-server
   ```

2. **Configure Redis**
   ```bash
   # Edit Redis config
   sudo nano /etc/redis/redis.conf

   # Set password (optional but recommended)
   requirepass your-redis-password
   ```

3. **Add to n8n**
   - **Credentials** > **Add Credential** > **Redis**
   - Configure:
     ```
     Name: Redis Cache
     Host: localhost
     Port: 6379
     Password: your-redis-password (if set)
     Database: 0
     ```

---

## Credential Management in Workflows

### Using Credentials in Nodes

Each workflow node references credentials by name. Update the credential references if your names differ:

```json
{
  "parameters": {
    "authentication": "genericCredentialType",
    "genericAuthType": "httpHeaderAuth"
  },
  "credentials": {
    "httpHeaderAuth": {
      "id": "DASHBOARD_API",
      "name": "Dashboard API"
    }
  }
}
```

### Credential ID Mapping

If you use different credential names, update these IDs in the workflow JSON files:

| Credential Name | Used In | Workflow Nodes |
|-----------------|---------|----------------|
| Dashboard API | All workflows | HTTP Request nodes |
| Gmail SMTP / SendGrid | Notifications, Reports | Email nodes |
| Slack Webhook | Notifications, Alerts | HTTP Request nodes |
| Teams Webhook | Notifications, Alerts | HTTP Request nodes |

---

## Security Best Practices

### 1. Credential Storage

- **NEVER** commit credentials to version control
- Use n8n's built-in credential encryption
- Store backups securely

### 2. API Key Rotation

Rotate credentials regularly:
```bash
# Generate new API key
lemonade-dashboard api-key generate --name "n8n-rotation-2026"

# Update in n8n UI
# Credentials > Dashboard API > Edit > Update value

# Revoke old key
lemonade-dashboard api-key revoke --name "n8n-automation-old"
```

### 3. Access Control

- Limit n8n access to authorized users
- Use separate credentials for dev/prod environments
- Implement IP whitelisting where possible

### 4. Environment Variables

For self-hosted n8n, use environment variables:

```bash
# .env file (NOT in version control)
N8N_ENCRYPTION_KEY=your-encryption-key
N8N_CREDENTIALS_OVERWRITE='{
  "httpHeaderAuth": {
    "Dashboard API": {
      "header": "Authorization",
      "value": "Bearer ledash_xxx"
    }
  }
}'
```

---

## Troubleshooting Credentials

### HTTP 401 Unauthorized

**Symptoms:** API calls fail with 401

**Solutions:**
1. Verify API key is valid: `lemonade-dashboard api-key verify`
2. Check credential name matches in workflow
3. Ensure "Bearer " prefix is included

### SMTP Authentication Failed

**Symptoms:** Email sending fails

**Solutions:**
1. Use App Password (not regular password) for Gmail
2. Enable "Less secure apps" (if using regular SMTP)
3. Verify SMTP host and port

### Webhook Not Received

**Symptoms:** Slack/Teams notifications don't arrive

**Solutions:**
1. Test webhook URL with curl (see above)
2. Check webhook URL is copied correctly (no trailing spaces)
3. Verify channel still exists

---

## Credential Export/Import

### Export Credentials (Self-hosted)

```bash
# Export credentials to JSON
curl http://localhost:5678/api/v1/credentials \
  -H "Authorization: Bearer $N8N_API_KEY" \
  -o credentials-export.json
```

### Import Credentials

```bash
# Import from JSON
curl -X POST http://localhost:5678/api/v1/credentials/import \
  -H "Authorization: Bearer $N8N_API_KEY" \
  -H "Content-Type: application/json" \
  -d @credentials-export.json
```

---

## Credential Checklist

Before deploying workflows, verify:

- [ ] Dashboard API credential created and tested
- [ ] Email credential configured (SMTP or SendGrid)
- [ ] Slack webhook URL added
- [ ] Teams webhook URL added (optional)
- [ ] All credential names match workflow references
- [ ] Test emails sent successfully
- [ ] Test Slack message received
- [ ] Test Teams message received (if configured)
- [ ] API key expiration noted in calendar

---

## Support

For credential issues:
- n8n docs: https://docs.n8n.io/hosting/authentication/
- Dashboard API: `/docs` endpoint
- Contact: team@lemonade.ai
