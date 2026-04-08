# Lemonade Eval Dashboard - Operations Guide

Complete guide for operating the Lemonade Eval Dashboard in production.

## Table of Contents

1. [Production Deployment Checklist](#production-deployment-checklist)
2. [Monitoring Setup](#monitoring-setup)
3. [Backup and Recovery Procedures](#backup-and-recovery-procedures)
4. [Scaling Guidelines](#scaling-guidelines)

---

## Production Deployment Checklist

### Pre-Deployment

#### Infrastructure Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **CPU** | 2 cores | 4+ cores | For backend workers |
| **RAM** | 4 GB | 8+ GB | PostgreSQL + application |
| **Storage** | 10 GB | 50+ GB | SSD recommended |
| **Network** | 100 Mbps | 1 Gbps | For API traffic |

#### Software Requirements

- [ ] Python 3.12+ installed
- [ ] Node.js 18+ installed (for frontend build)
- [ ] PostgreSQL 16+ installed and configured
- [ ] Redis 7+ installed (for caching/rate limiting)
- [ ] nginx installed (for reverse proxy)
- [ ] SSL certificate obtained
- [ ] Docker installed (if using containerized deployment)

#### Security Checklist

- [ ] SECRET_KEY generated (32+ characters, cryptographically secure)
- [ ] CLI_SECRET generated for signature verification
- [ ] Database password set (strong, unique password)
- [ ] Firewall rules configured
- [ ] CORS origins restricted to production domains
- [ ] DEBUG mode disabled
- [ ] Rate limiting enabled
- [ ] SSL/TLS configured
- [ ] Security headers configured

### Deployment Steps

#### Step 1: Prepare Environment

```bash
# Create application user
sudo useradd -r -m -d /opt/lemonade-eval -s /bin/bash lemonade
sudo usermod -aG www-data lemonade

# Create directories
sudo mkdir -p /opt/lemonade-eval/{backend,frontend,logs}
sudo chown -R lemonade:www-data /opt/lemonade-eval
sudo chmod -R 755 /opt/lemonade-eval
```

#### Step 2: Deploy Backend

```bash
cd /opt/lemonade-eval/backend

# Clone or copy code
git clone https://github.com/lemonade/lemonade-eval.git .
git checkout <release-tag>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create production .env file
cat > .env << EOF
# Application
APP_NAME=Lemonade Eval Dashboard
APP_VERSION=1.0.0
DEBUG=false

# Database
DATABASE_URL=postgresql://lemonade_user:SECURE_PASSWORD@localhost:5432/lemonade_dashboard
DATABASE_ASYNC_URL=postgresql+asyncpg://lemonade_user:SECURE_PASSWORD@localhost:5432/lemonade_dashboard

# Security
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
CLI_SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# API Configuration
API_V1_PREFIX=/api/v1
WS_V1_PREFIX=/ws/v1

# Redis
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_BURST=200

# Pagination
DEFAULT_PAGE_SIZE=20
MAX_PAGE_SIZE=100
EOF

# Secure .env file
chmod 600 .env

# Run migrations
alembic upgrade head

# Create logs directory
mkdir -p logs
chown lemonade:www-data logs
```

#### Step 3: Deploy Frontend

```bash
cd /opt/lemonade-eval/frontend

# Install dependencies
npm ci --production

# Build for production
npm run build

# Copy build to web root
sudo mkdir -p /var/www/lemonade-eval
sudo cp -r dist/* /var/www/lemonade-eval/
sudo chown -R www-data:www-data /var/www/lemonade-eval
```

#### Step 4: Configure nginx

Create `/etc/nginx/sites-available/lemonade-eval`:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;

    # HSTS (uncomment after verifying SSL works)
    # add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Frontend static files
    location / {
        root /var/www/lemonade-eval;
        index index.html;
        try_files $uri $uri/ /index.html;

        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;

        # Rate limiting (nginx level)
        limit_req zone=api burst=20 nodelay;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        proxy_pass http://127.0.0.1:8000/metrics;

        # Restrict to internal IPs
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }

    # Access and error logs
    access_log /var/log/nginx/lemonade-eval_access.log;
    error_log /var/log/nginx/lemonade-eval_error.log;
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/lemonade-eval /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### Step 5: Configure SSL

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

#### Step 6: Create Systemd Service

Create `/etc/systemd/system/lemonade-eval.service`:

```ini
[Unit]
Description=Lemonade Eval Dashboard Backend
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=notify
User=lemonade
Group=www-data
WorkingDirectory=/opt/lemonade-eval/backend
Environment="PATH=/opt/lemonade-eval/backend/venv/bin"
ExecStart=/opt/lemonade-eval/backend/venv/bin/gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --access-logfile /opt/lemonade-eval/logs/access.log \
    --error-logfile /opt/lemonade-eval/logs/error.log \
    --log-level info \
    --timeout 120 \
    --keep-alive 5
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lemonade-eval

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/lemonade-eval/logs

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lemonade-eval
sudo systemctl start lemonade-eval
sudo systemctl status lemonade-eval
```

#### Step 7: Configure Redis

```bash
# Edit Redis configuration
sudo nano /etc/redis/redis.conf

# Add/modify settings:
maxmemory 256mb
maxmemory-policy allkeys-lru
requirepass YOUR_REDIS_PASSWORD

# Restart Redis
sudo systemctl restart redis
```

Update backend `.env` with Redis password:
```
REDIS_URL=redis://:YOUR_REDIS_PASSWORD@localhost:6379/0
```

#### Step 8: Configure PostgreSQL

```bash
# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE lemonade_dashboard;
CREATE USER lemonade_user WITH PASSWORD 'SECURE_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE lemonade_dashboard TO lemonade_user;
\c lemonade_dashboard
GRANT ALL ON SCHEMA public TO lemonade_user;
EOF
```

#### Step 9: Verify Deployment

```bash
# Check health endpoint
curl -k https://your-domain.com/api/v1/health

# Check readiness
curl -k https://your-domain.com/api/v1/health/ready

# Check frontend
curl -k https://your-domain.com

# Check WebSocket
wscat -c "wss://your-domain.com/ws/v1/evaluations"
```

### Post-Deployment

#### Configure Monitoring

```bash
# Install monitoring agent (example: Prometheus node exporter)
sudo apt install prometheus-node-exporter
sudo systemctl enable prometheus-node-exporter
sudo systemctl start prometheus-node-exporter
```

#### Set Up Log Rotation

Create `/etc/logrotate.d/lemonade-eval`:

```
/opt/lemonade-eval/logs/*.log /var/log/nginx/lemonade-eval_*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 lemonade www-data
    sharedscripts
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
        systemctl kill -s HUP lemonade-eval > /dev/null 2>&1 || true
    endscript
}
```

#### Configure Backups

See [Backup and Recovery Procedures](#backup-and-recovery-procedures)

#### Document Deployment

Record:
- Deployment date and time
- Version deployed
- Server IPs and hostnames
- Configuration changes made
- Any issues encountered

---

## Monitoring Setup

### Application Metrics

#### Prometheus Metrics

The dashboard exposes Prometheus metrics at `/metrics`:

```bash
# Access metrics (restricted to internal IPs)
curl http://127.0.0.1:8000/metrics
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request duration |
| `http_requests_in_progress` | Gauge | Current requests |
| `database_connections` | Gauge | Active DB connections |
| `cache_hits_total` | Counter | Redis cache hits |
| `cache_misses_total` | Counter | Redis cache misses |
| `rate_limit_rejections_total` | Counter | Rate limited requests |

#### Prometheus Configuration

Create `/etc/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'lemonade-eval'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

### Grafana Dashboards

#### Import Dashboard

Create Grafana dashboard JSON for import:

```json
{
  "dashboard": {
    "title": "Lemonade Eval Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{path}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

Create `/etc/prometheus/alerts/lemonade-eval.yml`:

```yaml
groups:
  - name: lemonade-eval
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "p95 response time is {{ $value }}s"

      - alert: DatabaseDown
        expr: up{job="lemonade-eval"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Dashboard backend is down"

      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
```

### Log Monitoring

#### Structured Logging

The application uses structured logging with the following format:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "message": "Request processed",
  "path": "/api/v1/models",
  "method": "GET",
  "status": 200,
  "duration_ms": 45,
  "user_id": "user-123"
}
```

#### Log Aggregation

Configure rsyslog or Fluentd for log aggregation:

```bash
# Install Fluentd
sudo apt install fluentd

# Configure /etc/fluent/fluent.conf
<source>
  @type tail
  path /opt/lemonade-eval/logs/*.log
  pos_file /var/log/fluentd/lemonade.log.pos
  tag lemonade-eval.*
  <parse>
    @type json
  </parse>
</source>

<match lemonade-eval.**>
  @type elasticsearch
  host localhost
  port 9200
  index_name lemonade-eval-logs
</match>
```

### Health Checks

#### External Monitoring

Configure uptime monitoring with external services:

```yaml
# Example: Uptime Kuma configuration
{
  "name": "Lemonade Eval Dashboard",
  "type": "http",
  "url": "https://your-domain.com/api/v1/health",
  "interval": 60,
  "timeout": 10,
  "expectedStatus": 200
}
```

#### Internal Health Checks

Create `/etc/systemd/system/lemonade-eval-healthcheck.service`:

```ini
[Unit]
Description=Lemonade Eval Dashboard Health Check
After=lemonade-eval.service

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -sf http://127.0.0.1:8000/api/v1/health
User=lemonade

[Install]
WantedBy=multi-user.target
```

Create timer `/etc/systemd/system/lemonade-eval-healthcheck.timer`:

```ini
[Unit]
Description=Run health check every 5 minutes

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
Unit=lemonade-eval-healthcheck.service

[Install]
WantedBy=timers.target
```

### Performance Monitoring

#### Database Monitoring

```sql
-- Check connection count
SELECT count(*) FROM pg_stat_activity;

-- Check slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### Redis Monitoring

```bash
# Check Redis stats
redis-cli INFO

# Check memory usage
redis-cli INFO memory | grep used_memory_human

# Check connected clients
redis-cli CLIENT LIST | wc -l
```

---

## Backup and Recovery Procedures

### Backup Strategy

#### Backup Schedule

| Data Type | Frequency | Retention | Method |
|-----------|-----------|-----------|--------|
| PostgreSQL | Daily | 30 days | pg_dump |
| PostgreSQL WAL | Continuous | 7 days | WAL archiving |
| Redis | Daily | 7 days | RDB snapshot |
| Configuration | On change | All versions | Git |
| Logs | Weekly | 90 days | Log rotation |

### Database Backup

#### Full Backup Script

Create `/opt/lemonade-eval/scripts/backup-db.sh`:

```bash
#!/bin/bash

set -e

# Configuration
DB_NAME="lemonade_dashboard"
DB_USER="lemonade_user"
BACKUP_DIR="/opt/lemonade-eval/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR/postgresql"

# Create backup
pg_dump -U "$DB_USER" -h localhost -d "$DB_NAME" \
    --format=custom \
    --compress=9 \
    --verbose \
    "$BACKUP_DIR/postgresql/backup_${DATE}.dump"

# Verify backup
pg_restore -U "$DB_USER" -h localhost -d "$DB_NAME" \
    --list "$BACKUP_DIR/postgresql/backup_${DATE}.dump" > /dev/null

# Clean old backups
find "$BACKUP_DIR/postgresql" -name "backup_*.dump" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: backup_${DATE}.dump"
```

Make executable:
```bash
chmod +x /opt/lemonade-eval/scripts/backup-db.sh
```

#### Point-in-Time Recovery (PITR)

Configure WAL archiving in `/etc/postgresql/16/main/postgresql.conf`:

```conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /opt/lemonade-eval/backups/wal/%f'
archive_timeout = 300
```

### Redis Backup

Create `/opt/lemonade-eval/scripts/backup-redis.sh`:

```bash
#!/bin/bash

set -e

BACKUP_DIR="/opt/lemonade-eval/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Trigger Redis BGSAVE
redis-cli BGSAVE

# Wait for save to complete
while [ "$(redis-cli LASTSAVE)" == "$(redis-cli LASTSAVE)" ]; do
    sleep 1
done

# Copy RDB file
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis/dump_${DATE}.rdb"

# Clean old backups
find "$BACKUP_DIR/redis" -name "dump_*.rdb" -mtime +$RETENTION_DAYS -delete

echo "Redis backup completed"
```

### Automated Backups

Create systemd timer `/etc/systemd/system/lemonade-backup.timer`:

```ini
[Unit]
Description=Run Lemonade Eval backup daily

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
Unit=lemonade-backup.service

[Install]
WantedBy=timers.target
```

Create service `/etc/systemd/system/lemonade-backup.service`:

```ini
[Unit]
Description=Lemonade Eval Backup
After=postgresql.service redis.service

[Service]
Type=oneshot
User=lemonade
ExecStart=/opt/lemonade-eval/scripts/backup-db.sh
ExecStart=/opt/lemonade-eval/scripts/backup-redis.sh
```

Enable:
```bash
sudo systemctl enable lemonade-backup.timer
sudo systemctl start lemonade-backup.timer
```

### Recovery Procedures

#### Database Recovery

```bash
# Stop the application
sudo systemctl stop lemonade-eval

# Restore from backup
pg_restore -U lemonade_user -h localhost -d lemonade_dashboard \
    --clean \
    --if-exists \
    --verbose \
    /opt/lemonade-eval/backups/postgresql/backup_YYYYMMDD_HHMMSS.dump

# Start the application
sudo systemctl start lemonade-eval
```

#### Point-in-Time Recovery

```bash
# Stop PostgreSQL
sudo systemctl stop postgresql

# Restore base backup
pg_restore -U lemonade_user -h localhost -d lemonade_dashboard \
    /opt/lemonade-eval/backups/postgresql/backup_YYYYMMDD_HHMMSS.dump

# Apply WAL files
# Configure recovery.conf for PITR

# Start PostgreSQL
sudo systemctl start postgresql
```

#### Redis Recovery

```bash
# Stop Redis
sudo systemctl stop redis

# Copy backup to Redis data directory
cp /opt/lemonade-eval/backups/redis/dump_YYYYMMDD_HHMMSS.rdb \
   /var/lib/redis/dump.rdb

# Set permissions
chown redis:redis /var/lib/redis/dump.rdb

# Start Redis
sudo systemctl start redis
```

### Disaster Recovery

#### Full System Recovery

1. **Provision new server** with same specifications
2. **Restore database** from latest backup
3. **Restore Redis** from latest backup
4. **Deploy application** code from Git
5. **Restore configuration** from backup/Git
6. **Update DNS** if IP changed
7. **Verify all services** are running
8. **Run health checks** to confirm operation

#### Backup Verification

Test backups monthly:

```bash
# Test database restore in isolated environment
pg_restore --list backup_YYYYMMDD_HHMMSS.dump

# Create test database
createdb test_restore

# Restore to test database
pg_restore -d test_restore backup_YYYYMMDD_HHMMSS.dump

# Verify data
psql -d test_restore -c "SELECT COUNT(*) FROM runs;"

# Clean up
dropdb test_restore
```

### Backup Monitoring

Create alert for backup failures:

```yaml
# Prometheus alert
- alert: BackupFailed
  expr: lemonade_backup_success == 0
  for: 1h
  labels:
    severity: critical
  annotations:
    summary: "Backup failed"
    description: "Last backup failed at {{ $value }}"
```

---

## Scaling Guidelines

### Vertical Scaling

#### When to Scale Vertically

- CPU usage consistently > 70%
- Memory usage consistently > 80%
- Database query times increasing
- Single server can handle load

#### Scaling Steps

1. **Increase server resources**:
   - Add CPU cores
   - Add RAM
   - Upgrade to faster storage (SSD/NVMe)

2. **Tune application**:
   ```bash
   # Increase gunicorn workers
   --workers $(nproc * 2 + 1)

   # Increase database pool size
   # Update DATABASE_POOL_SIZE in config
   ```

3. **Tune database**:
   ```conf
   # postgresql.conf
   shared_buffers = 25% of RAM
   effective_cache_size = 75% of RAM
   work_mem = RAM / (max_connections * 3)
   ```

### Horizontal Scaling

#### When to Scale Horizontally

- Vertical scaling exhausted
- High availability required
- Geographic distribution needed
- Zero-downtime deployments needed

#### Load Balancer Setup

Configure nginx upstream:

```nginx
upstream lemonade_backend {
    least_conn;
    server 10.0.1.10:8000 weight=1;
    server 10.0.1.11:8000 weight=1;
    server 10.0.1.12:8000 weight=1 backup;
}

server {
    location /api/ {
        proxy_pass http://lemonade_backend;
        # ... other proxy settings
    }
}
```

#### Database Scaling

**Read Replicas:**

```conf
# Primary postgresql.conf
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1GB

# Replica postgresql.conf
hot_standby = on
```

**Connection Pooling (PgBouncer):**

```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
lemonade_dashboard = host=127.0.0.1 port=5432 dbname=lemonade_dashboard

[pgbouncer]
listen_port = 6432
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

Update application connection string:
```
DATABASE_URL=postgresql://user:pass@localhost:6432/lemonade_dashboard
```

### Caching Strategy

#### Redis Cache Configuration

```python
# Cache configuration
CACHE_TTL = {
    'models_list': 300,      # 5 minutes
    'runs_list': 60,         # 1 minute
    'metrics': 300,          # 5 minutes
    'user_session': 1800,    # 30 minutes
}

# Redis cluster for high availability
REDIS_CLUSTER = [
    'redis-1:6379',
    'redis-2:6379',
    'redis-3:6379',
]
```

#### Cache Invalidation

```python
# Invalidate cache on write operations
async def update_model(model_id, data):
    # Update database
    await db.update(model_id, data)

    # Invalidate cache
    await cache.delete(f"model:{model_id}")
    await cache.delete_pattern("models:*")
```

### Performance Optimization

#### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_runs_status_created ON runs(status, created_at);
CREATE INDEX idx_metrics_run_id ON metrics(run_id);
CREATE INDEX idx_models_family ON models(family);

-- Analyze tables
ANALYZE models;
ANALYZE runs;
ANALYZE metrics;

-- Vacuum regularly
VACUUM ANALYZE;
```

#### Application Optimization

```python
# Use async operations
async def get_model_runs(model_id: str) -> list:
    async with db.session() as session:
        result = await session.execute(
            select(Run).where(Run.model_id == model_id)
        )
        return result.scalars().all()

# Use connection pooling
# Configure in DATABASE_URL
# postgresql+asyncpg://user:pass@host/db?pool_size=10&max_overflow=20

# Batch operations
async def bulk_create_metrics(metrics: list):
    async with db.session() as session:
        session.add_all(metrics)
        await session.commit()
```

#### Frontend Optimization

```typescript
// Use React Query caching
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,  // 5 minutes
      cacheTime: 30 * 60 * 1000, // 30 minutes
      refetchOnWindowFocus: false,
    },
  },
});

// Lazy load components
const ComparePage = lazy(() => import('@/pages/compare/ComparePage'));
```

### Capacity Planning

#### Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Usage | > 70% | Scale up/out |
| Memory Usage | > 80% | Scale up |
| Disk Usage | > 80% | Expand storage |
| DB Connections | > 80% of max | Increase pool |
| Response Time | p95 > 2s | Optimize queries |
| Error Rate | > 1% | Investigate |

#### Scaling Calculator

```
Required instances = (Total RPS * Avg response time) / Target utilization

Example:
- Current RPS: 100
- Avg response time: 0.1s
- Target utilization: 70%

Required instances = (100 * 0.1) / 0.7 = 14.3 ≈ 15 instances
```

### High Availability

#### Multi-Region Deployment

```
┌─────────────────┐     ┌─────────────────┐
│   Region A      │     │   Region B      │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │  Load     │  │     │  │  Load     │  │
│  │ Balancer  │  │     │  │ Balancer  │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │
│  ┌─────▼─────┐  │     │  ┌─────▼─────┐  │
│  │  App      │  │     │  │  App      │  │
│  │ Servers   │  │     │  │ Servers   │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │
│  ┌─────▼─────┐  │     │  ┌─────▼─────┐  │
│  │  Primary  │◄─┼─────┼─►│  Replica  │  │
│  │    DB     │  │     │  │    DB     │  │
│  └───────────┘  │     │  └───────────┘  │
└─────────────────┘     └─────────────────┘
```

#### Failover Configuration

```yaml
# Keepalived configuration
vrrp_script check_backend {
    script "/usr/local/bin/check-backend.sh"
    interval 2
    fall 3
    rise 2
}

vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1

    virtual_ipaddress {
        10.0.0.100
    }

    track_script {
        check_backend
    }
}
```

---

## Incident Response

### Incident Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P0 | Critical | Immediate | Complete outage, data loss |
| P1 | High | 1 hour | Major feature broken |
| P2 | Medium | 4 hours | Minor feature broken |
| P3 | Low | 24 hours | Cosmetic issues |

### Incident Response Procedure

1. **Detect**: Identify the incident through monitoring/alerts
2. **Acknowledge**: Assign incident responder
3. **Assess**: Determine severity and impact
4. **Communicate**: Notify stakeholders
5. **Diagnose**: Find root cause
6. **Fix**: Implement and verify fix
7. **Document**: Record incident details
8. **Review**: Conduct post-mortem

### Runbook Examples

#### Database Connection Issues

```bash
# Check database status
sudo systemctl status postgresql

# Check connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check for locks
psql -c "SELECT * FROM pg_locks WHERE granted = false;"

# Restart if needed
sudo systemctl restart postgresql

# Check application
sudo systemctl status lemonade-eval
sudo journalctl -u lemonade-eval -n 100
```

#### High Memory Usage

```bash
# Check memory
free -h
top -p $(pgrep -f gunicorn)

# Check for memory leaks
ps aux | grep gunicorn

# Restart service
sudo systemctl restart lemonade-eval

# Monitor
watch -n 5 'free -h'
```

---

## Support and Maintenance

### Regular Maintenance Tasks

| Task | Frequency | Description |
|------|-----------|-------------|
| Security updates | Weekly | Apply OS and package updates |
| Database vacuum | Weekly | Run VACUUM ANALYZE |
| Log cleanup | Monthly | Remove old logs |
| Backup test | Monthly | Verify backup restoration |
| Performance review | Monthly | Review metrics and optimize |
| Dependency updates | Monthly | Update Python and npm packages |

### Contact Information

- **GitHub Issues**: https://github.com/lemonade/lemonade-eval/issues
- **Documentation**: https://github.com/lemonade/lemonade-eval/docs
- **Emergency Contact**: See internal documentation
