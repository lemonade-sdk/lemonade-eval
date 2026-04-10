# Lemonade Eval Dashboard - Deployment Guide

Complete guide for deploying the Lemonade Eval Dashboard to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Database Migration](#database-migration)
6. [SSL/HTTPS Setup](#sslhttps-setup)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB+ for database and logs
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows Server

### Software Requirements

- Python 3.10 or higher
- PostgreSQL 14+ or SQLite (for development)
- Node.js 18+ (for frontend)
- Docker 20+ (optional, for containerized deployment)
- nginx (for reverse proxy)

---

## Production Deployment

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/lemonade/lemonade-eval.git
cd lemonade-eval/dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install backend dependencies
pip install -e .

# Install frontend dependencies
cd frontend
npm install
npm run build
```

### Step 2: Configure Environment

Create `.env` file in `dashboard/backend/`:

```bash
# Production Environment
APP_NAME=Lemonade Eval Dashboard
APP_VERSION=1.0.0
DEBUG=false

# Database (PostgreSQL for production)
DATABASE_URL=postgresql://user:password@localhost:5432/lemonade_eval

# Security
SECRET_KEY=your-super-secret-key-change-in-production
CORS_ORIGINS=https://your-domain.com

# API Configuration
API_V1_PREFIX=/api/v1
WS_V1_PREFIX=/ws/v1
```

### Step 3: Initialize Database

```bash
# Run database migrations
cd backend
alembic upgrade head
```

### Step 4: Start Backend

```bash
# Using gunicorn (recommended for production)
pip install gunicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Step 5: Serve Frontend

The frontend should be built and served via nginx (see SSL/HTTPS section).

---

## Docker Deployment

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/lemonade_eval
      - SECRET_KEY=${SECRET_KEY}
      - DEBUG=false
    depends_on:
      - db
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=lemonade_eval
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### Backend Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "app.main:app", "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

### Frontend Dockerfile

Create `frontend/Dockerfile`:

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Deploy with Docker Compose

```bash
# Set environment variables
export SECRET_KEY=$(openssl rand -hex 32)

# Build and start containers
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## Environment Configuration

### Production Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_NAME` | Application name | Lemonade Eval Dashboard | No |
| `DEBUG` | Debug mode | false | Yes |
| `DATABASE_URL` | Database connection string | - | Yes |
| `SECRET_KEY` | Secret key for sessions | - | Yes |
| `CORS_ORIGINS` | Allowed CORS origins | http://localhost:3000 | Yes |
| `API_V1_PREFIX` | API prefix | /api/v1 | No |
| `WS_V1_PREFIX` | WebSocket prefix | /ws/v1 | No |

### Frontend Environment Variables

Create `frontend/.env.production`:

```bash
VITE_API_BASE_URL=https://your-domain.com/api
VITE_WS_BASE_URL=wss://your-domain.com/ws
VITE_APP_NAME=Lemonade Eval Dashboard
VITE_APP_VERSION=1.0.0

# Polling Configuration (in seconds)
VITE_POLLING_INTERVAL_FAST=30
VITE_POLLING_INTERVAL_SLOW=15
VITE_POLLING_INTERVAL_IMPORT=2
```

---

## Database Migration

### PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql

CREATE DATABASE lemonade_eval;
CREATE USER lemonade_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lemonade_eval TO lemonade_user;
\q
```

### Run Migrations

```bash
cd backend

# Initialize migrations (first time only)
alembic init alembic

# Create new migration
alembic revision -m "initial_schema"

# Apply all migrations
alembic upgrade head

# Check current version
alembic current
```

### Migration in Production

```bash
# Before deploying new version
docker-compose exec backend alembic upgrade head

# Or if running without Docker
source venv/bin/activate
cd backend
alembic upgrade head
```

---

## SSL/HTTPS Setup

### nginx Configuration

Create `/etc/nginx/sites-available/lemonade-eval`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend
    location / {
        root /var/www/lemonade-eval;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Let's Encrypt SSL Certificate

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
# Test renewal
sudo certbot renew --dry-run
```

---

## Monitoring and Logging

### Application Logs

Logs are stored in `backend/logs/`:

- `access.log` - HTTP access logs
- `error.log` - Application errors
- `app.log` - General application logs

Configure log rotation with `/etc/logrotate.d/lemonade-eval`:

```
/var/log/lemonade-eval/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload nginx
    endscript
}
```

### Health Check Endpoint

```bash
# Check API health
curl https://your-domain.com/api/v1/health

# Check readiness
curl https://your-domain.com/api/v1/health/ready
```

### Prometheus Metrics (Optional)

Add prometheus-client to monitor metrics:

```bash
pip install prometheus-client
```

Add to `backend/app/main.py`:

```python
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)
```

Access metrics at `/metrics`.

### System Monitoring

Create systemd service `/etc/systemd/system/lemonade-eval.service`:

```ini
[Unit]
Description=Lemonade Eval Dashboard Backend
After=network.target postgresql.service

[Service]
Type=notify
User=www-data
WorkingDirectory=/var/www/lemonade-eval/backend
Environment="PATH=/var/www/lemonade-eval/venv/bin"
ExecStart=/var/www/lemonade-eval/venv/bin/gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
Restart=always

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

---

## Troubleshooting

### Common Issues

**Database Connection Errors:**

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql postgresql://user:password@localhost:5432/lemonade_eval

# Check DATABASE_URL environment variable
echo $DATABASE_URL
```

**WebSocket Connection Issues:**

1. Verify nginx proxy configuration for WebSocket upgrade headers
2. Check firewall rules allow WebSocket connections
3. Ensure backend is binding to 0.0.0.0 not localhost

**Frontend Build Errors:**

```bash
cd frontend
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
npm run build
```

**High Memory Usage:**

1. Reduce number of gunicorn workers
2. Enable database connection pooling
3. Increase server RAM or add swap space

### Debug Mode

Enable debug mode for troubleshooting (NOT in production):

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

Check logs:

```bash
tail -f backend/logs/error.log
docker-compose logs -f backend
```

---

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Set DEBUG=false
- [ ] Configure CORS_ORIGINS properly
- [ ] Use HTTPS/SSL in production
- [ ] Set strong database password
- [ ] Enable firewall (ufw/iptables)
- [ ] Regular security updates
- [ ] Backup database regularly
- [ ] Monitor access logs
- [ ] Rate limit API endpoints

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/lemonade/lemonade-eval/issues
- Documentation: https://github.com/lemonade/lemonade-eval/docs
