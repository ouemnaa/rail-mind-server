# Rail-Mind Railway Deployment - Complete Step-by-Step Guide

This guide provides detailed, step-by-step instructions for deploying Rail-Mind microservices to Railway and ensuring proper communication between agents.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Step 1: Prepare Your Repository](#step-1-prepare-your-repository)
4. [Step 2: Deploy Detection Agent](#step-2-deploy-detection-agent)
5. [Step 3: Deploy Resolution Agent](#step-3-deploy-resolution-agent)
6. [Step 4: Deploy Backend](#step-4-deploy-backend)
7. [Step 5: Configure Inter-Service Communication](#step-5-configure-inter-service-communication)
8. [Step 6: Verify Deployment](#step-6-verify-deployment)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Topics](#advanced-topics)

---

## Prerequisites

Before starting, ensure you have:

- [ ] **GitHub Repository** - Your rail-mind-server code pushed to GitHub
- [ ] **Railway Account** - Sign up at [railway.app](https://railway.app)
- [ ] **Groq API Key** - Get from [console.groq.com](https://console.groq.com)
- [ ] **Qdrant Instance** - Either:
  - Qdrant Cloud (free tier): [cloud.qdrant.io](https://cloud.qdrant.io)
  - Or self-hosted on Railway

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAILWAY PLATFORM                              │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │  Backend         │    │  Detection       │    │  Resolution   │ │
│  │  Service         │───▶│  Agent           │    │  Agent        │ │
│  │                  │    │                  │    │               │ │
│  │  Port: $PORT     │    │  Port: $PORT     │    │  Port: $PORT  │ │
│  │                  │───▶│                  │    │               │ │
│  └────────┬─────────┘    └──────────────────┘    └───────────────┘ │
│           │                                              │          │
│           └──────────────────────────────────────────────┘          │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │  Qdrant          │◀───│  Resolution      │                       │
│  │  (Optional)      │    │  Agent           │                       │
│  └──────────────────┘    └──────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Frontend           │
                    │  (Vercel/Netlify)   │
                    └─────────────────────┘
```

**Communication Flow:**
1. Frontend calls Backend API
2. Backend calls Detection Agent for predictions/detections
3. Backend calls Resolution Agent for conflict resolution
4. Resolution Agent uses Qdrant for vector search
5. Resolution Agent uses Groq for LLM responses

---

## Step 1: Prepare Your Repository

### 1.1 Ensure Branch Structure

For production deployments, consider using branch-based deployments:

```bash
# Ensure you have the agent branches
git checkout main
git checkout -b agent/detection
git push origin agent/detection

git checkout main
git checkout -b agent/resolution
git push origin agent/resolution
```

### 1.2 Verify Dockerfiles Exist

Your repository should have:
```
rail-mind-server/
├── Dockerfile                           # Backend
├── agents/
│   ├── detection_agent/
│   │   └── Dockerfile                   # Detection Agent
│   └── resolution_agent/
│       └── Dockerfile                   # Resolution Agent
```

### 1.3 Create .env.example

```bash
# Create a template for required variables
cat > .env.example << 'EOF'
# Backend
AGENT_DETECTION_URL=https://detection-agent.railway.app
AGENT_RESOLUTION_URL=https://resolution-agent.railway.app
GROQ_API_KEY=gsk_your_key_here

# Resolution Agent
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
QDRANT_URL=https://your-qdrant.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key

# Detection Agent
# (no required env vars)
EOF
```

---

## Step 2: Deploy Detection Agent

### 2.1 Create New Project

1. Go to [railway.app/dashboard](https://railway.app/dashboard)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your repository
5. Select your `rail-mind-server` repository

### 2.2 Configure Service

After the project is created:

1. Click on the service card
2. Go to **Settings** tab
3. Configure the following:

**Source:**
```
Branch: main (or agent/detection if using branches)
Watch paths: agents/detection_agent/**
```

**Build:**
```
Builder: Dockerfile
Dockerfile Path: agents/detection_agent/Dockerfile
Root Directory: . (leave empty)
```

> **Important:** Railway builds from the repository root. The Dockerfile uses paths like `COPY . ./agents/detection_agent/` which assumes build context is the agent directory. You may need to adjust the Dockerfile or specify `Root Directory: agents/detection_agent`

### 2.3 Configure Networking

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Note the generated URL (e.g., `detection-agent-production-abc123.up.railway.app`)

### 2.4 Configure Health Check

1. Go to **Settings** → **Healthcheck**
2. Set:
   ```
   Path: /health
   Timeout: 120 seconds
   Interval: 30 seconds
   ```

### 2.5 Deploy

1. Click **"Deploy"** button
2. Wait for build to complete (may take 5-10 minutes first time)
3. Check logs for successful startup

### 2.6 Verify Deployment

```bash
# Replace with your actual URL
curl https://detection-agent-production-abc123.up.railway.app/health
```

Expected response:
```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": {
    "conflict_predictor": true,
    "detection_engine": true,
    "track_fault_detector": true
  }
}
```

**📝 Note:** Save this URL - you'll need it for the backend configuration.

---

## Step 3: Deploy Resolution Agent

### 3.1 Create New Project

1. Go to [railway.app/dashboard](https://railway.app/dashboard)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Select the same `rail-mind-server` repository

### 3.2 Configure Service

**Settings → Source:**
```
Branch: main (or agent/resolution if using branches)
Watch paths: agents/resolution_agent/**
```

**Settings → Build:**
```
Builder: Dockerfile
Dockerfile Path: agents/resolution_agent/Dockerfile
Root Directory: agents/resolution_agent
```

### 3.3 Add Environment Variables

1. Go to **Variables** tab
2. Add the following variables:

| Variable | Value | Notes |
|----------|-------|-------|
| `GROQ_API_KEY` | `gsk_xxxx` | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Optional, this is default |
| `QDRANT_URL` | `https://xxx.qdrant.io:6333` | Your Qdrant URL |
| `QDRANT_API_KEY` | `xxxx` | Your Qdrant API key |

**To add a variable:**
1. Click **"New Variable"**
2. Enter the key and value
3. Click **"Add"**

### 3.4 Configure Networking

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Note the URL (e.g., `resolution-agent-production-xyz789.up.railway.app`)

### 3.5 Configure Health Check

```
Path: /health
Timeout: 120 seconds
Interval: 30 seconds
```

### 3.6 Deploy and Verify

```bash
curl https://resolution-agent-production-xyz789.up.railway.app/health
```

Expected:
```json
{
  "status": "ok",
  "components": {
    "hybrid_rag": true,
    "mathematical_solver": true,
    "llm_judge": true
  }
}
```

**📝 Note:** Save this URL for backend configuration.

---

## Step 4: Deploy Backend

### 4.1 Create New Project

1. Create another new project from the same repository
2. This will be the main backend/API service

### 4.2 Configure Service

**Settings → Build:**
```
Builder: Dockerfile
Dockerfile Path: Dockerfile
Root Directory: (leave empty)
```

### 4.3 Add Environment Variables

| Variable | Value | Notes |
|----------|-------|-------|
| `AGENT_DETECTION_URL` | `https://detection-agent-production-abc123.up.railway.app` | From Step 2 |
| `AGENT_RESOLUTION_URL` | `https://resolution-agent-production-xyz789.up.railway.app` | From Step 3 |
| `GROQ_API_KEY` | `gsk_xxxx` | For direct LLM calls |

**Alternative:** Use a single JSON env var:
```
AGENT_MAP={"detection":"https://detection-agent.railway.app","resolution":"https://resolution-agent.railway.app"}
```

### 4.4 Configure Networking

Generate a domain for the backend. This will be your public API URL.

### 4.5 Deploy and Verify

```bash
curl https://backend-production-xxx.up.railway.app/health
```

---

## Step 5: Configure Inter-Service Communication

### 5.1 Understanding Railway Networking

Railway provides two ways for services to communicate:

1. **Public URLs** (HTTPS)
   - Use the generated domain URLs
   - Works across different Railway projects
   - Has slight latency overhead

2. **Private Networking** (Same Project)
   - Use internal URLs like `detection_agent.railway.internal`
   - Only works within the same Railway project
   - Lower latency, more secure

### 5.2 Option A: Separate Projects (Recommended)

If each service is in its own Railway project, use public URLs:

```
# In Backend project variables:
AGENT_DETECTION_URL=https://detection-agent-production-abc123.up.railway.app
AGENT_RESOLUTION_URL=https://resolution-agent-production-xyz789.up.railway.app
```

### 5.3 Option B: Same Project with Private Networking

If all services are in the same Railway project:

1. Enable Private Networking in project settings
2. Use internal URLs:

```
# In Backend project variables:
AGENT_DETECTION_URL=http://detection_agent.railway.internal:8000
AGENT_RESOLUTION_URL=http://resolution_agent.railway.internal:8000
```

### 5.4 Environment Variable Summary

**Backend Service:**
```env
# Required
AGENT_DETECTION_URL=https://detection-agent-xxx.up.railway.app
AGENT_RESOLUTION_URL=https://resolution-agent-xxx.up.railway.app
GROQ_API_KEY=gsk_your_key

# Optional
PORT=8080  # Railway sets this automatically
```

**Detection Agent:**
```env
# No required variables
# Optional:
PREDICTOR_MODEL_PATH=/app/models/xgboost.pkl
```

**Resolution Agent:**
```env
# Required
GROQ_API_KEY=gsk_your_key
QDRANT_URL=https://xxx.qdrant.io:6333
QDRANT_API_KEY=your_key

# Optional
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## Step 6: Verify Deployment

### 6.1 Check All Health Endpoints

Create a verification script:

```bash
#!/bin/bash
# verify_deployment.sh

BACKEND_URL="https://your-backend.up.railway.app"
DETECTION_URL="https://your-detection.up.railway.app"
RESOLUTION_URL="https://your-resolution.up.railway.app"

echo "Checking Detection Agent..."
curl -s "$DETECTION_URL/health" | jq .

echo -e "\nChecking Resolution Agent..."
curl -s "$RESOLUTION_URL/health" | jq .

echo -e "\nChecking Backend..."
curl -s "$BACKEND_URL/health" | jq .

echo -e "\nAll services checked!"
```

### 6.2 Test End-to-End Flow

**Test Simulation Tick (triggers detection agent):**
```bash
curl -X GET https://your-backend.up.railway.app/api/simulation/tick | jq .
```

**Test Resolution (triggers resolution agent):**
```bash
curl -X POST https://your-backend.up.railway.app/api/conflicts/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "conflict": {
      "conflict_id": "TEST-001",
      "conflict_type": "schedule_conflict",
      "train_ids": ["REG1", "IC2"],
      "station_ids": ["MILANO_CENTRALE"],
      "severity": 0.7
    }
  }' | jq .
```

### 6.3 Check Logs

In Railway dashboard:
1. Go to your project
2. Click on a service
3. Go to **"Logs"** tab
4. Look for any errors

---

## Troubleshooting

### Issue: "Connection Refused" from Backend to Agents

**Symptoms:**
- Backend health shows ok
- Agent URLs return "connection refused"

**Solutions:**
1. Verify URLs are correct (no typos, no trailing slash)
2. Check agent services are running (green checkmark in Railway)
3. Ensure domains are generated and active

### Issue: LLM Errors in Resolution Agent

**Symptoms:**
- `/health` shows `llm_judge: false`
- Resolution requests fail with LLM errors

**Solutions:**
1. Verify `GROQ_API_KEY` is set correctly
2. Check API key is valid at console.groq.com
3. Check for rate limiting (free tier has limits)

### Issue: Qdrant Connection Errors

**Symptoms:**
- `/health` shows `hybrid_rag: false`
- Connection timeout errors in logs

**Solutions:**
1. Verify `QDRANT_URL` is correct and accessible
2. For Qdrant Cloud: ensure `QDRANT_API_KEY` is set
3. Test Qdrant manually: `curl $QDRANT_URL/health`

### Issue: Slow Startup / Health Check Timeout

**Symptoms:**
- Service marked as unhealthy
- Deploy fails after build

**Solutions:**
1. Increase health check timeout to 180 seconds
2. Check service logs for loading errors
3. Verify all dependencies are available

### Issue: Memory Errors / Container Killed

**Symptoms:**
- Service restarts frequently
- "OOMKilled" in logs

**Solutions:**
1. Use CPU-only torch (smaller memory footprint)
2. Upgrade Railway plan for more memory
3. Reduce model sizes or disable unused features

---

## Advanced Topics

### Using Branch-Based Deployments

For cleaner separation, create branches per service:

```bash
# Create and push branches
git checkout -b agent/detection && git push origin agent/detection
git checkout main
git checkout -b agent/resolution && git push origin agent/resolution
git checkout main
```

Configure each Railway service to deploy from its respective branch.

### Setting Up CI/CD

Create `.github/workflows/railway.yml`:

```yaml
name: Deploy to Railway

on:
  push:
    branches: [main, agent/detection, agent/resolution]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # Railway automatically deploys on push
      # This workflow is for additional checks
      
      - name: Health Check
        run: |
          sleep 120  # Wait for Railway deploy
          curl -f ${{ secrets.BACKEND_URL }}/health
```

### Scaling Services

In Railway dashboard:
1. Go to service settings
2. Under **"Scaling"**, configure:
   - **Replicas**: Number of instances
   - **Auto-scaling**: Based on CPU/memory

### Monitoring and Alerting

1. Enable **Railway Metrics** in project settings
2. Set up **Observability** integrations (Datadog, etc.)
3. Configure **Healthcheck Notifications**

---

## Summary Checklist

Before going live, verify:

- [ ] Detection Agent deployed and health check passes
- [ ] Resolution Agent deployed with Groq and Qdrant configured
- [ ] Backend deployed with correct agent URLs
- [ ] All health endpoints return "ok" status
- [ ] End-to-end test (simulation tick + resolution) works
- [ ] Frontend configured with backend URL
- [ ] Error logs are clean
- [ ] Health check notifications enabled

---

## Support

If you encounter issues not covered here:

1. Check Railway status: [status.railway.app](https://status.railway.app)
2. Railway Discord: [discord.gg/railway](https://discord.gg/railway)
3. Project issues: Create a GitHub issue
