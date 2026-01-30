# Rail-Mind Backend Deployment Guide

This document explains how to deploy the Rail-Mind backend and agents to Railway.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│    Backend      │────▶│  Detection      │
│   (Vercel/etc)  │     │   (Railway)     │     │  Agent          │
└─────────────────┘     └────────┬────────┘     │  (Railway)      │
                                 │              └─────────────────┘
                                 │
                                 ▼              ┌─────────────────┐
                        ┌────────────────┐     │  Resolution     │
                        │  Resolution    │◀────│  Agent          │
                        │  Requests      │     │  (Railway)      │
                        └────────────────┘     └─────────────────┘
```

## Environment Variables Reference

### Backend Service

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `PORT` | No | Port to listen on (Railway sets this) | `8080` |
| `AGENT_DETECTION_URL` | **Yes** | URL of detection agent | `https://detection-agent-xxx.railway.app` |
| `AGENT_RESOLUTION_URL` | **Yes** | URL of resolution agent | `https://resolution-agent-xxx.railway.app` |
| `GROQ_API_KEY` | **Yes** | Groq API key for LLM | `gsk_...` |

**Alternative:** Use a single JSON env var:
```
AGENT_MAP={"detection":"https://detection.railway.app","resolution":"https://resolution.railway.app"}
```

### Detection Agent Service

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `PORT` | No | Port to listen on | `8000` |
| `PREDICTOR_MODEL_PATH` | No | Path to XGBoost model file | `/app/models/model.pkl` |

### Resolution Agent Service

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `PORT` | No | Port to listen on | `8000` |
| `GROQ_API_KEY` | **Yes** | Groq API key for LLM judge | `gsk_...` |
| `GROQ_MODEL` | No | LLM model to use | `llama-3.3-70b-versatile` |
| `QDRANT_URL` | **Yes** | Qdrant vector DB URL | `https://xxx.qdrant.io:6333` |
| `QDRANT_API_KEY` | Conditional | Qdrant API key (for cloud) | `xxx` |

---

## Deployment Options

### Option 1: Single Railway Project with Multiple Services

Best for: Testing, small deployments

1. Create a Railway project
2. Add 3 services from the same repo:
   - Backend (root Dockerfile)
   - Detection Agent (`agents/detection_agent/Dockerfile`)
   - Resolution Agent (`agents/resolution_agent/Dockerfile`)
3. Configure inter-service communication using Railway's internal network

### Option 2: Separate Railway Projects per Service (Recommended)

Best for: Production, independent scaling

Each service gets its own Railway project:
- `railmind-backend`
- `railmind-detection-agent`
- `railmind-resolution-agent`

---

## Step-by-Step Railway Deployment

### Step 1: Deploy Detection Agent

1. **Create Railway Project**
   ```
   Go to: https://railway.app/new
   Select: "Deploy from GitHub repo"
   Choose: Your rail-mind-server repository
   ```

2. **Configure Build**
   In Railway dashboard → Settings → Build:
   ```
   Dockerfile Path: agents/detection_agent/Dockerfile
   Root Directory: agents/detection_agent
   ```

3. **Add Environment Variables**
   In Railway dashboard → Variables:
   ```
   (No required env vars for detection agent)
   ```

4. **Configure Health Check**
   In Railway dashboard → Settings:
   ```
   Health Check Path: /health
   Health Check Timeout: 60s
   ```

5. **Get Service URL**
   After deployment, note the URL:
   ```
   https://detection-agent-production-xxx.up.railway.app
   ```

### Step 2: Deploy Resolution Agent

1. **Create New Railway Project**
   Repeat project creation for resolution agent.

2. **Configure Build**
   ```
   Dockerfile Path: agents/resolution_agent/Dockerfile
   Root Directory: agents/resolution_agent
   ```

3. **Add Environment Variables**
   ```
   GROQ_API_KEY=your-groq-api-key
   GROQ_MODEL=llama-3.3-70b-versatile
   QDRANT_URL=https://your-qdrant-instance.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-api-key
   ```

4. **Configure Health Check**
   ```
   Health Check Path: /health
   Health Check Timeout: 60s
   ```

5. **Get Service URL**
   ```
   https://resolution-agent-production-xxx.up.railway.app
   ```

### Step 3: Deploy Backend

1. **Create New Railway Project**
   Create the main backend project.

2. **Configure Build**
   ```
   Dockerfile Path: Dockerfile
   Root Directory: (leave empty - use repo root)
   ```

3. **Add Environment Variables**
   ```
   AGENT_DETECTION_URL=https://detection-agent-production-xxx.up.railway.app
   AGENT_RESOLUTION_URL=https://resolution-agent-production-xxx.up.railway.app
   GROQ_API_KEY=your-groq-api-key
   ```

4. **Configure Health Check**
   ```
   Health Check Path: /health
   Health Check Timeout: 30s
   ```

---

## Branch-Based Deployment

Railway supports automatic deployments from branches. For a microservices setup:

### Branch Strategy

```
main              → Backend production
agent/detection   → Detection agent production  
agent/resolution  → Resolution agent production
```

### Railway Configuration (Per-Branch)

1. **In Detection Agent Project:**
   - Settings → Deploy → Branch: `agent/detection`
   - Build → Dockerfile Path: `agents/detection_agent/Dockerfile`

2. **In Resolution Agent Project:**
   - Settings → Deploy → Branch: `agent/resolution`
   - Build → Dockerfile Path: `agents/resolution_agent/Dockerfile`

3. **In Backend Project:**
   - Settings → Deploy → Branch: `main`
   - Build → Dockerfile Path: `Dockerfile`

### Creating Agent Branches

```bash
# Create detection agent branch
git checkout -b agent/detection
git push origin agent/detection

# Create resolution agent branch
git checkout main
git checkout -b agent/resolution
git push origin agent/resolution
```

---

## Verifying Deployment

### 1. Check Health Endpoints

```bash
# Check detection agent
curl https://detection-agent-xxx.railway.app/health

# Expected response:
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": {
    "conflict_predictor": true,
    "detection_engine": true,
    "track_fault_detector": true
  },
  "timestamp": "2024-01-30T10:00:00"
}
```

```bash
# Check resolution agent
curl https://resolution-agent-xxx.railway.app/health

# Expected response:
{
  "status": "ok",
  "components": {
    "hybrid_rag": true,
    "mathematical_solver": true,
    "llm_judge": true
  },
  "timestamp": "2024-01-30T10:00:00"
}
```

```bash
# Check backend
curl https://backend-xxx.railway.app/health

# Expected response:
{
  "status": "healthy",
  "engine_initialized": true,
  "timestamp": "2024-01-30T10:00:00"
}
```

### 2. Test Agent Communication

```bash
# Test prediction via backend
curl -X POST https://backend-xxx.railway.app/api/simulation/tick
```

---

## Troubleshooting

### Common Issues

1. **"Connection refused" to agent URLs**
   - Ensure agent services are running
   - Check URL doesn't have trailing slash
   - Verify health check passes

2. **LLM errors in resolution agent**
   - Verify `GROQ_API_KEY` is set
   - Check API key has not expired

3. **Vector DB errors**
   - Verify `QDRANT_URL` is accessible from Railway
   - Check `QDRANT_API_KEY` for cloud Qdrant

4. **Slow startup / timeout**
   - Increase health check timeout to 120s
   - Check Railway logs for loading errors

### Viewing Logs

```bash
# Railway CLI
railway logs

# Or in dashboard: Project → Service → Logs
```

---

## Cost Optimization

1. **Use Hobby Plan** ($5/month)
   - Sufficient for testing
   - 512MB RAM per service

2. **Scale Down Unused Services**
   - Detection agent: Can run on minimal resources
   - Resolution agent: Needs more RAM for LLM

3. **Use External Qdrant**
   - Qdrant Cloud free tier: 1GB storage
   - Saves Railway resources

---

## Security Checklist

- [ ] All API keys stored as Railway secrets
- [ ] HTTPS enforced (Railway default)
- [ ] Health check paths are public (ok)
- [ ] No secret logging enabled
- [ ] CORS configured for frontend domain only (in production)
