# =============================================================================
# Resolution Agent Dockerfile
# =============================================================================
# 
# Build from repo root:
#   docker build -f agents/resolution_agent/Dockerfile -t resolution-agent .
#
# CPU-only Build (Railway/cloud):
#   docker build -f agents/resolution_agent/Dockerfile \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     -t resolution-agent-cpu .
#
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY agents/resolution_agent/requirements.service.txt ./requirements.txt

# Optional: Use CPU-only torch for smaller image
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir torch --index-url $TORCH_INDEX_URL; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agents/resolution_agent/ ./agents/resolution_agent/

# Copy shared utilities (if any imports from parent agents module)
COPY agents/__init__.py ./agents/__init__.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Required env vars (set at runtime):
# GROQ_API_KEY - For LLM judge
# QDRANT_URL - For vector database
# QDRANT_API_KEY - (optional) For cloud Qdrant

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn for production
CMD ["sh", "-c", "gunicorn agents.resolution_agent.app:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
