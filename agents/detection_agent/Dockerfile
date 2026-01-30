# =============================================================================
# Detection Agent Dockerfile
# =============================================================================
# 
# Build from repo root:
#   docker build -f agents/detection_agent/Dockerfile -t detection-agent .
#
# GPU Build:
#   docker build -f agents/detection_agent/Dockerfile \
#     --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime \
#     -t detection-agent-gpu .
#
# CPU-only Build (Railway/cloud):
#   docker build -f agents/detection_agent/Dockerfile \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     -t detection-agent-cpu .
#
# =============================================================================

ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY agents/detection_agent/requirements.service.txt ./requirements.txt

# For CPU builds, install CPU-only torch
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir torch torchvision --index-url $TORCH_INDEX_URL; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Download YOLOv8n model (small, ~6MB) - will be downloaded if not present
# The ultralytics package auto-downloads this, but we cache it for faster startup
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO model will be downloaded at runtime"

# Copy agent code
COPY agents/detection_agent/ ./agents/detection_agent/

# Copy shared utilities (if any imports from parent agents module)
COPY agents/__init__.py ./agents/__init__.py

# Try to copy YOLOv8 model if it exists in the repo (optional)
# If not present, the model will be downloaded at runtime by ultralytics
# Note: Using a wildcard pattern that won't fail if file doesn't exist
COPY yolov8n.p[t] ./

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn for production
CMD ["sh", "-c", "gunicorn agents.detection_agent.app:app -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
