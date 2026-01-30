# Detection Agent Microservice

This is the Detection Agent microservice for Rail-Mind. It provides:

- **Conflict Prediction** - XGBoost ML model for predicting railway conflicts
- **Deterministic Detection** - Rule-based real-time conflict detection
- **Track Fault Detection** - YOLOv8 vision model for detecting track defects

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Run conflict prediction on network state |
| `/detect` | POST | Run deterministic conflict detection |
| `/vision/detect` | POST | Detect track faults in single image |
| `/vision/batch` | POST | Detect track faults in folder |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `8000` | Port to listen on |
| `PREDICTOR_MODEL_PATH` | No | Auto-discover | Path to XGBoost model |

## Running Locally

### Without Docker

```bash
# From repo root
cd agents/detection_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.service.txt

# Run server
python app.py
```

Server will be available at `http://localhost:8001`

### With Docker

```bash
# From agent directory
docker build -t detection-agent .
docker run -p 8001:8000 detection-agent

# Or from repo root with compose
docker-compose up detection_agent
```

## API Examples

### Health Check

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": {
    "conflict_predictor": true,
    "detection_engine": true,
    "track_fault_detector": true
  },
  "timestamp": "2024-01-30T10:00:00"
}
```

### Run Prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "network_state": {
      "trains": [
        {
          "train_id": "REG123",
          "train_type": "REG",
          "current_station": "MI_CENTRALE",
          "next_station": "MI_GARIBALDI",
          "position_km": 0.5,
          "speed_kmh": 80,
          "delay_seconds": 120
        }
      ],
      "stations": [],
      "peak_hour": true
    }
  }'
```

### Run Deterministic Detection

```bash
curl -X POST http://localhost:8001/detect \
  -H "Content-Type: application/json" \
  -d '{
    "trains": [
      {
        "train_id": "REG123",
        "position_km": 10.5,
        "speed_kmh": 0,
        "current_station": "MI_CENTRALE",
        "delay_seconds": 600
      }
    ]
  }'
```

### Vision Detection

```bash
curl -X POST http://localhost:8001/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/app/agents/detection_agent/vision_track_fault/images/track1.jpg",
    "location": "MILANO--PAVIA"
  }'
```

## Docker Build Options

### CPU Build (default, smaller)

```bash
docker build -t detection-agent \
  --build-arg BASE_IMAGE=python:3.11-slim \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  .
```

### GPU Build

```bash
docker build -t detection-agent-gpu \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime \
  .
```

## Architecture

```
detection_agent/
├── app.py                     # FastAPI server
├── requirements.service.txt   # Dependencies
├── Dockerfile                 # Container build
├── prediction_conflict/       # XGBoost predictor
│   ├── predictor.py          
│   ├── feature_engine.py     
│   └── config.py             
├── deterministic_detection/   # Rule-based engine
│   ├── engine.py             
│   ├── rules.py              
│   └── models.py             
└── vision_track_fault/        # YOLOv8 detector
    ├── model.py              
    └── images/               
```
