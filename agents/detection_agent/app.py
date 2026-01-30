"""
Detection Agent FastAPI Server
==============================

This microservice exposes the detection agent's capabilities:
1. Conflict Prediction (XGBoost ML model)
2. Deterministic Conflict Detection (Rule-based)
3. Track Fault Detection (YOLOv8 Vision model)

Endpoints:
- GET /health - Health check with device info
- POST /predict - Run conflict prediction
- POST /detect - Run deterministic detection
- POST /vision/detect - Run track fault detection
"""

import os
import sys
import torch
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global models (loaded once at startup)
conflict_predictor = None
detection_engine = None
track_fault_detector = None
device = None


# =============================================================================
# Request/Response Models
# =============================================================================

class TrainStateInput(BaseModel):
    """Train state for prediction."""
    train_id: str
    train_type: str = "REG"
    current_station: Optional[str] = None
    next_station: Optional[str] = None
    position_km: float = 0.0
    speed_kmh: float = 80.0
    delay_seconds: float = 0.0
    scheduled_departure: Optional[str] = None
    actual_departure: Optional[str] = None


class NetworkStateInput(BaseModel):
    """Network state for batch prediction."""
    trains: List[TrainStateInput] = Field(default_factory=list)
    stations: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    current_time: Optional[str] = None
    peak_hour: bool = False
    weather: str = "clear"


class PredictRequest(BaseModel):
    """Request for conflict prediction."""
    network_state: NetworkStateInput
    force: bool = False
    horizon_minutes: Optional[int] = 30


class DetectRequest(BaseModel):
    """Request for deterministic detection."""
    trains: List[Dict[str, Any]]
    stations: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    current_time: Optional[str] = None


class VisionDetectRequest(BaseModel):
    """Request for track fault vision detection."""
    image_path: str
    location: str = "UNKNOWN"


class BatchVisionRequest(BaseModel):
    """Batch request for track fault detection."""
    folder_path: str
    location: str = "UNKNOWN"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    device: str
    models_loaded: Dict[str, bool]
    timestamp: str


# =============================================================================
# Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup on shutdown."""
    global conflict_predictor, detection_engine, track_fault_detector, device
    
    print("\n" + "=" * 60)
    print("DETECTION AGENT - Starting up...")
    print("=" * 60)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Startup] Device: {device}")
    
    # Load Conflict Predictor (XGBoost)
    try:
        from agents.detection_agent.prediction_conflict.predictor import ConflictPredictor
        model_path = os.environ.get("PREDICTOR_MODEL_PATH")
        conflict_predictor = ConflictPredictor(
            model_path=Path(model_path) if model_path else None,
            auto_load=True
        )
        print("[Startup] ✓ Conflict Predictor loaded")
    except Exception as e:
        print(f"[Startup] ✗ Conflict Predictor failed: {e}")
        traceback.print_exc()
    
    # Load Detection Engine (Rule-based)
    try:
        from agents.detection_agent.deterministic_detection.engine import DetectionEngine, ConflictEmitter
        from agents.detection_agent.deterministic_detection.state_tracker import StateTracker
        
        state_tracker = StateTracker()
        emitter = ConflictEmitter(enable_console=False)
        detection_engine = DetectionEngine(state_tracker=state_tracker, emitter=emitter)
        print("[Startup] ✓ Detection Engine loaded")
    except Exception as e:
        print(f"[Startup] ✗ Detection Engine failed: {e}")
        traceback.print_exc()
    
    # Load Track Fault Detector (YOLOv8)
    try:
        from agents.detection_agent.vision_track_fault.model import TrackFaultDetector
        track_fault_detector = TrackFaultDetector()
        print("[Startup] ✓ Track Fault Detector loaded")
    except Exception as e:
        print(f"[Startup] ✗ Track Fault Detector failed: {e}")
        traceback.print_exc()
    
    print("=" * 60)
    print("DETECTION AGENT - Ready!")
    print("=" * 60 + "\n")
    
    yield
    
    # Cleanup
    print("\n[Shutdown] Detection Agent shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Rail-Mind Detection Agent",
    description="Microservice for conflict prediction and detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def root():
    """API info."""
    return {
        "name": "Rail-Mind Detection Agent",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Conflict prediction (ML)",
            "/detect": "POST - Deterministic detection (rules)",
            "/vision/detect": "POST - Track fault detection (vision)",
            "/vision/batch": "POST - Batch track fault detection"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with model status."""
    return HealthResponse(
        status="ok",
        device=str(device) if device else "unknown",
        models_loaded={
            "conflict_predictor": conflict_predictor is not None,
            "detection_engine": detection_engine is not None,
            "track_fault_detector": track_fault_detector is not None
        },
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run conflict prediction on network state.
    
    Returns predictions for all trains with risk levels.
    """
    if conflict_predictor is None:
        raise HTTPException(status_code=503, detail="Conflict Predictor not loaded")
    
    try:
        from agents.detection_agent.prediction_conflict.feature_engine import (
            TrainState, StationState, NetworkState
        )
        
        # Convert input to internal format
        trains = {}
        for t in req.network_state.trains:
            trains[t.train_id] = TrainState(
                train_id=t.train_id,
                train_type=t.train_type,
                current_station=t.current_station,
                next_station=t.next_station,
                position_km=t.position_km,
                speed_kmh=t.speed_kmh,
                delay_seconds=t.delay_seconds,
                scheduled_departure=datetime.fromisoformat(t.scheduled_departure) if t.scheduled_departure else None,
                actual_departure=datetime.fromisoformat(t.actual_departure) if t.actual_departure else None
            )
        
        stations = {}
        for s in req.network_state.stations:
            stations[s.get("id", s.get("station_id"))] = StationState(
                station_id=s.get("id", s.get("station_id")),
                station_name=s.get("name", s.get("station_name", "")),
                platforms=s.get("platforms", 2),
                occupied_platforms=s.get("occupied_platforms", 0),
                trains_at_station=s.get("trains_at_station", []),
                pending_arrivals=s.get("pending_arrivals", []),
                is_major=s.get("is_major", False),
                congestion_level=s.get("congestion_level", 0.0)
            )
        
        # Build network state
        current_time = datetime.now()
        if req.network_state.current_time:
            try:
                current_time = datetime.fromisoformat(req.network_state.current_time)
            except:
                pass
        
        network_state = NetworkState(
            trains=trains,
            stations=stations,
            edges={},
            current_time=current_time,
            is_peak_hour=req.network_state.peak_hour,
            weather_condition=req.network_state.weather,
            network_congestion=0.0
        )
        
        # Run prediction
        batch = conflict_predictor.predict_batch(network_state)
        
        # Convert to response
        return {
            "timestamp": batch.timestamp.isoformat(),
            "network_risk_score": batch.network_risk_score,
            "high_risk_trains": batch.high_risk_trains,
            "critical_trains": batch.critical_trains,
            "recommended_actions": batch.recommended_actions,
            "predictions": [
                {
                    "train_id": p.train_id,
                    "probability": p.probability,
                    "risk_level": p.risk_level,
                    "color": p.color,
                    "emoji": p.emoji,
                    "predicted_conflict_type": p.predicted_conflict_type,
                    "predicted_time": p.predicted_time.isoformat() if p.predicted_time else None,
                    "predicted_location": p.predicted_location,
                    "contributing_factors": p.contributing_factors,
                    "confidence": p.confidence,
                    "model_used": p.model_used
                }
                for p in batch.predictions
            ]
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/detect")
def detect(req: DetectRequest):
    """
    Run deterministic conflict detection using rules.
    
    Returns list of detected conflicts.
    """
    if detection_engine is None:
        raise HTTPException(status_code=503, detail="Detection Engine not loaded")
    
    try:
        # Update state tracker with provided data
        state_tracker = detection_engine.state_tracker
        
        # Update trains
        for train in req.trains:
            state_tracker.update_train(
                train_id=train.get("train_id"),
                position_km=train.get("position_km", 0),
                speed_kmh=train.get("speed_kmh", 0),
                current_station=train.get("current_station"),
                next_station=train.get("next_station"),
                delay_seconds=train.get("delay_seconds", 0),
                current_edge=train.get("current_edge")
            )
        
        # Run detection
        if req.current_time:
            try:
                current_time = datetime.fromisoformat(req.current_time)
                conflicts = detection_engine.tick(current_time)
            except:
                conflicts = detection_engine.evaluate_all_rules()
        else:
            conflicts = detection_engine.evaluate_all_rules()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "conflicts_detected": len(conflicts),
            "conflicts": [c.to_dict() for c in conflicts],
            "statistics": detection_engine.get_statistics()
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/vision/detect")
def vision_detect(req: VisionDetectRequest):
    """
    Run track fault detection on a single image.
    
    Returns binary classification: DEFECTIVE or NOT DEFECTIVE.
    """
    if track_fault_detector is None:
        raise HTTPException(status_code=503, detail="Track Fault Detector not loaded")
    
    try:
        # Validate image path
        if not Path(req.image_path).exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")
        
        result = track_fault_detector.detect(req.image_path, req.location)
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Vision detection failed: {str(e)}")


@app.post("/vision/batch")
def vision_batch(req: BatchVisionRequest):
    """
    Run track fault detection on all images in a folder.
    
    Returns list of results with defective count.
    """
    if track_fault_detector is None:
        raise HTTPException(status_code=503, detail="Track Fault Detector not loaded")
    
    try:
        # Validate folder
        if not Path(req.folder_path).exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {req.folder_path}")
        
        results = track_fault_detector.scan_folder(req.folder_path, req.location)
        
        defective_count = sum(1 for r in results if r.is_defective)
        
        return {
            "folder": req.folder_path,
            "location": req.location,
            "total_images": len(results),
            "defective_count": defective_count,
            "healthy_count": len(results) - defective_count,
            "results": [r.to_dict() for r in results]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch vision detection failed: {str(e)}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
