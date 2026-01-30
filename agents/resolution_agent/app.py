"""
Resolution Agent FastAPI Server
===============================

This microservice exposes the resolution agent's capabilities:
1. Hybrid RAG Resolution (LLM + Vector DB)
2. Mathematical Resolution (Optimization algorithms)
3. Orchestrated Resolution (Both agents + LLM Judge)

Endpoints:
- GET /health - Health check with component status
- POST /resolve - Full orchestrated resolution
- POST /resolve/hybrid - Hybrid RAG only
- POST /resolve/mathematical - Mathematical solver only
"""

import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global components
hybrid_rag_available = False
mathematical_available = False
llm_judge_available = False

# Thread pool for running sync orchestrator code
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="resolver")


# =============================================================================
# Request/Response Models
# =============================================================================

class ConflictInput(BaseModel):
    """Conflict to resolve."""
    conflict_id: str
    conflict_type: str = "unknown"
    station_ids: List[str] = Field(default_factory=list)
    train_ids: List[str] = Field(default_factory=list)
    delay_values: Dict[str, float] = Field(default_factory=dict)
    timestamp: Optional[float] = None
    severity: float = 0.5
    blocking_behavior: str = "soft"
    # Optional additional context from detection
    original_detection: Optional[Dict[str, Any]] = None


class ContextInput(BaseModel):
    """Operational context."""
    time_of_day: float = 12.0
    day_of_week: int = 0
    is_peak_hour: bool = False
    weather_condition: str = "clear"
    network_load: float = 0.5


class ResolveRequest(BaseModel):
    """Request for conflict resolution."""
    conflict: ConflictInput
    context: Optional[ContextInput] = None
    timeout: float = 60.0
    llm_api_key: Optional[str] = None  # Override env var


class ResolveResponse(BaseModel):
    """Resolution response."""
    status: str
    conflict_id: str
    started_at: str
    finished_at: str
    total_execution_ms: int
    agents: Dict[str, Any]
    llm_judge: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    components: Dict[str, bool]
    timestamp: str


# =============================================================================
# Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components at startup."""
    global hybrid_rag_available, mathematical_available, llm_judge_available
    
    print("\n" + "=" * 60)
    print("RESOLUTION AGENT - Starting up...")
    print("=" * 60)
    
    # Check Hybrid RAG availability
    try:
        from agents.resolution_agent.hybrid_rag.agent.resolution_generator import ResolutionGenerationSystem
        hybrid_rag_available = True
        print("[Startup] ✓ Hybrid RAG module available")
    except ImportError as e:
        print(f"[Startup] ✗ Hybrid RAG not available: {e}")
    
    # Check Mathematical solver availability
    try:
        from agents.resolution_agent.mathematical_resolution import (
            Conflict, Context, ResolutionOrchestrator, OrchestratorConfig
        )
        mathematical_available = True
        print("[Startup] ✓ Mathematical Solver module available")
    except ImportError as e:
        print(f"[Startup] ✗ Mathematical Solver not available: {e}")
    
    # Check LLM Judge availability
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            llm_judge_available = True
            print("[Startup] ✓ LLM Judge available (GROQ_API_KEY set)")
        else:
            print("[Startup] ⚠ LLM Judge: GROQ_API_KEY not set")
    except ImportError as e:
        print(f"[Startup] ✗ LLM Judge not available: {e}")
    
    print("=" * 60)
    print("RESOLUTION AGENT - Ready!")
    print("=" * 60 + "\n")
    
    yield
    
    # Cleanup
    print("\n[Shutdown] Resolution Agent shutting down...")
    executor.shutdown(wait=False)


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Rail-Mind Resolution Agent",
    description="Microservice for conflict resolution with hybrid/mathematical approaches",
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
        "name": "Rail-Mind Resolution Agent",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/resolve": "POST - Full orchestrated resolution (both agents + judge)",
            "/resolve/hybrid": "POST - Hybrid RAG resolution only",
            "/resolve/mathematical": "POST - Mathematical solver only"
        },
        "env_vars": {
            "GROQ_API_KEY": "Required for LLM judge",
            "GROQ_MODEL": "Optional, default: llama-3.3-70b-versatile",
            "QDRANT_URL": "Required for Hybrid RAG",
            "QDRANT_API_KEY": "Optional, for cloud Qdrant"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with component status."""
    return HealthResponse(
        status="ok" if (hybrid_rag_available or mathematical_available) else "degraded",
        components={
            "hybrid_rag": hybrid_rag_available,
            "mathematical_solver": mathematical_available,
            "llm_judge": llm_judge_available
        },
        timestamp=datetime.now().isoformat()
    )


@app.post("/resolve")
async def resolve(req: ResolveRequest):
    """
    Run full orchestrated resolution.
    
    Runs both Hybrid RAG and Mathematical agents in parallel,
    normalizes their outputs, and uses LLM Judge to rank results.
    """
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    
    # Convert request to dict format expected by orchestrator
    conflict = {
        "conflict_id": req.conflict.conflict_id,
        "conflict_type": req.conflict.conflict_type,
        "station_ids": req.conflict.station_ids,
        "train_ids": req.conflict.train_ids,
        "delay_values": req.conflict.delay_values,
        "timestamp": req.conflict.timestamp or datetime.now().timestamp(),
        "severity": req.conflict.severity,
        "blocking_behavior": req.conflict.blocking_behavior
    }
    
    if req.conflict.original_detection:
        conflict["original_detection"] = req.conflict.original_detection
    
    context = None
    if req.context:
        context = {
            "time_of_day": req.context.time_of_day,
            "day_of_week": req.context.day_of_week,
            "is_peak_hour": req.context.is_peak_hour,
            "weather_condition": req.context.weather_condition,
            "network_load": req.context.network_load
        }
    
    api_key = req.llm_api_key or os.environ.get("GROQ_API_KEY")
    
    try:
        # Import and run orchestrator
        from agents.resolution_agent.resolver import orchestrate
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: orchestrate(
                conflict=conflict,
                context=context,
                timeout=req.timeout,
                api_key=api_key
            )
        )
        
        finished_at = datetime.now(timezone.utc)
        total_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Ensure result has required structure
        if isinstance(result, dict):
            result["started_at"] = started_at.isoformat()
            result["finished_at"] = finished_at.isoformat()
            result["total_execution_ms"] = total_ms
            return result
        else:
            return {
                "status": "ok",
                "conflict_id": req.conflict.conflict_id,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "total_execution_ms": total_ms,
                "agents": {},
                "llm_judge": {},
                "result": result
            }
            
    except Exception as e:
        traceback.print_exc()
        finished_at = datetime.now(timezone.utc)
        total_ms = int((time.perf_counter() - start_time) * 1000)
        
        return {
            "status": "error",
            "conflict_id": req.conflict.conflict_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total_execution_ms": total_ms,
            "error": f"{type(e).__name__}: {str(e)}",
            "agents": {},
            "llm_judge": {}
        }


@app.post("/resolve/hybrid")
async def resolve_hybrid(req: ResolveRequest):
    """
    Run Hybrid RAG resolution only.
    
    Uses vector database + LLM for context-aware resolution suggestions.
    """
    if not hybrid_rag_available:
        raise HTTPException(status_code=503, detail="Hybrid RAG not available")
    
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    
    conflict = {
        "conflict_id": req.conflict.conflict_id,
        "conflict_type": req.conflict.conflict_type,
        "station_ids": req.conflict.station_ids,
        "train_ids": req.conflict.train_ids,
        "delay_values": req.conflict.delay_values,
        "timestamp": req.conflict.timestamp or datetime.now().timestamp(),
        "severity": req.conflict.severity,
    }
    
    context = {}
    if req.context:
        context = {
            "time_of_day": req.context.time_of_day,
            "day_of_week": req.context.day_of_week,
            "is_peak_hour": req.context.is_peak_hour,
            "weather_condition": req.context.weather_condition,
            "network_load": req.context.network_load
        }
    
    try:
        from agents.resolution_agent.resolver import run_hybrid_rag_agent
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: run_hybrid_rag_agent(conflict, context, req.timeout)
        )
        
        execution_ms = int((time.perf_counter() - start_time) * 1000)
        
        return {
            "status": result.status,
            "conflict_id": req.conflict.conflict_id,
            "execution_ms": execution_ms,
            "raw_result": result.raw_result,
            "normalized_resolutions": result.normalized_resolutions,
            "error": result.error
        }
        
    except Exception as e:
        traceback.print_exc()
        execution_ms = int((time.perf_counter() - start_time) * 1000)
        raise HTTPException(
            status_code=500, 
            detail=f"Hybrid RAG failed: {str(e)}"
        )


@app.post("/resolve/mathematical")
async def resolve_mathematical(req: ResolveRequest):
    """
    Run Mathematical solver only.
    
    Uses optimization algorithms (genetic, simulated annealing, etc.)
    for precise delay minimization.
    """
    if not mathematical_available:
        raise HTTPException(status_code=503, detail="Mathematical Solver not available")
    
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    
    conflict = {
        "conflict_id": req.conflict.conflict_id,
        "conflict_type": req.conflict.conflict_type,
        "station_ids": req.conflict.station_ids,
        "train_ids": req.conflict.train_ids,
        "delay_values": req.conflict.delay_values,
        "timestamp": req.conflict.timestamp or datetime.now().timestamp(),
        "severity": req.conflict.severity,
        "blocking_behavior": req.conflict.blocking_behavior
    }
    
    context = {}
    if req.context:
        context = {
            "time_of_day": req.context.time_of_day,
            "day_of_week": req.context.day_of_week,
            "is_peak_hour": req.context.is_peak_hour,
            "weather_condition": req.context.weather_condition,
            "network_load": req.context.network_load
        }
    
    try:
        from agents.resolution_agent.resolver import run_mathematical_agent
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: run_mathematical_agent(conflict, context, req.timeout)
        )
        
        execution_ms = int((time.perf_counter() - start_time) * 1000)
        
        return {
            "status": result.status,
            "conflict_id": req.conflict.conflict_id,
            "execution_ms": execution_ms,
            "raw_result": result.raw_result,
            "normalized_resolutions": result.normalized_resolutions,
            "error": result.error
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Mathematical solver failed: {str(e)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
