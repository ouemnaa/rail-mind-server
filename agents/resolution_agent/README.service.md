# Resolution Agent Microservice

This is the Resolution Agent microservice for Rail-Mind. It provides:

- **Full Orchestration** - Runs both agents + LLM judge for ranked resolutions
- **Hybrid RAG Resolution** - LLM + vector database for context-aware suggestions
- **Mathematical Resolution** - Optimization algorithms (genetic, simulated annealing, etc.)

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check with component status |
| `/resolve` | POST | Full orchestrated resolution (both agents + judge) |
| `/resolve/hybrid` | POST | Hybrid RAG resolution only |
| `/resolve/mathematical` | POST | Mathematical solver only |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `8000` | Port to listen on |
| `GROQ_API_KEY` | **Yes** | - | Groq API key for LLM |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | LLM model to use |
| `QDRANT_URL` | **Yes** | - | Qdrant vector database URL |
| `QDRANT_API_KEY` | Conditional | - | Qdrant API key (required for cloud) |

## Running Locally

### Without Docker

```bash
# From repo root
cd agents/resolution_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.service.txt

# Set environment variables
export GROQ_API_KEY="your-groq-api-key"
export QDRANT_URL="http://localhost:6333"

# Run server
python app.py
```

Server will be available at `http://localhost:8002`

### With Docker

```bash
# From agent directory
docker build -t resolution-agent .
docker run -p 8002:8000 \
  -e GROQ_API_KEY="your-key" \
  -e QDRANT_URL="http://host.docker.internal:6333" \
  resolution-agent

# Or from repo root with compose
docker-compose up resolution_agent
```

## API Examples

### Health Check

```bash
curl http://localhost:8002/health
```

Response:
```json
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

### Full Orchestrated Resolution

```bash
curl -X POST http://localhost:8002/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "conflict": {
      "conflict_id": "CONF-2024-001",
      "conflict_type": "schedule_conflict",
      "station_ids": ["MI_CENTRALE", "MI_GARIBALDI"],
      "train_ids": ["REG123", "IC456"],
      "delay_values": {"REG123": 5.0, "IC456": 2.0},
      "severity": 0.7
    },
    "context": {
      "time_of_day": 8.5,
      "is_peak_hour": true,
      "network_load": 0.85
    },
    "timeout": 60
  }'
```

### Hybrid RAG Only

```bash
curl -X POST http://localhost:8002/resolve/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "conflict": {
      "conflict_id": "CONF-2024-001",
      "conflict_type": "schedule_conflict",
      "train_ids": ["REG123", "IC456"]
    }
  }'
```

### Mathematical Solver Only

```bash
curl -X POST http://localhost:8002/resolve/mathematical \
  -H "Content-Type: application/json" \
  -d '{
    "conflict": {
      "conflict_id": "CONF-2024-001",
      "conflict_type": "schedule_conflict",
      "train_ids": ["REG123", "IC456"],
      "delay_values": {"REG123": 5.0, "IC456": 2.0}
    }
  }'
```

## Response Format

The orchestrated `/resolve` endpoint returns:

```json
{
  "status": "ok",
  "conflict_id": "CONF-2024-001",
  "started_at": "2024-01-30T10:00:00Z",
  "finished_at": "2024-01-30T10:00:15Z",
  "total_execution_ms": 15432,
  "agents": {
    "hybrid_rag": {
      "status": "ok",
      "execution_ms": 8234,
      "normalized_resolutions": [...]
    },
    "mathematical": {
      "status": "ok", 
      "execution_ms": 5123,
      "normalized_resolutions": [...]
    }
  },
  "llm_judge": {
    "status": "ok",
    "execution_ms": 2075,
    "ranked_resolutions": [
      {
        "rank": 1,
        "resolution_id": "hybrid_historical_1",
        "overall_score": 87,
        "safety_rating": 9.0,
        "efficiency_rating": 8.5,
        "justification": "..."
      }
    ]
  }
}
```

## Docker Build Options

### CPU Build (smaller image)

```bash
docker build -t resolution-agent \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  .
```

## Architecture

```
resolution_agent/
├── app.py                     # FastAPI server
├── requirements.service.txt   # Dependencies
├── Dockerfile                 # Container build
├── resolver.py               # Main orchestrator
├── hybrid_rag/               # Hybrid RAG agent
│   └── agent/
│       └── resolution_generator.py
├── mathematical_resolution/   # Mathematical solvers
│   ├── __init__.py
│   └── solvers/
└── llm_judge/                 # LLM judge for ranking
    └── llm_judge_v2.py
```

## Qdrant Setup

For the Hybrid RAG agent to work, you need a Qdrant instance with indexed data.

### Local Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Cloud Qdrant

1. Create account at https://qdrant.io
2. Create a cluster
3. Get URL and API key
4. Set environment variables:
   ```
   QDRANT_URL=https://xxx.qdrant.io:6333
   QDRANT_API_KEY=your-api-key
   ```

## Integration with Backend

The backend calls this service via HTTP. Configure the backend with:

```
AGENT_RESOLUTION_URL=http://localhost:8002
```

Or in docker-compose:
```
AGENT_RESOLUTION_URL=http://resolution_agent:8000
```
