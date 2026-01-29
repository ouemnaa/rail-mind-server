import uvicorn
import os

# Import the app using absolute import (requires running with python -m backend.main)
from backend.integration.unified_api import app

def main():
    """Main entry point for the backend."""
    port = int(os.environ.get("PORT", 8002))
    print(f"Starting Rail-Mind Backend on port {port}...")
    uvicorn.run("backend.integration.unified_api:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
