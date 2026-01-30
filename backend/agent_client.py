"""
Agent HTTP Client
=================

Helper module to call agent microservices with retries and timeouts.

Usage:
    from backend.agent_client import agent_client
    
    # Call detection agent
    result = agent_client.predict(network_state)
    
    # Call resolution agent
    result = agent_client.resolve(conflict, context)
    
Environment Variables:
    AGENT_DETECTION_URL - URL for detection agent (e.g., http://localhost:8001)
    AGENT_RESOLUTION_URL - URL for resolution agent (e.g., http://localhost:8002)
    
    Alternative: AGENT_MAP (JSON) - e.g., {"detection": "http://...", "resolution": "http://..."}
"""

import os
import json
import time
import logging
from typing import Any, Dict, Optional
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def _get_agent_urls() -> Dict[str, str]:
    """
    Get agent URLs from environment variables.
    
    Supports two formats:
    1. Individual env vars: AGENT_DETECTION_URL, AGENT_RESOLUTION_URL
    2. JSON map: AGENT_MAP = {"detection": "...", "resolution": "..."}
    """
    urls = {}
    
    # Try JSON map first
    agent_map = os.environ.get("AGENT_MAP")
    if agent_map:
        try:
            urls = json.loads(agent_map)
            logger.info(f"Loaded agent URLs from AGENT_MAP: {list(urls.keys())}")
            return urls
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid AGENT_MAP JSON: {e}")
    
    # Fall back to individual env vars
    detection_url = os.environ.get("AGENT_DETECTION_URL")
    resolution_url = os.environ.get("AGENT_RESOLUTION_URL")
    
    if detection_url:
        urls["detection"] = detection_url.rstrip("/")
        logger.info(f"Detection agent URL: {detection_url}")
    
    if resolution_url:
        urls["resolution"] = resolution_url.rstrip("/")
        logger.info(f"Resolution agent URL: {resolution_url}")
    
    return urls


# Global configuration
AGENT_URLS = {}  # Populated on first use
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5
RETRY_STATUS_CODES = [502, 503, 504]


# =============================================================================
# Session Factory
# =============================================================================

def _create_session() -> requests.Session:
    """Create a requests session with retry configuration."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# Global session
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create the global session."""
    global _session
    if _session is None:
        _session = _create_session()
    return _session


# =============================================================================
# Agent Client Class
# =============================================================================

class AgentClient:
    """
    HTTP client for calling agent microservices.
    
    Provides methods for each agent with automatic retries and timeouts.
    """
    
    def __init__(self):
        self._urls: Dict[str, str] = {}
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of URLs."""
        if not self._initialized:
            self._urls = _get_agent_urls()
            self._initialized = True
    
    def _get_url(self, agent: str) -> str:
        """Get URL for an agent, with validation."""
        self._ensure_initialized()
        
        url = self._urls.get(agent)
        if not url:
            # Try to get from env again (for late binding)
            self._urls = _get_agent_urls()
            url = self._urls.get(agent)
        
        if not url:
            raise ValueError(
                f"No URL configured for agent '{agent}'. "
                f"Set AGENT_{agent.upper()}_URL or AGENT_MAP environment variable."
            )
        
        return url
    
    def _call(
        self, 
        agent: str, 
        endpoint: str, 
        method: str = "POST",
        payload: Optional[Dict] = None,
        timeout: int = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Make an HTTP call to an agent.
        
        Args:
            agent: Agent name (detection, resolution)
            endpoint: API endpoint (e.g., /predict)
            method: HTTP method
            payload: Request body for POST
            timeout: Request timeout in seconds
            
        Returns:
            Response JSON as dict
            
        Raises:
            requests.RequestException: On network errors
            ValueError: On invalid response
        """
        base_url = self._get_url(agent)
        url = f"{base_url}{endpoint}"
        session = _get_session()
        
        start_time = time.time()
        logger.debug(f"Calling {agent} agent: {method} {url}")
        
        try:
            if method.upper() == "GET":
                response = session.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = session.post(url, json=payload, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            elapsed = time.time() - start_time
            logger.debug(f"Agent {agent} responded in {elapsed:.2f}s with status {response.status_code}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling {agent} agent after {timeout}s")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to {agent} agent at {url}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {agent} agent: {e}")
            raise
    
    # =========================================================================
    # Detection Agent Methods
    # =========================================================================
    
    def detection_health(self, timeout: int = 5) -> Dict[str, Any]:
        """Check detection agent health."""
        return self._call("detection", "/health", method="GET", timeout=timeout)
    
    def predict(
        self, 
        network_state: Dict[str, Any],
        force: bool = False,
        horizon_minutes: int = 30,
        timeout: int = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Run conflict prediction on network state.
        
        Args:
            network_state: Dict with trains, stations, edges, etc.
            force: Force prediction even if triggers not met
            horizon_minutes: Prediction horizon
            timeout: Request timeout
            
        Returns:
            Prediction results with risk levels
        """
        payload = {
            "network_state": network_state,
            "force": force,
            "horizon_minutes": horizon_minutes
        }
        return self._call("detection", "/predict", payload=payload, timeout=timeout)
    
    def detect(
        self,
        trains: list,
        stations: list = None,
        edges: list = None,
        current_time: str = None,
        timeout: int = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Run deterministic conflict detection.
        
        Args:
            trains: List of train states
            stations: List of station states
            edges: List of edge states
            current_time: Current simulation time (ISO format)
            timeout: Request timeout
            
        Returns:
            Detection results with conflicts
        """
        payload = {
            "trains": trains,
            "stations": stations or [],
            "edges": edges or [],
            "current_time": current_time
        }
        return self._call("detection", "/detect", payload=payload, timeout=timeout)
    
    def vision_detect(
        self,
        image_path: str,
        location: str = "UNKNOWN",
        timeout: int = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Run track fault detection on an image.
        
        Args:
            image_path: Path to track image
            location: Edge ID for the location
            timeout: Request timeout
            
        Returns:
            Binary classification result
        """
        payload = {
            "image_path": image_path,
            "location": location
        }
        return self._call("detection", "/vision/detect", payload=payload, timeout=timeout)
    
    def vision_batch(
        self,
        folder_path: str,
        location: str = "UNKNOWN",
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Run track fault detection on a folder of images.
        
        Args:
            folder_path: Path to folder with images
            location: Edge ID for the location
            timeout: Request timeout (longer for batch)
            
        Returns:
            Batch results with defective count
        """
        payload = {
            "folder_path": folder_path,
            "location": location
        }
        return self._call("detection", "/vision/batch", payload=payload, timeout=timeout)
    
    # =========================================================================
    # Resolution Agent Methods
    # =========================================================================
    
    def resolution_health(self, timeout: int = 5) -> Dict[str, Any]:
        """Check resolution agent health."""
        return self._call("resolution", "/health", method="GET", timeout=timeout)
    
    def resolve(
        self,
        conflict: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
        llm_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full orchestrated resolution.
        
        Args:
            conflict: Conflict to resolve
            context: Operational context
            timeout: Per-agent timeout
            llm_api_key: Optional LLM API key override
            
        Returns:
            Orchestrated resolution with ranked results
        """
        payload = {
            "conflict": conflict,
            "context": context,
            "timeout": timeout
        }
        if llm_api_key:
            payload["llm_api_key"] = llm_api_key
        
        # Use longer request timeout for orchestration
        return self._call("resolution", "/resolve", payload=payload, timeout=int(timeout) + 30)
    
    def resolve_hybrid(
        self,
        conflict: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run Hybrid RAG resolution only.
        
        Args:
            conflict: Conflict to resolve
            context: Operational context
            timeout: Resolution timeout
            
        Returns:
            Hybrid RAG resolution results
        """
        payload = {
            "conflict": conflict,
            "context": context,
            "timeout": timeout
        }
        return self._call("resolution", "/resolve/hybrid", payload=payload, timeout=int(timeout) + 15)
    
    def resolve_mathematical(
        self,
        conflict: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run Mathematical solver only.
        
        Args:
            conflict: Conflict to resolve
            context: Operational context
            timeout: Resolution timeout
            
        Returns:
            Mathematical resolution results
        """
        payload = {
            "conflict": conflict,
            "context": context,
            "timeout": timeout
        }
        return self._call("resolution", "/resolve/mathematical", payload=payload, timeout=int(timeout) + 15)
    
    # =========================================================================
    # Health Check Utilities
    # =========================================================================
    
    def check_all_agents(self, timeout: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all configured agents.
        
        Returns:
            Dict mapping agent name to health response or error
        """
        results = {}
        
        for agent in ["detection", "resolution"]:
            try:
                if agent in self._urls or f"AGENT_{agent.upper()}_URL" in os.environ:
                    results[agent] = self._call(agent, "/health", method="GET", timeout=timeout)
                    results[agent]["reachable"] = True
            except Exception as e:
                results[agent] = {
                    "reachable": False,
                    "error": str(e)
                }
        
        return results


# =============================================================================
# Global Client Instance
# =============================================================================

# Singleton instance
agent_client = AgentClient()


# =============================================================================
# Convenience Functions
# =============================================================================

def call_detection_predict(payload: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Convenience function to call detection prediction."""
    return agent_client.predict(payload, timeout=timeout)


def call_detection_detect(trains: list, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Convenience function to call deterministic detection."""
    return agent_client.detect(trains, timeout=timeout)


def call_resolution(
    conflict: Dict[str, Any], 
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """Convenience function to call resolution orchestration."""
    return agent_client.resolve(conflict, context, timeout=timeout)


# =============================================================================
# Health Check for Backend Startup
# =============================================================================

def wait_for_agents(max_wait: int = 60, check_interval: int = 5) -> bool:
    """
    Wait for all configured agents to become available.
    
    Useful for startup synchronization in Docker Compose.
    
    Args:
        max_wait: Maximum seconds to wait
        check_interval: Seconds between checks
        
    Returns:
        True if all agents are healthy, False if timeout
    """
    logger.info("Waiting for agents to become available...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            results = agent_client.check_all_agents(timeout=check_interval)
            all_healthy = all(r.get("reachable", False) for r in results.values() if r)
            
            if all_healthy and len(results) > 0:
                logger.info("All agents are healthy!")
                return True
            
            unhealthy = [k for k, v in results.items() if not v.get("reachable", False)]
            logger.info(f"Waiting for agents: {unhealthy}")
            
        except Exception as e:
            logger.debug(f"Health check error: {e}")
        
        time.sleep(check_interval)
    
    logger.warning("Timeout waiting for agents")
    return False
