"""Canary router for A/B traffic splitting between model versions"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import random
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from typing import Dict

app = FastAPI(title="Canary Router", version="1.0.0")

# Configuration
API_V1_URL = "http://api-v1:8000"
MODEL_V2_URL = "http://model-v2:8080"
CANARY_RATIO = 0.30  # 30% traffic to new model

# Prometheus Metrics
ROUTE_COUNTER = Counter('route_requests_total', 'Total routed requests', ['destination', 'path'])
ROUTE_LATENCY = Histogram('route_latency_seconds', 'Routing latency', ['destination'])
ROUTE_ERRORS = Counter('route_errors_total', 'Routing errors', ['destination', 'error_type'])

logger = logging.getLogger(__name__)


@app.get("/")
def root():
    return {
        "service": "Canary Router",
        "version": "1.0.0",
        "routing": {
            "api_v1": f"{int((1-CANARY_RATIO)*100)}%",
            "model_v2": f"{int(CANARY_RATIO*100)}%"
        },
        "endpoints": {
            "api_v1": API_V1_URL,
            "model_v2": MODEL_V2_URL
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "canary_ratio": CANARY_RATIO
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_request(path: str, request: Request):
    """Route requests based on canary ratio"""
    
    # Determine destination
    use_canary = random.random() < CANARY_RATIO
    destination_url = MODEL_V2_URL if use_canary else API_V1_URL
    destination_name = "model_v2" if use_canary else "api_v1"
    
    # Build target URL
    target_url = f"{destination_url}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    # Update metrics
    ROUTE_COUNTER.labels(destination=destination_name, path=f"/{path}").inc()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Forward request
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=dict(request.headers),
                content=await request.body(),
            )
            
            # Add routing headers
            headers = dict(response.headers)
            headers['X-Routed-To'] = destination_name
            headers['X-Canary-Ratio'] = str(CANARY_RATIO)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=headers
            )
    
    except httpx.TimeoutException:
        ROUTE_ERRORS.labels(destination=destination_name, error_type="timeout").inc()
        logger.error(f"Timeout routing to {destination_name}")
        raise HTTPException(status_code=504, detail=f"Timeout routing to {destination_name}")
    
    except httpx.RequestError as e:
        ROUTE_ERRORS.labels(destination=destination_name, error_type="connection").inc()
        logger.error(f"Connection error routing to {destination_name}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to route to {destination_name}")
    
    except Exception as e:
        ROUTE_ERRORS.labels(destination=destination_name, error_type="unknown").inc()
        logger.error(f"Error routing to {destination_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal routing error")


@app.post("/admin/update-ratio")
async def update_canary_ratio(new_ratio: float):
    """Update canary routing ratio"""
    global CANARY_RATIO
    
    if not 0 <= new_ratio <= 1:
        raise HTTPException(status_code=400, detail="Ratio must be between 0 and 1")
    
    old_ratio = CANARY_RATIO
    CANARY_RATIO = new_ratio
    
    logger.info(f"Canary ratio updated: {old_ratio} -> {new_ratio}")
    
    return {
        "message": "Canary ratio updated",
        "old_ratio": old_ratio,
        "new_ratio": new_ratio,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/admin/stats")
async def get_stats():
    """Get routing statistics"""
    # In production, query Prometheus for actual stats
    return {
        "current_ratio": CANARY_RATIO,
        "api_v1_percentage": int((1 - CANARY_RATIO) * 100),
        "model_v2_percentage": int(CANARY_RATIO * 100),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
