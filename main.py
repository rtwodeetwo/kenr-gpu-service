"""
KENR GPU Service - FastAPI Application

Provides GPU-accelerated interpretability features:
- Top-K feature extraction using SAEs
- Circuit tracing (attribution graphs)
- Model comparison
- Steering vector computation
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware

from config import settings, MODEL_CONFIGS
from models import get_model_manager
from services import TopKService, CompareService
from schemas import (
    HealthResponse,
    TopKRequest,
    TopKResponse,
    CompareRequest,
    CompareResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting KENR GPU Service...")
    logger.info(f"Available models: {list(MODEL_CONFIGS.keys())}")

    # Check GPU availability
    manager = get_model_manager()
    gpu_available, free_mem, total_mem = manager.get_gpu_info()
    if gpu_available:
        logger.info(f"GPU available: {free_mem:.1f}GB free / {total_mem:.1f}GB total")
    else:
        logger.warning("No GPU available - running on CPU (will be slow!)")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down KENR GPU Service...")
    manager.unload_all()


# Create FastAPI app
app = FastAPI(
    title="KENR GPU Service",
    description="GPU-accelerated interpretability service for KENR",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Authentication ===

async def verify_api_key(x_api_key: Annotated[str, Header()]) -> str:
    """Verify the API key from request header."""
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key


# === Health Check ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns GPU status and loaded models.
    No authentication required.
    """
    manager = get_model_manager()
    gpu_available, free_mem, total_mem = manager.get_gpu_info()

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_memory_free_gb=free_mem,
        gpu_memory_total_gb=total_mem,
        loaded_models=manager.loaded_models,
    )


@app.get("/models")
async def list_models():
    """List available models and their configurations."""
    return {
        "models": [
            {
                "id": model_id,
                "n_layers": config["n_layers"],
                "d_model": config["d_model"],
                "requires_auth": config["requires_auth"],
            }
            for model_id, config in MODEL_CONFIGS.items()
        ]
    }


# === Feature Extraction ===

@app.post(
    "/features/topk",
    response_model=TopKResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_topk_features(
    request: TopKRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Get top-K feature activations per token.

    Runs the model on the input text and extracts the most active
    SAE features for each token position.
    """
    # Validate model
    if request.model_id not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {request.model_id}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    # Validate layer if specified
    config = MODEL_CONFIGS[request.model_id]
    if request.layer is not None:
        if request.layer < 0 or request.layer >= config["n_layers"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Layer {request.layer} out of range for {request.model_id} (0-{config['n_layers']-1})"
            )

    try:
        service = TopKService()
        return service.extract_topk(request)
    except Exception as e:
        logger.exception(f"Error in topk extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# === Model Comparison ===

@app.post(
    "/compare",
    response_model=CompareResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def compare_models(
    request: CompareRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Compare multiple models on domain-specific sentences.

    Analyzes which features activate for domain content across models
    and recommends the best model for the domain.
    """
    # Validate models
    for model_id in request.models:
        if model_id not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"
            )

    # Validate we have at least one sentence
    if not request.sentences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one sentence is required"
        )

    try:
        service = CompareService()
        return service.compare_models(request)
    except Exception as e:
        logger.exception(f"Error in model comparison: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# === Placeholder endpoints for future implementation ===

@app.post("/circuit/trace")
async def trace_circuit(api_key: str = Depends(verify_api_key)):
    """Circuit tracing - not yet implemented."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Circuit tracing coming soon"
    )


@app.post("/steering/compute")
async def compute_steering(api_key: str = Depends(verify_api_key)):
    """Steering vector computation - not yet implemented."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Steering computation coming soon"
    )


@app.post("/steering/generate")
async def generate_steered(api_key: str = Depends(verify_api_key)):
    """Steered generation - not yet implemented."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Steered generation coming soon"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
