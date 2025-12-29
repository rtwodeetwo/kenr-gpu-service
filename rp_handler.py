"""
RunPod Serverless Handler for KENR GPU Service.

This wraps the existing services in RunPod's handler format for serverless deployment.
Supports:
- Top-K feature extraction
- Model comparison
- Health checks
"""

import os
import logging
import traceback

import runpod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import services (lazy load to reduce cold start for health checks)
_topk_service = None
_compare_service = None


def get_topk_service():
    """Lazy load TopKService."""
    global _topk_service
    if _topk_service is None:
        from services import TopKService
        _topk_service = TopKService()
    return _topk_service


def get_compare_service():
    """Lazy load CompareService."""
    global _compare_service
    if _compare_service is None:
        from services import CompareService
        _compare_service = CompareService()
    return _compare_service


def verify_api_key(event: dict) -> bool:
    """Verify API key from request."""
    expected_key = os.environ.get("API_KEY", "")
    if not expected_key:
        # No API key configured = allow all (development mode)
        return True

    # Check headers
    headers = event.get("headers", {}) or {}
    provided_key = headers.get("x-api-key") or headers.get("X-Api-Key") or ""

    # Also check input for API key
    input_data = event.get("input", {}) or {}
    if not provided_key:
        provided_key = input_data.get("api_key", "")

    return provided_key == expected_key


def handler(event: dict) -> dict:
    """
    Main RunPod handler function.

    Expects input in format:
    {
        "input": {
            "action": "topk" | "compare" | "health" | "models",
            "api_key": "your-api-key",  # Optional if passed in headers
            ... action-specific parameters ...
        }
    }

    For top-K extraction:
    {
        "input": {
            "action": "topk",
            "model_id": "llama-3.1-8b",
            "text": "Your input text",
            "k": 10,
            "layer": null,
            "include_descriptions": true
        }
    }

    For model comparison:
    {
        "input": {
            "action": "compare",
            "models": ["gemma-2-2b", "llama-3.1-8b"],
            "sentences": ["Test sentence"],
            "domain_terms": ["term1", "term2"]
        }
    }
    """
    try:
        input_data = event.get("input", {})

        if not input_data:
            return {"error": "No input provided"}

        action = input_data.get("action", "topk")

        # Health check - no auth required
        if action == "health":
            return handle_health()

        # List models - no auth required
        if action == "models":
            return handle_models()

        # Verify API key for protected endpoints
        if not verify_api_key(event):
            return {"error": "Invalid or missing API key"}

        # Route to appropriate handler
        if action == "topk":
            return handle_topk(input_data)
        elif action == "compare":
            return handle_compare(input_data)
        else:
            return {"error": f"Unknown action: {action}. Valid actions: topk, compare, health, models"}

    except Exception as e:
        logger.exception(f"Handler error: {e}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def handle_health() -> dict:
    """Handle health check request."""
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        gpu_info = {}

        if gpu_available:
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "gpu_memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3),
            }

        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            **gpu_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def handle_models() -> dict:
    """Handle list models request."""
    from config import MODEL_CONFIGS

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


def handle_topk(input_data: dict) -> dict:
    """Handle top-K feature extraction request."""
    from config import MODEL_CONFIGS
    from schemas import TopKRequest

    # Validate model
    model_id = input_data.get("model_id")
    if not model_id:
        return {"error": "model_id is required"}

    if model_id not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"}

    # Validate text
    text = input_data.get("text")
    if not text:
        return {"error": "text is required"}

    # Build request
    request = TopKRequest(
        model_id=model_id,
        text=text,
        layer=input_data.get("layer"),
        k=input_data.get("k", 10),
        include_descriptions=input_data.get("include_descriptions", True),
    )

    # Get result
    service = get_topk_service()
    result = service.extract_topk(request)

    # Convert to dict for RunPod response
    return result.model_dump()


def handle_compare(input_data: dict) -> dict:
    """Handle model comparison request."""
    from config import MODEL_CONFIGS
    from schemas import CompareRequest

    # Validate models
    models = input_data.get("models", [])
    if not models:
        return {"error": "models list is required"}

    for model_id in models:
        if model_id not in MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"}

    # Validate sentences
    sentences = input_data.get("sentences", [])
    if not sentences:
        return {"error": "At least one sentence is required"}

    # Build request
    request = CompareRequest(
        models=models,
        sentences=sentences,
        domain_terms=input_data.get("domain_terms", []),
    )

    # Get result
    service = get_compare_service()
    result = service.compare_models(request)

    # Convert to dict for RunPod response
    return result.model_dump()


# Start the serverless worker
if __name__ == "__main__":
    logger.info("Starting KENR GPU Service (RunPod Serverless)...")
    runpod.serverless.start({"handler": handler})
