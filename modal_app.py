"""
KENR GPU Service - Modal Serverless Deployment

This replaces the RunPod serverless handler with Modal's simpler deployment model.
Deploy with: modal deploy modal_app.py
Test locally with: modal run modal_app.py

Version: 2025-12-30-fix-response-format
"""

import modal
import os
from typing import Optional, List, Dict, Any

# Define the Modal app
app = modal.App("kenr-gpu-service")

# Create the container image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core
        "fastapi",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        # ML/Deep Learning
        "torch>=2.1.0",
        "transformers>=4.37.0,<4.46.0",
        "accelerate>=0.26.0",
        "safetensors>=0.4.0",
        # Interpretability
        "transformer-lens>=2.0.0",
        "sae-lens>=3.0.0",
        "einops>=0.7.0",
        "jaxtyping>=0.2.25",
        # Hugging Face
        "huggingface-hub>=0.20.0",
    )
)

# Model configurations (same as config.py)
MODEL_CONFIGS = {
    "gpt2-small": {
        "hf_model_id": "gpt2",
        "transformer_lens_name": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.{layer}.hook_resid_pre",
        "n_layers": 12,
        "d_model": 768,
        "requires_auth": False,
    },
    "gemma-2-2b": {
        "hf_model_id": "google/gemma-2-2b",
        "transformer_lens_name": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_{layer}/width_16k/canonical",
        "n_layers": 26,
        "d_model": 2304,
        "requires_auth": True,
    },
    "gemma-2-2b-it": {
        "hf_model_id": "google/gemma-2-2b-it",
        "transformer_lens_name": "gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_{layer}/width_16k/canonical",
        "n_layers": 26,
        "d_model": 2304,
        "requires_auth": True,
    },
    # Llama 3.1 8B is commented out until we have a working SAE configuration
    # The fnlp/Llama-Scope release doesn't exist in sae_lens pretrained SAEs
    # See: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
    # "llama-3.1-8b": {
    #     "hf_model_id": "meta-llama/Llama-3.1-8B",
    #     "transformer_lens_name": "meta-llama/Llama-3.1-8B",
    #     "sae_release": "???",  # No official sae_lens release available
    #     "sae_id": "???",
    #     "n_layers": 32,
    #     "d_model": 4096,
    #     "requires_auth": True,
    # },
}

# Volume for caching models (persistent across invocations)
model_cache = modal.Volume.from_name("kenr-model-cache", create_if_missing=True)
CACHE_DIR = "/models"


@app.cls(
    gpu="L40S",  # 48GB GPU - same as RunPod
    image=gpu_image,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("kenr-secrets")],
    timeout=600,  # 10 minutes for long operations
    scaledown_window=300,  # Keep warm for 5 minutes (renamed from container_idle_timeout)
)
class GPUService:
    """
    GPU-accelerated feature extraction service.

    Uses TransformerLens + SAELens to extract top-K features per token.
    """

    # Instance variables initialized in @modal.enter() instead of __init__
    # to comply with Modal's deprecation of custom constructors
    _model: object = None
    _current_model_id: str = None
    _saes: dict = None

    @modal.enter()
    def setup(self):
        """Called once when container starts. Initializes instance state."""
        import torch

        # Initialize instance variables (replaces __init__)
        self._model = None
        self._current_model_id = None
        self._saes = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GPU Service initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def _verify_api_key(self, provided_key: Optional[str]) -> bool:
        """Verify the API key."""
        expected_key = os.environ.get("API_KEY", "")
        if not expected_key:
            return True  # No key configured = allow all
        return provided_key == expected_key

    def _load_model(self, model_id: str):
        """Load a model if not already loaded."""
        if self._current_model_id == model_id and self._model is not None:
            return

        # Unload previous model
        if self._model is not None:
            import torch
            del self._model
            self._saes = {}
            torch.cuda.empty_cache()

        from transformer_lens import HookedTransformer

        config = MODEL_CONFIGS[model_id]

        print(f"Loading model: {model_id}")
        # HF_TOKEN is already set via Modal secrets (kenr-secrets)
        # TransformerLens automatically reads from HF_TOKEN environment variable
        # DO NOT pass token via hf_model_kwargs - it causes "unexpected keyword argument" errors
        # with certain model architectures (e.g., LlamaForCausalLM)
        hf_token = os.environ.get("HF_TOKEN", "")
        if config.get("requires_auth") and not hf_token:
            print(f"WARNING: Model {model_id} requires auth but HF_TOKEN not set")

        self._model = HookedTransformer.from_pretrained(
            config["transformer_lens_name"],
            device=self.device,
            cache_dir=CACHE_DIR,
        )
        self._current_model_id = model_id
        self._saes = {}
        print(f"Model {model_id} loaded")

    def _get_sae(self, model_id: str, layer: int):
        """Get or load SAE for a layer."""
        if layer in self._saes:
            return self._saes[layer]

        from sae_lens import SAE

        config = MODEL_CONFIGS[model_id]
        sae_release = config["sae_release"]
        sae_id = config["sae_id"].format(layer=layer)

        print(f"Loading SAE: {sae_release}/{sae_id}")
        # SAE.from_pretrained now returns only the SAE object (no unpacking needed)
        sae = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=self.device,
        )
        self._saes[layer] = sae
        return sae

    # === Internal helper methods (not Modal-decorated) ===

    def _do_health(self) -> Dict[str, Any]:
        """Health check implementation."""
        import torch

        gpu_available = torch.cuda.is_available()
        result = {
            "status": "healthy",
            "gpu_available": gpu_available,
        }

        if gpu_available:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["gpu_memory_total_gb"] = props.total_memory / (1024**3)
            result["gpu_memory_free_gb"] = (
                props.total_memory - torch.cuda.memory_allocated(0)
            ) / (1024**3)

        return result

    def _do_models(self) -> Dict[str, Any]:
        """List available models implementation."""
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

    def _do_topk(
        self,
        api_key: Optional[str],
        model_id: str,
        text: str,
        k: int = 10,
        layer: Optional[int] = None,
        include_descriptions: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract top-K features per token.

        This is the main endpoint for feature extraction.
        """
        import time
        import torch

        # Verify API key
        if not self._verify_api_key(api_key):
            return {"error": "Invalid or missing API key"}

        # Validate model
        if model_id not in MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"}

        if not text:
            return {"error": "text is required"}

        start_time = time.time()

        # Load model
        self._load_model(model_id)
        config = MODEL_CONFIGS[model_id]

        # Tokenize
        tokens = self._model.to_tokens(text, prepend_bos=True)
        str_tokens = self._model.to_str_tokens(text, prepend_bos=True)

        # Determine layers
        if layer is not None:
            layers = [layer]
        else:
            # Use middle layer by default
            layers = [config["n_layers"] // 2]

        # Run forward pass
        with torch.no_grad():
            _, cache = self._model.run_with_cache(tokens)

        # Extract features for each token
        token_features = []

        for pos, str_token in enumerate(str_tokens):
            all_features = []

            for lyr in layers:
                try:
                    sae = self._get_sae(model_id, lyr)
                except Exception as e:
                    print(f"Failed to load SAE for layer {lyr}: {e}")
                    continue

                # Get hidden state
                hook_name = f"blocks.{lyr}.hook_resid_post"
                hidden = cache[hook_name][0, pos, :]

                # Encode through SAE
                sae_out = sae.encode(hidden.unsqueeze(0))
                activations = sae_out[0]

                # Get top-k
                topk_vals, topk_indices = torch.topk(
                    activations, min(k, len(activations))
                )

                for val, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
                    if val > 0:
                        desc = f"Feature L{lyr}:{idx}" if include_descriptions else None
                        all_features.append({
                            "layer": lyr,
                            "feature_index": idx,  # KENR client expects feature_index, not index
                            "activation": val,
                            "description": desc,
                        })

            # Sort and take top-k across all layers
            all_features.sort(key=lambda x: x["activation"], reverse=True)

            # KENR client expects "features" key, not "top_features"
            token_features.append({
                "token": str_token,
                "position": pos,
                "features": all_features[:k],
            })

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Return format matching KENR GPUTopKResponse schema
        return {
            "model_id": model_id,
            "layer": layers[0],  # Primary layer used
            "tokens": list(str_tokens),
            "token_features": token_features,  # KENR expects token_features, not features_by_token
            "processing_time_ms": processing_time_ms,
        }

    def _do_compare(
        self,
        api_key: Optional[str],
        models: List[str],
        sentences: List[str],
        domain_terms: Optional[List[str]] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Compare multiple models on domain-specific content (implementation)."""
        import time
        import uuid

        # Verify API key
        if not self._verify_api_key(api_key):
            return {"error": "Invalid or missing API key"}

        if not models:
            return {"error": "models list is required"}

        for model_id in models:
            if model_id not in MODEL_CONFIGS:
                return {"error": f"Unknown model: {model_id}"}

        if not sentences:
            return {"error": "At least one sentence is required"}

        start_time = time.time()
        comparison_id = f"cmp_{uuid.uuid4().hex[:8]}"
        domain_terms_lower = [t.lower() for t in (domain_terms or [])]

        results = {}

        for model_id in models:
            try:
                # Analyze this model
                all_features = []
                domain_activations = []

                for sentence in sentences:
                    # Call internal helper, not Modal method
                    topk_result = self._do_topk(
                        api_key=api_key,
                        model_id=model_id,
                        text=sentence,
                        k=20,
                        include_descriptions=True,
                    )

                    if "error" in topk_result:
                        continue

                    for token_data in topk_result["features_by_token"]:
                        token_lower = token_data["token"].strip().lower()

                        for feature in token_data["top_features"]:
                            desc_lower = (feature.get("description") or "").lower()

                            is_domain = any(
                                term in token_lower or term in desc_lower
                                for term in domain_terms_lower
                            )

                            if is_domain:
                                domain_activations.append(feature["activation"])
                                all_features.append(feature)

                # Calculate metrics
                all_features.sort(key=lambda x: x["activation"], reverse=True)

                results[model_id] = {
                    "domain_feature_count": len(all_features),
                    "avg_domain_activation": (
                        sum(domain_activations) / len(domain_activations)
                        if domain_activations else 0.0
                    ),
                    "prediction_accuracy": min(len(all_features) / 10, 1.0),
                    "notable_features": all_features[:10],
                }

            except Exception as e:
                print(f"Failed to analyze model {model_id}: {e}")
                results[model_id] = {
                    "domain_feature_count": 0,
                    "avg_domain_activation": 0.0,
                    "prediction_accuracy": 0.0,
                    "notable_features": [],
                }

        # Generate recommendation
        if results:
            scores = {}
            for model_id, result in results.items():
                scores[model_id] = (
                    result["domain_feature_count"] * 0.3 +
                    result["avg_domain_activation"] * 10 * 0.4 +
                    result["prediction_accuracy"] * 100 * 0.3
                )

            best_model = max(scores, key=scores.get)
            best_result = results[best_model]

            recommendation = {
                "best_model": best_model,
                "reason": (
                    f"Highest domain feature activation ({best_result['avg_domain_activation']:.2f}) "
                    f"with {best_result['domain_feature_count']} domain-relevant features detected"
                ),
            }
        else:
            recommendation = {
                "best_model": "unknown",
                "reason": "No models were successfully analyzed",
            }

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "comparison_id": comparison_id,
            "results": results,
            "recommendation": recommendation,
            "processing_time_ms": processing_time_ms,
        }

    # === Modal method wrappers (call internal helpers) ===

    @modal.method()
    def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return self._do_health()

    @modal.method()
    def models(self) -> Dict[str, Any]:
        """List available models."""
        return self._do_models()

    @modal.method()
    def topk(
        self,
        api_key: Optional[str],
        model_id: str,
        text: str,
        k: int = 10,
        layer: Optional[int] = None,
        include_descriptions: bool = True,
    ) -> Dict[str, Any]:
        """Extract top-K features per token."""
        return self._do_topk(api_key, model_id, text, k, layer, include_descriptions)

    @modal.method()
    def compare(
        self,
        api_key: Optional[str],
        models: List[str],
        sentences: List[str],
        domain_terms: Optional[List[str]] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Compare multiple models on domain-specific content."""
        return self._do_compare(api_key, models, sentences, domain_terms, k)

    @modal.fastapi_endpoint(method="GET")
    def web_health(self):
        """HTTP GET /health endpoint."""
        return self._do_health()

    @modal.fastapi_endpoint(method="GET")
    def web_models(self):
        """HTTP GET /models endpoint."""
        return self._do_models()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web_topk(self, item: dict):
        """HTTP POST /topk endpoint."""
        return self._do_topk(
            api_key=item.get("api_key"),
            model_id=item.get("model_id", ""),
            text=item.get("text", ""),
            k=item.get("k", 10),
            layer=item.get("layer"),
            include_descriptions=item.get("include_descriptions", True),
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web_compare(self, item: dict):
        """HTTP POST /compare endpoint."""
        return self._do_compare(
            api_key=item.get("api_key"),
            models=item.get("models", []),
            sentences=item.get("sentences", []),
            domain_terms=item.get("domain_terms"),
            k=item.get("k", 10),
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def runsync(self, item: dict):
        """
        RunPod-compatible endpoint.

        Accepts the same format as RunPod serverless:
        POST /runsync with {"input": {"action": "...", ...}}
        """
        input_data = item.get("input", {})
        action = input_data.get("action", "topk")
        api_key = input_data.get("api_key")

        if action == "health":
            output = self._do_health()
        elif action == "models":
            output = self._do_models()
        elif action == "topk":
            output = self._do_topk(
                api_key=api_key,
                model_id=input_data.get("model_id", ""),
                text=input_data.get("text", ""),
                k=input_data.get("k", 10),
                layer=input_data.get("layer"),
                include_descriptions=input_data.get("include_descriptions", True),
            )
        elif action == "compare":
            output = self._do_compare(
                api_key=api_key,
                models=input_data.get("models", []),
                sentences=input_data.get("sentences", []),
                domain_terms=input_data.get("domain_terms"),
                k=input_data.get("k", 10),
            )
        else:
            output = {"error": f"Unknown action: {action}"}

        # Return in RunPod format
        if isinstance(output, dict) and "error" in output:
            return {"status": "FAILED", "error": output["error"]}
        return {"status": "COMPLETED", "output": output}


# CLI for local testing
@app.local_entrypoint()
def main():
    """Test the service locally."""
    print("Testing KENR GPU Service...")

    service = GPUService()

    # Test health
    print("\n--- Health Check ---")
    health = service.health.remote()
    print(health)

    # Test models
    print("\n--- Available Models ---")
    models = service.models.remote()
    print(models)

    # Test top-k (small model for quick test)
    print("\n--- Top-K Test (gpt2-small) ---")
    result = service.topk.remote(
        api_key=None,
        model_id="gpt2-small",
        text="The quantum entanglement experiment succeeded.",
        k=5,
    )
    print(f"Tokens: {result.get('tokens', [])}")
    print(f"Processing time: {result.get('processing_time_ms', 0)}ms")
