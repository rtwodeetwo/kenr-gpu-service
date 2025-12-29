"""
Feature extraction using SAEs.

Extracts top-K feature activations per token from model hidden states.
"""

import logging
import torch
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from transformer_lens import HookedTransformer
from sae_lens import SAE

from models import get_model_manager
from config import MODEL_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """Result for a single feature activation."""
    layer: int
    index: int
    activation: float
    description: Optional[str] = None


@dataclass
class TokenResult:
    """Result for a single token."""
    token: str
    position: int
    features: List[FeatureResult]


class FeatureExtractor:
    """
    Extracts feature activations from model hidden states using SAEs.
    """

    def __init__(self):
        self._model_manager = get_model_manager()

    def get_topk_features(
        self,
        model_id: str,
        text: str,
        k: int = 10,
        layers: Optional[List[int]] = None,
        include_descriptions: bool = True,
    ) -> Tuple[List[str], List[TokenResult]]:
        """
        Get top-K feature activations per token.

        Args:
            model_id: Model to use
            text: Input text
            k: Number of top features per token
            layers: Specific layers to analyze (None = all)
            include_descriptions: Whether to include feature descriptions

        Returns:
            Tuple of (tokens, token_results)
        """
        # Get model and config
        model = self._model_manager.get_model(model_id)
        config = MODEL_CONFIGS[model_id]

        # Tokenize
        tokens = model.to_tokens(text, prepend_bos=True)
        str_tokens = model.to_str_tokens(text, prepend_bos=True)

        # Determine which layers to analyze
        if layers is None:
            # For efficiency, just use a middle layer for now
            # Full layer analysis would be expensive
            middle_layer = config["n_layers"] // 2
            layers = [middle_layer]

        # Run forward pass and cache activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Process each token
        token_results = []

        for pos, str_token in enumerate(str_tokens):
            all_features = []

            for layer in layers:
                # Get SAE for this layer
                try:
                    sae = self._model_manager.get_sae(model_id, layer)
                except Exception as e:
                    logger.warning(f"Failed to load SAE for layer {layer}: {e}")
                    continue

                # Get hidden state for this position
                hook_name = self._get_hook_name(model_id, layer)
                hidden = cache[hook_name][0, pos, :]  # [d_model]

                # Encode through SAE
                sae_out = sae.encode(hidden.unsqueeze(0))  # [1, n_features]
                activations = sae_out[0]  # [n_features]

                # Get top-k features for this layer
                topk_vals, topk_indices = torch.topk(activations, min(k, len(activations)))

                for val, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
                    if val > 0:  # Only include non-zero activations
                        description = None
                        if include_descriptions:
                            description = self._get_feature_description(model_id, layer, idx)

                        all_features.append(FeatureResult(
                            layer=layer,
                            index=idx,
                            activation=val,
                            description=description,
                        ))

            # Sort by activation and take top-k across all layers
            all_features.sort(key=lambda x: x.activation, reverse=True)
            top_features = all_features[:k]

            token_results.append(TokenResult(
                token=str_token,
                position=pos,
                features=top_features,
            ))

        return list(str_tokens), token_results

    def get_all_layer_features(
        self,
        model_id: str,
        text: str,
        k: int = 5,
    ) -> Tuple[List[str], Dict[int, List[TokenResult]]]:
        """
        Get top-K features for all layers (expensive operation).

        Returns features organized by layer.
        """
        config = MODEL_CONFIGS[model_id]
        all_layers = list(range(config["n_layers"]))

        results_by_layer = {}
        tokens = None

        for layer in all_layers:
            try:
                tokens, results = self.get_topk_features(
                    model_id, text, k=k, layers=[layer], include_descriptions=True
                )
                results_by_layer[layer] = results
            except Exception as e:
                logger.warning(f"Failed to process layer {layer}: {e}")
                continue

        return tokens or [], results_by_layer

    def _get_hook_name(self, model_id: str, layer: int) -> str:
        """Get the TransformerLens hook name for residual stream at a layer."""
        # Standard hook name for residual stream
        return f"blocks.{layer}.hook_resid_post"

    def _get_feature_description(
        self,
        model_id: str,
        layer: int,
        index: int
    ) -> Optional[str]:
        """
        Get a human-readable description for a feature.

        For now, returns a placeholder. Could be extended to:
        - Load from a local database of descriptions
        - Query Neuronpedia API
        - Use auto-interp descriptions from SAE metadata
        """
        # TODO: Load actual descriptions from SAE metadata or external source
        return f"Feature L{layer}:{index}"


# Singleton instance
_feature_extractor: Optional[FeatureExtractor] = None


def get_feature_extractor() -> FeatureExtractor:
    """Get the singleton feature extractor instance."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor
