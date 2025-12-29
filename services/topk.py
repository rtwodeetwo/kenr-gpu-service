"""
Top-K feature extraction service.
"""

import time
import logging
from typing import Optional, List

from sae import get_feature_extractor
from schemas import (
    TopKRequest,
    TopKResponse,
    TokenFeatures,
    FeatureActivation,
)

logger = logging.getLogger(__name__)


class TopKService:
    """Service for top-K feature extraction."""

    def __init__(self):
        self._extractor = get_feature_extractor()

    def extract_topk(self, request: TopKRequest) -> TopKResponse:
        """
        Extract top-K features for each token in the text.
        """
        start_time = time.time()

        # Determine layers to process
        layers = [request.layer] if request.layer is not None else None

        # Extract features
        tokens, token_results = self._extractor.get_topk_features(
            model_id=request.model_id,
            text=request.text,
            k=request.k,
            layers=layers,
            include_descriptions=request.include_descriptions,
        )

        # Convert to response format
        features_by_token = []
        for result in token_results:
            features_by_token.append(TokenFeatures(
                token=result.token,
                position=result.position,
                top_features=[
                    FeatureActivation(
                        layer=f.layer,
                        index=f.index,
                        activation=f.activation,
                        description=f.description,
                    )
                    for f in result.features
                ]
            ))

        processing_time_ms = int((time.time() - start_time) * 1000)

        return TopKResponse(
            model_id=request.model_id,
            tokens=tokens,
            features_by_token=features_by_token,
            processing_time_ms=processing_time_ms,
        )
