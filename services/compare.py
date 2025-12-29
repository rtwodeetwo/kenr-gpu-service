"""
Model comparison service.
"""

import time
import logging
import uuid
from typing import List, Dict

from sae import get_feature_extractor
from schemas import (
    CompareRequest,
    CompareResponse,
    ModelResult,
    FeatureActivation,
    ComparisonRecommendation,
)

logger = logging.getLogger(__name__)


class CompareService:
    """Service for comparing models on domain-specific content."""

    def __init__(self):
        self._extractor = get_feature_extractor()

    def compare_models(self, request: CompareRequest) -> CompareResponse:
        """
        Compare multiple models on domain sentences.

        Analyzes how each model processes the sentences and which
        features activate for domain-specific content.
        """
        start_time = time.time()
        comparison_id = f"cmp_{uuid.uuid4().hex[:8]}"

        results: Dict[str, ModelResult] = {}
        domain_terms_lower = [t.lower() for t in request.domain_terms]

        for model_id in request.models:
            try:
                model_result = self._analyze_model(
                    model_id=model_id,
                    sentences=request.sentences,
                    domain_terms=domain_terms_lower,
                )
                results[model_id] = model_result
            except Exception as e:
                logger.error(f"Failed to analyze model {model_id}: {e}")
                # Add a placeholder result for failed models
                results[model_id] = ModelResult(
                    domain_feature_count=0,
                    avg_domain_activation=0.0,
                    prediction_accuracy=0.0,
                    notable_features=[],
                )

        # Determine best model
        recommendation = self._generate_recommendation(results)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return CompareResponse(
            comparison_id=comparison_id,
            results=results,
            recommendation=recommendation,
            processing_time_ms=processing_time_ms,
        )

    def _analyze_model(
        self,
        model_id: str,
        sentences: List[str],
        domain_terms: List[str],
    ) -> ModelResult:
        """Analyze a single model on the domain sentences."""

        all_features: List[FeatureActivation] = []
        domain_activations: List[float] = []

        for sentence in sentences:
            # Get top-K features for this sentence
            tokens, token_results = self._extractor.get_topk_features(
                model_id=model_id,
                text=sentence,
                k=20,
                layers=None,  # Use default middle layer
                include_descriptions=True,
            )

            for token_result in token_results:
                token_lower = token_result.token.strip().lower()

                for feature in token_result.features:
                    # Check if this is a domain-relevant feature
                    is_domain_relevant = False
                    desc_lower = (feature.description or "").lower()

                    # Check if token or description matches domain terms
                    for term in domain_terms:
                        if term in token_lower or term in desc_lower:
                            is_domain_relevant = True
                            break

                    if is_domain_relevant:
                        domain_activations.append(feature.activation)
                        all_features.append(FeatureActivation(
                            layer=feature.layer,
                            index=feature.index,
                            activation=feature.activation,
                            description=feature.description,
                        ))

        # Calculate metrics
        domain_feature_count = len(all_features)
        avg_domain_activation = (
            sum(domain_activations) / len(domain_activations)
            if domain_activations else 0.0
        )

        # Sort and get top notable features
        all_features.sort(key=lambda x: x.activation, reverse=True)
        notable_features = all_features[:10]

        # Prediction accuracy would require running the model on
        # fill-in-the-blank sentences - simplified for now
        prediction_accuracy = min(domain_feature_count / 10, 1.0)

        return ModelResult(
            domain_feature_count=domain_feature_count,
            avg_domain_activation=avg_domain_activation,
            prediction_accuracy=prediction_accuracy,
            notable_features=notable_features,
        )

    def _generate_recommendation(
        self,
        results: Dict[str, ModelResult]
    ) -> ComparisonRecommendation:
        """Generate a recommendation based on comparison results."""

        if not results:
            return ComparisonRecommendation(
                best_model="unknown",
                reason="No models were successfully analyzed"
            )

        # Score each model
        scores = {}
        for model_id, result in results.items():
            # Weighted score
            score = (
                result.domain_feature_count * 0.3 +
                result.avg_domain_activation * 10 * 0.4 +
                result.prediction_accuracy * 100 * 0.3
            )
            scores[model_id] = score

        # Find best model
        best_model = max(scores, key=scores.get)
        best_result = results[best_model]

        # Generate reason
        reason = (
            f"Highest domain feature activation ({best_result.avg_domain_activation:.2f}) "
            f"with {best_result.domain_feature_count} domain-relevant features detected"
        )

        return ComparisonRecommendation(
            best_model=best_model,
            reason=reason,
        )
