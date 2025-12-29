"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# === Health Check ===

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    gpu_available: bool
    gpu_memory_free_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    loaded_models: List[str] = Field(default_factory=list)


# === Feature Extraction ===

class FeatureActivation(BaseModel):
    """A single feature activation."""
    layer: int
    index: int
    activation: float
    description: Optional[str] = None


class TokenFeatures(BaseModel):
    """Features for a single token."""
    token: str
    position: int
    top_features: List[FeatureActivation]


class TopKRequest(BaseModel):
    """Request for top-K feature extraction."""
    model_id: str
    text: str
    layer: Optional[int] = None  # None = all layers
    k: int = Field(default=10, ge=1, le=50)
    include_descriptions: bool = True


class TopKResponse(BaseModel):
    """Response from top-K feature extraction."""
    model_id: str
    tokens: List[str]
    features_by_token: List[TokenFeatures]
    processing_time_ms: int


# === Circuit Tracing ===

class CircuitNode(BaseModel):
    """Node in an attribution circuit."""
    id: str
    layer: int
    feature_index: int
    description: Optional[str] = None
    activation: float
    attribution_to_output: float


class CircuitEdge(BaseModel):
    """Edge in an attribution circuit."""
    source: str
    target: str
    weight: float


class TokenPrediction(BaseModel):
    """A predicted token with probability."""
    token: str
    probability: float


class CircuitTraceRequest(BaseModel):
    """Request for circuit tracing."""
    model_id: str
    text: str
    target_token_index: int = -1  # -1 = last token
    threshold: float = Field(default=0.01, ge=0.001, le=0.5)
    max_nodes: int = Field(default=100, ge=10, le=500)


class CircuitTraceResponse(BaseModel):
    """Response from circuit tracing."""
    model_id: str
    tokens: List[str]
    target_token: str
    top_predictions: List[TokenPrediction]
    nodes: List[CircuitNode]
    edges: List[CircuitEdge]
    processing_time_ms: int


# === Model Comparison ===

class ModelResult(BaseModel):
    """Results for a single model in comparison."""
    domain_feature_count: int
    avg_domain_activation: float
    prediction_accuracy: float
    notable_features: List[FeatureActivation]


class ComparisonRecommendation(BaseModel):
    """Recommendation from model comparison."""
    best_model: str
    reason: str


class CompareRequest(BaseModel):
    """Request for model comparison."""
    models: List[str]
    sentences: List[str]
    domain_terms: List[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    """Response from model comparison."""
    comparison_id: str
    results: Dict[str, ModelResult]
    recommendation: ComparisonRecommendation
    processing_time_ms: int


# === Steering ===

class FeatureConfig(BaseModel):
    """Configuration for a single feature to steer."""
    layer: int
    index: int
    strength: float = Field(ge=-2.0, le=2.0)


class SteeringComputeRequest(BaseModel):
    """Request to compute steering vector."""
    model_id: str
    amplify_features: List[FeatureConfig] = Field(default_factory=list)
    suppress_features: List[FeatureConfig] = Field(default_factory=list)


class SteeringComputeResponse(BaseModel):
    """Response from steering vector computation."""
    steering_vector_id: str
    model_id: str
    vector_shape: List[int]
    ready: bool


class SteeringGenerateRequest(BaseModel):
    """Request for steered generation."""
    model_id: str
    steering_vector_id: Optional[str] = None
    features: Optional[List[FeatureConfig]] = None  # Inline alternative
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class SteeringGenerateResponse(BaseModel):
    """Response from steered generation."""
    generated_text: str
    tokens: List[str]
    feature_activations: Optional[List[TokenFeatures]] = None
    processing_time_ms: int


# === Error Response ===

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
