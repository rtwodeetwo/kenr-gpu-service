# KENR GPU Service

GPU-accelerated interpretability service for KENR. Provides feature extraction, circuit tracing, and model comparison using TransformerLens and SAELens.

## Features

- **Top-K Feature Extraction**: Get the most active SAE features per token
- **Model Comparison**: Compare how different models process domain-specific content
- **Circuit Tracing**: Attribution graphs showing feature interactions (coming soon)
- **Steering**: Compute and apply steering vectors (coming soon)

## Supported Models

| Model | Parameters | VRAM Required |
|-------|------------|---------------|
| GPT-2 Small | 124M | ~1 GB |
| Gemma 2 2B | 2.6B | ~5 GB |
| Gemma 2 2B IT | 2.6B | ~5 GB |
| Llama 3.1 8B | 8B | ~16 GB |

## Local Development

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Hugging Face account with access to gated models

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd kenr-gpu-service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your API key and HF token

# Run the server
python main.py
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Get top-K features (requires API key)
curl -X POST http://localhost:8000/features/topk \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{
    "model_id": "gpt2-small",
    "text": "The plasma reached high temperature",
    "k": 10
  }'
```

## RunPod Deployment

### Build and Push Docker Image

```bash
# Build image
docker build -t kenr-gpu-service:latest .

# Tag for your registry
docker tag kenr-gpu-service:latest your-registry/kenr-gpu-service:latest

# Push
docker push your-registry/kenr-gpu-service:latest
```

### RunPod Serverless Template

1. Go to RunPod Serverless
2. Create new template:
   - Container Image: `your-registry/kenr-gpu-service:latest`
   - GPU Type: A100 40GB (or RTX 4090 for smaller models)
   - Container Disk: 50 GB
   - Volume Mount: `/models`
3. Set environment variables:
   - `API_KEY`: Your secure API key
   - `HF_TOKEN`: Your Hugging Face token

## API Reference

### GET /health

Health check endpoint. Returns GPU status and loaded models.

### GET /models

List available models and their configurations.

### POST /features/topk

Extract top-K SAE features per token.

**Request:**
```json
{
  "model_id": "gemma-2-2b",
  "text": "Your input text here",
  "layer": null,
  "k": 10,
  "include_descriptions": true
}
```

### POST /compare

Compare models on domain-specific sentences.

**Request:**
```json
{
  "models": ["gemma-2-2b", "llama-3.1-8b"],
  "sentences": ["The tokamak achieved H-mode"],
  "domain_terms": ["tokamak", "plasma", "confinement"]
}
```

## Architecture

```
kenr-gpu-service/
├── main.py              # FastAPI application
├── config.py            # Configuration & model configs
├── schemas.py           # Pydantic request/response models
├── models/
│   └── loader.py        # Model loading with TransformerLens
├── sae/
│   └── activation.py    # Feature extraction with SAELens
├── services/
│   ├── topk.py          # Top-K extraction service
│   └── compare.py       # Model comparison service
├── Dockerfile           # RunPod deployment
└── requirements.txt     # Python dependencies
```

## Cost Estimates

| Usage Level | Monthly Cost |
|-------------|--------------|
| Development | ~$8-15 |
| Light Use | ~$18-30 |
| Heavy Research | ~$50-100 |

Costs are based on RunPod Serverless pricing (~$0.00031/second for A100 40GB).
# Trigger rebuild
