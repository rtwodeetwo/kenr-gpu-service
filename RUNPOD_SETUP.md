# RunPod Serverless Setup Guide

This guide walks you through deploying the KENR GPU Service to RunPod Serverless.

## Prerequisites

1. RunPod account (you have this)
2. Hugging Face account with access to Llama 3.1 8B (required for model weights)
3. The GPU service code is already on GitHub: https://github.com/rtwodeetwo/kenr-gpu-service

## Step 1: Connect GitHub to RunPod

1. Go to https://www.runpod.io/console/serverless
2. Click **"New Endpoint"**
3. Under **"Import Git Repository"**, click **"Connect GitHub"**
4. Authorize RunPod to access your GitHub repositories
5. Select the `kenr-gpu-service` repository

## Step 2: Configure the Endpoint

After selecting the repository, configure these settings:

### Basic Settings
- **Name**: `kenr-gpu-service`
- **GPU Type**: Select **"NVIDIA A40"** (48GB VRAM, good balance of cost and capability)
  - Alternative: RTX 4090 (24GB, cheaper but may struggle with larger models)

### Scaling Settings
- **Min Workers**: `0` (scales to zero when not in use - saves money)
- **Max Workers**: `1` (adjust based on usage)
- **Idle Timeout**: `30` seconds (how long to wait before scaling down)

### Environment Variables
Click "Add Environment Variable" for each:

| Variable | Value | Notes |
|----------|-------|-------|
| `API_KEY` | Generate a secure key | Use: `openssl rand -hex 32` |
| `HF_TOKEN` | Your Hugging Face token | Get from https://huggingface.co/settings/tokens |

### Advanced Settings
- **Container Disk**: `50 GB` (for model weights)
- **Volume Mount Path**: `/runpod-volume` (for persistent model storage)

## Step 3: Deploy

1. Click **"Deploy"**
2. Wait for the build to complete (first build takes ~10-15 minutes)
3. Note your endpoint URL - it will look like: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/...`

## Step 4: Test the Endpoint

Once deployed, test with:

```bash
# Health check
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'

# Top-K extraction (requires API_KEY you set)
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "topk",
      "api_key": "YOUR_GPU_SERVICE_API_KEY",
      "model_id": "llama-3.1-8b",
      "text": "The tokamak achieved stable plasma confinement",
      "k": 10
    }
  }'
```

## Step 5: Configure KENR

Once the endpoint is working, add these environment variables to Railway:

1. Go to your Railway project: https://railway.app/project/...
2. Click on your service → Variables
3. Add:
   - `GPU_SERVICE_URL`: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID`
   - `GPU_SERVICE_API_KEY`: The API_KEY you set in Step 2

## Cost Estimates

RunPod Serverless pricing (A40):
- ~$0.00069/second when running
- $0 when scaled to zero

Typical usage:
- Single comparison (3 models, 10 sentences): ~30-60 seconds = ~$0.02-0.04
- Light development use: ~$10-20/month
- Regular research use: ~$30-50/month

## Troubleshooting

### Build fails
- Check the build logs in RunPod console
- Common issue: Missing dependencies in requirements.txt

### Cold starts are slow
- First request after idle takes 30-60 seconds to load models
- Increase idle timeout if this is problematic
- Consider min workers = 1 if you need instant responses (but costs more)

### Model loading errors
- Ensure HF_TOKEN has access to Llama 3.1 8B
- Check that you've accepted the model license on Hugging Face

### Out of memory
- Llama 3.1 8B needs ~16GB VRAM
- Ensure you're using A40 (48GB) or larger GPU
- If using RTX 4090 (24GB), only run one model at a time

## API Reference

### Health Check
```json
{"input": {"action": "health"}}
```

### List Models
```json
{"input": {"action": "models"}}
```

### Top-K Features
```json
{
  "input": {
    "action": "topk",
    "api_key": "your-api-key",
    "model_id": "llama-3.1-8b",
    "text": "Your input text",
    "k": 10,
    "include_descriptions": true
  }
}
```

### Compare Models
```json
{
  "input": {
    "action": "compare",
    "api_key": "your-api-key",
    "models": ["gemma-2-2b", "llama-3.1-8b"],
    "sentences": ["Test sentence 1", "Test sentence 2"],
    "domain_terms": ["term1", "term2"]
  }
}
```
