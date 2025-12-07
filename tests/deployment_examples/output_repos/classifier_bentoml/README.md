# Text Classification - BentoML Cloud Deployment

Production-ready ML text classification service deployed to BentoCloud with batching support.

## Structure

```
.
├── main.py            # Main entry point with predict() function
├── classifier.py      # Text classifier implementation
├── service.py         # BentoML service definition
├── bentofile.yaml     # BentoML build configuration
├── deploy.py          # BentoCloud deployment script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Local Testing

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Test the Predict Function

```bash
# Run with default example
python main.py

# Test with JSON input
cat > test.json << 'EOF'
{"text": "This is amazing!"}
EOF
python main.py < test.json

# Test batch prediction
cat > batch.json << 'EOF'
{"texts": ["I love this!", "Terrible product", "Great quality"]}
EOF
python main.py < batch.json
```

### Test BentoML Service Locally

```bash
# Start the service locally
bentoml serve service:TextClassifierService

# In another terminal, test the endpoints
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'

curl -X POST http://localhost:3000/predict_batch \
  -H "Content-Type: application/json" \
  -d '["Great product", "Terrible quality"]'

curl http://localhost:3000/health
```

## BentoCloud Deployment

### Prerequisites

1. Sign up for BentoCloud account at https://cloud.bentoml.com
2. Get your API key from the BentoCloud dashboard
3. Set environment variables:

```bash
export BENTO_CLOUD_API_KEY="your-api-key-here"
export BENTO_CLOUD_API_ENDPOINT="https://cloud.bentoml.com"  # Optional
```

### Deploy

#### Option 1: Using the deploy.py script

```bash
python deploy.py
```

#### Option 2: Manual deployment

```bash
# Login to BentoCloud
bentoml cloud login --api-token $BENTO_CLOUD_API_KEY --endpoint $BENTO_CLOUD_API_ENDPOINT

# Deploy the service
bentoml deploy .
```

### Test the Deployed Service

After deployment, you'll receive an endpoint URL. Test it:

```bash
ENDPOINT_URL="your-endpoint-url-here"

# Single prediction
curl -X POST $ENDPOINT_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'

# Batch prediction (with automatic batching)
curl -X POST $ENDPOINT_URL/predict_batch \
  -H "Content-Type: application/json" \
  -d '["Great product", "Terrible quality", "Best purchase ever"]'

# Health check
curl $ENDPOINT_URL/health
```

## API Reference

### `/predict`

Single text classification endpoint.

**Input:**
```json
{
  "text": "Your text here"
}
```

**Output:**
```json
{
  "status": "success",
  "output": {
    "text": "Your text here",
    "label": "positive",
    "confidence": 0.8542
  }
}
```

### `/predict_batch`

Batch prediction endpoint with automatic batching for throughput optimization.

**Input:**
```json
["Text 1", "Text 2", "Text 3"]
```

**Output:**
```json
[
  {
    "text": "Text 1",
    "label": "positive",
    "confidence": 0.8542
  },
  {
    "text": "Text 2",
    "label": "negative",
    "confidence": 0.7123
  }
]
```

### `/health`

Health check endpoint.

**Output:**
```json
{
  "status": "healthy"
}
```

## Features

- ✅ **Auto-scaling**: BentoCloud automatically scales based on traffic
- ✅ **Batching**: Automatic request batching for optimal throughput
- ✅ **Monitoring**: Built-in metrics and logging on BentoCloud dashboard
- ✅ **Serverless**: Pay only for what you use
- ✅ **Production-ready**: Configured for 2 CPU cores and 4GB memory

## Resource Configuration

The service is configured with the following resources (in `service.py`):

- **CPU**: 2 cores
- **Memory**: 4Gi
- **Timeout**: 300 seconds
- **Batch size**: Max 32 requests
- **Batch latency**: Max 1000ms

Adjust these in `service.py` as needed for your workload.
