# BentoML

## Summary
Production ML service deployment on BentoCloud with batching and monitoring.

## Best For
- Production ML model serving
- Need automatic request batching
- Production monitoring and observability
- Managed ML infrastructure
- Model versioning and A/B testing

## Not For
- Quick development (use local)
- Simple scripts (use local)
- LangGraph agents (use langgraph)
- GPU-heavy serverless (use modal)

## Resources
Requires resource specification:
- cpu: 1, 2, 4
- memory: 2Gi, 4Gi, 8Gi
- gpu: 0, 1 (optional)

Default: cpu=2, memory=4Gi

## Interface
http (BentoCloud API endpoint)

## Provider
bentocloud

