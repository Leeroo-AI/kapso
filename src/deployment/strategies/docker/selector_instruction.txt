# Docker

## Summary
Run in an isolated Docker container with HTTP API.

## Best For
- Reproducible deployments
- Isolated environments
- HTTP-based APIs
- Local testing of production setup
- CPU-only workloads with network access

## Not For
- Quick development iteration (use local)
- GPU workloads (use modal)
- Serverless auto-scaling (use modal or bentoml)
- Stateful agents (use langgraph)

## Resources
No resource specification needed (managed by Docker).

## Interface
http (FastAPI endpoint)

## Provider
None (local Docker)

