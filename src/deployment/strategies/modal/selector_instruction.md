# Modal

## Summary
Serverless GPU deployment on Modal.com with auto-scaling.

## Best For
- GPU workloads (PyTorch, TensorFlow, CUDA)
- ML model inference at scale
- Serverless auto-scaling
- Pay-per-use pricing
- Fast cold starts for ML

## Not For
- Simple local scripts (use local)
- Persistent HTTP servers (use docker)
- LangGraph/LangChain agents (use langgraph)
- Need on-premise deployment (use docker)

## Resources
Requires resource specification:
- gpu: T4, L4, A10G, A100, H100
- memory: 8Gi, 16Gi, 32Gi

Default: gpu=T4, memory=16Gi

## Interface
function (Modal SDK remote call)

## Provider
modal

