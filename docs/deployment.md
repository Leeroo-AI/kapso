# Deployment Module

Deploy your ML solutions with a unified interface. The deployment module handles the complexity of different deployment targets while providing a simple, consistent API.

## Overview

The deployment system works in three phases:

1. **Selection**: Choose or auto-detect the best deployment strategy
2. **Adaptation**: Transform your code for the target platform
3. **Running**: Execute with a unified interface regardless of backend

```
Solution Repository
       ↓
[Selector Agent] → Chooses: LOCAL, DOCKER, MODAL, or BENTOML
       ↓
[Adapter Agent] → Adds Dockerfile, modal_app.py, etc.
       ↓
[Runner] → FunctionRunner, HTTPRunner, ModalRunner
       ↓
    Software → Unified interface: run(), stop(), logs()
```

## Quick Start

### Basic Usage

```python
from src.deployment import DeploymentFactory, DeployStrategy, DeployConfig

# Configure deployment
config = DeployConfig(
    code_path="./my_solution",
    goal="Sentiment analysis API",
)

# Create deployment (AUTO selects best strategy)
software = DeploymentFactory.create(DeployStrategy.AUTO, config)

# Run predictions
result = software.run({"text": "I love this product!"})
print(result)
# {"status": "success", "output": {"sentiment": "positive", ...}}

# Clean up
software.stop()
```

### Using Context Manager

```python
with DeploymentFactory.create(DeployStrategy.LOCAL, config) as software:
    result = software.run({"input": "data"})
    print(result)
# Automatically stops when exiting context
```

### Specifying Strategy

```python
# Force specific strategy
software = DeploymentFactory.create(DeployStrategy.DOCKER, config)
software = DeploymentFactory.create(DeployStrategy.MODAL, config)
software = DeploymentFactory.create(DeployStrategy.BENTOML, config)
```

## Deployment Strategies

### LOCAL

Run directly as a Python process. Best for development and testing.

**Requirements**: None (just Python)

**Use when**:
- Developing and testing locally
- Simple scripts without heavy dependencies
- No GPU required

```python
software = DeploymentFactory.create(DeployStrategy.LOCAL, config)
result = software.run({"text": "hello"})
```

### DOCKER

Package as a Docker container. Best for consistent environments and microservices.

**Requirements**: Docker installed

**Use when**:
- Need consistent environment across machines
- Deploying as microservice
- HTTP API interface needed

```python
software = DeploymentFactory.create(DeployStrategy.DOCKER, config)

# Get deployment commands
print(software.get_deploy_command())
# docker build -t solution . && docker run -p 8000:8000 solution

# After running Docker container:
result = software.run({"input": "data"})
```

**Generated Files**:
- `Dockerfile` - Container definition
- `app.py` - FastAPI application (if HTTP interface)

### MODAL

Deploy to Modal for serverless GPU computing. Best for ML models needing GPUs.

**Requirements**:
- `pip install modal`
- Modal account: `modal token new`
- Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`

**Use when**:
- Need GPU acceleration
- Serverless scaling
- Pay-per-use model

```python
software = DeploymentFactory.create(DeployStrategy.MODAL, config)

# Get deployment command
print(software.get_deploy_command())
# modal deploy modal_app.py

# After deployment:
result = software.run({"text": "hello"})
```

**Generated Files**:
- `modal_app.py` - Modal application definition

### BENTOML

Deploy with BentoML for production ML serving. Best for ML model deployment with batching.

**Requirements**:
- `pip install bentoml`

**Use when**:
- Production ML model serving
- Need request batching
- Want model versioning

```python
software = DeploymentFactory.create(DeployStrategy.BENTOML, config)

# Get deployment commands
print(software.get_deploy_command())
# bentoml build && bentoml serve

# After running BentoML:
result = software.run({"question": "What is Python?", "context": "..."})
```

**Generated Files**:
- `service.py` - BentoML service definition
- `bentofile.yaml` - Build configuration

## Strategy Selection Guide

| Strategy | Best For | GPU | Complexity | Cost |
|----------|----------|-----|------------|------|
| LOCAL | Development, testing | No | Low | Free |
| DOCKER | Microservices, APIs | No | Medium | Varies |
| MODAL | ML inference, GPU tasks | Yes | Medium | Pay-per-use |
| BENTOML | Production ML serving | Optional | Medium | Self-hosted |

### AUTO Strategy

Let the system choose the best strategy:

```python
software = DeploymentFactory.create(DeployStrategy.AUTO, config)
print(f"Selected: {software.name}")  # Shows chosen strategy
```

The selector considers:
- Dependencies (torch, tensorflow → GPU likely needed)
- File structure (Dockerfile, modal_app.py → use existing)
- Goal description
- Available resources

## Credentials & Setup

### LOCAL

No setup required.

### DOCKER

```bash
# Install Docker
# macOS: brew install docker
# Ubuntu: sudo apt install docker.io

# Verify
docker --version
```

### MODAL

```bash
# Install
pip install modal

# Authenticate (opens browser)
modal token new

# Or set environment variables
export MODAL_TOKEN_ID=your_id
export MODAL_TOKEN_SECRET=your_secret
```

### BENTOML

```bash
# Install
pip install bentoml

# No authentication needed for local serving
```

## API Reference

### DeploymentFactory

```python
DeploymentFactory.create(
    strategy: DeployStrategy,  # Deployment target
    config: DeployConfig,      # Configuration
    coding_agent: str = "claude_code",  # Agent for adaptation
    validate: bool = True,     # Validate after adaptation
) -> Software
```

### DeployConfig

```python
@dataclass
class DeployConfig:
    code_path: str        # Path to solution code
    goal: str             # Description of what the code does
    env_vars: Dict = None # Environment variables
    port: int = 8000      # Port for HTTP services
    timeout: int = 300    # Timeout in seconds
```

### Software Interface

All deployments implement the `Software` interface:

```python
class Software(ABC):
    def run(self, inputs: Dict) -> Dict:
        """Run prediction. Returns {"status": "success/error", "output": ...}"""
    
    def stop(self) -> None:
        """Stop the software."""
    
    def logs(self) -> str:
        """Get execution logs."""
    
    def is_healthy(self) -> bool:
        """Check if software is running and healthy."""
    
    def get_deploy_command(self) -> str:
        """Get command to deploy the software."""
    
    def get_endpoint(self) -> Optional[str]:
        """Get HTTP endpoint if applicable."""
    
    def get_deployment_info(self) -> Dict:
        """Get full deployment metadata."""
```

### Response Format

All `run()` calls return a standardized response:

```python
{
    "status": "success",  # or "error"
    "output": {...},      # Function output
    "error": "...",       # Error message (if status == "error")
}
```

## Troubleshooting

### Common Issues

#### "Module not found" errors

Your solution's dependencies aren't installed. Install them:

```bash
pip install -r requirements.txt
```

#### Docker build fails

Check the Dockerfile and ensure all dependencies are specified:

```bash
cd your_solution
docker build -t test .
```

#### Modal function not deployed

Deploy the function first:

```bash
cd your_solution
modal deploy modal_app.py
```

#### BentoML service not running

Start the service:

```bash
cd your_solution
bentoml serve service:YourService
```

### Debug Mode

Get detailed logs:

```python
software = DeploymentFactory.create(strategy, config)
print(software.logs())
print(software.get_deployment_info())
```

## Examples

See `tests/deployment_examples/` for complete working examples:

- `sentiment_classifier/` - LOCAL deployment (TextBlob sentiment analysis)
- `image_api/` - DOCKER deployment (Pillow image processing)
- `text_embeddings/` - MODAL deployment (sentence-transformers with GPU)
- `qa_service/` - BENTOML deployment (transformers Q&A)

Each example includes:
- Complete source code
- `requirements.txt`
- README with usage instructions
- Deployment-specific configuration files

