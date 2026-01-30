# BentoML Cloud Deployment Instructions

BentoML deployment creates production-ready ML services deployed to **BentoCloud**.
Best for production ML APIs with batching, monitoring, auto-scaling, and serverless infrastructure.

## DEPLOY COMMAND

```bash
python deploy.py
```

Run this command to deploy to BentoCloud. Capture the endpoint URL from the output.
If deployment fails, debug and fix the error.

## RUN INTERFACE
- type: bentocloud
- deployment_name: derived from path
- path: /predict

After successful deployment, output this JSON (update deployment_name and endpoint):
```
<run_interface>{"type": "bentocloud", "deployment_name": "your-deployment-name", "path": "/predict"}</run_interface>
<endpoint_url>https://your-deployment.bentoml.cloud</endpoint_url>
```

## CRITICAL: YOU MUST ACTUALLY DEPLOY

**Do NOT just create files. You MUST run `python deploy.py` and verify it succeeds.**

After creating the BentoML service files:
1. Run: `python deploy.py`
2. Wait for deployment to complete
3. Capture the endpoint URL from the output
4. Test the endpoint with curl to verify it works
5. Report the deployment name and endpoint URL

**If deployment fails, debug the error and fix it. Do not give up.**

## Environment Variables Required

- `BENTO_CLOUD_API_KEY` - Your BentoCloud API key
- `BENTO_CLOUD_API_ENDPOINT` - BentoCloud API endpoint (optional)

## Required Structure

```
solution/
├── main.py           # Core logic with predict()
├── service.py        # BentoML service definition
├── bentofile.yaml    # BentoML build configuration
├── deploy.py         # BentoCloud deployment script
├── requirements.txt  # Dependencies
└── ...
```

## BentoML Service (service.py)

```python
"""BentoML service for BentoCloud deployment."""

import bentoml
from typing import Any, Dict, List, Union


@bentoml.service(
    name="solution-service",
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 300},
)
class SolutionService:
    def __init__(self):
        from main import predict as _predict
        self._predict = _predict
    
    @bentoml.api
    def predict(self, inputs: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        try:
            result = self._predict(inputs)
            return {"status": "success", "output": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @bentoml.api
    def health(self) -> Dict[str, str]:
        return {"status": "healthy"}
```

## BentoML Configuration (bentofile.yaml)

```yaml
service: "service:SolutionService"

labels:
  owner: expert-agent
  project: solution

include:
  - "*.py"
  - "requirements.txt"

python:
  requirements_txt: "./requirements.txt"

docker:
  distro: debian
  python_version: "3.11"
```

## Deployment Script (deploy.py)

```python
#!/usr/bin/env python3
"""BentoCloud deployment script."""

import os
import subprocess
import sys


def deploy_to_bentocloud():
    api_key = os.environ.get("BENTO_CLOUD_API_KEY")
    api_endpoint = os.environ.get("BENTO_CLOUD_API_ENDPOINT", "https://cloud.bentoml.com")
    
    if not api_key:
        print("ERROR: BENTO_CLOUD_API_KEY not set")
        sys.exit(1)
    
    print(f"Deploying to BentoCloud: {api_endpoint}")
    
    # Login
    subprocess.run(["bentoml", "cloud", "login", "--api-token", api_key, "--endpoint", api_endpoint], check=True)
    
    # Deploy
    result = subprocess.run(["bentoml", "deploy", "."], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Deployment successful!")
        print(result.stdout)
    else:
        print(f"✗ Deployment failed: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    deploy_to_bentocloud()
```

## Testing Locally

```bash
# Install BentoML
pip install bentoml

# Serve locally for testing
bentoml serve service:SolutionService

# Test
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

## Notes

- BentoML handles request batching automatically
- Use `bentoml.models` for model versioning
- Monitor with built-in metrics on BentoCloud dashboard

