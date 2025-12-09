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
