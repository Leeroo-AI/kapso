#!/usr/bin/env python3
"""
BentoCloud deployment script.

Deploys the service to BentoCloud using API credentials.
Requires: BENTO_CLOUD_API_KEY and BENTO_CLOUD_API_ENDPOINT env vars.
"""

import os
import subprocess
import sys


def deploy_to_bentocloud():
    """Deploy service to BentoCloud."""
    api_key = os.environ.get("BENTO_CLOUD_API_KEY")
    api_endpoint = os.environ.get("BENTO_CLOUD_API_ENDPOINT", "https://cloud.bentoml.com")

    if not api_key:
        print("ERROR: BENTO_CLOUD_API_KEY not set")
        print("Please set the environment variable:")
        print("  export BENTO_CLOUD_API_KEY='your-api-key'")
        sys.exit(1)

    print(f"Deploying to BentoCloud: {api_endpoint}")

    # Login to BentoCloud
    print("Logging in to BentoCloud...")
    login_cmd = ["bentoml", "cloud", "login", "--api-token", api_key, "--endpoint", api_endpoint]
    result = subprocess.run(login_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Login failed: {result.stderr}")
        sys.exit(1)

    print("✓ Login successful")

    # Deploy the service
    print("Deploying service...")
    deploy_cmd = ["bentoml", "deploy", "."]
    result = subprocess.run(deploy_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Deployment successful!")
        print(result.stdout)

        # Extract deployment URL
        for line in result.stdout.split("\n"):
            if "endpoint" in line.lower() or "url" in line.lower():
                print(f"Endpoint: {line}")
    else:
        print(f"✗ Deployment failed: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    deploy_to_bentocloud()
