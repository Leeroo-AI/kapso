#!/usr/bin/env python3
"""BentoCloud deployment script."""

import os
import subprocess
import sys
import json


def deploy_to_bentocloud():
    """Deploy the text classifier to BentoCloud."""
    api_key = os.environ.get("BENTO_CLOUD_API_KEY")
    api_endpoint = os.environ.get("BENTO_CLOUD_API_ENDPOINT", "https://cloud.bentoml.com")

    if not api_key:
        print("ERROR: BENTO_CLOUD_API_KEY environment variable not set")
        print("Please set your BentoCloud API key:")
        print("  export BENTO_CLOUD_API_KEY='your-api-key'")
        sys.exit(1)

    print(f"Deploying to BentoCloud: {api_endpoint}")
    print("-" * 60)

    # Login to BentoCloud
    print("Logging in to BentoCloud...")
    try:
        subprocess.run(
            ["bentoml", "cloud", "login", "--api-token", api_key, "--endpoint", api_endpoint],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Login successful")
    except subprocess.CalledProcessError as e:
        print(f"✗ Login failed: {e.stderr}")
        sys.exit(1)

    # Deploy to BentoCloud
    print("\nDeploying service...")
    try:
        result = subprocess.run(
            ["bentoml", "deploy", ".", "--wait"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if result.returncode == 0:
            print("✓ Deployment successful!")
            print("\n" + "=" * 60)
            print("DEPLOYMENT OUTPUT:")
            print("=" * 60)
            print(result.stdout)

            # Try to extract deployment info
            for line in result.stdout.split('\n'):
                if 'endpoint' in line.lower() or 'url' in line.lower():
                    print(f"\n*** {line} ***")

        else:
            print(f"✗ Deployment failed!")
            print("\nError output:")
            print(result.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"✗ Deployment error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    deploy_to_bentocloud()
