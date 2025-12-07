#!/usr/bin/env python3
"""
LangGraph Platform deployment script.

Deploys the chatbot agent to LangGraph Platform using LangSmith credentials.
"""
import os
import subprocess
import sys


def deploy_to_langgraph():
    """Deploy agent to LangGraph Platform."""
    api_key = os.environ.get("LANGSMITH_API_KEY")

    if not api_key:
        print("ERROR: LANGSMITH_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export LANGSMITH_API_KEY='your-api-key'")
        sys.exit(1)

    print("Deploying chatbot to LangGraph Platform...")
    print(f"Using API key: {api_key[:10]}...{api_key[-4:]}")

    # Deploy using langgraph CLI
    deploy_cmd = ["langgraph", "deploy"]
    result = subprocess.run(deploy_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("\n✓ Deployment successful!")
        print(result.stdout)
    else:
        print(f"\n✗ Deployment failed!")
        print(f"Error: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    deploy_to_langgraph()
