"""
Simple test for Claude Code Agent with AWS Bedrock.

Run: python tests/test_claude_code_agent.py
"""

import os
import shutil
import tempfile
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.execution.coding_agents.base import CodingAgentConfig
from src.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent


RUN_BEDROCK_TESTS = os.getenv("TINKERER_RUN_BEDROCK_TESTS") == "1"


def _has_bedrock_creds() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def run_bedrock_hello_world() -> bool:
    """Run Claude Code via Bedrock by writing a hello world file (returns success bool)."""
    
    print("=" * 60)
    print("Claude Code + AWS Bedrock Test")
    print("=" * 60)
    
    # Create temp workspace
    workspace = tempfile.mkdtemp(prefix="claude_test_")
    print(f"Workspace: {workspace}")
    
    # Configure agent with Bedrock
    config = CodingAgentConfig(
        agent_type="claude_code",
        model="us.anthropic.claude-opus-4-5-20251101-v1:0",
        debug_model="us.anthropic.claude-opus-4-5-20251101-v1:0",
        agent_specific={
            "use_bedrock": True,
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
            "streaming": True,
            "timeout": 120,
        }
    )
    
    # Initialize agent
    agent = ClaudeCodeCodingAgent(config)
    agent.initialize(workspace)
    
    print(f"Model: {config.model}")
    print(f"Bedrock: {agent._use_bedrock}")
    print()
    
    # Run simple task
    print("Task: Write a hello world Python file")
    print("-" * 60)
    
    result = agent.generate_code(
        "Create a file called hello.py that prints 'Hello from Bedrock!'"
    )
    
    print("-" * 60)
    print()
    
    # Check result
    print(f"Success: {result.success}")
    print(f"Cost: ${result.cost:.4f}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    # Check if file was created
    hello_file = os.path.join(workspace, "hello.py")
    file_exists = os.path.exists(hello_file)
    print(f"File created: {file_exists}")
    
    if file_exists:
        with open(hello_file) as f:
            content = f.read()
        print(f"Content:\n{content}")
    
    # Cleanup
    agent.cleanup()
    shutil.rmtree(workspace, ignore_errors=True)
    
    print()
    print("=" * 60)
    print(f"TEST {'PASSED' if result.success and file_exists else 'FAILED'}")
    print("=" * 60)
    
    return bool(result.success and file_exists)


def test_bedrock_hello_world():
    """Pytest entrypoint (skipped unless explicitly enabled)."""
    if not RUN_BEDROCK_TESTS:
        pytest.skip("Bedrock test disabled (set TINKERER_RUN_BEDROCK_TESTS=1 to enable).")
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI not installed (required for this test).")
    if not _has_bedrock_creds():
        pytest.skip("No AWS Bedrock credentials found in environment (required for this test).")

    assert run_bedrock_hello_world()


if __name__ == "__main__":
    success = run_bedrock_hello_world()
    exit(0 if success else 1)
