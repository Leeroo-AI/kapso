"""
Simple test for Claude Code Agent with AWS Bedrock.

Run: python tests/test_claude_code_agent.py
"""

import os
import shutil
import tempfile
from dotenv import load_dotenv

load_dotenv()

from src.execution.coding_agents.base import CodingAgentConfig
from src.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent


def test_bedrock_hello_world():
    """Test Claude Code via Bedrock by writing a hello world file."""
    
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
    
    return result.success and file_exists


if __name__ == "__main__":
    success = test_bedrock_hello_world()
    exit(0 if success else 1)
