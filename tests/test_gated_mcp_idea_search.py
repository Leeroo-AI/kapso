"""
Test gated MCP idea search with Claude Code + Bedrock.

This standalone script tests the idea search gate by:
1. Starting the gated MCP server with idea and research gates
2. Connecting Claude Code with Bedrock
3. Running a simple idea search task

Run: python tests/test_gated_mcp_idea_search.py

Environment:
    KG_INDEX_PATH              - Path to .index file (optional)
    AWS_REGION                 - AWS region (default: us-east-1)
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()


def _has_bedrock_creds() -> bool:
    """Check if AWS Bedrock credentials are available."""
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def run_idea_search_test() -> bool:
    """
    Run the idea search integration test.
    
    Returns:
        True if test passed, False otherwise
    """
    from src.execution.coding_agents.base import CodingAgentConfig
    from src.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
    from src.knowledge.gated_mcp import get_allowed_tools_for_gates
    
    print("=" * 60)
    print("Gated MCP Idea Search Test")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create temp workspace
    workspace = tempfile.mkdtemp(prefix="gated_mcp_idea_test_")
    print(f"Workspace: {workspace}")
    
    # Define which gates to enable
    gates = ["idea", "research"]
    
    # MCP server configuration
    mcp_env = {
        "PYTHONPATH": str(project_root),
        "MCP_ENABLED_GATES": ",".join(gates),
    }
    
    # Add KG_INDEX_PATH if available
    kg_index_path = os.environ.get("KG_INDEX_PATH")
    if kg_index_path:
        mcp_env["KG_INDEX_PATH"] = kg_index_path
        print(f"KG Index: {kg_index_path}")
    
    mcp_servers = {
        "gated-knowledge": {
            "command": "python",
            "args": ["-m", "src.knowledge.gated_mcp.server"],
            "cwd": str(project_root),
            "env": mcp_env,
        }
    }
    
    # Get allowed tools for the gates
    allowed_tools = get_allowed_tools_for_gates(gates, "gated-knowledge")
    print(f"Gates: {gates}")
    print(f"Allowed tools: {allowed_tools}")
    
    # Configure agent with Bedrock
    config = CodingAgentConfig(
        agent_type="claude_code",
        model="us.anthropic.claude-opus-4-5-20251101-v1:0",
        debug_model="us.anthropic.claude-opus-4-5-20251101-v1:0",
        agent_specific={
            "use_bedrock": True,
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed_tools,
            "timeout": 180,
            "streaming": True,
        }
    )
    
    # Initialize agent
    agent = ClaudeCodeCodingAgent(config)
    agent.initialize(workspace)
    
    print(f"Model: {config.model}")
    print(f"Bedrock: {agent._use_bedrock}")
    print()
    
    # Run idea search task
    print("Task: Search for principles about LoRA fine-tuning")
    print("-" * 60)
    
    result = agent.generate_code(
        "Search for principles about LoRA fine-tuning. "
        "First try wiki_idea_search, and if no results are found, use research_idea to search the web. "
        "Report what you find in a brief summary."
    )
    
    print("-" * 60)
    print()
    
    # Check result
    print(f"Success: {result.success}")
    print(f"Cost: ${result.cost:.4f}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.output:
        print(f"\nOutput preview:\n{result.output[:500]}...")
    
    # Cleanup
    agent.cleanup()
    shutil.rmtree(workspace, ignore_errors=True)
    
    print()
    print("=" * 60)
    print(f"TEST {'PASSED' if result.success else 'FAILED'}")
    print("=" * 60)
    
    return bool(result.success)


def test_gated_mcp_idea_search():
    """Pytest entrypoint."""
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI not installed")
    if not _has_bedrock_creds():
        pytest.skip("No AWS Bedrock credentials found")
    
    assert run_idea_search_test()


if __name__ == "__main__":
    # Check prerequisites
    if shutil.which("claude") is None:
        print("ERROR: Claude Code CLI not installed")
        print("Install with: npm install -g @anthropic-ai/claude-code")
        exit(1)
    
    if not _has_bedrock_creds():
        print("WARNING: No AWS Bedrock credentials found")
        print("Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE")
    
    success = run_idea_search_test()
    exit(0 if success else 1)
