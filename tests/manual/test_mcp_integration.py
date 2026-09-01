#!/usr/bin/env python3
"""
Simple MCP Integration Test

Tests the MCP server connection with Claude Code by running a simple task.
This verifies that:
1. MCP server starts correctly
2. Claude Code can connect to it
3. Knowledge search tools work

Usage:
    python tests/test_mcp_integration.py
    
    # Or with explicit env loading:
    export $(grep ANTHROPIC_API_KEY .env | xargs) && python tests/test_mcp_integration.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment from .env if available
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
    print(f"Loading environment from {ENV_FILE}")
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value


def run_test():
    """Run a simple MCP integration test with Claude Code."""
    
    print("=" * 60)
    print("MCP Integration Test")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY=your_key")
        print("\nAlternatively, load from .env:")
        print("  source <(grep ANTHROPIC_API_KEY .env | sed 's/^/export /')")
        return False
    
    # Check Claude Code CLI is available
    if not subprocess.run(["which", "claude"], capture_output=True).returncode == 0:
        print("\nERROR: Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return False
    
    # Create temporary MCP config
    mcp_config = {
        "mcpServers": {
            "kg-graph-search": {
                "command": "python",
                "args": ["-m", "src.gated_mcp.server"],
                "cwd": str(PROJECT_ROOT),
                "env": {
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "MCP_ENABLED_GATES": "kg"
                }
            }
        }
    }
    
    # Write temporary config
    config_path = Path(tempfile.gettempdir()) / "test_mcp_config.json"
    config_path.write_text(json.dumps(mcp_config, indent=2))
    print(f"\nMCP config written to: {config_path}")
    
    # Test prompt
    test_prompt = """Use the knowledge search tools to help with this task:

Build a llama 3 post-training with GRPO on a user preference dataset.

Search the knowledge base for relevant workflows and best practices.
Return a summary of what you found."""
    
    print(f"\nTest prompt:\n{test_prompt}")
    print("\n" + "-" * 60)
    print("Running Claude Code with MCP...")
    print("-" * 60 + "\n")
    
    # Run Claude Code with MCP config
    cmd = [
        "claude",
        "-p", test_prompt,
        "--mcp-config", str(config_path),
        "--output-format", "text",
        "--allowedTools", "Read,mcp__kg-graph-search__search_knowledge,mcp__kg-graph-search__get_wiki_page",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            env={**os.environ, "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")}
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("TEST PASSED: Claude Code completed successfully")
            return True
        else:
            print(f"TEST FAILED: Exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("TEST FAILED: Timeout after 120 seconds")
        return False
    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

