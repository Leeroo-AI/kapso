# Claude Code Coding Agent Adapter
#
# Uses Anthropic's Claude Code CLI for code generation.
# Claude Code is a professional-grade agentic CLI tool.
#
# Key features:
# - Planning modes with step-by-step approach
# - CLAUDE.md for project constitution
# - Superior for complex, multi-step tasks
#
# Requires: 
# - ANTHROPIC_API_KEY in environment
# - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code

import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.execution.coding_agents.base import (
    CodingAgentInterface, 
    CodingAgentConfig, 
    CodingResult
)


class ClaudeCodeCodingAgent(CodingAgentInterface):
    """
    Claude Code-based coding agent.
    
    Uses Anthropic's Claude Code CLI for code generation.
    Excellent for complex feature development and refactoring.
    
    Features:
    - Planning mode (outlines steps before executing)
    - CLAUDE.md project constitution support
    - Permission system for tools (Edit, Read, Write)
    
    Configuration (agent_specific):
    - claude_md_path: Path to CLAUDE.md file (optional)
    - planning_mode: True (default) - use planning
    - timeout: 300 (default) - CLI timeout in seconds
    - allowed_tools: ["Edit", "Read", "Write", "Bash"] (default)
    
    Environment:
    - ANTHROPIC_API_KEY: Required for authentication
    """
    
    def __init__(self, config: CodingAgentConfig):
        """Initialize Claude Code coding agent."""
        super().__init__(config)
        self.workspace: Optional[str] = None
        
        # Get Claude Code-specific settings
        self._claude_md_path = config.agent_specific.get("claude_md_path", None)
        self._planning_mode = config.agent_specific.get("planning_mode", True)
        self._timeout = config.agent_specific.get("timeout", 300)
        self._allowed_tools = config.agent_specific.get(
            "allowed_tools", 
            ["Edit", "Read", "Write", "Bash"]
        )
        
        # Verify Claude Code CLI is installed
        self._verify_cli()
    
    def _verify_cli(self):
        """Verify Claude Code CLI is installed."""
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude Code CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        
        # Verify API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
    
    def initialize(self, workspace: str) -> None:
        """
        Initialize Claude Code agent for the workspace.
        
        Args:
            workspace: Path to the working directory
        """
        self.workspace = workspace
        
        # Create CLAUDE.md if specified path exists
        if self._claude_md_path and os.path.exists(self._claude_md_path):
            target = Path(workspace) / "CLAUDE.md"
            if not target.exists():
                shutil.copy(self._claude_md_path, target)
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """
        Generate code using Claude Code CLI.
        
        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug model
            
        Returns:
            CodingResult with Claude Code's response
        """
        if self.workspace is None:
            return CodingResult(
                success=False,
                output="",
                error="Agent not initialized. Call initialize() first."
            )
        
        model = self.config.debug_model if debug_mode else self.config.model
        
        try:
            # Build the CLI command
            cmd = self._build_command(prompt, model)
            
            # Run Claude Code CLI
            result = subprocess.run(
                cmd,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=self._get_env()
            )
            
            output = result.stdout
            stderr = result.stderr
            
            if result.returncode != 0:
                # Check if it's a non-fatal warning
                if "warning" in stderr.lower() and output:
                    pass  # Continue with output
                else:
                    return CodingResult(
                        success=False,
                        output=output,
                        error=stderr or f"CLI exited with code {result.returncode}"
                    )
            
            # Parse the response
            files_changed = self._get_changed_files()
            
            # Estimate cost (Claude Code doesn't report directly)
            cost = self._estimate_cost(len(prompt), len(output))
            self._cumulative_cost += cost
            
            return CodingResult(
                success=True,
                output=output,
                files_changed=files_changed,
                cost=cost,
                metadata={
                    "model": model,
                    "planning_mode": self._planning_mode,
                }
            )
        except subprocess.TimeoutExpired:
            return CodingResult(
                success=False,
                output="",
                error=f"Claude Code CLI timed out after {self._timeout} seconds"
            )
        except Exception as e:
            return CodingResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def _build_command(self, prompt: str, model: str) -> List[str]:
        """Build the Claude Code CLI command."""
        cmd = [
            "claude",
            "-p", prompt,  # Non-interactive mode with prompt
            "--output-format", "text",  # Text output
        ]
        
        # Add model if specified (claude uses its own model selection)
        # Note: Claude Code CLI may use --model flag differently
        
        # Add allowed tools
        if self._allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._allowed_tools)])
        
        return cmd
    
    def _get_env(self) -> Dict[str, str]:
        """Get environment variables for subprocess."""
        env = os.environ.copy()
        # Ensure API key is available
        if "ANTHROPIC_API_KEY" not in env:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return env
    
    def _get_changed_files(self) -> List[str]:
        """
        Get list of files changed in the workspace.
        
        Uses git status to detect changes.
        """
        files = []
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.workspace,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        # Format: "XY filename" or "XY old -> new"
                        parts = line.split()
                        if len(parts) >= 2:
                            filename = parts[-1]
                            filepath = Path(self.workspace) / filename
                            files.append(str(filepath))
        except:
            pass
        return files
    
    def _estimate_cost(self, input_len: int, output_len: int) -> float:
        """
        Estimate cost for Claude Code usage.
        
        Claude Sonnet pricing: ~$3 per 1M input, ~$15 per 1M output tokens
        Rough estimate: 4 chars per token
        """
        input_tokens = input_len / 4
        output_tokens = output_len / 4
        
        cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        return cost
    
    def cleanup(self) -> None:
        """Clean up Claude Code resources."""
        self.workspace = None
    
    def supports_native_git(self) -> bool:
        """Claude Code doesn't handle git commits natively."""
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return Claude Code's capabilities."""
        return {
            "native_git": False,
            "sandbox": False,
            "planning_mode": True,  # Claude Code excels at planning
            "cost_tracking": True,
            "streaming": False,
        }

