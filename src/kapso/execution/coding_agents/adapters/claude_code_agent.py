# Claude Code Coding Agent Adapter
#
# Uses Anthropic's Claude Code CLI for code generation.
# Claude Code is a professional-grade agentic CLI tool.
#
# Key features:
# - Planning modes with step-by-step approach
# - CLAUDE.md for project constitution
# - Superior for complex, multi-step tasks
# - Streaming mode for live output visibility
# - Supports both direct Anthropic API and AWS Bedrock
#
# Requires:
# - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
#
# Authentication (one of):
# - Direct Anthropic: ANTHROPIC_API_KEY in environment
# - AWS Bedrock: AWS_BEARER_TOKEN_BEDROCK or AWS credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
#   Plus: AWS_REGION must be set for Bedrock mode

import json
import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
}

from kapso.execution.coding_agents.base import (
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
    - Supports both direct Anthropic API and AWS Bedrock
    
    Configuration (agent_specific):
    - claude_md_path: Path to CLAUDE.md file (optional)
    - planning_mode: True (default) - use planning
    - timeout: 3600 (default) - CLI timeout in seconds (1 hour)
    - allowed_tools: ["Edit", "Read", "Write", "Bash"] (default)
    - streaming: True (default) - stream output live to terminal for visibility
    - use_bedrock: False (default) - use AWS Bedrock instead of direct Anthropic API
    - aws_region: AWS region for Bedrock (required if use_bedrock=True, default: "us-east-1")
    - mcp_servers: Dict of MCP server configurations (optional)
      Example:
        {
            "kg-graph-search": {
                "command": "python",
                "args": ["-m", "kapso.gated_mcp.server"],
                "cwd": "/path/to/project",
                "env": {"MCP_ENABLED_GATES": "kg", "KG_INDEX_PATH": "/path/to/.index"}
            }
        }
    
    Environment (Direct Anthropic mode - default):
    - ANTHROPIC_API_KEY: Required for authentication
    
    Environment (AWS Bedrock mode - use_bedrock=True):
    - AWS_REGION: AWS region (can also be set via aws_region config)
    - One of:
      - AWS_BEARER_TOKEN_BEDROCK: Bedrock API key (simplest)
      - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY: IAM access keys
      - AWS_PROFILE: SSO profile name (after running aws sso login)
    """
    
    def __init__(self, config: CodingAgentConfig):
        """Initialize Claude Code coding agent."""
        super().__init__(config)
        self.workspace: Optional[str] = None
        
        # Get Claude Code-specific settings
        self._claude_md_path = config.agent_specific.get("claude_md_path", None)
        self._planning_mode = config.agent_specific.get("planning_mode", True)
        self._timeout = config.agent_specific.get("timeout", 3600)
        self._allowed_tools = config.agent_specific.get(
            "allowed_tools", 
            ["Edit", "Read", "Write", "Bash"]
        )
        # Optional environment overrides for the Claude Code subprocess.
        #
        # Why:
        # - Claude Code spawns MCP servers as subprocesses.
        # - The simplest way to pass per-run configuration (like KG_INDEX_PATH)
        #   into those MCP server processes is via inherited environment vars.
        # - We keep this explicit to avoid relying on global os.environ mutation.
        self._env_overrides: Dict[str, str] = {
            str(k): str(v)
            for k, v in (config.agent_specific.get("env_overrides") or {}).items()
            if v is not None
        }
        # Streaming: print Claude Code output live to terminal (default True for visibility)
        self._streaming = config.agent_specific.get("streaming", True)
        # Show heartbeat messages during long operations (default False to reduce noise)
        self._show_heartbeat = config.agent_specific.get("show_heartbeat", False)
        
        # AWS Bedrock settings
        # use_bedrock: If True, route requests through AWS Bedrock instead of direct Anthropic API
        self._use_bedrock = config.agent_specific.get("use_bedrock", False)
        # aws_region: AWS region for Bedrock (required if use_bedrock=True)
        self._aws_region = config.agent_specific.get("aws_region", "us-east-1")
        
        # MCP server configuration
        # mcp_servers: Dict of MCP server configs to enable for this agent
        # Format: {"server-name": {"command": "...", "args": [...], "cwd": "...", "env": {...}}}
        self._mcp_servers: Optional[Dict[str, Any]] = config.agent_specific.get("mcp_servers")
        self._mcp_config_path: Optional[Path] = None  # Set during initialize()
        
        # Verify Claude Code CLI is installed and credentials are available
        self._verify_cli()
    
    def _verify_cli(self):
        """
        Verify Claude Code CLI is installed and credentials are available.
        
        Checks for appropriate credentials based on mode:
        - Direct mode: ANTHROPIC_API_KEY
        - Bedrock mode: AWS credentials (bearer token, access keys, or profile)
        """
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude Code CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        
        if self._use_bedrock:
            # Bedrock mode: Check for AWS credentials
            self._verify_bedrock_credentials()
        else:
            # Direct mode: Check for Anthropic API key
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
    
    def _verify_bedrock_credentials(self):
        """
        Verify AWS Bedrock credentials are available.
        
        Checks for one of:
        - AWS_BEARER_TOKEN_BEDROCK (Bedrock API key - simplest)
        - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (IAM access keys)
        - AWS_PROFILE (SSO profile)
        
        Also verifies AWS_REGION is set (required for Bedrock).
        """
        # Check for AWS region
        aws_region = os.environ.get("AWS_REGION") or self._aws_region
        if not aws_region:
            raise ValueError(
                "AWS_REGION not set. Required for Bedrock mode. "
                "Set AWS_REGION environment variable or aws_region in config."
            )
        
        # Check for at least one authentication method
        has_bearer_token = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
        has_access_keys = bool(
            os.environ.get("AWS_ACCESS_KEY_ID") and 
            os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        has_profile = bool(os.environ.get("AWS_PROFILE"))
        
        if not (has_bearer_token or has_access_keys or has_profile):
            raise ValueError(
                "No AWS credentials found for Bedrock mode. Set one of:\n"
                "  - AWS_BEARER_TOKEN_BEDROCK (Bedrock API key)\n"
                "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (IAM access keys)\n"
                "  - AWS_PROFILE (SSO profile, after running 'aws sso login')"
            )
    
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
        
        # Write MCP config file if MCP servers are configured
        if self._mcp_servers:
            self._mcp_config_path = self._write_mcp_config()
            logger.info(f"MCP config written to: {self._mcp_config_path}")
    
    def _write_mcp_config(self) -> Path:
        """
        Write MCP server configuration to a temporary JSON file.
        
        The file is used by Claude Code CLI via --mcp-config flag.
        
        Returns:
            Path to the temporary config file
        """
        mcp_config = {"mcpServers": self._mcp_servers}
        
        # Create temp file that persists until cleanup()
        # Use workspace-based path for easier debugging
        # IMPORTANT: Use absolute path to avoid path duplication when Claude Code
        # runs with cwd=workspace and looks for the config relative to that directory
        config_dir = Path(self.workspace).resolve() / ".claude_mcp"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "mcp_config.json"
        
        config_path.write_text(json.dumps(mcp_config, indent=2))
        logger.debug(f"MCP config: {json.dumps(mcp_config, indent=2)}")
        
        return config_path
    
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
            # Use streaming or buffered mode
            if self._streaming:
                # Build command inside _run_streaming with stream-json
                cmd = self._build_command(prompt, model, use_stream_json=False)  # placeholder
                return self._run_streaming(cmd, model)
            else:
                cmd = self._build_command(prompt, model, use_stream_json=False)
                return self._run_buffered(cmd, model)
            
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
    
    def _run_buffered(self, cmd: List[str], model: str) -> CodingResult:
        """Run Claude Code CLI in buffered mode (no live output)."""
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
        cost = self._estimate_cost(len(cmd[2]) if len(cmd) > 2 else 0, len(output))
        self._cumulative_cost += cost
        
        return CodingResult(
            success=True,
            output=output,
            files_changed=files_changed,
            cost=cost,
            metadata={
                "model": model,
                "planning_mode": self._planning_mode,
                "use_bedrock": self._use_bedrock,
            }
        )
    
    def _run_streaming(self, cmd: List[str], model: str) -> CodingResult:
        """
        Run Claude Code CLI with live streaming output using stream-json format.
        
        Parses JSON events and displays Claude's thinking, tool calls, and results
        in real-time for maximum visibility.
        """
        # Rebuild command with stream-json format for structured output
        prompt = cmd[2] if len(cmd) > 2 else ""
        stream_cmd = self._build_command(prompt, model, use_stream_json=True)
        
        start_time = time.time()
        raw_lines: List[str] = []
        assistant_texts: List[str] = []
        result_text: str = ""
        total_cost: float = 0.0
        is_error: bool = False
        error_msg: str = ""
        
        c = _COLORS  # shorthand
        
        print(f"\n{c['cyan']}━━━ Claude Code Starting ━━━{c['reset']}", flush=True)

        # Ensure stdout/stderr pipes are always closed.
        #
        # Why:
        # - In Python, `subprocess.Popen(..., stdout=PIPE)` creates file objects.
        # - If we don't close them deterministically, Python can emit noisy
        #   `ResourceWarning: unclosed file <_io.TextIOWrapper ...>` at shutdown.
        # - This keeps logs clean and prevents leaking file descriptors in long runs.
        process = subprocess.Popen(
            stream_cmd,
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._get_env(),
            bufsize=1,
        )
        
        try:
            # Use select for non-blocking I/O on both stdout and stderr
            stdout_fd = process.stdout.fileno() if process.stdout else -1
            stderr_fd = process.stderr.fileno() if process.stderr else -1
            last_heartbeat = time.time()
            heartbeat_interval = 10.0  # Show heartbeat every 10 seconds of silence
            
            while True:
                retcode = process.poll()
                
                # Use select to check which streams have data (with 0.5s timeout)
                readable = []
                if stdout_fd >= 0 or stderr_fd >= 0:
                    fds_to_check = []
                    if stdout_fd >= 0:
                        fds_to_check.append(process.stdout)
                    if stderr_fd >= 0:
                        fds_to_check.append(process.stderr)
                    try:
                        readable, _, _ = select.select(fds_to_check, [], [], 0.5)
                    except (ValueError, OSError):
                        # File descriptor closed
                        pass
                
                got_output = False
                
                # Read from stdout if data available
                if process.stdout in readable:
                    line = process.stdout.readline()
                    if line:
                        line = line.rstrip('\n')
                        raw_lines.append(line)
                        self._display_stream_event(line, assistant_texts)
                        got_output = True
                        last_heartbeat = time.time()
                
                # Read from stderr if data available
                if process.stderr in readable:
                    err_line = process.stderr.readline()
                    if err_line:
                        err_line = err_line.rstrip('\n')
                        print(f"{c['yellow']}  [stderr] {err_line}{c['reset']}", file=sys.stderr, flush=True)
                        got_output = True
                        last_heartbeat = time.time()
                
                # Show heartbeat if no output for a while (Claude might be thinking)
                if not got_output and retcode is None and self._show_heartbeat:
                    now = time.time()
                    if now - last_heartbeat > heartbeat_interval:
                        elapsed = now - start_time
                        print(f"{c['dim']}  ... still working ({elapsed:.0f}s){c['reset']}", flush=True)
                        last_heartbeat = now
                
                if retcode is not None:
                    # Drain remaining output
                    if process.stdout:
                        for line in process.stdout:
                            line = line.rstrip('\n')
                            raw_lines.append(line)
                            self._display_stream_event(line, assistant_texts)
                    if process.stderr:
                        for err_line in process.stderr:
                            print(f"{c['yellow']}  [stderr] {err_line.rstrip()}{c['reset']}", file=sys.stderr, flush=True)
                    break
            
            elapsed = time.time() - start_time
            
            # Parse final result from last JSON line
            for line in reversed(raw_lines):
                try:
                    event = json.loads(line)
                    if event.get("type") == "result":
                        result_text = event.get("result", "")
                        total_cost = event.get("total_cost_usd", 0.0)
                        is_error = event.get("is_error", False)
                        break
                except json.JSONDecodeError:
                    continue
            
            print(f"{c['cyan']}━━━ Claude Code Finished ({elapsed:.1f}s, ${total_cost:.4f}) ━━━{c['reset']}\n", flush=True)
            
            if retcode != 0 or is_error:
                error_msg = result_text if is_error else f"CLI exited with code {retcode}"
                return CodingResult(
                    success=False,
                    output="\n".join(assistant_texts),
                    error=error_msg,
                    metadata={"elapsed_seconds": elapsed}
                )
            
            files_changed = self._get_changed_files()
            self._cumulative_cost += total_cost
            
            return CodingResult(
                success=True,
                output=result_text or "\n".join(assistant_texts),
                files_changed=files_changed,
                cost=total_cost,
                metadata={
                    "model": model,
                    "planning_mode": self._planning_mode,
                    "elapsed_seconds": elapsed,
                    "streaming": True,
                    "use_bedrock": self._use_bedrock,
                }
            )
            
        except Exception:
            # Make best-effort to stop the child process on error.
            try:
                process.kill()
            except Exception:
                pass
            raise
        finally:
            # Always close pipes so Python doesn't warn about unclosed file objects.
            # This also helps prevent leaking file descriptors in long Kapso runs.
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if process.stderr:
                    process.stderr.close()
            except Exception:
                pass
            
            # Reap the child process (best-effort). If it's already exited this returns fast.
            try:
                process.wait(timeout=1)
            except Exception:
                pass
    
    def _display_stream_event(self, line: str, assistant_texts: List[str]) -> None:
        """Parse and display a single stream-json event."""
        c = _COLORS
        
        if not line.strip():
            return  # Skip empty lines
        
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Not JSON, just print raw (might be progress indicator or other text)
            print(f"  {line}", flush=True)
            return
        
        event_type = event.get("type", "")
        subtype = event.get("subtype", "")
        
        if event_type == "system" and subtype == "init":
            # Initialization event
            model = event.get("model", "unknown")
            tools = event.get("tools", [])
            print(f"{c['dim']}  [init] model={model}, tools={len(tools)}{c['reset']}", flush=True)
        
        elif event_type == "assistant":
            # Assistant message (thinking + tool calls)
            message = event.get("message", {})
            content = message.get("content", [])
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        assistant_texts.append(text)
                        # Show full thinking text (no truncation)
                        print(f"{c['green']}  [thinking] {text}{c['reset']}", flush=True)
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    # Show tool call summary with arguments
                    if tool_name in ("Read", "Edit", "Write"):
                        path = tool_input.get("file_path", tool_input.get("path", "?"))
                        print(f"{c['blue']}  [tool:{tool_name}] {path}{c['reset']}", flush=True)
                    elif tool_name == "Bash":
                        cmd = tool_input.get("command", "")[:80]
                        print(f"{c['magenta']}  [tool:Bash] {cmd}{c['reset']}", flush=True)
                    elif tool_name.startswith("mcp__"):
                        # MCP tool - show the arguments
                        args_str = json.dumps(tool_input, ensure_ascii=False)
                        # Truncate if too long
                        if len(args_str) > 200:
                            args_str = args_str[:200] + "..."
                        print(f"{c['blue']}  [tool:{tool_name}] {args_str}{c['reset']}", flush=True)
                    else:
                        print(f"{c['blue']}  [tool:{tool_name}]{c['reset']}", flush=True)
        
        elif event_type == "user":
            # Tool result returned to Claude
            content = event.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")[:8]
                    is_error = block.get("is_error", False)
                    status = "error" if is_error else "ok"
                    # Show truncated result content
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        result_preview = result_content[:150].replace('\n', ' ')
                        if len(result_content) > 150:
                            result_preview += "..."
                    else:
                        result_preview = "..."
                    print(f"{c['dim']}  [result:{status}] {result_preview}{c['reset']}", flush=True)
        
        elif event_type == "result":
            # Final result - show summary
            duration = event.get("duration_ms", 0) / 1000
            cost = event.get("total_cost_usd", 0)
            print(f"{c['dim']}  [result] duration={duration:.1f}s, cost=${cost:.4f}{c['reset']}", flush=True)
        
        else:
            # Unknown event type - show it for debugging
            if event_type:
                print(f"{c['dim']}  [{event_type}:{subtype}]{c['reset']}", flush=True)
    
    def _build_command(self, prompt: str, model: str, use_stream_json: bool = False) -> List[str]:
        """Build the Claude Code CLI command."""
        cmd = [
            "claude",
            "-p", prompt,  # Non-interactive mode with prompt
            "--dangerously-skip-permissions",  # Auto-approve all tool calls
        ]
        
        # Output format: stream-json for live visibility, text for buffered
        if use_stream_json:
            cmd.extend(["--output-format", "stream-json", "--verbose"])
        else:
            cmd.extend(["--output-format", "text"])
        
        # Add model if specified
        if model:
            cmd.extend(["--model", model])
        
        # Add allowed tools
        if self._allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._allowed_tools)])
        
        # Add MCP config if available
        if self._mcp_config_path and self._mcp_config_path.exists():
            cmd.extend(["--mcp-config", str(self._mcp_config_path)])
        
        return cmd
    
    def _get_env(self) -> Dict[str, str]:
        """
        Get environment variables for subprocess.
        
        Sets up the appropriate environment based on mode:
        - Direct mode: Ensures ANTHROPIC_API_KEY is available
        - Bedrock mode: Sets CLAUDE_CODE_USE_BEDROCK=1 and AWS_REGION
        """
        env = os.environ.copy()
        
        if self._use_bedrock:
            # Bedrock mode: Set the flag and region
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"
            
            # Set AWS_REGION if not already in environment
            if "AWS_REGION" not in env:
                env["AWS_REGION"] = self._aws_region
            
            # Log which auth method is being used (for debugging)
            if env.get("AWS_BEARER_TOKEN_BEDROCK"):
                logger.debug("Using Bedrock with bearer token authentication")
            elif env.get("AWS_ACCESS_KEY_ID"):
                logger.debug("Using Bedrock with access key authentication")
            elif env.get("AWS_PROFILE"):
                logger.debug(f"Using Bedrock with SSO profile: {env.get('AWS_PROFILE')}")
        else:
            # Direct Anthropic mode: Ensure API key is available
            if "ANTHROPIC_API_KEY" not in env:
                raise ValueError("ANTHROPIC_API_KEY not set")

        # Apply caller-provided env overrides last so they take precedence.
        if self._env_overrides:
            env.update(self._env_overrides)
        
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
        # Clean up MCP config directory if it exists
        if self._mcp_config_path and self._mcp_config_path.exists():
            try:
                config_dir = self._mcp_config_path.parent
                if config_dir.name == ".claude_mcp":
                    shutil.rmtree(config_dir)
                    logger.debug(f"Cleaned up MCP config: {config_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up MCP config: {e}")
        
        self._mcp_config_path = None
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
            "streaming": self._streaming,  # Now supports live output streaming
            "bedrock": self._use_bedrock,  # Using AWS Bedrock for API calls
            "mcp": bool(self._mcp_servers),  # MCP server integration enabled
        }

