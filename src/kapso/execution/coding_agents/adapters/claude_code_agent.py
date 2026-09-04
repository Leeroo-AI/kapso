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
# - Supports OAuth (subscription) and direct Anthropic API keys
#
# Requires:
# - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
#
# Authentication (one of):
# - OAuth (subscription): a stored Claude CLI login or CLAUDE_CODE_OAUTH_TOKEN
# - Direct Anthropic: ANTHROPIC_API_KEY in environment

import json
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Shutdown mechanics for the streaming deadline: after SIGTERM, how long the
# CLI gets to flush its final stream-json result event (which carries the real
# cost) before SIGKILL. Process-teardown grace, not an operator knob.
_DEADLINE_GRACE_SECONDS = 2.0

# After a session's terminal result carries all completion markers, how long
# a silent-but-alive CLI gets before being reaped (process-lifecycle policy,
# not an operator knob — same class as the deadline grace above).
_POST_COMPLETION_GRACE_SECONDS = 60.0

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

from kapso.execution.inbox import new_requests_for_session
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
    - Supports OAuth (subscription) and direct Anthropic API keys
    
    Configuration (agent_specific):
    - claude_md_path: Path to CLAUDE.md file (optional)
    - planning_mode: True (default) - use planning
    - timeout: 3600 (default) - CLI timeout in seconds (1 hour)
    - allowed_tools: ["Edit", "Read", "Write", "Bash"] (default)
    - streaming: True (default) - stream output live to terminal for visibility
    - auth_mode: Authentication mode: auto (default), oauth, or api_key
    - append_system_prompt: Optional string appended to Claude Code's default system prompt
      Useful for injecting workspace restrictions (e.g. filesystem sandboxing)
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
    
    Environment (API-key mode):
    - ANTHROPIC_API_KEY: Required for authentication

    Environment (OAuth mode):
    - A stored Claude CLI login, or CLAUDE_CODE_OAUTH_TOKEN
    """

    AUTH_MODES = frozenset({"auto", "oauth", "api_key"})
    _PROVIDER_FLAGS = (
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    )
    
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
        # Extra tools to BAN for this session, merged into --disallowedTools.
        # Under --dangerously-skip-permissions, --allowedTools does NOT restrict
        # (it only gates permission prompts); --disallowedTools is what actually
        # removes a tool. Used to hard-disable WebSearch/WebFetch on web-off
        # (leakage-safe) ideation.
        self._disallowed_tools = config.agent_specific.get("disallowed_tools", [])
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
        
        # Authentication settings (subscription-first: bedrock was removed
        # 2026-08-26 by user direction — OAuth or an Anthropic API key).
        self._requested_auth_mode = self._get_requested_auth_mode(config.agent_specific)
        self._auth_mode = self._requested_auth_mode
        
        # MCP server configuration
        # mcp_servers: Dict of MCP server configs to enable for this agent
        # Format: {"server-name": {"command": "...", "args": [...], "cwd": "...", "env": {...}}}
        self._mcp_servers: Optional[Dict[str, Any]] = config.agent_specific.get("mcp_servers")
        self._mcp_config_path: Optional[Path] = None  # Set during initialize()
        
        # Optional system prompt to append to Claude Code's default system prompt.
        # Useful for injecting workspace restrictions, project rules, etc.
        self._append_system_prompt: Optional[str] = config.agent_specific.get("append_system_prompt")

        # Optional reasoning-effort level forwarded to the CLI (--effort).
        self._effort: Optional[str] = config.agent_specific.get("effort")

        # Env var names removed from the CLI child's environment (never from
        # this process). Used when the orchestrator legitimately holds a
        # credential the agent must not inherit — e.g. PostTrainBench strips
        # OPENAI_API_KEY from agent sessions on non-judge benchmarks while
        # kapso's own utility LLM keeps using it.
        self._env_strip: List[str] = list(
            config.agent_specific.get("env_strip", [])
        )

        # Env defaults applied set-if-absent to the CLI child env (ambient
        # values — e.g. a benchmark wrapper like solve.sh — keep precedence).
        # Used for the Bash-tool clock policy: without BASH_DEFAULT_TIMEOUT_MS
        # the CLI auto-backgrounds any foreground command at 120s, making
        # blocking evaluations structurally impossible (relbench finding 14).
        self._env_defaults: Dict[str, str] = {
            str(k): str(v)
            for k, v in (config.agent_specific.get("env_defaults") or {}).items()
        }

        # Completion markers: when a terminal result event's text contains
        # ALL of these, the session has declared itself finished per its
        # output contract. A CLI that then lingers in silence gets reaped
        # after a short grace instead of idling until the deadline
        # (runs #9/#16: 1.5h and 24min of idle GPU, R9-I-1/R16-P2-1).
        self._completion_markers: List[str] = list(
            config.agent_specific.get("completion_markers", [])
        )

        # When set, every raw stream-json event line is appended to this file
        # during streaming runs — per-session process forensics (the same
        # pattern as the codex ideation artifacts). Survives session kills:
        # lines are flushed as they arrive.
        self._stream_artifact_path: Optional[str] = config.agent_specific.get(
            "stream_artifact_path"
        )

        # The campaign inbox (docs/research/evolve-hub-design.md v4 §4.2):
        # kapso mints the session id up front (--session-id) so neither the
        # record nor the resume needs parsing; the run loop tails the inbox
        # file and ends the session once a request for this id appears,
        # after the configured grace (inbox.stop_grace_seconds).
        self._session_id: Optional[str] = config.agent_specific.get("session_id")
        self._inbox_path: Optional[str] = config.agent_specific.get("inbox_path")
        if self._inbox_path and not self._session_id:
            raise ValueError("inbox_path requires session_id")
        if self._inbox_path and "inbox_stop_grace_seconds" not in config.agent_specific:
            raise ValueError(
                "inbox_path requires inbox_stop_grace_seconds "
                "(config inbox.stop_grace_seconds)"
            )
        self._inbox_stop_grace_seconds: float = (
            float(config.agent_specific["inbox_stop_grace_seconds"])
            if self._inbox_path else 0.0
        )

        # Verify Claude Code CLI is installed and credentials are available
        self._verify_cli()
    
    def _verify_cli(self):
        """
        Verify Claude Code CLI is installed and resolve authentication.
        """
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude Code CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        
        env = self._get_effective_env()
        self._auth_mode = self._resolve_auth_mode(env)

        if self._auth_mode == "api_key":
            if not env.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Required when auth_mode='api_key'."
                )
        elif self._requested_auth_mode != "auto" and not self._has_oauth_credentials(env):
            raise ValueError(
                "Claude Code OAuth credentials not found. Run 'claude auth login' "
                "or set CLAUDE_CODE_OAUTH_TOKEN."
            )

    def _get_requested_auth_mode(self, agent_specific: Dict[str, Any]) -> str:
        """Normalize the configured auth mode."""
        explicit_mode = agent_specific.get("auth_mode")
        if explicit_mode is not None:
            mode = str(explicit_mode).strip().lower()
        else:
            mode = "auto"

        if mode not in self.AUTH_MODES:
            choices = ", ".join(sorted(self.AUTH_MODES))
            raise ValueError(f"Invalid Claude Code auth_mode {mode!r}. Expected one of: {choices}")

        return mode

    def _resolve_auth_mode(self, env: Dict[str, str]) -> str:
        """Resolve ``auto`` deterministically without exposing credentials."""
        if self._requested_auth_mode != "auto":
            return self._requested_auth_mode

        # API key first (an explicitly exported key is a deliberate choice),
        # then the CLI subscription login.
        if env.get("ANTHROPIC_API_KEY"):
            return "api_key"
        if self._has_oauth_credentials(env):
            return "oauth"

        raise ValueError(
            "No Claude Code credentials found for auth_mode='auto'. Run "
            "'claude auth login' or set ANTHROPIC_API_KEY."
        )

    def _get_effective_env(self) -> Dict[str, str]:
        """Return the process environment plus per-agent overrides."""
        env = os.environ.copy()
        env.update(self._env_overrides)
        return env

    def _has_oauth_credentials(self, env: Dict[str, str]) -> bool:
        """Check OAuth without reading Claude's platform-specific credential store."""
        if env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return True

        status_env = env.copy()
        self._remove_provider_flags(status_env)
        status_env.pop("ANTHROPIC_API_KEY", None)
        status_env.pop("ANTHROPIC_AUTH_TOKEN", None)

        try:
            result = subprocess.run(
                ["claude", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                env=status_env,
            )
        except (OSError, subprocess.SubprocessError):
            return False

        if result.returncode != 0:
            return False

        try:
            status = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return False

        auth_method = str(status.get("authMethod", "")).strip().lower()
        return bool(
            status.get("loggedIn")
            and auth_method not in {"api_key", "api-key", "apikey"}
        )

    @classmethod
    def _remove_provider_flags(cls, env: Dict[str, str]) -> None:
        for name in cls._PROVIDER_FLAGS:
            env.pop(name, None)

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
    
    def generate_code(
        self,
        prompt: str,
        debug_mode: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> CodingResult:
        """
        Generate code using Claude Code CLI.

        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug model
            timeout_seconds: Per-call deadline override; the configured
                timeout applies when None. This is how budget clamps reach
                individual calls on a long-lived agent.

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
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self._timeout
        )

        try:
            # Use streaming or buffered mode. The prompt travels via stdin,
            # never argv (see _build_command).
            if self._streaming:
                return self._run_streaming(prompt, model, effective_timeout)
            else:
                return self._run_buffered(prompt, model, effective_timeout)

        except subprocess.TimeoutExpired:
            return CodingResult(
                success=False,
                output="",
                error=f"Claude Code CLI timed out after {effective_timeout} seconds"
            )
        except Exception as e:
            return CodingResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def resume(
        self,
        cli_session_id: str,
        follow_up: str,
        timeout_seconds: Optional[float] = None,
    ) -> CodingResult:
        """Continue a stored session with one more user message (the inbox
        continuation). Always the streaming runner: the continuation needs
        the stream to capture how the session ends."""
        if self.workspace is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self._timeout
        )
        return self._run_streaming(
            follow_up,
            self.config.model,
            effective_timeout,
            resume_session_id=cli_session_id,
        )

    def _run_buffered(
        self, prompt: str, model: str, timeout_seconds: Optional[float]
    ) -> CodingResult:
        """Run Claude Code CLI in buffered mode (no live output)."""
        cmd = self._build_command(model, use_stream_json=False)
        result = subprocess.run(
            cmd,
            cwd=self.workspace,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
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
                "auth_mode": self._auth_mode,
            }
        )
    
    def _run_streaming(
        self,
        prompt: str,
        model: str,
        timeout_seconds: Optional[float],
        resume_session_id: Optional[str] = None,
    ) -> CodingResult:
        """
        Run Claude Code CLI with live streaming output using stream-json format.

        Parses JSON events and displays Claude's thinking, tool calls, and results
        in real-time for maximum visibility. With ``resume_session_id`` the
        CLI continues that stored session and ``prompt`` is its next user
        message (the inbox continuation).
        """
        if resume_session_id:
            stream_cmd = self._build_command(
                model, use_stream_json=True, resume_session_id=resume_session_id
            )
        else:
            stream_cmd = self._build_command(model, use_stream_json=True)

        start_time = time.time()
        raw_lines: List[str] = []
        artifact_fh = None
        if self._stream_artifact_path:
            os.makedirs(
                os.path.dirname(self._stream_artifact_path), exist_ok=True
            )
            artifact_fh = open(
                self._stream_artifact_path, "a", encoding="utf-8"
            )
        assistant_texts: List[str] = []
        result_text: str = ""
        total_cost: float = 0.0
        is_error: bool = False
        error_msg: str = ""
        # Metrics: count tool calls and track token usage
        tool_call_count: int = 0
        last_tool: str = ""
        input_tokens: int = 0
        output_tokens: int = 0
        
        c = _COLORS  # shorthand
        
        print(f"\n{c['cyan']}━━━ Claude Code Starting ━━━{c['reset']}", flush=True)

        # Ensure stdout/stderr pipes are always closed.
        #
        # Why:
        # - In Python, `subprocess.Popen(..., stdout=PIPE)` creates file objects.
        # - If we don't close them deterministically, Python can emit noisy
        #   `ResourceWarning: unclosed file <_io.TextIOWrapper ...>` at shutdown.
        # - This keeps logs clean and prevents leaking file descriptors in long runs.
        # start_new_session puts the CLI in its own process group, so the
        # deadline kill below can take down Bash-spawned grandchildren that
        # would otherwise outlive the CLI and keep consuming the budget.
        process = subprocess.Popen(
            stream_cmd,
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._get_env(),
            bufsize=1,
            start_new_session=True,
        )
        # Deliver the prompt and close stdin so the CLI starts its turn.
        process.stdin.write(prompt)
        process.stdin.close()
        deadline_exceeded = False
        completed_reaped = False
        completion_armed_at = None  # set when a result event carries all markers
        # The inbox tail: bytes of the inbox file already read, the request
        # ids filed for this session, when the grace runs out, and whether
        # the grace ran out before the session ended itself.
        inbox_seen_bytes = 0
        inbox_request_ids: List[int] = []
        inbox_stop_at: Optional[float] = None
        inbox_killed = False
        
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
                        if artifact_fh is not None:
                            artifact_fh.write(line + "\n")
                            artifact_fh.flush()
                        self._display_stream_event(line, assistant_texts)
                        got_output = True
                        last_heartbeat = time.time()
                        # Completion-reap arming: any activity disarms; a
                        # terminal result carrying ALL markers re-arms. Idle
                        # -wait turns never contain the markers, so healthy
                        # waiting sessions are never touched.
                        completion_armed_at = None
                        if self._completion_markers and '"result"' in line:
                            if all(m in line for m in self._completion_markers):
                                completion_armed_at = time.time()
                
                # Read from stderr if data available
                if process.stderr in readable:
                    err_line = process.stderr.readline()
                    if err_line:
                        err_line = err_line.rstrip('\n')
                        print(f"{c['yellow']}  [stderr] {err_line}{c['reset']}", file=sys.stderr, flush=True)
                        got_output = True
                        last_heartbeat = time.time()
                
                # Reap a session that declared completion (per its output
                # contract) and then went silent without exiting: the CLI
                # process alone is killed — NOT the group — so detached or
                # backgrounded work (e.g. a registered evaluation) survives
                # for the strategy-level guard to collect.
                if (
                    not got_output
                    and retcode is None
                    and completion_armed_at is not None
                    and time.time() - completion_armed_at
                    > _POST_COMPLETION_GRACE_SECONDS
                ):
                    print(
                        f"{c['yellow']}  Session completed its report but the "
                        f"CLI lingered — reaping after "
                        f"{_POST_COMPLETION_GRACE_SECONDS:.0f}s of silence"
                        f"{c['reset']}",
                        flush=True,
                    )
                    completed_reaped = True
                    process.terminate()
                    grace_end = time.time() + _DEADLINE_GRACE_SECONDS
                    while process.poll() is None and time.time() < grace_end:
                        time.sleep(0.1)
                    if process.poll() is None:
                        process.kill()

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
                            if artifact_fh is not None:
                                artifact_fh.write(line + "\n")
                                artifact_fh.flush()
                            self._display_stream_event(line, assistant_texts)
                    if process.stderr:
                        for err_line in process.stderr:
                            print(f"{c['yellow']}  [stderr] {err_line.rstrip()}{c['reset']}", file=sys.stderr, flush=True)
                    break

                # The inbox stop: a request filed for this session means the
                # session is done for now. It gets the grace to end its own
                # turn; then the same kill sequence as the deadline. Checked
                # before the deadline so a request always wins the tie.
                if self._inbox_path and inbox_stop_at is None:
                    inbox_seen_bytes, new_ids = new_requests_for_session(
                        self._inbox_path, self._session_id, inbox_seen_bytes
                    )
                    if new_ids:
                        inbox_request_ids.extend(new_ids)
                        inbox_stop_at = time.time() + self._inbox_stop_grace_seconds
                        print(
                            f"{c['yellow']}  Session asked the person (requests "
                            f"{', '.join(f'#{i}' for i in new_ids)}) — ending it "
                            f"after {self._inbox_stop_grace_seconds:.0f}s unless "
                            f"it ends itself{c['reset']}",
                            flush=True,
                        )
                if (
                    inbox_stop_at is not None
                    and retcode is None
                    and time.time() >= inbox_stop_at
                ):
                    inbox_killed = True
                    print(
                        f"{c['yellow']}  Grace over — ending the session for the "
                        f"inbox{c['reset']}",
                        flush=True,
                    )
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    grace_end = time.time() + _DEADLINE_GRACE_SECONDS
                    while process.poll() is None and time.time() < grace_end:
                        time.sleep(0.1)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    break

                # Enforce the configured deadline. SIGTERM first so the CLI can
                # flush its final result event (which carries the real cost),
                # SIGKILL after the grace window. The group kill relies on
                # start_new_session above; a target that exits between poll and
                # killpg raises and fails loud (accepted race, see design doc).
                if (
                    timeout_seconds is not None
                    and time.time() - start_time >= timeout_seconds
                ):
                    deadline_exceeded = True
                    print(
                        f"{c['yellow']}  Deadline of {timeout_seconds}s reached — "
                        f"terminating Claude Code{c['reset']}",
                        flush=True,
                    )
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    grace_end = time.time() + _DEADLINE_GRACE_SECONDS
                    while process.poll() is None and time.time() < grace_end:
                        time.sleep(0.1)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    break

            if self._inbox_path and not inbox_request_ids:
                # A session that asked and ended its own turn between two
                # polls still asked: one last read after the loop.
                inbox_seen_bytes, new_ids = new_requests_for_session(
                    self._inbox_path, self._session_id, inbox_seen_bytes
                )
                inbox_request_ids.extend(new_ids)

            if (deadline_exceeded or inbox_killed) and process.stdout:
                # Collect anything the CLI flushed during the grace window so a
                # terminated call still reports the cost it managed to emit.
                for line in process.stdout:
                    line = line.rstrip('\n')
                    raw_lines.append(line)
                    if artifact_fh is not None:
                        artifact_fh.write(line + "\n")
                        artifact_fh.flush()
                    self._display_stream_event(line, assistant_texts)

            if artifact_fh is not None:
                artifact_fh.close()

            elapsed = time.time() - start_time
            
            # Parse final result from last JSON line
            for line in reversed(raw_lines):
                try:
                    event = json.loads(line)
                    if event.get("type") == "result":
                        result_text = event.get("result", "")
                        total_cost = event.get("total_cost_usd", 0.0)
                        is_error = event.get("is_error", False)
                        # Extract token usage from result event if available
                        usage = event.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        break
                except json.JSONDecodeError:
                    continue
            
            # Count tool calls and sum per-turn token usage from assistant events.
            # The per-turn usage is a reliable fallback when the result event
            # doesn't include aggregated token counts.
            cumulative_input = 0
            cumulative_output = 0
            for line in raw_lines:
                try:
                    event = json.loads(line)
                    if event.get("type") == "assistant":
                        msg = event.get("message", {})
                        for block in msg.get("content", []):
                            if block.get("type") == "tool_use":
                                tool_call_count += 1
                                tool_input = block.get("input", {}) or {}
                                last_tool = (
                                    f"{block.get('name', '?')}: "
                                    f"{str(tool_input.get('command') or tool_input.get('file_path') or '')[:200]}"
                                )
                        # Sum per-turn usage (input_tokens, output_tokens)
                        usage = msg.get("usage", {})
                        cumulative_input += usage.get("input_tokens", 0)
                        cumulative_output += usage.get("output_tokens", 0)
                except json.JSONDecodeError:
                    continue
            
            # The CLI's own session id, from the init event: the handle a
            # later resume needs. A fresh session was told its id, so a
            # different one is a wiring bug, not a fact to record.
            cli_session_id = resume_session_id or self._session_id
            for line in raw_lines:
                if '"init"' not in line:
                    continue
                event = json.loads(line)
                if event.get("type") == "system" and event.get("subtype") == "init":
                    reported = event.get("session_id")
                    if reported and cli_session_id and reported != cli_session_id:
                        raise RuntimeError(
                            f"Claude Code reports session {reported!r} but "
                            f"this session is {cli_session_id!r}"
                        )
                    cli_session_id = reported or cli_session_id
                    break

            # Use result-level tokens if available, else fall back to summed per-turn
            if input_tokens == 0 and cumulative_input > 0:
                input_tokens = cumulative_input
            if output_tokens == 0 and cumulative_output > 0:
                output_tokens = cumulative_output
            
            print(f"{c['cyan']}━━━ Claude Code Finished ({elapsed:.1f}s, ${total_cost:.4f}, {tool_call_count} tools, {input_tokens}+{output_tokens} tokens) ━━━{c['reset']}\n", flush=True)
            
            # A kill after the session already delivered its complete final
            # report (all completion markers present in the captured stream)
            # is not a failure: classification keys on delivered CONTENT,
            # not on how the process died (R16-P2-1: a fully successful
            # session was logged "Implementation failed" after lingering
            # into the deadline).
            all_output = "\n".join(raw_lines)
            completed_before_kill = bool(self._completion_markers) and all(
                m in all_output for m in self._completion_markers
            )

            if inbox_request_ids:
                # The session asked the person and is done for now. Not a
                # failure however the process ended (its own turn end, our
                # grace kill, or the deadline in between): the node is
                # suspended and the CLI keeps the transcript for the resume.
                self._cumulative_cost += total_cost
                return CodingResult(
                    success=True,
                    output="\n".join(assistant_texts),
                    error=None,
                    cost=total_cost,
                    metadata={
                        "model": model,
                        "auth_mode": self._auth_mode,
                        "elapsed_seconds": elapsed,
                        "stopped_for_inbox": True,
                        "inbox_request_ids": list(inbox_request_ids),
                        "inbox_killed": inbox_killed,
                        "session_id": self._session_id,
                        "cli_session_id": cli_session_id,
                        "tool_call_count": tool_call_count,
                        "last_tool": last_tool,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "raw_log_lines": raw_lines,
                    },
                )

            if completed_reaped:
                self._cumulative_cost += total_cost
                return CodingResult(
                    success=True,
                    output="\n".join(assistant_texts),
                    error=None,
                    cost=total_cost,
                    metadata={
                        "model": model,
                        "auth_mode": self._auth_mode,
                        "elapsed_seconds": elapsed,
                        "cli_session_id": cli_session_id,
                        "completed_reaped": True,
                        "tool_call_count": tool_call_count,
                        "last_tool": last_tool,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "raw_log_lines": raw_lines,
                    },
                )

            if deadline_exceeded:
                # A terminated call still spent real money and real time —
                # record both instead of dropping them with the failure.
                self._cumulative_cost += total_cost
                return CodingResult(
                    success=completed_before_kill,
                    output="\n".join(assistant_texts),
                    error=(
                        None
                        if completed_before_kill
                        else (
                            f"Claude Code CLI exceeded its {timeout_seconds}s "
                            f"deadline and was terminated"
                        )
                    ),
                    cost=total_cost,
                    metadata={
                        "model": model,
                        "auth_mode": self._auth_mode,
                        "elapsed_seconds": elapsed,
                        "cli_session_id": cli_session_id,
                        "deadline_exceeded": True,
                        "completed_before_kill": completed_before_kill,
                        "tool_call_count": tool_call_count,
                        "last_tool": last_tool,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "raw_log_lines": raw_lines,
                    },
                )

            if retcode != 0 or is_error:
                error_msg = result_text if is_error else f"CLI exited with code {retcode}"
                # Failed calls report their parsed cost like successful ones do;
                # expensive failures are exactly what a cost budget must see.
                self._cumulative_cost += total_cost
                return CodingResult(
                    success=False,
                    output="\n".join(assistant_texts),
                    error=error_msg,
                    cost=total_cost,
                    metadata={
                        "model": model,
                        "auth_mode": self._auth_mode,
                        "elapsed_seconds": elapsed,
                        "cli_session_id": cli_session_id,
                        "tool_call_count": tool_call_count,
                        "last_tool": last_tool,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "raw_log_lines": raw_lines,
                    }
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
                    "cli_session_id": cli_session_id,
                    "streaming": True,
                    "auth_mode": self._auth_mode,
                    "tool_call_count": tool_call_count,
                        "last_tool": last_tool,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "raw_log_lines": raw_lines,
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
                        # MCP tool - show full arguments for transparency
                        args_str = json.dumps(tool_input, ensure_ascii=False)
                        print(f"{c['blue']}  [tool:{tool_name}] {args_str}{c['reset']}", flush=True)
                    else:
                        print(f"{c['blue']}  [tool:{tool_name}]{c['reset']}", flush=True)
        
        elif event_type == "user":
            # Tool result returned to Claude — show full content for transparency.
            #
            # Claude Code stream-json tool_result content can be either:
            #   - a plain str  (simple text result)
            #   - a list of content blocks, e.g. [{"type": "text", "text": "..."}]
            # We normalise both forms into a single string before printing.
            content = event.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "tool_result":
                    is_error = block.get("is_error", False)
                    status = "error" if is_error else "ok"
                    raw = block.get("content", "")

                    # Normalise content to a single string
                    if isinstance(raw, str):
                        result_text = raw
                    elif isinstance(raw, list):
                        # List of content blocks — extract text from each
                        parts = []
                        for item in raw:
                            if isinstance(item, dict) and item.get("text"):
                                parts.append(item["text"])
                            elif isinstance(item, str):
                                parts.append(item)
                        result_text = "\n".join(parts)
                    else:
                        result_text = str(raw) if raw else ""

                    if result_text.strip():
                        print(f"{c['dim']}  [result:{status}] ↓{c['reset']}", flush=True)
                        for result_line in result_text.splitlines():
                            print(f"{c['dim']}    {result_line}{c['reset']}", flush=True)
                    else:
                        print(f"{c['dim']}  [result:{status}] (empty){c['reset']}", flush=True)
        
        elif event_type == "result":
            # Final result - show summary
            duration = event.get("duration_ms", 0) / 1000
            cost = event.get("total_cost_usd", 0)
            print(f"{c['dim']}  [result] duration={duration:.1f}s, cost=${cost:.4f}{c['reset']}", flush=True)
        
        else:
            # Unknown event type - show it for debugging
            if event_type:
                print(f"{c['dim']}  [{event_type}:{subtype}]{c['reset']}", flush=True)
    
    # Tools whose delivery machinery does not run in print (-p) sessions —
    # the only mode this adapter ever spawns. ScheduleWakeup accepts the
    # call and prints "the harness re-invokes you when the wakeup fires",
    # but the scheduler that consumes it only runs in interactive /loop
    # dynamic mode: 0/50 fires outside it across our runs AND the official
    # PostTrainBench traces (vs 6/7 inside /loop), reproduced
    # deterministically on the pinned CLI 2.1.157. An agent that trusts the
    # promise and ends its turn with nothing else pending is stranded until
    # the session deadline (run #18 lost 3h GPU exactly this way). Banned
    # structurally rather than via config: no -p session can ever use it.
    PRINT_MODE_DEAD_TOOLS: List[str] = ["ScheduleWakeup"]

    def _build_command(
        self,
        model: str,
        use_stream_json: bool = False,
        resume_session_id: Optional[str] = None,
    ) -> List[str]:
        """Build the Claude Code CLI command.

        The prompt is deliberately NOT part of argv: it is piped via stdin.
        With the prompt in the command line, every process descendant carries
        the solution text in its cmdline, and an agent's own
        `pkill -f <word-from-its-plan>` matches its ancestor and kills the
        session (run #8 lost two sessions this way).
        """
        cmd = [
            "claude",
            "-p",  # Non-interactive (print) mode; prompt arrives on stdin
            "--dangerously-skip-permissions",  # Auto-approve all tool calls
        ]
        # A continuation resumes the stored session; a fresh session gets
        # the id kapso minted, so the inbox record and a later resume never
        # depend on parsing the stream.
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        elif self._session_id:
            cmd.extend(["--session-id", self._session_id])
        
        # Output format: stream-json for live visibility, text for buffered
        if use_stream_json:
            cmd.extend(["--output-format", "stream-json", "--verbose"])
        else:
            cmd.extend(["--output-format", "text"])
        
        # Add model if specified
        if model:
            cmd.extend(["--model", model])

        # Reasoning effort for the session (CLI >= 2.x supports --effort)
        if self._effort:
            cmd.extend(["--effort", str(self._effort)])
        
        # Add allowed tools
        if self._allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._allowed_tools)])

        # Ban tools that structurally cannot work in -p sessions (see
        # PRINT_MODE_DEAD_TOOLS). Verified on CLI 2.1.157: the flag removes
        # the tool from the session's tool list (init event).
        cmd.extend(["--disallowedTools",
                    ",".join(self.PRINT_MODE_DEAD_TOOLS + self._disallowed_tools)])
        
        # Add MCP config if available
        if self._mcp_config_path and self._mcp_config_path.exists():
            cmd.extend(["--mcp-config", str(self._mcp_config_path)])
        
        # Append system prompt if configured (e.g. workspace sandbox instructions)
        if self._append_system_prompt:
            cmd.extend(["--append-system-prompt", self._append_system_prompt])
        
        return cmd
    
    def _get_env(self) -> Dict[str, str]:
        """
        Get environment variables for subprocess.
        
        Sets up an isolated environment for the resolved mode. Provider flags
        and higher-precedence credentials from other modes are removed so an
        explicit selection cannot be silently overridden by Claude Code.
        """
        env = self._get_effective_env()

        if self._auth_mode == "api_key":
            self._remove_provider_flags(env)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            if not env.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not set")
        elif self._auth_mode == "oauth":
            self._remove_provider_flags(env)
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
        else:  # Defensive: auto is always resolved during initialization.
            raise RuntimeError(f"Unresolved Claude Code auth mode: {self._auth_mode}")

        for name in self._env_strip:
            env.pop(name, None)

        for name, value in self._env_defaults.items():
            env.setdefault(name, value)

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
            "oauth": self._auth_mode == "oauth",
            "api_key": self._auth_mode == "api_key",
            "mcp": bool(self._mcp_servers),  # MCP server integration enabled
        }
