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
# - Supports OAuth, direct Anthropic API keys, and AWS Bedrock
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
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
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

# OAuth session-limit failover (claude-swap): a Claude Max subscription's
# usage window exhausting mid-run emits hard 429s in the stream —
# 'rate_limit_event … session limit · resets 8:10pm (UTC)' — and the CLI
# starves until the window resets (killed the official fable-5 SmolLM3 run;
# blocked our runs 29-31 for ~1.8h on 2026-07-25). When recovery tokens are
# present (CLAUDE_CODE_OAUTH_RECOVERY_TOKENS, comma-separated — solve.sh
# derives it from lines 2+ of the oauth_token file), the adapter marks the
# active token exhausted, kills the CLI pid-only, and respawns the SAME
# session via --resume on the next healthy token. Detection threshold is
# stream-parsing policy, not an operator knob (same class as the graces).
_OAUTH_LIMIT_SWAP_THRESHOLD = 3
# When the reset time can't be parsed from the event, assume the provider's
# rolling-window length.
_OAUTH_EXHAUSTED_FALLBACK_SECONDS = 5 * 3600.0
# Process-wide registry: token -> unix time its usage window resets. Shared
# across every adapter instance in the orchestrator so ideation members,
# selectors, implementors and judges all avoid a token one of them exhausted.
_OAUTH_EXHAUSTED_UNTIL: Dict[str, float] = {}
_SESSION_LIMIT_RESET_RE = re.compile(
    r"resets\s+(\d{1,2}):(\d{2})\s*(am|pm)", re.IGNORECASE
)
# What the resumed session reads as its (only) new user turn.
_OAUTH_RESUME_MESSAGE = (
    "Your session was interrupted by a provider usage limit and has been "
    "resumed with restored capacity. Re-read PLAN.md and your workspace "
    "state, then continue exactly where you left off."
)


def _parse_session_limit_reset(line: str) -> Optional[float]:
    """Unix timestamp for 'resets H:MMam/pm (UTC)' in a stream line, if any.

    The CLI prints the reset clock in UTC; resolve it against today (UTC),
    rolling to tomorrow when the time already passed. None when the line
    carries no parseable reset time.
    """
    match = _SESSION_LIMIT_RESET_RE.search(line)
    if match is None:
        return None
    hour = int(match.group(1)) % 12
    if match.group(3).lower() == "pm":
        hour += 12
    now = time.time()
    day_start = now - (now % 86400.0)
    reset_ts = day_start + hour * 3600.0 + int(match.group(2)) * 60.0
    if reset_ts <= now:
        reset_ts += 86400.0
    return reset_ts


def _select_oauth_token(candidates: List[str]) -> str:
    """First candidate whose usage window is not marked exhausted.

    All exhausted → the one that resets soonest (least-bad; the CLI will
    surface the remaining wait in its own rate-limit events).
    """
    now = time.time()
    for token in candidates:
        if _OAUTH_EXHAUSTED_UNTIL.get(token, 0.0) <= now:
            return token
    return min(candidates, key=lambda t: _OAUTH_EXHAUSTED_UNTIL.get(t, 0.0))

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
    - Supports OAuth, direct Anthropic API keys, and AWS Bedrock
    
    Configuration (agent_specific):
    - claude_md_path: Path to CLAUDE.md file (optional)
    - planning_mode: True (default) - use planning
    - timeout: 3600 (default) - CLI timeout in seconds (1 hour)
    - allowed_tools: ["Edit", "Read", "Write", "Bash"] (default)
    - streaming: True (default) - stream output live to terminal for visibility
    - auth_mode: Authentication mode: auto (default), oauth, api_key, or bedrock
    - use_bedrock: Deprecated compatibility alias. True selects bedrock; False selects api_key.
    - aws_region: AWS region for Bedrock (default: "us-east-1")
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
    
    Environment (AWS Bedrock mode):
    - AWS_REGION: AWS region (can also be set via aws_region config)
    - One of:
      - AWS_BEARER_TOKEN_BEDROCK: Bedrock API key (simplest)
      - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY: IAM access keys
      - AWS_PROFILE: SSO profile name (after running aws sso login)
    """

    AUTH_MODES = frozenset({"auto", "oauth", "api_key", "bedrock"})
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
        
        # Authentication settings. ``use_bedrock`` remains an input alias for
        # compatibility, but all runtime behavior is based on the resolved mode.
        self._requested_auth_mode = self._get_requested_auth_mode(config.agent_specific)
        self._auth_mode = self._requested_auth_mode
        self._use_bedrock = self._auth_mode == "bedrock"
        self._aws_region = config.agent_specific.get("aws_region", "us-east-1")
        
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

        # OAuth failover state: which token the last-spawned CLI runs on and
        # which healthy standbys exist (both resolved per spawn in _get_env).
        self._active_oauth_token: str = ""
        self._standby_oauth_tokens: List[str] = []
        self._oauth_swap_count: int = 0

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
        self._use_bedrock = self._auth_mode == "bedrock"

        if self._auth_mode == "bedrock":
            self._verify_bedrock_credentials(env)
        elif self._auth_mode == "api_key":
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
        """Normalize the new auth setting and its deprecated alias."""
        explicit_mode = agent_specific.get("auth_mode")
        has_alias = "use_bedrock" in agent_specific

        if has_alias:
            warnings.warn(
                "Claude Code agent_specific.use_bedrock is deprecated; use "
                "agent_specific.auth_mode instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        if explicit_mode is not None:
            mode = str(explicit_mode).strip().lower()
        elif has_alias:
            mode = "bedrock" if bool(agent_specific["use_bedrock"]) else "api_key"
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

        # Keep Bedrock first for compatibility with Kapso's existing AWS-first
        # deployments, then preserve direct API-key behavior, then use a CLI
        # subscription login.
        if self._has_bedrock_credentials(env):
            return "bedrock"
        if env.get("ANTHROPIC_API_KEY"):
            return "api_key"
        if self._has_oauth_credentials(env):
            return "oauth"

        raise ValueError(
            "No Claude Code credentials found for auth_mode='auto'. Configure AWS "
            "Bedrock credentials, set ANTHROPIC_API_KEY, or run 'claude auth login'."
        )

    def _get_effective_env(self) -> Dict[str, str]:
        """Return the process environment plus per-agent overrides."""
        env = os.environ.copy()
        env.update(self._env_overrides)
        return env

    @staticmethod
    def _has_bedrock_credentials(env: Dict[str, str]) -> bool:
        """Return whether a complete supported AWS credential source is present."""
        return bool(
            env.get("AWS_BEARER_TOKEN_BEDROCK")
            or (
                env.get("AWS_ACCESS_KEY_ID")
                and env.get("AWS_SECRET_ACCESS_KEY")
            )
            or env.get("AWS_PROFILE")
        )

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

    def _verify_bedrock_credentials(self, env: Optional[Dict[str, str]] = None):
        """
        Verify AWS Bedrock credentials are available.
        
        Checks for one of:
        - AWS_BEARER_TOKEN_BEDROCK (Bedrock API key - simplest)
        - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (IAM access keys)
        - AWS_PROFILE (SSO profile)
        
        Also verifies AWS_REGION is set (required for Bedrock).
        """
        # Check for AWS region
        if env is None:
            env = self._get_effective_env()
        aws_region = env.get("AWS_REGION") or self._aws_region
        if not aws_region:
            raise ValueError(
                "AWS_REGION not set. Required for Bedrock mode. "
                "Set AWS_REGION environment variable or aws_region in config."
            )
        
        # Check for at least one authentication method
        has_bearer_token = bool(env.get("AWS_BEARER_TOKEN_BEDROCK"))
        has_access_keys = bool(
            env.get("AWS_ACCESS_KEY_ID") and
            env.get("AWS_SECRET_ACCESS_KEY")
        )
        has_profile = bool(env.get("AWS_PROFILE"))
        
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
                "use_bedrock": self._use_bedrock,
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
        in real-time for maximum visibility.

        resume_session_id is set only by the OAuth session-limit failover:
        this call continues the starved session's conversation on a fresh
        token, with the remaining share of the original deadline.
        """
        stream_cmd = self._build_command(
            model, use_stream_json=True, resume_session_id=resume_session_id
        )
        
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
        seen_session_id: Optional[str] = resume_session_id
        limit_hits = 0
        limit_reset_hint: Optional[float] = None

        try:
            # fd-level I/O. select() sees only the OS pipe, so pairing it
            # with buffered readline() strands the tail of an output burst
            # inside Python's reader buffer until the NEXT bytes arrive — a
            # session-limit storm followed by starved silence never reached
            # the swap threshold. Read raw bytes and split lines here; the
            # wrapper objects are never read from.
            stdout_fd = process.stdout.fileno() if process.stdout else -1
            stderr_fd = process.stderr.fileno() if process.stderr else -1
            stdout_buf = bytearray()
            stderr_buf = bytearray()

            def consume_stdout_line(line: str) -> None:
                nonlocal completion_armed_at, seen_session_id
                nonlocal limit_hits, limit_reset_hint
                raw_lines.append(line)
                if artifact_fh is not None:
                    artifact_fh.write(line + "\n")
                    artifact_fh.flush()
                self._display_stream_event(line, assistant_texts)
                # Completion-reap arming: any activity disarms; a terminal
                # result carrying ALL markers re-arms. Idle-wait turns never
                # contain the markers, so healthy waiting sessions are never
                # touched.
                completion_armed_at = None
                if self._completion_markers and '"result"' in line:
                    if all(m in line for m in self._completion_markers):
                        completion_armed_at = time.time()
                # OAuth failover bookkeeping: remember the session id (for
                # --resume) and count hard usage-window hits. Same malformed-
                # line tolerance as every other stream parse in this file:
                # the CLI interleaves non-JSON notices on stdout.
                if seen_session_id is None and '"session_id"' in line:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = {}
                    if event.get("type") == "system":
                        seen_session_id = event.get("session_id")
                if "session limit" in line:
                    limit_hits += 1
                    reset_hint = _parse_session_limit_reset(line)
                    if reset_hint is not None:
                        limit_reset_hint = reset_hint

            def consume_stderr_line(line: str) -> None:
                print(f"{c['yellow']}  [stderr] {line}{c['reset']}", file=sys.stderr, flush=True)

            def read_stream_chunk(which: str) -> bool:
                """Consume one os.read() chunk — every complete line in it.
                Returns True if bytes arrived; on EOF flushes any trailing
                partial line and retires the fd."""
                nonlocal stdout_fd, stderr_fd
                fd = stdout_fd if which == "stdout" else stderr_fd
                buf = stdout_buf if which == "stdout" else stderr_buf
                consume = consume_stdout_line if which == "stdout" else consume_stderr_line
                chunk = os.read(fd, 65536)
                if not chunk:
                    if buf:
                        consume(buf.decode("utf-8", errors="replace"))
                        buf.clear()
                    if which == "stdout":
                        stdout_fd = -1
                    else:
                        stderr_fd = -1
                    return False
                buf += chunk
                while True:
                    newline_at = buf.find(b"\n")
                    if newline_at < 0:
                        break
                    text = buf[:newline_at].decode("utf-8", errors="replace").rstrip("\r")
                    del buf[: newline_at + 1]
                    consume(text)
                return True

            def drain_stream(which: str, budget_seconds: float) -> None:
                """Bounded post-kill/exit drain. Never blocks past the budget
                on a write end inherited by a surviving background child
                (pid-only kills leave those alive by design)."""
                stop_at = time.time() + budget_seconds
                while time.time() < stop_at:
                    fd = stdout_fd if which == "stdout" else stderr_fd
                    if fd < 0:
                        return
                    try:
                        ready, _, _ = select.select([fd], [], [], 0.2)
                    except (ValueError, OSError):
                        return
                    if not ready:
                        if process.poll() is not None:
                            return
                        continue
                    read_stream_chunk(which)

            last_heartbeat = time.time()
            heartbeat_interval = 10.0  # Show heartbeat every 10 seconds of silence

            while True:
                retcode = process.poll()

                # select on the raw fds (0.5s tick drives deadline checks)
                readable = []
                fds_to_check = [fd for fd in (stdout_fd, stderr_fd) if fd >= 0]
                if fds_to_check:
                    try:
                        readable, _, _ = select.select(fds_to_check, [], [], 0.5)
                    except (ValueError, OSError):
                        # File descriptor closed
                        pass
                elif retcode is None:
                    time.sleep(0.5)  # both pipes at EOF; pace the poll loop

                got_output = False
                if stdout_fd >= 0 and stdout_fd in readable and read_stream_chunk("stdout"):
                    got_output = True
                    last_heartbeat = time.time()
                if stderr_fd >= 0 and stderr_fd in readable and read_stream_chunk("stderr"):
                    got_output = True
                    last_heartbeat = time.time()
                
                # OAuth session-limit failover: the usage window is
                # exhausted (repeated hard 429s) and a healthy recovery
                # token exists. Mark the active token exhausted until its
                # advertised reset, kill the CLI pid-only (backgrounded GPU
                # work survives, same stance as the reap), bank this
                # process's flushed cost, and continue the SAME conversation
                # on the next token via --resume with the remaining share of
                # the deadline.
                if (
                    limit_hits >= _OAUTH_LIMIT_SWAP_THRESHOLD
                    and self._auth_mode == "oauth"
                    and self._active_oauth_token
                    and any(
                        _OAUTH_EXHAUSTED_UNTIL.get(t, 0.0) <= time.time()
                        for t in self._standby_oauth_tokens
                    )
                ):
                    _OAUTH_EXHAUSTED_UNTIL[self._active_oauth_token] = (
                        limit_reset_hint
                        or time.time() + _OAUTH_EXHAUSTED_FALLBACK_SECONDS
                    )
                    self._oauth_swap_count += 1
                    print(
                        f"{c['yellow']}  OAuth usage window exhausted "
                        f"({limit_hits} session-limit events) — swapping to "
                        f"a recovery token and resuming session "
                        f"{seen_session_id or '(fresh start)'}"
                        f"{c['reset']}",
                        flush=True,
                    )
                    process.terminate()
                    grace_end = time.time() + _DEADLINE_GRACE_SECONDS
                    while process.poll() is None and time.time() < grace_end:
                        time.sleep(0.1)
                    if process.poll() is None:
                        process.kill()
                    # Collect the kill-flushed tail so this process's cost
                    # lands in the cumulative ledger before recursing.
                    drain_stream("stdout", _DEADLINE_GRACE_SECONDS)
                    for banked_line in reversed(raw_lines):
                        try:
                            banked_event = json.loads(banked_line)
                        except json.JSONDecodeError:
                            continue
                        if banked_event.get("type") == "result":
                            self._cumulative_cost += banked_event.get(
                                "total_cost_usd", 0.0
                            )
                            break
                    if artifact_fh is not None:
                        artifact_fh.close()
                    remaining = (
                        None
                        if timeout_seconds is None
                        else max(
                            60.0,
                            timeout_seconds - (time.time() - start_time),
                        )
                    )
                    return self._run_streaming(
                        _OAUTH_RESUME_MESSAGE if seen_session_id else prompt,
                        model,
                        remaining,
                        resume_session_id=seen_session_id,
                    )

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
                    drain_stream("stdout", _DEADLINE_GRACE_SECONDS)
                    drain_stream("stderr", _DEADLINE_GRACE_SECONDS)
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

            if deadline_exceeded:
                # Collect anything the CLI flushed during the grace window so a
                # terminated call still reports the cost it managed to emit.
                drain_stream("stdout", _DEADLINE_GRACE_SECONDS)

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
                        "oauth_token_swaps": self._oauth_swap_count,
                        "elapsed_seconds": elapsed,
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
                        "oauth_token_swaps": self._oauth_swap_count,
                        "elapsed_seconds": elapsed,
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
                        "oauth_token_swaps": self._oauth_swap_count,
                        "elapsed_seconds": elapsed,
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
                    "streaming": True,
                    "auth_mode": self._auth_mode,
                    "oauth_token_swaps": self._oauth_swap_count,
                    "use_bedrock": self._use_bedrock,
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

        resume_session_id continues a prior CLI conversation (used by the
        OAuth session-limit failover to restart a starved session on a
        recovery token without losing its context).
        """
        cmd = [
            "claude",
            "-p",  # Non-interactive (print) mode; prompt arrives on stdin
            "--dangerously-skip-permissions",  # Auto-approve all tool calls
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        
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

        if self._auth_mode == "bedrock":
            # Bedrock mode: Set the flag and region
            self._remove_provider_flags(env)
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            
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
        elif self._auth_mode == "api_key":
            self._remove_provider_flags(env)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            if not env.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not set")
        elif self._auth_mode == "oauth":
            self._remove_provider_flags(env)
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            # Failover pool: main token + recovery tokens. The child session
            # gets exactly ONE resolved token; the spare list never reaches
            # the agent's environment (same containment stance as env_strip).
            recovery_raw = env.pop("CLAUDE_CODE_OAUTH_RECOVERY_TOKENS", "")
            candidates = [
                t.strip()
                for t in [env.get("CLAUDE_CODE_OAUTH_TOKEN", ""), *recovery_raw.split(",")]
                if t.strip()
            ]
            if candidates:
                chosen = _select_oauth_token(candidates)
                env["CLAUDE_CODE_OAUTH_TOKEN"] = chosen
                self._active_oauth_token = chosen
                self._standby_oauth_tokens = [t for t in candidates if t != chosen]
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
            "bedrock": self._use_bedrock,  # Using AWS Bedrock for API calls
            "oauth": self._auth_mode == "oauth",
            "api_key": self._auth_mode == "api_key",
            "mcp": bool(self._mcp_servers),  # MCP server integration enabled
        }
