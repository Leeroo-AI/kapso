# Codex CLI coding agent adapter.
#
# Runs non-interactive `codex exec` sessions with a full-access sandbox so
# the agent can write code, install packages, and run the registered
# evaluation — the write-capable counterpart of the read-only ideation
# runner in search_strategies/generic/codex_ideation.py.
#
# Auth: the subprocess never sees OPENAI_API_KEY (same hygiene as the
# ideation runner); Codex authenticates via ~/.codex/auth.json (ChatGPT
# login) or CODEX_API_KEY. Cost: the Codex CLI reports no billing
# telemetry, so get_cumulative_cost() is always 0 and campaign ledgers
# undercount codex compute (documented behavior, matching ideation).
#
# Deadline semantics: on timeout the codex PID alone gets SIGTERM->SIGKILL
# — NOT the process group — so a registered evaluation the agent launched
# survives the kill for the strategy's teardown guard / durable-archive
# recovery (the same principle as the Claude adapter's completion-reap).

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from kapso.execution.inbox import new_requests_for_session
from kapso.execution.coding_agents.base import (
    CodingAgentConfig,
    CodingAgentInterface,
    CodingResult,
)

_DEADLINE_GRACE_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.5
# The first `--json` event names the thread; the stream is merged with
# stderr, so the id is matched on the line rather than parsed as JSON.
_THREAD_STARTED_PATTERN = re.compile(
    r'"type"\s*:\s*"thread\.started".*?"thread_id"\s*:\s*"([^"]+)"'
)
_DEFAULT_TIMEOUT_SECONDS = 3600.0
_STREAM_TAIL_CHARS = 2000


def mcp_config_overrides(mcp_servers: Dict[str, Any]) -> List[str]:
    """Translate a claude-shaped MCP server spec into codex `-c` overrides.

    {name: {command, args, cwd, env}} becomes dotted-path TOML overrides
    (`mcp_servers.<name>.command="..."` etc.) — the per-invocation
    equivalent of `codex mcp add`. JSON string encoding is valid TOML for
    these values (paths and identifiers).
    """
    overrides: List[str] = []
    for name, spec in (mcp_servers or {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"MCP server name not TOML-bare-key safe: {name!r}")
        overrides += ["-c", f"mcp_servers.{name}.command={json.dumps(spec['command'])}"]
        args = spec.get("args") or []
        overrides += [
            "-c",
            f"mcp_servers.{name}.args=[{', '.join(json.dumps(a) for a in args)}]",
        ]
        if spec.get("cwd"):
            overrides += ["-c", f"mcp_servers.{name}.cwd={json.dumps(spec['cwd'])}"]
        env = spec.get("env") or {}
        if env:
            body = ", ".join(f"{k} = {json.dumps(v)}" for k, v in env.items())
            overrides += ["-c", f"mcp_servers.{name}.env={{{body}}}"]
    return overrides


class CodexCodingAgent(CodingAgentInterface):
    """Non-interactive `codex exec` sessions as a pluggable coding agent.

    agent_specific keys:
        effort: model_reasoning_effort forwarded to the CLI (OpenAI ceiling
            is "xhigh"; there is no "max" tier)
        timeout: per-call deadline seconds (default 3600)
        sandbox: codex sandbox mode (default "danger-full-access" — the
            write+network parity of Claude's skip-permissions sessions)
        web_search: enable the native --search tool (default True)
        mcp_servers: claude-shaped MCP server spec {name: {command, args,
            cwd, env}} mounted via per-invocation `-c mcp_servers.*`
            overrides
        streaming: tee the live transcript to the console (default False;
            the artifact file always gets the full stream)
        env_strip: env var names removed from the child environment
        env_overrides: env vars SET on the child (e.g. expansion lane pins)
        env_defaults: env vars applied set-if-absent (ambient wins)
        stream_artifact_path: persist the transcript stream to this path
            (append mode — one session may span several calls)
        session_id: the id kapso minted for the campaign inbox (the gate
            files requests under it; the run loop stops the session when
            one appears)
        inbox_path: the campaign's inbox file to tail
        inbox_stop_grace_seconds: how long the session gets to end its own
            turn after asking, before it is ended (config
            inbox.stop_grace_seconds)
        capture_thread_id: pass --json and record the thread id from the
            thread.started event — what `resume` needs
    """

    def __init__(self, config: CodingAgentConfig):
        super().__init__(config)
        spec = config.agent_specific or {}
        self._effort: Optional[str] = spec.get("effort")
        # None is what a lane with no implementation_timeout and no time budget
        # passes (the strategy clamps to "unbounded"); the adapter needs a
        # deadline, so that is the default one (live L2, 2026-09-04).
        configured_timeout = spec.get("timeout")
        self._timeout: float = (
            float(configured_timeout) if configured_timeout is not None else _DEFAULT_TIMEOUT_SECONDS
        )
        self._sandbox: str = str(spec.get("sandbox", "danger-full-access"))
        self._web_search: bool = bool(spec.get("web_search", True))
        self._mcp_overrides: List[str] = mcp_config_overrides(
            spec.get("mcp_servers") or {}
        )
        self._streaming: bool = bool(spec.get("streaming", False))
        self._env_strip: List[str] = list(spec.get("env_strip", []))
        self._env_overrides: Dict[str, str] = {
            str(k): str(v) for k, v in (spec.get("env_overrides") or {}).items()
        }
        self._env_defaults: Dict[str, str] = {
            str(k): str(v) for k, v in (spec.get("env_defaults") or {}).items()
        }
        self._stream_artifact_path: Optional[str] = spec.get("stream_artifact_path")
        self._session_id: Optional[str] = spec.get("session_id")
        self._inbox_path: Optional[str] = spec.get("inbox_path")
        if self._inbox_path and not self._session_id:
            raise ValueError("inbox_path requires session_id")
        if self._inbox_path and "inbox_stop_grace_seconds" not in spec:
            raise ValueError(
                "inbox_path requires inbox_stop_grace_seconds "
                "(config inbox.stop_grace_seconds)"
            )
        self._inbox_stop_grace_seconds: float = (
            float(spec["inbox_stop_grace_seconds"]) if self._inbox_path else 0.0
        )
        self._capture_thread_id: bool = bool(spec.get("capture_thread_id", False))
        self._thread_id: Optional[str] = None
        self._workspace: str = ""
        if not shutil.which("codex"):
            raise RuntimeError(
                "Codex CLI not found. Install with: npm install -g @openai/codex"
            )

    def initialize(self, workspace: str) -> None:
        self._workspace = workspace

    def generate_code(
        self,
        prompt: str,
        debug_mode: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> CodingResult:
        if not self._workspace:
            raise RuntimeError("CodexCodingAgent used before initialize()")
        model = self.config.debug_model if debug_mode else self.config.model
        timeout = float(timeout_seconds) if timeout_seconds is not None else self._timeout
        return self._run_session(model, [], prompt, timeout)

    def resume(
        self,
        cli_session_id: str,
        follow_up: str,
        timeout_seconds: Optional[float] = None,
    ) -> CodingResult:
        """Continue a stored thread with one more user message (the inbox
        continuation): `codex exec <options> resume <thread> -`, the
        message on stdin. Exec options precede `resume` (verified on
        codex-cli 0.144.1)."""
        if not self._workspace:
            raise RuntimeError("CodexCodingAgent used before initialize()")
        timeout = float(timeout_seconds) if timeout_seconds is not None else self._timeout
        return self._run_session(
            self.config.model, ["resume", cli_session_id, "-"], follow_up, timeout
        )

    def _exec_command(self, model: str, last_message_path: str) -> List[str]:
        """`codex [--search] exec <exec options>` — the shared head of a
        fresh session and a resumed one."""
        cmd = ["codex"]
        if self._web_search:
            cmd.append("--search")
        cmd.extend(
            [
                "exec",
                "--sandbox",
                self._sandbox,
                "--skip-git-repo-check",
                "--color",
                "never",
                "--output-last-message",
                last_message_path,
                "-m",
                model,
            ]
        )
        if self._effort:
            cmd.extend(["-c", f'model_reasoning_effort="{self._effort}"'])
        cmd.extend(self._mcp_overrides)
        if self._capture_thread_id:
            cmd.append("--json")
        return cmd

    def _run_session(
        self, model: str, tail: List[str], prompt: str, timeout: float
    ) -> CodingResult:
        last_fd, last_path = tempfile.mkstemp(prefix="codex_agent_", suffix=".last")
        os.close(last_fd)
        if self._stream_artifact_path:
            os.makedirs(os.path.dirname(self._stream_artifact_path), exist_ok=True)
            stream_path = self._stream_artifact_path
            stream_file = open(stream_path, "a")
            persist_stream = True
        else:
            stream_fd, stream_path = tempfile.mkstemp(
                prefix="codex_agent_", suffix=".stream"
            )
            os.close(stream_fd)
            stream_file = open(stream_path, "w")
            persist_stream = False

        cmd = self._exec_command(model, last_path) + tail

        env = os.environ.copy()
        # OPENAI_API_KEY passes through: the CLI's billing is pinned to the
        # ChatGPT login via preferred_auth_method="chatgpt" in config.toml,
        # and the sessions' own tooling needs the key (lanes measured it
        # absent and shipped timeout-hedged 96-row hosted batches,
        # 2026-08-12 rel-amazon/user-churn).
        for name in self._env_strip:
            env.pop(name, None)
        for name, value in self._env_overrides.items():
            env[name] = value
        for name, value in self._env_defaults.items():
            env.setdefault(name, value)

        started = time.monotonic()
        # Prompt via stdin, never argv (argv-borne prompt text makes kill
        # patterns match ancestors — the self-pkill hazard).
        process = subprocess.Popen(
            cmd,
            cwd=self._workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )

        # Drain stdout continuously (a full pipe would deadlock the child):
        # every line lands in the stream artifact; streaming mode also tees
        # it to the console, matching the claude adapter's live transcript.
        # With --json the first event names the thread, the resume handle.
        def _drain() -> None:
            for line in process.stdout:
                stream_file.write(line)
                stream_file.flush()
                if self._capture_thread_id and self._thread_id is None:
                    match = _THREAD_STARTED_PATTERN.search(line)
                    if match:
                        self._thread_id = match.group(1)
                if self._streaming:
                    print(f"[codex] {line.rstrip()}", flush=True)

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        process.stdin.write(prompt)
        process.stdin.close()

        # The inbox stop (design v4 §4.2): a request filed for this session
        # means the session is done for now; it gets the grace to end its
        # own turn, then a PID-only SIGTERM like the deadline's.
        deadline = started + timeout
        inbox_seen_bytes = 0
        inbox_request_ids: List[int] = []
        inbox_stop_at: Optional[float] = None
        inbox_killed = False
        while process.poll() is None and time.monotonic() < deadline:
            if self._inbox_path and inbox_stop_at is None:
                inbox_seen_bytes, new_ids = new_requests_for_session(
                    self._inbox_path, self._session_id, inbox_seen_bytes
                )
                if new_ids:
                    inbox_request_ids.extend(new_ids)
                    inbox_stop_at = time.monotonic() + self._inbox_stop_grace_seconds
                    print(
                        "[codex] session asked the person (requests "
                        f"{', '.join(f'#{i}' for i in new_ids)}) — ending it after "
                        f"{self._inbox_stop_grace_seconds:.0f}s unless it ends itself",
                        flush=True,
                    )
            if inbox_stop_at is not None and time.monotonic() >= inbox_stop_at:
                inbox_killed = True
                break
            time.sleep(_POLL_INTERVAL_SECONDS)
        if self._inbox_path and not inbox_request_ids:
            # A session that asked and ended its own turn before the first
            # poll still asked: one last read after the loop.
            inbox_seen_bytes, new_ids = new_requests_for_session(
                self._inbox_path, self._session_id, inbox_seen_bytes
            )
            inbox_request_ids.extend(new_ids)

        timed_out = process.poll() is None and not inbox_killed
        if process.poll() is None:
            # PID only — children (e.g. a running registered evaluation)
            # survive for the strategy-level teardown guard.
            os.kill(process.pid, signal.SIGTERM)
            grace_end = time.monotonic() + _DEADLINE_GRACE_SECONDS
            while process.poll() is None and time.monotonic() < grace_end:
                time.sleep(0.1)
            if process.poll() is None:
                os.kill(process.pid, signal.SIGKILL)
            process.wait()
        elapsed = time.monotonic() - started

        reader.join(timeout=_DEADLINE_GRACE_SECONDS)
        stream_file.close()
        with open(last_path, "r", encoding="utf-8", errors="replace") as fh:
            last_message = fh.read().strip()
        os.unlink(last_path)
        with open(stream_path, "r", encoding="utf-8", errors="replace") as fh:
            stream = fh.read()
        if not persist_stream:
            os.unlink(stream_path)

        output = last_message if last_message else stream
        error: Optional[str] = None
        if inbox_request_ids:
            # Asked the person and stopped — not a failure however the
            # process ended; the node is suspended and the thread keeps
            # its context for the resume.
            error = None
        elif timed_out:
            error = f"Codex CLI killed by its deadline after {elapsed:.0f}s"
        elif process.returncode != 0:
            error = (
                f"Codex CLI exited with code {process.returncode}: "
                f"{stream[-_STREAM_TAIL_CHARS:]}"
            )
        elif not last_message:
            error = "Codex CLI produced no final message"
        success = error is None

        return CodingResult(
            success=success,
            output=output,
            files_changed=self._changed_files(),
            error=error,
            cost=0.0,
            metadata={
                "deadline_exceeded": timed_out,
                "elapsed_seconds": elapsed,
                "returncode": process.returncode,
                "stream_path": stream_path if persist_stream else None,
                "stopped_for_inbox": bool(inbox_request_ids),
                "inbox_request_ids": list(inbox_request_ids),
                "inbox_killed": inbox_killed,
                "session_id": self._session_id,
                "cli_session_id": self._thread_id,
            },
        )

    def _changed_files(self) -> List[str]:
        """Changed paths per git status; empty for non-repo workspaces."""
        if not os.path.isdir(os.path.join(self._workspace, ".git")):
            return []
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self._workspace,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return []
        return [
            re.sub(r"^.{3}", "", line).strip()
            for line in proc.stdout.splitlines()
            if line.strip()
        ]

    def cleanup(self) -> None:
        pass

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "native_git": False,
            "sandbox": True,
            "planning_mode": False,
            "cost_tracking": False,
            "streaming": True,
        }
