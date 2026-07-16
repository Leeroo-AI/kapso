"""Codex CLI ideation runner for the generic strategy's ensemble ideation.

Deliberately NOT a full CodingAgent adapter: ideation is a read-only,
text-out task, so this shells out one non-interactive `codex exec` in the
materialized parent worktree with a read-only sandbox. Codex brings its own
read tools and web search (--search); MCP parity with the Claude members is
not needed.

Env hygiene: the subprocess never sees OPENAI_API_KEY. On PostTrainBench
judge-scored tasks the harness exposes that key strictly for evaluate.py;
the scaffold's own Codex access must come from CODEX_API_KEY or the CLI
login (~/.codex/auth.json) — the same pattern the upstream scaffolds use.

Forensics (run #8, R8-F2): when ``artifacts_dir`` is given, the transcript
stream and the final-message file are PERSISTED there instead of deleted,
so a failed member turn leaves evidence on disk.
"""

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time

# Grace between SIGTERM and SIGKILL on deadline (mirrors the Claude adapter).
_DEADLINE_GRACE_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.5

# How much of a failed turn's stream tail to surface into the main trace.
STREAM_TAIL_CHARS = 400


def run_codex_ideation(
    prompt: str,
    model: str,
    cwd: str,
    timeout_seconds: float,
    effort: str | None = None,
    artifacts_dir: str | None = None,
) -> tuple[str, bool, float, dict]:
    """Run one read-only `codex exec` ideation call.

    Returns (output_text, timed_out, duration_seconds, meta) where meta is
    {"last_message_empty": bool, "stream_tail": str, "stream_path": str|None,
    "last_path": str|None}. Output prefers the clean final message; a killed
    or aborted turn falls back to the merged transcript stream (which echoes
    the submitted prompt — callers must filter echo artifacts).
    """
    if not shutil.which("codex"):
        raise RuntimeError(
            "Codex CLI not found. Install with: npm install -g @openai/codex"
        )

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        last_path = os.path.join(artifacts_dir, f"codex_{safe_model}.last.txt")
        out_path = os.path.join(artifacts_dir, f"codex_{safe_model}.stream.txt")
        open(last_path, "w").close()
        persist = True
    else:
        last_fd, last_path = tempfile.mkstemp(
            prefix="codex_ideation_", suffix=".last"
        )
        os.close(last_fd)
        out_fd, out_path = tempfile.mkstemp(
            prefix="codex_ideation_", suffix=".out"
        )
        os.close(out_fd)
        persist = False

    # --output-last-message isolates the agent's final answer from the
    # transcript stream; --search enables the native web_search tool.
    cmd = [
        "codex",
        "--search",
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-last-message",
        last_path,
        "-m",
        model,
    ]
    if effort:
        cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
    # Prompt via stdin, never argv (same self-pkill hazard as the Claude
    # adapter: argv-borne prompt text makes kill patterns match ancestors).

    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)

    # Stream to a file, not a PIPE: codex output can exceed the 64KB pipe
    # buffer mid-run, which would deadlock a poll loop.
    out_file = open(out_path, "w")
    started = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=out_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process.stdin.write(prompt)
    process.stdin.close()

    deadline = started + timeout_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)

    timed_out = process.poll() is None
    if timed_out:
        os.killpg(process.pid, signal.SIGTERM)
        grace_end = time.monotonic() + _DEADLINE_GRACE_SECONDS
        while process.poll() is None and time.monotonic() < grace_end:
            time.sleep(0.1)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    out_file.close()
    with open(last_path, "r", encoding="utf-8", errors="replace") as fh:
        last_message = fh.read().strip()
    with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
        stream = fh.read()
    if not persist:
        os.unlink(last_path)
        os.unlink(out_path)

    # Prefer the clean final message; a killed/aborted run has only its stream.
    output = last_message if last_message else stream
    meta = {
        "last_message_empty": not last_message,
        "stream_tail": stream[-STREAM_TAIL_CHARS:],
        "stream_path": out_path if persist else None,
        "last_path": last_path if persist else None,
    }
    return output, timed_out, time.monotonic() - started, meta
