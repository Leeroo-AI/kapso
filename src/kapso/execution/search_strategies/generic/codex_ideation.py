"""Codex CLI ideation runner for the generic strategy's ensemble ideation.

Deliberately NOT a full CodingAgent adapter: ideation is a read-only,
text-out task, so this shells out one non-interactive `codex exec` in the
materialized parent worktree with a read-only sandbox. Codex brings its own
read tools and web search; MCP parity with the Claude members is not needed.

Env hygiene: the subprocess never sees OPENAI_API_KEY. On PostTrainBench
judge-scored tasks the harness exposes that key strictly for evaluate.py;
the scaffold's own Codex access must come from CODEX_API_KEY or the CLI
login (~/.codex/auth.json) — the same pattern the upstream scaffolds use.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time

# Grace between SIGTERM and SIGKILL on deadline (mirrors the Claude adapter).
_DEADLINE_GRACE_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.5


def run_codex_ideation(
    prompt: str,
    model: str,
    cwd: str,
    timeout_seconds: float,
    effort: str | None = None,
) -> tuple[str, bool, float]:
    """Run one read-only `codex exec` ideation call.

    Returns (output_text, timed_out, duration_seconds). Output is stdout and
    stderr merged: on a deadline kill the partial stream is the salvageable
    research, exactly like the Claude members' partial output.
    """
    if not shutil.which("codex"):
        raise RuntimeError(
            "Codex CLI not found. Install with: npm install -g @openai/codex"
        )

    # --output-last-message isolates the agent's final answer from the
    # transcript stream, which echoes the prompt (whose tag examples would
    # otherwise self-match candidate extraction) and duplicates the final
    # message. The merged stream is kept only as timeout-salvage material.
    last_fd, last_path = tempfile.mkstemp(
        prefix="codex_ideation_", suffix=".last"
    )
    os.close(last_fd)
    cmd = [
        "codex",
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
    cmd.append(prompt)

    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)

    # Stream to a file, not a PIPE: codex output can exceed the 64KB pipe
    # buffer mid-run, which would deadlock a poll loop.
    out_fd, out_path = tempfile.mkstemp(prefix="codex_ideation_", suffix=".out")
    out_file = os.fdopen(out_fd, "w")
    started = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=out_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

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
    os.unlink(last_path)
    with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
        stream = fh.read()
    os.unlink(out_path)

    # Prefer the clean final message; a killed run has only its stream.
    output = last_message if last_message else stream
    return output, timed_out, time.monotonic() - started
