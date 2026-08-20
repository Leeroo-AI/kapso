"""Registered-evaluation governance for the generic strategy.

Owns the maintainer-registered evaluation feature: the manifest-of-record
parsing chain (the wrapper's machine-readable line is the score of record),
evaluation-attempt recording, frame-run execution under Kapso's own
deadline-bounded subprocess, the session-teardown guard with durable-archive
recovery, registered-tree sync into sessions, and the evaluation
instructions rendered into implementation prompts. Stateless functions only
— GenericSearch assembles arguments from its state and delegates here.
"""

import glob
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from kapso.execution.evaluation_integrity import verify_data_manifest
from kapso.execution.evaluation_maintainer.maintainer import (
    MANIFEST_MARKER,
    evaluation_command,
    parse_manifest_line,
)
from kapso.execution.fidelity import EvaluationAttempt, FidelityDecision
from kapso.execution.search_strategies.base import SearchNode

# Enforcement mechanic (mirrors the coding-agent adapter's deadline grace):
# time granted between SIGTERM and SIGKILL when a frame run overruns.
_FRAME_RUN_KILL_GRACE_SECONDS = 2.0


# Byte-identical to the pre-maintainer template text: rendered whenever no
# maintainer-registered evaluation exists, keeping default prompts unchanged.
DEFAULT_EVALUATION_INSTRUCTIONS = """You MUST build and run evaluation in `kapso_evaluation/` directory:

1. **Create evaluation script**: `kapso_evaluation/evaluate.py` (or similar)
2. **Evaluation should**:
   - Test your solution against the goal criteria
   - Output a clear score or success/failure indication
   - Be fair and actually test what it claims to test
   - NOT be hardcoded or trivially pass

3. **Run the evaluation**: Execute your evaluation script and capture output.

4. **Retry on crash**: If evaluation crashes, fix the issue and retry (max 3 attempts)."""


def manifest_of_record(
    node: SearchNode,
    *,
    registered_evaluation_command: Optional[str],
    fidelity_decision: Optional[FidelityDecision],
    registered_subsample_seed: int,
) -> Optional[Dict[str, Any]]:
    """The granted-class manifest from the session's last manifest line.

    Registered mode only: the wrapper contractually prints one
    machine-readable KAPSO_EVAL_MANIFEST line per run, so an LLM never
    has to be the parser of record (two live nodes lost real
    measurements to a killed feedback call). The line is model
    output: a present-but-malformed manifest raises. A well-formed
    line for a different class — the agent ran a custom fraction or
    the wrong fidelity — is not this node's canonical measurement and
    returns None (documented default).
    """
    if not registered_evaluation_command:
        return None
    output = node.evaluation_output or ""
    last_line = None
    for line in output.splitlines():
        if line.strip().startswith(MANIFEST_MARKER):
            last_line = line.strip()
    if last_line is None:
        return None
    manifest = parse_manifest_line(last_line)
    decision = fidelity_decision
    granted_fidelity = (
        decision.eval_fidelity if decision is not None else "full"
    )
    granted_fraction = (
        decision.eval_fraction if decision is not None else 1.0
    )
    if (
        manifest["fidelity"] != granted_fidelity
        or abs(float(manifest["fraction"]) - granted_fraction) > 1e-9
        or int(manifest["seed"]) != registered_subsample_seed
    ):
        print(
            "[GenericSearch] Manifest class mismatch: granted "
            f"{granted_fidelity}/{granted_fraction}/"
            f"{registered_subsample_seed}, session ran "
            f"{manifest['fidelity']}/{manifest['fraction']}/"
            f"{manifest['seed']} — no mechanical score of record"
        )
        return None
    if "score" not in manifest:
        return None
    return manifest


def manifest_score_of_record(
    node: SearchNode,
    *,
    registered_evaluation_command: Optional[str],
    fidelity_decision: Optional[FidelityDecision],
    registered_subsample_seed: int,
) -> Optional[float]:
    """The granted-class score from the session's last manifest line."""
    manifest = manifest_of_record(
        node,
        registered_evaluation_command=registered_evaluation_command,
        fidelity_decision=fidelity_decision,
        registered_subsample_seed=registered_subsample_seed,
    )
    if manifest is None:
        return None
    return float(manifest["score"])


def record_evaluation_attempt(
    node: SearchNode,
    *,
    registered_evaluator_id: Optional[str],
    fidelity_decision: Optional[FidelityDecision],
    registered_subsample_seed: int,
    workspace,
) -> None:
    """Append the node's measurement under the registered evaluator.

    Only trustworthy measurements become attempts: a registered
    evaluator must exist and the node must carry a valid score.
    """
    if (
        not registered_evaluator_id
        or node.score is None
        or node.had_error
        or not node.evaluation_valid
    ):
        return
    decision = fidelity_decision
    fraction = decision.eval_fraction if decision is not None else 1.0
    commit_sha = workspace.repo.commit(node.branch_name).hexsha
    node.evaluation_attempts.append(
        EvaluationAttempt(
            commit_sha=commit_sha,
            evaluator_id=registered_evaluator_id,
            fidelity=node.eval_fidelity,
            fraction=fraction,
            seed=registered_subsample_seed,
            score=node.score,
            duration_seconds=node.phase_telemetry.get(
                "implementation", {}
            ).get("duration_seconds"),
        )
    )


def execute_registered_evaluation(
    target: SearchNode,
    *,
    fidelity: str,
    fraction: float,
    deadline_seconds: Optional[float],
    registered_evaluator_id: Optional[str],
    registered_subsample_seed: int,
    registered_data_manifest: Optional[Dict[str, str]],
    workspace,
    workspace_dir: str,
    record_eval_duration,
) -> Optional[float]:
    """Frame-run the registered evaluation on an existing artifact.

    This is the staged-execution-ownership step from the design: the
    eval-only runs whose integrity matters most execute under Kapso's
    own deadline-bounded subprocess, not inside an agent session. The
    deadline is the affordability window and an overrun is an
    operational outcome, never a campaign failure: the process group
    is killed and the attempt reports None, exactly like a non-zero
    exit. Timing estimates gate admission; they do not kill campaigns.
    """
    command = shlex.split(
        evaluation_command(
            fidelity=fidelity,
            fraction=fraction,
            seed=registered_subsample_seed,
        )
    )
    run_started = time.monotonic()
    with workspace.materialize_ref(target.branch_name) as worktree:
        # The branch's own evaluation tree is whatever version its
        # session ran under — a frame run trusting it would execute a
        # RETIRED evaluator while labeling the attempt with the head's
        # id (observed live: a bridge labeled v2 executed the branch's
        # v1 tree). The registered head is the only ruler frame runs
        # execute.
        sync_registered_evaluation(worktree, workspace_dir)
        if registered_data_manifest:
            data_problem = verify_data_manifest(
                worktree, registered_data_manifest
            )
            if data_problem:
                print(
                    "[GenericSearch] Registered evaluation refused: "
                    f"{data_problem}"
                )
                return None
        # Spooled files, never PIPE: an evolved evaluator may emit
        # per-window progress lines, and an undrained 64KB pipe
        # deadlocks the child mid-write while this loop sleeps
        # (observed live: rel-event/user-ignore froze 6h inside
        # _emit_process_line, 2026-08-12).
        stdout_file = tempfile.TemporaryFile(mode="w+")
        stderr_file = tempfile.TemporaryFile(mode="w+")
        process = subprocess.Popen(
            command,
            cwd=worktree,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            overran = (
                deadline_seconds is not None
                and time.monotonic() - run_started >= deadline_seconds
            )
            if overran:
                os.killpg(process.pid, signal.SIGTERM)
                grace = time.monotonic() + _FRAME_RUN_KILL_GRACE_SECONDS
                while process.poll() is None and time.monotonic() < grace:
                    time.sleep(0.2)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                stdout_file.close()
                stderr_file.close()
                print(
                    "[GenericSearch] Registered evaluation exceeded its "
                    f"{deadline_seconds:.0f}s affordability window; "
                    "recorded as a failed attempt"
                )
                return None
            time.sleep(0.5)
        process.wait()
        stdout_file.seek(0)
        stdout = stdout_file.read()
        stdout_file.close()
        stderr_file.seek(0)
        stderr = stderr_file.read()
        stderr_file.close()
    duration = time.monotonic() - run_started
    if process.returncode != 0:
        print(
            "[GenericSearch] Registered evaluation failed "
            f"(exit {process.returncode}): {stderr}"
        )
        return None
    manifest = parse_manifest_line(stdout)
    score = float(manifest["score"])
    target.evaluation_attempts.append(
        EvaluationAttempt(
            commit_sha=workspace.repo.commit(
                target.branch_name
            ).hexsha,
            evaluator_id=registered_evaluator_id,
            fidelity=fidelity,
            fraction=fraction,
            seed=registered_subsample_seed,
            score=score,
            duration_seconds=duration,
        )
    )
    if record_eval_duration is not None:
        # Feed the measured duration back into the timing model: real
        # full-scale runs replace calibration extrapolation (samples
        # persist in the registry; the provider-backed policy sees the
        # tightened upper immediately).
        record_eval_duration(
            fraction=fraction, duration_seconds=duration
        )
    return score


def sync_registered_evaluation(session_folder: str, workspace_dir: str) -> None:
    """Overwrite the session's evaluation tree with the registered one."""
    source = os.path.join(workspace_dir, "kapso_evaluation")
    destination = os.path.join(session_folder, "kapso_evaluation")
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)


def evaluation_instructions(registered_evaluation_command: Optional[str]) -> str:
    """Registered-evaluation contract when a maintainer owns evaluation;
    the historical build-your-own instructions otherwise."""
    if not registered_evaluation_command:
        return DEFAULT_EVALUATION_INSTRUCTIONS
    return f"""The evaluation is maintained by the system and is read-and-execute only.

1. **Run the registered evaluation**: `{registered_evaluation_command}`
   and capture its full output, including the KAPSO_EVAL_MANIFEST line.
2. **Run it in the FOREGROUND and stay alive until it finishes.** Your
   session exists only while you are actively working: the moment you stop
   responding, the session ends and every process it started is killed. No
   background job survives you, and no completion notification can ever
   reach you — there is no later. Never launch the registered evaluation
   with `&`, `nohup`, or a background task. Full-fidelity builds taking
   many minutes is normal and expected: run the command blocking with a
   generous tool timeout, and if a single call hits its cap, keep
   re-issuing blocking foreground waits until KAPSO_EVAL_MANIFEST is in
   your transcript. Only then write your final response. An evaluation you
   background and abandon scores nothing — the entire iteration is wasted.
3. **Never alter evaluation behavior — at rest or at runtime.** Editing
   anything under `kapso_evaluation/`, rewriting protected data inputs,
   monkey-patching or hooking evaluation modules from your own code
   (e.g. via imports, `sys.modules`, or wrappers), or otherwise
   circumventing any evaluation check all count as tampering: the score
   is voided and the experiment loses. There is no sanctioned bypass.
4. **If you believe the evaluation itself is defective — broken OR
   mismeasuring — do not fix it, patch it, or route around it.** Broken
   means crashes or wrong wiring. MISMEASURING means the score ranks
   candidates in an order that will not hold, and you can detect it with
   two cheap checks that never touch test data:
   (a) Resolution — bootstrap the validation metric on your best candidate
   (resample rows with replacement, ~100 draws) to get its standard error,
   and measure how much your candidates' PREDICTIONS actually differ (mean
   pairwise rank correlation over validation rows). If materially different
   candidates score within about two standard errors of each other,
   validation is not separating them and its argmax is close to a coin
   flip.
   (b) Representativeness — when validation is a single time slice, compare
   its event volume and label rate against surrounding history and the
   prediction period. An irregular slice (calendar shock, outage) can rank
   candidates in the WRONG ORDER, not merely with less precision, and no
   amount of tuning fixes an inverted ordering.
   In either case file a request by including this tag in your final
   response:
   <evaluation_change_request>the defect, with measured evidence — numbers,
   not suspicion — and a concrete remedy, e.g. additional validation
   windows generated by the task's own label-generating code over
   training-era timestamps, each window closing before the prediction
   period, aggregated so no single slice decides the
   ranking</evaluation_change_request>
   Then still report your results from the run you attempted. Requests are
   triaged adversarially against your evidence and the budget is small (a
   few per campaign): file once, with your best case.
   TIMING — file at the FIRST confirmation, never the last iteration. Run
   both diagnostics during your first iteration, before optimizing against
   the score, and the moment they confirm a defect put the request in that
   same final response. Do not wait to build a stronger case: a later
   transition voids every measurement made under the old evaluation
   (scores never cross evaluator versions), so each candidate measured
   before you file adds to what the change throws away — filed in the
   first iteration it voids almost nothing; filed at the end it voids the
   campaign's rankings with no budget left to re-measure them.
   REMEDY SHAPE — propose the least-breaking remedy that fixes the defect,
   and say which kind yours is: (1) rescore stored outputs — a better
   metric, weighting, or aggregation over predictions every run already
   archives, so all prior candidates re-rank for free; (2) same-contract
   re-measurement — new windows, slices, or seeds prepared by the
   evaluator in the standard input layout and fed through the UNCHANGED
   candidate entrypoint, so prior candidates stay measurable at compute
   cost; (3) contract-breaking — candidates must produce outputs they
   never produced, orphaning all prior work; propose it only when nothing
   less fixes the defect.
   A confirmed defect is fixed, your work is re-measured first under the
   corrected evaluation, and every score is re-ranked under it — prior
   champions losing rank afterwards is the system working, not a
   regression. If a transition already happened, earlier designs whose
   scores now show as unmeasured were measured under the superseded
   evaluation: porting the strongest of them to the current evaluation
   contract and archiving the port is often the highest-value experiment
   available.
5. **Retry on transient crashes** of your own code (max 3 attempts)."""


def await_registered_evaluation(
    output_text: str,
    *,
    registered_evaluation_command: Optional[str],
    registered_evaluation_archive_glob: Optional[str],
    clamped_timeout: Callable[[float], float],
    implementation_timeout: float,
    session_started_ts: float,
):
    """Teardown guard for the registered evaluation (relbench finding 14 /
    Issue 2). MUST run BEFORE finalize_session: its rmtree destroys a
    still-running grader's working tree. If the session ended without a
    manifest in its output while the registered evaluation process is
    alive, wait for it (bounded by the live budget clamp). Then attempt
    recovery from the durable run archive — the grader archives the run
    (including manifest.txt) OUTSIDE the workspace before printing the
    manifest line — and return the recovered manifest line (or None).
    """
    if not registered_evaluation_command:
        return None
    if MANIFEST_MARKER in (output_text or ""):
        return None

    # A distinctive fragment of the registered command for /proc matching:
    # prefer the script path token; fall back to the full command string.
    tokens = [
        t for t in registered_evaluation_command.split() if ".py" in t
    ]
    needle = tokens[0] if tokens else registered_evaluation_command

    def _live_eval_pid():
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            cmdline_path = os.path.join("/proc", pid, "cmdline")
            if not os.path.exists(cmdline_path):
                continue
            with open(cmdline_path, "rb") as fh:
                cmdline = fh.read().replace(b"\0", b" ").decode(
                    "utf-8", "replace"
                )
            if needle in cmdline:
                return int(pid)
        return None

    bound = clamped_timeout(implementation_timeout)
    waited = 0.0
    pid = _live_eval_pid()
    if pid is not None:
        print(
            f"[GenericSearch] Registered evaluation still running "
            f"(pid {pid}) after session end — waiting up to {bound:.0f}s "
            "before teardown"
        )
    while pid is not None and waited < bound:
        time.sleep(5)
        waited += 5
        pid = _live_eval_pid()

    if not registered_evaluation_archive_glob:
        return None
    started = session_started_ts
    candidates = []
    for runs_root in glob.glob(registered_evaluation_archive_glob):
        for entry in glob.glob(os.path.join(runs_root, "run_*")):
            if os.path.isdir(entry) and os.path.getmtime(entry) > started:
                candidates.append(entry)
    for run_dir in sorted(
        candidates, key=os.path.getmtime, reverse=True
    ):
        manifest_path = os.path.join(run_dir, "manifest.txt")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as fh:
            line = fh.read().strip()
        if not line.startswith(MANIFEST_MARKER):
            continue
        print(
            "[GenericSearch] Recovered registered-evaluation manifest "
            f"from durable archive: {run_dir}"
        )
        return line
    return None
