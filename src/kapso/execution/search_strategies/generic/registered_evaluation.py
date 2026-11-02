"""Registered-evaluation governance for the generic strategy.

Owns the maintainer-registered evaluation feature: the manifest-of-record
parsing chain (the wrapper's machine-readable line is the score of record),
evaluation-attempt recording, frame-run execution under Kapso's own
deadline-bounded subprocess (the registered re-measurements and the cost
mode candidate run share one runner), the session-teardown guard with
durable-archive recovery, registered-tree sync into sessions, and the
evaluation instructions rendered into implementation prompts. Stateless
functions only — GenericSearch assembles arguments from its state and
delegates here.
"""

import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from kapso.core.processes import process_commands
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
# The provided suite's contract: run as given, never edited; the
# directory's rule (docs/build/evaluation.mdx) allows the data and the
# reports a run writes beside it.
PROVIDED_EVALUATION_INSTRUCTIONS = """The evaluation is provided in `kapso_evaluation/` and is read-and-execute only:

1. **Read it** to learn what it measures and how it is run.
2. **Run it as given** and capture its full output. Never modify, delete,
   replace or add evaluator source files; the runtime outputs it writes or
   you write beside it (`result.json`, predictions, a profile) are allowed.
3. **Retry on crash**: fix your solution, never the suite, and run it again
   (max 3 attempts). A suite you believe defective is reported in your
   final response, not patched."""

DEFAULT_EVALUATION_INSTRUCTIONS = """You MUST build and run evaluation in `kapso_evaluation/` directory:

1. **Create evaluation script**: `kapso_evaluation/evaluate.py` (or similar)
2. **Evaluation should**:
   - Test your solution against the goal criteria
   - Output a clear score or success/failure indication
   - Be fair and actually test what it claims to test
   - NOT be hardcoded or trivially pass

3. **Run the evaluation**: Execute your evaluation script and capture output.

4. **Retry on crash**: If evaluation crashes, fix the issue and retry (max 3 attempts)."""


def candidate_environment_command(
    command: List[str], *, env_strip: List[str], env_defaults: Dict[str, str],
    env_overrides: Optional[Dict[str, str]],
) -> List[str]:
    """Apply configured session environment transport without reading secrets."""
    overrides = env_overrides or {}
    for name in [*env_strip, *env_defaults, *overrides]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid candidate environment variable name: {name!r}")
    if not env_strip and not env_defaults and not overrides:
        return command
    strip_args = [argument for name in env_strip for argument in ("-u", name)]
    default_assignments = [
        f'if [ "${{{name}+x}}" != x ]; then\n'
        f"export {name}={shlex.quote(str(value))}\nfi"
        for name, value in env_defaults.items()
    ]
    script = "\n".join([*default_assignments, 'exec "$@"'])
    return [
        "/usr/bin/env", *strip_args,
        *(f"{name}={value}" for name, value in overrides.items()),
        "/bin/sh", "-c", script, "kapso-evaluation", *command,
    ]


def manifest_of_record(
    node: SearchNode,
    *,
    registered_evaluation_command: Optional[str],
    fidelity_decision: Optional[FidelityDecision],
    registered_subsample_seed: int,
) -> Optional[Dict[str, Any]]:
    """The granted-class manifest from the evaluation output's last
    manifest line.

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


@dataclass(frozen=True)
class FrameRun:
    """How one evaluator subprocess under Kapso's own deadline ended."""

    stdout: str
    stderr: str
    returncode: int
    overran: bool
    deadline_seconds: Optional[float]
    duration_seconds: float

    @property
    def completed(self) -> bool:
        """The process ran to a clean exit inside its deadline."""
        return not self.overran and self.returncode == 0

    @property
    def outcome(self) -> str:
        """One clause on how the run ended, predicated of the run: the
        node keeps it and the judge reads it, since a deadline kill
        leaves no trace in the captured streams."""
        if self.overran:
            return (
                f"was killed at its {self.deadline_seconds:g}s deadline "
                f"after {self.duration_seconds:.1f}s"
            )
        if self.returncode != 0:
            return f"exited {self.returncode} after {self.duration_seconds:.1f}s"
        return f"completed in {self.duration_seconds:.1f}s"


def run_frame_subprocess(
    command: List[str], *, cwd: str, deadline_seconds: Optional[float],
) -> FrameRun:
    """Run one evaluator command under Kapso's deadline, capturing both
    streams.

    This is the staged-execution-ownership step from the design: the
    eval-only runs whose integrity matters most execute under Kapso's
    own deadline-bounded subprocess, not inside an agent session. The
    deadline is the affordability window and an overrun is an
    operational outcome, never a campaign failure: the process group
    is killed and the run reports as not completed, exactly like a
    non-zero exit. Timing estimates gate admission; they do not kill
    campaigns.
    """
    # Spooled files, never PIPE: an evolved evaluator may emit
    # per-window progress lines, and an undrained 64KB pipe
    # deadlocks the child mid-write while this loop sleeps
    # (observed live: rel-event/user-ignore froze 6h inside
    # _emit_process_line, 2026-08-12).
    # errors="replace": the child's bytes are not ours to vet, and one stray
    # non-UTF-8 byte must not discard a finished run at the read-back.
    stdout_file = tempfile.TemporaryFile(mode="w+", errors="replace")
    stderr_file = tempfile.TemporaryFile(mode="w+", errors="replace")
    run_started = time.monotonic()
    overran = False
    process = subprocess.Popen(
        command,
        cwd=cwd,
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
            break
        time.sleep(0.5)
    process.wait()
    duration = time.monotonic() - run_started
    stdout_file.seek(0)
    stdout = stdout_file.read()
    stdout_file.close()
    stderr_file.seek(0)
    stderr = stderr_file.read()
    stderr_file.close()
    if overran:
        print(
            "[GenericSearch] Registered evaluation exceeded its "
            f"{deadline_seconds:.0f}s affordability window; "
            "recorded as a failed attempt"
        )
    elif process.returncode != 0:
        print(
            "[GenericSearch] Registered evaluation failed "
            f"(exit {process.returncode}): {stderr}"
        )
    return FrameRun(
        stdout=stdout, stderr=stderr, returncode=process.returncode,
        overran=overran, deadline_seconds=deadline_seconds,
        duration_seconds=duration,
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
    """Frame-run the registered evaluation on an existing artifact (a
    VALIDATE grant or a bridge), appending the measurement as an
    evaluation attempt. A run that failed or overran reports None."""
    command = shlex.split(evaluation_command(
        fidelity=fidelity,
        fraction=fraction,
        seed=registered_subsample_seed,
    ))
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
        run = run_frame_subprocess(
            command, cwd=worktree, deadline_seconds=deadline_seconds,
        )
    if not run.completed:
        return None
    manifest = parse_manifest_line(run.stdout)
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
            duration_seconds=run.duration_seconds,
        )
    )
    if record_eval_duration is not None:
        # Feed the measured duration back into the timing model: real
        # full-scale runs replace calibration extrapolation (samples
        # persist in the registry; the provider-backed policy sees the
        # tightened upper immediately).
        record_eval_duration(
            fraction=fraction, duration_seconds=run.duration_seconds
        )
    return score


def execute_candidate_evaluation(
    node: SearchNode,
    *,
    deadline_seconds: Optional[float],
    registered_evaluation_command: Optional[str],
    workspace,
    workspace_dir: str,
    env_strip: List[str],
    env_defaults: Dict[str, str],
    env_overrides: Optional[Dict[str, str]],
) -> FrameRun:
    """Cost mode: Kapso runs the admitted candidate's evaluator itself.

    The registered evaluator when one is registered (synced from the
    head, as every frame run), otherwise the candidate's own
    `kapso_evaluation/evaluate.py`, under the session's configured
    environment transport. The run's capture replaces the session's
    output as the judge's evidence and its outcome marks that a run
    happened; the node is valid only when the run completed, and in
    registered mode its score is then the manifest's. The data
    manifest is not re-verified here: enforce_evaluation_integrity
    verified this ref moments earlier in the same finalization.
    """
    command = candidate_environment_command(
        shlex.split(registered_evaluation_command)
        if registered_evaluation_command
        else [sys.executable, "kapso_evaluation/evaluate.py"],
        env_strip=env_strip, env_defaults=env_defaults,
        env_overrides=env_overrides,
    )
    with workspace.materialize_ref(node.branch_name) as worktree:
        if registered_evaluation_command:
            sync_registered_evaluation(worktree, workspace_dir)
        run = run_frame_subprocess(
            command, cwd=worktree, deadline_seconds=deadline_seconds,
        )
    node.evaluation_output = run.stdout + run.stderr
    node.evaluation_script_path = (
        "kapso_evaluation/kapso_eval.py" if registered_evaluation_command
        else "kapso_evaluation/evaluate.py"
    )
    node.frame_run_outcome = run.outcome
    node.evaluation_valid = run.completed
    node.score = (
        float(parse_manifest_line(run.stdout)["score"])
        if registered_evaluation_command and run.completed
        else None
    )
    return run


def sync_registered_evaluation(session_folder: str, workspace_dir: str) -> None:
    """Overwrite the session's evaluation tree with the registered one."""
    source = os.path.join(workspace_dir, "kapso_evaluation")
    destination = os.path.join(session_folder, "kapso_evaluation")
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)


def evaluation_instructions(registered_evaluation_command: Optional[str], *, provided: bool = False) -> str:
    """The evaluation contract, rendered once in the implementation prompt,
    in one of three forms: the registered one when a maintainer owns
    evaluation, the provided suite's (run it, never edit it) when the
    build brought one, build-your-own otherwise."""
    if not registered_evaluation_command:
        return PROVIDED_EVALUATION_INSTRUCTIONS if provided else DEFAULT_EVALUATION_INSTRUCTIONS
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


def deferred_evaluation_instructions(registered_evaluation_command: Optional[str], *, provided: bool = False) -> str:
    """The build-only contract of a cost-aware campaign: Kapso estimates the
    finalized candidate's platform cost and runs the evaluator itself once
    the candidate passes its cap and the campaign budget."""
    if registered_evaluation_command:
        evaluation_code = (
            "The evaluation is maintained by the system and is read-and-execute "
            f"only: `{registered_evaluation_command}` keeps its manifest fields and "
            "its entrypoint, and altering its behavior at rest or at runtime voids "
            "the score."
        )
    elif provided:
        evaluation_code = (
            "The evaluation is provided in `kapso_evaluation/` and is read-and-execute "
            "only: Kapso runs it after admission; never modify, delete, replace or add "
            "evaluator source files."
        )
    else:
        evaluation_code = (
            "Create `kapso_evaluation/evaluate.py`; it must test the solution "
            "against the goal fairly, print a clear score, and not be hardcoded or "
            "trivially pass. It may report measured platform cost with one manifest "
            'line: `KAPSO_EVAL_MANIFEST {"cost": "2.5 credits"}`.'
        )
    return f"""This is a build-only session. Kapso estimates platform cost on the finalized
candidate before granting execution and runs the evaluator itself once the
candidate passes its cost cap and the campaign budget.

1. **Do not run the candidate or evaluation**: no training, no platform jobs, no
   paid platform operations, no expensive artifacts in the shared cache. Reading
   code and writing the candidate are allowed.
2. **Evaluation code**: {evaluation_code}
3. **Final tags**: leave `<evaluation_output>` empty and report `<score>null</score>`.
   The measurement recon above still applies to how you build; per-stratum
   scores come from Kapso's run."""


def await_registered_evaluation(
    output_text: str,
    *,
    session: str,
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
    Parallel lanes share the archive root, so only a run whose selection
    label names ``session`` is this session's to recover.
    """
    if not registered_evaluation_command:
        return None
    if MANIFEST_MARKER in (output_text or ""):
        return None

    # A distinctive fragment of the registered command to find it running by:
    # prefer the script path token; fall back to the full command string.
    tokens = [
        t for t in registered_evaluation_command.split() if ".py" in t
    ]
    needle = tokens[0] if tokens else registered_evaluation_command

    def _live_eval_pid():
        for pid, command in process_commands().items():
            if needle in command:
                return pid
        return None

    bound = clamped_timeout(implementation_timeout)
    waited = 0.0
    pid = _live_eval_pid()
    if pid is not None:
        bound_label = "without a deadline" if bound is None else f"up to {bound:.0f}s"
        print(
            f"[GenericSearch] Registered evaluation still running "
            f"(pid {pid}) after session end — waiting {bound_label} "
            "before teardown"
        )
    while pid is not None and (bound is None or waited < bound):
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
        with open(
            os.path.join(run_dir, "private", "selection.json"), "r", encoding="utf-8"
        ) as fh:
            if json.load(fh)["session"] != session:
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
