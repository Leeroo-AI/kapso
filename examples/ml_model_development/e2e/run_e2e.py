# run_e2e.py
#
# The budgeted E2E runner for the budget-aware experimentation track
# (docs/evolve/budget-aware-experimentation.mdx, "End-to-end verification
# track"). One file, two roles:
#
#   parent (default): stages a fresh campaign (initial_repo copy + synthetic
#     data), spawns the child under a hard wall of 2x the mode's time budget,
#     then verifies the milestone invariants against the run artifacts.
#   --child: actually calls Kapso.evolve() and writes result.json.
#
# Every knob lives in config.e2e.yaml; this runner only selects a mode.
#
# Usage (from the repo root, in an environment with kapso installed and
# coding-agent credentials configured):
#
#   python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET
#   python examples/ml_model_development/e2e/run_e2e.py --mode GATE_BUDGET
#   python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET \
#       --interrupt-after-seconds 300          # M2: simulated crash
#   python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET \
#       --resume                               # M2: clock continues
#   python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET \
#       --resume --top-up-minutes 60           # M3: budget top-up
#   python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET \
#       --verify-only                          # re-run checks, no spawn

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

E2E_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = E2E_DIR.parent
REPO_ROOT = E2E_DIR.parents[2]
CONFIG_PATH = E2E_DIR / "config.e2e.yaml"

sys.path.insert(0, str(REPO_ROOT))

from examples.ml_model_development.e2e.generate_data import generate

GOAL_HEADER = """
Optimize the ML model in `train.py` to improve accuracy on a synthetic
Spaceship-Titanic dataset (same schema as the Kaggle competition).

## Target File
The file `train.py` contains:
- `train_model()`: Function that trains and returns a model
- `predict_with_model()`: Function that makes predictions with the trained model

IMPORTANT: Do NOT change function names or signatures. Only modify internal
implementation.

## Data
The data/ directory contains:
- train.csv: Training data with features and the Transported target
- test.csv: Test data for final predictions (no target column)

The data/ directory is a fixed evaluation input. Modifying any file in
data/ is prohibited and mechanically voids the candidate's score.
"""

GOAL_EVALUATION_BUILD = """
## Evaluation
Run: python evaluate.py --data-dir ./data --seed 0

The evaluation trains the model, splits training data into train/validation
(90/10), and prints the validation accuracy as `__SCORE__: <value>`.
"""

GOAL_EVALUATION_PROVIDED = """
## Evaluation
The official grader is provided at `kapso_evaluation/grader.py` and is
immutable. It trains the model via train.py's contract, splits training
data into train/validation (90/10), validates the submission, and scores
accuracy. All scoring must route through the grader's
`evaluate_for_accuracy`. Run evaluation only through the registered kapso
evaluation entrypoint.
"""

GOAL_FOOTER = """
## Success Criteria
- Accuracy: higher is better (baseline ~0.50 with DummyClassifier)
- Target: 0.97+ accuracy. This is a hard target: iterate on feature
  engineering, model selection, and hyperparameters until the budget is
  spent. Do not declare success below the target.
"""


def goal_text(seed_eval_defect: bool) -> str:
    evaluation = (
        GOAL_EVALUATION_PROVIDED if seed_eval_defect else GOAL_EVALUATION_BUILD
    )
    return GOAL_HEADER + evaluation + GOAL_FOOTER

WALL_GRACE_SECONDS = 30.0
DATA_SEED = 1337
DATA_ROWS = 2000
DATA_TEST_ROWS = 500


def load_mode_block(mode: str) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    modes = config["modes"]
    if mode not in modes:
        raise SystemExit(
            f"Mode {mode!r} not defined in {CONFIG_PATH}; "
            f"available: {sorted(modes)}"
        )
    return modes[mode]


def stage_campaign(run_dir: Path, *, seed_eval_defect: bool) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    staged_repo = run_dir / "initial_repo"
    shutil.copytree(EXAMPLE_DIR / "initial_repo", staged_repo)
    train_df, test_df = generate(
        rows=DATA_ROWS, test_rows=DATA_TEST_ROWS, seed=DATA_SEED
    )
    data_dir = staged_repo / "data"
    data_dir.mkdir()
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    if seed_eval_defect:
        # The provided grader with the buried consistency-guard defect:
        # passes the single-class baseline, rejects every honest mixed
        # submission — the change-request flow's deterministic trigger.
        shutil.copytree(E2E_DIR / "provided_eval", run_dir / "provided_eval")
        # The defective grader replaces the repo's own evaluate.py as the
        # scoring authority; remove it so nothing routes around the trap.
        (staged_repo / "evaluate.py").unlink()
    print(f"[e2e] staged campaign at {run_dir}")


# =========================================================================
# Child: the actual evolve call
# =========================================================================

def run_child(args: argparse.Namespace, run_dir: Path) -> None:
    from kapso import Kapso

    kapso = Kapso(config_path=str(CONFIG_PATH))
    solution = kapso.evolve(
        goal=goal_text(args.seed_eval_defect),
        initial_repo=str(run_dir / "initial_repo"),
        output_path=str(run_dir / "workspace"),
        max_iterations=args.max_iterations,
        mode=args.mode,
        resume=args.resume,
        time_budget_minutes=args.top_up_minutes,
        eval_dir=(
            str(run_dir / "provided_eval")
            if args.seed_eval_defect
            else None
        ),
    )
    result = {
        "final_score": solution.final_score,
        "succeeded": solution.succeeded,
        "code_path": solution.code_path,
        "metadata": solution.metadata,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(f"[e2e] child finished: score={solution.final_score}")


# =========================================================================
# Parent: hard wall + interruption
# =========================================================================

def kill_group(process: subprocess.Popen, sig: int) -> None:
    # Poll-before-kill; the residual race (child exits between poll and
    # kill) crashes loud, matching the adapter's enforcement pattern.
    if process.poll() is None:
        os.killpg(process.pid, sig)


def sweep_run_orphans(run_dir: Path) -> None:
    """Kill detached agent sessions whose cwd lives under this run.

    The adapter deliberately detaches agents into their own sessions
    (per-call deadline enforcement), so killing the child's process group
    cannot reach them; after a simulated crash or a wall kill they would
    keep spending against a dead campaign. cwd is the precise marker:
    agents run inside the campaign's session/worktree directories.
    """
    run_root = str(run_dir.resolve())
    own_group = os.getpgrp()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        # realpath is non-strict: for a vanished pid it returns the
        # unresolved /proc path, which never matches run_root.
        target = os.path.realpath(str(proc_dir / "cwd"))
        if target != run_root and not target.startswith(run_root + os.sep):
            continue
        pid = int(proc_dir.name)
        if (proc_dir / "stat").exists():
            group = os.getpgid(pid)
            if group != own_group:
                print(f"[e2e] sweeping orphan pgid={group} cwd={target}")
                os.killpg(group, signal.SIGTERM)


def run_walled_child(
    args: argparse.Namespace, wall_seconds: float, run_dir: Path
) -> tuple:
    """Returns (outcome, returncode): 'exited', 'interrupted', 'walled'."""
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--child",
        "--mode", args.mode,
        "--runs-dir", str(args.runs_dir),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.resume:
        command.append("--resume")
    if args.top_up_minutes is not None:
        command += ["--top-up-minutes", str(args.top_up_minutes)]
    if args.seed_eval_defect:
        command.append("--seed-eval-defect")

    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    interrupted = False
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if (
            not interrupted
            and args.interrupt_after_seconds is not None
            and elapsed >= args.interrupt_after_seconds
        ):
            print(
                f"[e2e] simulating crash: SIGTERM after {elapsed:.0f}s"
            )
            kill_group(process, signal.SIGTERM)
            interrupted = True
        if elapsed >= wall_seconds:
            print(
                f"[e2e] HARD WALL: run exceeded {wall_seconds:.0f}s; killing"
            )
            kill_group(process, signal.SIGTERM)
            grace_deadline = time.monotonic() + WALL_GRACE_SECONDS
            while process.poll() is None and time.monotonic() < grace_deadline:
                time.sleep(1.0)
            kill_group(process, signal.SIGKILL)
            process.wait()
            sweep_run_orphans(run_dir)
            return "walled", process.returncode
        time.sleep(2.0)
    if interrupted:
        sweep_run_orphans(run_dir)
        return "interrupted", process.returncode
    return "exited", process.returncode


# =========================================================================
# Verification: the milestone invariants, read from run artifacts
# =========================================================================

def load_checkpoint(run_dir: Path) -> dict:
    path = run_dir / "workspace" / ".kapso" / "run_state.json"
    if not path.exists():
        raise SystemExit(f"FAIL: no checkpoint at {path}")
    return json.loads(path.read_text())


def check_checkpoint_v2(checkpoint: dict, evidence: list) -> bool:
    ok = (
        checkpoint.get("elapsed_seconds", 0) > 0
        and isinstance(checkpoint.get("cost_by_component"), dict)
        and checkpoint.get("status") in {"running", "completed"}
    )
    evidence.append(
        f"status={checkpoint.get('status')} "
        f"elapsed={checkpoint.get('elapsed_seconds', 0):.0f}s "
        f"cost_by_component={checkpoint.get('cost_by_component')}"
    )
    return ok


def check_stop_semantics(checkpoint: dict, result: dict, evidence: list) -> bool:
    from kapso.execution.run_checkpoint import RunCheckpoint

    last_stop = checkpoint.get("last_stop")
    status = checkpoint.get("status")
    evidence.append(f"last_stop={last_stop}")
    if status == "completed":
        return result is None or bool(result.get("succeeded"))
    if (
        last_stop is not None
        and last_stop not in RunCheckpoint.VALID_LAST_STOPS
    ):
        return False
    if result is not None:
        return result.get("metadata", {}).get("stop_detail") == last_stop
    return True


def check_node_telemetry(nodes: list, evidence: list) -> bool:
    missing = [
        node["node_id"]
        for node in nodes
        if not node.get("phase_telemetry")
        or node.get("duration_seconds") is None
    ]
    evidence.append(f"nodes={len(nodes)} missing_telemetry={missing}")
    return len(nodes) > 0 and not missing


def check_single_class_scores(nodes: list, evidence: list) -> bool:
    """A node's score is the mean of exactly one comparability class."""
    blended = []
    for node in nodes:
        score = node.get("score")
        attempts = node.get("evaluation_attempts") or []
        if score is None or not attempts:
            continue
        class_means = {}
        for attempt in attempts:
            key = (
                attempt["evaluator_id"], attempt["fidelity"],
                attempt["fraction"], attempt["seed"],
            )
            class_means.setdefault(key, []).append(attempt["score"])
        means = {
            sum(values) / len(values) for values in class_means.values()
        }
        if not any(abs(score - mean) < 1e-6 for mean in means):
            blended.append(node["node_id"])
    evidence.append(f"blended_scores={blended}")
    return not blended


def check_registry(run_dir: Path, checkpoint: dict, evidence: list) -> bool:
    path = run_dir / "workspace" / ".kapso" / "evaluation_registry.json"
    if not path.exists():
        evidence.append("no evaluation_registry.json")
        return False
    versions = json.loads(path.read_text())
    head_id = versions[-1]["evaluator_id"]
    known_ids = {version["evaluator_id"] for version in versions}
    strategy_state = checkpoint["strategy_state"]
    nodes = strategy_state.get("node_history", [])
    foreign = {
        attempt["evaluator_id"]
        for node in nodes
        for attempt in node.get("evaluation_attempts") or []
    } - known_ids
    anchored = strategy_state.get("scores_evaluator_id") == head_id
    evidence.append(
        f"versions={len(versions)} anchored_on_head={anchored} "
        f"foreign_evaluator_ids={sorted(foreign)}"
    )
    return anchored and not foreign


def check_fractions(
    nodes: list, fast_fraction: float, seed: int, evidence: list
) -> bool:
    wrong = []
    counts = {"fast": 0, "full": 0}
    for node in nodes:
        for attempt in node.get("evaluation_attempts") or []:
            counts[attempt["fidelity"]] += 1
            expected = (
                fast_fraction if attempt["fidelity"] == "fast" else 1.0
            )
            if (
                abs(attempt["fraction"] - expected) > 1e-9
                or attempt["seed"] != seed
            ):
                wrong.append(node["node_id"])
    evidence.append(f"attempts={counts} off_contract={wrong}")
    return not wrong


def check_best_is_full(run_dir: Path, checkpoint: dict, evidence: list) -> bool:
    from kapso.execution.fidelity import (
        TIER_FULL,
        evidence_tier,
        select_committed_candidate,
    )
    from kapso.execution.search_strategies.base import SearchNode

    path = run_dir / "workspace" / ".kapso" / "evaluation_registry.json"
    head_id = json.loads(path.read_text())[-1]["evaluator_id"]
    nodes = [
        SearchNode.from_dict(node)
        for node in checkpoint["strategy_state"].get("node_history", [])
    ]
    any_full = any(
        attempt.fidelity == "full" and attempt.evaluator_id == head_id
        for node in nodes
        for attempt in node.evaluation_attempts
    )
    if not any_full:
        evidence.append("no full-fidelity attempt recorded (budget stop?)")
        return False
    winner = select_committed_candidate(nodes, evaluator_id=head_id)
    tier = (
        evidence_tier(winner, head_id) if winner is not None else None
    )
    head_branch = subprocess.run(
        [
            "git", "-C", str(run_dir / "workspace"),
            "rev-parse", "--abbrev-ref", "HEAD",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    delivered = (
        winner is not None and head_branch == winner.branch_name
    )
    evidence.append(
        f"committed_winner={getattr(winner, 'node_id', None)} tier={tier} "
        f"delivered_branch={head_branch}"
    )
    return winner is not None and tier == TIER_FULL and delivered


def check_all_full_fidelity(nodes: list, evidence: list) -> bool:
    off_profile = [
        node["node_id"]
        for node in nodes
        if node.get("build_fidelity") != "full"
        or node.get("eval_fidelity") != "full"
        or any(
            attempt["fidelity"] != "full"
            for attempt in node.get("evaluation_attempts") or []
        )
    ]
    evidence.append(f"non_full_nodes={off_profile}")
    return not off_profile


def check_child_exit(returncode, result, evidence: list) -> bool:
    """A crash after the checkpoint is still a failed run, not a pass."""
    evidence.append(
        f"returncode={returncode} result_json={result is not None}"
    )
    return returncode in (None, 0) and result is not None


def check_change_request_flow(
    run_dir: Path, checkpoint: dict, evidence: list
) -> bool:
    """The seeded-defect leg must fire the whole CR chain: a second
    registered version, an anchored transition carrying the requester's
    priority, and the requester actually measured under the new head.
    """
    registry_path = (
        run_dir / "workspace" / ".kapso" / "evaluation_registry.json"
    )
    versions = json.loads(registry_path.read_text())
    strategy_state = checkpoint.get("strategy_state", {})
    transition = strategy_state.get("evaluator_transition") or {}
    head_id = versions[-1]["evaluator_id"]
    priority = transition.get("priority_node_id")
    nodes = strategy_state.get("node_history", [])
    requester_measured = any(
        attempt["evaluator_id"] == head_id
        for node in nodes
        if node["node_id"] == priority
        for attempt in node.get("evaluation_attempts") or []
    )
    evidence.append(
        f"versions={len(versions)} transition={transition.get('status')} "
        f"priority_node={priority} requester_measured={requester_measured}"
    )
    return (
        len(versions) >= 2
        and transition.get("status") == "anchored"
        and priority is not None
        and requester_measured
    )


def check_artifact(
    run_dir: Path, checkpoint: dict, evidence: list, *, required: tuple
) -> bool:
    """The final checkout must actually land on an experiment branch.

    File existence alone is not evidence — the baseline files exist on
    main too, which is exactly how a crashed checkout previously slipped
    past this check.
    """
    workspace = run_dir / "workspace"
    present = [name for name in required if (workspace / name).exists()]
    head = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    nodes = checkpoint.get("strategy_state", {}).get("node_history", [])
    any_scored = any(node.get("score") is not None for node in nodes)
    evidence.append(f"checked_out={present} head={head} scored={any_scored}")
    if len(present) != 2:
        return False
    if any_scored:
        return head.startswith("generic_exp_")
    return True


def check_gate_wall(
    checkpoint: dict, mode_block: dict, evidence: list
) -> bool:
    """Elapsed is bounded by budget plus the enforcement residue.

    The residue is derived, not guessed: the last admitted phase may run
    its clamp floor, the feedback generator is unclamped by design (its
    configured timeout bounds it), plus kill grace. Anything beyond that
    means a clamp failed to bind.
    """
    budget_seconds = mode_block["budget"]["time_budget_minutes"] * 60.0
    feedback_timeout = mode_block["feedback_generator"]["agent_specific"][
        "timeout"
    ]
    residue = 60.0 + feedback_timeout + 60.0
    nodes = checkpoint.get("strategy_state", {}).get("node_history", [])
    elapsed = checkpoint.get("elapsed_seconds", 0.0)
    slow_phases = [
        (node["node_id"], phase, values.get("duration_seconds", 0.0))
        for node in nodes
        for phase, values in (node.get("phase_telemetry") or {}).items()
        if values.get("duration_seconds", 0.0) >= budget_seconds
    ]
    evidence.append(
        f"elapsed={elapsed:.0f}s bound={budget_seconds + residue:.0f}s "
        f"unclamped_phases={slow_phases}"
    )
    return elapsed <= budget_seconds + residue and not slow_phases


def verify(
    args: argparse.Namespace,
    run_dir: Path,
    outcome: str,
    child_returncode=None,
) -> int:
    mode_block = load_mode_block(args.mode)
    budget_block = mode_block["budget"]
    has_maintainer = "evaluation_maintainer" in mode_block
    fidelity_block = budget_block.get("fidelity") or {}
    fidelity_on = fidelity_block.get("mode", "off") != "off"

    if outcome == "walled":
        print("[e2e] VERDICT: FAIL (hard wall breached — hung campaign)")
        return 2

    checkpoint = load_checkpoint(run_dir)
    nodes = checkpoint.get("strategy_state", {}).get("node_history", [])
    result_path = run_dir / "result.json"
    result = (
        json.loads(result_path.read_text()) if result_path.exists() else None
    )

    checks = [
        ("checkpoint_v2", lambda ev: check_checkpoint_v2(checkpoint, ev)),
        (
            "single_class_scores",
            lambda ev: check_single_class_scores(nodes, ev),
        ),
    ]
    if outcome == "interrupted":
        # A pre-iteration interrupt legitimately has zero nodes — the
        # bootstrap checkpoint is what makes it resumable. Completed nodes,
        # when present, must still carry telemetry.
        if nodes:
            checks.append(
                ("node_telemetry", lambda ev: check_node_telemetry(nodes, ev))
            )
        checks.append(
            (
                "interrupt_left_resumable_state",
                lambda ev: checkpoint.get("status") == "running",
            )
        )
    else:
        checks += [
            ("node_telemetry", lambda ev: check_node_telemetry(nodes, ev)),
            (
                "child_clean_exit",
                lambda ev: check_child_exit(child_returncode, result, ev),
            ),
            (
                "stop_semantics",
                lambda ev: check_stop_semantics(checkpoint, result, ev),
            ),
            (
                "artifact_checked_out",
                lambda ev: check_artifact(
                    run_dir,
                    checkpoint,
                    ev,
                    # The seeded-defect variant replaces evaluate.py with
                    # the provided grader; it is legitimately absent.
                    required=(
                        ("train.py",)
                        if args.seed_eval_defect
                        else ("train.py", "evaluate.py")
                    ),
                ),
            ),
        ]
    if has_maintainer:
        checks.append(
            ("registry", lambda ev: check_registry(run_dir, checkpoint, ev))
        )
    if args.seed_eval_defect and outcome == "exited":
        checks.append(
            (
                "change_request_flow",
                lambda ev: check_change_request_flow(run_dir, checkpoint, ev),
            )
        )
    if fidelity_on:
        fast_fraction = fidelity_block["eval"]["fast_fraction"]
        seed = mode_block["evaluation_maintainer"]["subsample_seed"]
        checks.append(
            (
                "fraction_contract",
                lambda ev: check_fractions(nodes, fast_fraction, seed, ev),
            )
        )
        if outcome == "exited":
            checks.append(
                (
                    "committed_best_is_full",
                    lambda ev: check_best_is_full(run_dir, checkpoint, ev),
                )
            )
    if has_maintainer and not fidelity_on:
        checks.append(
            (
                "fidelity_off_is_full_passthrough",
                lambda ev: check_all_full_fidelity(nodes, ev),
            )
        )
    if args.mode == "GATE_BUDGET" and outcome == "exited":
        checks.append(
            (
                "gate_wall_and_clamps",
                lambda ev: check_gate_wall(checkpoint, mode_block, ev),
            )
        )

    failures = 0
    print(f"\n[e2e] verification — mode={args.mode} outcome={outcome}")
    for name, runner in checks:
        evidence: list = []
        ok = runner(evidence)
        failures += 0 if ok else 1
        detail = f"  ({'; '.join(evidence)})" if evidence else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")

    print(
        f"[e2e] VERDICT: {'PASS' if failures == 0 else 'FAIL'} "
        f"({len(checks) - failures}/{len(checks)} checks)"
    )
    return 0 if failures == 0 else 1


# =========================================================================
# Entry
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted E2E runner")
    parser.add_argument(
        "--mode",
        default="E2E_BUDGET",
        choices=["E2E_BUDGET", "E2E_BUDGET_FIDELITY_OFF", "GATE_BUDGET"],
    )
    parser.add_argument("--runs-dir", type=Path, default=E2E_DIR / "runs")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top-up-minutes", type=float, default=None)
    parser.add_argument("--interrupt-after-seconds", type=float, default=None)
    parser.add_argument("--seed-eval-defect", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    run_dir = args.runs_dir / (
        args.mode.lower() + ("_cr" if args.seed_eval_defect else "")
    )

    if args.child:
        run_child(args, run_dir)
        return

    if args.verify_only:
        result_exists = (run_dir / "result.json").exists()
        outcome = "exited" if result_exists else "interrupted"
        raise SystemExit(verify(args, run_dir, outcome))

    if args.resume:
        checkpoint_path = run_dir / "workspace" / ".kapso" / "run_state.json"
        if not checkpoint_path.exists():
            raise SystemExit(
                f"--resume requires an existing campaign at {checkpoint_path}"
            )
        # A fresh child run supersedes the previous attempt's result.
        stale_result = run_dir / "result.json"
        if stale_result.exists():
            stale_result.unlink()
    else:
        stage_campaign(run_dir, seed_eval_defect=args.seed_eval_defect)
        if args.stage_only:
            print("[e2e] staged only; not running")
            return

    budget_minutes = load_mode_block(args.mode)["budget"][
        "time_budget_minutes"
    ]
    wall_seconds = 2.0 * budget_minutes * 60.0
    print(
        f"[e2e] mode={args.mode} budget={budget_minutes}min "
        f"hard_wall={wall_seconds / 60:.0f}min"
    )
    outcome, child_returncode = run_walled_child(args, wall_seconds, run_dir)
    raise SystemExit(verify(args, run_dir, outcome, child_returncode))


if __name__ == "__main__":
    main()
