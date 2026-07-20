"""IOAI 2026 Home Task 3 ("Animal Deduction") problem handler.

Hands the coding agent the official task statement plus the operational
discipline a short, single-GPU, hard-deadline run demands. Scoring during the
search is self-reported (the agent runs dataset/evaluate.py on dev.csv and
reports the mean score via <score> tags / kapso_evaluation/result.json); the
runner scores the held-out split privately after the campaign.
"""

import os
import time

from kapso.environment.handlers.base import ProblemHandler


class AnimalDeductionHandler(ProblemHandler):
    """Handler for the IOAI 2026 Animal Deduction home task."""

    maximize_scoring = True

    def __init__(
        self,
        task_dir: str,
        statement: str,
        deadline_ts: float,
        session_caps: dict,
    ):
        super().__init__(additional_context="")
        if not isinstance(session_caps, dict) or not {
            "ideation_timeout",
            "implementation_timeout",
        } <= session_caps.keys():
            raise ValueError(
                "session_caps must be the runner's shaped session timeouts "
                "(ideation_timeout/implementation_timeout, seconds)"
            )
        self.task_dir = os.path.abspath(task_dir)
        self.statement = statement.strip()
        self.deadline_ts = deadline_ts
        self.session_caps = session_caps
        self.dataset_dir = os.path.join(self.task_dir, "dataset")
        self.artifacts_dir = os.path.join(self.task_dir, "artifacts")
        self.submission_dir = os.path.join(self.task_dir, "submission")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def _remaining_str(self) -> str:
        remaining = max(0, int(self.deadline_ts - time.time()))
        return f"{remaining // 3600}h {(remaining % 3600) // 60:02d}m"

    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        return f"""{self.statement}

---

# Kapso operational requirements (your harness — follow these exactly)

You are the implementation agent of kapso, an autonomous experimentation
loop. Each iteration you design or refine ONE experiment, implement it, and
evaluate it on dev. A hard kill at the deadline must still leave a valid,
best-known submission on disk.

## Ground truth paths (always use absolute paths)
- Task directory (the ONLY area you may modify, plus subdirectories): {self.task_dir}
- Dataset & harness (READ-ONLY, never modify): {self.dataset_dir}
  (interactor.py, evaluate.py, animals_pool.txt, questions_pool.txt, dev.csv)
- Big files (precomputed tables, logs, checkpoints): {self.artifacts_dir} —
  never inside the git workspace; .gitignore *.npy, *.npz, *.pt, logs.
- Time remaining in the run as of writing this context: {self._remaining_str()}.

## The one non-negotiable deliverable
{self.submission_dir} must AT ALL TIMES contain your best solution so far:
- `solution.py` defining `class MySolution` with
  `__init__(self, animals_pool, questions_pool)` and `solve(self, interactor)`
  — exactly the notebook contract.
- Any data files it needs (e.g. the precomputed answer table), stored NEXT TO
  solution.py and loaded relative to `os.path.dirname(__file__)` — the file
  will be executed from a DIFFERENT working directory at final scoring.
- `__init__` must complete in under 5 minutes and must NOT recompute large
  tables: persist them once to the submission dir and load them.
Final scoring runs your submission on a HELD-OUT split with a pristine copy
of the harness: `python evaluate.py --csv <hidden.csv> --solution
solution.py:MySolution`. Verify after every improvement that this exact
invocation works against dev.csv from a scratch directory.
Update procedure when a candidate beats the current best dev score: write to
`submission.tmp/`, verify it loads and solves a few rows, then atomically
swap into {self.submission_dir}, and append `<dev_score> <iso-time>
<experiment-name>` to {self.task_dir}/best_score.log. A mediocre working
submission beats an empty directory — install your first working solution
immediately.

## Evaluation discipline
- Iterate with: `python {self.dataset_dir}/evaluate.py --csv {self.dataset_dir}/dev.csv \\
  --solution {self.submission_dir}/solution.py:MySolution` (or call
  `evaluate()` from your own driver for subsets).
- dev.csv has 150 rows; a full dev eval costs rows × queries oracle calls.
  Use a fixed ~30-row subset for quick iteration and the full 150 only to
  confirm an improvement. Report full-dev scores when you have them.
- The oracle model and your own copy share ONE GPU. The Interactor loads the
  LLM once per process — reuse processes; never run two evaluations
  concurrently; make sure a finished evaluation's process has exited before
  starting a big precompute.
- Batch your own model calls when precomputing (the statement's speed tip);
  measure throughput on a small slice first and size the precompute to fit
  your session (the full 1471×558 table is ~820k calls — budget it or shrink
  the question set).

## Rules you must never break
1. Inside `solve`, interact with the oracle ONLY via `ask` / `guess` /
   `is_done` / `remaining_budget`. Never read `interactor.gold`,
   `interactor.history`, its internals, or monkeypatch/subclass the
   Interactor. The final harness is pristine and your solution source is
   audited; violations zero the run.
2. Never modify anything under {self.dataset_dir}.
3. Do not special-case dev rows (e.g. memorizing dev's gold sequence) — the
   final split is disjoint; only strategies that deduce from oracle answers
   transfer.

## Session discipline
Hard caps enforced by a process-group kill: implementation sessions ≈
{self.session_caps['implementation_timeout'] // 60} min, ideation ≈
{self.session_caps['ideation_timeout'] // 60} min. Only files on disk survive.
- Start every implementation session by writing PLAN.md (session start +
  deadline, chosen approach, exact next command, status); keep it current.
- Any command expected to exceed 10 minutes: run it detached
  (`nohup ... > {self.artifacts_dir}/log.txt 2>&1 &`, record the PID), poll in
  bounded waits (≤5 min), and do useful work between polls.
- Kill processes by recorded PID only — never `pkill -f python` or group
  kills: your own session and its orchestrator run on this machine too.
- Persist partial precompute progress incrementally (append/save every few
  hundred rows) so a session kill never loses the work.

## Reporting (kapso convention)
At the end of every experiment report the measured DEV mean_score (0..1)
inside <score></score> tags AND write kapso_evaluation/result.json in your
workspace: {{"score": <float>, "notes": "<rows evaluated, solved_rate,
mean_queries>"}}. Never fabricate a score; a failed run is reported as such.

## Budget strategy
Budget progress: ~{budget_progress:.0f}%. Rough guide: first minutes —
baseline solution installed in submission/ (even the fixed-questions
reference beats an empty dir), measure oracle throughput; core — precompute
the answer table in batches while iterating on the adaptive
information-gain solver (they parallelize badly on one GPU: table first,
then solver work is CPU-only); last 15% — freeze, full-dev confirm, verify
the scratch-directory invocation. Use the whole budget; do not stop while
another improve+confirm cycle fits.
"""

    def stop_condition(self) -> bool:
        return False

    def final_evaluate(self, file_path: str, **kwargs):
        # Held-out scoring is done by the runner after the campaign.
        return {"submission_dir": self.submission_dir}
