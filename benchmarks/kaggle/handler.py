"""Kaggle competition problem handler (IOAI AI Models Track practice).

Hands the coding agent the task statement plus the invariant kapso contract:
paths, the best-public-score objective, and score reporting. Every
per-competition submission mechanic (kernel push vs. file upload, format,
compute limits, quota) lives in the statement itself, authored by the preflight
(benchmarks/kaggle/preflight_spec.md). best_score.log records PUBLIC leaderboard
scores only.
"""

import os
import time

from kapso.environment.handlers.base import ProblemHandler

class KaggleNotebookHandler(ProblemHandler):
    """Handler for Kaggle competitions; submission mechanics live in the statement."""

    maximize_scoring = True

    def __init__(
        self,
        task_dir: str,
        statement: str,
        deadline_ts: float,
        session_caps: dict,
        kaggle: dict,
        insured_reserve_seconds: float,
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
        if not isinstance(kaggle, dict) or not kaggle.get("competition"):
            raise ValueError("kaggle must carry the competition slug")
        self.task_dir = os.path.abspath(task_dir)
        self.statement = statement.strip()
        self.deadline_ts = deadline_ts
        self.session_caps = session_caps
        self.kaggle = kaggle
        self.insured_reserve_seconds = float(insured_reserve_seconds)
        self.dataset_dir = os.path.join(self.task_dir, "dataset")
        self.artifacts_dir = os.path.join(self.task_dir, "artifacts")
        self.submission_dir = os.path.join(self.task_dir, "submission")
        # The organizers' binding rules; the runner stages them into the task
        # dir. A run without them would let the agent build a kernel that
        # breaks a rule (two GPUs, an external checkpoint) and be voided.
        self.rules_path = os.path.join(self.task_dir, "RULES.md")
        if not os.path.isfile(self.rules_path):
            raise FileNotFoundError(
                f"{self.rules_path} missing — the runner stages "
                "benchmarks/kaggle/RULES.md there at launch"
            )
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def _remaining_str(self) -> str:
        remaining = max(0, int(self.deadline_ts - time.time()))
        return f"{remaining // 3600}h {(remaining % 3600) // 60:02d}m"

    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        competition = self.kaggle["competition"]
        return f"""{self.statement}

---

# Kapso operational context

You are the implementation agent of kapso, an autonomous experimentation
loop: design ONE experiment per iteration, implement it, evaluate it.

- Task directory (yours to modify): {self.task_dir}
- Dataset (READ-ONLY, never modify): {self.dataset_dir}
- Big files, checkpoints, logs: {self.artifacts_dir}
- Time remaining: {self._remaining_str()}
- Competition: `{competition}` — the kaggle CLI is installed and
  authenticated; develop and validate locally before spending a submission.

READ {self.rules_path} BEFORE WRITING ANY KERNEL, and obey it. It carries the
organizers' binding rules — single GPU (`cuda:0`), T4-or-CPU only, no internet,
no outside models or data, `.py` only — plus the fixed package list your kernel
must run against. A kernel that breaks one of those rules can be voided no
matter how well it scores, so treat the rules as a hard constraint on the design
space rather than something to check at the end.

What counts is the best PUBLIC leaderboard score among your submissions — a
stable baseline that scores low earns nothing. Go for the approach with the
highest expected final score you can execute in the time remaining. The
statement's Submission section is authoritative for how to build and submit a
scored entry; keep the exact artifact of your best submitted attempt under
{self.submission_dir}/.

The job is END-TO-END within this run's time budget: develop → submit → public
score, all before the deadline. Budget the submit-and-score round trip into
your plan from the start — an unsubmitted or unscored model counts for nothing,
and the last submission must leave enough margin for its run and scoring.

Only 2 GPU kernels run at once per account; the CPU pool is separate.

After a submission scores, append `<public_score> <iso-time> <label>` to
{self.task_dir}/best_score.log (public scores only). Report each experiment's
local validation score in <score></score> tags AND write
kapso_evaluation/result.json: {{"score": <float>, "notes": "..."}}. Never
fabricate a score; a failed run is reported as such.
"""

    def deliverable_ready_reserve_seconds(self):
        """Insured once a public score is banked in best_score.log.

        The full finalization reserve exists to cover one submission round trip
        (push -> kernel run -> submit -> score). Once a score is actually on the
        leaderboard that insurance is already paid, so the endgame only needs
        the residual and late iterations stay available. A missing log is the
        documented "nothing banked yet" case; a malformed line raises.
        """
        score_log = os.path.join(self.task_dir, "best_score.log")
        if not os.path.isfile(score_log):
            return None
        with open(score_log, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        for line in lines:
            score = float(line.split()[0])
            if score > 0:
                return self.insured_reserve_seconds
        return None

    def stop_condition(self) -> bool:
        return False

    def final_evaluate(self, file_path: str, **kwargs):
        # Ground truth is the Kaggle leaderboard; the runner polls it after
        # the campaign.
        return {"submission_dir": self.submission_dir}
