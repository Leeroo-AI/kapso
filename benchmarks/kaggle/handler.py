"""Kaggle competition problem handler (IOAI AI Models Track practice).

Hands the coding agent the task statement plus the invariant kapso contract:
paths, the best-public-score objective, and score reporting. Every
per-competition submission mechanic (kernel push vs. file upload, format,
compute limits, quota) lives in the statement itself, authored by the preflight
(benchmarks/kaggle/preflight_spec.md). Lanes learn from each other through
Kaggle itself — the account holds every submission's score, message and code.
best_score.log stays as the run's own record of PUBLIC scores actually banked;
the finalization reserve is released against it.
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
        # dir. A run without them would let the agent build a solution that
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

You are the implementation agent of kapso: design ONE experiment per
iteration, implement it, evaluate it.

- Task directory (yours to modify): {self.task_dir}
- Dataset (READ-ONLY, never modify): {self.dataset_dir}
- Big files, checkpoints, logs: {self.artifacts_dir}
- Time remaining: {self._remaining_str()}
- Competition: `{competition}` — the kaggle CLI is installed and authenticated.

READ {self.rules_path} BEFORE YOU BUILD ANYTHING, and obey it — the organizers'
binding rules and the fixed package list your code runs against. Breaking one
voids the submission however well it scores, so treat them as constraints on
the design space, not an end-of-run checklist.

What counts is the best PUBLIC leaderboard score among your submissions — a
stable baseline that scores low earns nothing. Go for the approach with the
highest expected final score you can execute in the time remaining. The
statement's Submission section is authoritative on the modality and the exact
file required; work out the CLI commands yourself. Keep your best submission
under {self.submission_dir}/.

The job is END-TO-END: develop → submit → public score, all before the
deadline. Budget the round trip from the start — an unsubmitted or unscored
model counts for nothing, and the last submission needs margin for its run and
scoring. When one scores, append `<public_score> <iso-time> <one-line idea>` to
{self.task_dir}/best_score.log — the run's record of what is actually banked
(public scores only, never a local validation number).

Every lane submits through one account, so Kaggle holds the whole team's
history — read it before committing to an idea, and don't redo what a sibling
already scored. `kaggle competitions submissions {competition}` gives each
attempt's public score and message; where kernels were pushed, `kaggle kernels
list -m` then `kaggle kernels pull <ref> -m -p <dir>` gets the code behind one.
Always submit with a `-m` message naming your idea: it is what the others read.

Report each experiment's local validation score in <score></score> tags AND
write kapso_evaluation/result.json: {{"score": <float>, "notes": "..."}} — name
the validation split in `notes`, since scores measured different ways do not
compare. Never fabricate a score; a failed run is reported as such.
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
