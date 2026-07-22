"""Kaggle code-competition problem handler (IOAI AI Models Track practice).

Hands the coding agent the task statement plus the minimal kapso contract:
paths, the scored-submission deliverable, the operator approval gate, and
score reporting. best_score.log holds PUBLIC leaderboard scores only —
banking one arms the insured (shrunk) finalization reserve.
"""

import os
import time

from kapso.environment.handlers.base import ProblemHandler

class KaggleNotebookHandler(ProblemHandler):
    """Handler for notebook-based (code) Kaggle competitions."""

    maximize_scoring = True

    def __init__(
        self,
        task_dir: str,
        statement: str,
        deadline_ts: float,
        session_caps: dict,
        contest_economics: dict,
        kaggle: dict,
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
        if not isinstance(contest_economics, dict) or (
            "insured_freeze_minutes" not in contest_economics
        ):
            raise ValueError(
                "contest_economics must carry insured_freeze_minutes"
            )
        if not isinstance(kaggle, dict) or not kaggle.get("competition"):
            raise ValueError("kaggle must carry the competition slug")
        self.task_dir = os.path.abspath(task_dir)
        self.statement = statement.strip()
        self.deadline_ts = deadline_ts
        self.session_caps = session_caps
        self.contest_economics = contest_economics
        self.kaggle = kaggle
        self.dataset_dir = os.path.join(self.task_dir, "dataset")
        self.artifacts_dir = os.path.join(self.task_dir, "artifacts")
        self.submission_dir = os.path.join(self.task_dir, "submission")
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
loop: design ONE experiment per iteration, implement it, evaluate it. A
hard kill at the deadline must still leave a submitted, scored entry.

- Task directory (yours to modify): {self.task_dir}
- Dataset (READ-ONLY, never modify): {self.dataset_dir}
- Big files, checkpoints, logs: {self.artifacts_dir}
- Time remaining: {self._remaining_str()}
- Competition: `{competition}` — the kaggle CLI is installed and
  authenticated; develop and validate locally before spending a kernel run.

Deliverable: at least one Kaggle submission RUN and SCORED on the public
leaderboard, and {self.submission_dir}/kernel/ holding the exact pushed
kernel (script + kernel-metadata.json) — self-contained: it trains from
the competition data (and the provided checkpoint, if any) inside the
kernel and writes submission.csv itself.

`kaggle competitions submit` requires operator approval: the CLI logs
your exact command to ~/kaggle_submit_requests.log and the approved
submission's full output appears in ~/kaggle_submit_executed.log — poll
it and continue local work while waiting; never retry or bypass the gate.

After a submission scores, append `<public_score> <iso-time> <label>` to
{self.task_dir}/best_score.log (public scores only). Report each
experiment's local validation score in <score></score> tags AND write
kapso_evaluation/result.json: {{"score": <float>, "notes": "..."}}. Never
fabricate a score; a failed run is reported as such.
"""

    def deliverable_ready_reserve_seconds(self):
        """Insured once a >0 public score is banked with a kernel on disk.

        best_score.log holds public leaderboard scores only (the handler
        prompt forbids local numbers there), so any >0 line plus the
        current kernel folder means the endgame needs only the freeze
        residual. A corrupt line raises — fail loud.
        """
        kernel_script = os.path.join(self.submission_dir, "kernel", "script.py")
        if not os.path.isfile(kernel_script):
            return None
        score_log = os.path.join(self.task_dir, "best_score.log")
        if not os.path.isfile(score_log):
            return None
        with open(score_log, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        for line in lines:
            if float(line.split()[0]) > 0:
                return self.contest_economics["insured_freeze_minutes"] * 60.0
        return None

    def stop_condition(self) -> bool:
        return False

    def final_evaluate(self, file_path: str, **kwargs):
        # Ground truth is the Kaggle leaderboard; the runner polls it after
        # the campaign.
        return {"submission_dir": self.submission_dir}
