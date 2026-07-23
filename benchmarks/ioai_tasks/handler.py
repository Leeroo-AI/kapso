"""Generic local-task problem handler (IOAI cross-year tasks).

A minimal handler for any local IOAI notebook task that scores a
`submission/solution.py` with a local evaluate command against a held-out
labeled split. Unlike the Kaggle handler there is no external submission —
ground truth is a local labeled file the runner scores after the campaign.

The point of these runs is not to win the task but to GENERATE a real kapso
execution trace (logs + artifacts + solution) that the harvest agents mine
for lessons transferable to a target task (Night Watch audio CIL).
"""

import os
import time

from kapso.environment.handlers.base import ProblemHandler


class LocalTaskHandler(ProblemHandler):
    """Handler for a local, self-scored IOAI task."""

    maximize_scoring = True

    def __init__(
        self,
        task_dir: str,
        statement: str,
        deadline_ts: float,
        session_caps: dict,
        eval_spec: dict,
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
        if not isinstance(eval_spec, dict) or not {
            "self_check_command",
            "metric_name",
        } <= eval_spec.keys():
            raise ValueError(
                "eval_spec must carry self_check_command + metric_name"
            )
        self.task_dir = os.path.abspath(task_dir)
        self.statement = statement.strip()
        self.deadline_ts = deadline_ts
        self.session_caps = session_caps
        self.eval_spec = eval_spec
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

# Kapso operational context

You are the implementation agent of kapso, an autonomous experimentation
loop: design ONE experiment per iteration, implement it, evaluate it.

- Task directory (yours to modify): {self.task_dir}
- Dataset (READ-ONLY, never modify): {self.dataset_dir}
- Big files, checkpoints, logs: {self.artifacts_dir}
- Time remaining: {self._remaining_str()}

Deliverable: {self.submission_dir}/solution.py implementing the class the
task statement specifies. It is scored on a HELD-OUT labeled split after the
campaign; a hard kill at the deadline must leave your best solution.py in
place. Self-check it any time from a scratch directory with:
    {self.eval_spec['self_check_command']}
Go for the highest {self.eval_spec['metric_name']} you can execute in the
time remaining; a trivial baseline that scores low earns nothing.

## Validation discipline (the recurring trap)
Measure honestly. If the task has any distribution shift between the data
you train on and the data you are scored on (class balance, source/session
structure, unseen conditions), a naive random split will over-report — carve
a split that reflects the real evaluation, and trust that, not raw training
accuracy. Report each experiment's measured {self.eval_spec['metric_name']}
in <score></score> tags AND write kapso_evaluation/result.json:
{{"score": <float>, "notes": "..."}}. Never fabricate a score; a failed run
is reported as such.
"""

    def stop_condition(self) -> bool:
        return False

    def final_evaluate(self, file_path: str, **kwargs):
        # Held-out scoring is done by the runner after the campaign.
        return {"submission_dir": self.submission_dir}
