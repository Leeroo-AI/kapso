"""Kaggle competition problem handler (IOAI AI Models Track practice).

Hands the coding agent the task statement plus the invariant kapso contract:
paths, the best-public-score objective, the up-to-three submit-and-learn
rounds each lane runs inside its session, and score reporting. Every
per-competition submission mechanic (kernel push vs. file upload, format,
compute limits, quota) lives in the statement itself, authored by the preflight
(benchmarks/ioai2026/preflight_spec.md). Lanes learn from each other through
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
    # A competition has no "goal achieved": every remaining minute can buy
    # leaderboard position, so agent stop votes are advisory and the campaign
    # runs to its time budget.
    honor_agent_stop = False

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
                "benchmarks/ioai2026/RULES.md there at launch"
            )
        # The kaggle CLI playbook, staged by the runner from the repo's
        # kaggle-cli-submission skill. The statement deliberately authors no
        # CLI mechanics and codex/claude session clones cannot load the skill
        # natively, so this path is how every coding-agent CLI receives it.
        self.skill_path = os.path.join(self.task_dir, "KAGGLE_CLI.md")
        if not os.path.isfile(self.skill_path):
            raise FileNotFoundError(
                f"{self.skill_path} missing — the runner stages the "
                "kaggle-cli-submission skill there at launch"
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

# Shared knowledge bank — first search priority

`{self.task_dir}/knowledge_bank/` is a curated book of IOAI-style problems, each
carrying its best-known solution: open `INDEX.md` first — its router
maps a task's TWIST to a section, and every line reads
`problem → winning solution (score)` with tier marks (`✓` proof-grade,
`~` author-claimed max, `●` under review). Each problem dir holds
`problem.md` (the task), `idea.md` (the winning idea, provenance,
verification), and `solution/` (runnable code).

This applies to EVERY module — ideation, selection, implementation,
feedback: when you search for ideas, methods, or code, the bank is the
FIRST priority; the open web is second: Kaggle competitions on the same
problem family (winning solutions, top public notebooks, discussion
forums) and GitHub are the richest sources. Route via `INDEX.md`, read the
matched problems' idea.md and solution/, then take your web search
wherever you judge best.

# Kapso operational context

You are one implementation lane of kapso. Your session is a LOOP of up to
TEN submit-and-learn rounds:
1. Implement an idea, validate locally, and — after writing down the
   public score you PREDICT — submit once it clears the ROI bar. The
   <solution> you received is round 1's starting point, not a cage: diverge
   from it, or from your own lineage, whenever evidence points elsewhere.
2. WAIT for the public score and bank it. Score minus prediction is your
   calibration gap — local validation runs optimistic — and it prices every
   later ROI estimate. Then study the evidence: the gap, every sibling
   submission, the account's WHOLE history on this competition (previous
   campaigns' submissions are fair inspiration — pull any kernel behind a
   score), and — you have web search — how similar problems were pushed
   further.
3. Propose the best successor that evidence supports, and go again. A
   failed or under-target round is material: learn from it and keep working
   it; when it stops paying, switch to fresh ideas from the web or from any
   previous run's submissions.
Your session ends ONLY when the remaining time cannot fit a full
submit-and-score trip — never because a round succeeded, failed, or looks
hard to beat. Banking a score starts the next round; a failed build starts
the next round (fix it or switch angles — you have the GPU and the hours);
"my ideas can't beat the board" means find the idea that CAN, using the gap
evidence, the siblings' code, and the web. Write your final report only when
that time gate closes. Ten submissions is your per-lane ceiling, but
the competition budget is SHARED by every lane and binds first — past your
fifth spend, check `kaggle competitions submissions` before each one. Working until the gate
closes is the contract. Exiting with an idle clock — worst of all with zero
submissions — is the one unacceptable outcome.

- Task directory (yours to modify): {self.task_dir}
- Dataset (READ-ONLY, never modify): {self.dataset_dir}
- Big files, checkpoints, logs: {self.artifacts_dir}
- Time remaining: {self._remaining_str()}
- Competition: `{competition}` — the kaggle CLI is installed and authenticated.

READ {self.rules_path} BEFORE YOU BUILD ANYTHING, and obey it — the organizers'
binding rules and the fixed package list your code runs against. Breaking one
voids the submission however well it scores, so treat them as constraints on
the design space, not an end-of-run checklist.

What counts is the best PUBLIC leaderboard score among your submissions, so
a submission's ROI is its chance of BEATING the board's best times the size
of the beat. Banking a stable baseline or re-proving a sibling's result is
worth ZERO — every round, submit the candidate with the
highest expected final score you can execute in the time remaining, even
when a tamer option is likelier to merely succeed. The
statement's Submission section is authoritative on the modality and the exact
file required; {self.skill_path} is the CLI playbook — push, poll,
submit, read the score. Keep your best submission under {self.submission_dir}/.
The public score is your feedback signal, not the prize: final ranking is
decided on the PRIVATE leaderboard — an unseen split scored only at the end.
Prefer moves that improve the underlying model over moves that fit
public-split quirks, and treat a public gain your own validation cannot
explain with suspicion rather than celebration: it may not transfer.

The job is END-TO-END: develop → submit → public score, all before the
deadline. Budget the round trip from the start — an unsubmitted or unscored
model counts for nothing, and the last submission needs margin for its run and
scoring. When one scores, append `<public_score> <iso-time> <kernel-ref>
<one-line idea>` (ref `-` on a file upload) to {self.task_dir}/best_score.log —
the banked-score record (public scores only).

Every lane submits through one account, so Kaggle holds the whole team's
history. Before proposing EACH new experiment, read the board —
`kaggle competitions submissions {competition}` (scores + idea messages) —
and learn from it, in this order:

1. SKIP: a sibling already scored your idea? Don't rerun it — change what
   matters or move on.
2. STEAL: pull the code behind results stronger than yours
   (`kaggle kernels list -m`, then `kaggle kernels pull <ref> -m -p <dir>`)
   and integrate the components that beat you — a proven feature, schedule,
   or head is free progress.
3. COMPLEMENT: when your plan lands within noise of a sibling's, switch to
   the variant most DIFFERENT from the board's leaders (other model family,
   other feature view, other objective) — the team needs diverse strong
   models, not a third copy of the leader.
4. ENSEMBLE: once the board holds two or more diverse strong models and
   single-model gains go sub-point, blending their probabilities (yours,
   or rebuilt from siblings' pulled code) is usually the cheapest points
   available — build the blend kernel rather than another solo tweak.

THE BUDGET IS TEAM-SHARED. The task's stated spend limit (scored submissions
or notebook versions — whichever the statement caps) covers ALL lanes
together, and the board's row count is the live spend counter: count it
before EVERY spend. Once more than 85% of the budget is gone, PRIORITIZE
spends your calibration says can beat the CURRENT BEST SUBMITTED score —
map your local score to a public estimate using the local-vs-public gaps
already banked on the board. This is a priority, NOT a stop sign: budget
left unspent at the deadline is thrown away, so keep submitting your best
available candidates to the end — near the deadline a plausible improvement
is always worth scoring, and a diverse strong candidate teaches the team
even when it might not take the lead. What to avoid is pure waste:
re-measuring a variant the board has already answered. To keep calibration
computable for everyone, every -m message must include your local score
(local=<x>) next to the idea.

Then propose the experiment, and submit it with a `-m` message naming the
idea: the board is the team's memory and your message is your entry in it.

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
