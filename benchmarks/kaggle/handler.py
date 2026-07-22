"""Kaggle code-competition problem handler (IOAI AI Models Track practice).

Hands the coding agent the task statement plus the operational discipline a
short, single-GPU, hard-deadline run with a METERED ground-truth channel
demands: local validation is free, but the only real score is a Kaggle
submission, capped per day by the competition rules and per run by our own
budget. best_score.log holds PUBLIC leaderboard scores only — banking one is
what arms the insured (shrunk) finalization reserve.
"""

import os
import time

from kapso.environment.handlers.base import ProblemHandler

KAGGLE_ECONOMICS_KEYS = {"daily_submission_cap", "run_submission_cap"}


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
        if not isinstance(contest_economics, dict) or not {
            "insurance_minutes",
            "confirm_gain_ratio",
            "insured_freeze_minutes",
        } <= contest_economics.keys():
            raise ValueError(
                "contest_economics must carry the config's reward-policy "
                "knobs (insurance_minutes/confirm_gain_ratio/"
                "insured_freeze_minutes)"
            )
        if not isinstance(kaggle, dict) or not (
            KAGGLE_ECONOMICS_KEYS <= kaggle.keys() and kaggle.get("competition")
        ):
            raise ValueError(
                "kaggle must carry the competition slug plus the config's "
                "submission caps (competition/daily_submission_cap/"
                "run_submission_cap)"
            )
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

# Kapso operational requirements (your harness — follow these exactly)

You are the implementation agent of kapso, an autonomous experimentation
loop. Each iteration you design or refine ONE experiment, implement it, and
evaluate it. A hard kill at the deadline must still leave a submitted,
scored Kaggle entry and a reproducible kernel on disk.

## Ground truth paths (always use absolute paths)
- Task directory (the ONLY area you may modify, plus subdirectories): {self.task_dir}
- Dataset & provided checkpoint (READ-ONLY, never modify): {self.dataset_dir}
  (audio/, train.csv, fine_tune.csv, submission.csv, model/)
- Big files (features, checkpoints, logs): {self.artifacts_dir} — never
  inside the git workspace; .gitignore *.npy, *.npz, *.pt, *.safetensors, logs.
- Time remaining in the run as of writing this context: {self._remaining_str()}.

## The one non-negotiable deliverable
1. At least ONE Kaggle submission that has RUN and SCORED on the public
   leaderboard, and
2. {self.submission_dir}/kernel/ holding the CURRENT best kernel exactly as
   pushed: `script.py` + `kernel-metadata.json` (kernel_type "script",
   enable_internet "false", competition_sources ["{competition}"], GPU
   enabled). The kernel must TRAIN from the provided checkpoint + competition
   data (≤ ~10 GPU-minutes) and then write `/kaggle/working/submission.csv`
   itself — self-contained and reproducible; never depend on locally-trained
   weights the kernel cannot rebuild.

## Kaggle protocol (the CLI is installed and authenticated)
- SUBMISSION APPROVAL GATE: every `kaggle competitions submit` requires
  HUMAN OPERATOR APPROVAL. The CLI blocks the submit, logs your exact
  command to ~/kaggle_submit_requests.log, and notifies the operator; the
  approved submission is executed by the operator and its full output
  appears in ~/kaggle_submit_executed.log — poll that file. When you
  request, include a one-line rationale (local val score + expected gain)
  via the -m message. Do not retry blocked submits or attempt to bypass
  the gate; continue local work while waiting.
- Submission flow, exactly this sequence (competition: `{competition}`):
  `kaggle kernels push -p {self.submission_dir}/kernel/` — push TWICE on
  first creation (the first run often starts before the data mounts) and
  record the printed version number N;
  poll `kaggle kernels status <user>/<slug>` every ~30 s until COMPLETE
  (ERROR → `kaggle kernels output <user>/<slug> -p out/` and read the log);
  `kaggle competitions submit {competition} -k <user>/<slug> -v N -f
  submission.csv -m "<label>"`;
  poll `kaggle competitions submissions {competition} --format json` until
  your entry's publicScore is populated.
- Develop `script.py` LOCALLY first (this box has the same data layout) and
  validate it end-to-end before any push: locate inputs by filename search
  (on Kaggle the data mounts under /kaggle/input/..., locally use
  {self.dataset_dir}) and write cheap assertions on the output (row count
  363, columns path,target, same row order as the reference file).
- One submission in flight at a time; poll status before pushing again.
- After a publicScore lands, bank it (append-only, under flock):
  `( flock 9; echo "<public_score> <iso-time> kernel=<user>/<slug>:vN <label>" \\
    >> {self.task_dir}/best_score.log ) 9>>{self.task_dir}/best_score.log`
  best_score.log holds PUBLIC scores only — local validation numbers never
  go there.

## SUBMISSION BUDGET (the scarce resource — treat it like GPU-hours)
- Account-wide daily cap: {self.kaggle['daily_submission_cap']} submissions.
  This run's own cap: {self.kaggle['run_submission_cap']} — NEVER exceed it.
- A submission is the ONLY ground-truth read and costs ~20 minutes round
  trip (push + kernel run + scoring). Spend them deliberately:
  one early insurance submission, mid-run submissions only when local
  validation projects a clear improvement, and the final best-recipe
  submission with enough margin before the deadline.

## Evaluation discipline (local, free tier)
- Carve a stratified local validation split from the 920 labeled rows
  (e.g. ~20% per class of train.csv + fine_tune.csv), train on the rest,
  and measure the metric exactly: 0.5*Acc_old + 0.5*Acc_new. This is your
  iteration signal; report it in <score> tags.
- The 363 submission.csv rows are UNLABELED — they cannot leak; do not
  waste time trying to exploit them.
- Local GPU and training budget mirror the task rule: single GPU, ~10 min
  per full training run — profile a short run first, then size epochs to
  fit. Never run two GPU trainings concurrently.
- Match the kernel and local environments: same preprocessing
  (dataset/model/preprocessor_config.json), same seed policy, so the
  kernel's re-training reproduces your local result.

## Rules you must never break
1. Only the provided checkpoint ({self.dataset_dir}/model) and the
   competition data may enter the solution — no other pretrained
   weights/embeddings/models, no external data, no external AI APIs inside
   the kernel. Libraries that auto-download pretrained resources are banned.
2. Never modify anything under {self.dataset_dir}.
3. The submitted kernel runs with Internet OFF — it must not download
   anything; everything it needs comes from the competition mount and the
   code itself.
4. The submitted kernel source is audited after the run; violations zero
   the result.

## Session discipline
Session caps: implementation sessions ≈
{self.session_caps['implementation_timeout'] // 60} min, ideation ≈
{self.session_caps['ideation_timeout'] // 60} min. The generic
session-runtime rules (detached long jobs, watcher/alarm/notification
discipline, kill-by-PID, no orphaned value, incremental persistence) are in
your core instructions. Benchmark-specific:
- Start every implementation session by writing PLAN.md (session start +
  deadline, chosen approach, exact next command, status); keep it current.
- Long jobs (training, feature precompute, kernel polling loops) log and
  record PIDs under {self.artifacts_dir}.

## Reporting (kapso convention)
At the end of every experiment report the measured LOCAL validation score
(0.5*Acc_old + 0.5*Acc_new, 0..1) inside <score></score> tags AND write
kapso_evaluation/result.json in your workspace: {{"score": <float>,
"notes": "<val split sizes, Acc_old, Acc_new, any Kaggle publicScore>"}}.
Never fabricate a score; a failed run is reported as such.

## Reward & time economics (contest mode)
Budget progress: ~{budget_progress:.0f}%.
- REWARD: you are rewarded ONLY for the best PUBLIC leaderboard score among
  this run's submissions. Local tidiness and unverified progress earn
  NOTHING. A failed ambitious attempt (with insurance banked) costs the
  same as never trying — attempt the strongest design you can execute.
- INSURANCE: get ONE valid submission SCORED on the leaderboard within your
  first ~{self.contest_economics['insurance_minutes']} minutes (a simple
  head-extension baseline is fine). That is the only permitted safety
  spend.
- CONFIRMATIONS are expensive here in a specific way: a Kaggle submission
  costs ~20 min AND one of your {self.kaggle['run_submission_cap']}. Submit
  mid-run ONLY when local validation projects a gain over
  {self.contest_economics['confirm_gain_ratio']:g}× the round-trip cost;
  otherwise keep iterating on the free local tier.
- FREEZE: reserve the final ~15% to freeze: confirm your best recipe's
  kernel is pushed, submitted, and SCORED, and {self.submission_dir}/kernel
  matches that exact version. The freeze submission is mandatory and sits
  OUTSIDE these economics — never skip it, and never leave your best local
  recipe unsubmitted.
- Once a >0 public score is banked in best_score.log, the campaign
  automatically shrinks its endgame reserve to
  ~{self.contest_economics['insured_freeze_minutes']} minutes and lowers
  the iteration-admission floor — late, short, bold iterations stay
  available. Banking the insurance early literally buys search time.
- NOT negotiable for speed: the competition rules, the run submission cap,
  and the freeze submission. Boldness applies to allocation, never to
  measurement.
Use the whole budget; do not stop while another improve+submit cycle fits.
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
