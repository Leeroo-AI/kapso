"""
PostTrainBench Problem Handler

Wraps a PostTrainBench task directory (evaluate.py, timer.sh, templates/) as a
kapso ProblemHandler. The handler's job is to hand the coding agent the
official task prompt plus the operational discipline a 10-hour, single-H100,
hard-deadline run demands: absolute-path conventions, GPU exclusivity,
chat-template fidelity, best-so-far final_model maintenance, and the benchmark
rules it must never break.

Scoring is self-reported: the agent runs {task_dir}/evaluate.py itself and
reports the score via <score> tags / kapso_evaluation/result.json, which the
generic search strategy's FeedbackGenerator validates.
"""

import os
import time

from kapso.environment.handlers.base import ProblemHandler

# Per-benchmark iteration-time eval sample limits. Full test sets are used by
# the organizers' final scoring; during the run smaller --limit values keep the
# train/eval cycle short. -1 means the set is small enough to always run fully.
ITERATION_EVAL_LIMITS = {
    "aime2025": -1,  # 30 problems
    "gsm8k": 150,  # of 1319
    "gpqamain": 150,  # of 448
    "humaneval": -1,  # 164 problems
    "bfcl": 200,
    "arenahardwriting": 50,  # GPT-judge scored: each eval costs OpenAI credits
    "healthbench": 50,  # GPT-judge scored: each eval costs OpenAI credits
}

# Condensed findings from the PostTrainBench paper (arXiv:2603.08640) about
# what worked for prior agents. This is scaffold-level guidance (fair game,
# like any agent's system prompt) — it references no benchmark test data.
PRIOR_RUN_INSIGHTS = """
- SFT on well-chosen open datasets is the dominant winning method (TRL
  SFTTrainer). LoRA trains fast; full fine-tuning can win when time allows.
  Distilling reasoning traces from stronger open models' published datasets
  works well for math/code.
- The single biggest lever is matching the EVALUATION chat template exactly.
  The eval applies a fixed jinja template (see templates/) with vLLM; training
  data must be rendered with that exact template or scores collapse.
- Largest headroom historically: function calling (BFCL) and GSM8K. GPQA runs
  often score below the 25% random baseline purely due to answer-format
  problems — verify the model emits parseable answers early. AIME is the
  hardest; do not sink the whole budget there without early evidence of gains.
- Weak runs died from: leaving no time to save the final model, breaking the
  model with an aggressive late experiment and submitting it, format drift
  between training and eval, and idling after 2-3 hours instead of using the
  full budget.
"""


class PostTrainBenchHandler(ProblemHandler):
    """Problem handler for a single PostTrainBench model+benchmark task."""

    maximize_scoring = True

    def __init__(
        self,
        task_dir: str,
        official_prompt: str,
        model_id: str = "",
        benchmark_name: str = "",
        benchmark_id: str = "",
        deadline_ts: float | None = None,
        num_gpus: int = 1,
        session_caps: dict | None = None,
    ):
        super().__init__(additional_context="")
        # The agent can only manage a deadline it has been told about (run #7,
        # finding F5): these are the shaped per-session timeouts the runner
        # computes, rendered verbatim into the agent's context.
        if (
            not isinstance(session_caps, dict)
            or not {
                "implementation_timeout",
            }
            <= session_caps.keys()
        ):
            raise ValueError(
                "session_caps must be the runner's shaped session timeouts "
                "(implementation_timeout, seconds)"
            )
        self.task_dir = os.path.abspath(task_dir)
        self.official_prompt = official_prompt.strip()
        self.model_id = model_id
        self.benchmark_name = benchmark_name
        self.benchmark_id = benchmark_id
        self.deadline_ts = deadline_ts
        self.num_gpus = num_gpus
        self.session_caps = session_caps
        self.artifacts_dir = os.path.join(self.task_dir, "artifacts")
        self.final_model_dir = os.path.join(self.task_dir, "final_model")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _remaining_str(self) -> str:
        if self.deadline_ts is None:
            return "unknown — run `bash timer.sh` in the task directory"
        remaining = max(0, int(self.deadline_ts - time.time()))
        return f"{remaining // 3600}h {(remaining % 3600) // 60:02d}m"

    def _eval_limit_hint(self) -> str:
        limit = ITERATION_EVAL_LIMITS.get(self.benchmark_id)
        if limit is None:
            return "pick a --limit that keeps one eval under ~15 minutes"
        if limit == -1:
            return "this benchmark is small; use --limit -1 (full set) every time"
        return f"use --limit {limit} for iteration evals (full set only if time clearly allows)"

    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        evaluate_py = os.path.join(self.task_dir, "evaluate.py")
        templates_dir = os.path.join(self.task_dir, "templates")
        timer_sh = os.path.join(self.task_dir, "timer.sh")

        return f"""{self.official_prompt}

---

# Kapso operational requirements (your harness — follow these exactly)

You are the implementation agent of kapso, an autonomous experimentation loop.
Each iteration you design or refine ONE experiment, implement it, train, and
evaluate. Everything below exists so that a hard kill at the deadline still
leaves a valid, best-known submission on disk.

## Ground truth paths (always use absolute paths)
- Task directory (the ONLY area you may modify, plus subdirectories): {self.task_dir}
- Evaluation script (READ-ONLY, never modify): {evaluate_py}
- Chat templates used by the evaluator (READ-ONLY, never modify): {templates_dir}
- Time left: run `bash {timer_sh}` — at the START of every session, and before
  starting anything expected to take more than 30 minutes.
  Time remaining as of writing this context: {self._remaining_str()}.
- Store training pipelines/code in your current git workspace (it lives under
  the task directory). Store ALL model weights, checkpoints and large files
  under {self.artifacts_dir}/<experiment-name>/ — NEVER inside the git
  workspace. Before your first commit, add a .gitignore covering
  *.safetensors, *.bin, *.pt, checkpoints/, and dataset dumps.

## The one non-negotiable deliverable
{self.final_model_dir} must AT ALL TIMES contain the best model found so far,
loadable with transformers/vLLM from the unmodified starting environment.
Update procedure whenever a candidate beats the current best score:
1. Save/copy the candidate to {self.task_dir}/final_model.tmp
2. Verify it loads (config.json + tokenizer files + weights present)
3. Replace atomically: rm -rf final_model && mv final_model.tmp final_model
4. Append a line to {self.task_dir}/best_score.log: `<score> <iso-time> <experiment-name>`
Never leave final_model empty, partial, or pointing at an unverified model.
As soon as your FIRST fine-tuned model evaluates at or above the base model,
put it there — a mediocre model beats an empty directory.

## Evaluation discipline
- Evaluate candidates with:
  `python {evaluate_py} --model-path <ABS_PATH_TO_CANDIDATE> --templates-dir {templates_dir} --limit <N>`
  For this benchmark: {self._eval_limit_hint()}.
- First establish the BASE model's score the same way (small --limit is fine)
  so you know what "better" means.
- The GPU is shared between training and evaluation: make sure training has
  fully exited (no lingering python processes holding GPU memory) before
  launching an eval, and vice versa. If vLLM startup fails with memory errors,
  kill stray processes and lower --max-connections or --max-tokens.
- The evaluator renders conversations through the template in {templates_dir}
  appropriate to the model family. READ that template before building training
  data, and render your SFT examples with the exact same template.

## Session discipline & long-running processes
You operate in bounded SESSIONS inside the overall run. Hard caps, enforced
by a process-group kill that takes down EVERY process you started (including
training): implementation sessions ≈ {self.session_caps['implementation_timeout'] // 60}
minutes.
Only files on disk survive a session kill. `timer.sh` shows the RUN clock;
your SESSION clock is shorter and is the one that kills you.
- FIRST actions of every implementation session: run `date -u`, then create
  or refresh PLAN.md at the workspace root with (a) session start time and
  your session deadline (start + the cap above), (b) the run-level time from
  `bash timer.sh`, (c) chosen approach and dataset, (d) the exact next
  command, (e) current status. Update it whenever the plan changes — a
  killed session must leave a consumable plan behind.
- NEVER run a command expected to exceed 10 minutes in the foreground.
  Launch it detached (plain nohup — never setsid) and log to a file:
      nohup python train.py > {self.artifacts_dir}/train_log.txt 2>&1 &
      echo $! > {self.artifacts_dir}/train.pid
  then poll in BOUNDED waits, each ≤5 minutes, e.g.
  `sleep 240 && tail -5 {self.artifacts_dir}/train_log.txt && bash timer.sh`,
  comparing progress against the session deadline in PLAN.md. Do useful work
  between polls (prepare the eval command, update PLAN.md). Note: if you do
  block on a long foreground call, the CLI converts it to a background task
  after ~2 minutes and hands you its output-file path — poll that file; but
  prefer explicit nohup so the pattern is under your control.
- SIZE BEFORE YOU COMMIT: before any training run projected to exceed
  15 minutes, measure throughput first (≤50 steps or one logging interval),
  compute total_steps × seconds_per_step, and choose max_steps / dataset
  size so the projected duration fits within 60% of YOUR SESSION's
  remaining time (the other 40% covers merge + eval + promotion + one
  corrective action). Write the arithmetic into PLAN.md.
- At every poll apply the rule: if projected completion exceeds your
  session's remaining time minus 20 minutes, kill training, promote the
  best checkpoint to final_model, and evaluate it.
- Checkpoint at least every ~15 minutes of training (size save_steps
  accordingly) so a kill never loses more than one interval, and commit
  code early and often so nothing important lives only in your session
  context.
- KILL DISCIPLINE: terminate processes by specific PID only
  (`kill $(cat {self.artifacts_dir}/train.pid)`). NEVER use pattern kills
  (`pkill -f python`, `pkill -f vllm`) or group kills (`kill 0`,
  `kill -- -PID`): this container also runs YOUR OWN session and its
  orchestrator, and a pattern/group kill will terminate you mid-work
  (this has destroyed whole sessions before). To free the GPU, kill the
  recorded PIDs, then verify with `nvidia-smi`.

## Rules you must never break (violations get judged and zero the run)
1. Do not train on {self.benchmark_name or 'the target benchmark'} test data —
   no questions, no answers, no paraphrases of specific test items. This
   INCLUDES reading per-sample eval-log questions or gold targets to design
   training data: error analysis must stay at the class/count level
   (categories, pass/fail tallies), using your own held-out splits of
   permitted data. Eval logs may tell you WHICH classes fail, never WHAT
   the answers look like.
2. Do not modify {evaluate_py} or {templates_dir}.
3. Only fine-tune `{self.model_id or 'the assigned base model'}`; never
   download or submit an instruction-tuned variant or any other model.
4. Do not call the OpenAI API for anything except what evaluate.py itself does
   internally. Your own LLM calls run on Anthropic via kapso; keep it that way.
5. Work only inside {self.task_dir} (the HuggingFace cache in the home
   directory is fine to use).
6. final_model must run with the starting environment's packages, even if you
   installed extra packages for training.

## Insights from prior runs of this benchmark
{PRIOR_RUN_INSIGHTS}

## Reporting (kapso convention)
At the end of every experiment, report the measured benchmark score (0-100) of
this iteration's candidate inside <score></score> tags AND write
kapso_evaluation/result.json in your workspace: {{"score": <float>, "notes": "..."}}.
If the run failed, report the failure honestly — never fabricate a score.

## Budget strategy
Budget progress is ~{budget_progress:.0f}%. Rough guide: first ~10% research +
baseline eval + template study; until ~75% training iterations (start with the
highest-expected-value recipe, iterate on data quality/mix, LR, epochs); by
~85% stop risky new ideas and consolidate the best candidate (optionally a
final full-set eval); final stretch: verify final_model integrity and loading.
Use the WHOLE time budget — do not stop early while measurable headroom
remains and time allows another train+eval cycle.
"""

    def stop_condition(self) -> bool:
        # Never stop early: remaining budget is always better spent iterating.
        return False

    def final_evaluate(self, file_path: str, **kwargs):
        # Official scoring is done by the PostTrainBench harness (full-set
        # evaluate.py against task_dir/final_model) after the agent exits.
        return {"final_model": self.final_model_dir}
