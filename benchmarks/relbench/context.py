"""Problem-context builder for RelBench tasks.

Assembles the static problem description handed to the ideation LLM and the
coding agent: schema, task definition (including the exact label-generating
SQL), split protocol, prediction contract, hard leakage rules, and resource
constraints.

The context is built once per task (the orchestrator reads it a single time),
so anything dynamic (budget progress, best-so-far) is surfaced through
handler.run() outputs instead.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import List

import numpy as np
import pandas as pd

from benchmarks.relbench.task_specs import (
    AUTOCOMPLETE_BINARY,
    AUTOCOMPLETE_MULTICLASS,
    AUTOCOMPLETE_REGRESSION,
    DATASET_NOTES,
    ENTITY_BINARY,
    ENTITY_MULTICLASS,
    ENTITY_REGRESSION,
    RECOMMENDATION,
    TaskSpec,
)

MAX_SCHEMA_COLS = 40


# ---------------------------------------------------------------------------
# Schema description
# ---------------------------------------------------------------------------

def describe_database(db, dataset) -> str:
    lines: List[str] = []
    lines.append(
        f"Database time range: {db.min_timestamp} .. {db.max_timestamp} | "
        f"val cutoff V = {dataset.val_timestamp} | test cutoff T = {dataset.test_timestamp}"
    )
    lines.append("")
    for name, table in sorted(db.table_dict.items()):
        df = table.df
        fkeys = ", ".join(f"{c}->{t}" for c, t in table.fkey_col_to_pkey_table.items()) or "none"
        lines.append(
            f"### table `{name}` — {len(df):,} rows | pkey: {table.pkey_col} | "
            f"time_col: {table.time_col} | fkeys: {fkeys}"
        )
        col_bits = []
        for col in df.columns[:MAX_SCHEMA_COLS]:
            dtype = str(df[col].dtype)
            null_pct = float(df[col].isna().mean()) * 100
            bit = f"{col}:{dtype}"
            if null_pct >= 1:
                bit += f"({null_pct:.0f}% null)"
            col_bits.append(bit)
        if len(df.columns) > MAX_SCHEMA_COLS:
            col_bits.append(f"... +{len(df.columns) - MAX_SCHEMA_COLS} more")
        lines.append("  columns: " + ", ".join(col_bits))
    return "\n".join(lines)


def _label_stats(spec: TaskSpec, train_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
    lines = []
    for split, df in (("train", train_df), ("val", val_df)):
        if spec.is_recommendation:
            sizes = df[spec.dst_entity_col].map(len)
            lines.append(
                f"- {split}: {len(df):,} rows | ground-truth list size "
                f"mean={sizes.mean():.2f} median={sizes.median():.0f} max={sizes.max()}"
            )
        elif spec.family in (ENTITY_BINARY, AUTOCOMPLETE_BINARY):
            pos = float((df[spec.target_col] == 1).mean())
            lines.append(f"- {split}: {len(df):,} rows | positive rate = {pos:.4f}")
        elif spec.is_multiclass:
            top = df[spec.target_col].value_counts(normalize=True).head(5)
            top_str = ", ".join(f"{int(k)}:{v:.2%}" for k, v in top.items())
            lines.append(
                f"- {split}: {len(df):,} rows | {spec.num_classes} classes | top classes: {top_str}"
            )
        else:
            t = df[spec.target_col]
            lines.append(
                f"- {split}: {len(df):,} rows | target min={t.min():.3g} "
                f"q25={t.quantile(0.25):.3g} median={t.median():.3g} "
                f"q75={t.quantile(0.75):.3g} max={t.max():.3g} mean={t.mean():.3g}"
            )
    return "\n".join(lines)


def _task_definition(task, spec: TaskSpec) -> str:
    doc = inspect.getdoc(type(task)) or ""
    lines = [f"Task `{spec.dataset_name}/{spec.task_name}` — family: {spec.family}."]
    if doc:
        lines.append(f"Official description: {doc}")
    if spec.is_recommendation:
        lines.append(
            f"For each seed row (source `{spec.src_entity_col}` from table "
            f"`{spec.src_entity_table}`, seed time in `{spec.time_col}`), predict the "
            f"ranked top-{spec.eval_k} destination `{spec.dst_entity_col}` ids from table "
            f"`{spec.dst_entity_table}` ({spec.num_dst_nodes:,} candidate ids, integer "
            f"indices 0..{spec.num_dst_nodes - 1}) that the source will interact with in "
            f"(t, t + {spec.timedelta_str}]."
        )
    elif spec.is_autocomplete:
        lines.append(
            f"A row of table `{spec.entity_table}` identified by `{spec.entity_col}` at "
            f"time `{spec.time_col}` is being entered; predict its `{spec.target_col}` "
            "column from the relational context. The target column has been removed "
            "from the database you can access."
        )
        if spec.removed_columns:
            lines.append(
                "Also removed for leakage prevention (do not attempt to reconstruct): "
                + ", ".join(spec.removed_columns)
            )
    else:
        lines.append(
            f"For each seed row (entity `{spec.entity_col}` from table "
            f"`{spec.entity_table}`, seed time in `{spec.time_col}`), predict "
            f"`{spec.target_col}` over the window (t, t + {spec.timedelta_str}]."
        )
    if spec.num_eval_timestamps > 1:
        lines.append(
            f"This task uses {spec.num_eval_timestamps} consecutive evaluation windows "
            "per split (multiple seed timestamps in val/test)."
        )

    # The exact label-generating SQL is public task definition — extremely
    # useful for feature engineering (and for generating extra training
    # windows from allowed history).
    if not spec.is_autocomplete:
        try:
            src = inspect.getsource(type(task).make_table)
            lines.append(
                "Exact label-generation code (`make_table`) for reference:\n```python\n"
                + textwrap.dedent(src)
                + "\n```"
            )
        except (OSError, TypeError):
            pass
    return "\n".join(lines)
# ---------------------------------------------------------------------------
# Contract / rules / resources
# ---------------------------------------------------------------------------

def _prediction_contract(spec: TaskSpec, n_val: int, n_test: int) -> str:
    shape_val = spec.expected_pred_shape(n_val)
    shape_test = spec.expected_pred_shape(n_test)
    if spec.is_recommendation:
        dtype_line = (
            "dtype: integer destination ids in [0, "
            f"{spec.num_dst_nodes}), ranked best-first, distinct within each row"
        )
    elif spec.is_multiclass:
        dtype_line = "dtype: float scores, one column per class (higher = more likely)"
    elif spec.family in (ENTITY_BINARY, AUTOCOMPLETE_BINARY):
        dtype_line = "dtype: float probabilities in [0, 1]"
    else:
        dtype_line = "dtype: float predictions on the original target scale"
    return f"""
Your program `main.py` must, on EVERY run (debug and full), write exactly these files into
the directory given by the environment variable KAPSO_RUN_DATA_DIR:
- val_predictions.npy  — numpy array, shape {shape_val}, for the val split, row i aligned
  with row i of `task.get_table("val")` in its original order.
- test_predictions.npy — numpy array, shape {shape_test}, for the test split, row i aligned
  with row i of `task.get_table("test")` in its original order.
- {dtype_line}.
- Save with np.save; never reorder, drop, or deduplicate task-table rows.
- CRITICAL — validation predictions must be OUT-OF-SAMPLE: nothing in the
  chain that produces your validation predictions may have been fit on
  validation labels — the model, calibrators, decision thresholds, feature
  selectors, early-stopping criteria, stacking meta-learners, and ensemble
  weights all included. Training on train+validation IS allowed — and
  encouraged — for the chain that produces your test predictions (the
  two-model pattern: model A, fit without validation labels, produces the
  validation predictions; model B, refit on train+validation, produces the
  test predictions). If one pipeline refits on train+validation for test,
  keep the validation predictions from the pre-refit model — never
  regenerate them from the refit. Before every evaluation you submit, verify
  which fit produced the validation predictions. An in-sample validation
  score is self-defeating: validation is the only selection and feedback
  signal, so inflating it selects a weaker model over your genuinely better
  ones.
- CRITICAL — the validation split is ONE finite sample; the hidden test set
  is ANOTHER, and its distribution can differ (temporal shift, regime
  change, different sampling). A small validation edge bought by tuning
  against the validation score usually does not transfer and often inverts.
  Concretely: never use the official validation score as a tuning target —
  no hyperparameter search, feature/blend selection, early stopping, or
  design pruning driven by it; tune only on internal resampling of the
  training data whose splits mirror how test differs from train
  (forward-chained splits for time-ordered data, grouped splits for grouped
  data, K-fold otherwise). Prefer the design with the best and most stable
  internal-resampling mean over the one maximizing the single validation
  number; treat validation differences within noise (~1-2 standard errors)
  as ties and break ties toward the simpler, more regularized, more
  resampling-stable design. Do not resubmit near-identical variants of your
  validation-best solution to polish the last decimals — repeats that keep
  the same validation predictions add no information and overfit the
  selection.
- Optionally write metrics.json with any self-measured diagnostics.

The evaluation harness computes the official metrics itself from these files. Your score
for the search is the VALIDATION {spec.primary_metric} ({'higher' if spec.maximize else 'lower'}
is better). Test metrics are computed but hidden from you.

Run modes:
- `python main.py --debug` must finish in under {spec.debug_timeout // 60} minutes — it is a
  pipeline-correctness gate, not a mini training run: cut the work however you like
  (subsample rows, truncate training, skip expensive blocks full mode rebuilds) as long
  as the complete pipeline is exercised end to end and both full-shape prediction files
  are written (cheap/constant predictions are fine for rows you skip). Exceeding the
  debug budget kills the run before it ever scores; budget roughly half the limit to be
  safe.
- `python main.py` (full mode) must finish in under {spec.full_timeout // 3600:.1f} hours
  including all embedding/feature computation. Budget time explicitly: print elapsed time
  after each phase; leave a safety margin to write predictions.
"""


def _data_access_rules(spec: TaskSpec) -> str:
    ac_note = ""
    if spec.is_autocomplete:
        ac_note = (
            "- The database keeps rows after T (test rows must be predictable), but the "
            "target column has been physically blanked for rows after T, and correlated "
            "leak columns were removed. Do not try to recover them.\n"
        )
    else:
        ac_note = (
            "- The database is physically truncated at the test cutoff T; rows after T "
            "do not exist in your copy.\n"
        )
    return f"""
Data access (violations invalidate the run):
- INTEGRITY — do NOT look up this problem's published solution. This task may
  derive from a public competition: never search for, read, or port a
  published SOLUTION to THIS specific task/dataset (winning write-ups,
  leaderboard code, answer keys, feature recipes tuned to it). General
  methods, domain background, and library usage are fine. Solve it yourself.
- PRETRAINED MODELS (encouraged) — any pretrained model may be downloaded
  and used however helps (fine-tune, distill, feature-extract).
- EXTERNAL DATASETS (encouraged) — allowed under ONE condition: ZERO leakage
  into the test windows; ANY leakage voids the experiment. Test labels are
  public real-world history, so truncate any source covering this database's
  domain at the cutoff before features touch it, and document each source in
  changes.log (provenance + leak-free argument).
- SYNTHETIC DATA — generating synthetic data yourself is legal.
{ac_note}- Temporal censoring is YOUR responsibility: every feature/join for a seed
  row at time t uses only rows with time <= t.
- Val labels: model selection, and training ONLY for the test-predicting
  model — never for the model producing val predictions (two-model
  contract). Test rows expose only ({spec.time_col}, {'src id' if spec.is_recommendation else 'entity id'}).
- Never call task.stats(), mask_input_cols=False on the test split, or
  db.table_dict[...].removed_cols.
"""


def _resources(spec: TaskSpec, has_gpu: bool, num_cpus: int, mem_gb: int, gpu_name: str = "") -> str:
    if has_gpu:
        detail = f" — {gpu_name}" if gpu_name else ""
        gpu_line = (
            f"this instance HAS a dedicated CUDA GPU{detail} — available to any "
            "candidate; set device from env CUDA_DEVICE, default 0"
        )
    else:
        gpu_line = "no GPU on this machine"
    return f"""
Resources & engineering:
- Hardware: {gpu_line}; ~{num_cpus} CPUs; ~{mem_gb} GB RAM. Prefer duckdb over
  pandas for big joins/aggregations; project columns early.
- Persistent cache: $KAPSO_SHARED_CACHE_DIR survives across experiments —
  store embeddings, feature matrices, and per-model val/test predictions
  there, keyed by content/version. Check-before-compute.
- pip install -q anything missing at the top of main.py. Any library or
  approach is allowed subject to the data-access rules above.
- Match INSTALLED library APIs (print versions in EDA). Known traps:
  lightgbm 4.x wants callbacks=[lgb.early_stopping(N), lgb.log_evaluation(0)]
  (early_stopping_rounds/verbose_eval raise TypeError); sklearn >=1.2 wants
  loss='absolute_error'/'squared_error'. Fix API usage on such errors — do
  not abandon the approach.
- Suppress warnings/progress bars; a few small modules, main.py orchestrates.
"""


def _iteration_protocol(spec: TaskSpec) -> str:
    return f"""
Experimentation notes for this search:
- Each experiment's evaluation output includes the official VALIDATION metrics computed
  by the harness plus your printed logs — read previous experiments' outputs carefully
  and address what they reveal (overfitting, timeouts, weak segments, degenerate preds).
- The evaluation output's run/budget notes tell you how much of the campaign budget
  remains; cached val/test predictions from earlier experiments persist in
  $KAPSO_SHARED_CACHE_DIR. How to spend the remaining budget is your call.
- Every iteration, before anything else: read features_history.md and apply the
  FEATURE ENGINEERING rules above — new features first, all tables covered.
"""


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

FEATURE_ENGINEERING_NOTE = (
    "Feature engineering has been the highest-value direction on these "
    "tasks. Representation work over the relational database — new joins "
    "across tables, temporally-censored transforms, cohort-level priors, "
    "interaction and trend encodings — has repeatedly decided them, and a "
    "recurring failure mode of past campaigns was freezing the feature "
    "matrix once an early champion emerged and never re-widening it while "
    "later iterations swapped model mechanisms. Treating the matrix as "
    "never finished — regularly asking 'which features have we not "
    "tried?' — has paid off consistently.\n"
    "Two strong suggestions (guidance, not constraints — follow the "
    "evidence your own measurements produce):\n"
    "1. Consider ALL tables — features drawing on every table in the "
    "database tend to win, and an unread table is usually unexplored "
    "signal. When you set a table aside, it is worth recording why "
    "(ideally a measured reason) in features_history.md.\n"
    "2. Lean features-over-architecture — on this benchmark, architecture "
    "swaps have repeatedly measured dead while feature widening kept "
    "paying, so new features are usually the better first bet for "
    "iteration budget."
)

ROLLING_CONTRACT_NOTE = (
    "This task's eval timestamps roll YEARS past the database freeze, so it is "
    "evaluated tick-by-tick under per-row seed-time censoring (the same regime "
    "the published bars were set in). The evaluation invokes your `main.py` "
    "ONCE PER TICK. On each invocation:\n"
    "- $RELBENCH_CACHE_DIR is a snapshot containing ONLY data dated <= that "
    "tick: db tables truncated at the tick; `train.parquet` = every label "
    "window already CLOSED by the tick (fully labeled — train on it); "
    "`test.parquet` = just that tick's rows (inputs only); the snapshot's "
    "val table exists but is EMPTY by design — per-tick tuning splits come "
    "from train.parquet.\n"
    "- Write predictions for the cache's test table rows, in order, to "
    "$KAPSO_RUN_DATA_DIR/test_predictions.npy. Do NOT write "
    "val_predictions.npy — the grader assembles ticks itself.\n"
    "- Later ticks see more history (earlier windows' outcomes are ordinary "
    "closed results by then): recompute features per tick; retraining per tick "
    "is allowed and usually pays.\n"
    "- Budget: the per-run timeout covers ALL ticks together (~56 invocations); "
    "`--debug` must finish a tick in a few seconds (subsample origins/trees). "
    "Cache expensive per-origin artifacts in $KAPSO_SHARED_CACHE_DIR keyed by "
    "origin date — snapshots grow monotonically, so earlier-origin features "
    "are reusable verbatim across ticks.\n"
    "- Never try to locate other ticks' snapshots or the rolling root: each "
    "invocation legally sees its own snapshot only, and reaching outside it "
    "is flagged by the audit."
)

LIVING_DOCUMENTS_NOTE = (
    "Two agent-maintained files live in the shared artifact workspace "
    "($KAPSO_SHARED_CACHE_DIR) and persist across iterations and campaigns:\n"
    "- table_information.md — YOUR knowledge base about the database, "
    "starting empty. Build it as you explore: table and column semantics, "
    "the join graph and multi-hop join paths you use, unit quirks, null "
    "semantics, joins that worked, dead ends with their measured reasons. "
    "Keep it factual; do not delete earlier notes unless they are measured "
    "wrong.\n"
    "- features_history.md — the campaign's persistent feature memory. READ "
    "it before proposing features; APPEND one entry for every feature (or "
    "feature group) you propose or test: what it is, its status (PROPOSED / "
    "TESTED-KEPT / TESTED-REJECTED / BLOCKED), the measured outcome, and "
    "the takeaway. Append-only — never delete or rewrite existing entries; "
    "parallel lanes append concurrently.\n"
    "Consult both files at the start of every session — they are the "
    "campaign's memory of what has been learned and tried."
)

FEATURES_HISTORY_TEMPLATE = """# Feature history — {problem_id} (living memory)

Append ONE entry per proposed or experimented feature (or coherent feature
group). This file is the campaign's persistent memory of what has been
tried: READ it before proposing features, APPEND after every experiment.
Append-only — never delete or rewrite prior entries (parallel lanes append
concurrently; earlier campaigns' entries are evidence, not clutter).

Entry format:
### <feature or group name>
- run/experiment: <id> | status: PROPOSED | TESTED-KEPT | TESTED-REJECTED | BLOCKED
- what: <tables/columns/transform in one line>
- outcome: <measured metric deltas / gate numbers, or why it was blocked>
- takeaway: <one line the next proposer should know>

## Entries

(none yet)
"""


def build_table_information(dataset_name: str) -> str:
    """Seed for the agent-built table_information.md living doc.

    Deliberately near-empty: the full schema is already in the problem
    context, so this file holds only what agents VERIFY themselves —
    populated during exploration, persistent across iterations and
    campaigns."""
    return "\n".join([
        f"# Table information — {dataset_name} (LIVING DOCUMENT — build it yourself)",
        "",
        "Agent-maintained knowledge base for this database, empty by design: "
        "populate it as you explore and keep it current. Document what you "
        "verify: table and column semantics, the foreign-key join graph and "
        "the multi-hop join paths you actually use, unit quirks, null "
        "semantics, joins that worked, and dead ends with their measured "
        "reasons. Append; do not delete factual notes from earlier sessions "
        "unless they are measured wrong. (The full schema listing is in your "
        "problem context — this file is for everything the schema does NOT "
        "tell you.)",
        "",
        "## Notes (append below)",
        "",
    ])


def build_problem_context(
    task,
    dataset,
    spec: TaskSpec,
    db,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_test: int,
    has_gpu: bool,
    num_cpus: int,
    mem_gb: int,
    extra_knowledge: str = "",
    gpu_name: str = "",
    rolling: bool = False,
) -> str:
    sections = [
        "# RelBench task",
        "You are competing on RelBench (the relational deep learning benchmark, "
        "relbench.stanford.edu). Goal: the best possible TEST metric. "
        "Model selection uses the VALIDATION metric — a finite-sample proxy for it, not the objective itself. No form of test leakage is permitted. "
        f"Primary metric: **{spec.primary_metric}** "
        f"({'higher' if spec.maximize else 'lower'} is better). "
        f"All official metrics computed: {', '.join(spec.metrics)}.",
        DATASET_NOTES.get(spec.dataset_name, ""),
        "\n## Task definition\n" + _task_definition(task, spec),
        "\n## Label statistics\n" + _label_stats(spec, train_df, val_df),
        "\n## Database schema (your sanitized copy)\n" + describe_database(db, dataset),
        "\n## Prediction contract\n" + _prediction_contract(spec, len(val_df), n_test),
        ("\n## ROLLING EVALUATION — read carefully\n" + ROLLING_CONTRACT_NOTE) if rolling else "",
        "\n## Data access rules\n" + _data_access_rules(spec),
        "\n## Feature engineering (high-value direction — suggestion)\n"
        + FEATURE_ENGINEERING_NOTE,
        "\n## Resources\n" + _resources(spec, has_gpu, num_cpus, mem_gb, gpu_name),
        "\n## Living documents (shared artifact workspace)\n"
        + LIVING_DOCUMENTS_NOTE,
        "\n## Iteration protocol\n" + _iteration_protocol(spec),
    ]
    if extra_knowledge:
        sections.append("\n## Additional knowledge\n" + extra_knowledge)
    sections.append(
        "\nStarter kit: `kapso_datasets/` contains contract helpers only — "
        "`common.py` (env/task loading and prediction-saving that already respect "
        "the contract) and `check_predictions.py` (pre-validate prediction shapes). "
        "Method choice is entirely yours. Do not modify files under "
        "`kapso_evaluation/`."
    )
    sections.append(
        "\nRun retraction: every full evaluation is archived and competes in "
        "final selection. If you later conclude an archived evaluation of yours "
        "was invalid — e.g. you found and fixed leakage after running it — void "
        "it explicitly: `python kapso_evaluation/grader.py --void run_XXXX "
        '--reason "<what was wrong>"`. Voided runs are excluded from final '
        "selection; an unvoided invalid run can be selected over your corrected "
        "work purely on its inflated validation score."
    )
    return "\n".join(s for s in sections if s)
