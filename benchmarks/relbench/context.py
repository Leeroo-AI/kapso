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
- CRITICAL — val predictions must be OUT-OF-SAMPLE: the model that produces
  val_predictions.npy must never have seen val labels during training or tuning-fit.
  Training on train+val is allowed ONLY for the model producing test_predictions.npy
  (the two-model pattern: model A trained on train -> val preds; model B trained on
  train+val -> test preds). Val predictions from a val-trained model inflate the
  selection signal and the solution will collapse at the final test evaluation.
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
- Beat-the-number focus: the current published state of the art for this task is shown
  below (if known). Treat it as the bar; report progress against it in your logs.
- Every iteration, before anything else: read features_history.md and apply the
  FEATURE ENGINEERING rules above — new features first, all tables covered.
"""


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

FEATURE_ENGINEERING_NOTE = (
    "FEATURE ENGINEERING IS THE PRIMARY DIRECTION OF THIS CAMPAIGN — the "
    "axis where iteration budget goes first and the one this instruction "
    "emphasizes above everything else. Representation work over the "
    "relational database — new joins across tables, temporally-censored "
    "transforms, cohort-level priors, interaction and trend encodings — "
    "repeatedly decides these tasks, and a documented failure mode of "
    "past campaigns is freezing the feature matrix once an early "
    "champion emerges and never re-widening it while later iterations "
    "swap model mechanisms. Treat the feature matrix as never finished: "
    "every iteration should ask 'which features have we not tried?' "
    "before it asks anything else, and an iteration may leave the matrix "
    "unchanged only on measured saturation evidence (an ablation or "
    "importance study showing no headroom), never by default.\n"
    "Two standing rules:\n"
    "1. ALL TABLES (hard rule) — your features must draw on EVERY table in "
    "the database. Excluding a table is allowed only with a justified, "
    "MEASURED reason that it carries no extractable signal (a diagnostic "
    "you ran, recorded in features_history.md) — and 99% of tables are not "
    "useless, so an unread table is unfinished work, never a default. "
    "Prior campaigns left entire tables unread without ever evaluating "
    "them; this rule exists to kill exactly that failure.\n"
    "2. FEATURES OVER ARCHITECTURE — when allocating iteration budget, "
    "prefer proposing and experimenting with NEW FEATURES over new model "
    "architectures: on this benchmark architecture swaps have repeatedly "
    "measured dead while feature widening kept paying."
)

BEST_PRACTICES_NOTE = (
    "**Error decomposition discipline.** Decompose error by INFORMATION "
    "AVAILABILITY, not by feature family. Backtest under deployment-matched "
    "conditions (for forward-prediction tasks: rolling frozen-origin "
    "replays — freeze the available data at historical cutoffs, predict the "
    "windows that follow, score on labels that lie before your real cutoff). "
    "Split the measured error into two classes: (1) REGIME-SHIFTED rows — "
    "entities whose state changed after the information boundary, so the "
    "change is unobservable in your inputs; no feature engineered from the "
    "available data can reduce this class — only recovering legal "
    "additional information or exploiting structure in the task's own "
    "revealed inputs can; (2) PREDICTABLE rows — entities continuing the "
    "trajectory your inputs imply; this class rewards better features, "
    "ordering, and calibration. Report the split every iteration, alongside "
    "rank-quality metrics (not just aggregate error — compressed "
    "predictions with weak ordering and sharp predictions with strong "
    "ordering fail differently). Direct effort at whichever class carries "
    "the dominant MEASURED mass, and treat any steering hypothesis as "
    "subordinate to your own measurement: if the decomposition contradicts "
    "an assumed error story, follow the measurement."
)

LIVING_DOCUMENTS_NOTE = (
    "Two agent-maintained files live in the shared artifact workspace "
    "($KAPSO_SHARED_CACHE_DIR) and persist across iterations and campaigns:\n"
    "- table_information.md — schema, join graph, and table semantics. You "
    "are allowed and expected to EDIT it as you learn new facts about the "
    "tables during experimentation: append discovered semantics, unit "
    "quirks, joins that worked, dead ends with their measured reasons. "
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


def build_table_information(db, dataset, dataset_name: str) -> str:
    """Seed content for the agent-editable table_information.md living doc.

    Auto-derived from the actual database so it works for every RelBench
    dataset: full schema, the foreign-key join graph, and two-hop join
    paths through bridge tables (the paths past campaigns never built).
    """
    edges = []
    for name, table in sorted(db.table_dict.items()):
        for col, parent in table.fkey_col_to_pkey_table.items():
            edges.append((name, col, parent))
    edge_lines = [f"- `{child}.{col}` -> `{parent}`" for child, col, parent in edges]
    parents_of: dict = {}
    for child, col, parent in edges:
        parents_of.setdefault(child, []).append((col, parent))
    two_hop = []
    for child, col, parent in edges:
        for col2, grand in parents_of.get(parent, []):
            two_hop.append(
                f"- `{child}` -> `{parent}` (via `{col}`) -> `{grand}` (via `{col2}`)"
            )
    sections = [
        f"# Table information — {dataset_name} (LIVING DOCUMENT)",
        "",
        "Agent-maintained: extend and correct this file as you learn table "
        "semantics during experimentation. Append; do not delete factual "
        "notes from earlier sessions unless they are measured wrong.",
        "",
        "## Schema",
        "",
        describe_database(db, dataset),
        "",
        "## Join graph (foreign keys)",
        "",
        "\n".join(edge_lines) if edge_lines else "(no foreign keys)",
    ]
    if two_hop:
        sections += ["", "Two-hop join paths through bridge tables:", "", "\n".join(two_hop)]
    sections += [
        "",
        "## Semantics and gotchas (append below)",
        "",
        "(column meanings, unit quirks, null semantics, joins that worked, "
        "dead ends with their measured reasons)",
        "",
    ]
    return "\n".join(sections)


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
    sota_note: str = "",
    extra_knowledge: str = "",
    gpu_name: str = "",
) -> str:
    sections = [
        "# RelBench task",
        "You are competing on RelBench (the relational deep learning benchmark, "
        "relbench.stanford.edu). Goal: the best possible TEST metric, achieved by "
        "maximizing the validation metric without any form of test leakage. "
        f"Primary metric: **{spec.primary_metric}** "
        f"({'higher' if spec.maximize else 'lower'} is better). "
        f"All official metrics computed: {', '.join(spec.metrics)}.",
        DATASET_NOTES.get(spec.dataset_name, ""),
        "\n## Task definition\n" + _task_definition(task, spec),
        "\n## Label statistics\n" + _label_stats(spec, train_df, val_df),
        "\n## Database schema (your sanitized copy)\n" + describe_database(db, dataset),
        "\n## Prediction contract\n" + _prediction_contract(spec, len(val_df), n_test),
        "\n## Data access rules\n" + _data_access_rules(spec),
        "\n## FEATURE ENGINEERING — THE PRIMARY DIRECTION\n"
        + FEATURE_ENGINEERING_NOTE,
        "\n## Best practices (strongly recommended — follow unless you have a measured reason not to)\n"
        + BEST_PRACTICES_NOTE,
        "\n## Resources\n" + _resources(spec, has_gpu, num_cpus, mem_gb, gpu_name),
        "\n## Living documents (shared artifact workspace)\n"
        + LIVING_DOCUMENTS_NOTE,
        "\n## Iteration protocol\n" + _iteration_protocol(spec),
    ]
    if sota_note:
        sections.append("\n## Published state of the art for this task\n" + sota_note)
    if extra_knowledge:
        sections.append("\n## Additional knowledge\n" + extra_knowledge)
    sections.append(
        "\nStarter kit: `kapso_datasets/` contains contract helpers only — "
        "`common.py` (env/task loading and prediction-saving that already respect "
        "the contract) and `check_predictions.py` (pre-validate prediction shapes). "
        "Method choice is entirely yours. Do not modify files under "
        "`kapso_evaluation/`."
    )
    return "\n".join(s for s in sections if s)
