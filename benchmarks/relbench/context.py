"""Problem-context builder for RelBench tasks.

Assembles the static problem description handed to the ideation LLM and the
coding agent: schema, task definition (including the exact label-generating
SQL), split protocol, prediction contract, hard leakage rules, resource
constraints, and a family-specific playbook of techniques that are known to
move the needle on RelBench.

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
# Family playbooks
# ---------------------------------------------------------------------------

_COMMON_PLAYBOOK = """
- Baseline first, then iterate: get a correct end-to-end run (even a simple model) writing
  both prediction files before attempting anything sophisticated. A scored run beats a
  brilliant crash.
- The official SOTA reference for this benchmark is Relational Deep Learning (RDL): a
  heterogeneous temporal GNN (GraphSAGE-style) over the pkey-fkey graph with per-table
  PyTorch Frame column encoders and text embeddings, trained with time-aware neighbor
  sampling. Starter implementations are vendored in `kapso_datasets/` — adapt
  them (they are known-correct on temporal handling) rather than writing samplers from scratch.
- Strong non-GNN contender on many tasks: temporally-censored SQL feature engineering
  (duckdb) + gradient boosting (LightGBM/XGBoost/CatBoost). Aggregates over each fkey
  relation at multiple lookback windows (7/30/90/365 days, all-time): counts, recency
  (days since last event), velocity (ratio of recent to older counts), means/sums of
  numeric columns, distinct counts, plus target-history features where legal. Every SQL
  aggregation MUST carry `event_time <= seed_time`.
- Extra training data is legal and valuable: labels for any seed time t are derivable from
  the database as long as the full label window (t, t + timedelta] lies at or before the
  test cutoff T. The official train table stops at V - timedelta; you may additionally
  generate seed rows in (V - timedelta, T - timedelta] with the task's own `make_table`
  logic and train on train + these + val. More-recent windows are closer to the test
  distribution — use them (a recency-weighted loss often helps too).
- For the final model, retrain on train+val (after model selection on val) when the val
  metric is stable; keep an untouched early-stopping slice carved from the tail of train.
- Text columns: GloVe averages (starter default) are weak; upgrade to a stronger sentence
  encoder (e.g. intfloat/e5-base-v2 or BAAI/bge-base-en-v1.5 if available locally) when
  text is informative. Embed ONCE and cache in $KAPSO_SHARED_CACHE_DIR keyed by
  (table, column, model); re-embedding every run wastes the time budget.
- Ensembling reliably wins on this benchmark: combine diverse models (GNN + GBDT + linear
  autoregressors). For AUROC/MAP use rank-averaging; for MAE use weighted means with
  weights fit on val. Keep per-model val predictions in $KAPSO_SHARED_CACHE_DIR so later
  experiments can ensemble earlier models without retraining.
- Log the val metric per epoch, early-stop on the PRIMARY metric, and print final
  val metrics clearly. Print a compact EDA at the start (label stats, key table sizes)
  and an error analysis at the end (worst segments, prediction distribution).
"""

_ENTITY_CLS_PLAYBOOK = """
Binary entity classification specifics (primary: roc_auc, higher is better):
- AUROC is rank-based: monotone transforms of scores don't matter; focus on ordering.
  Output probabilities in [0,1] anyway (accuracy/F1 at 0.5 are also reported).
- Class imbalance: prefer more informative negatives per epoch over loss reweighting.
- Label-history autoregression is powerful: the entity's own past label windows
  (churned last month? active streak length?) computed with `make_table` logic at past
  seed times are legal features at time t (window fully before t). The starter GNN
  supports including past task tables as features.
- GNN: 2 layers, 128 channels, fanout ~[128, 128], sum aggregation, uniform temporal
  sampling is the known-good starting point; try attention/PNA aggregation, deeper
  fanouts on small DBs, and shallow entity embeddings for high-overlap entity sets.
- GBDT on censored SQL features frequently matches or beats the GNN on sparse/behavioral
  churn tasks — always field it, then ensemble both.
"""

_ENTITY_REG_PLAYBOOK = """
Regression specifics (optimize the PRIMARY metric named above — the two published bars
differ: the official leaderboard ranks by NMAE = MAE / std(train targets), which is
per-task monotone with plain MAE; the v2 paper ranks the newer tasks by R^2, which is
MSE-flavored — a median-style MAE-optimal model can score terribly on R^2 and vice versa):
- If primary is MAE: the optimal point prediction is the conditional MEDIAN — train with
  L1/quantile-0.5 objectives (LightGBM objective='regression_l1'), not MSE.
- If primary is R^2: train with MSE/L2 (optionally on log1p scale with careful
  back-transform), and mind outliers via clipping of TRAINING targets, not predictions.
- Targets are typically heavy-tailed, non-negative, and often ZERO-INFLATED (counts,
  LTV, attendance): the single biggest known win on such tasks is distribution-aware
  prediction — hurdle models (P(y>0) x E[y|y>0]), tweedie objectives, or plain
  quantile regression. On rel-event user-attendance this class of trick is worth ~10x
  NMAE vs naive regression (0.03 vs the 0.31 cluster on the board).
- Clip predictions to [0, q99.5(train)]; if the target is integer-valued, compare
  rounded vs raw predictions on val.
- All of mae/rmse/r2 are computed each run — track them all, optimize the primary.
"""

_ENTITY_MC_PLAYBOOK = """
Multiclass specifics (primary: accuracy, higher is better):
- Output an (N, num_classes) score matrix; argmax drives accuracy, full ordering drives
  MRR (also reported). Softmax probabilities are fine.
- Class priors shift over time; consider recency-weighted training and prior calibration
  toward the val-period class distribution.
"""

_RECOMMENDATION_PLAYBOOK = """
Recommendation specifics (primary: link_prediction_map = MAP@K, higher is better):
- Output exactly (N, K) integer destination ids per val/test seed row, ranked best-first,
  no duplicates within a row, all ids in [0, num_dst_nodes).
- REPEAT BEHAVIOR IS KING: on purchase/visit-style tasks, most future interactions are
  re-interactions. A time-decayed frequency+recency ranking of the source's own past
  destinations, backfilled with global/recent popularity, is a brutal baseline that beats
  most GNNs — implement it FIRST (see starter_kit/recommendation_heuristics.py).
- The winning pattern on H&M-style data is candidate-generation + GBDT re-ranking:
  candidates = user's past items + item-item co-occurrence neighbors (items bought
  together within sessions/windows) + recent-popularity by segment; features = per
  (src, dst) recency/frequency/co-occurrence stats + entity features; train LightGBM
  ranker (lambdarank) or binary classifier on historical windows built the same way.
- GNN alternative: identity-aware GNN (ID-GNN) scoring destinations against each source's
  temporal subgraph; 4-layer ID-GNN beats 2-layer nearly everywhere. The published SOTA
  recipe (ContextGNN) is a hybrid: pair-wise NBFNet-style scores for destinations inside
  the source's sampled subgraph + a two-tower with SHALLOW learnable destination
  embeddings for everything outside it, fused per source; train with sampled softmax over
  very many negatives (10^5-10^6), not BPR pairs. It roughly tripled two-tower MAP on
  rel-amazon and doubled the board number on rel-trial site-sponsor-run. Starter provides
  ID-GNN; upgrading it toward the hybrid is a high-value experiment.
- Evaluate MAP@K on val exactly like the harness does before trusting a model; a custom
  fast MAP implementation avoids surprises.
- Popularity fallback for cold sources; never emit fewer than K distinct ids.
"""

_AUTOCOMPLETE_PLAYBOOK = """
Autocomplete specifics:
- This is row-attribute prediction at row-insert time: the split is by the row's own
  timestamp (train <= V < val <= T < test). Inputs = the row's other columns + anything
  reachable through its foreign keys, censored at the row's timestamp.
- Gradient boosting over joined + censored-aggregate features is the strongest known
  simple recipe here (the v2 baselines are beatable): join parent tables' attributes,
  add per-parent historical aggregates of the TARGET where legal (e.g. the user's mean
  past rating, the product's mean past rating for review-rating), plus counts/recency.
  Historical target aggregates must only use rows with time strictly <= the seed row's
  time, from train/val labels you legitimately have.
- Temporal shift is real (test rows are the newest): recency-weight training rows and
  retrain on train+val for the final test model.
- Many autocomplete targets are JOIN-LOOKUPS in disguise: the same customer's previous
  orders carry the same payterms/incoterms; an item's historical price nearly determines
  the next transaction price; a user's and beer's past ratings pin down the next score.
  Published GNN baselines beat LightGBM by 40-96 accuracy points on rel-salt only because
  naive LightGBM never joined those lookups — an explicit most-recent-value-from-history
  feature per (entity, target) closes that gap instantly. Build 'previous value of the
  target for the same parent entity' features FIRST.
- Text columns: TF-IDF features are surprisingly strong for autocomplete (worth up to
  +10 AUROC on text-heavy tasks) — cheap to try alongside sentence embeddings.
- The GNN starter (gnn_autocomplete) treats the new row as a node with its known columns
  and predicts the target from the sampled temporal subgraph — good complement for
  ensembling with GBDT.
- Some 'known' columns of the target row may be missing at prediction time in spirit —
  use exactly the columns present in the sanitized database, nothing else.
"""

_FAMILY_PLAYBOOKS = {
    ENTITY_BINARY: _ENTITY_CLS_PLAYBOOK,
    ENTITY_MULTICLASS: _ENTITY_MC_PLAYBOOK,
    ENTITY_REGRESSION: _ENTITY_REG_PLAYBOOK,
    RECOMMENDATION: _RECOMMENDATION_PLAYBOOK,
    AUTOCOMPLETE_BINARY: _ENTITY_CLS_PLAYBOOK + _AUTOCOMPLETE_PLAYBOOK,
    AUTOCOMPLETE_MULTICLASS: _ENTITY_MC_PLAYBOOK + _AUTOCOMPLETE_PLAYBOOK,
    AUTOCOMPLETE_REGRESSION: _ENTITY_REG_PLAYBOOK + _AUTOCOMPLETE_PLAYBOOK,
}


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
- `python main.py --debug` must finish in under {spec.debug_timeout // 60} minutes: subsample
  training work aggressively (e.g. a few thousand training rows, 1 epoch, small model) but
  STILL produce both full-shape prediction files (cheap/constant predictions are fine for
  rows you skip). This validates the pipeline end to end.
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
Data access (read carefully — violations invalidate the run):
- Load data ONLY through the relbench API with download=False:
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    dataset = get_dataset("{spec.dataset_name}", download=False)
    task = get_task("{spec.dataset_name}", "{spec.task_name}", download=False)
    db = dataset.get_db()                      # input database
    train = task.get_table("train")            # seed rows + labels
    val   = task.get_table("val")              # seed rows + labels (tuning allowed)
    test  = task.get_table("test")             # seed rows ONLY (labels held out)
  RELBENCH_CACHE_DIR is preset to a sanitized read-only cache prepared for this task.
- `download=True` will fail (read-only cache) — never use it. Never change
  RELBENCH_CACHE_DIR, never read or write ~/.cache/relbench, never fetch dataset files
  from the network.
{ac_note}- Temporal censoring is YOUR responsibility inside allowed data: every feature,
  aggregation, join, or sampled neighborhood for a seed row at time t must only use
  rows with time <= t. The starter kit's samplers/SQL templates do this correctly.
- Validation labels are for model selection AND may be used as training data for the
  model that produces TEST predictions (they lie before the test cutoff) — but never
  for the model that produces VAL predictions (see the prediction contract). Test rows
  expose only ({spec.time_col}, {'src id' if spec.is_recommendation else 'entity id'}).
- Do not call task.stats(), mask_input_cols=False on the test split, or
  db.table_dict[...].removed_cols — they either crash on the sanitized cache or are
  off-limits.
- Determinism: seed numpy/torch/random; keep a fixed seed across runs so improvements
  are attributable to ideas, not noise. If variance is suspected, average 2-3 seeds.
"""


def _resources(spec: TaskSpec, has_gpu: bool, num_cpus: int, mem_gb: int) -> str:
    gpu_line = (
        "one CUDA GPU (set device from env CUDA_DEVICE, default 0); use it for any "
        "neural model; keep GBDT on CPU"
        if has_gpu
        else "NO GPU — prefer GBDT/duckdb pipelines; keep any torch model tiny and CPU-friendly"
    )
    return f"""
Resources & engineering:
- Hardware: {gpu_line}; ~{num_cpus} CPUs; ~{mem_gb} GB RAM. Parallelize dataloading and
  duckdb with the CPU count; watch memory on big tables (project columns early,
  prefer duckdb over pandas for joins/aggregations on millions of rows).
- Persistent cache: $KAPSO_SHARED_CACHE_DIR survives across experiments. Store text
  embeddings, materialized graphs, engineered feature matrices, and per-model val/test
  predictions there, keyed by a content/version string. Check-before-compute.
- Install any missing pip package quietly at the top of main.py (pip install -q).
  Allowed/typical: torch, torch_geometric, pytorch_frame, relbench, lightgbm, xgboost,
  catboost, sentence-transformers, duckdb, polars.
- Match the INSTALLED library APIs (print versions in your EDA) — modern majors have
  removed legacy kwargs. Known traps here: lightgbm 4.x (`lgb.train` takes
  callbacks=[lgb.early_stopping(N), lgb.log_evaluation(0)]; early_stopping_rounds /
  verbose_eval raise TypeError) and sklearn >=1.2 (GradientBoostingRegressor
  loss='absolute_error'/'squared_error', not 'lad'/'ls'). When a run fails on such a
  TypeError/InvalidParameterError, fix the API usage — do not abandon the approach.
- Suppress warnings and progress bars (tqdm disable) — logs must stay readable.
- Structure code across a few small modules; main.py orchestrates end to end.
"""


def _iteration_protocol(spec: TaskSpec) -> str:
    return f"""
Experimentation protocol for this search:
- Each experiment's evaluation output includes the official VALIDATION metrics computed
  by the harness plus your printed logs — read previous experiments' outputs carefully
  and address what they reveal (overfitting, timeouts, weak segments, degenerate preds).
- When the parent experiment succeeded, at least one child must be the same solution with
  meaningfully better hyperparameters/training budget (deeper search around a winner).
- Keep diversity: at least one child per expansion should try a structurally different
  model family (GNN vs GBDT vs heuristic/ensemble) until it's clear which dominates here.
- When run/budget notes in the evaluation output indicate the LATE phase (>75% budget),
  stop exploring: ensemble the best distinct models from earlier experiments (their
  cached val/test predictions are in $KAPSO_SHARED_CACHE_DIR) and fine-tune the winner.
- Beat-the-number focus: the current published state of the art for this task is shown
  below (if known). Treat it as the bar; report progress against it in your logs.
"""


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

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
        "\n## Resources\n" + _resources(spec, has_gpu, num_cpus, mem_gb),
        "\n## Playbook (battle-tested guidance — use it)\n"
        + _COMMON_PLAYBOOK
        + _FAMILY_PLAYBOOKS[spec.family],
        "\n## Iteration protocol\n" + _iteration_protocol(spec),
    ]
    if sota_note:
        sections.append("\n## Published state of the art for this task\n" + sota_note)
    if extra_knowledge:
        sections.append("\n## Additional knowledge\n" + extra_knowledge)
    sections.append(
        "\nStarter kit: `kapso_datasets/` contains vendored official RelBench "
        "example implementations (temporal GNNs for all three families, LightGBM "
        "baselines, ID-GNN recommendation) plus `common.py` (env/task loading and "
        "prediction-saving helpers that already respect the contract) and "
        "`recommendation_heuristics.py`. Adapt them; do not modify files under "
        "`kapso_evaluation/`."
    )
    return "\n".join(s for s in sections if s)
