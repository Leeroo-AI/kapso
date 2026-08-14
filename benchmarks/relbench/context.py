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
from typing import List, Optional

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
- HOSTED MODEL APIS (encouraged) — calling hosted LLM APIs with the
  provided credentials (OPENAI_API_KEY) is explicitly PERMITTED, for
  feature extraction and anything else that helps. There is NO
  competition-style prohibition on third-party model APIs here — do not
  assume one, and do not design around one.
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
- When the schema has text columns, your FIRST full evaluation must EXECUTE the
  LLM text measurement of modelling practice 8: the run itself executes a
  non-empty LLM extraction batch — hosted-LLM calls, or a locally served open
  instruct model validated against a hosted-scored panel — for feature
  extraction over the text or direct classification, whichever you designed —
  and your logs report how many rows were scored and the measured score of the
  resulting features. Code that is wired up but scores zero rows at run time
  is not the measurement; a campaign must not reach final selection without
  this comparison having actually run.
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

MODELLING_PRACTICE_NOTE = (
    "Practices that measured positive when ablated on tasks of this "
    "benchmark. Each is stated as a thing to MEASURE, not to assume — keep "
    "it only if your own numbers agree.\n"
    "1. NORMALISE FEATURES WITHIN THE COMPETING SET. This applies whenever "
    "several prediction rows share a natural grouping key — the same seed "
    "timestamp, session, parent entity, auction, or batch — so that the rows "
    "are effectively scored against one another. Alongside each informative "
    "raw feature, add its within-group rank, percentile and z-score, plus the "
    "gap to the group's leading value. An absolute value cannot distinguish "
    "'strong in a weak group' from 'strong overall', while the label often "
    "encodes exactly that relative standing. Adding this group improved the "
    "held-out metric across every model family tried (linear, shallow trees, "
    "boosted ensembles), which is the kind of consistency that makes a "
    "technique worth a default rather than a one-off.\n"
    "2. MATCH MODEL CAPACITY TO THE NUMBER OF LABELLED ROWS, AND MAKE ADDED "
    "COMPLEXITY EARN ITS PLACE. When labelled training rows number in the low "
    "thousands or fewer, always score a strongly-regularised or plainly linear "
    "model against the richer ensemble instead of assuming more capacity wins. "
    "On a small-sample task here, a linear model on a compact feature set beat "
    "every boosted ensemble AND a much larger hand-engineered blend on the "
    "held-out split — the extra capacity was fitting sample noise. Wide feature "
    "matrices and stacked models are worth building, but each added layer "
    "should be kept only where it measurably beats the simpler version.\n"
    "A companion habit for both: REPORT THE METRIC BY SLICE, not only in "
    "aggregate. Split the evaluation rows by the conditions you suspect matter "
    "(sparse vs rich history, recently-changed context, rare vs common "
    "classes) and read the metric per slice. A change that looks flat overall "
    "can be moving one slice by a large margin and another the opposite way, "
    "and the aggregate alone will never show you which.\n"
    "3. SEGMENT COVERAGE — MEASURE IT BEFORE YOU MODEL. Split the evaluation "
    "rows by how much pre-cutoff history the target entity has (none / sparse "
    "/ rich) and report, per segment: its share of train, validation, and "
    "prediction-era origins; its label rate; your metric on it. Two failure "
    "modes hide in the aggregate: (a) a segment whose share GROWS from "
    "validation to the prediction period makes validation overstate "
    "history-derived features; (b) any feature that reaches the row by "
    "joining through the entity's own past events is NULL by construction "
    "for zero-history entities, so a model can score well overall while "
    "coin-flipping that whole segment. If a material segment scores near "
    "chance: build features that do not route through the entity's own "
    "history — attributes of the entity itself, attributes of the "
    "items/context in the row, cohort aggregates through shared non-history "
    "keys (device, location, category, demographics), and system-level "
    "activity at the origin; match the training mix to the prediction-era "
    "mix with per-row weights, or fit a router plus segment-specific models; "
    "and judge candidates on the segment-weighted metric, not the "
    "rich-history-dominated average. Adopt each step only where your gated "
    "forward-fold test agrees.\n"
    "4. WHEN THE POPULATION IS DORMANT, EGO-FEATURES GO EMPTY — AND NEVER "
    "DISCARD AN UNCORRELATED FINALIST. Report your metric by entity-history "
    "recency and depth (last event >365d / 92-365d / <92d; counts 0 / 1-5 / "
    "5+). If most evaluation entities are long-inactive, features built "
    "solely from the entity's own past (counts, recencies, rates) are "
    "near-empty exactly where the evaluation mass is. Invest in signal that "
    "survives dormancy: as-of-cutoff aggregates over RELATED entities "
    "(activity and outcomes of neighbours through foreign keys), "
    "graph-derived scalars computed at the cutoff (degree, co-participation "
    "counts, community statistics), attribute-cohort baselines, and the "
    "content/text state of the entity's artefacts — and put them IN the "
    "main model's feature set; a separate weak graph model blended at the "
    "end is not a substitute for neighbourhood features the strong model "
    "can split on. Go deeper than one hop: propagate the label-generating "
    "activity itself through the relation graph — as-of-cutoff counts and "
    "outcome rates of the entity's one- AND two-hop neighbourhood over the "
    "same interaction table that defines the label, with quantiles taken "
    "within the arrival batch — and add the row's arrival context (how many "
    "prediction rows arrived together for the same origin, and that batch's "
    "aggregate reach). Where an entity-entity relation exists, a low-rank "
    "factorization of its adjacency (SVD embeddings) and community-level "
    "empirical-Bayes outcome rates were the strongest history-free signals "
    "measured on relational tasks here. Separately, before finalising, "
    "compute pairwise "
    "prediction rank-correlations among your surviving candidates: two "
    "candidates scoring within noise of each other but correlating weakly "
    "are an ensemble waiting to happen, and their rank-average is the most "
    "reliable single improvement available — gate it on your forward folds "
    "like any other change.\n"
    "5. CALIBRATE EVERY ACCEPT/SHIP GATE TO YOUR MEASURED NOISE — NEVER A "
    "ROUND NUMBER. Before adopting or rejecting any change, bootstrap your "
    "own validation metric with the CORRECT clustering unit (resample whole "
    "entities/origins, not rows, when rows within a group are correlated) "
    "and set the acceptance threshold from that measured standard error: "
    "keep a change at roughly one clustered SE of paired improvement, or "
    "equivalently when the paired bootstrap gives P(improvement > 0) of "
    "about 0.8 or better. A pre-picked bar like 'must gain 0.005' or 'must "
    "gain 0.01 pooled' feels rigorous but fails two ways at once: on a "
    "mature design the honest per-step gain is often a fraction of such a "
    "bar, so every real improvement is rejected as a tie and the search "
    "converges to re-shipping its incumbent unchanged; and when the bar "
    "exceeds what your validation SE can even resolve, the gate is not "
    "strict — it is mathematically impassable, and hours of genuine "
    "positive results get discarded. Symmetrically, a run of small "
    "accepted gains needs an occasional full-suite re-measure to confirm "
    "they compound. If two of your own significance tests disagree, trust "
    "the one whose resampling unit matches the data's correlation "
    "structure — do not pick the test that supports rejecting. And if a "
    "gate-selected weight or capacity lands on the edge of its grid, the "
    "grid is wrong: extend it and re-select before freezing — a winning "
    "add-on here chose the maximum of its weight grid and the campaign "
    "never learned how much more it wanted.\n"
    "6. BUILD AN EXPLICIT ENSEMBLE CANDIDATE FROM THE CAMPAIGN'S "
    "DECORRELATED FINALISTS. Near the end of the budget, list every "
    "archived candidate across ALL branches of the search — not just your "
    "own lineage — with its validation score and pairwise prediction "
    "rank-correlation. If two or more sit within about two clustered SE "
    "(practice 5) of the leader and correlate below roughly 0.95, submit a "
    "dedicated candidate that rank-averages them, weights fit on a forward "
    "fold; where the per-slice read shows one finalist paying "
    "disproportionately on a segment, weight it up there. Argmax selection "
    "among near-tied decorrelated finalists discards the one improvement "
    "that is reliable in expectation, and blending only within your own "
    "lineage captures almost none of it — measured here, campaigns "
    "repeatedly blended 0.98-correlated siblings while leaving 0.92-0.94 "
    "cross-branch finalists, two SE apart, unblended. The ensemble must be "
    "submitted as its own candidate run: final selection only ever sees "
    "candidates, so an ensemble that is never a candidate can never win. "
    "And any family intended for this blend must ship training-period OOF "
    "predictions at build time — a family without them cannot be selected "
    "later, and its compute strands (a campaign here spent its final "
    "iteration retrofitting exactly this).\n"
    "7. WHEN LABELLED ROWS ARE SCARCE, REPLAY THE LABEL QUERY AT EARLIER "
    "CUTOFFS. The training table's timestamps are usually a thin sample of "
    "the timestamps at which the label is well-defined; regenerate training "
    "rows at additional earlier cutoffs under exactly the same leakage "
    "discipline (features strictly pre-cutoff, labels from the window after "
    "it), multiplying the labelled set several-fold, and judge the expansion "
    "on forward-chaining folds it never saw. On a low-row task here a "
    "seven-phase replay expanded training seven-fold and was the largest "
    "single contributor to the winning design. For groups that stay sparse "
    "even after expansion, borrow strength instead of abandoning them: "
    "hierarchical/empirical-Bayes shrinkage of group rates toward their "
    "parent population, kept as first-class features the model can weigh. "
    "And never let a stacker learn a sparse-slice weight from a validation "
    "set dominated by the rich slice — a stacker that drives the sparse "
    "expert to zero is usually mis-calibrated on that slice, not proof the "
    "expert is worthless (calibrate per-slice, practice 5).\n"
    "8. TEXT COLUMNS: MEASURE AN LLM-EXTRACTED FEATURE BLOCK — A LOCAL "
    "ENCODER IS NOT THAT MEASUREMENT. An OpenAI key is present in the "
    "environment (OPENAI_API_KEY); credits are free — do not optimise for "
    "or worry about API cost. Whenever the schema has text columns:\n"
    "- First action: a single-row live probe that round-trips one call "
    "through your schema parser. Build the pipeline only on a passed "
    "probe — wired-but-never-executed channels are the measured failure "
    "mode of this practice.\n"
    "- Write the prompt as a domain-expert reviewer and ask for "
    "JUDGMENTS — adequacy, ambition, difficulty, risk, coherence — not "
    "descriptions. Two hard rules, both measured: (i) never ask for "
    "anything a join or regex over the structured columns can recover — "
    "the model is a noisier copy of the database there, and a "
    "facts-restating schema measured barely above chance standalone with "
    "a near-zero incremental delta; audit the schema against the tables "
    "first. (ii) Pass the complete document PLUS the row's structured "
    "fields rendered as context — the model's unique value is judging "
    "interactions between text and numbers (is the stated size adequate "
    "for the implied effect?), which no join can compute. Window long "
    "documents into multiple complete calls; a truncated document "
    "measures the truncation, not the text. 'Do not predict the label' "
    "means the recorded label — judgment dimensions are the point, not a "
    "violation; only a direct label guess as the channel's main output "
    "is the measured anti-pattern.\n"
    "- Outputs need not be numeric: closed-vocabulary labels (perfect "
    "categoricals — joinable and group-by-able), normalized short "
    "phrases and one-sentence judgments (embed or cluster downstream), "
    "and micro-summaries (clean inputs for cheaper models) are all valid "
    "targets. Prefer the most constrained modality that captures the "
    "signal; anchor numeric scales to the corpus ('2 = typical for this "
    "domain') so scores compare across documents; and return an explicit "
    "insufficient-evidence marker instead of a fabricated midpoint.\n"
    "- Before gating the block, regress each extracted dimension on your "
    "existing no-LLM feature matrix (OOF): high R-squared means that "
    "dimension is a copy of features you already have — drop it or keep "
    "only its residual. And treat the prompt/schema as a gated variable: "
    "run two or three variants through the same forward gates and keep "
    "the winner — one variant failing is evidence about that variant, "
    "not about the text.\n"
    "- Cache every response in $KAPSO_SHARED_CACHE_DIR, keyed by (model, "
    "prompt version, content hash). The extraction is then paid once per "
    "campaign: parallel lanes, later candidates, and re-runs consume it "
    "for free, and final candidates that reuse the cache re-evaluate in "
    "seconds.\n"
    "- When the text corpus is large relative to your hosted throughput, "
    "serve a fast open instruct model locally (e.g. vLLM with a small or "
    "small-active-MoE model from practice 9's list) and use it as the "
    "extraction engine for FULL direct coverage — then validate it "
    "against a few-thousand-row hosted-scored panel: keep the local "
    "scorer where the two agree; the panel is your quality gate, not a "
    "teacher to distill from. Prefer this over teacher-distillation "
    "whenever direct coverage is affordable on your GPUs.\n"
    "- PRIMARY WIRING: a per-fold supervised head over the judgments "
    "(plus sentence embeddings) producing OOF-complete train/val/test "
    "prediction vectors, entered as its own decorrelated finalist for "
    "practice 6 — this is the form that has repeatedly survived final "
    "selection; its errors decorrelate from history-derived channels "
    "because it reads the documents while they read the past. Secondary: "
    "per-event marks aggregated at several horizons (last event, last 4, "
    "16, 64) where a model consumes sequences — a banked winner used "
    "exactly this form. Attributes dissolved as extra columns in a large "
    "feature matrix are the measured failure mode: gates cannot see a "
    "small orthogonal lift among thousands of columns.\n"
    "- A rejected zero-shot block is evidence about that block, not "
    "about the text. Before concluding the text carries no signal, test "
    "a SUPERVISED text channel: fit a head on the frozen judgments or "
    "embeddings against your labels, or escalate to practice 9. "
    "Zero-shot rejection followed by supervised success is the measured "
    "norm on document-heavy tasks.\n"
    "9. WHEN A FROZEN TEXT CHANNEL PAYS BUT A LARGE TEXT GAP REMAINS, "
    "ESCALATE TO FINE-TUNING AN OPEN LLM ON THE TASK. Any frozen text "
    "channel — practice 8 extraction or a fine-tuned small encoder — "
    "compresses each document into a bounded schema or embedding, which "
    "discards most of the text's information about the label. Escalate to "
    "task-adaptive fine-tuning when ALL of: (a) the documents are rich "
    "enough that an expert reading them could plausibly know things your "
    "features don't — proven either by a positive frozen channel OR by a "
    "rejected zero-shot block on a document-heavy corpus (zero-shot "
    "failure does not clear the text: a fine-tune has become a winning "
    "add-on in a campaign where every zero-shot block was rejected); "
    "(b) the documents are long "
    "or rich, or a published bar set by a fine-tuned language model sits "
    "far above your best, telling you the text carries far more than your "
    "schema recovers; (c) you have at least a few thousand labelled rows "
    "after practice 7's replay expansion, and a GPU. Method: LoRA/QLoRA a "
    "small open instruction model scoring the label directly — strong "
    "verified-downloadable picks: Qwen/Qwen3.5-9B (best "
    "quality-per-parameter of the current small models), "
    "google/gemma-4-12B-it, the cheap-to-tune small-activation MoEs "
    "Qwen/Qwen3.6-35B-A3B and google/gemma-4-26B-A4B-it, "
    "mistralai/Ministral-3-14B-Instruct-2512; prefer a domain-pretrained "
    "variant when one exists (e.g. google/medgemma-1.5-4b-it / "
    "medgemma-27b-text-it, baichuan-inc/Baichuan-M2-32B, "
    "Intelligent-Internet/II-Medical-8B-1706 for biomedical). A 9B LoRA "
    "fits on one 40GB GPU; QLoRA and the ~3-4B-active MoEs stretch much "
    "further. For the encoder rung, ModernBERT-large, chandar-lab/NeoBERT "
    "and jhu-clsp/ettin-encoder-1b are the current strong choices. Serialize into its input BOTH "
    "the row's text AND its relational/tabular context rendered as text "
    "(history counts, cohort rates, neighbor outcomes) so the model can "
    "learn their interaction — text alone rediscovers what your features "
    "already know. Multiply sample efficiency by distilling hosted-LLM "
    "rationales: on TRAINING-period rows whose outcomes are already "
    "history, have the hosted model write the reasoning from document to "
    "known outcome, and fine-tune on rationale-then-verdict rather than "
    "bare labels. Validate on forward folds under the same leakage and "
    "two-model discipline as everything else, and enter the fine-tuned "
    "scorer as one more decorrelated family for practice 6 — gated "
    "against the frozen channel it escalates from, never assumed better.\n"
    "10. OPTIMIZE THE METRIC YOU ARE SCORED ON — FOR MAE-FAMILY METRICS "
    "THAT MEANS MEDIANS, NOT MEANS. When the evaluation is MAE/NMAE, "
    "L2-trained means target the wrong statistic: train with L1/"
    "quantile-0.5 objectives and predict conditional MEDIANS end-to-end — "
    "including through any target transform (log1p-then-expm1 preserves "
    "the median, not the mean, so it pairs correctly with L1; use it for "
    "heavy-tailed targets). For zero-inflated counts — most activity and "
    "monetary labels — model hurdle-style: a P(zero) classifier times a "
    "positive-part regressor (or a Tweedie objective), gate the zero "
    "component like any family, and remember the MAE-optimal point "
    "prediction is exactly 0 whenever P(zero) exceeds 0.5. Calibrate "
    "per-slice (practice 5) on the metric's own scale, never on a proxy.\n"
    "11. RUN A LABEL-DETERMINISM AUDIT BEFORE DEEP MODELING. Some labels "
    "are partially DETERMINED at cutoff: the database already contains "
    "rows that fix the outcome (posted results, registrations, RSVPs, "
    "schedules, completed transactions). Before building anything deep, "
    "probe every pre-cutoff table as a direct label source with cheap "
    "single-join tests — per table: join to the label rows, aggregate, "
    "measure standalone score on a sample — and treat a published method "
    "family beating the whole field by several times on one task as "
    "strong evidence such a source exists and standard featureization "
    "misses it. Signal found this way is legal by task construction: the "
    "leakage rules forbid post-cutoff reads, not reading the cutoff "
    "state well. Measured precedent: a registry snapshot-direct funnel "
    "(a few percent of rows fixed at cutoff, model fallback elsewhere) "
    "was the single largest banked gain of this campaign. Wire such "
    "sources as covered-rows-direct plus model-fallback, with coverage "
    "counts and per-slice scores logged."
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
    "- CRITICAL — DO NOT RE-TRUNCATE THE SNAPSHOT. The cache you are handed is "
    "already the correct view for this tick. Loading the database with "
    "`get_db(upto_test_timestamp=True)` clamps every tick back to the "
    "dataset's STATIC freeze and silently throws away all history between that "
    "freeze and the current tick — which, on a task whose ticks roll years "
    "past the freeze, is most of the evidence you need. Pass "
    "`upto_test_timestamp=False`, or read the cache tables directly. Note why "
    "your own measurements will NOT catch this: validation ticks fall BEFORE "
    "the freeze, so they lose nothing and score normally, while every test "
    "tick is stripped. A model crippled this way looks its best on "
    "validation. If you find your per-tick history stops growing at a fixed "
    "date, suspect this first.\n"
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

CHAMPION_NOTE_TEMPLATE = (
    "A previous campaign's winning solution for this task is staged in your "
    "shared artifact workspace at $KAPSO_SHARED_CACHE_DIR/champion/: full "
    "code (code/), design notes (solution.md, when present), and its "
    "validation metrics (champion_report.json). Its validation {metric} was "
    "{val:.4f}. Treat it as the baseline to beat, not a suggestion:\n"
    "- Reproduce it as one of your FIRST candidates. Task tables are "
    "resampled per cache build, so retrain it in this environment rather "
    "than reusing any saved outputs; expect its validation score here to "
    "land near the quoted number, and investigate before building on it if "
    "it does not.\n"
    "- Then spend the budget IMPROVING on it: add the signal it lacks, "
    "apply the practices above, and ensemble it with your own designs — a "
    "reproduced champion is a free decorrelated finalist for practice 6.\n"
    "- The champion's toolset is a floor, not a boundary: the practices "
    "above may name signal channels the champion never used (e.g. practice "
    "8's LLM extraction) — measure those against it rather than skipping "
    "them because the champion lacked them.\n"
    "- Ship nothing that scores below your reproduced champion."
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
    champion: Optional[dict] = None,
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
        "\n## Modelling practices (measured — suggestion)\n"
        + MODELLING_PRACTICE_NOTE,
        ("\n## Prior champion (provided — reproduce it, then beat it)\n"
         + CHAMPION_NOTE_TEMPLATE.format(metric=spec.primary_metric, val=champion["val"]))
        if champion else "",
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
