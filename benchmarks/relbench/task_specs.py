"""Per-task metadata for the RelBench integration.

Most metadata is derived live from the relbench task object; the static maps
below only carry information the package does not expose (leaderboard-primary
metrics, runtime tiers, domain notes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Task families -------------------------------------------------------------

ENTITY_BINARY = "entity_binary_classification"
ENTITY_MULTICLASS = "entity_multiclass_classification"
ENTITY_REGRESSION = "entity_regression"
RECOMMENDATION = "recommendation"
AUTOCOMPLETE_BINARY = "autocomplete_binary_classification"
AUTOCOMPLETE_MULTICLASS = "autocomplete_multiclass_classification"
AUTOCOMPLETE_REGRESSION = "autocomplete_regression"

# Leaderboard-primary metric per family: (metric_fn_name, maximize).
# The RelBench leaderboard reports AUROC for binary classification, MAE for
# regression, and MAP@K for recommendation. The v2 paper additionally
# emphasizes R^2 (regression) and accuracy/MRR (multiclass autocomplete);
# task.evaluate returns all of them — these choices only drive the search.
PRIMARY_METRIC = {
    ENTITY_BINARY: ("roc_auc", True),
    ENTITY_MULTICLASS: ("accuracy", True),
    ENTITY_REGRESSION: ("mae", False),
    RECOMMENDATION: ("link_prediction_map", True),
    AUTOCOMPLETE_BINARY: ("roc_auc", True),
    AUTOCOMPLETE_MULTICLASS: ("accuracy", True),
    AUTOCOMPLETE_REGRESSION: ("mae", False),
}

# Optional per-task override: {"rel-xyz/task": ("metric", maximize)}
# Regression tasks on the official leaderboard are ranked by NMAE
# (= MAE / std(train targets)), which is per-task monotone with MAE — so MAE
# is the right search signal there. The v2 paper instead reports R^2 for the
# new regression tasks (incl. all autocomplete regression); to beat those
# published bars the search must optimize R^2 (MSE-flavored), not MAE.
PRIMARY_METRIC_OVERRIDES: Dict[str, tuple] = {
    "rel-arxiv/author-publication": ("r2", True),
    "rel-ratebeer/user-count": ("r2", True),
    "rel-amazon/review-rating": ("r2", True),
    "rel-event/users-birthyear": ("r2", True),
    "rel-f1/results-position": ("r2", True),
    "rel-f1/qualifying-position": ("r2", True),
    "rel-hm/transactions-price": ("r2", True),
    "rel-ratebeer/beer_ratings-total_score": ("r2", True),
    "rel-trial/studies-enrollment": ("r2", True),
}

# Native RelBench v2 databases (11). dbinfer-*/tgb*/ctu integrations are out
# of scope for the leaderboard runs.
NATIVE_DATASETS = [
    "rel-amazon",
    "rel-avito",
    "rel-event",
    "rel-f1",
    "rel-hm",
    "rel-stack",
    "rel-trial",
    "rel-arxiv",
    "rel-salt",
    "rel-ratebeer",
    "rel-mimic",
]

# Rough database size tiers -> default runtime budget for one full training
# run of generated code (seconds). Override per run with
# RELBENCH_FULL_TIMEOUT / RELBENCH_DEBUG_TIMEOUT.
SIZE_TIER = {
    "rel-f1": "small",
    "rel-event": "small",
    "rel-avito": "medium",
    "rel-trial": "medium",
    "rel-arxiv": "medium",
    "rel-ratebeer": "medium",
    "rel-salt": "medium",
    "rel-mimic": "medium",
    "rel-stack": "large",
    "rel-hm": "large",
    "rel-amazon": "large",
}

FULL_TIMEOUT_BY_TIER = {"small": 2 * 60 * 60, "medium": 4 * 60 * 60, "large": 8 * 60 * 60}
DEBUG_TIMEOUT_BY_TIER = {"small": 15 * 60, "medium": 20 * 60, "large": 30 * 60}

# Domain notes injected into the problem context (kept short; the heavy
# lifting is in the family playbooks in context.py).
DATASET_NOTES = {
    "rel-amazon": "Amazon product reviews (books). Tens of millions of review rows; "
    "review text/summary are strong signals where present. Users/items are long-tailed.",
    "rel-avito": "Avito classified ads: search logs, ad visits/clicks. Heavy interaction "
    "tables; CTR-style signals; many categorical columns.",
    "rel-event": "Event recommendation social data: users, events, friendships, "
    "interest responses. Strong social-graph and demographic signals.",
    "rel-f1": "Formula 1 racing history. Tiny database; entity tasks have 40 evaluation "
    "timestamps. Standings/results history is highly predictive; k-fold-style stability matters.",
    "rel-hm": "H&M fashion transactions. Strong recency/repeat-purchase structure; "
    "weekly seasonality; large transaction table.",
    "rel-stack": "Stack Exchange dump: posts, votes, badges, comments. User activity "
    "history dominates engagement/badge tasks; text columns are informative.",
    "rel-trial": "Clinical trials (AACT). Rich text (eligibility criteria, titles); "
    "sponsor/site/condition relations; labels are sparse for some tasks.",
    "rel-arxiv": "arXiv citation/authorship graph. Citation-count style targets are "
    "heavy-tailed; category prediction is text+graph.",
    "rel-salt": "SAP SALT enterprise sales orders. All tasks are autocomplete on order "
    "fields; high-cardinality categoricals; strong header/item hierarchy signals.",
    "rel-ratebeer": "RateBeer social beer-rating data. Churn/count tasks + several "
    "recommendation tasks; MRR is also reported for recs.",
    "rel-mimic": "MIMIC-IV clinical DB (credentialed PhysioNet access required; must be "
    "built locally via BigQuery). ICU length-of-stay regression.",
}


@dataclass
class TaskSpec:
    dataset_name: str
    task_name: str
    family: str
    primary_metric: str
    maximize: bool
    entity_col: Optional[str] = None
    entity_table: Optional[str] = None
    src_entity_col: Optional[str] = None
    src_entity_table: Optional[str] = None
    dst_entity_col: Optional[str] = None
    dst_entity_table: Optional[str] = None
    time_col: Optional[str] = None
    target_col: Optional[str] = None
    eval_k: Optional[int] = None
    num_classes: Optional[int] = None
    num_dst_nodes: Optional[int] = None
    timedelta_str: Optional[str] = None
    num_eval_timestamps: int = 1
    metrics: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)
    full_timeout: int = 4 * 60 * 60
    debug_timeout: int = 20 * 60

    @property
    def task_id(self) -> str:
        return f"{self.dataset_name}--{self.task_name}"

    @property
    def is_recommendation(self) -> bool:
        return self.family == RECOMMENDATION

    @property
    def is_multiclass(self) -> bool:
        return self.family in (ENTITY_MULTICLASS, AUTOCOMPLETE_MULTICLASS)

    @property
    def is_autocomplete(self) -> bool:
        return self.family.startswith("autocomplete")

    def expected_pred_shape(self, n_rows: int) -> tuple:
        if self.is_recommendation:
            return (n_rows, self.eval_k)
        if self.is_multiclass:
            return (n_rows, self.num_classes)
        return (n_rows,)


def resolve_spec(task, dataset_name: str, task_name: str) -> TaskSpec:
    """Build a TaskSpec from a live relbench task object."""
    from relbench.base import AutoCompleteTask, RecommendationTask, TaskType

    task_type = task.task_type
    if isinstance(task, AutoCompleteTask):
        family = {
            TaskType.BINARY_CLASSIFICATION: AUTOCOMPLETE_BINARY,
            TaskType.MULTICLASS_CLASSIFICATION: AUTOCOMPLETE_MULTICLASS,
            TaskType.REGRESSION: AUTOCOMPLETE_REGRESSION,
        }[task_type]
    elif isinstance(task, RecommendationTask):
        family = RECOMMENDATION
    else:
        family = {
            TaskType.BINARY_CLASSIFICATION: ENTITY_BINARY,
            TaskType.MULTICLASS_CLASSIFICATION: ENTITY_MULTICLASS,
            TaskType.REGRESSION: ENTITY_REGRESSION,
        }[task_type]

    task_id = f"{dataset_name}/{task_name}"
    primary_metric, maximize = PRIMARY_METRIC_OVERRIDES.get(
        task_id, PRIMARY_METRIC[family]
    )

    tier = SIZE_TIER.get(dataset_name, "medium")
    spec = TaskSpec(
        dataset_name=dataset_name,
        task_name=task_name,
        family=family,
        primary_metric=primary_metric,
        maximize=maximize,
        time_col=getattr(task, "time_col", None),
        target_col=getattr(task, "target_col", None),
        timedelta_str=str(getattr(task, "timedelta", None)),
        num_eval_timestamps=getattr(task, "num_eval_timestamps", 1),
        metrics=[fn.__name__ for fn in task.metrics],
        full_timeout=FULL_TIMEOUT_BY_TIER[tier],
        debug_timeout=DEBUG_TIMEOUT_BY_TIER[tier],
    )

    if family == RECOMMENDATION:
        spec.src_entity_col = task.src_entity_col
        spec.src_entity_table = task.src_entity_table
        spec.dst_entity_col = task.dst_entity_col
        spec.dst_entity_table = task.dst_entity_table
        spec.eval_k = task.eval_k
        spec.num_dst_nodes = task.num_dst_nodes
    else:
        spec.entity_col = task.entity_col
        spec.entity_table = task.entity_table
        if isinstance(task, AutoCompleteTask):
            spec.removed_columns = [
                f"{table}.{col}" for table, col in getattr(task, "remove_columns", [])
            ]
        if spec.is_multiclass:
            num_classes = getattr(task, "num_classes", None)
            if num_classes is None:
                train = task.get_table("train", mask_input_cols=False)
                num_classes = int(train.df[task.target_col].max()) + 1
            spec.num_classes = int(num_classes)

    return spec
