# Experiment History Store
#
# Stores experiment history in a JSON file, with semantic search powered by
# solution embeddings (the LLM backend's "embedding" role — provider
# credentials are read by the SDK, never by this code):
# - Write time: each record's solution text is embedded IN FULL and the
#   vector persists on the record.
# - Query time: the query is embedded with the same role and records are
#   ranked by cosine similarity.
# Without a backend (llm=None) the store is recency-only: search_similar
# degrades to get_recent_experiments — a documented capability absence,
# not an error.
#
# Features:
# - The per-experiment lesson artifact is technical_difficulties
#   (implementor-authored, fallback-generated when missing)
#
# The file is a projection of the run checkpoint's node history, so agents
# never read an experiment the campaign does not have:
# - The orchestrator (in-process) rewrites it right after every checkpoint
# - MCP server (separate process) reads it through tools, and only ever sees
#   a whole file

import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kapso.core.llm import LLMBackend
from kapso.execution.cost_estimator import (
    campaign_cost_used,
    normalize_cost_config,
    selection_score,
)
from kapso.execution.run_checkpoint import RunCheckpointStore

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 on zero norms."""
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimensions differ: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class ExperimentRecord:
    """
    Stored experiment record.

    Contains all information about a single experiment attempt,
    including the implementor's technical_difficulties report.
    """
    node_id: int
    solution: str
    score: Optional[float]
    feedback: str
    branch_name: str
    had_error: bool
    error_message: str
    timestamp: str
    # Implementor-reported (or fallback-generated) build difficulties.
    technical_difficulties: str = ""
    # Embedding of the FULL solution text (the store's llm embedding role),
    # written once, at the node's first reconcile; empty when the store had
    # no backend or the solution was blank.
    solution_embedding: List[float] = field(default_factory=list)
    # Observational caller-owned metrics. Internal ``score`` remains the search
    # and ranking signal used by Kapso.
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: Optional[str] = None
    external_evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    external_evaluation_error: str = ""
    evaluation_valid: bool = True
    evaluation_provenance: str = "agent_generated"
    evaluation_integrity_error: str = ""
    estimated_cost: Optional[Dict[str, Any]] = None
    measured_cost: Optional[Dict[str, Any]] = None
    cost_refusal: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Format for display."""
        if self.had_error:
            return f"Experiment {self.node_id} FAILED: {self.error_message[:100]}"
        return f"Experiment {self.node_id} (score={self.score}): {self.solution[:200]}..."



class ExperimentHistoryStore:
    """
    Store for experiment history.

    Provides:
    - reconcile(): Make the store say what the checkpoint's nodes say
    - get_top_experiments(): Get best experiments by score
    - get_recent_experiments(): Get most recent experiments
    - search_similar(): Semantic search over solution embeddings

    Storage: a JSON file — records carry their own solution embeddings, so
    the file is the single durable artifact (no external vector service).
    """

    def __init__(
        self,
        json_path: str,
        goal: Optional[str] = None,
        llm: Optional[LLMBackend] = None,
        cost_config: Optional[Mapping[str, Any]] = None,
        maximize_scoring: bool = True,
    ):
        """
        Initialize experiment history store.

        Args:
            json_path: Path to JSON file for persistence
            goal: Goal description (kept for context/rendering)
            llm: Backend whose "embedding" role powers write-time solution
                 embeddings and query-time semantic search. None → the
                 store is recency-only (search_similar returns recent).
            cost_config: Platform cost policy the ranking uses. The MCP
                 process takes it from the campaign checkpoint, see
                 open_campaign_history.
            maximize_scoring: Problem direction used by cost-aware ranking.
        """
        self.json_path = json_path
        self.goal = goal
        self._llm = llm
        self.cost_config = normalize_cost_config(cost_config)
        if not isinstance(maximize_scoring, bool):
            raise ValueError("Experiment history maximize_scoring must be a boolean")
        self.maximize_scoring = maximize_scoring
        self.experiments: List[ExperimentRecord] = []
        self._load_from_json()

    def reconcile(self, nodes: Sequence[Any]) -> None:
        """
        Make the store hold exactly these nodes, and persist it.

        ``nodes`` are the finalized nodes of the run checkpoint, the one
        record of what the campaign has. A record for any other node is a
        round that died before its checkpoint and is gone. A node's score and
        verdict follow the node, so a re-measurement or an evaluator
        transition reaches the agents. A record keeps its timestamp and, for
        an unchanged solution, its embedding, so this costs nothing when
        nothing is new.

        Args:
            nodes: The checkpoint's finalized SearchNodes, in history order
        """
        known = {(record.node_id, record.solution): record for record in self.experiments}
        records = []
        for node in nodes:
            record = self._record_of(node)
            earlier = known.get((record.node_id, record.solution))
            if earlier is not None:
                record.timestamp = earlier.timestamp
                record.solution_embedding = earlier.solution_embedding
            else:
                print(f"[ExperimentHistoryStore] Added experiment {record.node_id} (score={record.score})")
            # Embed the full solution BEFORE persisting so the saved record is
            # complete. A blank solution has nothing to embed.
            if (
                self._llm is not None
                and record.solution.strip()
                and not record.solution_embedding
            ):
                record.solution_embedding = self._llm.create_embedding(
                    record.solution
                )
            records.append(record)
        if records != self.experiments:
            self.experiments = records
            self._save_to_json()

    @staticmethod
    def _record_of(node: Any) -> ExperimentRecord:
        """The record for one node, carrying its technical_difficulties
        verbatim; the caller supplies the embedding."""
        raw_metrics = getattr(node, "metrics", {})
        metrics = (
            dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
        )
        raw_metadata = getattr(node, "external_evaluation_metadata", {})
        metadata = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        primary_metric = getattr(node, "primary_metric", None)
        if not isinstance(primary_metric, str):
            primary_metric = None
        external_evaluation_error = getattr(
            node,
            "external_evaluation_error",
            "",
        )
        if not isinstance(external_evaluation_error, str):
            external_evaluation_error = ""
        evaluation_valid = getattr(node, "evaluation_valid", True)
        if not isinstance(evaluation_valid, bool):
            evaluation_valid = True
        evaluation_provenance = getattr(
            node,
            "evaluation_provenance",
            "agent_generated",
        )
        if evaluation_provenance not in {"provided", "agent_generated"}:
            evaluation_provenance = "agent_generated"
        evaluation_integrity_error = getattr(
            node,
            "evaluation_integrity_error",
            "",
        )
        if not isinstance(evaluation_integrity_error, str):
            evaluation_integrity_error = ""

        return ExperimentRecord(
            node_id=node.node_id,
            solution=node.solution or "",
            score=node.score,
            feedback=node.feedback or "",
            branch_name=node.branch_name or "",
            had_error=node.had_error,
            error_message=node.error_message or "",
            technical_difficulties=getattr(node, "technical_difficulties", "") or "",
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            primary_metric=primary_metric,
            external_evaluation_metadata=metadata,
            external_evaluation_error=external_evaluation_error,
            evaluation_valid=evaluation_valid,
            evaluation_provenance=evaluation_provenance,
            evaluation_integrity_error=evaluation_integrity_error,
            estimated_cost=getattr(node, "estimated_cost", None),
            measured_cost=getattr(node, "measured_cost", None),
            cost_refusal=getattr(node, "cost_refusal", None),
        )

    def get_top_experiments(self, k: int = 5) -> List[ExperimentRecord]:
        """
        Get top k experiments by score.

        Args:
            k: Number of experiments to return

        Returns:
            List of experiments sorted by score (best first)
        """
        valid = [
            e
            for e in self.experiments
            if not e.had_error
            and e.evaluation_valid
            and e.score is not None
        ]
        ranking_config = {
            "cost": self.cost_config,
            "maximize_scoring": (
                self.maximize_scoring if self.cost_config["enabled"] else True
            ),
        }
        return sorted(
            valid,
            key=lambda record: selection_score(record, ranking_config),
            reverse=True,
        )[:k]

    def format_experiments(self, experiments: List[ExperimentRecord]) -> str:
        """Render a subset with the full campaign's cost accounting."""
        return format_experiments(
            experiments,
            cost_config=self.cost_config,
            budget_used=campaign_cost_used(self.experiments, self.cost_config),
        )

    def get_recent_experiments(self, k: int = 5) -> List[ExperimentRecord]:
        """
        Get most recent k experiments.

        Args:
            k: Number of experiments to return

        Returns:
            List of experiments in chronological order (most recent last)
        """
        return self.experiments[-k:]

    def search_similar(self, query: str, k: int = 3) -> List[ExperimentRecord]:
        """
        Semantic search over stored solution embeddings.

        The query is embedded with the same role that embedded the
        solutions; records are ranked by cosine similarity. Records
        without a vector (written before embeddings existed, or blank
        solutions) cannot be ranked and are excluded.

        Without a configured backend, or with no embedded records yet,
        degrades to get_recent_experiments (documented capability
        absence).

        Args:
            query: Search query (description of approach or problem)
            k: Number of results to return

        Returns:
            List of similar experiments (most similar first)
        """
        embedded = [e for e in self.experiments if e.solution_embedding]
        if self._llm is None or not embedded:
            print(
                "[ExperimentHistoryStore] Semantic search unavailable "
                "(no embedding backend or no embedded records); "
                "returning recent experiments"
            )
            return self.get_recent_experiments(k)

        query_embedding = self._llm.create_embedding(query)
        ranked = sorted(
            embedded,
            key=lambda e: cosine_similarity(
                query_embedding, e.solution_embedding
            ),
            reverse=True,
        )
        return ranked[:k]

    def get_experiment_count(self) -> int:
        """Get total number of experiments."""
        return len(self.experiments)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _load_from_json(self) -> None:
        """Load experiments from the JSON file.

        A missing file is the documented empty store; a corrupt file
        raises (never silently resets history).
        """
        if not os.path.exists(self.json_path):
            return
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self.experiments = []
        for e in data:
            record = ExperimentRecord(
                node_id=e.get("node_id", 0),
                solution=e.get("solution", ""),
                score=e.get("score"),
                feedback=e.get("feedback", ""),
                branch_name=e.get("branch_name", ""),
                had_error=e.get("had_error", False),
                error_message=e.get("error_message", ""),
                technical_difficulties=e.get(
                    "technical_difficulties", ""
                ),
                timestamp=e.get("timestamp", ""),
                solution_embedding=e.get("solution_embedding", []),
                metrics=e.get("metrics", {}),
                primary_metric=e.get("primary_metric"),
                external_evaluation_metadata=e.get(
                    "external_evaluation_metadata", {}
                ),
                external_evaluation_error=e.get(
                    "external_evaluation_error", ""
                ),
                evaluation_valid=e.get(
                    "evaluation_valid", True
                ),
                evaluation_provenance=e.get(
                    "evaluation_provenance",
                    "agent_generated",
                ),
                evaluation_integrity_error=e.get(
                    "evaluation_integrity_error", ""
                ),
                estimated_cost=e.get("estimated_cost"),
                measured_cost=e.get("measured_cost"),
                cost_refusal=e.get("cost_refusal"),
            )
            self.experiments.append(record)
        print(f"[ExperimentHistoryStore] Loaded {len(self.experiments)} experiments from {self.json_path}")

    def _save_to_json(self) -> None:
        """Replace the JSON file atomically (failures propagate). A corrupt
        file raises on load, so a half-written one would keep the campaign
        from resuming and fail any session reading it meanwhile."""
        directory = os.path.dirname(self.json_path)
        os.makedirs(directory, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            dir=directory, prefix=".experiment_history.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w") as handle:
            json.dump([asdict(e) for e in self.experiments], handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.json_path)


# =============================================================================
# Standalone Functions for MCP Server
# =============================================================================

def open_campaign_history(
    json_path: str, *, goal: Optional[str], llm: Optional[LLMBackend]
) -> ExperimentHistoryStore:
    """Open a campaign's history ranked the way the campaign ranks it.

    The history file and the run checkpoint share the campaign's ``.kapso``
    directory, so the checkpoint supplies the cost policy and the problem's
    scoring direction. A history opened outside a campaign, with no
    checkpoint beside it, gets the disabled default policy; a corrupt
    checkpoint raises.
    """
    checkpoint_store = RunCheckpointStore(str(Path(json_path).parent.parent))
    if not checkpoint_store.exists():
        return ExperimentHistoryStore(json_path=json_path, goal=goal, llm=llm)
    checkpoint = checkpoint_store.load()
    return ExperimentHistoryStore(
        json_path=json_path,
        goal=goal,
        llm=llm,
        cost_config=checkpoint.cost_config,
        maximize_scoring=checkpoint.maximize_scoring,
    )


def load_store_from_env() -> ExperimentHistoryStore:
    """
    Open the campaign's experiment store from environment variables.

    Used by the MCP server (env is the transport across the server-process
    boundary; see get_mcp_config). The cost policy and the scoring
    direction come from the campaign checkpoint, see open_campaign_history.

    Environment variables:
    - EXPERIMENT_HISTORY_PATH: Path to JSON file (required)
    - EXPERIMENT_EMBEDDING_MODEL: Embedding model for semantic search
      (optional; absent → recency-only store)
    - EXPERIMENT_GOAL: Goal description (optional)
    """
    json_path = os.environ.get("EXPERIMENT_HISTORY_PATH", ".kapso/experiment_history.json")
    embedding_model = os.environ.get("EXPERIMENT_EMBEDDING_MODEL")
    llm = (
        LLMBackend(models={"embedding": embedding_model})
        if embedding_model
        else None
    )
    return open_campaign_history(
        json_path, goal=os.environ.get("EXPERIMENT_GOAL"), llm=llm
    )


def format_experiments(
    experiments: List[ExperimentRecord],
    cost_config: Optional[Mapping[str, Any]] = None,
    budget_used: Optional[float] = None,
) -> str:
    """
    Format experiments as markdown for agent consumption.

    Args:
        experiments: List of experiment records

    Returns:
        Formatted markdown string
    """
    lines = []
    show_cost = cost_config is not None and cost_config["enabled"]
    if show_cost:
        if budget_used is None:
            raise ValueError("Cost history requires full campaign budget_used")
        budget = cost_config["campaign_budget"]
        cap = f" / {budget}" if budget is not None else ""
        lines.append(
            f"Campaign budget used: {budget_used}{cap} {cost_config['unit']}\n"
        )
    if not experiments:
        lines.append("No experiments found.")
        return "\n".join(lines)
    for exp in experiments:
        if not exp.evaluation_valid:
            status = "INVALID EVALUATION"
        else:
            status = f"score={exp.score}"

        # Full content, never clipped: agent-consumed via the MCP tools.
        lines.append(f"""
## Experiment {exp.node_id} ({status})

**Solution:**
{exp.solution}

**Feedback:**
{exp.feedback}""")

        if show_cost:
            for cost_field in ("estimated_cost", "measured_cost"):
                lines.append(
                    f"\n**{cost_field}:** {json.dumps(getattr(exp, cost_field))}"
                )
            if exp.cost_refusal is not None:
                lines.append(f"\n**cost_refusal:** {json.dumps(exp.cost_refusal)}")

        if exp.technical_difficulties:
            lines.append(f"""
**Technical difficulties:**
{exp.technical_difficulties}""")

    return "\n".join(lines)
