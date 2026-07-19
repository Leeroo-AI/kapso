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
# The store is designed to be accessed by both:
# - The orchestrator (in-process, for adding experiments)
# - MCP server (separate process, for querying via tools)

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from kapso.core.llm import LLMBackend

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
    # written at add time; empty when the store had no backend or the
    # solution was blank.
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

    def __str__(self) -> str:
        """Format for display."""
        if self.had_error:
            return f"Experiment {self.node_id} FAILED: {self.error_message[:100]}"
        return f"Experiment {self.node_id} (score={self.score}): {self.solution[:200]}..."



class ExperimentHistoryStore:
    """
    Store for experiment history.

    Provides:
    - add_experiment(): Add new experiment result (embeds its solution)
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
    ):
        """
        Initialize experiment history store.

        Args:
            json_path: Path to JSON file for persistence
            goal: Goal description (kept for context/rendering)
            llm: Backend whose "embedding" role powers write-time solution
                 embeddings and query-time semantic search. None → the
                 store is recency-only (search_similar returns recent).
        """
        self.json_path = json_path
        self.goal = goal
        self._llm = llm
        self.experiments: List[ExperimentRecord] = []
        self._load_from_json()

    def add_experiment(self, node: Any) -> None:
        """
        Add experiment to the store and persist it.

        The record carries the node's technical_difficulties verbatim, and
        — when a backend is configured — an embedding of its full solution
        text, computed before the record is persisted so the JSON is the
        complete durable artifact.

        Args:
            node: SearchNode with experiment results
        """
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

        record = ExperimentRecord(
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
        )

        # Embed the full solution BEFORE persisting so the saved record is
        # complete. A blank solution has nothing to embed.
        if self._llm is not None and record.solution.strip():
            record.solution_embedding = self._llm.create_embedding(
                record.solution
            )

        # Add to in-memory list
        self.experiments.append(record)

        # Persist to JSON
        self._save_to_json()

        print(f"[ExperimentHistoryStore] Added experiment {record.node_id} (score={record.score})")

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
        return sorted(valid, key=lambda x: x.score or 0, reverse=True)[:k]

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
            )
            self.experiments.append(record)
        print(f"[ExperimentHistoryStore] Loaded {len(self.experiments)} experiments from {self.json_path}")

    def _save_to_json(self) -> None:
        """Save experiments to the JSON file (failures propagate)."""
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, 'w') as f:
            json.dump([asdict(e) for e in self.experiments], f, indent=2)


# =============================================================================
# Standalone Functions for MCP Server
# =============================================================================

def load_store_from_env() -> ExperimentHistoryStore:
    """
    Load experiment store from environment variables.

    Used by MCP server to access the store (env is the transport across
    the server-process boundary; see get_mcp_config).

    Environment variables:
    - EXPERIMENT_HISTORY_PATH: Path to JSON file (required)
    - EXPERIMENT_EMBEDDING_MODEL: Embedding model for semantic search
      (optional; absent → recency-only store)
    - EXPERIMENT_GOAL: Goal description (optional)
    """
    json_path = os.environ.get("EXPERIMENT_HISTORY_PATH", ".kapso/experiment_history.json")
    embedding_model = os.environ.get("EXPERIMENT_EMBEDDING_MODEL")
    goal = os.environ.get("EXPERIMENT_GOAL")

    llm = (
        LLMBackend(models={"embedding": embedding_model})
        if embedding_model
        else None
    )

    return ExperimentHistoryStore(
        json_path=json_path,
        goal=goal,
        llm=llm,
    )


def format_experiments(experiments: List[ExperimentRecord]) -> str:
    """
    Format experiments as markdown for agent consumption.

    Args:
        experiments: List of experiment records

    Returns:
        Formatted markdown string
    """
    if not experiments:
        return "No experiments found."

    lines = []
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

        if exp.technical_difficulties:
            lines.append(f"""
**Technical difficulties:**
{exp.technical_difficulties}""")

    return "\n".join(lines)
