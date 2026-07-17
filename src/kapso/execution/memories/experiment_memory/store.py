# Experiment History Store
#
# Stores experiment history with dual storage:
# - JSON file for basic retrieval (top, recent)
# - Weaviate for semantic search (optional)
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

# Weaviate imports (optional - graceful fallback if not available)
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    Store for experiment history with dual storage.
    
    Provides:
    - add_experiment(): Add new experiment result
    - get_top_experiments(): Get best experiments by score
    - get_recent_experiments(): Get most recent experiments
    - search_similar(): Semantic search for similar experiments (via Weaviate)
    
    Storage:
    - JSON file: Always used, provides persistence and basic retrieval
    - Weaviate: Optional, provides semantic search capability
    
    """
    
    WEAVIATE_COLLECTION = "ExperimentHistory"
    DUPLICATE_THRESHOLD = 0.95  # Cosine similarity threshold for duplicate detection
    
    def __init__(
        self, 
        json_path: str,
        weaviate_url: Optional[str] = None,
        goal: Optional[str] = None,
        llm: Any = None,
    ):
        """
        Initialize experiment history store.
        
        Args:
            json_path: Path to JSON file for persistence
            weaviate_url: Optional Weaviate URL for semantic search
            goal: Goal description (kept for context/rendering)
            llm: Shared configured backend (reserved for future use)
        """
        self.json_path = json_path
        self.goal = goal
        self._llm = llm
        self.experiments: List[ExperimentRecord] = []
        
        # Connect to Weaviate if available
        self.weaviate = None
        if weaviate_url and WEAVIATE_AVAILABLE:
            try:
                self.weaviate = weaviate.connect_to_local(
                    host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                    port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080,
                )
                self._ensure_weaviate_collection()
                print(f"[ExperimentHistoryStore] Connected to Weaviate at {weaviate_url}")
            except Exception as e:
                print(f"[ExperimentHistoryStore] Warning: Could not connect to Weaviate: {e}")
                self.weaviate = None
        
        # Load existing experiments from JSON
        self._load_from_json()
    
    def add_experiment(self, node: Any) -> None:
        """
        Add experiment to both JSON and Weaviate.
        
        The record carries the node's technical_difficulties verbatim.
        
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
        
        # Add to in-memory list
        self.experiments.append(record)
        
        # Persist to JSON
        self._save_to_json()
        
        # Index in Weaviate (for semantic search)
        if self.weaviate and record.evaluation_valid:
            self._index_in_weaviate(record)
        
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
        Semantic search for similar experiments via Weaviate.
        
        Args:
            query: Search query (description of approach or problem)
            k: Number of results to return
            
        Returns:
            List of similar experiments
        """
        if not self.weaviate:
            # Fallback: return recent if no Weaviate
            print("[ExperimentHistoryStore] Weaviate not available, falling back to recent experiments")
            return self.get_recent_experiments(k)
        
        try:
            collection = self.weaviate.collections.get(self.WEAVIATE_COLLECTION)
            results = collection.query.near_text(
                query=query,
                limit=k,
            )
            
            # Convert Weaviate objects to ExperimentRecord
            records = []
            for obj in results.objects:
                props = obj.properties
                records.append(ExperimentRecord(
                    node_id=props.get("node_id", 0),
                    solution=props.get("solution", ""),
                    score=props.get("score"),
                    feedback=props.get("feedback", ""),
                    branch_name=props.get("branch_name", ""),
                    had_error=props.get("had_error", False),
                    error_message=props.get("error_message", ""),
                    technical_difficulties=props.get("technical_difficulties", ""),
                    timestamp=props.get("timestamp", ""),
                    metrics={},
                    primary_metric=None,
                    external_evaluation_metadata={},
                    external_evaluation_error="",
                    evaluation_valid=True,
                    evaluation_provenance="agent_generated",
                    evaluation_integrity_error="",
                ))
            return records
            
        except Exception as e:
            print(f"[ExperimentHistoryStore] Weaviate search failed: {e}")
            return self.get_recent_experiments(k)
    
    def get_experiment_count(self) -> int:
        """Get total number of experiments."""
        return len(self.experiments)
    
    def close(self) -> None:
        """Close connections."""
        if self.weaviate:
            try:
                self.weaviate.close()
            except Exception:
                pass
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _load_from_json(self) -> None:
        """Load experiments from JSON file."""
        if os.path.exists(self.json_path):
            try:
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
            except Exception as e:
                print(f"[ExperimentHistoryStore] Warning: Could not load from JSON: {e}")
                self.experiments = []
    
    def _save_to_json(self) -> None:
        """Save experiments to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, 'w') as f:
                json.dump([asdict(e) for e in self.experiments], f, indent=2)
        except Exception as e:
            print(f"[ExperimentHistoryStore] Warning: Could not save to JSON: {e}")
    
    def _ensure_weaviate_collection(self) -> None:
        """Create Weaviate collection if it doesn't exist."""
        if not self.weaviate:
            return
        
        try:
            if not self.weaviate.collections.exists(self.WEAVIATE_COLLECTION):
                self.weaviate.collections.create(
                    name=self.WEAVIATE_COLLECTION,
                    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                    properties=[
                        Property(name="node_id", data_type=DataType.INT),
                        Property(name="solution", data_type=DataType.TEXT),
                        Property(name="score", data_type=DataType.NUMBER),
                        Property(name="feedback", data_type=DataType.TEXT),
                        Property(name="branch_name", data_type=DataType.TEXT),
                        Property(name="had_error", data_type=DataType.BOOL),
                        Property(name="error_message", data_type=DataType.TEXT),
                        Property(name="timestamp", data_type=DataType.TEXT),
                        Property(name="text", data_type=DataType.TEXT),  # Vectorized field
                        Property(name="technical_difficulties", data_type=DataType.TEXT),
                    ]
                )
                print(f"[ExperimentHistoryStore] Created Weaviate collection: {self.WEAVIATE_COLLECTION}")
        except Exception as e:
            print(f"[ExperimentHistoryStore] Warning: Could not create Weaviate collection: {e}")
    
    def _index_in_weaviate(self, record: ExperimentRecord) -> None:
        """Index experiment in Weaviate for semantic search."""
        if not self.weaviate:
            return
        
        try:
            collection = self.weaviate.collections.get(self.WEAVIATE_COLLECTION)
            
            # Text to embed: solution + feedback + difficulties
            text_parts = [f"Solution: {record.solution}", f"Feedback: {record.feedback}"]
            if record.technical_difficulties:
                text_parts.append(
                    f"Difficulties: {record.technical_difficulties}"
                )
            text_for_embedding = "\n".join(text_parts)
            
            collection.data.insert({
                "node_id": record.node_id,
                "solution": record.solution,
                "score": record.score,
                "feedback": record.feedback,
                "branch_name": record.branch_name,
                "had_error": record.had_error,
                "error_message": record.error_message,
                "timestamp": record.timestamp,
                "text": text_for_embedding,
                "technical_difficulties": record.technical_difficulties,
            })
        except Exception as e:
            print(f"[ExperimentHistoryStore] Warning: Could not index in Weaviate: {e}")


# =============================================================================
# Standalone Functions for MCP Server
# =============================================================================

def load_store_from_env() -> ExperimentHistoryStore:
    """
    Load experiment store from environment variables.
    
    Used by MCP server to access the store.
    
    Environment variables:
    - EXPERIMENT_HISTORY_PATH: Path to JSON file (required)
    - WEAVIATE_URL: Weaviate URL (optional)
    - EXPERIMENT_GOAL: Goal description (optional)
    """
    json_path = os.environ.get("EXPERIMENT_HISTORY_PATH", ".kapso/experiment_history.json")
    weaviate_url = os.environ.get("WEAVIATE_URL")
    goal = os.environ.get("EXPERIMENT_GOAL")
    
    return ExperimentHistoryStore(
        json_path=json_path, 
        weaviate_url=weaviate_url,
        goal=goal,
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
        elif exp.had_error:
            status = f"FAILED: {exp.error_message[:100]}"
        else:
            status = f"score={exp.score}"
        
        lines.append(f"""
## Experiment {exp.node_id} ({status})

**Solution:**
{exp.solution[:500]}{'...' if len(exp.solution) > 500 else ''}

**Feedback:**
{exp.feedback[:300]}{'...' if len(exp.feedback) > 300 else ''}""")

        if exp.technical_difficulties:
            lines.append(f"""
**Technical difficulties:**
{exp.technical_difficulties[:300]}{'...' if len(exp.technical_difficulties) > 300 else ''}""")
    
    return "\n".join(lines)
