# Experiment History Store
#
# Stores experiment history with dual storage:
# - JSON file for basic retrieval (top, recent)
# - Weaviate for semantic search (optional)
#
# The store is designed to be accessed by both:
# - The orchestrator (in-process, for adding experiments)
# - MCP server (separate process, for querying via tools)

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# Weaviate imports (optional - graceful fallback if not available)
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


@dataclass
class ExperimentRecord:
    """
    Stored experiment record.
    
    Contains all information about a single experiment attempt.
    """
    node_id: int
    solution: str
    score: Optional[float]
    feedback: str
    branch_name: str
    had_error: bool
    error_message: str
    timestamp: str
    
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
    
    def __init__(
        self, 
        json_path: str,
        weaviate_url: Optional[str] = None,
    ):
        """
        Initialize experiment history store.
        
        Args:
            json_path: Path to JSON file for persistence
            weaviate_url: Optional Weaviate URL for semantic search
        """
        self.json_path = json_path
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
        
        Args:
            node: SearchNode with experiment results
        """
        record = ExperimentRecord(
            node_id=node.node_id,
            solution=node.solution or "",
            score=node.score,
            feedback=node.feedback or "",
            branch_name=node.branch_name or "",
            had_error=node.had_error,
            error_message=node.error_message or "",
            timestamp=datetime.now().isoformat(),
        )
        
        # Add to in-memory list
        self.experiments.append(record)
        
        # Persist to JSON
        self._save_to_json()
        
        # Index in Weaviate (for semantic search)
        if self.weaviate:
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
        valid = [e for e in self.experiments if not e.had_error and e.score is not None]
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
                    timestamp=props.get("timestamp", ""),
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
                    self.experiments = [ExperimentRecord(**e) for e in data]
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
            
            # Text to embed: solution + feedback
            text_for_embedding = f"Solution: {record.solution}\nFeedback: {record.feedback}"
            
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
    """
    json_path = os.environ.get("EXPERIMENT_HISTORY_PATH", ".kapso/experiment_history.json")
    weaviate_url = os.environ.get("WEAVIATE_URL")
    
    return ExperimentHistoryStore(json_path=json_path, weaviate_url=weaviate_url)


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
        if exp.had_error:
            status = f"FAILED: {exp.error_message[:100]}"
        else:
            status = f"score={exp.score}"
        
        lines.append(f"""
## Experiment {exp.node_id} ({status})

**Solution:**
{exp.solution[:500]}{'...' if len(exp.solution) > 500 else ''}

**Feedback:**
{exp.feedback[:300]}{'...' if len(exp.feedback) > 300 else ''}
""")
    
    return "\n".join(lines)
