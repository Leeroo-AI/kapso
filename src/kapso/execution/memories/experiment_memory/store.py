# Experiment History Store
#
# Stores experiment history with dual storage:
# - JSON file for basic retrieval (top, recent)
# - Weaviate for semantic search (optional)
#
# Features:
# - LLM-based insight extraction from errors and successes
# - Duplicate detection to prevent storing redundant insights
# - Confidence-based filtering for high-quality insights
#
# The store is designed to be accessed by both:
# - The orchestrator (in-process, for adding experiments)
# - MCP server (separate process, for querying via tools)

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kapso.execution.memories.experiment_memory.insight_extractor import InsightExtractor

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
    including extracted insights for learning.
    """
    node_id: int
    solution: str
    score: Optional[float]
    feedback: str
    branch_name: str
    had_error: bool
    error_message: str
    timestamp: str
    # Insight fields (optional, extracted by LLM)
    insight: Optional[str] = None
    insight_type: Optional[str] = None  # "critical_error" or "best_practice"
    insight_confidence: Optional[float] = None
    insight_tags: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Format for display."""
        if self.had_error:
            return f"Experiment {self.node_id} FAILED: {self.error_message[:100]}"
        return f"Experiment {self.node_id} (score={self.score}): {self.solution[:200]}..."
    
    def has_insight(self) -> bool:
        """Check if this record has an extracted insight."""
        return self.insight is not None and len(self.insight) > 0


class ExperimentHistoryStore:
    """
    Store for experiment history with dual storage and insight extraction.
    
    Provides:
    - add_experiment(): Add new experiment result with automatic insight extraction
    - get_top_experiments(): Get best experiments by score
    - get_recent_experiments(): Get most recent experiments
    - get_experiments_with_insights(): Get experiments that have extracted insights
    - search_similar(): Semantic search for similar experiments (via Weaviate)
    
    Storage:
    - JSON file: Always used, provides persistence and basic retrieval
    - Weaviate: Optional, provides semantic search capability
    
    Insight Extraction:
    - Errors are generalized into reusable lessons
    - High-scoring successes are extracted as best practices
    - Duplicate insights are detected and skipped
    """
    
    WEAVIATE_COLLECTION = "ExperimentHistory"
    DUPLICATE_THRESHOLD = 0.95  # Cosine similarity threshold for duplicate detection
    SUCCESS_SCORE_THRESHOLD = 0.7  # Minimum score to extract success insight
    
    def __init__(
        self, 
        json_path: str,
        weaviate_url: Optional[str] = None,
        goal: Optional[str] = None,
        enable_insights: bool = True,
    ):
        """
        Initialize experiment history store.
        
        Args:
            json_path: Path to JSON file for persistence
            weaviate_url: Optional Weaviate URL for semantic search
            goal: Goal description (used for insight extraction context)
            enable_insights: Whether to extract insights from experiments
        """
        self.json_path = json_path
        self.goal = goal
        self.enable_insights = enable_insights
        self.experiments: List[ExperimentRecord] = []
        
        # Lazy-loaded insight extractor
        self._insight_extractor: Optional["InsightExtractor"] = None
        
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
    
    def _get_insight_extractor(self) -> "InsightExtractor":
        """Lazy-load insight extractor."""
        if self._insight_extractor is None:
            from kapso.execution.memories.experiment_memory.insight_extractor import InsightExtractor
            self._insight_extractor = InsightExtractor()
        return self._insight_extractor
    
    def add_experiment(self, node: Any) -> None:
        """
        Add experiment to both JSON and Weaviate.
        
        Automatically extracts insights from errors and high-scoring successes.
        
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
        
        # Extract insight if enabled
        if self.enable_insights:
            self._extract_and_attach_insight(record)
        
        # Check for duplicate insight before adding
        if record.insight and self._is_duplicate_insight(record.insight):
            logger.info(f"[ExperimentHistoryStore] Skipping duplicate insight for experiment {record.node_id}")
            record.insight = None
            record.insight_type = None
            record.insight_confidence = None
            record.insight_tags = []
        
        # Add to in-memory list
        self.experiments.append(record)
        
        # Persist to JSON
        self._save_to_json()
        
        # Index in Weaviate (for semantic search)
        if self.weaviate:
            self._index_in_weaviate(record)
        
        insight_info = f", insight={record.insight_type}" if record.insight else ""
        print(f"[ExperimentHistoryStore] Added experiment {record.node_id} (score={record.score}{insight_info})")
    
    def _extract_and_attach_insight(self, record: ExperimentRecord) -> None:
        """Extract insight from experiment and attach to record."""
        try:
            extractor = self._get_insight_extractor()
            goal = self.goal or "Unknown goal"
            
            if record.had_error and record.error_message:
                # Extract error insight
                insight = extractor.extract_from_error(
                    error_message=record.error_message,
                    goal=goal,
                    solution=record.solution,
                )
                record.insight = insight.to_formatted_string()
                record.insight_type = insight.insight_type.value
                record.insight_confidence = insight.confidence
                record.insight_tags = insight.tags
                logger.info(f"[ExperimentHistoryStore] Extracted error insight: {insight.lesson[:100]}...")
                
            elif record.score is not None and record.score >= self.SUCCESS_SCORE_THRESHOLD:
                # Extract success insight for high-scoring experiments
                insight = extractor.extract_from_success(
                    feedback=record.feedback,
                    score=record.score,
                    goal=goal,
                    solution=record.solution,
                )
                record.insight = insight.to_formatted_string()
                record.insight_type = insight.insight_type.value
                record.insight_confidence = insight.confidence
                record.insight_tags = insight.tags
                logger.info(f"[ExperimentHistoryStore] Extracted success insight: {insight.lesson[:100]}...")
                
        except Exception as e:
            logger.warning(f"[ExperimentHistoryStore] Insight extraction failed: {e}")
    
    def _is_duplicate_insight(self, insight_text: str) -> bool:
        """
        Check if similar insight already exists.
        
        Uses Weaviate for semantic similarity if available,
        falls back to exact string match.
        """
        if not insight_text:
            return False
        
        # Check exact match in local list
        for exp in self.experiments:
            if exp.insight and exp.insight == insight_text:
                return True
        
        # TODO: Add Weaviate semantic similarity check when needed
        # For now, exact match is sufficient
        
        return False
    
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
    
    def get_experiments_with_insights(
        self, 
        k: int = 10,
        insight_type: Optional[str] = None,
    ) -> List[ExperimentRecord]:
        """
        Get experiments that have extracted insights.
        
        Args:
            k: Maximum number of experiments to return
            insight_type: Filter by insight type ("critical_error" or "best_practice")
            
        Returns:
            List of experiments with insights, sorted by confidence
        """
        with_insights = [e for e in self.experiments if e.has_insight()]
        
        if insight_type:
            with_insights = [e for e in with_insights if e.insight_type == insight_type]
        
        # Sort by confidence (highest first)
        with_insights.sort(key=lambda x: x.insight_confidence or 0, reverse=True)
        
        return with_insights[:k]
    
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
                    insight=props.get("insight"),
                    insight_type=props.get("insight_type"),
                    insight_confidence=props.get("insight_confidence"),
                    insight_tags=props.get("insight_tags", []),
                ))
            return records
            
        except Exception as e:
            print(f"[ExperimentHistoryStore] Weaviate search failed: {e}")
            return self.get_recent_experiments(k)
    
    def get_experiment_count(self) -> int:
        """Get total number of experiments."""
        return len(self.experiments)
    
    def get_insight_count(self) -> int:
        """Get number of experiments with insights."""
        return len([e for e in self.experiments if e.has_insight()])
    
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
                        # Handle backward compatibility (old records without insight fields)
                        record = ExperimentRecord(
                            node_id=e.get("node_id", 0),
                            solution=e.get("solution", ""),
                            score=e.get("score"),
                            feedback=e.get("feedback", ""),
                            branch_name=e.get("branch_name", ""),
                            had_error=e.get("had_error", False),
                            error_message=e.get("error_message", ""),
                            timestamp=e.get("timestamp", ""),
                            insight=e.get("insight"),
                            insight_type=e.get("insight_type"),
                            insight_confidence=e.get("insight_confidence"),
                            insight_tags=e.get("insight_tags", []),
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
                        # Insight fields
                        Property(name="insight", data_type=DataType.TEXT),
                        Property(name="insight_type", data_type=DataType.TEXT),
                        Property(name="insight_confidence", data_type=DataType.NUMBER),
                        Property(name="insight_tags", data_type=DataType.TEXT_ARRAY),
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
            
            # Text to embed: solution + feedback + insight
            text_parts = [f"Solution: {record.solution}", f"Feedback: {record.feedback}"]
            if record.insight:
                text_parts.append(f"Insight: {record.insight}")
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
                "insight": record.insight,
                "insight_type": record.insight_type,
                "insight_confidence": record.insight_confidence,
                "insight_tags": record.insight_tags,
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
        enable_insights=False,  # MCP server is read-only, don't extract insights
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
        if exp.had_error:
            status = f"FAILED: {exp.error_message[:100]}"
        else:
            status = f"score={exp.score}"
        
        lines.append(f"""
## Experiment {exp.node_id} ({status})

**Solution:**
{exp.solution[:500]}{'...' if len(exp.solution) > 500 else ''}

**Feedback:**
{exp.feedback[:300]}{'...' if len(exp.feedback) > 300 else ''}""")
        
        # Include insight if available
        if exp.insight:
            lines.append(f"""
**Insight ({exp.insight_type}, confidence={exp.insight_confidence:.2f}):**
{exp.insight}""")
    
    return "\n".join(lines)
