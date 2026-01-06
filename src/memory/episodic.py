# =============================================================================
# Episodic Memory Store
# =============================================================================
#
# Stores generalized insights from past experiments using:
# - Weaviate: Vector database for semantic search
# - JSON: File-based fallback for persistence
#
# Embeddings are generated client-side using OpenAI API to avoid
# needing the API key in the Weaviate container.
#
# Configuration: See cognitive_memory.yaml for all tunable parameters.
# =============================================================================

import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING

try:
    import weaviate
    import weaviate.classes as wvc
    HAS_WEAVIATE = True
except ImportError:
    weaviate = None
    wvc = None
    HAS_WEAVIATE = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    openai = None
    HAS_OPENAI = False

from src.memory.types import Insight, InsightType

if TYPE_CHECKING:
    from src.memory.config import EpisodicConfig

logger = logging.getLogger(__name__)


class EpisodicStore:
    """
    Stores generalized insights from past experiments.
    
    Uses Weaviate for semantic search when available, falls back to JSON.
    Embeddings are generated client-side using OpenAI API.
    
    Configuration can be passed directly or loaded from cognitive_memory.yaml.
    
    Example:
        # Using defaults from config file
        store = EpisodicStore()
        
        # With custom config
        from src.memory.config import CognitiveMemoryConfig
        config = CognitiveMemoryConfig.load(preset="high_quality")
        store = EpisodicStore(config=config.episodic)
    """
    
    # Similarity threshold for duplicate detection (0.95 = very similar)
    DUPLICATE_THRESHOLD = 0.95
    
    def __init__(
        self, 
        persist_path: Optional[str] = None,
        config: Optional["EpisodicConfig"] = None
    ):
        """
        Initialize EpisodicStore.
        
        Args:
            persist_path: Override JSON file path (or use config.persist_path)
            config: EpisodicConfig instance (or loads from YAML)
        """
        # Load config if not provided
        if config is None:
            from src.memory.config import CognitiveMemoryConfig
            config = CognitiveMemoryConfig.load().episodic
        
        # Store config values
        self.collection_name = config.collection_name
        self.embedding_model = config.embedding_model
        self.weaviate_host = config.weaviate_host
        self.weaviate_port = config.weaviate_port
        self.retrieval_top_k = config.retrieval_top_k
        self.min_confidence = config.min_confidence
        self.max_insights = config.max_insights
        self.persist_path = persist_path or config.persist_path
        
        # Runtime state
        self.insights: List[Insight] = []
        self._weaviate_client = None
        self._openai_client = None
        
        # Init OpenAI for embeddings
        self._init_openai()
        
        # Try Weaviate connection
        self._init_weaviate()
        
        # Load from JSON as fallback/supplement
        self._load_from_json()
    
    def _init_openai(self):
        """Initialize OpenAI client for embedding generation."""
        if not HAS_OPENAI:
            logger.info("EpisodicStore: openai not installed, Weaviate disabled")
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("EpisodicStore: OPENAI_API_KEY not set, Weaviate disabled")
            return
        
        try:
            self._openai_client = openai.OpenAI(api_key=api_key)
            logger.info(f"EpisodicStore: OpenAI initialized (model: {self.embedding_model})")
        except Exception as e:
            logger.warning(f"EpisodicStore: OpenAI init failed ({e})")
    
    def _init_weaviate(self):
        """Initialize Weaviate connection if available."""
        if not HAS_WEAVIATE:
            logger.info("EpisodicStore: weaviate-client not installed, using JSON only")
            return
        
        # Need OpenAI for embeddings
        if not self._openai_client:
            logger.info("EpisodicStore: OpenAI unavailable, using JSON only")
            return
            
        try:
            self._weaviate_client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port
            )
            self._ensure_collection()
            logger.info(f"EpisodicStore: Connected to Weaviate at {self.weaviate_host}:{self.weaviate_port}")
        except Exception as e:
            logger.warning(f"EpisodicStore: Weaviate unavailable ({e}), using JSON")
            self._weaviate_client = None
    
    def _ensure_collection(self):
        """Create Weaviate collection with explicit vector config (no auto-vectorizer)."""
        if not self._weaviate_client:
            return
            
        try:
            collections = self._weaviate_client.collections.list_all()
            if self.collection_name not in collections.keys():
                # vectorizer_config=none means we provide vectors client-side
                self._weaviate_client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="insight_type", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="confidence", data_type=wvc.config.DataType.NUMBER),
                        wvc.config.Property(name="source_experiment_id", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                        wvc.config.Property(name="created_at", data_type=wvc.config.DataType.TEXT),
                    ]
                )
                logger.info(f"EpisodicStore: Created collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Failed to ensure Weaviate collection: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI client-side."""
        if not self._openai_client:
            return None
        
        try:
            response = self._openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"EpisodicStore: Embedding failed ({e})")
            return None
    
    def _is_duplicate(self, embedding: List[float]) -> bool:
        """Check if a similar insight already exists in Weaviate."""
        if not self._weaviate_client:
            return False
        
        try:
            collection = self._weaviate_client.collections.get(self.collection_name)
            response = collection.query.near_vector(
                near_vector=embedding,
                limit=1,
                return_metadata=["distance"]
            )
            
            if response.objects:
                # distance is cosine distance: 0 = identical, 2 = opposite
                # similarity = 1 - (distance / 2)
                distance = response.objects[0].metadata.distance or 0
                similarity = 1 - (distance / 2)
                if similarity >= self.DUPLICATE_THRESHOLD:
                    logger.debug(f"Duplicate insight detected (similarity: {similarity:.2%})")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Duplicate check failed ({e})")
            return False
    
    def add_insight(self, insight: Insight) -> bool:
        """
        Add a new insight to the store.
        
        Args:
            insight: Insight to add
            
        Returns:
            True if added, False if duplicate or failed
            
        Note: Insights below min_confidence threshold are stored in JSON only.
        """
        # Check confidence threshold
        if insight.confidence < self.min_confidence:
            logger.debug(f"Insight below min_confidence ({insight.confidence} < {self.min_confidence}), JSON only")
        
        # Store in Weaviate if available and above confidence threshold
        if self._weaviate_client and insight.confidence >= self.min_confidence:
            try:
                embedding = self._generate_embedding(insight.content)
                if embedding:
                    # Check for duplicates before inserting
                    if self._is_duplicate(embedding):
                        logger.info("Skipping duplicate insight")
                        return False
                    
                    collection = self._weaviate_client.collections.get(self.collection_name)
                    collection.data.insert(
                        properties={
                            "content": insight.content,
                            "insight_type": insight.insight_type.value,
                            "confidence": insight.confidence,
                            "source_experiment_id": insight.source_experiment_id,
                            "tags": insight.tags,
                            "created_at": insight.created_at.isoformat(),
                        },
                        vector=embedding
                    )
            except Exception as e:
                logger.error(f"Failed to add insight to Weaviate: {e}")
        
        # Check for duplicates in local list
        if any(i.content == insight.content for i in self.insights):
            logger.debug("Skipping duplicate insight (local)")
            return False
        
        self.insights.append(insight)
        
        # Enforce max_insights limit (remove oldest)
        while len(self.insights) > self.max_insights:
            removed = self.insights.pop(0)
            logger.debug(f"Removed oldest insight to stay under limit: {removed.content}")
        
        # Always save to JSON as backup
        self._save_to_json()
        return True
    
    def retrieve_relevant(self, query: str, top_k: Optional[int] = None) -> List[Insight]:
        """
        Retrieve insights relevant to the query using semantic search.
        
        Args:
            query: Search query
            top_k: Number of results (default: config.retrieval_top_k)
            
        Returns:
            List of relevant Insight objects
        """
        top_k = top_k or self.retrieval_top_k
        
        # Try Weaviate semantic search first
        if self._weaviate_client:
            try:
                query_embedding = self._generate_embedding(query)
                if query_embedding:
                    collection = self._weaviate_client.collections.get(self.collection_name)
                    response = collection.query.near_vector(
                        near_vector=query_embedding,
                        limit=top_k,
                        return_properties=["content", "insight_type", "confidence", 
                                          "source_experiment_id", "tags", "created_at"]
                    )
                    results = []
                    for obj in response.objects:
                        props = obj.properties
                        created_at = datetime.now(timezone.utc)
                        if props.get("created_at"):
                            try:
                                created_at = datetime.fromisoformat(props["created_at"])
                            except:
                                pass
                        results.append(Insight(
                            content=props["content"],
                            insight_type=InsightType(props["insight_type"]),
                            confidence=props["confidence"],
                            source_experiment_id=props["source_experiment_id"],
                            tags=props.get("tags", []),
                            created_at=created_at
                        ))
                    return results
            except Exception as e:
                logger.warning(f"Weaviate search failed ({e}), using JSON fallback")
        
        # JSON fallback: simple keyword matching
        query_terms = set(query.lower().split())
        scored = []
        for insight in self.insights:
            score = sum(1 for term in query_terms if term in insight.content.lower())
            if score > 0:
                scored.append((score, insight))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]
    
    def clear(self):
        """Clear all insights from both Weaviate and JSON."""
        self.insights = []
        self._save_to_json()
        
        if self._weaviate_client:
            try:
                self._weaviate_client.collections.delete(self.collection_name)
                self._ensure_collection()
                logger.info("Cleared all insights from Weaviate")
            except Exception as e:
                logger.warning(f"Failed to clear Weaviate collection: {e}")
    
    def _save_to_json(self):
        """Persist insights to JSON file (atomic write)."""
        data = [
            {
                "content": i.content,
                "type": i.insight_type.value,
                "confidence": i.confidence,
                "source": i.source_experiment_id,
                "tags": i.tags,
                "created_at": i.created_at.isoformat()
            }
            for i in self.insights
        ]
        try:
            # Atomic write: write to temp file then rename
            temp_path = self.persist_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.persist_path)
        except Exception as e:
            logger.error(f"Failed to save insights to JSON: {e}")
    
    def _load_from_json(self):
        """Load insights from JSON file."""
        if not os.path.exists(self.persist_path):
            return
            
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
                for d in data:
                    created_at = datetime.now(timezone.utc)
                    if "created_at" in d:
                        try:
                            created_at = datetime.fromisoformat(d["created_at"])
                        except:
                            pass
                    insight = Insight(
                        content=d["content"],
                        insight_type=InsightType(d["type"]),
                        confidence=d["confidence"],
                        source_experiment_id=d["source"],
                        tags=d.get("tags", []),
                        created_at=created_at
                    )
                    # Avoid duplicates
                    if not any(i.content == insight.content for i in self.insights):
                        self.insights.append(insight)
        except Exception as e:
            logger.warning(f"Failed to load insights from JSON: {e}")
    
    def close(self):
        """Close external clients (Weaviate + OpenAI)."""
        if self._weaviate_client:
            try:
                self._weaviate_client.close()
            except:
                pass
        
        # Close OpenAI client if present.
        #
        # Why:
        # - The OpenAI SDK keeps an underlying HTTP client alive (often httpx).
        # - If we don't close it, Python can emit `ResourceWarning: unclosed <ssl.SSLSocket ...>`
        #   at process shutdown, which is noisy and can hide real issues.
        if self._openai_client:
            try:
                if hasattr(self._openai_client, "close"):
                    self._openai_client.close()
            except Exception:
                # Best-effort cleanup. Never fail callers during shutdown.
                pass
            self._openai_client = None
    
    def get_stats(self) -> dict:
        """Get statistics about the episodic store."""
        return {
            "total_insights": len(self.insights),
            "weaviate_connected": self._weaviate_client is not None,
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "persist_path": self.persist_path,
        }
