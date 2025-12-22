# Episodic Memory Store - persists insights from past experiments
import json
import os
import logging
from typing import List, Optional

try:
    import weaviate
    import weaviate.classes as wvc
    HAS_WEAVIATE = True
except ImportError:
    weaviate = None
    wvc = None
    HAS_WEAVIATE = False

from src.memory.types import Insight, InsightType

logger = logging.getLogger(__name__)


class EpisodicStore:
    """
    Stores generalized insights from past experiments.
    Uses Weaviate for semantic search when available, falls back to JSON.
    """
    
    COLLECTION_NAME = "EpisodicInsights"
    
    def __init__(self, persist_path: str = ".memory_store.json"):
        self.persist_path = persist_path
        self.insights: List[Insight] = []
        self._weaviate_client: Optional[weaviate.WeaviateClient] = None
        
        # Try Weaviate first
        self._init_weaviate()
        
        # Load from JSON as fallback/supplement
        self._load_from_json()
    
    def _init_weaviate(self):
        """Initialize Weaviate connection if available."""
        if not HAS_WEAVIATE:
            logger.info("EpisodicStore: weaviate-client not installed, using JSON only")
            return
            
        try:
            self._weaviate_client = weaviate.connect_to_local(
                host="localhost",
                port=8080
            )
            self._ensure_collection()
            logger.info(f"EpisodicStore: Connected to Weaviate")
        except Exception as e:
            logger.warning(f"EpisodicStore: Weaviate unavailable ({e}), using JSON")
            self._weaviate_client = None
    
    def _ensure_collection(self):
        """Create Weaviate collection if needed."""
        if not self._weaviate_client:
            return
            
        try:
            collections = self._weaviate_client.collections.list_all()
            if self.COLLECTION_NAME not in [c for c in collections.keys()]:
                self._weaviate_client.collections.create(
                    name=self.COLLECTION_NAME,
                    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
                    properties=[
                        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="insight_type", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="confidence", data_type=wvc.config.DataType.NUMBER),
                        wvc.config.Property(name="source_experiment_id", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                    ]
                )
                logger.info(f"EpisodicStore: Created collection '{self.COLLECTION_NAME}'")
        except Exception as e:
            logger.warning(f"Failed to ensure Weaviate collection: {e}")
    
    def add_insight(self, insight: Insight):
        """Add a new insight to the store."""
        self.insights.append(insight)
        
        # Store in Weaviate if available
        if self._weaviate_client:
            try:
                collection = self._weaviate_client.collections.get(self.COLLECTION_NAME)
                collection.data.insert({
                    "content": insight.content,
                    "insight_type": insight.insight_type.value,
                    "confidence": insight.confidence,
                    "source_experiment_id": insight.source_experiment_id,
                    "tags": insight.tags,
                })
            except Exception as e:
                logger.error(f"Failed to add insight to Weaviate: {e}")
        
        # Always save to JSON as backup
        self._save_to_json()
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Insight]:
        """Retrieve insights relevant to the query."""
        
        # Try Weaviate semantic search first
        if self._weaviate_client:
            try:
                collection = self._weaviate_client.collections.get(self.COLLECTION_NAME)
                response = collection.query.near_text(
                    query=query,
                    limit=top_k,
                    return_properties=["content", "insight_type", "confidence", "source_experiment_id", "tags"]
                )
                return [
                    Insight(
                        content=obj.properties["content"],
                        insight_type=InsightType(obj.properties["insight_type"]),
                        confidence=obj.properties["confidence"],
                        source_experiment_id=obj.properties["source_experiment_id"],
                        tags=obj.properties.get("tags", [])
                    )
                    for obj in response.objects
                ]
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
    
    def _save_to_json(self):
        """Persist insights to JSON file."""
        data = [
            {
                "content": i.content,
                "type": i.insight_type.value,
                "confidence": i.confidence,
                "source": i.source_experiment_id,
                "tags": i.tags
            }
            for i in self.insights
        ]
        try:
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2)
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
                    insight = Insight(
                        content=d["content"],
                        insight_type=InsightType(d["type"]),
                        confidence=d["confidence"],
                        source_experiment_id=d["source"],
                        tags=d.get("tags", [])
                    )
                    # Avoid duplicates
                    if not any(i.content == insight.content for i in self.insights):
                        self.insights.append(insight)
        except Exception as e:
            logger.warning(f"Failed to load insights from JSON: {e}")
