# Learner Base Classes
#
# Abstract interface for knowledge learners.
# Each implementation handles a specific source type (Repo, Paper, File, etc.)
# and converts it into KnowledgeChunks for KG indexing.
#
# To create a new learner:
# 1. Subclass Learner
# 2. Implement learn()
# 3. Register with @register_learner("your_name") decorator

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeChunk:
    """
    A single unit of knowledge to be indexed into the KG.
    
    This is the universal output format. All learners produce these.
    The KG search backend consumes these for indexing.
    
    Attributes:
        content: The main text or code content
        chunk_type: Category of knowledge ("text", "code", "concept", "workflow")
        source: Origin identifier (URL, file path, experiment ID, etc.)
        metadata: Additional tags, language info, timestamps, etc.
    """
    content: str
    chunk_type: str = "text"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for KG indexing."""
        return {
            "content": self.content,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "metadata": self.metadata,
        }


class Learner(ABC):
    """
    Abstract base class for knowledge learners.
    
    Each implementation handles a specific source type (Repo, Paper, File, etc.)
    and converts it into a list of KnowledgeChunks for the KG.
    
    The Expert.learn() method dispatches to the appropriate Learner based on
    the Source type provided by the user.
    
    Subclasses must implement:
    - learn(): Extract knowledge from source data
    - name: Property returning the learner's identifier
    
    To create a new learner:
    1. Subclass Learner
    2. Implement learn() and name property
    3. Register with @register_learner("your_name") decorator in factory.py
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize learner.
        
        Args:
            params: Implementation-specific parameters
        """
        self.params = params or {}
    
    @abstractmethod
    def learn(self, source_data: Any) -> List[KnowledgeChunk]:
        """
        Extract knowledge from source data.
        
        Args:
            source_data: The input data (URL, path, Solution object, etc.)
                         Format depends on the specific learner implementation.
            
        Returns:
            List of KnowledgeChunk objects ready for KG indexing
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the learner's identifier (e.g., 'repo', 'paper')."""
        pass

