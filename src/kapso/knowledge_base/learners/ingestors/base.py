# Ingestor Base Class
#
# Abstract interface for knowledge ingestors.
# Each implementation handles a specific source type (Repo, Solution)
# and converts it into WikiPage objects for the KG.
#
# To create a new ingestor:
# 1. Subclass Ingestor
# 2. Implement ingest() and source_type property
# 3. Register with @register_ingestor("your_name") decorator

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from kapso.knowledge_base.search.base import WikiPage


class Ingestor(ABC):
    """
    Abstract base class for knowledge ingestors.
    
    Each implementation handles a specific source type (Repo, Solution)
    and converts it into a list of WikiPage objects for the Knowledge Graph.
    
    The KnowledgePipeline dispatches to the appropriate Ingestor based on
    the Source type provided by the user.
    
    Subclasses must implement:
    - ingest(): Extract knowledge from source, return WikiPages
    - source_type: Property returning the source type this handles
    
    Example:
        @register_ingestor("repo")
        class RepoIngestor(Ingestor):
            @property
            def source_type(self) -> str:
                return "repo"
            
            def ingest(self, source) -> List[WikiPage]:
                # Clone repo, extract knowledge, return pages
                ...
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ingestor.
        
        Args:
            params: Implementation-specific parameters
        """
        self.params = params or {}
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """
        Return the source type this ingestor handles.
        
        Returns:
            Source type identifier (e.g., 'repo', 'solution')
        """
        pass
    
    @abstractmethod
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Extract knowledge from source and return proposed WikiPages.
        
        This is the main method that processes a knowledge source.
        The ingestor should:
        1. Access/download the source content
        2. Extract structured knowledge
        3. Return WikiPage objects ready for merging into KG
        
        Args:
            source: The input source (Source.Repo, Source.Solution)
                    or a dict with source data
            
        Returns:
            List of WikiPage objects representing extracted knowledge
        """
        pass

