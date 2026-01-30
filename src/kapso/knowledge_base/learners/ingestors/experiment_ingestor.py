# Experiment Ingestor
#
# Extracts knowledge from completed Solutions (experiment logs).
# This is the "backward pass" - learning from experience.
#
# Part of Stage 1 of the knowledge learning pipeline.
#
# Status: Under development

from typing import Any, Dict, List, Optional

from kapso.knowledge_base.learners.ingestors.base import Ingestor
from kapso.knowledge_base.learners.ingestors.factory import register_ingestor
from kapso.knowledge_base.search.base import WikiPage


@register_ingestor("solution")
class ExperimentIngestor(Ingestor):
    """
    Extract knowledge from experiment logs (the backward pass).
    
    Extracts knowledge from:
    - Goal and constraints that were provided
    - Experiment logs (what worked, what failed)
    - Final solution patterns
    - Error patterns to avoid
    
    This ingestor is key for the "reinforcement" step where Kapso
    learns from its own experiments to improve future builds.
    
    Input formats:
        Source.Solution(solution_result_obj)
    
    Example:
        ingestor = ExperimentIngestor()
        pages = ingestor.ingest(Source.Solution(solution_result))
    
    Status: Under development - will be implemented soon.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
    
    @property
    def source_type(self) -> str:
        return "solution"
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Extract knowledge from a SolutionResult object.
        
        Args:
            source: Source.Solution wrapper or SolutionResult object
            
        Returns:
            List of proposed WikiPage objects
            
        Raises:
            NotImplementedError: This feature is under development.
        """
        raise NotImplementedError(
            "ExperimentIngestor is under development and will be added soon. "
            "This ingestor will extract knowledge from experiment logs to learn "
            "successful patterns and failures to avoid."
        )
