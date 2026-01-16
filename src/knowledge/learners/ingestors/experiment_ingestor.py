# Experiment Ingestor
#
# Extracts knowledge from completed Solutions (experiment logs).
# This is the "backward pass" - learning from experience.
#
# Part of Stage 1 of the knowledge learning pipeline.

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

# Avoid circular import
if TYPE_CHECKING:
    from src.kapso import SolutionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_ingestor("solution")
class ExperimentIngestor(Ingestor):
    """
    Extract knowledge from experiment logs (the backward pass).
    
    Extracts knowledge from:
    - Goal and constraints that were provided
    - Experiment logs (what worked, what failed)
    - Final solution patterns
    - Error patterns to avoid
    
    This ingestor is key for the "reinforcement" step where the Kapso
    learns from its own experiments to improve future builds.
    
    Input formats:
        Source.Solution(solution_result_obj)
        SolutionResult object directly
    
    Example:
        ingestor = ExperimentIngestor()
        pages = ingestor.ingest(Source.Solution(solution_result))
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ExperimentIngestor.
        
        Args:
            params: Optional parameters (for future use)
        """
        super().__init__(params)
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "solution"
    
    def _normalize_source(self, source: Any) -> Any:
        """
        Normalize source input.
        
        For experiment ingestor, we expect either:
        - Source.Solution wrapper with .obj attribute
        - Raw SolutionResult object
        
        Args:
            source: Source.Solution or SolutionResult object
            
        Returns:
            The underlying SolutionResult object
        """
        # If it's a Source.Solution wrapper, extract the obj
        if hasattr(source, 'obj'):
            return source.obj
        # Otherwise assume it's the raw SolutionResult
        return source
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Extract knowledge from a SolutionResult object.
        
        Args:
            source: Source.Solution wrapper or SolutionResult object
            
        Returns:
            List of proposed WikiPage objects (Workflow, Heuristic types)
        """
        # Normalize to get SolutionResult
        solution = self._normalize_source(source)
        
        # Extract attributes
        goal = getattr(solution, 'goal', 'Unknown goal')
        experiment_logs = getattr(solution, 'experiment_logs', [])
        code_path = getattr(solution, 'code_path', '')
        
        pages = []
        
        # TODO: Implement proper knowledge extraction with Claude Code agent
        # 1. Analyze experiment logs to identify patterns
        # 2. Extract successful approaches as Workflow pages
        # 3. Extract optimizations/tips as Heuristic pages
        # 4. Extract failure patterns as Heuristics (what to avoid)
        
        # For now, create a placeholder Workflow page if there's data
        if goal and goal != 'Unknown goal':
            # Analyze logs
            successful_logs = [log for log in experiment_logs if "Success" in str(log)]
            failed_logs = [log for log in experiment_logs if "Failed" in str(log)]
            
            # Create a summary page
            content = f"""== Overview ==
Experiment workflow for: {goal}

=== Description ===
This workflow was learned from a completed experiment.

== Experiment Summary ==
- Goal: {goal}
- Code Path: {code_path}
- Successful Approaches: {len(successful_logs)}
- Failed Approaches: {len(failed_logs)}

== Key Learnings ==
"""
            if successful_logs:
                content += "\n=== What Worked ===\n"
                for log in successful_logs[:5]:  # Limit to first 5
                    content += f"* {str(log)[:200]}\n"
            
            if failed_logs:
                content += "\n=== What to Avoid ===\n"
                for log in failed_logs[:5]:  # Limit to first 5
                    content += f"* {str(log)[:200]}\n"
            
            # Create a simple title from goal
            title = goal.replace(" ", "_")[:50]
            
            page = WikiPage(
                id=f"Workflow/{title}_Experiment",
                page_title=f"{title}_Experiment",
                page_type="Workflow",
                overview=f"Learned workflow for: {goal}",
                content=content,
                domains=["Experiments", "Learned"],
                sources=[{"type": "Experiment", "title": "Local Experiment", "url": code_path}],
                outgoing_links=[],
            )
            pages.append(page)
            
            logger.info(f"[ExperimentIngestor] Extracted workflow from experiment: {goal}")
        else:
            logger.warning("[ExperimentIngestor] No goal found in solution")
        
        return pages


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    """Test the ExperimentIngestor."""
    from dataclasses import dataclass
    
    print("=" * 60)
    print("ExperimentIngestor Test")
    print("=" * 60)
    
    # Create a mock SolutionResult
    @dataclass
    class MockSolution:
        goal: str = "Train a classifier"
        experiment_logs: list = None
        code_path: str = "/tmp/experiment"
        
        def __post_init__(self):
            if self.experiment_logs is None:
                self.experiment_logs = [
                    "Success: Model trained with 95% accuracy",
                    "Failed: OOM with batch_size=64",
                    "Success: Reduced batch_size to 32",
                ]
    
    mock_solution = MockSolution()
    
    print(f"\nTest solution goal: {mock_solution.goal}")
    print("-" * 60)
    
    ingestor = ExperimentIngestor()
    
    try:
        pages = ingestor.ingest(mock_solution)
        print(f"\nExtracted {len(pages)} proposed pages:")
        for page in pages:
            print(f"  - {page.page_title} ({page.page_type})")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")

