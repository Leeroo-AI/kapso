# Experiment Learner
#
# Extracts knowledge from completed Solutions (experiment logs).
# This is the "backward pass" - learning from experience.

from typing import Any, Dict, List, TYPE_CHECKING

from src.knowledge.learners.base import Learner, KnowledgeChunk
from src.knowledge.learners.factory import register_learner

# Avoid circular import
if TYPE_CHECKING:
    from src.expert import SolutionResult


@register_learner("experiment")
class ExperimentLearner(Learner):
    """
    Learn from experiment logs (the backward pass).
    
    Extracts knowledge from:
    - Goal and constraints that were provided
    - Experiment logs (what worked, what failed)
    - Final solution patterns
    - Error patterns to avoid
    
    This learner is key for the "reinforcement" step where the Expert
    learns from its own experiments to improve future builds.
    
    Input format:
        SolutionResult object with .goal, .experiment_logs, .code_path
    """
    
    @property
    def name(self) -> str:
        return "experiment"
    
    def learn(self, source_data: Any) -> List[KnowledgeChunk]:
        """
        Extract knowledge from a SolutionResult object.
        
        Args:
            source_data: SolutionResult object with goal, experiment_logs, code_path
            
        Returns:
            List of KnowledgeChunk capturing the learnings
        """
        chunks = []
        
        # Extract attributes from SolutionResult object
        goal = getattr(source_data, 'goal', 'Unknown goal')
        experiment_logs = getattr(source_data, 'experiment_logs', [])
        code_path = getattr(source_data, 'code_path', '')
        
        # Create a workflow chunk summarizing the successful approach
        if experiment_logs:
            # Identify successful experiments
            successful_logs = [log for log in experiment_logs if "Success" in str(log)]
            failed_logs = [log for log in experiment_logs if "Failed" in str(log)]
            
            # Create chunk for successful patterns
            if successful_logs:
                chunks.append(KnowledgeChunk(
                    content=f"Goal: {goal}\nSuccessful approaches:\n" + "\n".join(str(l) for l in successful_logs),
                    chunk_type="workflow",
                    source="experiment",
                    metadata={"goal": goal, "outcome": "success", "learner": "experiment"}
                ))
            
            # Create chunk for patterns to avoid
            if failed_logs:
                chunks.append(KnowledgeChunk(
                    content=f"Goal: {goal}\nApproaches to avoid:\n" + "\n".join(str(l) for l in failed_logs),
                    chunk_type="concept",
                    source="experiment",
                    metadata={"goal": goal, "outcome": "failure", "learner": "experiment"}
                ))
        
        # If no specific logs, create a general chunk
        if not chunks:
            chunks.append(KnowledgeChunk(
                content=f"Completed solution for goal: {goal}\nCode at: {code_path}",
                chunk_type="workflow",
                source="experiment",
                metadata={"goal": goal, "learner": "experiment"}
            ))
        
        print(f"[ExperimentLearner] Learned from experiment: {goal}")
        return chunks

