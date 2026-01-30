# Search Strategy Base Classes
#
# Minimal base class for all search strategies.
# Contains only shared infrastructure and abstract method signatures.
#
# To create a new strategy:
# 1. Subclass SearchStrategy
# 2. Implement abstract methods: run(), get_experiment_history(), get_best_experiment(), etc.
# 3. Register with @register_strategy("your_name") decorator in factory.py

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kapso.execution.types import ContextData
from kapso.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.environment.handlers.base import ProblemHandler
from kapso.core.llm import LLMBackend
from kapso.execution.memories.repo_memory import RepoMemoryManager

# Avoid circular import - FeedbackGenerator is optional
if TYPE_CHECKING:
    from kapso.execution.search_strategies.generic import FeedbackGenerator


@dataclass
class SearchNode:
    """
    Unified node structure for search strategies.
    
    Accumulates data through the node lifecycle:
    1. Solution generation -> solution populated
    2. Implementation -> branch_name, code_changes_summary populated
    3. Evaluation -> evaluation_script_path, evaluation_output populated
    4. Feedback -> feedback, score, should_stop populated
    """
    node_id: int
    parent_node_id: Optional[int] = None
    
    # Step 1: Solution generation
    solution: str = ""
    
    # Step 2: Implementation
    branch_name: str = ""
    code_changes_summary: str = ""
    agent_output: str = ""  # Raw output from developer agent
    
    # Step 3: Evaluation (extracted from agent output or result.json)
    evaluation_script_path: str = ""
    evaluation_output: str = ""
    
    # Step 4: Feedback
    feedback: str = ""
    score: Optional[float] = None
    should_stop: bool = False
    evaluation_valid: bool = True
    
    # Metadata
    had_error: bool = False
    error_message: str = ""
    workspace_dir: str = ""
    code_diff: str = ""
    
    def __str__(self) -> str:
        if self.had_error:
            return f"- Node {self.node_id} failed: {self.error_message[:100]}...\n  Solution: {self.solution[:200]}..."
        else:
            return (
                f"- Node {self.node_id} (score={self.score}):\n"
                f"  Solution: {self.solution[:200]}...\n"
                + (f"  Feedback: {self.feedback[:200]}...\n" if self.feedback else "")
            )


@dataclass
class ExperimentResult:
    """
    Result of a single experiment.
    
    DEPRECATED: Use SearchNode instead. Kept for backward compatibility.
    """
    node_id: int
    solution: str
    score: float
    branch_name: str
    had_error: bool
    error_message: str = ""
    output: str = ""
    detailed_output: str = ""
    feedbacks: str = ""
    embedding: List[float] = None
    evaluation_output: str = ""
    evaluation_script_path: str = ""
    code_diff: str = ""
    workspace_dir: str = ""
    
    def __str__(self) -> str:
        if self.had_error:
            return f"- Experiment with failed implementation error {self.error_message}. :\n  {self.solution} "
        else:
            return (
                f"- Experiment with final score {self.score} :\n # Solution : {self.solution}" 
                + (f"\n\n  # Runtime output: {self.output}" if self.output else "")
                + (f"\n\n  # Feedbacks: {self.feedbacks} \n" if self.feedbacks else "")
            )
    
    def get_embedding(self, llm: LLMBackend) -> List[float]:
        if self.embedding is None:
            self.embedding = llm.create_embedding(self.__str__())
        return self.embedding
    
    @classmethod
    def from_search_node(cls, node: SearchNode) -> "ExperimentResult":
        """Convert SearchNode to ExperimentResult for backward compatibility."""
        return cls(
            node_id=node.node_id,
            solution=node.solution,
            score=node.score or 0.0,
            branch_name=node.branch_name,
            had_error=node.had_error,
            error_message=node.error_message,
            output=node.agent_output,
            detailed_output=node.agent_output,
            feedbacks=node.feedback,
            evaluation_output=node.evaluation_output,
            evaluation_script_path=node.evaluation_script_path,
            code_diff=node.code_diff,
            workspace_dir=node.workspace_dir,
        )


@dataclass 
class SearchStrategyConfig:
    """Configuration passed to search strategies."""
    problem_handler: ProblemHandler
    llm: LLMBackend
    coding_agent_config: CodingAgentConfig
    # Strategy-specific params (from YAML config)
    params: Dict[str, Any] = field(default_factory=dict)
    # Optional: start experiments from an existing local repo (copy/clone into workspace)
    initial_repo: Optional[str] = None
    # Optional: directories to copy into workspace
    eval_dir: Optional[str] = None
    data_dir: Optional[str] = None
    # Optional: FeedbackGenerator for generating feedback after each experiment
    feedback_generator: Optional["FeedbackGenerator"] = None
    # Goal string for feedback generation
    goal: str = ""


class SearchStrategy(ABC):
    """
    Abstract base class for experiment search strategies.
    
    Subclasses must implement:
    - run(): Execute one iteration of the search, returns SearchNode
    - get_experiment_history(): Return all experiments
    - get_best_experiment(): Return best experiment so far
    - checkout_to_best_experiment_branch(): Checkout to best solution
    - export_checkpoint(): Save state to disk
    - import_checkpoint(): Load state from disk
    
    Shared infrastructure provided:
    - Workspace creation and management
    - RepoMemory bootstrap
    - Kapso directory setup (eval_dir, data_dir)
    """
    
    WORKSPACE_FOLDER_BASE = 'tmp/search_strategy_workspace'
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """
        Initialize search strategy.
        
        Args:
            config: SearchStrategyConfig with problem_handler, llm, coding_agent_config, params
            workspace_dir: Path to the workspace directory (optional)
            import_from_checkpoint: Whether to import state from checkpoint
        """
        self.problem_handler = config.problem_handler
        self.llm = config.llm
        self.params = config.params
        
        # Feedback generator and goal for generating feedback after experiments
        self.feedback_generator = config.feedback_generator
        self.goal = config.goal
        
        # Create experiment workspace with coding agent config
        if workspace_dir is None:
            self.workspace_dir = os.path.join(self.WORKSPACE_FOLDER_BASE, str(uuid.uuid4()))
        else:
            self.workspace_dir = workspace_dir
        self.workspace = ExperimentWorkspace(
            coding_agent_config=config.coding_agent_config,
            workspace_dir=self.workspace_dir,
            initial_repo=config.initial_repo,
        )

        # Setup kapso directories (eval_dir -> kapso_evaluation/, data_dir -> kapso_datasets/)
        if not import_from_checkpoint:
            self._setup_kapso_directories(config.eval_dir, config.data_dir)

        # Ensure baseline RepoMemory exists in the workspace repo.
        if not import_from_checkpoint:
            if self.workspace.is_seeded:
                RepoMemoryManager.bootstrap_baseline_model(
                    repo_root=self.workspace_dir,
                    llm=self.llm,
                    initial_repo=self.workspace.initial_repo,
                )
            else:
                RepoMemoryManager.ensure_exists_in_worktree(self.workspace_dir)

            # Commit baseline memory file if it is new/updated
            self.workspace.repo.git.add([RepoMemoryManager.MEMORY_REL_PATH])
            if self.workspace.repo.is_dirty(untracked_files=True):
                self.workspace.repo.git.commit("-m", "chore(kapso): add baseline repo memory")

        if import_from_checkpoint:
            self.import_checkpoint()
    
    # =========================================================================
    # Directory Setup
    # =========================================================================
    
    def _setup_kapso_directories(
        self, 
        eval_dir: Optional[str], 
        data_dir: Optional[str]
    ) -> None:
        """
        Setup kapso_evaluation/ and kapso_datasets/ directories in workspace.
        
        Copies user-provided directories into the workspace repo so the agent
        has access to evaluation scripts and datasets.
        """
        workspace = self.workspace.workspace_dir
        dirs_created = []
        
        # Setup kapso_evaluation/
        kapso_eval = os.path.join(workspace, "kapso_evaluation")
        os.makedirs(kapso_eval, exist_ok=True)
        if eval_dir and os.path.exists(eval_dir):
            shutil.copytree(eval_dir, kapso_eval, dirs_exist_ok=True)
            print(f"  Copied eval_dir to kapso_evaluation/")
        dirs_created.append("kapso_evaluation")
        
        # Setup kapso_datasets/
        kapso_data = os.path.join(workspace, "kapso_datasets")
        os.makedirs(kapso_data, exist_ok=True)
        if data_dir and os.path.exists(data_dir):
            shutil.copytree(data_dir, kapso_data, dirs_exist_ok=True)
            print(f"  Copied data_dir to kapso_datasets/")
        dirs_created.append("kapso_datasets")
        
        # Add placeholder files to empty directories so git tracks them
        for dir_name in dirs_created:
            dir_path = os.path.join(workspace, dir_name)
            if not os.listdir(dir_path):
                placeholder = os.path.join(dir_path, ".gitkeep")
                with open(placeholder, "w") as f:
                    f.write("# Placeholder to track empty directory\n")
        
        # Commit the directories to the workspace repo
        self.workspace.repo.git.add(dirs_created)
        if self.workspace.repo.is_dirty(untracked_files=True):
            self.workspace.repo.git.commit("-m", "chore(kapso): setup evaluation and data directories")
    
    # =========================================================================
    # Shared Helpers
    # =========================================================================

    def _get_code_diff(self, branch_name: str, parent_branch: str) -> str:
        """Get git diff between branch and parent."""
        try:
            diff = self.workspace.repo.git.diff(parent_branch, branch_name)
            return diff
        except Exception as e:
            print(f"[SearchStrategy] Warning: Could not get diff: {e}")
            return ""

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def run(self, context: ContextData, budget_progress: float = 0.0) -> Optional[SearchNode]:
        """
        Execute one iteration of the search strategy.
        
        Args:
            context: Problem context, KG results, experiment history
            budget_progress: 0-100 indicating budget consumed
            
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        pass
    
    @abstractmethod
    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """
        Get all experiment results.
        
        Args:
            best_last: If True, sort by score (best last)
            
        Returns:
            List of SearchNode
        """
        pass
    
    @abstractmethod
    def get_best_experiment(self) -> Optional[SearchNode]:
        """Get the best experiment result so far."""
        pass
    
    @abstractmethod
    def checkout_to_best_experiment_branch(self) -> None:
        """Checkout git to the best experiment's branch."""
        pass

    @abstractmethod
    def export_checkpoint(self) -> None:
        """Export checkpoint to the workspace folder."""
        pass

    @abstractmethod
    def import_checkpoint(self) -> None:
        """Import checkpoint from the workspace folder."""
        pass
