# Search Strategy Base Classes
#
# Base class for all search strategies with shared implementation.
#
# To create a new strategy:
# 1. Subclass SearchStrategy
# 2. Implement abstract methods: run(), get_experiment_history(), get_best_experiment()
# 3. Register with @register_strategy("your_name") decorator in factory.py

import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.execution.context_manager.types import ContextData
from src.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from src.execution.experiment_workspace.experiment_session import ExperimentSession
from src.execution.coding_agents.base import CodingAgentConfig
from src.environment.handlers.base import ProblemHandler, ProblemRunResult
from src.core.llm import LLMBackend


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
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


@dataclass 
class SearchStrategyConfig:
    """Configuration passed to search strategies."""
    problem_handler: ProblemHandler
    llm: LLMBackend
    coding_agent_config: CodingAgentConfig
    # Strategy-specific params (from YAML config)
    params: Dict[str, Any] = field(default_factory=dict)


class SearchStrategy(ABC):
    """
    Abstract base class for experiment search strategies.
    
    Subclasses must implement:
    - run(): Execute one iteration of the search
    - get_experiment_history(): Return all experiments
    - get_best_experiment(): Return best experiment so far
    - checkout_to_best_experiment_branch(): Checkout to best solution
    
    Shared functionality provided:
    - implement_solution(): Generate code for a solution
    - debug_solution(): Debug failed code
    - _implement_n_debug(): Full implement + debug loop
    """
    
    WORKSPACE_FOLDER_BASE = 'tmp/search_strategy_workspace'
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """
        Initialize search strategy.
        
        Args:
            config: SearchStrategyConfig with problem_handler, llm, coding_agent_config, params
            workspace_dir: Path to the workspace directory (optional)
        """
        self.problem_handler = config.problem_handler
        self.llm = config.llm
        self.params = config.params
        
        # Create experiment workspace with coding agent config
        if workspace_dir is None:
            self.workspace_dir = os.path.join(self.WORKSPACE_FOLDER_BASE, str(uuid.uuid4()))
        else:
            self.workspace_dir = workspace_dir
        self.workspace = ExperimentWorkspace(coding_agent_config=config.coding_agent_config, workspace_dir=self.workspace_dir)

        if import_from_checkpoint:
            self.import_checkpoint()

        # Shared state for tracking errors
        self.previous_errors: List[str] = []
        self.recent_error_count = 10
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def run(self, context: ContextData, budget_progress: float = 0.0) -> None:
        """
        Execute one iteration of the search strategy.
        
        Args:
            context: Problem context, KG results, experiment history
            budget_progress: 0-100 indicating budget consumed
        """
        pass
    
    @abstractmethod
    def get_experiment_history(self, best_last: bool = False) -> List[ExperimentResult]:
        """
        Get all experiment results.
        
        Args:
            best_last: If True, sort by score (best last)
            
        Returns:
            List of ExperimentResult
        """
        pass
    
    @abstractmethod
    def get_best_experiment(self) -> Optional[ExperimentResult]:
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

    # =========================================================================
    # Shared Implementation - Available to all subclasses
    # =========================================================================
    
    def implement_solution(
        self, 
        solution: str, 
        context: ContextData, 
        session: ExperimentSession
    ) -> ProblemRunResult:
        """
        Generate code for a solution and run it.
        
        Args:
            solution: The solution description to implement
            context: Problem context with KG results
            session: Experiment session with coding agent
            
        Returns:
            ProblemRunResult with score and error info
        """
        developer_prompt = f"""
            You are a world class developer and programmer. Modify the repo or Implement the provided <solution> for <problem>.
            Requirements:
            - Write a clean and functional code.
            - You must implement the <solution> exactly as it is provided.
                - Read Sections and Steps of <solution> carefully and implement them exactly.
            - Output code and format must be as mentioned in the problem statement.
            - Do not write any comments in the code. Just the start of each section.
            - Choose the names of the variables and functions according to the the solution.
            - The code must be highly structured and well organized. Separate sections for different functionalities.
            - Run the code only if asked in the problem context.
            - Use the informaton from knowledge base to develop with less error. under no circumstances deviate from the <solution> provided because of the knowledge base.
            - CRITICAL: Never print or allow interactive or multiline outputs like tqdm, progress bar, etc.
            - You have access to a list of your recent errors that you have made in the past. make sure to no repeat them.
                <previous_errors>
                {self.previous_errors[-self.recent_error_count:]}
                </previous_errors>
            - Directories:
                - Codes must be implemented in the current directory and git root.
                - Experiment Output Data Directory: For the outputs of running the main code like checkpoints, data files and final outputs use Experiment Output Data Directory : "./output_data_{session.branch_name}".
                    -- Always use this path relative and always set it in each implementation.
                - It is highly critical that not using absolute path for the above directories but if the problem provided absolute folders, it is ok to use them.
            - At the end create a changes.log file and summarize the changes you made to implement the <solution> in a few sentences.
            - CRITICAL: You are an AI code editor. Your ONLY job is to edit code files. Do NOT write any conversational text, explanations, or descriptions.  Do not respond with "I'll implement..." or any other conversational text.
            - The most critical part of development: Read the <solution> line by line, understand the logic and details and implement the code exactly as <solution> is provided.    
            \n\n
            <Relible information from knowledge base>
             {context.kg_code_results}
            <Relible information from knowledge base>
            \n\n
            <problem>
             {context.problem}
            </problem>
            \n\n
            <solution>
             {solution}
            </solution>
            
            - Do not ask any questions from the user. Just implement everything as you said. It is highly critical to implement everything as you said so be through.
        """

        session.generate_code(developer_prompt)

        return self.problem_handler.run(
            session.session_folder,
            run_data_dir=session.run_dir,
            solution=solution,
        )

    def debug_solution(
        self, 
        solution: str, 
        context: ContextData, 
        error: str, 
        session: ExperimentSession
    ) -> ProblemRunResult:
        """
        Debug a failed solution.
        
        Args:
            solution: The solution description
            context: Problem context
            error: Error message from failed run
            session: Experiment session
            
        Returns:
            ProblemRunResult after debug attempt
        """
        developer_prompt = f"""
            You are a world class developer. Debug the Implemented <solution> for <problem>.
            \n\n
            <problem>
             {context.problem}
            </problem>
            \n\n
            <solution>
             {solution}
            </solution>
            \n\n
            current output: {error}
            Requirements:
            - Read the code line by line and understand the logic.
            - Make sure every part of the <solution> is implemented correctly.
                - Read sections and steps of <solution> carefully and implement them exactly.            
            - Do not propose a new solution or drift away from the current implementation for <solution> and only fix the errors.
            - Write clean, functional code, that can be improved iteratively later.
            - Output code and format must be as mentioned in the problem statement.
            - Do not add logics like fallback and functionality discarding to avoid the error. you must fix the error directly.
            - Never and under no circumstances use try except blocks to fix the errors. you should fix the error directly.
            - Beside fixing the current error, read the code and make sure other parts of the code will be run correctly and without errors.
            - Do not change any hyper parameter or logic of the solution to fix the error.
            - Do not ask any questions from the user. just do as you said.
        """
        session.generate_code(developer_prompt, debug_mode=True)
        return self.problem_handler.run(
            session.session_folder,
            run_data_dir=session.run_dir,
            solution=solution,
            debug=True,
        )

    def _implement_n_debug(
        self, 
        solution: str, 
        context: ContextData, 
        code_debug_tries: int, 
        branch_name: str, 
        parent_branch_name: str = "main"
    ) -> ProblemRunResult:
        """
        Full implementation + debugging loop.
        
        Args:
            solution: Solution description to implement
            context: Problem context
            code_debug_tries: Max debug attempts
            branch_name: Git branch for this experiment
            parent_branch_name: Parent branch to inherit code from
            
        Returns:
            Final ProblemRunResult
        """
        session = self.workspace.create_experiment_session(branch_name, parent_branch_name)
        result = self.implement_solution(solution, context, session)
        
        for i in range(code_debug_tries):
            if result.run_had_error and result.continue_debugging:
                print("Run had error, error details: ", result.error_details)
                result = self.debug_solution(solution, context, result.error_details, session)
            else:
                break
        
        self.workspace.finalize_session(session)
        return result
