# Search Strategy Base Classes
#
# Base class for all search strategies with shared implementation.
#
# To create a new strategy:
# 1. Subclass SearchStrategy
# 2. Implement abstract methods: run(), get_experiment_history(), get_best_experiment()
# 3. Register with @register_strategy("your_name") decorator in factory.py

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.execution.context_manager.types import ContextData
from src.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from src.execution.experiment_workspace.experiment_session import ExperimentSession
from src.execution.coding_agents.base import CodingAgentConfig
from src.environment.handlers.base import ProblemHandler
from src.core.llm import LLMBackend
from src.core.prompt_loader import load_prompt, render_prompt
from src.repo_memory import RepoMemoryManager
from src.repo_memory.observation import extract_repo_memory_sections_consulted


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
    # New fields for feedback generator
    evaluation_output: str = ""  # Output from running evaluation script
    evaluation_script_path: str = ""  # Path to evaluation script (from developer agent)
    code_diff: str = ""  # Git diff of implementation changes
    workspace_dir: str = ""  # Path to workspace for this experiment
    
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
    # Optional: start experiments from an existing local repo (copy/clone into workspace)
    initial_repo: Optional[str] = None
    # Optional: directories to copy into workspace
    eval_dir: Optional[str] = None
    data_dir: Optional[str] = None


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
        self.workspace = ExperimentWorkspace(
            coding_agent_config=config.coding_agent_config,
            workspace_dir=self.workspace_dir,
            initial_repo=config.initial_repo,
        )

        # Setup kapso directories (eval_dir -> kapso_evaluation/, data_dir -> kapso_datasets/)
        # This must happen before RepoMemory is built so the directories are included.
        if not import_from_checkpoint:
            self._setup_kapso_directories(config.eval_dir, config.data_dir)

        # Ensure baseline RepoMemory exists in the workspace repo.
        #
        # - For seeded repos: build an evidence-backed RepoModel once at start so
        #   ideation and implementation can be grounded in the repo's actual design.
        # - For empty workspaces: create a lightweight skeleton (RepoMap only).
        #
        # RepoMemory is committed into the workspace's "main" branch under `.kapso/`,
        # so all experiment branches inherit it automatically.
        if not import_from_checkpoint:
            # Build baseline RepoMemory and commit it to the workspace's main branch.
            # - For seeded repos: build evidence-backed RepoModel via LLM.
            # - For empty workspaces: create a lightweight skeleton (RepoMap only).
            if self.workspace.is_seeded:
                RepoMemoryManager.bootstrap_baseline_model(
                    repo_root=self.workspace_dir,
                    llm=self.llm,
                    initial_repo=self.workspace.initial_repo,
                )
            else:
                RepoMemoryManager.ensure_exists_in_worktree(self.workspace_dir)

            # Commit baseline memory file if it is new/updated.
            self.workspace.repo.git.add([RepoMemoryManager.MEMORY_REL_PATH])
            if self.workspace.repo.is_dirty(untracked_files=True):
                self.workspace.repo.git.commit("-m", "chore(kapso): add baseline repo memory")

        if import_from_checkpoint:
            self.import_checkpoint()

        # Shared state for tracking errors
        self.previous_errors: List[str] = []
        self.recent_error_count = 10
    
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
        
        Args:
            eval_dir: Path to evaluation files (copied to kapso_evaluation/)
            data_dir: Path to data files (copied to kapso_datasets/)
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
        
        # Commit the directories to the workspace repo (use relative paths)
        self.workspace.repo.git.add(dirs_created)
        if self.workspace.repo.is_dirty(untracked_files=True):
            self.workspace.repo.git.commit("-m", "chore(kapso): setup evaluation and data directories")
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def run(self, context: ContextData, budget_progress: float = 0.0) -> Optional[ExperimentResult]:
        """
        Execute one iteration of the search strategy.
        
        Args:
            context: Problem context, KG results, experiment history
            budget_progress: 0-100 indicating budget consumed
            
        Returns:
            ExperimentResult with solution, evaluation_output, code_diff, workspace_dir
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
    ) -> str:
        """
        Have the developer agent implement a solution.
        
        The developer agent is responsible for:
        - Implementing the solution
        - Building evaluation in kapso_evaluation/
        - Running the evaluation
        - Handling any errors/retries internally
        
        Args:
            solution: The solution description to implement
            context: Problem context with KG results
            session: Experiment session with coding agent
            
        Returns:
            The agent's output (contains implementation results)
        """
        # RepoMemory is committed inside branches under `.kapso/`.
        # This means when we start from a parent branch, we also inherit the memory
        # corresponding to that code state. We still render a short briefing here
        # so coding agents don't need to rediscover basic repo structure every time.
        repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(session.session_folder)
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(repo_memory_doc, max_chars=2500)

        # By default, agents can read the JSON file directly.
        # For Claude Code specifically, we also provide a tiny CLI so it can fetch
        # a section via the existing "Bash" tool (auditable + easy to use).
        agent_type = getattr(getattr(self.workspace, "coding_agent_config", None), "agent_type", "")
        if agent_type == "claude_code":
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "use the CLI (preferred): `python3 tools/repo_memory_cli.py get-section <section_id>`\n"
                "Example: `python3 tools/repo_memory_cli.py get-section core.architecture`\n"
                "Fallback: open `.kapso/repo_memory.json` and read `book.sections[section_id]`."
            )
        else:
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "read: `.kapso/repo_memory.json` and look up by section ID from the TOC."
            )

        template = load_prompt("execution/prompts/coding_agent_implement.md")
        developer_prompt = render_prompt(
            template,
            {
                "previous_errors": "\n".join(
                    str(e) for e in self.previous_errors[-self.recent_error_count :]
                ),
                "branch_name": session.branch_name,
                "repo_memory_brief": repo_memory_brief,
                "repo_memory_detail_access_instructions": repo_memory_detail_access_instructions,
                "kg_code_results": str(getattr(context, "kg_code_results", "")),
                "problem": str(getattr(context, "problem", "")),
                "solution": str(solution or ""),
            },
        )

        # Run the developer agent - it handles implementation, evaluation, and retries
        result = session.generate_code(developer_prompt)
        return result.output if hasattr(result, 'output') else str(result)

    def _implement(
        self, 
        solution: str, 
        context: ContextData, 
        branch_name: str, 
        parent_branch_name: str = "main",
        ideation_repo_memory_sections_consulted: Optional[List[str]] = None,
    ) -> str:
        """
        Full implementation flow.
        
        Creates a session, runs the developer agent, and finalizes.
        The developer agent handles everything: implementation, evaluation, retries.
        
        Args:
            solution: Solution description to implement
            context: Problem context
            branch_name: Git branch for this experiment
            parent_branch_name: Parent branch to inherit code from
            ideation_repo_memory_sections_consulted: RepoMemory sections used during ideation
            
        Returns:
            The agent's output string
        """
        session = self.workspace.create_experiment_session(branch_name, parent_branch_name, llm=self.llm)
        agent_output = self.implement_solution(solution, context, session)

        # Update RepoMemory for this experiment branch.
        # This makes memory correct across the experiment tree: child experiments inherit
        # the memory file from the parent branch they start from.
        run_result_payload = {
            "score": 0,  # Score will be extracted by feedback generator
            "run_had_error": False,
            "error_message": "",
            "error_details": "",
            "feedbacks": "",
            "ideation_repo_memory_sections_consulted": ideation_repo_memory_sections_consulted or [],
        }

        # Observability: record which RepoMemory sections the agent claims to have consulted.
        # We instruct the agent to write this into `changes.log`.
        sections_consulted = []
        try:
            changes_log_path = os.path.join(session.session_folder, "changes.log")
            if os.path.exists(changes_log_path):
                with open(changes_log_path, "r", encoding="utf-8", errors="replace") as f:
                    sections_consulted = extract_repo_memory_sections_consulted(f.read())
        except Exception:
            sections_consulted = []
        run_result_payload["repo_memory_sections_consulted"] = sections_consulted

        # Schedule RepoMemory update for session close.
        session.schedule_repo_memory_update(
            solution_spec=solution,
            run_result=run_result_payload,
        )
        
        self.workspace.finalize_session(session)
        return agent_output
