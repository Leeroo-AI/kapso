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
    seed_repo_path: Optional[str] = None


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
            seed_repo_path=config.seed_repo_path,
        )

        # Ensure baseline RepoMemory exists in the workspace repo.
        #
        # - For seeded repos: build an evidence-backed RepoModel once at start so
        #   ideation and implementation can be grounded in the repo's actual design.
        # - For empty workspaces: create a lightweight skeleton (RepoMap only).
        #
        # RepoMemory is committed into the workspace's "main" branch under `.praxium/`,
        # so all experiment branches inherit it automatically.
        if not import_from_checkpoint:
            # Build baseline RepoMemory and commit it to the workspace's main branch.
            # - For seeded repos: build evidence-backed RepoModel via LLM.
            # - For empty workspaces: create a lightweight skeleton (RepoMap only).
            if self.workspace.is_seeded:
                RepoMemoryManager.bootstrap_baseline_model(
                    repo_root=self.workspace_dir,
                    llm=self.llm,
                    seed_repo_path=self.workspace.seed_repo_path,
                )
            else:
                RepoMemoryManager.ensure_exists_in_worktree(self.workspace_dir)

            # Commit baseline memory file if it is new/updated.
            self.workspace.repo.git.add([RepoMemoryManager.MEMORY_REL_PATH])
            if self.workspace.repo.is_dirty(untracked_files=True):
                self.workspace.repo.git.commit("-m", "chore(praxium): add baseline repo memory")

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
        # RepoMemory is committed inside branches under `.praxium/`.
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
                "Fallback: open `.praxium/repo_memory.json` and read `book.sections[section_id]`."
            )
        else:
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "read: `.praxium/repo_memory.json` and look up by section ID from the TOC."
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
        repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(session.session_folder)
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(repo_memory_doc, max_chars=2500)

        agent_type = getattr(getattr(self.workspace, "coding_agent_config", None), "agent_type", "")
        if agent_type == "claude_code":
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "use the CLI (preferred): `python3 tools/repo_memory_cli.py get-section <section_id>`\n"
                "Example: `python3 tools/repo_memory_cli.py get-section core.architecture`\n"
                "Fallback: open `.praxium/repo_memory.json` and read `book.sections[section_id]`."
            )
        else:
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "read: `.praxium/repo_memory.json` and look up by section ID from the TOC."
            )

        template = load_prompt("execution/prompts/coding_agent_debug.md")
        developer_prompt = render_prompt(
            template,
            {
                "repo_memory_brief": repo_memory_brief,
                "repo_memory_detail_access_instructions": repo_memory_detail_access_instructions,
                "problem": str(getattr(context, "problem", "")),
                "solution": str(solution or ""),
                "error_details": str(error or ""),
            },
        )
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
        parent_branch_name: str = "main",
        ideation_repo_memory_sections_consulted: Optional[List[str]] = None,
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
        session = self.workspace.create_experiment_session(branch_name, parent_branch_name, llm=self.llm)
        result = self.implement_solution(solution, context, session)
        
        for i in range(code_debug_tries):
            if result.run_had_error and result.continue_debugging:
                print("Run had error, error details: ", result.error_details)
                result = self.debug_solution(solution, context, result.error_details, session)
            else:
                break

        # Update RepoMemory for this experiment branch.
        # This makes memory correct across the experiment tree: child experiments inherit
        # the memory file from the parent branch they start from.
        run_result_payload = {
            "score": getattr(result, "score", 0),
            "run_had_error": getattr(result, "run_had_error", False),
            "error_message": getattr(result, "error_message", "")[:5000],
            "error_details": getattr(result, "error_details", "")[:10000],
            "feedbacks": getattr(result, "feedbacks", "")[:10000],
            # Observability: which RepoMemory sections were consulted during ideation (engine-mediated).
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
        # This ensures the update runs AFTER final commits (run data + remaining changes),
        # and BEFORE push/cleanup (latest-commit semantics).
        session.schedule_repo_memory_update(
            solution_spec=solution,
            run_result=run_result_payload,
        )
        
        self.workspace.finalize_session(session)
        return result
