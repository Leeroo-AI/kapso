# Experiment Session with Pluggable Coding Agents
#
# This class manages a single experiment session with:
# - Git branch setup/teardown (agent-agnostic)
# - Pluggable coding agent for code generation
# - Commit message generation for meaningful commits
#
# Key design: ExperimentSession owns ALL git operations.
# Coding agents only generate code - they don't handle git.

import os
import shutil
import copy
from typing import Dict, Optional

import git

from src.execution.coding_agents.base import CodingAgentConfig, CodingResult
from src.execution.coding_agents.factory import CodingAgentFactory
from src.execution.coding_agents.commit_message_generator import CommitMessageGenerator


class ExperimentSession:
    """
    Experiment session with pluggable coding agent.
    
    Supports multiple coding agents (Aider, Gemini, Claude Code, OpenHands)
    through the CodingAgentInterface abstraction.
    
    This class owns ALL git operations:
    - Branch setup (clone, checkout parent, create branch)
    - Commits after code generation (if agent doesn't handle it)
    - Push on session close
    
    The coding agent only generates code - no git responsibility.
    """
    
    def __init__(
        self, 
        main_repo: git.Repo, 
        session_folder: str, 
        coding_agent_config: CodingAgentConfig,
        parent_branch_name: str, 
        branch_name: str,
    ):
        """
        Initialize an experiment session.
        
        Args:
            main_repo: The main git repository
            session_folder: Path to session working directory
            coding_agent_config: Configuration for the coding agent
            parent_branch_name: Branch to inherit code from
            branch_name: Name for this experiment's branch
        """
        self.main_repo = main_repo
        self.session_folder = session_folder
        self.branch_name = branch_name
        self.parent_branch_name = parent_branch_name
        
        # === GIT SETUP ===
        os.makedirs(os.path.dirname(self.session_folder), exist_ok=True)
        
        # Clone from main repo to isolated session folder
        self.repo = git.Repo.clone_from(
            f"file://{main_repo.working_dir}", 
            self.session_folder
        )
        with self.repo.config_writer() as git_config:
            git_config.set_value("user", "name", branch_name)
            git_config.set_value("user", "email", branch_name+"@experiment.com")
        
        # CRITICAL: Checkout parent branch first (inherit parent's code)
        self.repo.git.checkout(parent_branch_name)
        
        # Create new branch from parent
        if branch_name in [ref.name for ref in self.repo.heads]:
            self.repo.git.checkout(branch_name)
        else:
            self.repo.git.checkout('-b', branch_name)

        # Record the base commit SHA for this experiment branch.
        # This is the exact repo state we "started from" (inherited from parent_branch_name).
        # We use it to compute diffs and update RepoMemory with an accurate change log.
        self.base_commit_sha = self.repo.head.commit.hexsha
        
        # Create run directory for output data
        self.run_dir = os.path.join(self.session_folder, f"output_data_{branch_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # === CREATE CODING AGENT ===
        # Create a deep copy of config with workspace set (don't mutate original)
        session_config = copy.deepcopy(coding_agent_config)
        session_config.workspace = self.session_folder
        
        # Create agent via factory
        self.coding_agent = CodingAgentFactory.create(session_config)
        self.coding_agent.initialize(self.session_folder)
        
        # Track if agent handles its own commits
        self._agent_handles_git = self.coding_agent.supports_native_git()
        
        # Initialize commit message generator
        self.commit_generator = CommitMessageGenerator()
        
        # Store solution context for richer commit messages
        self._current_solution_summary: Optional[str] = None
    
    def set_solution_context(self, solution_summary: str) -> None:
        """
        Set context for richer commit messages.
        
        Args:
            solution_summary: Summary of what's being implemented
        """
        self._current_solution_summary = solution_summary
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> Dict[str, str]:
        """
        Generate code using the configured coding agent.
        
        This is the unified interface - works with any registered agent.
        Handles git commits for agents that don't manage their own.
        
        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug mode/model
            
        Returns:
            Dictionary with success, code, error, files_changed, cost
        """
        result: CodingResult = self.coding_agent.generate_code(prompt, debug_mode)
        
        # === COMMIT HANDLING ===
        # If agent doesn't handle git, we commit the changes with meaningful message
        if not self._agent_handles_git and result.success:
            self._commit_with_message(result)
        
        return {
            'success': result.success,
            'code': result.output,
            'error': result.error or '',
            'files_changed': result.files_changed,
            'cost': result.cost,
        }
    
    def _commit_with_message(self, result: CodingResult) -> None:
        """
        Commit changes with a meaningful message.
        
        Uses hybrid approach:
        1. Agent's suggestion if provided
        2. Generate from diff + context
        
        Args:
            result: CodingResult from the coding agent
        """
        if not self.repo.is_dirty(untracked_files=True):
            return
        
        try:
            # Stage all changes first
            self.repo.git.add('.')
            
            # Get diff for message generation
            diff = self.repo.git.diff('--cached')
            
            # Generate meaningful commit message
            message = self.commit_generator.generate(
                diff=diff,
                solution_summary=self._current_solution_summary,
                agent_suggestion=result.commit_message,
            )
            
            # Commit with generated message
            self.repo.git.commit('-m', message)
        except git.GitCommandError as e:
            # Nothing to commit
            if "nothing to commit" in str(e).lower():
                pass
            else:
                print(f"[ExperimentSession] Commit failed: {e}")
    
    def commit_folder(self, folder_name: str) -> None:
        """
        Commit a specific folder (for output data).
        
        Args:
            folder_name: Path to folder to commit
        """
        if os.path.exists(folder_name) and os.listdir(folder_name):
            print('*'*100)
            print(f"Committing folder: {folder_name}")
            try:
                self.repo.git.add(folder_name)
                self.repo.git.commit('-m', 'chore: commit run data')
            except git.GitCommandError:
                pass  # Nothing to commit or folder empty
    
    def close_session(self) -> None:
        """
        Close session: final commit, push, cleanup.
        
        Ensures all changes are committed and pushed before cleanup.
        """
        # Commit run directory
        self.commit_folder(self.run_dir)
        
        # Final commit of any remaining changes
        if self.repo.is_dirty(untracked_files=True):
            self.repo.git.add('.')
            try:
                self.repo.git.commit('-m', 'chore: final session commit')
            except git.GitCommandError:
                pass
        
        # Push to origin (makes branch available for child nodes)
        try:
            self.repo.git.push('origin', self.branch_name)
        except git.GitCommandError as e:
            print(f"[ExperimentSession] Push failed: {e}")
        
        # Cleanup coding agent resources
        self.coding_agent.cleanup()
        
        # Remove session folder
        shutil.rmtree(self.session_folder, ignore_errors=True)
    
    def get_cumulative_cost(self) -> float:
        """
        Get cumulative cost from the coding agent.
        
        Returns:
            Total cost in dollars
        """
        return self.coding_agent.get_cumulative_cost()

