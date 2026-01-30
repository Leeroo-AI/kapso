# Experiment Workspace - Manages Experiment Sessions with Pluggable Coding Agents
#
# This class manages:
# - Git workspace for experiments
# - Branch management per experiment
# - Session creation with pluggable coding agents
# - Cost tracking across sessions

import os
import uuid
import shutil
import threading
from typing import Optional

import git

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.experiment_workspace.experiment_session import ExperimentSession


class ExperimentWorkspace:
    """
    Manages experiment sessions with pluggable coding agents.
    
    Creates isolated git workspaces for experimentation. Each experiment
    runs in its own branch, allowing tree-based exploration of solutions.
    
    Supports multiple coding agents (Aider, Gemini, Claude Code, OpenHands)
    through the CodingAgentConfig system.
    """
    

    def __init__(
        self,
        coding_agent_config: CodingAgentConfig,
        workspace_dir: str,
        initial_repo: Optional[str] = None,
    ):
        """
        Initialize the Experiment Workspace.
        
        Args:
            coding_agent_config: Configuration for the coding agent (required)
            workspace_dir: Path to the workspace directory (required)
            initial_repo: Optional local filesystem path to a repository to COPY/CLONE
                into this workspace. This enables "improve an existing repo" workflows.
        """
        
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.initial_repo = os.path.abspath(initial_repo) if initial_repo else None
        self.is_seeded = self.initial_repo is not None
        
        # Initialize git repository.
        #
        # Two modes:
        # - Empty workspace (default): start from a fresh git repo (used by many benchmarks).
        # - Seeded workspace: start from an existing local repo (copy/clone) so experiments
        #   mutate an input codebase rather than creating everything from scratch.
        if self.is_seeded:
            self.repo = self._init_from_seed_repo(self.initial_repo)
        else:
            self.repo = git.Repo.init(self.workspace_dir)

        # Repo-local git config that helps push branches back into this workspace repo.
        # This is intentionally local-only (not committed).
        with self.repo.config_writer() as git_config:
            git_config.set_value("user", "name", "Experiment Workspace")
            git_config.set_value("user", "email", "workspace@experiment.com")
            # Needed because we may push to non-bare repos (this workspace is a working repo).
            git_config.set_value("receive", "denyCurrentBranch", "ignore")
        
        # Store coding agent config
        self.coding_agent_config = coding_agent_config
        
        # Cost tracking
        self.previous_sessions_cost = 0
        self.repo_lock = threading.Lock()
        
        # Ensure we have a stable baseline branch called "main".
        # Many parts of the execution engine assume "main" exists and is the default parent.
        self._ensure_main_branch()

        # Ensure `sessions/` is ignored in the workspace repo.
        # Sessions contain nested git clones and should never appear as "untracked noise"
        # in the workspace repo status.
        self._ensure_workspace_gitignore()

    @classmethod
    def with_default_config(
        cls,
        workspace_dir: Optional[str] = None,
        initial_repo: Optional[str] = None,
    ) -> 'ExperimentWorkspace':
        """
        Create ExperimentWorkspace with default coding agent from agents.yaml.
        
        Returns:
            ExperimentWorkspace configured with default agent
        """
        config = CodingAgentFactory.build_config()
        # Keep this helper usable in standalone scripts.
        # If workspace_dir is not provided, create a unique temp path.
        workspace_dir = workspace_dir or os.path.join("tmp", "experiment_workspace", str(uuid.uuid4()))
        return cls(coding_agent_config=config, workspace_dir=workspace_dir, initial_repo=initial_repo)

    def get_current_branch(self) -> str:
        """Get the current active branch name."""
        return self.repo.active_branch.name
    
    def switch_branch(self, branch_name: str) -> None:
        """
        Switch to an existing branch.
        
        Args:
            branch_name: Name of branch to switch to
        """
        # Clean untracked files that might block checkout (e.g., __pycache__)
        # Use -f to force, -d to remove directories, -x to remove ignored files too
        try:
            self.repo.git.clean('-fdx')
        except Exception:
            pass  # Best effort - continue even if clean fails
        self.repo.git.checkout(branch_name)
    
    def create_branch(self, branch_name: str) -> None:
        """
        Create and switch to a new branch.
        
        Args:
            branch_name: Name for the new branch
        """
        self.repo.git.checkout('-b', branch_name)

    # =========================================================================
    # Seeding / bootstrap helpers
    # =========================================================================

    def _init_from_seed_repo(self, initial_repo: str) -> git.Repo:
        """
        Initialize this workspace from an existing local repository path.
        
        IMPORTANT DESIGN NOTE:
        - We do NOT mutate the seed repo in-place.
        - We clone/copy it into this workspace directory so we can diff "evolved"
          branches against the baseline without touching the original.
        """
        if not os.path.exists(initial_repo):
            raise FileNotFoundError(f"Initial repo path does not exist: {initial_repo}")

        # If seed is a git repo, do a proper git clone to preserve history.
        # Otherwise, copy the directory and initialize a new git repo.
        try:
            _ = git.Repo(initial_repo)
            is_git_repo = True
        except Exception:
            is_git_repo = False

        # Workspace dir must be empty before we populate it.
        if os.path.exists(self.workspace_dir) and os.listdir(self.workspace_dir):
            raise ValueError(
                f"Workspace directory must be empty to seed it: {self.workspace_dir}"
            )

        if is_git_repo:
            repo = git.Repo.clone_from(initial_repo, self.workspace_dir)
        else:
            shutil.copytree(initial_repo, self.workspace_dir, dirs_exist_ok=True)
            repo = git.Repo.init(self.workspace_dir)
            repo.git.add(".")
            repo.git.commit("-m", "chore(kapso): seed workspace from directory")

        return repo

    def _ensure_main_branch(self) -> None:
        """
        Ensure the workspace has a branch named "main" checked out.
        
        This keeps downstream logic simple because ExperimentSession defaults to
        parent_branch_name="main".
        """
        try:
            current = self.repo.active_branch.name
        except Exception:
            # Detached HEAD or unusual repo state - create main at HEAD.
            self.repo.git.checkout("-b", "main")
            return

        if current != "main":
            # Force rename current branch to main (works even if current is "master").
            self.repo.git.branch("-M", "main")
        else:
            self.repo.git.checkout("main")

    def _ensure_workspace_gitignore(self) -> None:
        """
        Ensure `.gitignore` includes patterns needed by the experimentation engine.
        
        We append patterns instead of overwriting existing .gitignore, because
        seeded repos often have important ignore rules already.
        """
        gitignore_path = os.path.join(self.workspace_dir, ".gitignore")
        # Keep session clones out of the workspace repo.
        # Keep generic logs ignored, BUT explicitly allow `changes.log` so it is committed
        # into experiment branches as an audit trail (RepoMemory observability).
        #
        # IMPORTANT: Ordering matters in .gitignore.
        # The negation rule (`!changes.log`) must appear after `*.log`.
        required_lines = ["sessions/*", "*.log", "!changes.log"]

        existing = ""
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                existing = f.read()

        to_add = [line for line in required_lines if line not in existing]
        if not to_add:
            return

        # Append with a clear marker so humans know this is infrastructure.
        with open(gitignore_path, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n# Kapso experimentation engine\n")
            for line in to_add:
                f.write(line + "\n")

        # Commit the ignore change so all experiment branches inherit it.
        self.repo.git.add([".gitignore"])
        try:
            self.repo.git.commit("-m", "chore(kapso): ignore experiment sessions")
        except git.GitCommandError:
            # Nothing to commit (rare). Keep silent.
            pass
    
    def create_experiment_session(
        self, 
        branch_name: str, 
        parent_branch_name: str = "main",
        llm=None,
    ) -> ExperimentSession:
        """
        Create a new experiment session.
        
        Each session:
        - Clones the repo to an isolated folder
        - Checks out from parent branch (inherits parent's code)
        - Creates a new experiment branch
        - Uses the configured coding agent
        
        Args:
            branch_name: Name for the experiment branch
            parent_branch_name: Branch to inherit code from
            
        Returns:
            ExperimentSession ready for code generation
        """
        print(f"Creating experiment session for branch {branch_name} with parent {parent_branch_name}")
        
        session_folder = os.path.join(self.workspace_dir, 'sessions', branch_name)
        
        # Create session with coding agent config
        session = ExperimentSession(
            main_repo=self.repo,
            session_folder=session_folder,
            coding_agent_config=self.coding_agent_config,
            parent_branch_name=parent_branch_name,
            branch_name=branch_name,
            repo_memory_llm=llm,
        )
        
        return session
    
    def finalize_session(self, session: ExperimentSession) -> None:
        """
        Finalize an experiment session.
        
        Collects cost and closes the session (commits, pushes, cleanup).
        
        Args:
            session: The session to finalize
        """
        cost = session.get_cumulative_cost()
        with self.repo_lock:
            self.previous_sessions_cost += cost
            session.close_session()
    
    def cleanup(self) -> None:
        """
        Clean up the entire workspace.
        
        Removes the workspace folder and all its contents.
        """
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def get_cumulative_cost(self) -> float:
        """
        Get total cost across all sessions.
        
        Returns:
            Total cost in dollars
        """
        return self.previous_sessions_cost


if __name__ == "__main__":
    # Test with default agent from agents.yaml
    print("Testing ExperimentWorkspace with default config...")
    
    workspace = ExperimentWorkspace.with_default_config()
    print(f"Workspace: {workspace.workspace_dir}")
    print(f"Agent type: {workspace.coding_agent_config.agent_type}")
    
    session = workspace.create_experiment_session("test_branch")
    result = session.generate_code("implement a main.py file that prints 'Hello World'")
    
    print(f"Success: {result['success']}")
    print(f"Code: {result['code'][:200]}..." if result['code'] else "No code")
    if result['error']:
        print(f"Error: {result['error']}")
    
    workspace.finalize_session(session)
    print(f"Cumulative cost: ${workspace.get_cumulative_cost():.4f}")
    
    # Cleanup
    workspace.cleanup()
    print("Done!")

