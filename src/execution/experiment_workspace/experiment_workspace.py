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

from src.execution.coding_agents.base import CodingAgentConfig
from src.execution.coding_agents.factory import CodingAgentFactory
from src.execution.experiment_workspace.experiment_session import ExperimentSession


class ExperimentWorkspace:
    """
    Manages experiment sessions with pluggable coding agents.
    
    Creates isolated git workspaces for experimentation. Each experiment
    runs in its own branch, allowing tree-based exploration of solutions.
    
    Supports multiple coding agents (Aider, Gemini, Claude Code, OpenHands)
    through the CodingAgentConfig system.
    """
    
    WORKSPACE_FOLDER_BASE = 'tmp/experiment_workspace'

    def __init__(self, coding_agent_config: CodingAgentConfig):
        """
        Initialize the Experiment Workspace.
        
        Args:
            coding_agent_config: Configuration for the coding agent (required)
        """
        self.uuid = str(uuid.uuid4())
        self.workspace_folder = os.path.join(self.WORKSPACE_FOLDER_BASE, self.uuid)
        os.makedirs(self.workspace_folder, exist_ok=True)
        
        # Initialize git repository
        self.repo = git.Repo.init(self.workspace_folder)
        with self.repo.config_writer() as git_config:
            git_config.set_value("user", "name", "Experiment Workspace")
            git_config.set_value("user", "email", "workspace@experiment.com")
            git_config.set_value("receive", "denyCurrentBranch", "ignore")
        
        # Store coding agent config
        self.coding_agent_config = coding_agent_config
        
        # Cost tracking
        self.previous_sessions_cost = 0
        self.repo_lock = threading.Lock()
        
        # Initialize main branch
        self.create_branch('main')
        with open(os.path.join(self.workspace_folder, '.gitignore'), 'w') as f:
            f.write('sessions/*\n*.log')
        self.repo.git.add(['.gitignore'])
        self.repo.git.commit('-m', 'chore: initialize repository')

    @classmethod
    def with_default_config(cls) -> 'ExperimentWorkspace':
        """
        Create ExperimentWorkspace with default coding agent from agents.yaml.
        
        Returns:
            ExperimentWorkspace configured with default agent
        """
        config = CodingAgentFactory.build_config()
        return cls(coding_agent_config=config)

    def get_current_branch(self) -> str:
        """Get the current active branch name."""
        return self.repo.active_branch.name
    
    def switch_branch(self, branch_name: str) -> None:
        """
        Switch to an existing branch.
        
        Args:
            branch_name: Name of branch to switch to
        """
        self.repo.git.checkout(branch_name)
    
    def create_branch(self, branch_name: str) -> None:
        """
        Create and switch to a new branch.
        
        Args:
            branch_name: Name for the new branch
        """
        self.repo.git.checkout('-b', branch_name)
    
    def create_experiment_session(
        self, 
        branch_name: str, 
        parent_branch_name: str = "main"
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
        
        session_folder = os.path.join(self.workspace_folder, 'sessions', branch_name)
        
        # Create session with coding agent config
        session = ExperimentSession(
            main_repo=self.repo,
            session_folder=session_folder,
            coding_agent_config=self.coding_agent_config,
            parent_branch_name=parent_branch_name,
            branch_name=branch_name,
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
        shutil.rmtree(self.workspace_folder, ignore_errors=True)

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
    print(f"Workspace: {workspace.workspace_folder}")
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

