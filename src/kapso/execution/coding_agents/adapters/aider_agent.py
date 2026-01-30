# Aider Coding Agent Adapter
#
# Wraps the Aider library for use in the orchestrator.
# Aider is Git-centric with native diff-based editing.
#
# Key features:
# - Native git support (auto-commits after each edit)
# - Diff-based editing format
# - Cost tracking via coder.total_cost
# - Separate models for implement/debug

import os
from typing import Dict, List, Optional
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from kapso.execution.coding_agents.base import (
    CodingAgentInterface, 
    CodingAgentConfig, 
    CodingResult
)


class AiderCodingAgent(CodingAgentInterface):
    """
    Aider-based coding agent.
    
    Aider is a pair-programming agent that excels at small, atomic code edits.
    It's Git-centric, automatically staging and committing changes after
    every successful interaction.
    
    Features:
    - Native Git integration (use_git=True)
    - Diff-based editing for efficient changes
    - Cost tracking via coder.total_cost
    - Separate coders for implement/debug (can use different models)
    
    Configuration (agent_specific):
    - edit_format: "diff" (default), "whole", "udiff"
    - auto_commits: True (default)
    """
    
    def __init__(self, config: CodingAgentConfig):
        """Initialize Aider coding agent."""
        super().__init__(config)
        self.coder: Optional[Coder] = None
        self.debug_coder: Optional[Coder] = None
        self.workspace: Optional[str] = None
        
        # Get Aider-specific settings
        self._edit_format = config.agent_specific.get("edit_format", "diff")
        self._auto_commits = config.agent_specific.get("auto_commits", True)
    
    def _get_files_in_workspace(self, workspace: str) -> List[str]:
        """
        Get list of Python/code files in workspace.
        
        If no files exist, returns empty list (Aider will create them).
        """
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        files = []
        
        if os.path.isdir(workspace):
            for item in os.listdir(workspace):
                item_path = os.path.join(workspace, item)
                if os.path.isfile(item_path):
                    _, ext = os.path.splitext(item)
                    if ext.lower() in code_extensions:
                        files.append(item_path)
        
        return files
    
    def initialize(self, workspace: str) -> None:
        """
        Initialize Aider coders for the workspace.
        
        Creates two Aider Coder instances:
        - Main coder: For implementing solutions
        - Debug coder: For fixing errors (can use different model)
        
        CRITICAL: We must change to the workspace directory BEFORE creating
        Aider Coders. This ensures Aider discovers the session's git repo,
        not the parent project's git repo (which may have blocking .gitignore).
        
        Args:
            workspace: Path to the working directory
        """
        self.workspace = os.path.abspath(workspace)
        
        try:
            fnames = [self.workspace] 
            
            # Check if this workspace has its own git repo
            use_git = self._auto_commits and self._is_git_repo(self.workspace)
            
            # Create main coder for implementation
            # Since we're in the workspace dir, Aider will discover the session's git repo
            io = InputOutput(yes=True)  # Non-interactive mode
            main_model = Model(self.config.model)
            self.coder = Coder.create(
                main_model=main_model,
                fnames=fnames if fnames else None,
                io=io,
                stream=False,
                use_git=use_git,
                edit_format=self._edit_format,
            )
            # Ensure root is set to workspace
            if hasattr(self.coder, 'root'):
                self.coder.root = self.workspace
            
            # Create debug coder (potentially different model)
            io = InputOutput(yes=True)
            debug_model = Model(self.config.debug_model)
            self.debug_coder = Coder.create(
                main_model=debug_model,
                fnames=fnames if fnames else None,
                io=io,
                stream=False,
                use_git=use_git,
                edit_format=self._edit_format,
            )
            if hasattr(self.debug_coder, 'root'):
                self.debug_coder.root = self.workspace
        finally:
            pass
    
    def _is_git_repo(self, path: str) -> bool:
        """Check if path is inside a git repository."""
        git_dir = os.path.join(path, '.git')
        return os.path.exists(git_dir)
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """
        Generate code using Aider.
        
        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug coder
            
        Returns:
            CodingResult with Aider's response and cost
        """
        coder = self.debug_coder if debug_mode else self.coder
        
        if coder is None:
            return CodingResult(
                success=False,
                output="",
                error="Agent not initialized. Call initialize() first."
            )
        
        try:
            
            # Run Aider with the prompt
            result = coder.run(prompt)
            
            # Update cumulative cost from both coders
            self._cumulative_cost = (
                (self.coder.total_cost if self.coder else 0) +
                (self.debug_coder.total_cost if self.debug_coder else 0)
            )
            
            # Get list of changed files
            files_changed = []
            if hasattr(coder, 'abs_fnames'):
                files_changed = list(coder.abs_fnames)
            
            return CodingResult(
                success=True,
                output=result or "",
                files_changed=files_changed,
                cost=coder.total_cost,
                metadata={
                    "model": coder.main_model.name,
                    "edit_format": self._edit_format,
                }
            )
        except Exception as e:
            return CodingResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def cleanup(self) -> None:
        """Clean up Aider resources."""
        self.coder = None
        self.debug_coder = None
        self.workspace = None
    
    def supports_native_git(self) -> bool:
        """
        Aider handles its own git commits.
        
        When use_git=True (default), Aider automatically commits
        after each successful code edit with a descriptive message.
        """
        return self._auto_commits
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return Aider's capabilities."""
        return {
            "native_git": self._auto_commits,
            "sandbox": False,
            "planning_mode": False,
            "cost_tracking": True,
            "streaming": True,
        }
