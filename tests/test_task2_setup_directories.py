#!/usr/bin/env python3
"""
Task 2: Setup Directories - Tests

Tests for the kapso_evaluation/ and kapso_datasets/ directory setup:
1. Both directories provided
2. Only eval_dir provided
3. Only data_dir provided
4. Neither provided (empty directories created)
5. Directories committed and inherited by branches

Usage:
    conda activate praxium_conda
    python -m pytest tests/test_task2_setup_directories.py -v
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.search_strategies.base import SearchStrategyConfig
from src.execution.search_strategies.factory import SearchStrategyFactory
from src.execution.coding_agents.factory import CodingAgentFactory
from src.environment.handlers.generic import GenericProblemHandler
from src.core.llm import LLMBackend


class TestSetupKapsoDirectories:
    """Test the _setup_kapso_directories method."""
    
    @classmethod
    def setup_class(cls):
        """Create test fixtures once for all tests."""
        # Create test eval_dir with some files
        cls.eval_dir = tempfile.mkdtemp(prefix="test_eval_")
        with open(os.path.join(cls.eval_dir, "evaluate.py"), "w") as f:
            f.write("# Evaluation script\nprint('score: 0.95')\n")
        with open(os.path.join(cls.eval_dir, "config.yaml"), "w") as f:
            f.write("threshold: 0.9\n")
        
        # Create test data_dir with some files
        cls.data_dir = tempfile.mkdtemp(prefix="test_data_")
        with open(os.path.join(cls.data_dir, "train.csv"), "w") as f:
            f.write("id,label\n1,positive\n2,negative\n")
        os.makedirs(os.path.join(cls.data_dir, "subdir"))
        with open(os.path.join(cls.data_dir, "subdir", "test.csv"), "w") as f:
            f.write("id,label\n3,positive\n")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test fixtures."""
        shutil.rmtree(cls.eval_dir, ignore_errors=True)
        shutil.rmtree(cls.data_dir, ignore_errors=True)
    
    def _create_strategy(self, workspace_dir: str, eval_dir=None, data_dir=None):
        """Helper to create a search strategy with given directories."""
        handler = GenericProblemHandler(
            problem_description="Test problem",
            main_file="main.py",
            language="python",
        )
        coding_agent_config = CodingAgentFactory.build_config()
        llm = LLMBackend()
        
        return SearchStrategyFactory.create(
            strategy_type="linear_search",
            problem_handler=handler,
            llm=llm,
            coding_agent_config=coding_agent_config,
            workspace_dir=workspace_dir,
            eval_dir=eval_dir,
            data_dir=data_dir,
        )
    
    def test_both_directories_provided(self):
        """Test with both eval_dir and data_dir provided."""
        workspace = tempfile.mkdtemp(prefix="test_workspace_")
        try:
            strategy = self._create_strategy(
                workspace_dir=workspace,
                eval_dir=self.eval_dir,
                data_dir=self.data_dir,
            )
            
            # Check kapso_evaluation/ exists and has files
            kapso_eval = os.path.join(workspace, "kapso_evaluation")
            assert os.path.exists(kapso_eval), "kapso_evaluation/ should exist"
            assert os.path.exists(os.path.join(kapso_eval, "evaluate.py")), "evaluate.py should be copied"
            assert os.path.exists(os.path.join(kapso_eval, "config.yaml")), "config.yaml should be copied"
            
            # Check kapso_datasets/ exists and has files
            kapso_data = os.path.join(workspace, "kapso_datasets")
            assert os.path.exists(kapso_data), "kapso_datasets/ should exist"
            assert os.path.exists(os.path.join(kapso_data, "train.csv")), "train.csv should be copied"
            assert os.path.exists(os.path.join(kapso_data, "subdir", "test.csv")), "subdir/test.csv should be copied"
            
            # Check directories are committed
            repo = strategy.workspace.repo
            # Get list of tracked files
            tracked = repo.git.ls_files().split('\n')
            assert any("kapso_evaluation" in f for f in tracked), "kapso_evaluation should be tracked"
            assert any("kapso_datasets" in f for f in tracked), "kapso_datasets should be tracked"
            
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    
    def test_only_eval_dir_provided(self):
        """Test with only eval_dir provided."""
        workspace = tempfile.mkdtemp(prefix="test_workspace_")
        try:
            strategy = self._create_strategy(
                workspace_dir=workspace,
                eval_dir=self.eval_dir,
                data_dir=None,
            )
            
            # Check kapso_evaluation/ has files
            kapso_eval = os.path.join(workspace, "kapso_evaluation")
            assert os.path.exists(os.path.join(kapso_eval, "evaluate.py"))
            
            # Check kapso_datasets/ exists but is empty (has .gitkeep)
            kapso_data = os.path.join(workspace, "kapso_datasets")
            assert os.path.exists(kapso_data)
            assert os.path.exists(os.path.join(kapso_data, ".gitkeep"))
            
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    
    def test_only_data_dir_provided(self):
        """Test with only data_dir provided."""
        workspace = tempfile.mkdtemp(prefix="test_workspace_")
        try:
            strategy = self._create_strategy(
                workspace_dir=workspace,
                eval_dir=None,
                data_dir=self.data_dir,
            )
            
            # Check kapso_evaluation/ exists but is empty (has .gitkeep)
            kapso_eval = os.path.join(workspace, "kapso_evaluation")
            assert os.path.exists(kapso_eval)
            assert os.path.exists(os.path.join(kapso_eval, ".gitkeep"))
            
            # Check kapso_datasets/ has files
            kapso_data = os.path.join(workspace, "kapso_datasets")
            assert os.path.exists(os.path.join(kapso_data, "train.csv"))
            
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    
    def test_neither_directory_provided(self):
        """Test with neither eval_dir nor data_dir provided."""
        workspace = tempfile.mkdtemp(prefix="test_workspace_")
        try:
            strategy = self._create_strategy(
                workspace_dir=workspace,
                eval_dir=None,
                data_dir=None,
            )
            
            # Both directories should exist with .gitkeep
            kapso_eval = os.path.join(workspace, "kapso_evaluation")
            kapso_data = os.path.join(workspace, "kapso_datasets")
            
            assert os.path.exists(kapso_eval)
            assert os.path.exists(kapso_data)
            assert os.path.exists(os.path.join(kapso_eval, ".gitkeep"))
            assert os.path.exists(os.path.join(kapso_data, ".gitkeep"))
            
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    
    def test_directories_inherited_by_branches(self):
        """Test that directories are inherited by experiment branches."""
        workspace = tempfile.mkdtemp(prefix="test_workspace_")
        try:
            strategy = self._create_strategy(
                workspace_dir=workspace,
                eval_dir=self.eval_dir,
                data_dir=self.data_dir,
            )
            
            # Create a new branch
            repo = strategy.workspace.repo
            repo.git.checkout("-b", "test_branch")
            
            # Directories should still exist on the new branch
            kapso_eval = os.path.join(workspace, "kapso_evaluation")
            kapso_data = os.path.join(workspace, "kapso_datasets")
            
            assert os.path.exists(os.path.join(kapso_eval, "evaluate.py"))
            assert os.path.exists(os.path.join(kapso_data, "train.csv"))
            
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    
    def test_nonexistent_eval_dir_ignored(self):
        """Test that non-existent eval_dir is handled gracefully."""
        workspace = tempfile.mkdtemp(prefix="test_workspace_")
        try:
            strategy = self._create_strategy(
                workspace_dir=workspace,
                eval_dir="/nonexistent/path/12345",
                data_dir=None,
            )
            
            # kapso_evaluation/ should exist but be empty (with .gitkeep)
            kapso_eval = os.path.join(workspace, "kapso_evaluation")
            assert os.path.exists(kapso_eval)
            assert os.path.exists(os.path.join(kapso_eval, ".gitkeep"))
            
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


class TestKapsoEvolveWithDirectories:
    """Test Kapso.evolve() with eval_dir and data_dir parameters."""
    
    @classmethod
    def setup_class(cls):
        """Create test fixtures."""
        cls.eval_dir = tempfile.mkdtemp(prefix="test_eval_")
        with open(os.path.join(cls.eval_dir, "evaluate.py"), "w") as f:
            f.write("print('SCORE: 1.0')\n")
        
        cls.data_dir = tempfile.mkdtemp(prefix="test_data_")
        with open(os.path.join(cls.data_dir, "data.txt"), "w") as f:
            f.write("test data\n")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup."""
        shutil.rmtree(cls.eval_dir, ignore_errors=True)
        shutil.rmtree(cls.data_dir, ignore_errors=True)
    
    def test_evolve_accepts_eval_dir_and_data_dir(self):
        """Test that evolve() accepts the new parameters."""
        from src.kapso import Kapso
        import inspect
        
        sig = inspect.signature(Kapso.evolve)
        params = list(sig.parameters.keys())
        
        assert "eval_dir" in params, "evolve() should accept eval_dir"
        assert "data_dir" in params, "evolve() should accept data_dir"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
