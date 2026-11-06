#!/usr/bin/env python3
"""
The initial repository a build starts from

Tests for the initial_repo resolution logic in Kapso.build():
1. Local path handling
2. GitHub URL cloning
3. Workflow search integration (real KG indexing)
4. Empty repo fallback

Usage:
    conda activate praxium_conda
    python -m pytest tests/test_task1_initialize_repo.py -v
"""

import os
import shutil
import tempfile

import git
import pytest
from dotenv import load_dotenv

# Load environment variables (OpenAI API key)
load_dotenv()

# Test the helper methods directly
from kapso.kapso import Kapso


class TestInitialRepoResolution:
    """Test the _resolve_initial_repo and helper methods."""
    
    def setup_method(self):
        """Create a Kapso instance for testing."""
        self.kapso = Kapso()
    
    def test_is_github_url_https(self):
        """Test GitHub URL detection for https URLs."""
        assert self.kapso._is_github_url("https://github.com/owner/repo") == True
        assert self.kapso._is_github_url("https://github.com/owner/repo.git") == True
        assert self.kapso._is_github_url("https://github.com/owner/repo/tree/main") == True
    
    def test_is_github_url_git(self):
        """Test GitHub URL detection for git@ URLs."""
        assert self.kapso._is_github_url("git@github.com:owner/repo.git") == True
    
    def test_is_github_url_local_path(self):
        """Test that local paths are not detected as GitHub URLs."""
        assert self.kapso._is_github_url("/path/to/repo") == False
        assert self.kapso._is_github_url("./relative/path") == False
        assert self.kapso._is_github_url("relative/path") == False
    
    def test_resolve_initial_repo_local_path(self):
        """Test that local paths are returned as-is."""
        local_path = "/tmp/test_repo"
        result = self.kapso._resolve_initial_repo(local_path, "test goal")
        assert result == local_path
    
    def test_resolve_initial_repo_none_no_kg(self):
        """Test that None returns None when KG is not enabled."""
        result = self.kapso._resolve_initial_repo(None, "test goal")
        assert result is None


class TestCloneGithubRepo:
    """Test the _clone_github_repo method with real GitHub cloning."""
    
    def setup_method(self):
        """Create a Kapso instance for testing."""
        self.kapso = Kapso()
        self.cloned_dirs = []
    
    def teardown_method(self):
        """Cleanup cloned directories."""
        for d in self.cloned_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
    
    def test_clone_github_repo_real(self):
        """Test real GitHub repo cloning with a small public repo."""
        # Use a small, stable public repo
        url = "https://github.com/jaymody/picoGPT"
        
        result = self.kapso._clone_github_repo(url)
        
        # Track for cleanup
        if result:
            self.cloned_dirs.append(result)
        
        # Should return a temp directory path
        assert result is not None, "Clone should succeed"
        assert os.path.exists(result), f"Cloned directory should exist: {result}"
        
        # Verify it's a git repo with expected files
        assert os.path.exists(os.path.join(result, ".git")), "Should be a git repo"
        assert os.path.exists(os.path.join(result, "gpt2.py")), "Should contain gpt2.py"
    
    def test_clone_github_repo_invalid_url(self):
        """A clone that fails propagates: a build never starts from an
        empty repository when the person named one."""
        with pytest.raises(git.exc.GitCommandError):
            self.kapso._clone_github_repo("https://github.com/invalid-user-12345/nonexistent-repo-67890")


class TestLocalPathIntegration:
    """Integration tests with actual local paths."""
    
    def setup_method(self):
        """Create test fixtures."""
        self.kapso = Kapso()
        self.test_repo = tempfile.mkdtemp(prefix="test_repo_")
        
        # Create a simple test file
        with open(os.path.join(self.test_repo, "main.py"), "w") as f:
            f.write("print('hello')")
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo, ignore_errors=True)
    
    def test_resolve_local_path_exists(self):
        """Test resolving an existing local path."""
        result = self.kapso._resolve_initial_repo(self.test_repo, "test goal")
        assert result == self.test_repo
    
    def test_resolve_local_path_not_exists(self):
        """Test resolving a non-existent local path (should still return it)."""
        non_existent = "/tmp/non_existent_repo_12345"
        result = self.kapso._resolve_initial_repo(non_existent, "test goal")
        # The method returns the path as-is; validation happens later
        assert result == non_existent


class TestWorkflowSearchExtraction:
    """Test that workflow search correctly extracts GitHub URLs from wiki pages."""
    
    def test_extract_github_url_from_section(self):
        """Test extracting GitHub URL from == Github URL == section."""
        from kapso.knowledge_base.search.workflow_search import extract_github_url
        
        content = """
== Overview ==
Some overview text.

== GitHub URL ==
https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation

== Other Section ==
"""
        
        result = extract_github_url(content)
        assert result == "https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation"
    
    def test_extract_github_url_from_source_syntax(self):
        """Test extracting GitHub URL from [[source::Repo|name|URL]] syntax."""
        from kapso.knowledge_base.search.workflow_search import extract_github_url
        
        content = """
{| class="wikitable"
|-
! Knowledge Sources
||
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|}
"""
        
        result = extract_github_url(content)
        assert result == "https://github.com/jaymody/picoGPT"
    
    def test_extract_github_url_raw(self):
        """Test extracting raw GitHub URL from content."""
        from kapso.knowledge_base.search.workflow_search import extract_github_url
        
        content = """
Check out the repo at https://github.com/owner/repo for more info.
"""
        
        result = extract_github_url(content)
        assert result == "https://github.com/owner/repo"


class TestResolveInitialRepoWithGitHubURL:
    """Test _resolve_initial_repo with GitHub URLs."""
    
    def setup_method(self):
        """Create a Kapso instance for testing."""
        self.kapso = Kapso()
        self.cloned_dirs = []
    
    def teardown_method(self):
        """Cleanup cloned directories."""
        for d in self.cloned_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
    
    def test_resolve_github_url_clones_repo(self):
        """Test that GitHub URLs are cloned."""
        result = self.kapso._resolve_initial_repo(
            "https://github.com/jaymody/picoGPT",
            "test goal"
        )
        
        if result:
            self.cloned_dirs.append(result)
        
        assert result is not None, "Should clone the repo"
        assert os.path.exists(result), f"Cloned repo should exist: {result}"
        assert os.path.exists(os.path.join(result, "gpt2.py")), "Should contain gpt2.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

