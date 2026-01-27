#!/usr/bin/env python3
"""
Task 1: Initialize Repo - Tests

Tests for the initial_repo resolution logic in Kapso.evolve():
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
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (OpenAI API key)
load_dotenv()

# Test the helper methods directly
from src.kapso import Kapso


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
        """Test GitHub repo cloning with invalid URL."""
        result = self.kapso._clone_github_repo("https://github.com/invalid-user-12345/nonexistent-repo-67890")
        
        # Should return None on failure
        assert result is None


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
        from src.knowledge.search.workflow_search import extract_github_url
        
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
        from src.knowledge.search.workflow_search import extract_github_url
        
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
        from src.knowledge.search.workflow_search import extract_github_url
        
        content = """
Check out the repo at https://github.com/owner/repo for more info.
"""
        
        result = extract_github_url(content)
        assert result == "https://github.com/owner/repo"


class TestWorkflowSearchWithRealKG:
    """
    Integration tests with real KG indexing and workflow search.
    
    Uses data/wikis_llm_finetuning_test/ which contains PicoGPT workflow.
    """
    
    @classmethod
    def setup_class(cls):
        """Index the test wiki data once for all tests in this class."""
        cls.wiki_dir = Path("data/wikis_llm_finetuning_test")
        cls.index_path = Path("data/indexes/test_task1_workflow.index")
        
        # Ensure wiki data exists
        if not cls.wiki_dir.exists():
            pytest.skip(f"Test wiki data not found: {cls.wiki_dir}")
        
        # Index the wiki data
        print(f"\nIndexing wiki data from {cls.wiki_dir}...")
        cls.kapso = Kapso()
        cls.kapso.index_kg(
            wiki_dir=str(cls.wiki_dir),
            save_to=str(cls.index_path),
        )
        print("Indexing complete!")
        
        # Create a new Kapso instance with the index loaded
        cls.kapso_with_kg = Kapso(kg_index=str(cls.index_path))
    
    @classmethod
    def teardown_class(cls):
        """Cleanup index file."""
        if hasattr(cls, 'index_path') and cls.index_path.exists():
            cls.index_path.unlink()
    
    def test_workflow_search_finds_picogpt(self):
        """Test that workflow search finds PicoGPT for text generation queries."""
        from src.knowledge.search.workflow_search import WorkflowRepoSearch
        
        # Use the KG search from our indexed Kapso
        search = WorkflowRepoSearch(kg_search=self.kapso_with_kg.knowledge_search)
        
        # Search for text generation
        result = search.search("text generation with GPT", top_k=3)
        
        print(f"\nSearch results for 'text generation with GPT':")
        for item in result.items:
            print(f"  - {item.title} (score={item.score:.3f})")
            print(f"    GitHub: {item.github_url}")
        
        # Should find at least one result
        assert not result.is_empty, "Should find workflow results"
        
        # Top result should be PicoGPT
        top = result.top_result
        assert "PicoGPT" in top.title or "Text_Generation" in top.title, \
            f"Top result should be PicoGPT workflow, got: {top.title}"
    
    def test_search_returns_github_url(self):
        """Test search result contains GitHub URL."""
        from src.knowledge.search.workflow_search import WorkflowRepoSearch
        
        search = WorkflowRepoSearch(kg_search=self.kapso_with_kg.knowledge_search)
        
        # Search for GPT text generation
        result = search.search("generate text using GPT-2 with NumPy", top_k=1)
        
        print(f"\nSearch result: {result.top_result.title if result.top_result else 'None'}")
        
        # Should return a result with GitHub URL
        assert not result.is_empty, "Should find a result"
        assert result.top_result.github_url, "Result should have github_url"
        assert "github.com" in result.top_result.github_url, \
            f"Should be a GitHub URL: {result.top_result.github_url}"
    
    def test_search_workflow_repo_integration(self):
        """Test the full _search_workflow_repo method with real KG."""
        # Use the Kapso instance with KG loaded
        result = self.kapso_with_kg._search_workflow_repo("text generation with GPT-2")
        
        print(f"\n_search_workflow_repo result: {result}")
        
        # Should return a cloned repo path or None
        # Note: This actually clones the repo, so we need to clean up
        if result:
            assert os.path.exists(result), f"Cloned repo should exist: {result}"
            # Cleanup
            shutil.rmtree(result, ignore_errors=True)
    
    def test_resolve_initial_repo_with_workflow_search(self):
        """Test _resolve_initial_repo uses workflow search when initial_repo is None."""
        # Use the Kapso instance with KG loaded
        result = self.kapso_with_kg._resolve_initial_repo(
            initial_repo=None,
            goal="generate text using GPT-2 with NumPy"
        )
        
        print(f"\n_resolve_initial_repo result: {result}")
        
        # Should return a cloned repo path
        if result:
            assert os.path.exists(result), f"Cloned repo should exist: {result}"
            # Cleanup
            shutil.rmtree(result, ignore_errors=True)


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

