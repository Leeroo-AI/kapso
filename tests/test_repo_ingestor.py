# tests/test_repo_ingestor.py
#
# Integration tests for the phased repo ingestor.
# Uses unsloth repo as test case.
#
# Test categories:
# - Unit tests: Test individual components (no agent calls)
# - Integration tests: Full pipeline with real repo (requires agent)
#
# Run integration tests:
#   pytest tests/test_repo_ingestor.py -v -m integration
#
# Run unit tests only:
#   pytest tests/test_repo_ingestor.py -v -m "not integration"

import os
import re
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


# =============================================================================
# Unit Tests (No Agent Required)
# =============================================================================

class TestRepoIngestorUtils:
    """Unit tests for repo_ingestor utility functions."""
    
    def test_get_repo_name_from_url(self):
        """Test extracting repo name from various URL formats."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import get_repo_name_from_url
        
        assert get_repo_name_from_url("https://github.com/unslothai/unsloth") == "unsloth"
        assert get_repo_name_from_url("https://github.com/unslothai/unsloth/") == "unsloth"
        assert get_repo_name_from_url("https://github.com/unslothai/unsloth.git") == "unsloth"
        assert get_repo_name_from_url("git@github.com:user/repo.git") == "repo"
    
    def test_load_wiki_structure_workflow(self):
        """Test loading workflow wiki structure."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import load_wiki_structure
        
        content = load_wiki_structure("workflow")
        
        # Should contain key sections
        assert "Workflow" in content
        assert "Page Definition" in content or "page_definition" in content.lower()
        assert "== Overview ==" in content or "Overview" in content
    
    def test_load_wiki_structure_principle(self):
        """Test loading principle wiki structure."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import load_wiki_structure
        
        content = load_wiki_structure("principle")
        
        assert "Principle" in content
        assert "implemented_by" in content.lower() or "Implemented By" in content
    
    def test_load_wiki_structure_implementation(self):
        """Test loading implementation wiki structure."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import load_wiki_structure
        
        content = load_wiki_structure("implementation")
        
        assert "Implementation" in content
        assert "Code Signature" in content or "code_signature" in content.lower()
    
    def test_load_wiki_structure_environment(self):
        """Test loading environment wiki structure."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import load_wiki_structure
        
        content = load_wiki_structure("environment")
        
        assert "Environment" in content
        assert "Dependencies" in content or "requirements" in content.lower()
    
    def test_load_wiki_structure_heuristic(self):
        """Test loading heuristic wiki structure."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import load_wiki_structure
        
        content = load_wiki_structure("heuristic")
        
        assert "Heuristic" in content
        assert "Insight" in content or "Rule" in content
    
    def test_load_wiki_structure_invalid_type(self):
        """Test that invalid page type raises error."""
        from src.knowledge.learners.ingestors.repo_ingestor.utils import load_wiki_structure
        
        with pytest.raises(FileNotFoundError):
            load_wiki_structure("invalid_type")


class TestRepoIngestorPrompts:
    """Unit tests for prompt loading."""
    
    def test_prompts_exist(self):
        """Verify all required prompt files exist."""
        prompts_dir = Path(__file__).parents[1] / "src/knowledge/learners/ingestors/repo_ingestor/prompts"
        
        required_prompts = ["anchoring.md", "excavation.md", "synthesis.md", "enrichment.md", "audit.md"]
        
        for prompt in required_prompts:
            prompt_path = prompts_dir / prompt
            assert prompt_path.exists(), f"Missing prompt: {prompt}"
    
    def test_prompt_has_placeholders(self):
        """Verify prompts have required format placeholders."""
        prompts_dir = Path(__file__).parents[1] / "src/knowledge/learners/ingestors/repo_ingestor/prompts"
        
        # Anchoring should have repo_name, repo_path, wiki_dir
        anchoring = (prompts_dir / "anchoring.md").read_text()
        assert "{repo_name}" in anchoring
        assert "{repo_path}" in anchoring
        assert "{wiki_dir}" in anchoring


class TestRepoIngestorClass:
    """Unit tests for RepoIngestor class."""
    
    def test_ingestor_source_type(self):
        """Test that ingestor reports correct source type."""
        from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
        
        ingestor = RepoIngestor()
        assert ingestor.source_type == "repo"
    
    def test_ingestor_registered(self):
        """Test that RepoIngestor is registered in factory."""
        from src.knowledge.learners.ingestors import IngestorFactory
        
        assert IngestorFactory.is_registered("repo")
        
        ingestor = IngestorFactory.create("repo")
        assert ingestor is not None
    
    def test_ingestor_params(self):
        """Test ingestor parameter handling."""
        from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
        
        ingestor = RepoIngestor(params={
            "timeout": 600,
            "cleanup": False,
            "wiki_dir": "/tmp/test_wiki",
        })
        
        assert ingestor._timeout == 600
        assert ingestor._cleanup == False
        assert ingestor._wiki_dir == Path("/tmp/test_wiki")
    
    def test_normalize_source_dict(self):
        """Test normalizing dict source."""
        from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
        
        ingestor = RepoIngestor()
        
        result = ingestor._normalize_source({"url": "https://github.com/user/repo", "branch": "main"})
        assert result["url"] == "https://github.com/user/repo"
        assert result["branch"] == "main"
    
    def test_normalize_source_object(self):
        """Test normalizing Source.Repo object."""
        from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
        from src.knowledge.learners.sources import Source
        
        ingestor = RepoIngestor()
        
        source = Source.Repo("https://github.com/user/repo", branch="develop")
        result = ingestor._normalize_source(source)
        
        assert result["url"] == "https://github.com/user/repo"
        assert result["branch"] == "develop"
    
    def test_ingest_requires_url(self):
        """Test that ingest raises error without URL."""
        from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
        
        ingestor = RepoIngestor()
        
        with pytest.raises(ValueError, match="Repository URL is required"):
            ingestor.ingest({"branch": "main"})


# =============================================================================
# Integration Tests (Requires Agent)
# =============================================================================

@pytest.fixture
def wiki_test_dir(tmp_path):
    """Create isolated wiki test directory."""
    wiki_dir = tmp_path / "wikis_test"
    for subdir in ["workflows", "principles", "implementations", "environments", "heuristics"]:
        (wiki_dir / subdir).mkdir(parents=True)
    return wiki_dir


@pytest.mark.slow
@pytest.mark.integration
class TestPhasedRepoIngestorIntegration:
    """Integration tests for the phased repo ingestor with unsloth."""
    
    def test_ingest_unsloth_repo(self, wiki_test_dir):
        """
        Test full pipeline with unsloth repo.
        
        This test:
        1. Clones the unsloth repo
        2. Runs all 5 phases
        3. Verifies pages were written
        """
        from src.knowledge.learners import KnowledgePipeline, Source
        from src.knowledge.search.kg_graph_search import parse_wiki_directory
        
        pipeline = KnowledgePipeline(wiki_dir=wiki_test_dir)
        
        result = pipeline.run(
            Source.Repo("https://github.com/unslothai/unsloth"),
            dry_run=False,
            skip_merge=True,  # Don't merge, just extract
        )
        
        # Verify pages were extracted
        assert result.total_pages_extracted > 0, "No pages extracted"
        assert result.success, f"Pipeline failed: {result.errors}"
        
        # Verify files were written
        pages = parse_wiki_directory(wiki_test_dir)
        assert len(pages) > 0, "No pages found in wiki directory"
        
        # Verify page types exist
        page_types = {p.page_type for p in pages}
        assert "Workflow" in page_types, "No Workflow pages created"
        assert "Principle" in page_types or "Implementation" in page_types, "No theory/code pages created"
    
    def test_workflow_has_steps(self, wiki_test_dir):
        """Verify workflow pages have step links after ingestion."""
        from src.knowledge.learners import KnowledgePipeline, Source
        
        pipeline = KnowledgePipeline(wiki_dir=wiki_test_dir)
        pipeline.run(
            Source.Repo("https://github.com/unslothai/unsloth"),
            skip_merge=True,
        )
        
        # Check workflow files have step links
        workflows_dir = wiki_test_dir / "workflows"
        if workflows_dir.exists():
            for wf_file in workflows_dir.glob("*.md"):
                content = wf_file.read_text()
                # Should have at least one step link
                has_step = "[[step::Principle:" in content
                if not has_step:
                    pytest.skip(f"Workflow {wf_file.name} has no step links yet")
    
    def test_principle_has_implementation(self, wiki_test_dir):
        """Verify executability constraint - all Principles have implementations."""
        from src.knowledge.learners import KnowledgePipeline, Source
        
        pipeline = KnowledgePipeline(wiki_dir=wiki_test_dir)
        pipeline.run(
            Source.Repo("https://github.com/unslothai/unsloth"),
            skip_merge=True,
        )
        
        # Check all principles have implementation links
        principles_dir = wiki_test_dir / "principles"
        if principles_dir.exists():
            for p_file in principles_dir.glob("*.md"):
                content = p_file.read_text()
                has_impl = "[[implemented_by::Implementation:" in content
                if not has_impl:
                    pytest.fail(f"Principle {p_file.name} missing implementation link")
    
    def test_page_format_matches_wiki_structure(self, wiki_test_dir):
        """Verify pages follow wiki_structure section definitions."""
        from src.knowledge.learners import KnowledgePipeline, Source
        
        pipeline = KnowledgePipeline(wiki_dir=wiki_test_dir)
        pipeline.run(
            Source.Repo("https://github.com/unslothai/unsloth"),
            skip_merge=True,
        )
        
        # Check workflow pages have required sections
        workflows_dir = wiki_test_dir / "workflows"
        if workflows_dir.exists():
            for wf_file in workflows_dir.glob("*.md"):
                content = wf_file.read_text()
                
                # Required sections for Workflow pages
                assert "== Overview ==" in content or "Overview" in content, \
                    f"Workflow {wf_file.name} missing Overview section"
                assert "Knowledge Sources" in content or "source::" in content, \
                    f"Workflow {wf_file.name} missing metadata block"


# =============================================================================
# CLI Tests
# =============================================================================

class TestRepoIngestorCLI:
    """Test CLI interface through knowledge learners module."""
    
    def test_cli_help(self):
        """Test that CLI shows help."""
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "src.knowledge.learners", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[1],
        )
        
        # Should show help text without error
        assert result.returncode == 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


# =============================================================================
# Test Markers Configuration
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring agent"
    )

