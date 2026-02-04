"""Tests for path/title transformations."""

import pytest
from pathlib import Path

from ..transforms import (
    path_to_title,
    title_to_path,
    is_synced_title,
    is_synced_path,
    FOLDER_TO_NAMESPACE,
    NAMESPACE_TO_FOLDER,
)


class TestPathToTitle:
    """Test path_to_title function."""

    def test_heuristic_path(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "heuristics" / "Test_Page.md"
        assert path_to_title(wiki_dir, file_path) == "Heuristic:Test_Page"

    def test_workflow_path(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "workflows" / "My_Workflow.md"
        assert path_to_title(wiki_dir, file_path) == "Workflow:My_Workflow"

    def test_implementation_path(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "implementations" / "Some_Implementation.md"
        assert path_to_title(wiki_dir, file_path) == "Implementation:Some_Implementation"

    def test_skip_conflicts_folder(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "_conflicts" / "Some_File.md"
        assert path_to_title(wiki_dir, file_path) is None

    def test_skip_staging_folder(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "_staging" / "Some_File.md"
        assert path_to_title(wiki_dir, file_path) is None

    def test_non_md_file(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "heuristics" / "Test_Page.txt"
        assert path_to_title(wiki_dir, file_path) is None

    def test_unsupported_folder(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "unsupported" / "Test_Page.md"
        assert path_to_title(wiki_dir, file_path) is None


class TestTitleToPath:
    """Test title_to_path function."""

    def test_heuristic_title(self):
        wiki_dir = Path("/wikis")
        title = "Heuristic:Test_Page"
        expected = wiki_dir / "heuristics" / "Test_Page.md"
        assert title_to_path(wiki_dir, title) == expected

    def test_workflow_title(self):
        wiki_dir = Path("/wikis")
        title = "Workflow:My_Workflow"
        expected = wiki_dir / "workflows" / "My_Workflow.md"
        assert title_to_path(wiki_dir, title) == expected

    def test_unsupported_namespace(self):
        wiki_dir = Path("/wikis")
        title = "Category:Some_Category"
        assert title_to_path(wiki_dir, title) is None

    def test_main_namespace(self):
        wiki_dir = Path("/wikis")
        title = "Main_Page"
        assert title_to_path(wiki_dir, title) is None


class TestIsSyncedTitle:
    """Test is_synced_title function."""

    def test_heuristic_synced(self):
        assert is_synced_title("Heuristic:Test") is True

    def test_workflow_synced(self):
        assert is_synced_title("Workflow:Test") is True

    def test_category_not_synced(self):
        assert is_synced_title("Category:Test") is False

    def test_main_namespace_not_synced(self):
        assert is_synced_title("Main_Page") is False


class TestIsSyncedPath:
    """Test is_synced_path function."""

    def test_heuristics_synced(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "heuristics" / "Test.md"
        assert is_synced_path(wiki_dir, file_path) is True

    def test_conflicts_not_synced(self):
        wiki_dir = Path("/wikis")
        file_path = wiki_dir / "_conflicts" / "Test.md"
        assert is_synced_path(wiki_dir, file_path) is False

    def test_outside_wiki_dir(self):
        wiki_dir = Path("/wikis")
        file_path = Path("/other/heuristics/Test.md")
        assert is_synced_path(wiki_dir, file_path) is False


class TestMappings:
    """Test that folder/namespace mappings are consistent."""

    def test_reverse_mapping(self):
        for folder, namespace in FOLDER_TO_NAMESPACE.items():
            assert NAMESPACE_TO_FOLDER[namespace] == folder

    def test_all_namespaces_mapped(self):
        expected = {"Heuristic", "Workflow", "Principle", "Implementation", "Environment"}
        assert set(NAMESPACE_TO_FOLDER.keys()) == expected
