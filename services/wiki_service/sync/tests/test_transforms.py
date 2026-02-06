"""Tests for path/title transformations."""

import pytest
from pathlib import Path

from ..transforms import (
    path_to_title,
    title_to_path,
    is_synced_title,
    is_synced_path,
    strip_h1_heading,
    transform_source_links,
    add_page_metadata,
    prepare_content_for_wiki,
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


# =============================================================================
# Content Transform Tests
# =============================================================================


class TestStripH1Heading:
    """Test strip_h1_heading function."""

    def test_strip_markdown_h1(self):
        """Markdown H1 (# Title) should be stripped."""
        content = "# Workflow: QLoRA_Finetuning\n\n{| class=\"wikitable\""
        result = strip_h1_heading(content)
        assert result == "{| class=\"wikitable\""

    def test_strip_mediawiki_h1(self):
        """MediaWiki H1 (= Title =) should be stripped."""
        content = "= My Page Title =\n\n== Overview =="
        result = strip_h1_heading(content)
        assert result == "== Overview =="

    def test_strip_heading_with_blank_lines(self):
        """Blank lines after heading should also be stripped."""
        content = "# Workflow: Test\n\n\n{| class=\"wikitable\""
        result = strip_h1_heading(content)
        assert result == "{| class=\"wikitable\""

    def test_no_heading_unchanged(self):
        """Content without H1 heading should be returned unchanged."""
        content = "{| class=\"wikitable\"\n|}\n\n== Overview =="
        result = strip_h1_heading(content)
        assert result == content

    def test_h2_not_stripped(self):
        """H2 headings (## or ==) should NOT be stripped."""
        content = "## Section Title\n\nSome text"
        result = strip_h1_heading(content)
        assert result == content

    def test_empty_content(self):
        """Empty content should be returned unchanged."""
        assert strip_h1_heading("") == ""

    def test_heading_only(self):
        """Content that is only a heading should return empty."""
        result = strip_h1_heading("# Just a Heading")
        assert result == ""


class TestTransformSourceLinks:
    """Test transform_source_links function."""

    def test_repo_source(self):
        """Repo source link should become external link."""
        content = "* [[source::Repo|OpenClaw|https://github.com/openclaw/openclaw]]"
        result = transform_source_links(content)
        assert result == "* [https://github.com/openclaw/openclaw OpenClaw]"

    def test_doc_source(self):
        """Doc source link should become external link."""
        content = "* [[source::Doc|Agent Loop|https://docs.openclaw.ai/concepts/agent-loop]]"
        result = transform_source_links(content)
        assert result == "* [https://docs.openclaw.ai/concepts/agent-loop Agent Loop]"

    def test_blog_source(self):
        """Blog source link should become external link."""
        content = "* [[source::Blog|Fine-tuning Guide|https://unsloth.ai/docs/get-started]]"
        result = transform_source_links(content)
        assert result == "* [https://unsloth.ai/docs/get-started Fine-tuning Guide]"

    def test_multiple_sources(self):
        """Multiple source links on separate lines should all be transformed."""
        content = (
            "* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]\n"
            "* [[source::Doc|Docs|https://unsloth.ai/docs]]\n"
            "* [[source::Blog|Guide|https://unsloth.ai/blog/grpo]]"
        )
        result = transform_source_links(content)
        assert "* [https://github.com/unslothai/unsloth Unsloth]" in result
        assert "* [https://unsloth.ai/docs Docs]" in result
        assert "* [https://unsloth.ai/blog/grpo Guide]" in result

    def test_no_source_links_unchanged(self):
        """Content without source links should be returned unchanged."""
        content = "== Overview ==\n\nSome regular text."
        result = transform_source_links(content)
        assert result == content

    def test_other_semantic_links_unchanged(self):
        """Other SMW annotations should not be touched."""
        content = "[[domain::LLMs]], [[last_updated::2026-01-01]]"
        result = transform_source_links(content)
        assert result == content

    def test_source_in_table(self):
        """Source link inside a wikitable cell should be transformed."""
        content = "|| [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]"
        result = transform_source_links(content)
        assert result == "|| [https://github.com/unslothai/unsloth Unsloth]"


class TestAddPageMetadata:
    """Test add_page_metadata function."""

    def test_adds_pageinfo_and_category(self):
        """Should prepend PageInfo and append Category."""
        content = "== Overview =="
        result = add_page_metadata(content, "Workflow", "My_Page")
        assert result.startswith("{{PageInfo|type=Workflow|title=My_Page}}")
        assert result.endswith("[[Category:Workflows]]")
        assert "== Overview ==" in result

    def test_no_duplicate_pageinfo(self):
        """Should not add PageInfo if already present."""
        content = "{{PageInfo|type=Workflow|title=X}}\n== Overview =="
        result = add_page_metadata(content, "Workflow", "X")
        assert result.count("{{PageInfo|") == 1

    def test_no_duplicate_category(self):
        """Should not add Category if already present."""
        content = "== Overview ==\n\n[[Category:Workflows]]"
        result = add_page_metadata(content, "Workflow", "X")
        assert result.count("[[Category:Workflows]]") == 1

    def test_pluralizes_namespace(self):
        """Category name should be pluralized."""
        content = "Text"
        result = add_page_metadata(content, "Heuristic", "Tip")
        assert "[[Category:Heuristics]]" in result

    def test_implementation_namespace(self):
        """Implementation namespace should work correctly."""
        content = "Text"
        result = add_page_metadata(content, "Implementation", "Foo")
        assert "{{PageInfo|type=Implementation|title=Foo}}" in result
        assert "[[Category:Implementations]]" in result


class TestPrepareContentForWiki:
    """Test the full prepare_content_for_wiki pipeline."""

    def test_full_pipeline(self):
        """All transforms should be applied in order."""
        content = (
            "# Workflow: Agent_Message_Loop\n\n"
            "{| class=\"wikitable\"\n"
            "|-\n"
            "! Knowledge Sources\n"
            "||\n"
            "* [[source::Repo|OpenClaw|https://github.com/openclaw/openclaw]]\n"
            "|}\n\n"
            "== Overview ==\n"
            "Some text."
        )
        result = prepare_content_for_wiki(content, "Workflow", "Agent_Message_Loop")

        # H1 heading should be stripped
        assert "# Workflow:" not in result

        # Source link should be transformed
        assert "[https://github.com/openclaw/openclaw OpenClaw]" in result
        assert "[[source::" not in result

        # PageInfo and Category should be added
        assert "{{PageInfo|type=Workflow|title=Agent_Message_Loop}}" in result
        assert "[[Category:Workflows]]" in result

    def test_without_namespace(self):
        """Without namespace, only H1 and source transforms should apply."""
        content = "# Title\n\n* [[source::Repo|X|https://example.com]]\nBody"
        result = prepare_content_for_wiki(content)

        assert "# Title" not in result
        assert "[https://example.com X]" in result
        # No PageInfo or Category added
        assert "{{PageInfo" not in result
        assert "[[Category:" not in result

    def test_real_page_content(self):
        """Test with actual page content from the repo."""
        content = (
            "# Workflow: QLoRA_Finetuning\n\n"
            "{| class=\"wikitable\" style=\"float:right; margin-left:1em; width:300px;\"\n"
            "|-\n"
            "! Knowledge Sources\n"
            "||\n"
            "* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]\n"
            "* [[source::Doc|Unsloth Docs|https://unsloth.ai/docs]]\n"
            "* [[source::Blog|Fine-tuning Guide|https://unsloth.ai/docs/get-started]]\n"
            "|-\n"
            "! Domains\n"
            "|| [[domain::LLMs]], [[domain::Fine_Tuning]]\n"
            "|-\n"
            "! Last Updated\n"
            "|| [[last_updated::2026-02-04 18:00 GMT]]\n"
            "|}\n\n"
            "== Overview ==\n"
            "End-to-end process."
        )
        result = prepare_content_for_wiki(
            content, "Workflow", "Unslothai_Unsloth_QLoRA_Finetuning"
        )

        # H1 stripped
        assert "# Workflow:" not in result

        # All three source links transformed
        assert "[https://github.com/unslothai/unsloth Unsloth]" in result
        assert "[https://unsloth.ai/docs Unsloth Docs]" in result
        assert "[https://unsloth.ai/docs/get-started Fine-tuning Guide]" in result

        # Other SMW annotations untouched
        assert "[[domain::LLMs]]" in result
        assert "[[last_updated::2026-02-04 18:00 GMT]]" in result

        # Metadata added
        assert "{{PageInfo|type=Workflow|title=Unslothai_Unsloth_QLoRA_Finetuning}}" in result
        assert "[[Category:Workflows]]" in result
