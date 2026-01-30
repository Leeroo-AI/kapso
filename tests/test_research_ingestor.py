import pytest

from src.knowledge_base.learners import Source
from src.knowledge_base.learners.ingestors import IngestorFactory


def test_source_research_to_context_string_includes_objective_and_report():
    research = Source.Research(
        objective="Test objective",
        mode="idea",
        report_markdown="## Summary\n- Point A\n",
    )

    ctx = research.to_context_string()
    assert "Test objective" in ctx
    assert "## Summary" in ctx
    assert "Point A" in ctx


@pytest.mark.parametrize(
    "mode,expected_types,expected_count",
    [
        ("idea", {"Principle"}, 1),
        ("implementation", {"Implementation"}, 1),
        ("both", {"Principle", "Implementation"}, 2),
    ],
)
def test_research_ingestor_outputs_expected_page_types(mode, expected_types, expected_count):
    research = Source.Research(
        objective="How to pick LoRA rank?",
        mode=mode,
        report_markdown="A short research report with a citation (https://example.com).",
    )

    ingestor = IngestorFactory.for_source(research)
    pages = ingestor.ingest(research)

    assert len(pages) == expected_count
    assert {p.page_type for p in pages} == expected_types

    # The artifact should be embedded in page content for KG retention.
    for p in pages:
        assert "How to pick LoRA rank?" in p.content
        assert "https://example.com" in p.content


def test_research_ingestor_rejects_invalid_mode():
    research = Source.Research(
        objective="Objective",
        mode="invalid_mode",
        report_markdown="Report",
    )

    ingestor = IngestorFactory.for_source(research)
    with pytest.raises(ValueError):
        ingestor.ingest(research)

