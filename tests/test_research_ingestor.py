"""Research-source dispatch and formatting.

These previously asserted that `Source.Research(mode=...)` produced a fixed
number of pages of a fixed type. That design is gone: research() now returns
distinct Source types, and the ingestors are agentic — a Claude Code pipeline
decides which page types a source deserves. "One idea yields exactly one
Principle page" is no longer a contract anything can honour, and asserting it
against a mocked agent would prove nothing (Rule 9).

What is still a contract, and is checked here: the factory dispatches on the
source's class name, an unregistered type fails loud, and each Source formats
itself for a prompt without calling anything.
"""

import pytest

from kapso.knowledge_base.learners import Source
from kapso.knowledge_base.learners.ingestors import IngestorFactory


def test_factory_dispatches_each_research_source_to_its_ingestor():
    """for_source() keys off the class name, so a renamed variant breaks here."""
    cases = [
        (Source.Idea(query="How to pick LoRA rank?", source="https://example.com",
                     content="Use rank 16."), "idea"),
        (Source.Implementation(query="QLoRA setup", source="https://example.com",
                               content="peft config"),
         "implementation"),
        (Source.ResearchReport(query="PEFT survey", content="## Summary"),
         "researchreport"),
    ]
    for source, expected_type in cases:
        ingestor = IngestorFactory.for_source(source)
        assert ingestor.source_type == expected_type


def test_unregistered_source_type_raises_and_names_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        IngestorFactory.create("not_a_real_source")
    message = str(excinfo.value)
    assert "not_a_real_source" in message
    # the error has to be actionable, so it lists what is registered
    assert "idea" in message


def test_research_sources_render_their_query_and_content_for_a_prompt():
    """to_string() is what reaches a model, so both halves must survive it."""
    idea = Source.Idea(query="How to pick LoRA rank?", source="https://example.com",
                       content="Start at 16.")
    rendered = idea.to_string()
    assert "How to pick LoRA rank?" in rendered
    assert "Start at 16." in rendered

    report = Source.ResearchReport(query="PEFT survey", content="## Summary\n- Point A")
    rendered = report.to_string()
    assert "PEFT survey" in rendered
    assert "Point A" in rendered


def test_registered_ingestor_types_cover_the_research_sources():
    available = IngestorFactory.list_ingestors()
    for source_type in ("idea", "implementation", "researchreport"):
        assert source_type in available
