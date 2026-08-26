"""Research-findings parsers under CLI-session narration.

The CLI-only inference conversion changed what the research parsers see:
a completion returned exactly the model's text, but a coding-agent
session returns a transcript where the tagged block is surrounded by
narration ("I searched for...", tool chatter, closing remarks). The
design (cli-only-inference §6.4) calls this the conversion's real risk
and asks for a test per parser: the regexes must dig the tagged content
out of narration, and the truncated-transcript fallback (missing closing
tag) must still recover items.
"""

from kapso.knowledge_base.types import Source
from kapso.researcher.research_findings import (
    parse_idea_results,
    parse_implementation_results,
    parse_study_result,
)


NARRATED_TRANSCRIPT = """I searched the web for recent techniques.

Let me look at a few sources first... done. Here are my findings:

<research_result>
<research_item>
<source>https://example.com/a</source>
<content>Use cosine LR decay for stability.</content>
</research_item>
<research_item>
<source>https://example.com/b</source>
<content>Merge LoRA adapters before evaluation.</content>
</research_item>
</research_result>

That covers the strongest ideas I found. Let me know if you want more.
"""


def test_idea_and_implementation_parsers_survive_session_narration():
    ideas = parse_idea_results(NARRATED_TRANSCRIPT, "q")
    assert [i.source for i in ideas] == [
        "https://example.com/a", "https://example.com/b"
    ]
    assert ideas[0].content == "Use cosine LR decay for stability."
    assert all(isinstance(i, Source.Idea) and i.query == "q" for i in ideas)

    impls = parse_implementation_results(NARRATED_TRANSCRIPT, "q")
    assert [i.content for i in impls] == [
        "Use cosine LR decay for stability.",
        "Merge LoRA adapters before evaluation.",
    ]
    assert all(isinstance(i, Source.Implementation) for i in impls)


def test_study_parser_takes_only_the_tagged_block_not_the_narration():
    report = parse_study_result(NARRATED_TRANSCRIPT, "q")
    assert report.content.startswith("<research_item>")
    assert "I searched the web" not in report.content
    assert "Let me know if you want more" not in report.content


def test_truncated_transcript_missing_closing_tag_still_recovers_items():
    # A session cut off mid-write loses </research_result>; the fallback
    # takes everything after the opening tag so complete items survive.
    truncated = NARRATED_TRANSCRIPT.split("</research_result>")[0]
    ideas = parse_idea_results(truncated, "q")
    assert [i.source for i in ideas] == [
        "https://example.com/a", "https://example.com/b"
    ]
