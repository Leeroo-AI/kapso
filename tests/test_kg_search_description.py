"""search_knowledge shows each hit's === Description === section, and only
that section.

Every page type in wiki_structure puts === Usage === right after
=== Description ===, under the same == Overview ==. The description must
end at that next heading, whatever its level.
"""

from kapso.gated_mcp.gates.base import GateConfig
from kapso.gated_mcp.gates.kg_gate import KGGate
from kapso.knowledge_base.search.base import KGOutput, KGResultItem

PAGE = """== Overview ==
Loads a language model in 4-bit for fine-tuning.

=== Description ===
The FastLanguageModel class handles quantization and LoRA injection.

=== Usage ===
Use it before attaching LoRA adapters.

== Code Reference ==
unsloth/models/loader.py
"""


def search_text(content):
    gate = KGGate(GateConfig(enabled=True, params={}))
    result = KGOutput(
        query="4-bit loading",
        results=[
            KGResultItem(
                id="Implementation/FastLanguageModel",
                score=0.9,
                page_type="Implementation",
                overview="Loads a language model in 4-bit.",
                content=content,
            )
        ],
        total_found=1,
    )
    return gate._format_search_results("4-bit loading", result, True)


def test_description_stops_at_the_usage_subsection():
    text = search_text(PAGE)
    assert (
        "### Description\n"
        "The FastLanguageModel class handles quantization and LoRA "
        "injection.\n" in text
    )
    assert "=== Usage ===" not in text
    assert "Use it before attaching LoRA adapters." not in text


def test_description_runs_to_the_end_of_a_page_with_no_later_heading():
    text = search_text(PAGE.split("\n=== Usage ===")[0])
    assert "LoRA injection.\n" in text
