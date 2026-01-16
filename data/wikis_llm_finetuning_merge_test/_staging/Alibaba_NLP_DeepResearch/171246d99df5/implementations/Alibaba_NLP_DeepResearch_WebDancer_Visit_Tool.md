# Implementation: WebDancer_Visit_Tool

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|Jina Reader|https://r.jina.ai/]]
|-
! Domains
| [[domain::Web_Scraping]], [[domain::Content_Extraction]], [[domain::Tool_Use]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Goal-directed webpage visitor tool using Jina Reader for content extraction and LLM-based summarization for WebDancer agents.

=== Description ===
The `Visit` class in WebDancer provides intelligent webpage visitation with:

- Jina Reader API for webpage-to-markdown conversion
- LLM-based content extraction focused on user goal
- Support for single URL or array of URLs
- Parallel URL processing with ThreadPoolExecutor
- Structured output with evidence and summary

The tool uses a secondary LLM call to extract goal-relevant information from raw webpage content.

=== Usage ===
Register this tool in agents needing webpage content extraction. Requires JINA_API_KEY and DASHSCOPE_API_KEY environment variables.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/tools/private/visit.py WebAgent/WebDancer/demos/tools/private/visit.py]
* '''Lines:''' 1-173

=== Signature ===
<syntaxhighlight lang="python">
@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    """Goal-directed webpage visitor with LLM extraction."""

    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "URL(s) to visit"
            },
            "goal": {
                "type": "string",
                "description": "Goal of the visit"
            }
        },
        "required": ["url", "goal"]
    }

    def call(self, params: str, **kwargs) -> str:
        """Execute webpage visit with goal-directed extraction."""
        ...

    def readpage(self, url: str, goal: str) -> str:
        """Read page via Jina and extract goal-relevant info."""
        ...

    def llm(self, messages: List[Dict]) -> str:
        """Call LLM for content extraction."""
        ...

def jina_readpage(url: str) -> str:
    """Fetch webpage content via Jina Reader API."""
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.tools.private.visit import Visit, jina_readpage
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| url || str/List[str] || Yes || URL(s) to visit
|-
| goal || str || Yes || Information extraction goal
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| useful_information || str || Extracted evidence and summary
|}

== Usage Examples ==

=== Basic Visit ===
<syntaxhighlight lang="python">
import os
from WebAgent.WebDancer.demos.tools.private.visit import Visit

os.environ['JINA_API_KEY'] = 'jina-key'
os.environ['DASHSCOPE_API_KEY'] = 'dashscope-key'

visit = Visit()
result = visit.call({
    "url": "https://2025.aclweb.org",
    "goal": "Find the paper submission deadline"
})
print(result)
# Output:
# Evidence in page: Paper submission deadline: February 15, 2025
# Summary: The ACL 2025 paper deadline is February 15, 2025.
</syntaxhighlight>

=== Multiple URLs ===
<syntaxhighlight lang="python">
result = visit.call({
    "url": [
        "https://2025.aclweb.org",
        "https://2025.emnlp.org"
    ],
    "goal": "Find submission deadlines"
})
# Results for each URL separated by "======="
</syntaxhighlight>

=== Raw Jina Access ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.tools.private.visit import jina_readpage

# Get raw markdown content
content = jina_readpage("https://example.com")
print(content[:500])
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
