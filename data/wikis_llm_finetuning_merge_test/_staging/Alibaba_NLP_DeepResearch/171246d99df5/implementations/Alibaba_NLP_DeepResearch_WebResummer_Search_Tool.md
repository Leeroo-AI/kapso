# Implementation: WebResummer_Search_Tool

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|Serper API|https://serper.dev/]]
|-
! Domains
| [[domain::Web_Search]], [[domain::Tool_Use]], [[domain::Google_Search]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Batched Google Serper search tool for WebResummer with parallel query execution and formatted result output.

=== Description ===
The `Search` class in WebResummer provides batched web search functionality via Google Serper API:

- Array-based query input for batch searching
- Parallel execution with ThreadPoolExecutor (3 workers)
- Formatted results with title, link, date, source, and snippet
- Retry logic with 5 attempts for resilience
- Country-specific results (default: English)

Registered as `'search'` tool in Qwen-Agent system.

=== Usage ===
Register in ReAct agents needing web search. Requires GOOGLE_SEARCH_KEY environment variable.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebResummer/src/tool_search.py WebAgent/WebResummer/src/tool_search.py]
* '''Lines:''' 1-112

=== Signature ===
<syntaxhighlight lang="python">
@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    """Batched web search via Serper API."""

    name = "search"
    description = "Performs batched web searches..."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings"
            }
        },
        "required": ["query"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute batch search queries."""
        ...

    def google_search(self, query: str) -> str:
        """Execute single Serper API search."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.tool_search import Search
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| query || List[str] || Yes || Search query strings
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || str || Formatted search results
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
import os
from WebAgent.WebResummer.src.tool_search import Search

os.environ['GOOGLE_SEARCH_KEY'] = 'your-key'

search = Search()
results = search.call({"query": ["NeurIPS 2025 deadline"]})
print(results)
</syntaxhighlight>

=== In Agent Context ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.react_agent import MultiTurnReactAgent

agent = MultiTurnReactAgent(
    llm=llm_cfg,
    function_list=['search', 'visit'],  # Auto-registers Search
    system_message=SYSTEM_PROMPT
)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
