# Implementation: WebDancer_Search_Tool

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
Batched web search tool using Google Serper API that supports multiple concurrent queries with formatted result output.

=== Description ===
The `Search` class in WebDancer provides batched web search functionality via the Google Serper API. Key features:

- Accepts array of query strings for batch search
- Parallel execution using ThreadPoolExecutor
- Configurable max queries per call (MAX_MULTIQUERY_NUM)
- Formatted results with title, URL, date, source, and snippet
- Retry logic for API resilience (5 attempts)

The tool registers as `'search'` in the Qwen-Agent tool system.

=== Usage ===
Register this tool in agents that need web search capability. Requires GOOGLE_SEARCH_KEY environment variable.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/tools/private/search.py WebAgent/WebDancer/demos/tools/private/search.py]
* '''Lines:''' 1-99

=== Signature ===
<syntaxhighlight lang="python">
@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    """Batched web search tool using Serper API."""

    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings."
            }
        },
        "required": ["query"]
    }

    def call(self, params: str, **kwargs) -> str:
        """
        Execute search queries.

        Args:
            params: JSON with 'query' array

        Returns:
            Formatted search results
        """
        ...

    def google_search(self, query: str) -> str:
        """
        Execute single Google search via Serper API.

        Args:
            query: Search query string

        Returns:
            Formatted results for query
        """
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.tools.private.search import Search
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| query || List[str] || Yes || Array of search queries
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || str || Formatted search results with URLs
|}

== Usage Examples ==

=== Basic Search ===
<syntaxhighlight lang="python">
import os
from WebAgent.WebDancer.demos.tools.private.search import Search

os.environ['GOOGLE_SEARCH_KEY'] = 'your-serper-key'

search = Search()
results = search.call({"query": ["ACL 2025 deadline"]})
print(results)
</syntaxhighlight>

=== Batched Search ===
<syntaxhighlight lang="python">
# Multiple queries in one call
results = search.call({
    "query": [
        "ACL 2025 conference",
        "EMNLP 2025 deadline",
        "NeurIPS 2025 venue"
    ]
})
# Results separated by "======="
for section in results.split("======="):
    print(section[:200])
</syntaxhighlight>

=== In Agent ===
<syntaxhighlight lang="python">
from qwen_agent.agents import Assistant

agent = Assistant(
    llm=llm_config,
    function_list=['search']  # Auto-registered
)

for response in agent.run([{'role': 'user', 'content': 'Search for AI news'}]):
    print(response[-1].content)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
