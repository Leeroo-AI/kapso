# Implementation: WebSailor_Search_Tool

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
Batched Google Serper search tool for WebSailor with parallel query execution.

=== Description ===
The `Search` class in WebSailor provides batched Google search via Serper API:

- Array-based query input for batch searching
- ThreadPoolExecutor with 3 workers
- Formatted results with title, link, date, source, snippet
- 5 retry attempts for resilience
- 10 results per query

Registered as `'search'` tool.

=== Usage ===
Register in WebSailor agents. Requires GOOGLE_SEARCH_KEY environment variable.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebSailor/src/tool_search.py WebAgent/WebSailor/src/tool_search.py]
* '''Lines:''' 1-103

=== Signature ===
<syntaxhighlight lang="python">
@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches..."
    parameters = {...}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        ...

    def google_search(self, query: str) -> str:
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.tool_search import Search
</syntaxhighlight>

== Usage Examples ==

<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.tool_search import Search

search = Search()
results = search.call({"query": ["topic 1", "topic 2"]})
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
