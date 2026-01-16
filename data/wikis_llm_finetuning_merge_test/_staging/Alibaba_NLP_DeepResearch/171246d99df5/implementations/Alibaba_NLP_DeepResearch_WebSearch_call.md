# Implementation: WebSearch_call

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|nlp_web_search.py|WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/nlp_web_search.py]]
* [[source::API|Serper API|https://serper.dev]]
|-
! Domains
| [[domain::Information_Retrieval]], [[domain::Web_Search]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Web search tool implementation that queries Serper API and returns formatted search results for the multimodal agent.

=== Description ===

The `call()` method in the `WebSearch` class (registered as 'web_search' tool) provides web search capabilities to the multimodal agent. It accepts search queries, invokes the Serper API (Google search backend), and returns formatted results containing titles, URLs, dates, sources, and snippets.

The implementation features:
- Batch query support via ThreadPoolExecutor
- Retry logic (5 attempts) for API reliability
- Markdown-formatted output with numbered results
- Configurable via environment variables (TEXT_SEARCH_KEY, MAX_CHAR)

=== Usage ===

Use the `web_search` tool when:
- The agent needs to find relevant web pages for a topic
- Gathering information about entities identified in images
- Finding current news or recent information
- Locating authoritative sources for verification

The tool returns URLs that can be subsequently visited using the 'visit' tool.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/nlp_web_search.py
* '''Lines:''' 116-130

=== Signature ===
<syntaxhighlight lang="python">
def call(self, params: Union[str, dict], **kwargs) -> str:
    """
    Execute web search queries and return formatted results.

    Args:
        params: Union[str, dict] - Search parameters:
            - If str: JSON string with "queries" key
            - If dict: Dictionary with "queries" key
            - params["queries"]: List[str] - List of search queries

    Returns:
        str - Formatted search results with:
            - Numbered entries
            - Title with link in markdown format
            - Date and source metadata
            - Snippet preview

    Note:
        - Returns top 10 results per query
        - Parallel execution for multiple queries
        - Results are cached based on query string
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from qwen_agent.tools.private.nlp_web_search import WebSearch

web_search = WebSearch()
results = web_search.call({"queries": ["Python programming tutorial"]})
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| params || Union[str, dict] || Yes || Search parameters containing queries
|-
| params["queries"] || List[str] || Yes || List of search query strings
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || str || Formatted search results as markdown string
|}

'''Output Format:'''
<pre>
1. [Page Title](https://example.com/page)
   Date: 2024-01-15 | Source: example.com
   Snippet preview text describing the page content...

2. [Another Page](https://example2.com)
   Date: 2024-01-10 | Source: example2.com
   Another snippet with relevant information...
</pre>

== Usage Examples ==

=== Basic Search Query ===
<syntaxhighlight lang="python">
from qwen_agent.tools.private.nlp_web_search import WebSearch

web_search = WebSearch()

# Single query search
results = web_search.call({
    "queries": ["Eiffel Tower history"]
})

print(results)
# Output:
# 1. [Eiffel Tower - Wikipedia](https://en.wikipedia.org/wiki/Eiffel_Tower)
#    Date: | Source: wikipedia.org
#    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars...
# ...
</syntaxhighlight>

=== Multiple Queries ===
<syntaxhighlight lang="python">
from qwen_agent.tools.private.nlp_web_search import WebSearch

web_search = WebSearch()

# Multiple queries executed in parallel
results = web_search.call({
    "queries": [
        "Mount Fuji elevation",
        "Mount Fuji climbing season"
    ]
})

print(results)
</syntaxhighlight>

=== Agent Tool Call Integration ===
<syntaxhighlight lang="python">
# In agent context, tool calls come as parsed JSON
tool_call = {
    "name": "web_search",
    "arguments": {
        "queries": ["species of bird with blue feathers"]
    }
}

# The Qwen_agent routes this to WebSearch.call()
result = qwen_agent.execute_tool(tool_call)
# Result is appended to conversation as tool response
</syntaxhighlight>

=== With JSON String Input ===
<syntaxhighlight lang="python">
import json
from qwen_agent.tools.private.nlp_web_search import WebSearch

web_search = WebSearch()

# JSON string format (common in tool parsing)
params_str = json.dumps({
    "queries": ["Leonardo da Vinci paintings"]
})

results = web_search.call(params_str)
print(results)
</syntaxhighlight>

=== Extracting URLs for Visitation ===
<syntaxhighlight lang="python">
import re
from qwen_agent.tools.private.nlp_web_search import WebSearch

web_search = WebSearch()

results = web_search.call({"queries": ["Python decorators explained"]})

# Extract URLs from markdown links
urls = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', results)
print(f"Found {len(urls)} URLs to potentially visit:")
for url in urls[:3]:
    print(f"  - {url}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Text_Web_Search]]
