# Implementation: Search_call

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::API|Serper API|https://serper.dev/]]
|-
! Domains
| [[domain::Web_Search]], [[domain::Information_Retrieval]], [[domain::Agent_Tools]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

The call method of the Search tool that executes web searches using the Serper (Google Search) API.

=== Description ===

The `Search.call()` method is the entry point for web search execution in the DeepResearch agent. It wraps the Serper API to provide Google search capabilities with support for:

- Single query execution
- Batched query execution (multiple queries in one call)
- Automatic locale detection for Chinese vs English queries
- Formatted markdown output with titles, links, dates, and snippets

The method delegates to `search_with_serp()` which in turn calls `google_search_with_serp()` for the actual API interaction.

=== Usage ===

Use `Search.call()` when:
- The agent needs to search the web for information
- Multiple related queries should be executed together
- Finding URLs for subsequent webpage visits

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/tool_search.py
* '''Lines:''' 113-130

=== Signature ===
<syntaxhighlight lang="python">
def call(self, params: Union[str, dict], **kwargs) -> str:
    """
    Execute web search with single or multiple queries.

    Args:
        params: Union[str, dict] - Input parameters:
            - If dict: Must contain "query" key with str or List[str] value
            - Array queries are executed sequentially and joined
        **kwargs: Additional arguments (unused)

    Returns:
        str: Formatted search results in markdown format.
            - For single query: Results for that query
            - For multiple queries: Results joined by "\\n=======\\n"

    Raises:
        Returns error string if params format is invalid.

    Environment:
        SERPER_KEY_ID: API key for Serper service
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from tool_search import Search

# Or access via TOOL_MAP
from react_agent import TOOL_MAP
search_tool = TOOL_MAP["search"]
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| params || Union[str, dict] || Yes || Search parameters
|-
| params["query"] || Union[str, List[str]] || Yes || Single query string or array of queries
|-
| **kwargs || dict || No || Additional arguments (unused)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || str || Formatted markdown search results
|}

=== Output Format ===
<syntaxhighlight lang="text">
A Google search for 'query' found N results:

## Web Results
1. [Page Title](https://url.com)
Date published: 2024-01-15
Source: example.com
Snippet text from the page...

2. [Another Title](https://url2.com)
...
</syntaxhighlight>

== Usage Examples ==

=== Single Query Search ===
<syntaxhighlight lang="python">
from tool_search import Search
import os

os.environ['SERPER_KEY_ID'] = 'your-api-key'

search = Search()

# Single query
result = search.call({"query": "latest AI research papers 2024"})
print(result)
</syntaxhighlight>

=== Batched Multi-Query Search ===
<syntaxhighlight lang="python">
from tool_search import Search

search = Search()

# Multiple queries in one call
params = {
    "query": [
        "machine learning best practices",
        "deep learning optimization techniques",
        "transformer architecture improvements"
    ]
}

result = search.call(params)

# Results are separated by "======="
for i, section in enumerate(result.split("\n=======\n")):
    print(f"--- Query {i+1} Results ---")
    print(section[:500])
    print()
</syntaxhighlight>

=== Agent Tool Call Format ===
<syntaxhighlight lang="python">
# How the agent formats tool calls
tool_call_json = '''
{
    "name": "search",
    "arguments": {
        "query": ["population of Tokyo 2023", "Tokyo metropolitan area statistics"]
    }
}
'''

# Executed by react_agent.py
import json5
tool_call = json5.loads(tool_call_json)
result = TOOL_MAP[tool_call["name"]].call(tool_call["arguments"])
</syntaxhighlight>

=== Handling Chinese Queries ===
<syntaxhighlight lang="python">
from tool_search import Search

search = Search()

# Chinese query - automatically uses China locale
chinese_result = search.call({"query": "2024年人工智能发展趋势"})

# English query - uses US locale
english_result = search.call({"query": "AI development trends 2024"})
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from tool_search import Search

search = Search()

# Invalid format returns error message
result = search.call("not a dict")
# Returns: "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

# Query with no results
result = search.call({"query": "xyzabc123nonexistent"})
# Returns: "No results found for 'xyzabc123nonexistent'. Try with a more general query."

# Timeout after retries
result = search.call({"query": "test"})
# May return: "Google search Timeout, return None, Please try again later."
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_Visit_call]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]

=== Requires Environment ===
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:Alibaba_NLP_DeepResearch_Locale_Detection_Search]]
