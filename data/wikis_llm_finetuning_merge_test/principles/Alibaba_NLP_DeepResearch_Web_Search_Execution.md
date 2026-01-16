# Principle: Web_Search_Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::API|Serper API|https://serper.dev/]]
* [[source::Paper|WebGPT: Browser-assisted question-answering with human feedback|https://arxiv.org/abs/2112.09332]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Web_Search]], [[domain::Information_Retrieval]], [[domain::Agent_Tools]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Web search tool execution using external search APIs. Batched queries are supported for efficiency, allowing multiple related searches in a single tool call.

=== Description ===

Web search execution is a fundamental capability for autonomous research agents. It allows the agent to retrieve current information from the internet by querying search engines and processing the results.

The implementation uses the Serper API (Google Search API) with the following features:

1. **Batched Queries** - Multiple search queries can be executed in a single tool call
2. **Locale Detection** - Automatically detects Chinese characters and adjusts search locale (China/zh-cn vs US/en)
3. **Result Formatting** - Returns structured results with title, link, date, source, and snippet
4. **Retry Logic** - Implements 5 retries for transient failures
5. **Top-10 Results** - Returns the top 10 organic search results per query

The search tool is registered under the name "search" and accepts an array of query strings.

=== Usage ===

Use Web Search Execution when:
- The agent needs to find current information not in its training data
- Multiple related queries should be executed together
- Discovering URLs for subsequent webpage visits

Search result structure per query:
| Field | Description |
|-------|-------------|
| title | Page title |
| link | URL to the webpage |
| date | Publication date (if available) |
| source | Source website name |
| snippet | Text excerpt from the page |

== Theoretical Basis ==

Web search serves as the primary information retrieval mechanism in research agents. The process follows:

<math>
\text{Results} = \text{SearchAPI}(\text{query}, \text{locale}, \text{top\_k})
</math>

Where locale is determined by:
<math>
\text{locale} = \begin{cases} \text{(cn, zh-cn)} & \text{if Chinese characters detected} \\ \text{(us, en)} & \text{otherwise} \end{cases}
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Web Search Execution Pattern
def search(queries: Union[str, List[str]]) -> str:
    """Execute web search with optional batching."""

    def single_search(query: str) -> str:
        # Detect locale based on character set
        if contains_chinese(query):
            locale = {"location": "China", "gl": "cn", "hl": "zh-cn"}
        else:
            locale = {"location": "United States", "gl": "us", "hl": "en"}

        # Call search API with retry
        for attempt in range(5):
            try:
                response = search_api.call(
                    query=query,
                    **locale
                )
                break
            except Exception:
                if attempt == 4:
                    return f"Search timeout for '{query}'"
                continue

        # Format results
        results = []
        for idx, page in enumerate(response["organic"][:10], 1):
            result = f"{idx}. [{page['title']}]({page['link']})"
            if "date" in page:
                result += f"\nDate published: {page['date']}"
            if "source" in page:
                result += f"\nSource: {page['source']}"
            if "snippet" in page:
                result += f"\n{page['snippet']}"
            results.append(result)

        return f"A Google search for '{query}' found {len(results)} results:\n\n" + \
               "## Web Results\n" + "\n\n".join(results)

    # Handle single or batch queries
    if isinstance(queries, str):
        return single_search(queries)
    else:
        results = [single_search(q) for q in queries]
        return "\n=======\n".join(results)
</syntaxhighlight>

Key search principles:
- **Batch Efficiency**: Multiple queries reduce round-trip overhead
- **Locale Awareness**: Results quality improves with appropriate locale settings
- **Graceful Degradation**: Timeout errors return informative messages
- **Structured Output**: Markdown formatting enables easy parsing by the LLM

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Search_call]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
