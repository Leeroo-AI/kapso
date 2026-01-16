# Principle: Text_Web_Search

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

Text-based web search for multimodal workflows. Uses Serper API to find relevant web pages.

=== Description ===

Text Web Search provides the ability to query web search engines and retrieve relevant results. In multimodal workflows, this complements visual search by enabling text-based information gathering about entities, facts, or concepts identified in images.

The search tool:

1. **Query Processing** - Accept search queries from the agent
2. **API Invocation** - Call Serper API (Google search backend) with the query
3. **Result Parsing** - Extract title, URL, date, source, and snippet from organic results
4. **Formatting** - Return structured results ready for agent consumption

Key features:
- Returns top 10 organic search results
- Parallel query processing via ThreadPoolExecutor
- Retry logic (5 attempts) for API resilience
- Formatted markdown-style output with clickable links

=== Usage ===

Use text web search when:
- The agent needs factual information not in its training data
- Following up on entities identified in images
- Gathering current/recent information about topics
- Finding authoritative sources for verification

Text web search is typically used after visual analysis to gather supporting information.

== Theoretical Basis ==

The text web search pipeline:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Text web search via Serper API
def google_search(query: str) -> List[Dict]:
    # Step 1: Configure request
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": 10  # Request top 10 results
    }

    # Step 2: Execute search with retry logic
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, headers=headers)
            results = response.json()
            break
        except Exception as e:
            time.sleep(1)
            continue

    # Step 3: Parse organic results
    parsed_results = []
    for item in results.get("organic", []):
        parsed_results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "date": item.get("date", ""),
            "source": item.get("source", ""),
            "snippet": item.get("snippet", "")
        })

    return parsed_results

# Format for agent consumption
def format_results(results: List[Dict]) -> str:
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. [{r['title']}]({r['link']})")
        formatted.append(f"   Date: {r['date']} | Source: {r['source']}")
        formatted.append(f"   {r['snippet']}")
    return "\n".join(formatted)
</syntaxhighlight>

The search results provide:
- '''Title''' - Page title for quick identification
- '''Link''' - URL for subsequent visitation
- '''Date''' - Publication/update date for recency assessment
- '''Snippet''' - Preview text for relevance evaluation

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebSearch_call]]
