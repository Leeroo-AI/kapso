# Implementation: Visit_call_multimodal

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|jialong_visit.py|WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/jialong_visit.py]]
|-
! Domains
| [[domain::Web_Scraping]], [[domain::Information_Retrieval]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Webpage visitation tool implementation that fetches content via multiple fallback services and summarizes with LLM.

=== Description ===

The `call()` method in the `Visit` class (registered as 'visit' tool) provides robust webpage content retrieval for the multimodal agent. It accepts URLs and a goal, attempts content fetch through multiple services in priority order, and returns LLM-summarized content focused on the goal.

The implementation features:
- Parallel URL processing via ThreadPoolExecutor (3 workers)
- Multi-service fallback: Wikipedia dict, AIData cache, AIData online, Jina
- LLM summarization using Qwen2.5-72B-Instruct-SummaryModel
- Progressive content truncation for pages exceeding token limits
- Structured output with rational, evidence, and summary

=== Usage ===

Use the `visit` tool when:
- The agent needs detailed content from URLs found in search
- Gathering specific information to answer a question
- Building knowledge from multiple web sources
- Verifying information from authoritative sources

The tool is typically called after web_search identifies relevant URLs.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/jialong_visit.py
* '''Lines:''' 68-90

=== Signature ===
<syntaxhighlight lang="python">
def call(self, params: Union[str, dict], **kwargs) -> str:
    """
    Visit URLs and extract goal-relevant information.

    Args:
        params: Union[str, dict] - Visit parameters:
            - If str: JSON string with "urls" and "goal" keys
            - If dict: Dictionary with "urls" and "goal" keys
            - params["urls"]: List[str] - URLs to visit
            - params["goal"]: str - Information extraction goal

    Returns:
        str - Summarized content from all URLs, containing:
            - rational: Why the content is relevant
            - evidence: Specific facts/quotes
            - summary: Concise goal-focused summary

    Note:
        - Uses ThreadPoolExecutor with 3 workers for parallel fetch
        - Falls back through multiple services if fetch fails
        - Long content is progressively truncated
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from qwen_agent.tools.private.jialong_visit import Visit

visit = Visit()
result = visit.call({
    "urls": ["https://en.wikipedia.org/wiki/Eiffel_Tower"],
    "goal": "When was the Eiffel Tower built?"
})
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| params || Union[str, dict] || Yes || Visit parameters
|-
| params["urls"] || List[str] || Yes || List of URLs to visit
|-
| params["goal"] || str || Yes || Information extraction goal/question
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || str || Combined summarized content from all URLs
|}

'''Output Structure:'''
<pre>
URL: https://example.com/page
Rational: This page discusses the topic in detail...
Evidence: "The construction began in 1887 and was completed in 1889..."
Summary: The structure was built between 1887-1889 for the World's Fair.
</pre>

== Usage Examples ==

=== Basic URL Visitation ===
<syntaxhighlight lang="python">
from qwen_agent.tools.private.jialong_visit import Visit

visit = Visit()

result = visit.call({
    "urls": ["https://en.wikipedia.org/wiki/Mona_Lisa"],
    "goal": "Who painted the Mona Lisa and when?"
})

print(result)
# Output includes:
# - Rational: The Wikipedia article about Mona Lisa...
# - Evidence: "painted by Leonardo da Vinci between 1503 and 1519"
# - Summary: The Mona Lisa was painted by Leonardo da Vinci...
</syntaxhighlight>

=== Multiple URLs ===
<syntaxhighlight lang="python">
from qwen_agent.tools.private.jialong_visit import Visit

visit = Visit()

# Visit multiple URLs in parallel
result = visit.call({
    "urls": [
        "https://en.wikipedia.org/wiki/Mount_Fuji",
        "https://www.japan-guide.com/e/e2172.html"
    ],
    "goal": "What is the best time to climb Mount Fuji?"
})

print(result)
</syntaxhighlight>

=== Agent Tool Call Integration ===
<syntaxhighlight lang="python">
# In agent context
tool_call = {
    "name": "visit",
    "arguments": {
        "urls": ["https://example.com/article"],
        "goal": "Extract the main findings"
    }
}

# Routed to Visit.call() by the agent
result = qwen_agent.execute_tool(tool_call)
</syntaxhighlight>

=== With JSON String Input ===
<syntaxhighlight lang="python">
import json
from qwen_agent.tools.private.jialong_visit import Visit

visit = Visit()

params_str = json.dumps({
    "urls": ["https://example.com/research-paper"],
    "goal": "What methodology was used in the study?"
})

result = visit.call(params_str)
print(result)
</syntaxhighlight>

=== Chaining with Web Search ===
<syntaxhighlight lang="python">
import re
from qwen_agent.tools.private.nlp_web_search import WebSearch
from qwen_agent.tools.private.jialong_visit import Visit

web_search = WebSearch()
visit = Visit()

# Step 1: Search for relevant pages
search_results = web_search.call({"queries": ["climate change effects on coral reefs"]})

# Step 2: Extract URLs
urls = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', search_results)[:3]

# Step 3: Visit top URLs for detailed content
if urls:
    detailed_info = visit.call({
        "urls": urls,
        "goal": "What are the main effects of climate change on coral reefs?"
    })
    print(detailed_info)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation_Multimodal]]
