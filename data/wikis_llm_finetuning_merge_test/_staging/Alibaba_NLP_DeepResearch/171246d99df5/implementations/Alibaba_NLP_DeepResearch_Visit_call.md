# Implementation: Visit_call

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::API|Jina AI Reader|https://jina.ai/reader/]]
|-
! Domains
| [[domain::Web_Scraping]], [[domain::Content_Extraction]], [[domain::Agent_Tools]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

The call method of the Visit tool that retrieves webpage content and extracts goal-relevant information using LLM summarization.

=== Description ===

The `Visit.call()` method is the entry point for webpage content extraction in the DeepResearch agent. It implements a two-stage process:

1. **Content Fetching** - Uses Jina AI reader service to convert webpages to markdown
2. **LLM Extraction** - Applies EXTRACTOR_PROMPT to extract evidence and summary

Key implementation details:
- Token truncation using tiktoken (cl100k_base encoding) to 95,000 tokens
- Retry logic with progressive content truncation on failure
- Batch URL processing with 15-minute timeout
- JSON output parsing with fallback extraction

=== Usage ===

Use `Visit.call()` when:
- The agent needs to read full webpage content
- Extracting specific information based on a research goal
- Processing multiple URLs discovered through search

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/tool_visit.py
* '''Lines:''' 64-97

=== Signature ===
<syntaxhighlight lang="python">
def call(self, params: Union[str, dict], **kwargs) -> str:
    """
    Visit webpage(s) and return goal-relevant summary.

    Args:
        params: Union[str, dict] - Input parameters:
            - Must be dict containing "url" and "goal" keys
            - url: Union[str, List[str]] - Single URL or array of URLs
            - goal: str - The extraction goal/purpose
        **kwargs: Additional arguments (unused)

    Returns:
        str: Formatted extraction results containing:
            - Evidence: Direct relevant quotes from the page
            - Summary: Synthesized information paragraph

    Environment Variables:
        JINA_API_KEYS: API key for Jina reader service
        API_KEY: API key for summary LLM
        API_BASE: Base URL for summary LLM
        SUMMARY_MODEL_NAME: Model name for summarization
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from tool_visit import Visit

# Or access via TOOL_MAP
from react_agent import TOOL_MAP
visit_tool = TOOL_MAP["visit"]
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| params || Union[str, dict] || Yes || Visit parameters
|-
| params["url"] || Union[str, List[str]] || Yes || Single URL or array of URLs to visit
|-
| params["goal"] || str || Yes || The extraction goal/purpose
|-
| **kwargs || dict || No || Additional arguments (unused)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || str || Formatted extraction results with evidence and summary
|}

=== Output Format ===
<syntaxhighlight lang="text">
The useful information in {url} for user goal {goal} as follows:

Evidence in page:
[Extracted evidence from the webpage...]

Summary:
[Synthesized summary paragraph...]

</syntaxhighlight>

== Usage Examples ==

=== Single URL Visit ===
<syntaxhighlight lang="python">
from tool_visit import Visit
import os

# Set environment variables
os.environ['JINA_API_KEYS'] = 'your-jina-api-key'
os.environ['API_KEY'] = 'your-llm-api-key'
os.environ['API_BASE'] = 'https://api.openai.com/v1'
os.environ['SUMMARY_MODEL_NAME'] = 'gpt-4'

visit = Visit()

# Visit a single URL
result = visit.call({
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "goal": "Find the definition and history of artificial intelligence"
})

print(result)
</syntaxhighlight>

=== Batch URL Processing ===
<syntaxhighlight lang="python">
from tool_visit import Visit

visit = Visit()

# Visit multiple URLs
params = {
    "url": [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3"
    ],
    "goal": "Extract key statistics about climate change"
}

result = visit.call(params)

# Results are separated by "======="
for i, section in enumerate(result.split("\n=======\n")):
    print(f"--- URL {i+1} Results ---")
    print(section[:500])
    print()
</syntaxhighlight>

=== Agent Tool Call Format ===
<syntaxhighlight lang="python">
# How the agent formats tool calls
tool_call_json = '''
{
    "name": "visit",
    "arguments": {
        "url": "https://www.nature.com/articles/some-paper",
        "goal": "Extract the main findings and methodology of the study"
    }
}
'''

# Executed by react_agent.py
import json5
tool_call = json5.loads(tool_call_json)
result = TOOL_MAP[tool_call["name"]].call(tool_call["arguments"])
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from tool_visit import Visit

visit = Visit()

# Invalid format returns error message
result = visit.call({"url": "https://example.com"})
# Returns: "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

# Failed page access
result = visit.call({
    "url": "https://nonexistent-domain-12345.com/page",
    "goal": "Find information"
})
# Returns:
# The useful information in {url} for user goal {goal} as follows:
#
# Evidence in page:
# The provided webpage content could not be accessed. Please check the URL or file format.
#
# Summary:
# The webpage content could not be processed, and therefore, no information is available.
</syntaxhighlight>

=== Processing Search Results ===
<syntaxhighlight lang="python">
from tool_search import Search
from tool_visit import Visit
import re

search = Search()
visit = Visit()

# First, search for relevant pages
search_results = search.call({"query": "latest quantum computing breakthroughs 2024"})

# Extract URLs from search results
urls = re.findall(r'\((https?://[^\)]+)\)', search_results)[:3]  # Top 3 URLs

# Visit the URLs with a specific goal
visit_results = visit.call({
    "url": urls,
    "goal": "Extract specific quantum computing achievements and breakthroughs in 2024"
})

print(visit_results)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_Search_call]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]

=== Requires Environment ===
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:Alibaba_NLP_DeepResearch_Exponential_Backoff_Retry]]
