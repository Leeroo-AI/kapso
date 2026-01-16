# Implementation: Tool_Explore

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Information_Extraction]], [[domain::NLP]], [[domain::Summarization]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Content extraction and summarization tool that processes webpage responses to extract relevant information for NestBrowse agent memory.

=== Description ===
The `tool_explore.py` module provides the `process_response` function that takes raw webpage content and extracts key information relevant to the user's query. It implements:

- Content cleaning and normalization
- Key information extraction based on query context
- Summary generation for agent memory
- Structured output formatting

This tool is essential for converting verbose webpage content into concise, actionable information for the agent.

=== Usage ===
Call `process_response` after visiting a webpage to extract and summarize the content for use in subsequent agent reasoning.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/NestBrowse/toolkit/tool_explore.py WebAgent/NestBrowse/toolkit/tool_explore.py]
* '''Lines:''' 1-47

=== Signature ===
<syntaxhighlight lang="python">
def process_response(
    content: str,
    query: str,
    max_length: int = 2000
) -> str:
    """
    Process webpage content and extract relevant information.

    Args:
        content: Raw webpage content
        query: User's original query for context
        max_length: Maximum output length

    Returns:
        Processed and summarized content
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.tool_explore import process_response
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| content || str || Yes || Raw webpage content
|-
| query || str || Yes || User query for context
|-
| max_length || int || No || Max output length (default 2000)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| summary || str || Extracted and summarized content
|}

== Usage Examples ==

=== Basic Content Processing ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.tool_explore import process_response

# Raw webpage content
raw_content = """
ACL 2025 Conference Information
The 63rd Annual Meeting of the Association for Computational Linguistics
will be held in Vienna, Austria from July 27 to August 1, 2025.
Important Dates:
- Paper submission deadline: February 15, 2025
- Notification: April 20, 2025
- Camera-ready: May 10, 2025
... (more content)
"""

# Process with query context
query = "When is the paper deadline for ACL 2025?"
summary = process_response(
    content=raw_content,
    query=query,
    max_length=500
)
print(summary)
# Output: "Paper submission deadline: February 15, 2025"
</syntaxhighlight>

=== Integration with Visit Tool ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.browser import Visit
from WebAgent.NestBrowse.toolkit.tool_explore import process_response

# Visit page
visit = Visit()
raw_content = visit.call({"url": "https://2025.aclweb.org"})

# Process for agent memory
query = "Find the venue address"
processed = process_response(raw_content, query)

# Add to agent memory
agent_memory.append({
    "url": "https://2025.aclweb.org",
    "summary": processed
})
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Page_Content_Extraction]]
