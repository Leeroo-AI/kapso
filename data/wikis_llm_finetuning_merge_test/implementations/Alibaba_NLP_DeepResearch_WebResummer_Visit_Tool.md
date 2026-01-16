# Implementation: WebResummer_Visit_Tool

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|Jina Reader|https://r.jina.ai/]]
|-
! Domains
| [[domain::Web_Scraping]], [[domain::Content_Extraction]], [[domain::LLM_Summarization]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Webpage visitor tool for WebResummer with Jina Reader for content extraction, LLM-based summarization, and token truncation handling.

=== Description ===
The `Visit` class in WebResummer provides intelligent webpage extraction:

- Jina Reader API for webpage-to-markdown conversion
- LLM-based extraction with goal-directed prompts
- Token truncation with tiktoken (max 95k tokens)
- Retry logic with progressive truncation on failures
- Support for single or multiple URLs
- Structured output with evidence and summary

The tool handles long webpages gracefully through progressive truncation.

=== Usage ===
Register in ReAct agents needing webpage content. Requires JINA_API_KEYS, SUMMARY_API_KEY, and SUMMARY_URL environment variables.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebResummer/src/tool_visit.py WebAgent/WebResummer/src/tool_visit.py]
* '''Lines:''' 1-240

=== Signature ===
<syntaxhighlight lang="python">
@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    """Webpage visitor with LLM extraction."""

    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": ["string", "array"], ...},
            "goal": {"type": "string", ...}
        },
        "required": ["url", "goal"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute webpage visits."""
        ...

    def readpage(self, url: str, goal: str) -> str:
        """Read and extract from single URL."""
        ...

    def call_server(self, msgs: List[Dict], max_retries: int = 2) -> str:
        """Call summarization LLM server."""
        ...

    def jina_readpage(self, url: str) -> str:
        """Fetch content via Jina Reader."""
        ...

@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """Truncate text to token limit."""
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.tool_visit import Visit, truncate_to_tokens
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| url || str/List[str] || Yes || URL(s) to visit
|-
| goal || str || Yes || Extraction goal
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| useful_information || str || Evidence and summary
|}

== Usage Examples ==

=== Basic Visit ===
<syntaxhighlight lang="python">
import os
from WebAgent.WebResummer.src.tool_visit import Visit

os.environ['JINA_API_KEYS'] = 'jina-key'
os.environ['SUMMARY_API_KEY'] = 'api-key'
os.environ['SUMMARY_URL'] = 'http://localhost:8000/v1'
os.environ['SUMMARY_MODEL_NAME'] = 'qwen2.5-72b-instruct'

visit = Visit()
result = visit.call({
    "url": "https://arxiv.org/abs/2401.12345",
    "goal": "Summarize the paper contributions"
})
print(result)
</syntaxhighlight>

=== Token Truncation ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.tool_visit import truncate_to_tokens

long_content = "..." * 100000
truncated = truncate_to_tokens(long_content, max_tokens=95000)
print(f"Truncated to {len(truncated)} chars")
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
