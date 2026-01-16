# Implementation: WebSailor_Visit_Tool

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
Webpage visitor tool for WebSailor with Jina Reader extraction and vLLM-based summarization.

=== Description ===
The `Visit` class in WebSailor provides intelligent webpage extraction:

- Jina Reader API for webpage content extraction
- vLLM server for goal-directed summarization
- Configurable max content length (WEBCONTENT_MAXLENGTH)
- Parallel URL processing with ThreadPoolExecutor
- Progressive truncation on extraction failures
- Structured output with evidence and summary

The tool alternates between retry attempts when extraction fails.

=== Usage ===
Register in WebSailor agents. Requires JINA_API_KEY environment variable and local vLLM server.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebSailor/src/tool_visit.py WebAgent/WebSailor/src/tool_visit.py]
* '''Lines:''' 1-220

=== Signature ===
<syntaxhighlight lang="python">
@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    """Webpage visitor with vLLM extraction."""

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
        ...

    def readpage(self, url: str, goal: str) -> str:
        ...

    def call_server(self, msgs: List[Dict], max_tries: int = 10) -> str:
        """Call vLLM server at localhost:6002."""
        ...

    def jina_readpage(self, url: str) -> str:
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.tool_visit import Visit
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

<syntaxhighlight lang="python">
import os
from WebAgent.WebSailor.src.tool_visit import Visit

os.environ['JINA_API_KEY'] = 'your-key'
os.environ['WEBCONTENT_MAXLENGTH'] = '150000'

visit = Visit()
result = visit.call({
    "url": "https://example.com",
    "goal": "Extract main information"
})
print(result)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
