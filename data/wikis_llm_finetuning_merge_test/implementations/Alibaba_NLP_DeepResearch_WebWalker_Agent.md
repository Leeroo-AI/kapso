# Implementation: WebWalker_Agent

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::ReAct_Agent]], [[domain::Website_Navigation]], [[domain::Information_Extraction]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Website navigator agent that explores web pages using ReAct with action/observation format and critic-based information accumulation.

=== Description ===
The `WebWalker` class implements a specialized ReAct agent for website navigation:

- ReAct format with `Action:`, `Action Input:`, `Observation:` tokens
- Information extraction from observations via `observation_information_extraction`
- Critic-based answer validation via `critic_information`
- Memory accumulation across navigation steps
- JSON-formatted response extraction

The agent iteratively explores webpages, extracts relevant information, and determines when sufficient evidence has been gathered.

=== Usage ===
Use for website-specific information seeking tasks where the agent navigates within a single site to find answers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWalker/src/agent.py WebAgent/WebWalker/src/agent.py]
* '''Lines:''' 1-208

=== Signature ===
<syntaxhighlight lang="python">
class WebWalker(FnCallAgent):
    """Website navigation agent with ReAct format."""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        name: Optional[str] = None,
        description: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize WebWalker.

        Args:
            llm: Dict with 'api_key', 'model_server', 'model', 'query'
        """
        ...

    def observation_information_extraction(
        self,
        query: str,
        observation: str
    ) -> Optional[str]:
        """Extract relevant info from observation."""
        ...

    def critic_information(
        self,
        query: str,
        memory: List[str]
    ) -> Optional[str]:
        """Validate if memory contains answer."""
        ...

    def _run(
        self,
        messages: List[Message],
        lang: Literal['en', 'zh'] = 'en',
        **kwargs
    ) -> Iterator[List[Message]]:
        """Execute navigation loop."""
        ...

    def _detect_tool(self, text: str) -> Tuple[bool, str, str, str]:
        """Parse Action/Action Input from text."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebWalker.src.agent import WebWalker
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| llm || Dict || Yes || Contains api_key, model_server, model, query
|-
| function_list || List || No || Navigation tools
|-
| messages || List[Message] || Yes || Initial messages
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| response || Iterator[List[Message]] || Streaming navigation steps
|-
| memory || List[str] || Accumulated information
|-
| final_answer || str || Answer via critic validation
|}

== Usage Examples ==

=== Basic Navigation ===
<syntaxhighlight lang="python">
from WebAgent.WebWalker.src.agent import WebWalker

llm_cfg = {
    'model': 'qwen-plus',
    'api_key': 'your-key',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'query': 'When is the ACL 2025 deadline?',
    'action_count': 10,
    'generate_cfg': {'top_p': 0.8, 'max_input_tokens': 120000}
}

agent = WebWalker(
    llm=llm_cfg,
    function_list=['visit_page']
)

messages = [{'role': 'user', 'content': 'Starting from https://2025.aclweb.org'}]
for response in agent.run(messages=messages):
    if 'Final Answer' in response[-1].content:
        print(response[-1].content)
        break
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Website_Navigation]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
