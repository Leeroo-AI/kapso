# Implementation: WebSailor_ReActAgent

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::ReAct_Agent]], [[domain::vLLM]], [[domain::Tool_Use]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
vLLM-powered ReAct agent for WebSailor with XML-based tool calling and configurable context management.

=== Description ===
The `MultiTurnReactAgent` class in WebSailor implements a ReAct agent optimized for vLLM serving:

- Multi-turn tool-calling loop with XML format (`<tool_call>`, `<tool_response>`, `<answer>`)
- Token counting via AutoTokenizer or tiktoken
- Configurable max LLM calls and token limits
- Graceful context overflow handling with forced answer generation
- Custom user prompt injection

The agent communicates with a local vLLM server for efficient inference.

=== Usage ===
Use for web research tasks requiring efficient vLLM serving. The agent handles tool calling through XML-formatted responses.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebSailor/src/react_agent.py WebAgent/WebSailor/src/react_agent.py]
* '''Lines:''' 1-162

=== Signature ===
<syntaxhighlight lang="python">
class MultiTurnReactAgent(FnCallAgent):
    """vLLM ReAct agent with XML tool calling."""

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
        ...

    def _run(
        self,
        data: Dict,
        model: str,
        user_prompt: str,
        **kwargs
    ) -> Dict:
        """
        Execute ReAct loop.

        Args:
            data: Dict with 'item' containing question/answer
            model: Model identifier for vLLM
            user_prompt: Custom prompt prefix

        Returns:
            Dict with prediction and trajectory
        """
        ...

    def call_server(self, msgs: List[Dict], max_tries: int = 10) -> str:
        """Call vLLM server at localhost:6001."""
        ...

    def count_tokens(self, messages: List[Dict], model: str = "gpt-4o") -> int:
        """Count tokens in conversation."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.react_agent import MultiTurnReactAgent
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| data || Dict || Yes || Item with question/answer
|-
| model || str || Yes || vLLM model identifier
|-
| user_prompt || str || Yes || Custom prompt prefix
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| question || str || Original question
|-
| prediction || str || Agent's answer
|-
| messages || List[Dict] || Full trajectory
|-
| termination || str || Termination reason
|}

== Usage Examples ==

=== Basic Execution ===
<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.react_agent import MultiTurnReactAgent
from WebAgent.WebSailor.src.prompt import SYSTEM_PROMPT_MULTI, USER_PROMPT

llm_cfg = {
    'model': '/path/to/model',
    'generate_cfg': {
        'temperature': 0.6,
        'top_p': 0.95
    },
    'model_type': 'qwen_dashscope'
}

agent = MultiTurnReactAgent(
    llm=llm_cfg,
    function_list=['search', 'visit'],
    system_message=SYSTEM_PROMPT_MULTI
)

result = agent._run(
    data={'item': {'question': 'ACL 2025 venue?', 'answer': 'Vienna'}, 'rollout_id': 1},
    model='/path/to/model',
    user_prompt=USER_PROMPT
)
print(result['prediction'])
</syntaxhighlight>

=== Environment Configuration ===
<syntaxhighlight lang="bash">
export MAX_LLM_CALL_PER_RUN=40
export MAX_LENGTH=31744  # 31k tokens
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
