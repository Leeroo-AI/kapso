# Implementation: WebResummer_ReActAgent

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::ReAct_Agent]], [[domain::Summarization]], [[domain::Context_Management]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
ReAct agent with integrated ReSum (conversation summarization) for maintaining long context through periodic compression of conversation history.

=== Description ===
The `MultiTurnReactAgent` class in WebResummer extends the base ReAct agent with ReSum capability. Key features:

- Multi-turn tool-calling loop with XML-formatted tool calls
- Periodic conversation summarization via ReSum-Tool model
- Token counting with automatic context overflow handling
- Configurable max LLM calls and context limits
- Full trajectory logging for analysis

The ReSum mechanism triggers when context exceeds threshold, compressing conversation history while preserving key information.

=== Usage ===
Use this agent for long-context web research tasks where conversation history would exceed model limits. The ReSum feature maintains coherent context.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebResummer/src/react_agent.py WebAgent/WebResummer/src/react_agent.py]
* '''Lines:''' 1-202

=== Signature ===
<syntaxhighlight lang="python">
class MultiTurnReactAgent(FnCallAgent):
    """ReAct agent with ReSum conversation summarization."""

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
        """Initialize agent with LLM config containing generate_cfg and model path."""
        ...

    def _run(
        self,
        data: Dict,
        model: str,
        summary_iteration: int,
        **kwargs
    ) -> Dict:
        """
        Execute ReAct loop with ReSum.

        Args:
            data: Dict with 'item' containing 'question' and 'answer'
            model: Model identifier
            summary_iteration: Rounds between summarizations

        Returns:
            Dict with question, answer, messages, prediction, termination
        """
        ...

    def call_server(self, msgs: List[Dict], max_tries: int = 10) -> str:
        """Call vLLM server for generation."""
        ...

    def count_tokens(self, messages: List[Dict], model: str = "gpt-4o") -> int:
        """Count tokens in message history."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.react_agent import MultiTurnReactAgent
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| data || Dict || Yes || Contains 'item' with question/answer
|-
| model || str || Yes || Model path or identifier
|-
| summary_iteration || int || Yes || Rounds between ReSum calls
|-
| llm || Dict || Yes || LLM config with generate_cfg
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| question || str || Original question
|-
| answer || str || Reference answer
|-
| messages || List[Dict] || Full trajectory
|-
| prediction || str || Agent's final answer
|-
| termination || str || Reason for termination
|}

== Usage Examples ==

=== Basic Agent Run ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.react_agent import MultiTurnReactAgent
from WebAgent.WebResummer.src.prompt import SYSTEM_PROMPT

llm_cfg = {
    'model': '/path/to/model',
    'generate_cfg': {
        'temperature': 0.85,
        'top_p': 0.95
    },
    'model_type': 'qwen_dashscope'
}

agent = MultiTurnReactAgent(
    llm=llm_cfg,
    function_list=['search', 'visit'],
    system_message=SYSTEM_PROMPT
)

data = {
    'item': {
        'question': 'When is the ACL 2025 deadline?',
        'answer': 'February 15, 2025'
    },
    'rollout_id': 1
}

result = agent._run(
    data=data,
    model='/path/to/model',
    summary_iteration=10  # Summarize every 10 rounds
)

print(f"Prediction: {result['prediction']}")
print(f"Termination: {result['termination']}")
</syntaxhighlight>

=== Environment Variables ===
<syntaxhighlight lang="bash">
# Configure agent behavior
export MAX_LLM_CALL_PER_RUN=60
export RESUM=True  # Enable ReSum
export MAX_CONTEXT=32  # 32k context limit
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[implements::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
