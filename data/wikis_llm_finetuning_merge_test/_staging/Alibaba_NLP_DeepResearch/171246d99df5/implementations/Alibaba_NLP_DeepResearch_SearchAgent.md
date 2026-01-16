# Implementation: SearchAgent

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|Qwen-Agent|https://github.com/QwenLM/Qwen-Agent]]
|-
! Domains
| [[domain::Web_Search]], [[domain::Agent]], [[domain::Tool_Use]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Extended Assistant agent class for web search with custom user prompts, reasoning mode support, and optional secondary agent chaining.

=== Description ===
The `SearchAgent` class extends Qwen-Agent's `Assistant` to create a specialized web search agent. It adds:

- Custom user prompt injection for task-specific instructions
- Dynamic system prompt generation via `make_system_prompt` callback
- Reasoning mode toggle for chain-of-thought outputs
- Configurable max LLM calls per conversation
- Secondary agent chaining via `addtional_agent` for post-processing

The agent implements a multi-turn tool-calling loop with search and visit tools.

=== Usage ===
Use `SearchAgent` when building web research assistants that need custom prompting, reasoning traces, or multi-agent pipelines.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/agents/search_agent.py WebAgent/WebDancer/demos/agents/search_agent.py]
* '''Lines:''' 1-113

=== Signature ===
<syntaxhighlight lang="python">
class SearchAgent(Assistant):
    """Web search agent with extended capabilities."""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        name: Optional[str] = None,
        description: Optional[str] = None,
        files: Optional[List[str]] = None,
        rag_cfg: Optional[Dict] = None,
        extra: Optional[Dict] = {},
        custom_user_prompt: Optional[str] = '',
        make_system_prompt: Optional[Callable] = None,
        addtional_agent: Optional[Any] = None
    ):
        """
        Initialize SearchAgent.

        Args:
            function_list: Tools to register (e.g., ['search', 'visit'])
            llm: LLM configuration dict or model
            system_message: Default system message
            name: Agent name
            description: Agent description
            extra: Dict with 'reasoning' (bool) and 'max_llm_calls' (int)
            custom_user_prompt: Prompt prefix for user messages
            make_system_prompt: Callback to generate system prompt
            addtional_agent: Secondary agent for post-processing
        """
        ...

    def _run(
        self,
        messages: List[Message],
        lang: Literal['en', 'zh'] = 'zh',
        knowledge: str = '',
        **kwargs
    ) -> Iterator[List[Message]]:
        """Execute search agent loop with tool calling."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.agents.search_agent import SearchAgent
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| function_list || List || No || Tool names or definitions
|-
| llm || Dict/Model || Yes || LLM configuration
|-
| extra || Dict || No || Contains 'reasoning', 'max_llm_calls'
|-
| custom_user_prompt || str || No || Prompt prefix for users
|-
| make_system_prompt || Callable || No || System prompt generator
|-
| addtional_agent || Agent || No || Post-processing agent
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| response || Iterator[List[Message]] || Streaming response messages
|-
| tool_calls || List || Tool invocation records
|}

== Usage Examples ==

=== Basic SearchAgent ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.agents.search_agent import SearchAgent
from WebAgent.WebDancer.demos.llm.oai import TextChatAtOAI

# Initialize LLM
llm = TextChatAtOAI({
    'model': 'gpt-4o',
    'api_key': 'your-key',
    'generate_cfg': {'temperature': 0.6}
})

# Create search agent
agent = SearchAgent(
    llm=llm,
    function_list=['search', 'visit'],
    name='WebSearcher',
    extra={'reasoning': True, 'max_llm_calls': 20}
)

# Run query
messages = [{'role': 'user', 'content': 'Find ACL 2025 deadline'}]
for response in agent.run(messages=messages):
    print(response[-1].content)
</syntaxhighlight>

=== With Custom System Prompt ===
<syntaxhighlight lang="python">
from datetime import datetime

def make_system_prompt():
    return f"""You are a web research assistant.
Current date: {datetime.now().strftime('%Y-%m-%d')}
Search thoroughly and verify information."""

agent = SearchAgent(
    llm=llm,
    function_list=['search', 'visit'],
    make_system_prompt=make_system_prompt,
    extra={'reasoning': True}
)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Agent_Initialization]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
