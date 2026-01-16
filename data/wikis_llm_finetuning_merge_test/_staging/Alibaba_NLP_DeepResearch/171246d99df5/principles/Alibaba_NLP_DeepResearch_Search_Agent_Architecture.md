# Principle: Search_Agent_Architecture

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::Architecture]], [[domain::Web_Research]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Component architecture for web search agents combining LLM reasoning with registered tools (search, visit) for autonomous research tasks.

=== Description ===

Search Agent Architecture defines how to structure a complete web research agent:

1. **LLM Backend** - Configurable language model for reasoning
2. **Tool Registry** - Search and visit tools with function calling
3. **Agent Loop** - Iterative reasoning-action cycle
4. **Memory** - Conversation history and extracted knowledge
5. **Configuration** - Model, tool, and behavior settings

The WebDancer SearchAgent implementation provides a reference architecture.

=== Usage ===

Use Search Agent Architecture when:
- Building new web research agents
- Need modular, configurable agent design
- Want to swap LLM backends or tools
- Creating demo or production search assistants

== Theoretical Basis ==

Agent composition pattern:

'''Search Agent Pattern:'''
<syntaxhighlight lang="python">
from qwen_agent.agent import Agent

class SearchAgent(Agent):
    """Web search agent with configurable tools."""

    def __init__(
        self,
        llm: dict,
        function_list: list[str],
        name: str = "SearchAgent",
        description: str = "Web research assistant",
        **kwargs
    ):
        """
        Initialize search agent.

        Args:
            llm: LLM configuration dict with model, api_key, etc.
            function_list: List of tool names ['search', 'visit']
            name: Agent display name
            description: Agent description
        """
        super().__init__(
            function_list=function_list,
            llm=llm,
            name=name,
            description=description,
            **kwargs
        )

    async def run(self, messages: list[dict]) -> str:
        """Execute agent reasoning loop."""
        # Inherited from base Agent class
        # Implements ReAct-style loop with tool calling
        ...
</syntaxhighlight>

Architecture components:
- **Base class**: Inherit from Agent framework
- **Tool registration**: Use @register_tool decorator
- **LLM configuration**: Dict with model settings
- **Function list**: Tools available to agent

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_SearchAgent]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_LLM_Backend]]
