# Principle: Agent_Initialization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|ReAct: Synergizing Reasoning and Acting in Language Models|https://arxiv.org/abs/2210.03629]]
* [[source::Paper|Language Agent Tree Search|https://arxiv.org/abs/2310.04406]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::NLP]], [[domain::Web_Research]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

The ReAct agent pattern for autonomous web research. ReAct (Reasoning + Acting) is a framework where agents alternate between reasoning (thinking about what to do) and acting (executing tools).

=== Description ===

The ReAct (Reasoning and Acting) paradigm, introduced in "ReAct: Synergizing Reasoning and Acting in Language Models" (arXiv:2210.03629), represents a fundamental approach to building autonomous language agents. Unlike pure reasoning (chain-of-thought) or pure acting (tool-calling) approaches, ReAct interleaves both capabilities in a unified framework.

Key components of agent initialization:

1. **Tool Registration** - The agent is initialized with a list of available tools (search, visit, python, file parsing, scholar)
2. **LLM Configuration** - Model path and generation parameters (temperature, top_p, presence_penalty) are configured
3. **Agent Inheritance** - Extends FnCallAgent from qwen_agent for function calling capabilities
4. **Tool Mapping** - Creates a global TOOL_MAP dictionary mapping tool names to tool instances

The MultiTurnReactAgent class serves as the core orchestrator that:
- Maintains conversation state across multiple turns
- Dispatches tool calls to appropriate handlers
- Manages the reasoning-action loop until an answer is produced

=== Usage ===

Use Agent Initialization when:
- Setting up a new ReAct agent instance for web research tasks
- Configuring LLM parameters for the reasoning model
- Registering custom tools for the agent to use

Configuration parameters include:
| Parameter | Type | Description |
|-----------|------|-------------|
| function_list | List[Union[str, Dict, BaseTool]] | Available tools for the agent |
| llm.model | str | Path to the LLM model |
| llm.generate_cfg.temperature | float | Sampling temperature (default: 0.6) |
| llm.generate_cfg.top_p | float | Nucleus sampling parameter (default: 0.95) |
| llm.generate_cfg.presence_penalty | float | Repetition penalty (default: 1.1) |

== Theoretical Basis ==

The ReAct paradigm operates on the principle that effective problem-solving requires both internal reasoning and external information gathering. The agent follows this cycle:

<math>
s_{t+1} = \text{Agent}(s_t, \text{observation}_t)
</math>

Where each state transition involves:
1. **Thought**: Internal reasoning about current progress and next steps
2. **Action**: Tool selection and parameter generation
3. **Observation**: External feedback from tool execution

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# ReAct Agent Initialization Pattern
class ReActAgent:
    def __init__(self, tools: List[Tool], llm_config: Dict):
        # Store LLM configuration for generation
        self.generate_cfg = llm_config["generate_cfg"]
        self.model_path = llm_config["model"]

        # Register available tools
        self.tool_map = {tool.name: tool for tool in tools}

        # Initialize conversation state
        self.messages = []

    def run(self, query: str) -> str:
        # Initialize with system prompt and user query
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        # Enter ReAct loop
        while not done:
            # Generate thought + action
            response = self.llm_call(messages)

            # Parse and execute tool call
            if has_tool_call(response):
                result = self.execute_tool(response)
                messages.append({"role": "user", "content": result})

            # Check for answer
            if has_answer(response):
                return extract_answer(response)

        return final_answer
</syntaxhighlight>

Key initialization principles:
- **Lazy Loading**: Tools are instantiated once and reused across calls
- **Configuration Injection**: LLM parameters are passed at initialization, not runtime
- **Extensibility**: New tools can be added by extending the tool list

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__init__]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Web_Search_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Webpage_Visitation]]
