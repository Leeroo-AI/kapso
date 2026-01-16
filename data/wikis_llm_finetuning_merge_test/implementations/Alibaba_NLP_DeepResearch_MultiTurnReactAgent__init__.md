# Implementation: MultiTurnReactAgent__init__

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::NLP]], [[domain::Web_Research]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Constructor for the MultiTurnReactAgent class that initializes a ReAct-style agent for autonomous web research.

=== Description ===

The `__init__` method of `MultiTurnReactAgent` sets up the agent with LLM configuration parameters needed for the reasoning and generation process. It inherits from `FnCallAgent` (qwen_agent) to leverage function calling capabilities.

The initialization extracts two key configuration items:
1. **generate_cfg** - Generation parameters (temperature, top_p, presence_penalty, max_tokens)
2. **model** - Path to the local LLM model for tokenization

The class works in conjunction with a global TOOL_MAP that maps tool names to pre-instantiated tool objects (FileParser, Scholar, Visit, Search, PythonInterpreter).

=== Usage ===

Use `MultiTurnReactAgent.__init__()` when:
- Creating a new agent instance for web research tasks
- Configuring the LLM backend for the agent
- Setting up generation parameters for reasoning model calls

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/react_agent.py
* '''Lines:''' 47-55

=== Signature ===
<syntaxhighlight lang="python">
class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):
        """
        Initialize the MultiTurnReactAgent.

        Args:
            function_list: Optional list of tools/functions available to the agent.
                          Can be strings, dicts, or BaseTool instances.
            llm: LLM configuration dictionary containing:
                - 'model': Path to the local model for tokenization
                - 'generate_cfg': Generation parameters dict with keys:
                    - temperature: float (default 0.6)
                    - top_p: float (default 0.95)
                    - presence_penalty: float (default 1.1)
            **kwargs: Additional keyword arguments passed to parent class.

        Note:
            Does not call super().__init__() - relies on global TOOL_MAP for tools.
        """
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| function_list || Optional[List[Union[str, Dict, BaseTool]]] || No || List of available tools (not used directly; relies on TOOL_MAP)
|-
| llm || Optional[Union[Dict, BaseChatModel]] || Yes || LLM configuration dict with 'model' and 'generate_cfg' keys
|-
| **kwargs || dict || No || Additional arguments (passed through)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (instance) || MultiTurnReactAgent || Initialized agent instance ready for _run() calls
|}

== Usage Examples ==

=== Basic Initialization ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

# Configure LLM settings
llm_config = {
    "model": "/path/to/qwen-model",
    "generate_cfg": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 1.1
    }
}

# Create agent instance
agent = MultiTurnReactAgent(
    function_list=None,  # Uses global TOOL_MAP
    llm=llm_config
)
</syntaxhighlight>

=== Full Pipeline Setup ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent
import os

# Set environment variables for tools
os.environ['SERPER_KEY_ID'] = 'your-serper-api-key'
os.environ['JINA_API_KEYS'] = 'your-jina-api-key'
os.environ['SANDBOX_FUSION_ENDPOINT'] = 'http://localhost:8080'

# LLM configuration
llm_config = {
    "model": "/models/Qwen2.5-72B-Instruct",
    "generate_cfg": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 1.1
    }
}

# Initialize agent
agent = MultiTurnReactAgent(llm=llm_config)

# Prepare data for _run method
data = {
    "item": {
        "question": "What is the capital of France?",
        "answer": "Paris"  # Ground truth for evaluation
    },
    "planning_port": 8000
}

# Execute research
result = agent._run(data, model="qwen-72b")
print(result["prediction"])
</syntaxhighlight>

=== Accessing Instance Attributes ===
<syntaxhighlight lang="python">
from react_agent import MultiTurnReactAgent

llm_config = {
    "model": "/models/Qwen2.5-72B-Instruct",
    "generate_cfg": {
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 1.0
    }
}

agent = MultiTurnReactAgent(llm=llm_config)

# Access stored configuration
print(f"Model path: {agent.llm_local_path}")
print(f"Temperature: {agent.llm_generate_cfg['temperature']}")
print(f"Top-p: {agent.llm_generate_cfg['top_p']}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Agent_Initialization]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_count_tokens]]
