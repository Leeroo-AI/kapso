# Implementation: WebDancer_Demo

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Demo]], [[domain::Web_Interface]], [[domain::Agent]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
WebDancer demo entry point that initializes search agents and launches the Gradio web interface for interactive web research.

=== Description ===
The `assistant_qwq_chat.py` module serves as the main entry point for the WebDancer demo application. It provides:

- `init_dev_search_agent_service`: Factory function to create configured SearchAgent instances
- `app_gui`: Main function that sets up multiple agents and launches WebUI
- Pre-configured example prompts in Chinese and English
- Integration with local vLLM servers for model serving

The demo showcases WebDancer's capabilities for deep web research with reasoning traces.

=== Usage ===
Run this module directly to launch the WebDancer demo interface, or import `init_dev_search_agent_service` to create agents programmatically.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/assistant_qwq_chat.py WebAgent/WebDancer/demos/assistant_qwq_chat.py]
* '''Lines:''' 1-140

=== Signature ===
<syntaxhighlight lang="python">
def init_dev_search_agent_service(
    name: str = 'SEARCH',
    port: int = 8002,
    desc: str = 'Description',
    reasoning: bool = True,
    max_llm_calls: int = 20,
    tools: List[str] = ['search', 'visit'],
    addtional_agent: Optional[Any] = None
) -> SearchAgent:
    """
    Initialize a configured SearchAgent.

    Args:
        name: Agent name for UI display
        port: vLLM server port
        desc: Agent description
        reasoning: Enable chain-of-thought
        max_llm_calls: Max tool-calling iterations
        tools: List of tool names
        addtional_agent: Optional secondary agent

    Returns:
        Configured SearchAgent instance
    """
    ...

def app_gui() -> None:
    """
    Launch WebDancer Gradio demo interface.

    Sets up agents and WebUI with example prompts.
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.assistant_qwq_chat import (
    init_dev_search_agent_service,
    app_gui
)
</syntaxhighlight>

== I/O Contract ==

=== init_dev_search_agent_service ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| name || str || Display name
|-
| port || int || vLLM server port
|-
| reasoning || bool || Enable CoT
|-
| max_llm_calls || int || Max iterations
|-
| tools || List[str] || Tool names
|}

{| class="wikitable"
|-
! Output !! Type !! Description
|-
| agent || SearchAgent || Configured agent
|}

== Usage Examples ==

=== Launch Demo Interface ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.assistant_qwq_chat import app_gui

# Launch Gradio interface
if __name__ == '__main__':
    app_gui()
</syntaxhighlight>

=== Create Custom Agent ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.assistant_qwq_chat import init_dev_search_agent_service

# Create agent with custom config
agent = init_dev_search_agent_service(
    name='CustomSearcher',
    port=8004,
    reasoning=True,
    max_llm_calls=30,
    tools=['search', 'visit']
)

# Use agent
messages = [{'role': 'user', 'content': 'Research question...'}]
for response in agent.run(messages=messages):
    print(response[-1].content)
</syntaxhighlight>

=== Command Line Launch ===
<syntaxhighlight lang="bash">
# Run the demo
python WebAgent/WebDancer/demos/assistant_qwq_chat.py

# Access at http://127.0.0.1:7860
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Demo_Interface]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
