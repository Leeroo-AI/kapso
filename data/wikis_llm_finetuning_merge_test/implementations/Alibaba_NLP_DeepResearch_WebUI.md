# Implementation: WebUI

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|Gradio Documentation|https://www.gradio.app/docs]]
|-
! Domains
| [[domain::Web_Interface]], [[domain::Gradio]], [[domain::Chat_UI]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Gradio-based chat web interface for WebDancer search agents with multi-agent selection, streaming responses, and customizable chatbot configurations.

=== Description ===
The `WebUI` class provides a production-ready web chat interface built on Gradio for interacting with WebDancer search agents. It supports multiple agent configurations in a dropdown, streaming message display with typing indicators, conversation history management, and customizable chatbot UI settings like suggested prompts. The interface handles multi-turn conversations with proper message formatting and displays both user queries and agent reasoning/tool calls.

Key features include:
- Multi-agent selection dropdown for comparing different model configurations
- Streaming response display with real-time token rendering
- Conversation history with reset functionality
- Configurable prompt suggestions for quick queries
- Support for concurrent users with proper session management

=== Usage ===
Import and instantiate WebUI when building a web demo for WebDancer or similar search agents. Use this for creating interactive demos, user testing, or production chat interfaces.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebDancer/demos/gui/web_ui.py WebAgent/WebDancer/demos/gui/web_ui.py]
* '''Lines:''' 1-389

=== Signature ===
<syntaxhighlight lang="python">
class WebUI:
    """Gradio-based chat interface for search agents."""

    def __init__(
        self,
        agent: Union[List, object],
        chatbot_config: Optional[Dict] = None
    ):
        """
        Initialize WebUI with agent(s) and configuration.

        Args:
            agent: Single agent or list of agents for selection dropdown
            chatbot_config: Configuration dict with keys:
                - 'prompt.suggestions': List of suggested prompts
                - 'user.name': Display name for user (default 'User')
                - 'verbose': Enable verbose logging (default False)
        """
        ...

    def run(
        self,
        message: Optional[Dict] = None,
        share: bool = False,
        server_name: str = '127.0.0.1',
        server_port: int = 7860,
        concurrency_limit: int = 20,
        enable_mention: bool = False
    ) -> None:
        """
        Launch the Gradio web interface.

        Args:
            message: Initial message to display
            share: Create public Gradio share link
            server_name: Server host address
            server_port: Server port number
            concurrency_limit: Max concurrent users
            enable_mention: Enable @agent mentions
        """
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.gui.web_ui import WebUI
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| agent || Union[List, Agent] || Yes || Agent or list of agents to use
|-
| chatbot_config || Dict || No || UI configuration options
|-
| share || bool || No || Create public share link (default False)
|-
| server_name || str || No || Server host (default '127.0.0.1')
|-
| server_port || int || No || Server port (default 7860)
|-
| concurrency_limit || int || No || Max concurrent users (default 20)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Gradio Interface || gr.Blocks || Running Gradio web application
|-
| Chat History || List[Dict] || Conversation messages stored in session
|-
| Share URL || str || Public URL if share=True
|}

== Usage Examples ==

=== Basic WebUI Setup ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.gui.web_ui import WebUI
from WebAgent.WebDancer.demos.agents.search_agent import SearchAgent
from WebAgent.WebDancer.demos.llm.oai import TextChatAtOAI

# Initialize LLM backend
llm_cfg = TextChatAtOAI({
    'model': 'gpt-4o',
    'api_key': 'your-api-key',
    'generate_cfg': {
        'temperature': 0.6,
        'max_tokens': 4096
    }
})

# Create search agent
agent = SearchAgent(
    llm=llm_cfg,
    function_list=['search', 'visit'],
    name='WebDancer-GPT4',
    description='Web search assistant'
)

# Configure chatbot UI
chatbot_config = {
    'prompt.suggestions': [
        'What is the latest news about AI?',
        'Find information about climate change',
        'Research quantum computing advances'
    ],
    'user.name': 'User',
    'verbose': True
}

# Launch WebUI
WebUI(
    agent=agent,
    chatbot_config=chatbot_config
).run(
    server_name='0.0.0.0',
    server_port=7860,
    share=False
)
</syntaxhighlight>

=== Multi-Agent Comparison ===
<syntaxhighlight lang="python">
from WebAgent.WebDancer.demos.gui.web_ui import WebUI

# Create multiple agents with different configs
agents = []
for name, model in [('GPT-4o', 'gpt-4o'), ('Qwen-Max', 'qwen-max')]:
    agent = SearchAgent(
        llm=TextChatAtOAI({'model': model, ...}),
        function_list=['search', 'visit'],
        name=name
    )
    agents.append(agent)

# Launch with agent selector dropdown
WebUI(agent=agents).run(
    server_port=7860,
    concurrency_limit=50
)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Demo_Interface]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
