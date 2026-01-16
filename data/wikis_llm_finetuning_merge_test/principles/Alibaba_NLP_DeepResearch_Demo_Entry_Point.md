# Principle: Demo_Entry_Point

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Demo]], [[domain::Application]], [[domain::Entry_Point]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Application entry point pattern for launching demo agents with proper initialization, configuration loading, and service startup.

=== Description ===

Demo Entry Point defines how to structure the main script that launches a demo application:

1. **Configuration loading** - Read API keys, model settings from environment
2. **Component initialization** - Create LLM backends, agents, tools
3. **Service registration** - Register agents with the demo service
4. **Server startup** - Launch the web interface
5. **Graceful shutdown** - Handle cleanup on exit

This pattern ensures demos are easy to run and configure.

=== Usage ===

Use Demo Entry Point when:
- Creating runnable demo scripts
- Need clean separation of config and logic
- Want standardized demo launch experience
- Building production-ready applications

== Theoretical Basis ==

Entry point pattern:

'''Demo Entry Point Pattern:'''
<syntaxhighlight lang="python">
import os
from dotenv import load_dotenv

def init_service():
    """Initialize demo service with agents."""
    # Load environment configuration
    load_dotenv()

    # Configure LLM backend
    llm_config = {
        'model': os.environ.get('MODEL_NAME', 'qwen-max'),
        'api_key': os.environ['DASHSCOPE_API_KEY'],
        'generate_cfg': {
            'temperature': 0.6,
            'max_tokens': 4096
        }
    }

    # Create agent
    agent = SearchAgent(
        llm=llm_config,
        function_list=['search', 'visit'],
        name='WebDancer'
    )

    return agent

def app_gui():
    """Launch the Gradio web interface."""
    agent = init_service()

    # Create and launch UI
    ui = WebUI(agent=agent, chatbot_config={
        'prompt.suggestions': [
            'Search for recent AI news',
            'Find information about climate change'
        ]
    })

    ui.run(
        server_name='0.0.0.0',
        server_port=7860,
        share=False
    )

if __name__ == '__main__':
    app_gui()
</syntaxhighlight>

Design principles:
- **Environment-based config**: No hardcoded secrets
- **Modular initialization**: Each component initialized separately
- **Fail-fast**: Check requirements early
- **Logging**: Provide visibility into startup

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebDancer_Demo]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Demo_Interface]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_LLM_Backend]]
