# Principle: Demo_Interface

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Gradio Documentation|https://www.gradio.app/docs]]
* [[source::Doc|Streamlit Documentation|https://docs.streamlit.io/]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Web_Interface]], [[domain::Demo]], [[domain::User_Experience]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Web-based interactive demonstration interfaces for showcasing agent capabilities with real-time response streaming, conversation management, and visual feedback.

=== Description ===

Demo Interface is the principle of building user-facing web applications that allow non-technical users to interact with research agents. Effective demo interfaces:

1. **Provide intuitive input** - Text boxes for queries, dropdowns for configuration
2. **Show agent reasoning** - Display thinking process, tool calls, and intermediate results
3. **Support streaming** - Real-time token-by-token response rendering
4. **Manage conversations** - History, reset, and context persistence
5. **Handle errors gracefully** - Timeout messages, retry options

The DeepResearch repository implements demos using both Gradio (for WebDancer) and Streamlit (for WebWalker).

=== Usage ===

Create Demo Interfaces when:
- Showcasing agent capabilities to stakeholders
- Running user studies or evaluations
- Building production-ready chat applications
- Comparing multiple agent configurations

== Theoretical Basis ==

Demo interface design follows human-computer interaction principles:

'''Key Components:'''
<syntaxhighlight lang="python">
# Demo Interface Pattern
class DemoInterface:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.history = []

    def render_input(self):
        """Create input widgets (query box, dropdowns)."""
        ...

    def render_output(self, response):
        """Display agent response with formatting."""
        ...

    def stream_response(self, query):
        """Yield tokens as they're generated."""
        for token in self.agent.generate_stream(query):
            yield token

    def reset_conversation(self):
        """Clear history and state."""
        self.history = []
</syntaxhighlight>

Design principles:
- **Responsiveness**: Show progress indicators during processing
- **Transparency**: Display intermediate reasoning steps
- **Accessibility**: Support keyboard navigation and screen readers
- **Robustness**: Handle edge cases (empty input, long responses)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebUI]]
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebWalker_App]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Demo_Entry_Point]]
