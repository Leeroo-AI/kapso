# Principle: Browser_Interaction

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Playwright Actions|https://playwright.dev/docs/input]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Browser_Automation]], [[domain::Tools]], [[domain::Web_Interaction]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Low-level browser interaction tools (Visit, Click, Fill) that provide atomic actions for browser agents to navigate and interact with web pages.

=== Description ===

Browser Interaction defines the primitive tool interface for browser automation. Each tool provides a single atomic action:

1. **Visit** - Navigate to a URL and return page content
2. **Click** - Click on an element by selector or text
3. **Fill** - Type text into input fields
4. **Scroll** - Scroll the page in a direction
5. **Extract** - Get text/attributes from elements

These tools are registered with the agent's tool system and called via function calling.

=== Usage ===

Use Browser Interaction tools when:
- Implementing browser agent actions
- Building automated testing scripts
- Extracting content from interactive pages

Tool selection:
- **Visit**: Initial page load, following links
- **Click**: Buttons, checkboxes, navigation
- **Fill**: Search boxes, forms, inputs

== Theoretical Basis ==

Browser tools follow a consistent interface:

'''Tool Interface Pattern:'''
<syntaxhighlight lang="python">
from qwen_agent.tools import BaseTool, register_tool

@register_tool('visit')
class Visit(BaseTool):
    """Navigate to URL and return content."""

    description = "Visit a webpage and return its content."
    parameters = [{
        'name': 'url',
        'type': 'string',
        'description': 'The URL to visit',
        'required': True
    }]

    def call(self, params: dict, **kwargs) -> str:
        url = params['url']
        # Use Playwright to navigate
        page = self.browser.goto(url)
        return page.content()

@register_tool('click')
class Click(BaseTool):
    """Click on page element."""

    description = "Click on an element by selector."
    parameters = [{
        'name': 'selector',
        'type': 'string',
        'description': 'CSS selector or text to click',
        'required': True
    }]

    def call(self, params: dict, **kwargs) -> str:
        selector = params['selector']
        self.browser.click(selector)
        return f"Clicked: {selector}"

@register_tool('fill')
class Fill(BaseTool):
    """Fill input field."""

    description = "Type text into an input field."
    parameters = [
        {'name': 'selector', 'type': 'string', 'required': True},
        {'name': 'text', 'type': 'string', 'required': True}
    ]

    def call(self, params: dict, **kwargs) -> str:
        self.browser.fill(params['selector'], params['text'])
        return f"Filled: {params['selector']}"
</syntaxhighlight>

Design principles:
- **Atomic**: Each tool does one thing
- **Idempotent**: Safe to retry on failure
- **Observable**: Return confirmation of action

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_NestBrowse_Browser_Tools]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Browser_Agent]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_MCP_Protocol]]
