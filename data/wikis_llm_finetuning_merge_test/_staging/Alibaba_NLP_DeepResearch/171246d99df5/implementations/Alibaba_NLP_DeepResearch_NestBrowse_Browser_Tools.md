# Implementation: NestBrowse_Browser_Tools

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Browser_Automation]], [[domain::Web_Scraping]], [[domain::Tool_Use]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Browser interaction tool classes (Visit, Click, Fill) for NestBrowse agent that enable webpage navigation, element clicking, and form input through MCP.

=== Description ===
The `browser.py` module defines three core tool classes for browser automation:

- `Visit`: Navigate to a URL and extract page content
- `Click`: Click on page elements by selector or text
- `Fill`: Fill form inputs with provided values

These tools integrate with the MCP (Model Context Protocol) browser server to execute browser actions in a controlled environment. Each tool implements the qwen_agent `BaseTool` interface for seamless agent integration.

=== Usage ===
Register these tools when creating a NestBrowse agent. They are called automatically by the agent during web browsing tasks.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/NestBrowse/toolkit/browser.py WebAgent/NestBrowse/toolkit/browser.py]
* '''Lines:''' 1-192

=== Signature ===
<syntaxhighlight lang="python">
@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    """Navigate to URL and extract content."""
    name = 'visit'
    description = 'Visit a webpage and return its content'
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to visit"}
        },
        "required": ["url"]
    }

    def call(self, params: str, **kwargs) -> str:
        ...

@register_tool('click', allow_overwrite=True)
class Click(BaseTool):
    """Click on page elements."""
    name = 'click'
    description = 'Click on an element on the page'
    parameters = {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector or text"}
        },
        "required": ["selector"]
    }

    def call(self, params: str, **kwargs) -> str:
        ...

@register_tool('fill', allow_overwrite=True)
class Fill(BaseTool):
    """Fill form inputs."""
    name = 'fill'
    description = 'Fill a form field with a value'
    parameters = {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "Input selector"},
            "value": {"type": "string", "description": "Value to fill"}
        },
        "required": ["selector", "value"]
    }

    def call(self, params: str, **kwargs) -> str:
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.browser import Visit, Click, Fill
</syntaxhighlight>

== I/O Contract ==

=== Visit Tool ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| url || str || URL to navigate to
|}

{| class="wikitable"
|-
! Output !! Type !! Description
|-
| content || str || Page content in markdown format
|}

=== Click Tool ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| selector || str || CSS selector or element text
|}

=== Fill Tool ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| selector || str || Form field selector
|-
| value || str || Value to enter
|}

== Usage Examples ==

=== Manual Tool Usage ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.browser import Visit, Click, Fill

# Visit a webpage
visit_tool = Visit()
content = visit_tool.call({"url": "https://example.com"})
print(content)

# Click a button
click_tool = Click()
result = click_tool.call({"selector": "button.submit"})

# Fill a form
fill_tool = Fill()
result = fill_tool.call({
    "selector": "input[name='email']",
    "value": "user@example.com"
})
</syntaxhighlight>

=== Agent Registration ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.browser import Visit, Click, Fill

# Tools are auto-registered via @register_tool
# Use in agent function_list:
agent = SomeAgent(
    function_list=['visit', 'click', 'fill'],
    llm=llm_config
)
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Browser_Interaction]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
