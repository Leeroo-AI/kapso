# Principle: Browser_Agent

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|WebArena|https://arxiv.org/abs/2307.13854]]
* [[source::Doc|Playwright Documentation|https://playwright.dev/]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Browser_Automation]], [[domain::Agent_Systems]], [[domain::Web_Interaction]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Autonomous browser-controlling agent that navigates websites through programmatic interactions (clicks, typing, scrolling) guided by LLM reasoning.

=== Description ===

Browser Agent is an agentic paradigm where an LLM directly controls a web browser to accomplish tasks. Unlike API-based web search, browser agents:

1. **Navigate interactively** - Click links, fill forms, scroll pages
2. **Process visual context** - Screenshots, DOM elements, page structure
3. **Maintain browser state** - Cookies, sessions, navigation history
4. **Execute multi-step tasks** - Login, search, extract, submit

The NestBrowse implementation uses async programming for efficient browser control with MCP protocol integration.

=== Usage ===

Use Browser Agent when:
- Tasks require interactive website navigation
- Information is behind login walls or forms
- API access is unavailable
- Tasks involve clicking, typing, or scrolling

Not suitable for:
- Simple information retrieval (use Search tool instead)
- High-volume data extraction (may be rate-limited)
- Sites with anti-automation measures

== Theoretical Basis ==

Browser agents map observations to actions:

<math>
a_t = \pi(o_t, h_t, g)
</math>

Where a_t is action at time t, o_t is observation (screenshot/DOM), h_t is history, and g is goal.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Browser Agent Pattern
async def agentic_loop(browser, goal, max_steps=30):
    """Main browser agent loop."""
    history = []

    for step in range(max_steps):
        # Get current observation
        screenshot = await browser.screenshot()
        dom = await browser.get_accessibility_tree()

        # LLM decides action
        action = await llm.decide_action(
            goal=goal,
            screenshot=screenshot,
            dom=dom,
            history=history
        )

        # Execute action
        if action.type == "click":
            await browser.click(action.selector)
        elif action.type == "type":
            await browser.type(action.selector, action.text)
        elif action.type == "scroll":
            await browser.scroll(action.direction)
        elif action.type == "done":
            return action.result

        history.append(action)

    return "Max steps reached"
</syntaxhighlight>

Key capabilities:
- **Click**: Navigate links, buttons, checkboxes
- **Type**: Fill text fields, search boxes
- **Scroll**: Access below-fold content
- **Extract**: Read page content, tables

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_NestBrowse_Infer_Async]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Browser_Interaction]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_MCP_Protocol]]
