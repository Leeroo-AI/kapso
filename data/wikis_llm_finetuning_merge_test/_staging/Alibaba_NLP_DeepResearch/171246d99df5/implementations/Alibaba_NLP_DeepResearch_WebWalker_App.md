# Implementation: WebWalker_App

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|Streamlit|https://streamlit.io/]]
|-
! Domains
| [[domain::Demo]], [[domain::Streamlit]], [[domain::Web_Interface]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Streamlit-based demo application for WebWalker with visual navigation, screenshot display, and interactive button clicking.

=== Description ===
The `app.py` module provides an interactive Streamlit demo for WebWalker:

- Two-column layout with thoughts/memory on left, screenshots on right
- URL extraction from HTML with clickable button mapping
- Screenshot capture and display during navigation
- Memory visualization as agent accumulates information
- Configurable max rounds and example queries

The demo includes a `VisitPage` tool that integrates with the Streamlit UI.

=== Usage ===
Run via `streamlit run app.py` to launch the interactive WebWalker demo.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWalker/src/app.py WebAgent/WebWalker/src/app.py]
* '''Lines:''' 1-271

=== Signature ===
<syntaxhighlight lang="python">
# Streamlit app entry
if __name__ == "__main__":
    st.title('WebWalker')
    # ... UI setup

@register_tool('visit_page', allow_overwrite=True)
class VisitPage(BaseTool):
    """Button click navigation tool for Streamlit UI."""

    description = 'Analyzes webpage content and extracts clickable buttons...'
    parameters = [{
        'name': 'button',
        'type': 'string',
        'description': 'the button to click',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        """Execute button click and return page content."""
        ...

def extract_links_with_text(html: str) -> str:
    """Extract clickable links from HTML."""
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as Streamlit app
streamlit run WebAgent/WebWalker/src/app.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs (UI) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Website || str || Starting URL
|-
| Query || str || User question
|-
| Max Rounds || int || Navigation step limit
|}

=== Outputs (UI) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Screenshots || Image || Page captures
|-
| Thoughts || Text || Agent reasoning
|-
| Memory || Text || Accumulated info
|-
| Answer || Text || Final response
|}

== Usage Examples ==

=== Launch Demo ===
<syntaxhighlight lang="bash">
# Set API keys
export DASHSCOPE_API_KEY=your-key
# or
export OPENAI_API_KEY=your-key

# Run Streamlit app
cd WebAgent/WebWalker/src
streamlit run app.py
</syntaxhighlight>

=== Access Demo ===
<syntaxhighlight lang="text">
1. Open http://localhost:8501 in browser
2. Enter website URL (e.g., https://2025.aclweb.org)
3. Enter query (e.g., "What is the paper deadline?")
4. Click "Start!!!!" to begin navigation
5. Watch screenshots and thoughts as agent navigates
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Demo_Interface]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
