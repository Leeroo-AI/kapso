# File: `WebAgent/WebDancer/demos/gui/html_decorate.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 157 |
| Functions | `get_style_css`, `decorate_writing` |
| Imports | html, markdown_it, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Transforms raw markdown/text output into richly formatted HTML for display in the web interface.

**Mechanism:** The `decorate_writing()` function performs several transformations: (1) processes custom `<qwen:cite>` tags into clickable citation links with superscript numbering, (2) converts `<qwen:takeaway>` tags into styled div elements, (3) renders Mermaid diagram blocks as `<pre class="mermaid">` elements, (4) converts ECharts code blocks into executable chart containers with JavaScript, (5) uses markdown_it to render standard markdown to HTML, (6) wraps everything in a complete HTML document with KaTeX (math), ECharts, and Mermaid libraries. The `get_style_css()` function loads different CSS themes (Default, MBE, Glassmorphism, Apple, Paper). Final output is an iframe with escaped HTML content.

**Significance:** Key UI component that enables rich content rendering including citations, diagrams, charts, and mathematical formulas in WebDancer responses.
