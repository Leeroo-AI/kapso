# File: `libs/text-splitters/langchain_text_splitters/jsx.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 102 |
| Classes | `JSFrameworkTextSplitter` |
| Imports | langchain_text_splitters, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits React (JSX), Vue, and Svelte component code by detecting custom component tags and combining them with JavaScript syntax separators.

**Mechanism:** Extends RecursiveCharacterTextSplitter with dynamic separator detection. Uses regex pattern `<\s*([a-zA-Z0-9]+)[^>]*>` to extract unique opening HTML/component tags from the input text, excluding self-closing tags. Constructs component-specific separators (e.g., "<Component", "<div") and combines them with predefined JavaScript separators (export, function, async function, const, let, var, class, if, for, while, switch, case, default) plus JSX-specific separators ("<>", "&&\n", "||\n"). Passes the combined separator list to parent's split_text() for recursive splitting.

**Significance:** Framework-aware code splitter that understands component boundaries in modern JavaScript frameworks. Essential for RAG systems indexing React/Vue/Svelte codebases where component structure provides semantic meaning. The dynamic tag detection adapts to custom component libraries without hardcoded patterns.
