# Researcher Redesign

This document describes the new design for the `Researcher` module.

## Overview

The researcher performs deep public web research using OpenAI's `web_search` tool. It has three modes:

| Mode | Purpose | Output |
|------|---------|--------|
| `idea` | Conceptual understanding, inspiration | List of `IdeaResult` |
| `implementation` | Working code snippets to solve a problem | List of `ImplementationResult` |
| `freeform` | Comprehensive research report | `ResearchReport` |

## API

```python
from src.knowledge.researcher import Researcher, ResearchFindings

researcher = Researcher()

# Idea mode only - get top 5 ideas
findings = researcher.research("How to implement RAG?", mode="idea", top_k=5)
for idea in findings.ideas:
    print(idea.description, idea.source)

# Implementation mode only - get top 3 code snippets
findings = researcher.research("How to stream OpenAI responses?", mode="implementation", top_k=3)
for impl in findings.implementations:
    print(impl.description)
    print(impl.code_snippet)
    print(impl.dependencies)

# Freeform mode only - get research report
findings = researcher.research("Compare LoRA vs QLoRA for fine-tuning", mode="freeform")
print(findings.report.content)

# Default: all three modes
findings = researcher.research("How to fine-tune LLMs?", top_k=5)
print(findings.ideas)           # From idea mode
print(findings.implementations) # From implementation mode
print(findings.report.content)  # From freeform mode

# Explicit subset of modes
findings = researcher.research("RAG architectures", mode=["idea", "implementation"], top_k=3)
```

## Data Structures

### IdeaResult

Used in `idea` mode. Each result represents a conceptual idea or approach.

```python
@dataclass
class IdeaResult:
    """Single idea result from research."""
    source: str   # URL where this was found
    content: str  # Description of the idea/approach
    
    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "content": self.content}
    
    def to_context_string(self) -> str:
        return f"- {self.content} ({self.source})"
```

### ImplementationResult

Used in `implementation` mode. Each result includes working code.

```python
@dataclass
class ImplementationResult:
    """Single implementation result from research."""
    source: str   # URL where this was found
    content: str  # Freeform: description + code snippet + dependencies
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
        }
    
    def to_context_string(self) -> str:
        return f"**Source:** {self.source}\n\n{self.content}"
```

### ResearchReport

Used in `freeform` mode. Contains a full research report.

```python
@dataclass
class ResearchReport:
    """Freeform research report (academic paper style)."""
    content: str  # Full markdown with all sections
    
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content}
    
    def to_context_string(self) -> str:
        return self.content
```

### ResearchFindings

Wrapper that holds the research output. Supports multiple modes in a single call.

```python
@dataclass
class ResearchFindings:
    """Wrapper for all research outputs."""
    query: str
    modes: List[Literal["idea", "implementation", "freeform"]]  # Can be multiple
    top_k: int
    
    # Populated based on mode(s) - multiple can be filled if multiple modes requested:
    ideas: List[IdeaResult] = field(default_factory=list)
    implementations: List[ImplementationResult] = field(default_factory=list)
    report: Optional[ResearchReport] = None
    
    # For KG ingestion compatibility
    def to_source(self) -> Source.Research:
        """Convert to Source.Research for KnowledgePipeline."""
        ...
```

## Output Format

### Idea and Implementation Modes

The LLM outputs free-form analysis first, then structured results in XML:

```xml
<research_result>
<research_item>
<source>https://...</source>
<content>
Description of the idea/implementation...

For implementation mode, includes code snippet and dependencies:
```python
# code here
```

Dependencies: pip install torch
</content>
</research_item>
<!-- more items... -->
</research_result>
```

### Freeform Mode

The LLM outputs the full report inside the tags:

```xml
<research_result>
## Abstract
...

## Introduction
...

## Literature Review
...

## Methodology
...

## Implementations
...
</research_result>
```

## Prompt Templates

### Envelope (`research_envelope.md`)

Wraps all modes with common instructions.

```markdown
You are a senior engineer-researcher.

Task: Do deep public web research for the following query:

QUERY: {query}

Use the web_search tool. Perform multiple searches and read multiple sources.
Prioritize authoritative sources (official docs, standards, major vendors, reputable blogs, papers).

Mode: {mode}
Top K: {top_k}

{mode_instructions}
```

### Idea Mode (`idea.md`)

```markdown
## Research mindset (idea mode)

You are helping a user who wants conceptual understanding or inspiration.
Search for ideas, approaches, and insights related to their query.

## Source quality rules

Prioritize sources in this order:
1. Official documentation / standards / maintainers
2. Original papers / arXiv + well-known followups
3. Major vendors / reputable labs (OpenAI, Google, Meta, Microsoft, etc.)
4. Well-known educators / engineers with strong track records

Avoid:
- SEO content farms, scraped content, generic posts with no evidence
- Single-source claims that cannot be corroborated

## Task

Return the top {top_k} most relevant and popular ideas/approaches for the query.
Rank them by a combination of:
- Relevance to the query
- Popularity/authority of the source

## Output format (MANDATORY)

First, write free-form analysis and reasoning about what you found.

Then, at the end, output the structured results wrapped exactly like this:

<research_result>
<research_item>
<source>https://exact-url-where-you-found-this</source>
<content>
Clear, concise description of the idea or approach.
</content>
</research_item>
<research_item>
<source>...</source>
<content>...</content>
</research_item>
<!-- Repeat for up to {top_k} items, ranked best first -->
</research_result>

Rules:
- Each <research_item> must have exactly one <source> and one <content>
- <source> must be a real, valid URL (not invented)
- <content> is the description of the idea/approach
- Order items by relevance + popularity (best first)
- Do NOT include code snippets in idea mode
- If you cannot find {top_k} quality results, return fewer
```

### Implementation Mode (`implementation.md`)

```markdown
## Research mindset (implementation mode)

You are helping a user who is stuck implementing something.
They need working code snippets they can apply to their problem.

## Source quality rules

Prioritize sources in this order:
1. Official documentation / maintainer reference implementations
2. Widely adopted OSS repos from reputable orgs
3. Major vendors / reputable labs
4. High-quality tutorials by known experts

Avoid:
- Content farms, low-effort SEO posts
- Tiny repos with unclear ownership, no maintenance
- Copy-pasted snippets with no context

## Task

Return the top {top_k} most relevant and popular implementation approaches.
Each result must include a working code snippet with a sample example.
Rank them by a combination of:
- Relevance to the query
- Popularity/authority of the source
- Code quality and completeness

## Output format (MANDATORY)

First, write free-form analysis and reasoning about what you found.

Then, at the end, output the structured results wrapped exactly like this:

<research_result>
<research_item>
<source>https://exact-url-where-you-found-this</source>
<content>
What this implementation does and when to use it.

```python
# Complete, runnable code example
# Include imports and a minimal working example

def example():
    ...
```

Dependencies: pip install package1 package2 (or "none" if no dependencies)
</content>
</research_item>
<research_item>
<source>...</source>
<content>...</content>
</research_item>
<!-- Repeat for up to {top_k} items, ranked best first -->
</research_result>

Rules:
- Each <research_item> must have <source> and <content>
- <source> must be a real, valid URL (not invented)
- <content> must include: description, code snippet (complete and runnable), and dependencies
- Order items by relevance + popularity + code quality (best first)
- If you cannot find {top_k} quality results, return fewer
```

### Freeform Mode (`freeform.md`)

```markdown
## Research mindset (freeform mode)

You are writing a research report about the query.
Think like an academic researcher producing a comprehensive literature review.

## Source quality rules

Prioritize sources in this order:
1. Original papers / arXiv / peer-reviewed publications
2. Official documentation / standards
3. Major vendors / reputable labs
4. Well-known educators / engineers

Cross-check important claims across multiple sources when possible.

## Task

Write a comprehensive research report structured like an academic paper.
Include inline citations (URLs in parentheses) for all non-trivial claims.

## Output format (MANDATORY)

First, you may include brief notes about your search process.

Then, output the full report wrapped exactly like this:

<research_result>
## Abstract

3-5 sentences summarizing the key findings and conclusions.

## Introduction

- What is the problem/topic?
- Why does it matter?
- What is the scope of this report?

## Literature Review

- What prior work exists?
- Key papers, tools, and approaches
- Include inline URLs for citations: "Smith et al. proposed X (https://...)"

## Methodology

- What approaches/techniques are used to solve this problem?
- Compare different methodologies
- Trade-offs between approaches

## Implementations

- Concrete ways to implement the methodologies
- Key libraries, frameworks, tools
- Configuration considerations
- Include code examples where helpful

</research_result>

Rules:
- The content inside <research_result> is freeform markdown
- Include inline URLs (in parentheses) for all significant claims
- Do NOT invent citations - if you cannot find a source, say so
- Be comprehensive but concise
- Focus on actionable insights
```

## Parsing Logic

```python
import re
from typing import List, Optional

def parse_research_result(raw_output: str, mode: str, query: str, top_k: int) -> ResearchFindings:
    """Parse LLM output into ResearchFindings."""
    
    # Extract content between <research_result>...</research_result>
    match = re.search(r'<research_result>(.*?)</research_result>', raw_output, re.DOTALL)
    if not match:
        # Fallback: return empty findings
        return ResearchFindings(query=query, modes=[mode], top_k=top_k)
    
    content = match.group(1).strip()
    
    if mode == "freeform":
        return ResearchFindings(
            query=query,
            modes=[mode],
            top_k=top_k,
            report=ResearchReport(content=content)
        )
    
    # For idea/implementation, parse <research_item> tags
    items = re.findall(r'<research_item>(.*?)</research_item>', content, re.DOTALL)
    
    results = []
    for item in items:
        source = _extract_tag(item, "source")
        content_text = _extract_tag(item, "content")
        if source and content_text:
            if mode == "idea":
                results.append(IdeaResult(source=source, content=content_text))
            else:  # implementation
                results.append(ImplementationResult(source=source, content=content_text))
    
    if mode == "idea":
        return ResearchFindings(query=query, modes=[mode], top_k=top_k, ideas=results)
    else:
        return ResearchFindings(query=query, modes=[mode], top_k=top_k, implementations=results)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from a single XML tag."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None
```

## Files to Change

### 1. `src/knowledge/researcher/research_findings.py`

- Remove `RepoInfo`, `IdeaInfo` classes
- Add `IdeaResult`, `ImplementationResult`, `ResearchReport` dataclasses
- Update `ResearchFindings` class with new structure
- Remove `parse_repos_from_report`, `parse_ideas_from_report` functions
- Add `parse_research_result` function with XML parsing

### 2. `src/knowledge/researcher/researcher.py`

- Update `ResearchMode` type: `Literal["idea", "implementation", "freeform"]`
- Add `ResearchModeInput` type: `Union[ResearchMode, List[ResearchMode]]`
- Add `top_k` parameter to `research()` method (default: 5)
- Update `research()` to accept list of modes and run each sequentially
- Merge results into single `ResearchFindings` when multiple modes
- Update `_build_research_prompt()` to include `top_k`
- Update return type handling for new `ResearchFindings` structure

### 3. `src/knowledge/researcher/prompts/`

- Delete `both.md` (no longer needed)
- Rewrite `idea.md` with new format
- Rewrite `implementation.md` with new format
- Create `freeform.md` with new format
- Update `research_envelope.md` to include `top_k`

### 4. `src/knowledge/researcher/__init__.py`

- Export new types: `IdeaResult`, `ImplementationResult`, `ResearchReport`

### 5. `src/knowledge/learners/sources.py`

- Update `Source.Research` to handle new modes
- Consider adding `Source.IdeaResult`, `Source.ImplementationResult` if needed for KG ingestion

### 6. `src/knowledge/learners/ingestors/research_ingestor.py`

- Update `_ALLOWED_MODES` to include `freeform`
- Update ingestion logic to handle new output structures

## Migration Notes

- The `both` mode is removed. Users should use `freeform` for comprehensive reports.
- The output format changes from markdown-based parsing to XML-based parsing.
- `top_k` is a new required concept (with default of 5).
- The `depth` parameter remains unchanged (`light` or `deep`).

## Multi-Mode Behavior

When `mode` is a list, the researcher runs each mode sequentially and merges results.
**Default is all three modes** (`["idea", "implementation", "freeform"]`).

```python
# Default: all three modes
findings = researcher.research("query")
# findings.ideas populated from idea mode
# findings.implementations populated from implementation mode
# findings.report populated from freeform mode

# Single mode (string)
findings = researcher.research("query", mode="idea")
# findings.ideas populated, others empty

# Subset of modes (list)
findings = researcher.research("query", mode=["idea", "implementation"])
# findings.ideas and findings.implementations populated
# findings.report is None (freeform not requested)
```

Implementation approach:
1. Default `mode` to `["idea", "implementation", "freeform"]`
2. Normalize `mode` to a list (if string, wrap in list)
3. For each mode in the list, run the research and parse results
4. Merge into a single `ResearchFindings` object
5. `modes` field in `ResearchFindings` tracks which modes were run
