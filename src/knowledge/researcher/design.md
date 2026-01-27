# Researcher Redesign

This document describes the new design for the `Researcher` module.

## Overview

The researcher performs deep public web research using OpenAI's `web_search` tool. It has three modes:

| Mode | Purpose | Output |
|------|---------|--------|
| `idea` | Conceptual understanding, inspiration | List of `IdeaResult` |
| `implementation` | Working code snippets to solve a problem | List of `ImplementationResult` |
| `study` | Comprehensive research report | `ResearchReport` |

## API

```python
from src.knowledge.researcher import Researcher, ResearchFindings

researcher = Researcher()

# Idea mode only - get top 5 ideas
findings = researcher.research("How to implement RAG?", mode="idea", top_k=5)
for idea in findings.ideas:
    print(idea.source, idea.content)

# Implementation mode only - get top 3 code snippets
findings = researcher.research("How to stream OpenAI responses?", mode="implementation", top_k=3)
for impl in findings.implementations:
    print(impl.source)
    print(impl.content)  # Contains description, code snippet, and dependencies

# Study mode only - get research report
findings = researcher.research("Compare LoRA vs QLoRA for fine-tuning", mode="study")
print(findings.report.content)

# Default: all three modes
findings = researcher.research("How to fine-tune LLMs?", top_k=5)
print(findings.ideas)           # From idea mode
print(findings.implementations) # From implementation mode
print(findings.report.content)  # From study mode

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

Used in `study` mode. Contains a full research report.

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
    modes: List[Literal["idea", "implementation", "study"]]  # Can be multiple
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

### Study Mode

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
Provide comprehensive, actionable information for each idea.

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

For each idea, provide comprehensive information across multiple sections.

## Output format (MANDATORY)

First, write free-form analysis and reasoning about what you found.

Then, at the end, output the structured results wrapped exactly like this:

<research_result>
<research_item>
<source>https://exact-url-where-you-found-this</source>
<content>
## Description
Clear, concise description of the idea or approach. What is it? What problem does it solve?

## How to Apply
Concrete steps to apply this idea:
1. Step one...
2. Step two...
3. Step three...

## When to Use
- Scenario 1: When you need X...
- Scenario 2: When dealing with Y...
- Avoid when: Z...

## Why Related
Explicit explanation of how this idea connects to the user's query. Why is this relevant to what they're trying to achieve?

## Trade-offs
**Pros:**
- Advantage 1
- Advantage 2

**Cons:**
- Limitation 1
- Limitation 2

## Examples
Real-world examples or case studies where this idea has been applied successfully:
- Example 1: Brief description of how company/project X used this...
- Example 2: Brief description of another application...

## Prerequisites
What you need to know or have before applying this:
- Prerequisite 1
- Prerequisite 2

## Related Concepts
Other ideas/techniques that complement this approach:
- Related concept 1
- Related concept 2
</content>
</research_item>
<!-- Repeat for up to {top_k} items, ranked best first -->
</research_result>

Rules:
- Each <research_item> must have exactly one <source> and one <content>
- <source> must be a real, valid URL (not invented)
- <content> must include ALL sections: Description, How to Apply, When to Use, Why Related, Trade-offs, Examples, Prerequisites, Related Concepts
- Order items by relevance + popularity (best first)
- Do NOT include code snippets in idea mode (save those for implementation mode)
- If you cannot find {top_k} quality results, return fewer
- Be comprehensive but concise in each section
```

### Implementation Mode (`implementation.md`)

```markdown
## Research mindset (implementation mode)

You are helping a user who is stuck implementing something.
They need working code snippets they can apply to their problem.
Provide comprehensive, production-ready implementations with full context.

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

For each implementation, provide comprehensive information across multiple sections.

## Output format (MANDATORY)

First, write free-form analysis and reasoning about what you found.

Then, at the end, output the structured results wrapped exactly like this:

<research_result>
<research_item>
<source>https://exact-url-where-you-found-this</source>
<content>
## Description
What this implementation does and what problem it solves.

## Why Related
How this directly addresses the query. Why is this relevant to what the user is trying to achieve?

## When to Use
- Use when: scenario 1...
- Use when: scenario 2...
- Avoid when: scenario where this isn't ideal...

## Code Snippet
```python
# Complete, runnable code example
# Include imports and a minimal working example

def example():
    ...
```

## Dependencies
pip install package1 package2 (or "none" if no dependencies)

## Configuration Options
Key parameters and what they control:
- `param1`: What it does, default value, when to change it
- `param2`: What it does, default value, when to change it

## Trade-offs
**Pros:**
- Advantage 1
- Advantage 2

**Cons:**
- Limitation 1
- Limitation 2

## Common Pitfalls
- Pitfall 1: Description and how to avoid/fix
- Pitfall 2: Description and how to avoid/fix

## Performance Notes
Expected throughput, memory usage, scaling characteristics, benchmarks if available.
</content>
</research_item>
<!-- Repeat for up to {top_k} items, ranked best first -->
</research_result>

Rules:
- Each <research_item> must have <source> and <content>
- <source> must be a real, valid URL (not invented)
- <content> must include ALL sections: Description, Why Related, When to Use, Code Snippet, Dependencies, Configuration Options, Trade-offs, Common Pitfalls, Performance Notes
- Code snippets must be complete and runnable (include imports, show usage)
- Order items by relevance + popularity + code quality (best first)
- If you cannot find {top_k} quality results, return fewer
- Be comprehensive but concise in each section
```

### Study Mode (`study.md`)

```markdown
## Research mindset (study mode)

You are writing a comprehensive research report about the query.
Think like an academic researcher producing a thorough literature review and practical guide.
The goal is to provide knowledge-grounded, actionable insights.

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
Be thorough, analytical, and practical.

## Output format (MANDATORY)

First, you may include brief notes about your search process.

Then, output the full report wrapped exactly like this:

<research_result>
## Key Takeaways
A numbered list of 5-7 most important insights for busy readers:
1. Key insight 1
2. Key insight 2
3. ...

## Abstract
3-5 sentences summarizing the key findings, methodology, and conclusions.

## Introduction
- **Problem Statement**: What is the problem/topic being addressed?
- **Motivation**: Why does this matter? What's the impact?
- **Research Questions**: What specific questions does this report answer?
- **Scope**: What is covered and what is explicitly out of scope?

## Background
- Key concepts and definitions
- Historical context / evolution of the field
- Prerequisites for understanding this topic

## Literature Review
Systematic review of prior work, organized by theme or approach:
- Categories of approaches with descriptions and citations
- Comparison table of key approaches
- Gaps in existing solutions

## Methodology Comparison
Detailed analysis of different approaches:
- How each approach works
- When to use each
- Trade-offs matrix

## Implementation Guide
Practical steps to implement:
- Prerequisites
- Step-by-step guide
- Code examples
- Configuration recommendations
- Best practices

## Evaluation & Benchmarks
- Performance comparisons from literature
- Metrics to consider
- Real-world results

## Limitations
- What this report doesn't cover
- Caveats and assumptions
- Areas where evidence is limited

## Conclusion
- Summary of main findings
- Practical recommendations (numbered list)
- Open questions

## References
Complete list of all cited sources with URLs

</research_result>

Rules:
- The content inside <research_result> is study markdown
- Include inline URLs (in parentheses) for all significant claims
- Do NOT invent citations - if you cannot find a source, say so
- Be comprehensive but concise
- Focus on actionable, practical insights
- Use tables for comparisons where appropriate
- Include code examples where they add value
- All sections are required
```

## Parsing Logic

```python
import re
from typing import List, Optional

def parse_research_result(raw_output: str, mode: str, query: str, top_k: int) -> ResearchFindings:
    """Parse LLM output into ResearchFindings."""
    
    # Extract content between <research_result>...</research_result>
    # First try with closing tag
    match = re.search(r'<research_result>(.*?)</research_result>', raw_output, re.DOTALL)
    
    # If no closing tag found, try to extract everything after opening tag
    # (handles case where output was truncated)
    if not match:
        match = re.search(r'<research_result>(.*)', raw_output, re.DOTALL)
        if match:
            logger.warning("Missing </research_result> closing tag; output may have been truncated")
    
    if not match:
        # Fallback: return empty findings
        return ResearchFindings(query=query, modes=[mode], top_k=top_k)
    
    content = match.group(1).strip()
    
    if mode == "study":
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

- Update `ResearchMode` type: `Literal["idea", "implementation", "study"]`
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
- Create `study.md` with new format
- Update `research_envelope.md` to include `top_k`

### 4. `src/knowledge/researcher/__init__.py`

- Export new types: `IdeaResult`, `ImplementationResult`, `ResearchReport`

### 5. `src/knowledge/learners/sources.py`

- Update `Source.Research` to handle new modes
- Consider adding `Source.IdeaResult`, `Source.ImplementationResult` if needed for KG ingestion

### 6. `src/knowledge/learners/ingestors/research_ingestor.py`

- Update `_ALLOWED_MODES` to include `study`
- Update ingestion logic to handle new output structures

## Migration Notes

- The `both` mode is removed. Users should use `study` for comprehensive reports.
- The output format changes from markdown-based parsing to XML-based parsing.
- `top_k` is a new required concept (with default of 5).
- The `depth` parameter remains unchanged (`light` or `deep`).

## Configuration

- **Model**: `gpt-5.2` (default)
- **Max output tokens**: 32000
- **Reasoning effort**: Only applied for o1/o3 models

## Multi-Mode Behavior

When `mode` is a list, the researcher runs each mode sequentially and merges results.
**Default is all three modes** (`["idea", "implementation", "study"]`).

```python
# Default: all three modes
findings = researcher.research("query")
# findings.ideas populated from idea mode
# findings.implementations populated from implementation mode
# findings.report populated from study mode

# Single mode (string)
findings = researcher.research("query", mode="idea")
# findings.ideas populated, others empty

# Subset of modes (list)
findings = researcher.research("query", mode=["idea", "implementation"])
# findings.ideas and findings.implementations populated
# findings.report is None (study not requested)
```

Implementation approach:
1. Default `mode` to `["idea", "implementation", "study"]`
2. Normalize `mode` to a list (if string, wrap in list)
3. For each mode in the list, run the research and parse results
4. Merge into a single `ResearchFindings` object
5. `modes` field in `ResearchFindings` tracks which modes were run
