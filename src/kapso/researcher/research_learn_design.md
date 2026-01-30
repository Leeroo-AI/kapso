# Research Output & Knowledge Learning Integration

This document describes the design for research output data structures and their integration with the knowledge learning pipeline.

## Overview

Research outputs (`Idea`, `Implementation`, `ResearchReport`) are designed to:
1. Be standalone, self-contained results from web research
2. Provide `to_string()` for use as context in `.evolve()` calls
3. Be acceptable sources for `KnowledgePipeline.run()`

## Data Structures

### Idea

Produced by `idea` mode. Represents a single research idea/insight.

```python
@dataclass
class Idea:
    """
    A single research idea from web research.
    
    Produced by: researcher.research(query, mode="idea")
    Used in: kapso.evolve(context=[idea.to_string()])
    Learnable: pipeline.run(idea)
    """
    query: str      # Original research query
    source: str     # URL where this idea came from
    content: str    # Full content with sections:
                    #   - Description
                    #   - How to Apply
                    #   - When to Use
                    #   - Why Related
                    #   - Trade-offs
                    #   - Examples
                    #   - Prerequisites
                    #   - Related Concepts
    
    def to_string(self) -> str:
        """Format idea as context string for LLM prompts."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...
```

### Implementation

Produced by `implementation` mode. Represents a code snippet/solution.

```python
@dataclass
class Implementation:
    """
    A single implementation from web research.
    
    Produced by: researcher.research(query, mode="implementation")
    Used in: kapso.evolve(context=[impl.to_string()])
    Learnable: pipeline.run(impl)
    """
    query: str      # Original research query
    source: str     # URL where this implementation came from
    content: str    # Full content with sections:
                    #   - Description
                    #   - Why Related
                    #   - When to Use
                    #   - Code Snippet
                    #   - Dependencies
                    #   - Configuration Options
                    #   - Trade-offs
                    #   - Common Pitfalls
                    #   - Performance Notes
    
    def to_string(self) -> str:
        """Format implementation as context string for LLM prompts."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...
```

### ResearchReport

Produced by `study` mode. Represents a comprehensive research report.

```python
@dataclass
class ResearchReport:
    """
    A comprehensive research report (academic paper style).
    
    Produced by: researcher.research(query, mode="study")
    Used in: kapso.evolve(context=[report.to_string()])
    Learnable: pipeline.run(report)
    """
    query: str      # Original research query
    content: str    # Full markdown report with sections:
                    #   - Key Takeaways
                    #   - Abstract
                    #   - Introduction
                    #   - Background
                    #   - Literature Review
                    #   - Methodology Comparison
                    #   - Implementation Guide
                    #   - Evaluation & Benchmarks
                    #   - Limitations
                    #   - Conclusion
                    #   - References
    
    def to_string(self) -> str:
        """Format report as context string for LLM prompts."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...
```

## API Usage

### Research API

```python
from kapso.knowledge.researcher import Researcher

researcher = Researcher()

# Idea mode - returns List[Idea]
ideas: List[Idea] = researcher.research("How to implement RAG?", mode="idea", top_k=5)
for idea in ideas:
    print(idea.query)   # "How to implement RAG?"
    print(idea.source)  # URL
    print(idea.content) # Full content with all sections

# Implementation mode - returns List[Implementation]
impls: List[Implementation] = researcher.research("How to stream OpenAI?", mode="implementation", top_k=3)
for impl in impls:
    print(impl.query)   # "How to stream OpenAI?"
    print(impl.source)  # URL
    print(impl.content) # Full content with code snippet

# Study mode - returns ResearchReport
report: ResearchReport = researcher.research("LLM fine-tuning techniques", mode="study")
print(report.query)     # "LLM fine-tuning techniques"
print(report.content)   # Full academic-style report
```

### Using as Context in Evolve

```python
from kapso.kapso import Kapso

kapso = Kapso()

# Research ideas
ideas = kapso.research("How to optimize transformer inference?", mode="idea", top_k=5)

# Use as context in evolve
result = kapso.evolve(
    repo_path="./my_project",
    goal="Optimize model inference speed",
    context=[idea.to_string() for idea in ideas],  # Pass as context strings
)

# Or use a single implementation
impls = kapso.research("PyTorch model quantization", mode="implementation", top_k=1)
result = kapso.evolve(
    repo_path="./my_project",
    goal="Add INT8 quantization",
    context=[impls[0].to_string()],
)
```

### Learning from Research

Research outputs are valid sources for the knowledge learning pipeline.

```python
from kapso.knowledge.learners import KnowledgePipeline

pipeline = KnowledgePipeline()

# Learn from a single idea
ideas = researcher.research("LoRA fine-tuning", mode="idea", top_k=5)
for idea in ideas:
    result = pipeline.run(idea)
    print(f"Created: {result.created}, Edited: {result.edited}")

# Learn from implementations
impls = researcher.research("FastAPI streaming", mode="implementation", top_k=3)
for impl in impls:
    result = pipeline.run(impl)

# Learn from a research report
report = researcher.research("Vector database comparison", mode="study")
result = pipeline.run(report)
```

## to_string() Format

### Idea.to_string()

```
# Research Idea
Query: {query}
Source: {source}

{content}
```

### Implementation.to_string()

```
# Implementation
Query: {query}
Source: {source}

{content}
```

### ResearchReport.to_string()

```
# Research Report
Query: {query}

{content}
```

## Integration with Source Types

These research outputs integrate with the existing `Source` namespace in `sources.py`:

```python
class Source:
    # Existing sources
    class Repo: ...
    class Solution: ...
    class Research: ...  # Legacy, to be deprecated
    class Idea: ...      # Existing simple idea
    
    # New research output sources (aliases)
    # These are the same classes from researcher module
    # Idea = researcher.Idea
    # Implementation = researcher.Implementation
    # ResearchReport = researcher.ResearchReport
```

The `IngestorFactory` will be updated to handle these new source types:

```python
class IngestorFactory:
    @staticmethod
    def for_source(source, **kwargs) -> Ingestor:
        if isinstance(source, Idea):
            return IdeaIngestor(**kwargs)
        elif isinstance(source, Implementation):
            return ImplementationIngestor(**kwargs)
        elif isinstance(source, ResearchReport):
            return ResearchReportIngestor(**kwargs)
        # ... existing handlers
```

## Ingestor Design

Each research output type has a corresponding ingestor that converts it to WikiPages:

### IdeaIngestor

Extracts:
- Main concept as a WikiPage
- Related concepts as linked pages
- Prerequisites as linked pages

### ImplementationIngestor

Extracts:
- Main implementation as a WikiPage (with code)
- Dependencies as linked pages
- Related patterns as linked pages

### ResearchReportIngestor

Extracts:
- Each major section as a WikiPage
- Cross-references between sections
- References as external links

## Migration Notes

1. The existing `ResearchFindings` wrapper is replaced by direct return types:
   - `mode="idea"` returns `List[Idea]`
   - `mode="implementation"` returns `List[Implementation]`
   - `mode="study"` returns `ResearchReport`

2. The `Source.Research` class in `sources.py` is deprecated in favor of the new types.

3. The `Source.Idea` class in `sources.py` is replaced by the new `Idea` class from researcher.

## File Changes Required

1. **src/knowledge/researcher/research_findings.py**
   - Update `IdeaResult` → `Idea` with `query` attribute
   - Update `ImplementationResult` → `Implementation` with `query` attribute
   - Update `ResearchReport` with `query` attribute
   - Add `to_string()` method to all three classes
   - Remove `ResearchFindings` wrapper (or keep for backward compatibility)

2. **src/knowledge/researcher/researcher.py**
   - Update return types to `List[Idea]`, `List[Implementation]`, `ResearchReport`
   - Pass `query` to each result object

3. **src/knowledge/learners/sources.py**
   - Import and re-export `Idea`, `Implementation`, `ResearchReport`
   - Deprecate `Source.Research` and `Source.Idea`

4. **src/knowledge/learners/ingestors/factory.py**
   - Add handlers for `Idea`, `Implementation`, `ResearchReport`

5. **src/knowledge/learners/ingestors/** (new files)
   - `idea_ingestor.py`
   - `implementation_ingestor.py`
   - `research_report_ingestor.py`
