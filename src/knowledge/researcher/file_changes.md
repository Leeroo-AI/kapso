# Research Output Integration - File Changes

This document details the specific file changes required to implement the design in `research_learn_design.md`.

## Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `src/knowledge/researcher/research_findings.py` | Modify | Rename classes, add `query` attr, add `to_string()`, remove `ResearchFindings` |
| `src/knowledge/researcher/researcher.py` | Modify | Update return types to `List[Idea]`, `List[Implementation]`, `ResearchReport` |
| `src/knowledge/researcher/__init__.py` | Modify | Update exports |
| `src/knowledge/learners/sources.py` | Modify | Remove `Source.Research`, `Source.Idea`, `IdeaList` |
| `src/knowledge/learners/ingestors/factory.py` | No change | Uses class name (already works) |
| `src/knowledge/learners/ingestors/idea_ingestor.py` | Create | New ingestor for `Idea` |
| `src/knowledge/learners/ingestors/implementation_ingestor.py` | Create | New ingestor for `Implementation` |
| `src/knowledge/learners/ingestors/research_report_ingestor.py` | Create | New ingestor for `ResearchReport` |
| `src/knowledge/learners/ingestors/research_ingestor.py` | Delete | No longer needed |
| `src/knowledge/learners/ingestors/__init__.py` | Modify | Export new ingestors, remove old |
| `tests/test_researcher_modes.py` | Modify | Update tests for new return types |
| `README.md` | Modify | Update Basic Usage example |
| `docs/quickstart.mdx` | Modify | Update Web Research section |
| `docs/research/overview.mdx` | Modify | Major rewrite for new API |
| `docs/reference/kapso-api.mdx` | Modify | Update research() and learn() signatures |

---

## 1. src/knowledge/researcher/research_findings.py

### Changes Required

**1.1 Rename `IdeaResult` → `Idea`**

```python
# BEFORE
@dataclass
class IdeaResult:
    source: str
    content: str

# AFTER
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
    content: str    # Full content with sections
    
    def to_string(self) -> str:
        """Format idea as context string for LLM prompts."""
        return f"# Research Idea\nQuery: {self.query}\nSource: {self.source}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "source": self.source, "content": self.content}
    
    def __str__(self) -> str:
        return self.to_string()
```

**1.2 Rename `ImplementationResult` → `Implementation`**

```python
# BEFORE
@dataclass
class ImplementationResult:
    source: str
    content: str

# AFTER
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
    content: str    # Full content with code snippet
    
    def to_string(self) -> str:
        """Format implementation as context string for LLM prompts."""
        return f"# Implementation\nQuery: {self.query}\nSource: {self.source}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "source": self.source, "content": self.content}
    
    def __str__(self) -> str:
        return self.to_string()
```

**1.3 Update `ResearchReport`**

```python
# BEFORE
@dataclass
class ResearchReport:
    content: str

# AFTER
@dataclass
class ResearchReport:
    """
    A comprehensive research report (academic paper style).
    
    Produced by: researcher.research(query, mode="study")
    Used in: kapso.evolve(context=[report.to_string()])
    Learnable: pipeline.run(report)
    """
    query: str      # Original research query
    content: str    # Full markdown report
    
    def to_string(self) -> str:
        """Format report as context string for LLM prompts."""
        return f"# Research Report\nQuery: {self.query}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "content": self.content}
    
    def __str__(self) -> str:
        return self.to_string()
```

**1.4 Remove `ResearchFindings` wrapper class**

The `ResearchFindings` class is no longer needed. The `research()` method will return:
- `List[Idea]` for idea mode
- `List[Implementation]` for implementation mode
- `ResearchReport` for study mode

**1.5 Update `parse_research_result` function**

Change return type and pass `query` to all result objects:

```python
def parse_idea_results(raw_output: str, query: str) -> List[Idea]:
    """Parse LLM output into List[Idea]."""
    ...
    return [Idea(query=query, source=source, content=content_text) for ...]

def parse_implementation_results(raw_output: str, query: str) -> List[Implementation]:
    """Parse LLM output into List[Implementation]."""
    ...
    return [Implementation(query=query, source=source, content=content_text) for ...]

def parse_study_result(raw_output: str, query: str) -> ResearchReport:
    """Parse LLM output into ResearchReport."""
    ...
    return ResearchReport(query=query, content=content)
```

**1.6 Remove `merge_findings` function**

No longer needed since we return direct types.

---

## 2. src/knowledge/researcher/researcher.py

### Changes Required

**2.1 Update imports**

```python
from src.knowledge.researcher.research_findings import (
    Idea,
    Implementation,
    ResearchReport,
    parse_idea_results,
    parse_implementation_results,
    parse_study_result,
)
```

**2.2 Update type definitions**

```python
from typing import List, Literal, Union, overload

ResearchMode = Literal["idea", "implementation", "study"]
ResearchDepth = Literal["light", "deep"]

# Return type union
ResearchResult = Union[List[Idea], List[Implementation], ResearchReport]
```

**2.3 Update `research()` method signature and return type**

```python
def research(
    self,
    query: str,
    *,
    mode: ResearchMode,  # Now required, no default
    top_k: int = 5,
    depth: ResearchDepth = "deep",
) -> ResearchResult:
    """
    Run deep web research.
    
    Args:
        query: What we want to learn from public sources.
        mode: Research mode (required):
            - "idea": Returns List[Idea]
            - "implementation": Returns List[Implementation]
            - "study": Returns ResearchReport
        top_k: Maximum number of results (for idea/implementation modes).
        depth: Research depth ("light" or "deep").
    
    Returns:
        - List[Idea] if mode="idea"
        - List[Implementation] if mode="implementation"
        - ResearchReport if mode="study"
    """
    ...
```

**2.4 Update `_run_single_mode` to return correct types**

```python
def _run_single_mode(
    self,
    query: str,
    mode: ResearchMode,
    top_k: int,
    reasoning_effort: str,
) -> ResearchResult:
    """Run research for a single mode."""
    ...
    
    # Parse based on mode
    if mode == "idea":
        return parse_idea_results(raw_text, query)
    elif mode == "implementation":
        return parse_implementation_results(raw_text, query)
    else:  # study
        return parse_study_result(raw_text, query)
```

**2.5 Remove multi-mode support**

Since each mode returns a different type, remove the ability to pass a list of modes.
Users should call `research()` multiple times if they need multiple modes.

---

## 3. src/knowledge/researcher/__init__.py

### Changes Required

Update exports (remove old types):

```python
from src.knowledge.researcher.researcher import (
    Researcher,
    ResearchMode,
    ResearchDepth,
    ResearchResult,
)
from src.knowledge.researcher.research_findings import (
    Idea,
    Implementation,
    ResearchReport,
)

__all__ = [
    "Researcher",
    "ResearchMode",
    "ResearchDepth",
    "ResearchResult",
    "Idea",
    "Implementation",
    "ResearchReport",
]
```

---

## 4. src/knowledge/learners/sources.py

### Changes Required

**4.1 Remove `Source.Research` class**

Delete the entire `Source.Research` class (lines 99-131).

**4.2 Remove `Source.Idea` class**

Delete the entire `Source.Idea` class (lines 73-97).

**4.3 Remove `IdeaList` class**

Delete the entire `IdeaList` class (lines 134-163).

**4.4 Update header comment**

Remove references to `Source.Idea` and `Source.Research` in the usage examples.

**4.5 Final `sources.py` structure**

```python
class Source:
    """Namespace for knowledge source types."""
    
    @dataclass
    class Repo:
        """Source from a Git repository."""
        url: str
        branch: str = "main"
        ...
    
    @dataclass
    class Solution:
        """Source from a completed Solution (experiment logs)."""
        obj: "SolutionResult"
        ...
```

Note: `Idea`, `Implementation`, `ResearchReport` from `src.knowledge.researcher` are used directly, not wrapped in `Source`.

---

## 5. src/knowledge/learners/ingestors/factory.py

### Changes Required

No changes needed. The factory uses `source.__class__.__name__.lower()` to determine ingestor type.

New ingestors will be registered as:
- `"idea"` → `IdeaIngestor`
- `"implementation"` → `ImplementationIngestor`
- `"researchreport"` → `ResearchReportIngestor`

---

## 6. src/knowledge/learners/ingestors/idea_ingestor.py (NEW FILE)

```python
# Idea Ingestor
#
# Converts `Idea` (from researcher) into `WikiPage` objects.

import logging
import re
from datetime import datetime, timezone
from typing import Any, List

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug for WikiPage IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_")
    return (cleaned or "idea")[:max_len]


@register_ingestor("idea")
class IdeaIngestor(Ingestor):
    """
    Ingest researcher.Idea into Principle WikiPage.
    
    Extracts:
    - Main concept as a WikiPage (type: Principle)
    """

    @property
    def source_type(self) -> str:
        return "idea"

    def ingest(self, source: Any) -> List[WikiPage]:
        # Extract attributes
        if isinstance(source, dict):
            query = source.get("query", "")
            src_url = source.get("source", "")
            content = source.get("content", "")
        else:
            query = getattr(source, "query", "")
            src_url = getattr(source, "source", "")
            content = getattr(source, "content", "")

        if not query:
            raise ValueError("IdeaIngestor expected a non-empty 'query'")

        slug = _slugify(query)
        now = datetime.now(timezone.utc).isoformat()

        page_id = f"Principle/Research_Idea_{slug}"
        page_title = f"Research_Idea_{slug}"
        overview = f"Research idea for: {query}"
        
        page_content = (
            "== Overview ==\n"
            f"Research idea extracted from web research.\n\n"
            "== Query ==\n"
            f"{query}\n\n"
            "== Source ==\n"
            f"{src_url}\n\n"
            "== Content ==\n"
            f"{content}\n\n"
            "== Metadata ==\n"
            f"* Generated at (UTC): {now}\n"
        )

        return [WikiPage(
            id=page_id,
            page_title=page_title,
            page_type="Principle",
            overview=overview,
            content=page_content,
            domains=["Research", "Idea"],
            sources=[src_url] if src_url else [],
            outgoing_links=[],
        )]
```

---

## 7. src/knowledge/learners/ingestors/implementation_ingestor.py (NEW FILE)

```python
# Implementation Ingestor
#
# Converts `Implementation` (from researcher) into `WikiPage` objects.

import logging
import re
from datetime import datetime, timezone
from typing import Any, List

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug for WikiPage IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_")
    return (cleaned or "impl")[:max_len]


@register_ingestor("implementation")
class ImplementationIngestor(Ingestor):
    """
    Ingest researcher.Implementation into Implementation WikiPage.
    
    Extracts:
    - Main implementation as a WikiPage (type: Implementation)
    """

    @property
    def source_type(self) -> str:
        return "implementation"

    def ingest(self, source: Any) -> List[WikiPage]:
        # Extract attributes
        if isinstance(source, dict):
            query = source.get("query", "")
            src_url = source.get("source", "")
            content = source.get("content", "")
        else:
            query = getattr(source, "query", "")
            src_url = getattr(source, "source", "")
            content = getattr(source, "content", "")

        if not query:
            raise ValueError("ImplementationIngestor expected a non-empty 'query'")

        slug = _slugify(query)
        now = datetime.now(timezone.utc).isoformat()

        page_id = f"Implementation/Research_Impl_{slug}"
        page_title = f"Research_Impl_{slug}"
        overview = f"Implementation for: {query}"
        
        page_content = (
            "== Overview ==\n"
            f"Implementation extracted from web research.\n\n"
            "== Query ==\n"
            f"{query}\n\n"
            "== Source ==\n"
            f"{src_url}\n\n"
            "== Content ==\n"
            f"{content}\n\n"
            "== Metadata ==\n"
            f"* Generated at (UTC): {now}\n"
        )

        return [WikiPage(
            id=page_id,
            page_title=page_title,
            page_type="Implementation",
            overview=overview,
            content=page_content,
            domains=["Research", "Implementation"],
            sources=[src_url] if src_url else [],
            outgoing_links=[],
        )]
```

---

## 8. src/knowledge/learners/ingestors/research_report_ingestor.py (NEW FILE)

```python
# Research Report Ingestor
#
# Converts `ResearchReport` (from researcher) into `WikiPage` objects.

import logging
import re
from datetime import datetime, timezone
from typing import Any, List

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug for WikiPage IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_")
    return (cleaned or "report")[:max_len]


@register_ingestor("researchreport")
class ResearchReportIngestor(Ingestor):
    """
    Ingest researcher.ResearchReport into Principle WikiPage.
    
    Extracts:
    - Full report as a WikiPage (type: Principle)
    """

    @property
    def source_type(self) -> str:
        return "researchreport"

    def ingest(self, source: Any) -> List[WikiPage]:
        # Extract attributes
        if isinstance(source, dict):
            query = source.get("query", "")
            content = source.get("content", "")
        else:
            query = getattr(source, "query", "")
            content = getattr(source, "content", "")

        if not query:
            raise ValueError("ResearchReportIngestor expected a non-empty 'query'")

        slug = _slugify(query)
        now = datetime.now(timezone.utc).isoformat()

        page_id = f"Principle/Research_Report_{slug}"
        page_title = f"Research_Report_{slug}"
        overview = f"Research report for: {query}"
        
        page_content = (
            "== Overview ==\n"
            f"Comprehensive research report from web research.\n\n"
            "== Query ==\n"
            f"{query}\n\n"
            "== Report ==\n"
            f"{content}\n\n"
            "== Metadata ==\n"
            f"* Generated at (UTC): {now}\n"
        )

        return [WikiPage(
            id=page_id,
            page_title=page_title,
            page_type="Principle",
            overview=overview,
            content=page_content,
            domains=["Research", "Report"],
            sources=[],
            outgoing_links=[],
        )]
```

---

## 9. src/knowledge/learners/ingestors/__init__.py

### Changes Required

Remove old ingestor, add new ones:

```python
# Existing imports
from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import IngestorFactory, register_ingestor

# Import ingestors to trigger registration
from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor
from src.knowledge.learners.ingestors.experiment_ingestor import ExperimentIngestor

# NEW: Import new ingestors (replaces research_ingestor)
from src.knowledge.learners.ingestors.idea_ingestor import IdeaIngestor
from src.knowledge.learners.ingestors.implementation_ingestor import ImplementationIngestor
from src.knowledge.learners.ingestors.research_report_ingestor import ResearchReportIngestor

__all__ = [
    "Ingestor",
    "IngestorFactory",
    "register_ingestor",
    "RepoIngestor",
    "ExperimentIngestor",
    # NEW
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
```

---

## 10. src/knowledge/learners/ingestors/research_ingestor.py

### Changes Required

**Delete this file.** It is replaced by the three new ingestors.

---

## 11. README.md

### Changes Required

Update the Basic Usage example (lines 94-137):

```python
from src.kapso import Kapso, Source, DeployStrategy

# Initialize Kapso
kapso = Kapso(kg_index="data/indexes/legal_contracts.index")

# Research: Gather domain-specific techniques from the web
# Each mode returns a different type:
# - mode="idea" returns List[Idea]
# - mode="implementation" returns List[Implementation]
# - mode="study" returns ResearchReport

ideas = kapso.research(
    "RLHF and DPO fine-tuning for legal contract analysis",
    mode="idea",
    top_k=10,
    depth="deep",
)

implementations = kapso.research(
    "RLHF and DPO fine-tuning for legal contract analysis",
    mode="implementation",
    top_k=5,
    depth="deep",
)

# Learn: Ingest knowledge from repositories and research into the KG
kapso.learn(
    sources=[
        Source.Repo("https://github.com/huggingface/trl"),
        *ideas,           # List[Idea] - each idea is a valid source
        *implementations, # List[Implementation] - each impl is a valid source
    ],
    wiki_dir="data/wikis",
)

# Evolve: Build a solution through experimentation
# Use research results as context via to_string()
solution = kapso.evolve(
    goal="Fine-tune Llama-3.1-8B for legal clause risk classification, target F1 > 0.85",
    data_dir="./data/cuad_dataset",
    output_path="./models/legal_risk_v1",
    context=[idea.to_string() for idea in ideas],
)

# Deploy: Turn solution into running deployed_program
deployed_program = kapso.deploy(solution, strategy=DeployStrategy.MODAL)
deployed_program.stop()
```

---

## 12. docs/quickstart.mdx

### Changes Required

**12.1 Update "Web Research (Optional)" section (lines 156-178)**

```python
from src.kapso import Kapso

kapso = Kapso()

# Research implementations
implementations = kapso.research(
    "unsloth FastLanguageModel example",
    mode="implementation",  # Returns List[Implementation]
    top_k=5,
    depth="deep",
)

# Use research as context for evolving
solution = kapso.evolve(
    goal="Fine-tune a model with Unsloth + LoRA",
    context=[impl.to_string() for impl in implementations],
    output_path="./models/unsloth_v1",
)
```

---

## 13. docs/research/overview.mdx

### Changes Required

This file needs significant updates to reflect the new API.

**13.1 Update Overview section**

Replace:
```
4. **Return** a `Source.Research` object that can be ingested into the Knowledge Graph
```

With:
```
4. **Return** structured results:
   - `List[Idea]` for idea mode
   - `List[Implementation]` for implementation mode
   - `ResearchReport` for study mode
```

**13.2 Update Research Modes table**

| Mode | Description | Returns | Best For |
|------|-------------|---------|----------|
| `idea` | Conceptual understanding, principles, trade-offs | `List[Idea]` | Learning new domains, understanding best practices |
| `implementation` | Code examples, APIs, libraries, configuration | `List[Implementation]` | Building features, finding working code |
| `study` | Comprehensive research report (academic style) | `ResearchReport` | Deep understanding, literature review |

**13.3 Update Basic Usage examples**

```python
from src.kapso import Kapso

kapso = Kapso()

# Research ideas - returns List[Idea]
ideas = kapso.research(
    "QLoRA fine-tuning best practices for LLaMA models",
    mode="idea",
    top_k=10,
    depth="deep",
)

for idea in ideas:
    print(idea.source)   # URL
    print(idea.content)  # Full content
    print(idea.to_string())  # Formatted for LLM context
```

**13.4 Update Research → Evolve example**

```python
from src.kapso import Kapso

kapso = Kapso()

# Research implementation details
implementations = kapso.research(
    "unsloth FastLanguageModel example",
    mode="implementation",
    top_k=5,
    depth="deep",
)

# Use research as context for evolve
solution = kapso.evolve(
    goal="Fine-tune a model with Unsloth + LoRA",
    context=[impl.to_string() for impl in implementations],
    output_path="./models/unsloth_v1",
)
```

**13.5 Update Research → Learn example**

```python
from src.kapso import Kapso

kapso = Kapso()

# Research a topic - returns List[Idea]
ideas = kapso.research(
    "LoRA rank selection best practices",
    mode="idea",
    top_k=10,
    depth="deep",
)

# Ingest each idea into Knowledge Graph
for idea in ideas:
    kapso.learn(idea, wiki_dir="data/wikis")

# Or learn from a study report
report = kapso.research(
    "LoRA rank selection best practices",
    mode="study",
    depth="deep",
)
kapso.learn(report, wiki_dir="data/wikis")
```

**13.6 Update API Reference section**

```python
def research(
    self,
    query: str,
    *,
    mode: Literal["idea", "implementation", "study"],  # Required
    top_k: int = 5,  # For idea/implementation modes
    depth: Literal["light", "deep"] = "deep",
) -> Union[List[Idea], List[Implementation], ResearchReport]:
    """
    Perform deep public web research.
    
    Args:
        query: What to research (be specific)
        mode: Research mode (required):
            - "idea": Returns List[Idea]
            - "implementation": Returns List[Implementation]
            - "study": Returns ResearchReport
        top_k: Max results for idea/implementation modes
        depth: How thorough - light (faster) or deep (more comprehensive)
    
    Returns:
        - List[Idea] if mode="idea"
        - List[Implementation] if mode="implementation"
        - ResearchReport if mode="study"
    """
```

**13.7 Update data class documentation**

```python
@dataclass
class Idea:
    query: str    # Original research query
    source: str   # URL where this idea came from
    content: str  # Full content with sections
    
    def to_string(self) -> str:
        """Format for use as context in evolve()"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""

@dataclass
class Implementation:
    query: str    # Original research query
    source: str   # URL where this implementation came from
    content: str  # Full content with code snippet
    
    def to_string(self) -> str:
        """Format for use as context in evolve()"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""

@dataclass
class ResearchReport:
    query: str    # Original research query
    content: str  # Full markdown report
    
    def to_string(self) -> str:
        """Format for use as context in evolve()"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
```

---

## 14. docs/reference/kapso-api.mdx

### Changes Required

**14.1 Update research() method signature (lines 49-76)**

```python
def research(
    self,
    query: str,
    *,
    mode: Literal["idea", "implementation", "study"],  # Required
    top_k: int = 5,
    depth: str = "deep",
) -> Union[List[Idea], List[Implementation], ResearchReport]
```

**14.2 Update Parameters table**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | - | What to research |
| `mode` | `str` | - | `"idea"`, `"implementation"`, or `"study"` (required) |
| `top_k` | `int` | `5` | Max results for idea/implementation modes |
| `depth` | `str` | `"deep"` | `"light"` or `"deep"` |

**14.3 Update Returns section**

Returns based on mode:
- `List[Idea]` if mode="idea"
- `List[Implementation]` if mode="implementation"
- `ResearchReport` if mode="study"

**14.4 Update Example**

```python
# Research ideas
ideas = kapso.research(
    "QLoRA best practices for fine-tuning",
    mode="idea",
    top_k=10,
    depth="deep",
)

# Use as context
solution = kapso.evolve(
    goal="Fine-tune LLaMA",
    context=[idea.to_string() for idea in ideas],
)

# Or learn into KG
for idea in ideas:
    kapso.learn(idea, wiki_dir="data/wikis")
```

**14.5 Update learn() method signature (lines 96-137)**

Remove `Source.Research` from the type hint:

```python
def learn(
    self,
    *sources: Union[Source.Repo, Source.Solution, Idea, Implementation, ResearchReport],
    wiki_dir: str = "data/wikis",
    skip_merge: bool = False,
    kg_index: Optional[str] = None,
) -> PipelineResult
```

**14.6 Update Source Types section (lines 338-351)**

Remove `Source.Research`:

```python
from src.kapso import Source
from src.knowledge.researcher import Idea, Implementation, ResearchReport

# Repository
Source.Repo(url: str, branch: str = "main")

# Solution (from evolve)
Source.Solution(solution: SolutionResult)

# Research outputs (from research)
Idea       # Returned by kapso.research(mode="idea")
Implementation  # Returned by kapso.research(mode="implementation")
ResearchReport  # Returned by kapso.research(mode="study")
```

**14.7 Update Complete Example (lines 381-419)**

```python
from src.kapso import Kapso, Source, DeployStrategy

# Initialize with KG
kapso = Kapso(kg_index="data/indexes/ml.index")

# Research
ideas = kapso.research("XGBoost best practices", mode="idea", top_k=10)

# Learn from repo and research
kapso.learn(
    Source.Repo("https://github.com/dmlc/xgboost"),
    *ideas,  # Each idea is a valid source
)

# Evolve with research context
solution = kapso.evolve(
    goal="XGBoost classifier with AUC > 0.9",
    context=[idea.to_string() for idea in ideas],
    max_iterations=10,
)

# Check if goal was achieved
if solution.succeeded:
    print(f"Goal achieved with score: {solution.final_score}")

# Deploy
deployed_program = kapso.deploy(solution, strategy=DeployStrategy.LOCAL)
result = deployed_program.run({"data_path": "./test.csv"})
deployed_program.stop()

# Learn from experience
kapso.learn(Source.Solution(solution))
```

---

## Implementation Order (Updated)

1. **research_findings.py** - Rename classes, add `query`, add `to_string()`, remove `ResearchFindings`
2. **researcher.py** - Update return types, remove multi-mode support
3. **__init__.py** (researcher) - Update exports
4. **Delete research_ingestor.py**
5. **Create new ingestor files** - idea, implementation, research_report
6. **ingestors/__init__.py** - Update imports
7. **sources.py** - Remove `Source.Research`, `Source.Idea`, `IdeaList`
8. **tests** - Update test file
9. **README.md** - Update Basic Usage example
10. **docs/quickstart.mdx** - Update Web Research section
11. **docs/research/overview.mdx** - Major rewrite for new API
12. **docs/reference/kapso-api.mdx** - Update research() and learn() signatures
