# Research Ingestor Implementation Tasks

Based on `design.md`, this document outlines the specific changes needed to implement the agentic research ingestors.

---

## Task 0: Move Existing Skeleton Files

**Action:** Move the existing skeleton ingestor files into the `research_ingestor/` package as starting points.

### 0.1 Files to Move

| From | To |
|------|-----|
| `src/knowledge/learners/ingestors/idea_ingestor.py` | `src/knowledge/learners/ingestors/research_ingestor/idea_ingestor.py` |
| `src/knowledge/learners/ingestors/implementation_ingestor.py` | `src/knowledge/learners/ingestors/research_ingestor/implementation_ingestor.py` |
| `src/knowledge/learners/ingestors/research_report_ingestor.py` | `src/knowledge/learners/ingestors/research_ingestor/research_report_ingestor.py` |

### 0.2 After Moving

- Delete the original files from `src/knowledge/learners/ingestors/`
- Rewrite each file to inherit from `ResearchIngestorBase` (Task 4)
- Update imports in the moved files

---

## Task 1: Create Directory Structure

**Location:** `src/knowledge/learners/ingestors/research_ingestor/`

Create the following files:
- [ ] `__init__.py` - Package exports
- [ ] `base.py` - `ResearchIngestorBase` class
- [ ] `idea_ingestor.py` - `IdeaIngestor` class
- [ ] `implementation_ingestor.py` - `ImplementationIngestor` class  
- [ ] `research_report_ingestor.py` - `ResearchReportIngestor` class
- [ ] `utils.py` - Shared utility functions
- [ ] `prompts/` directory with prompt templates

---

## Task 2: Create Prompt Templates

**Location:** `src/knowledge/learners/ingestors/research_ingestor/prompts/`

### 2.1 Planning Prompt (`planning.md`)

Create prompt that instructs agent to:
- Read input content (provided as context)
- Read ALL wiki structure definitions (Principle, Implementation, Environment, Heuristic)
- Read page_connections.md for graph relationships
- Analyze content and decide what page types to create
- Write `_plan.md` with:
  - List of pages to create (type, name, reasoning)
  - Notes about content mapping for each page
  - Graph connections to establish

**Template variables:**
- `{content}` - The research content to ingest
- `{query}` - The original research query
- `{source_url}` - Source URL (if available)
- `{wiki_dir}` - Output wiki directory
- `{wiki_structure_dir}` - Path to wiki_structure definitions

### 2.2 Writing Prompt (`writing.md`)

Create prompt that instructs agent to:
- Read `_plan.md` from planning phase
- For each planned page:
  - Read sections_definition.md for that page type
  - Extract relevant content from research input
  - Write page with ALL required sections
  - Add proper semantic wiki links for graph connections
- Follow WikiMedia naming conventions
- Place files in correct subdirectories

**Template variables:**
- `{content}` - The research content
- `{query}` - The original research query
- `{source_url}` - Source URL
- `{wiki_dir}` - Output wiki directory
- `{plan_path}` - Path to _plan.md
- `{wiki_structure_dir}` - Path to wiki_structure definitions

### 2.3 Auditing Prompt (`auditing.md`)

Create prompt that instructs agent to:
- Read all created wiki pages
- Validate against wiki structure definitions:
  - All required sections present?
  - Metadata block correct?
  - Page title format correct (WikiMedia compliance)?
- Validate graph connections:
  - All link targets exist or are valid?
  - Required connections present (e.g., Principle → Implementation)?
- Fix any issues found
- Write `_audit_report.md` with results

**Template variables:**
- `{wiki_dir}` - Wiki directory with created pages
- `{wiki_structure_dir}` - Path to wiki_structure definitions

---

## Task 3: Implement Base Class

**File:** `src/knowledge/learners/ingestors/research_ingestor/base.py`

### 3.1 Class Definition

```python
class ResearchIngestorBase(Ingestor):
    """
    Base class for agentic research ingestors.
    
    Provides:
    - Claude Code agent initialization (Bedrock by default)
    - Wiki structure loading
    - Three-phase pipeline execution
    - Page collection from wiki directory
    """
```

### 3.2 Constructor (`__init__`)

Parameters to support:
- `timeout` - Agent timeout (default: 600 seconds)
- `use_bedrock` - Use AWS Bedrock (default: True)
- `aws_region` - AWS region for Bedrock (default: "us-east-1")
- `model` - Model override (default: Sonnet on Bedrock)
- `wiki_dir` - Output directory (default: DEFAULT_WIKI_DIR)
- `staging_subdir` - Staging subdirectory (default: "_staging")
- `cleanup_staging` - Remove staging after ingest (default: False)

### 3.3 Agent Initialization (`_initialize_agent`)

- Configure Claude Code with Read, Write, Edit, Bash tools
- Set up Bedrock by default (configurable)
- Use Sonnet model by default for cost efficiency

### 3.4 Phase Execution Methods

- `_run_planning_phase(content, query, source_url)` → bool
- `_run_writing_phase(content, query, source_url)` → bool
- `_run_auditing_phase()` → bool
- `_build_phase_prompt(phase, **kwargs)` → str

### 3.5 Utility Methods

- `_load_prompt(name)` - Load prompt template
- `_collect_pages()` - Parse wiki directory for WikiPage objects
- `_normalize_source(source)` - Extract query, source, content from input

### 3.6 Main `ingest` Method

```python
def ingest(self, source: Any) -> List[WikiPage]:
    # 1. Normalize source to extract content
    # 2. Create staging directory
    # 3. Initialize agent
    # 4. Run planning phase
    # 5. Run writing phase
    # 6. Run auditing phase
    # 7. Collect and return pages
```

---

## Task 4: Implement Ingestors

### 4.1 IdeaIngestor

**File:** `src/knowledge/learners/ingestors/research_ingestor/idea_ingestor.py`

- Inherit from `ResearchIngestorBase`
- Register with `@register_ingestor("idea")`
- Override `source_type` property to return `"idea"`
- No other overrides needed (base class handles everything)

### 4.2 ImplementationIngestor

**File:** `src/knowledge/learners/ingestors/research_ingestor/implementation_ingestor.py`

- Inherit from `ResearchIngestorBase`
- Register with `@register_ingestor("implementation")`
- Override `source_type` property to return `"implementation"`
- No other overrides needed

### 4.3 ResearchReportIngestor

**File:** `src/knowledge/learners/ingestors/research_ingestor/research_report_ingestor.py`

- Inherit from `ResearchIngestorBase`
- Register with `@register_ingestor("researchreport")`
- Override `source_type` property to return `"researchreport"`
- No other overrides needed

---

## Task 5: Implement Utilities

**File:** `src/knowledge/learners/ingestors/research_ingestor/utils.py`

### 5.1 Functions to Implement

- `slugify(text, max_len)` - Create filesystem-safe slug
- `sanitize_wiki_title(text)` - WikiMedia-compliant title
- `load_wiki_structure(page_type)` - Load page definitions (reuse from repo_ingestor)
- `load_all_wiki_structures()` - Load all page type definitions
- `load_page_connections()` - Load page_connections.md
- `get_wiki_structure_dir()` - Get path to wiki_structure directory

---

## Task 6: Create Package Exports

**File:** `src/knowledge/learners/ingestors/research_ingestor/__init__.py`

```python
from src.knowledge.learners.ingestors.research_ingestor.base import ResearchIngestorBase
from src.knowledge.learners.ingestors.research_ingestor.idea_ingestor import IdeaIngestor
from src.knowledge.learners.ingestors.research_ingestor.implementation_ingestor import ImplementationIngestor
from src.knowledge.learners.ingestors.research_ingestor.research_report_ingestor import ResearchReportIngestor

__all__ = [
    "ResearchIngestorBase",
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
```

---

## Task 7: Update Parent Package

**File:** `src/knowledge/learners/ingestors/__init__.py`

### 7.1 Remove Old Imports

Remove:
```python
from src.knowledge.learners.ingestors.idea_ingestor import IdeaIngestor
from src.knowledge.learners.ingestors.implementation_ingestor import ImplementationIngestor
from src.knowledge.learners.ingestors.research_report_ingestor import ResearchReportIngestor
```

### 7.2 Add New Imports

Add:
```python
from src.knowledge.learners.ingestors.research_ingestor import (
    IdeaIngestor,
    ImplementationIngestor,
    ResearchReportIngestor,
)
```

### 7.3 Update `__all__`

Keep the same exports (class names unchanged).

---

## Task 8: Delete Old Ingestor Files

**Note:** These files are moved in Task 0, not deleted. After Task 0, the original locations should be empty.

Verify these files no longer exist at the old locations:
- [ ] `src/knowledge/learners/ingestors/idea_ingestor.py` (moved to research_ingestor/)
- [ ] `src/knowledge/learners/ingestors/implementation_ingestor.py` (moved to research_ingestor/)
- [ ] `src/knowledge/learners/ingestors/research_report_ingestor.py` (moved to research_ingestor/)

---

## Task 9: Write Tests

**File:** `tests/test_research_ingestors.py`

### 9.1 End-to-End Test

Create a complete test that:
1. Runs `Researcher` in all three modes (idea, implementation, study)
2. Passes results to the new agentic ingestors
3. Verifies WikiPage objects are created with proper structure
4. Validates pages follow wiki structure definitions

```python
#!/usr/bin/env python3
"""
End-to-end test for agentic research ingestors.

Tests the full pipeline:
1. Research → Idea/Implementation/ResearchReport
2. Ingest → WikiPage objects with proper structure
3. Validate → Pages follow wiki structure definitions

Usage:
    conda activate praxium_conda
    cd /home/ubuntu/kapso
    python tests/test_research_ingestors.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.knowledge.researcher import Researcher, Idea, Implementation, ResearchReport
from src.knowledge.learners.ingestors import IdeaIngestor, ImplementationIngestor, ResearchReportIngestor
from src.knowledge.search.base import WikiPage


def test_idea_ingestor_e2e():
    """Test: Research idea mode → IdeaIngestor → WikiPages."""
    print("\n" + "=" * 60)
    print("Test: Idea Mode → Ingestor → WikiPages")
    print("=" * 60 + "\n")
    
    # Step 1: Research
    researcher = Researcher()
    query = "Best practices for LLM fine-tuning with LoRA"
    
    ideas = researcher.research(
        query=query,
        mode="idea",
        top_k=2,  # Small for testing
        depth="light",
    )
    
    print(f"Research returned {len(ideas)} ideas")
    assert len(ideas) > 0, "Expected at least 1 idea"
    
    # Step 2: Ingest each idea
    ingestor = IdeaIngestor()
    
    for i, idea in enumerate(ideas, 1):
        print(f"\nIngesting idea {i}...")
        pages = ingestor.ingest(idea)
        
        print(f"  Created {len(pages)} pages")
        assert len(pages) > 0, "Expected at least 1 page"
        
        for page in pages:
            assert isinstance(page, WikiPage), f"Expected WikiPage, got {type(page)}"
            print(f"  - {page.page_type}: {page.page_title}")
            
            # Validate page has required sections
            assert page.content, "Page content should not be empty"
            assert "== Overview ==" in page.content, "Page should have Overview section"
    
    print("\n✅ Idea ingestor test passed!")


def test_implementation_ingestor_e2e():
    """Test: Research implementation mode → ImplementationIngestor → WikiPages."""
    print("\n" + "=" * 60)
    print("Test: Implementation Mode → Ingestor → WikiPages")
    print("=" * 60 + "\n")
    
    # Step 1: Research
    researcher = Researcher()
    query = "How to implement RAG with LangChain"
    
    impls = researcher.research(
        query=query,
        mode="implementation",
        top_k=2,
        depth="light",
    )
    
    print(f"Research returned {len(impls)} implementations")
    assert len(impls) > 0, "Expected at least 1 implementation"
    
    # Step 2: Ingest each implementation
    ingestor = ImplementationIngestor()
    
    for i, impl in enumerate(impls, 1):
        print(f"\nIngesting implementation {i}...")
        pages = ingestor.ingest(impl)
        
        print(f"  Created {len(pages)} pages")
        assert len(pages) > 0, "Expected at least 1 page"
        
        for page in pages:
            assert isinstance(page, WikiPage), f"Expected WikiPage, got {type(page)}"
            print(f"  - {page.page_type}: {page.page_title}")
            
            # Validate page has required sections
            assert page.content, "Page content should not be empty"
            assert "== Overview ==" in page.content, "Page should have Overview section"
    
    print("\n✅ Implementation ingestor test passed!")


def test_research_report_ingestor_e2e():
    """Test: Research study mode → ResearchReportIngestor → WikiPages."""
    print("\n" + "=" * 60)
    print("Test: Study Mode → Ingestor → WikiPages")
    print("=" * 60 + "\n")
    
    # Step 1: Research
    researcher = Researcher()
    query = "Comparison of LoRA, QLoRA, and full fine-tuning for LLMs"
    
    report = researcher.research(
        query=query,
        mode="study",
        depth="light",
    )
    
    print(f"Research returned report with {len(report.content)} chars")
    assert isinstance(report, ResearchReport), f"Expected ResearchReport, got {type(report)}"
    
    # Step 2: Ingest the report
    ingestor = ResearchReportIngestor()
    
    print(f"\nIngesting research report...")
    pages = ingestor.ingest(report)
    
    print(f"  Created {len(pages)} pages")
    assert len(pages) > 0, "Expected at least 1 page"
    
    for page in pages:
        assert isinstance(page, WikiPage), f"Expected WikiPage, got {type(page)}"
        print(f"  - {page.page_type}: {page.page_title}")
        
        # Validate page has required sections
        assert page.content, "Page content should not be empty"
        assert "== Overview ==" in page.content, "Page should have Overview section"
    
    print("\n✅ Research report ingestor test passed!")


def test_full_pipeline_with_learn():
    """Test: Research → Ingest → Learn (full pipeline)."""
    print("\n" + "=" * 60)
    print("Test: Full Pipeline (Research → Ingest → Learn)")
    print("=" * 60 + "\n")
    
    from src.knowledge.learners import KnowledgePipeline
    
    # Step 1: Research in multiple modes
    researcher = Researcher()
    
    ideas = researcher.research(
        query="LoRA fine-tuning best practices",
        mode="idea",
        top_k=1,
        depth="light",
    )
    
    impls = researcher.research(
        query="How to use Unsloth for fine-tuning",
        mode="implementation",
        top_k=1,
        depth="light",
    )
    
    print(f"Research returned {len(ideas)} ideas, {len(impls)} implementations")
    
    # Step 2: Run through pipeline (ingest only, skip merge)
    pipeline = KnowledgePipeline(wiki_dir="data/wikis_test_research")
    
    # Ingest ideas
    for idea in ideas:
        result = pipeline.run(idea, skip_merge=True)
        print(f"Idea ingestion: {result.total_pages_extracted} pages extracted")
        assert result.total_pages_extracted > 0, "Expected pages from idea"
    
    # Ingest implementations
    for impl in impls:
        result = pipeline.run(impl, skip_merge=True)
        print(f"Implementation ingestion: {result.total_pages_extracted} pages extracted")
        assert result.total_pages_extracted > 0, "Expected pages from implementation"
    
    print("\n✅ Full pipeline test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Research Ingestor End-to-End Tests")
    print("=" * 60)
    print("\nEnvironment: praxium_conda")
    
    # Run tests one at a time to avoid rate limits
    # Uncomment the tests you want to run:
    
    test_idea_ingestor_e2e()
    # test_implementation_ingestor_e2e()
    # test_research_report_ingestor_e2e()
    # test_full_pipeline_with_learn()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### 9.2 Test Validation Criteria

For each test, verify:
- [ ] WikiPage objects are returned (not empty list)
- [ ] Page types are valid (Principle, Implementation, Environment, Heuristic)
- [ ] Page content contains required sections (Overview, etc.)
- [ ] Page follows WikiMedia naming conventions
- [ ] Graph connections are established (semantic wiki links)

### 9.3 Test Configuration

Tests should work with:
- Default Bedrock configuration
- Custom model override
- Different wiki_dir paths

---

## Task 10: Update Documentation

### 10.1 Update design.md

Mark implementation status for each section.

### 10.2 Add README

**File:** `src/knowledge/learners/ingestors/research_ingestor/README.md`

Document:
- Purpose of agentic ingestors
- How to use each ingestor
- Configuration options
- Example usage

---

## Execution Order

1. Task 0: Move existing skeleton files
2. Task 1: Create directory structure (prompts/ folder)
3. Task 5: Implement utilities (needed by other tasks)
4. Task 2: Create prompt templates
5. Task 3: Implement base class
6. Task 4: Rewrite ingestors to use base class
7. Task 6: Create package exports
8. Task 7: Update parent package
9. Task 8: Verify old files removed
10. Task 9: Write tests
11. Task 10: Update documentation

---

## Dependencies

- `src.execution.coding_agents.factory.CodingAgentFactory` - For Claude Code agent
- `src.knowledge.search.base.WikiPage` - Output type
- `src.knowledge.search.kg_graph_search.parse_wiki_directory` - Page collection
- AWS Bedrock access (for default configuration)

---

## Estimated Effort

| Task | Complexity | Est. Lines |
|------|------------|------------|
| Task 1 | Low | 10 |
| Task 2 | Medium | 300 |
| Task 3 | High | 250 |
| Task 4 | Low | 60 |
| Task 5 | Low | 80 |
| Task 6 | Low | 15 |
| Task 7 | Low | 10 |
| Task 8 | Low | 0 |
| Task 9 | Medium | 200 |
| Task 10 | Low | 50 |
| **Total** | | **~975** |
