# Research Ingestor Design

## Overview

This document describes the agentic design for research output ingestors. These ingestors convert research outputs (`Idea`, `Implementation`, `ResearchReport`) from the researcher module into properly structured wiki pages that conform to the Knowledge Graph schema.

## Problem Statement

The current ingestors are simple template-based converters that:
1. Don't follow the wiki structure definitions properly
2. Create pages with minimal content that don't match the expected sections
3. Don't establish proper graph connections between pages
4. Don't validate output against the wiki schema

## Solution: Agentic Ingestors

Inspired by the `RepoIngestor`, we implement agentic ingestors that use Claude Code to:
1. Analyze the input content
2. Plan what pages to create based on wiki structure definitions
3. Write pages following section definitions
4. Audit pages for schema compliance and graph integrity

## Architecture

### Shared Base Class: `ResearchIngestorBase`

All three ingestors share a common base class that handles:
- Claude Code agent initialization (Bedrock by default)
- Wiki structure loading
- Phase execution
- Page collection

```
ResearchIngestorBase
├── IdeaIngestor        (idea → agent decides page types)
├── ImplementationIngestor (implementation → agent decides page types)
└── ResearchReportIngestor (researchreport → agent decides page types)
```

**Key Principle:** The agent reads the wiki structure definitions and analyzes the input content to decide what page types to create. The mapping is NOT prescribed - the agent uses its judgment based on:
- The nature of the content (theoretical vs practical vs tips)
- The wiki page definitions (what each page type is for)
- The graph connection rules (what links are required/optional)

### Directory Structure

```
src/knowledge/learners/ingestors/research_ingestor/
├── __init__.py              # Exports all three ingestors
├── base.py                  # ResearchIngestorBase class
├── idea_ingestor.py         # IdeaIngestor
├── implementation_ingestor.py # ImplementationIngestor
├── research_report_ingestor.py # ResearchReportIngestor
├── utils.py                 # Shared utilities
└── prompts/
    ├── planning.md          # Phase 1: Planning prompt
    ├── writing.md           # Phase 2: Writing prompt
    └── auditing.md          # Phase 3: Auditing prompt
```

## Three-Phase Pipeline

Each ingestor runs a three-phase pipeline:

### Phase 1: Planning

**Goal:** Analyze input content and decide what pages to create.

**Input:**
- Research content (Idea/Implementation/ResearchReport)
- Wiki structure definitions (page_definition.md for relevant types)
- Page connections schema (page_connections.md)

**Output:**
- `_plan.md` file containing:
  - List of pages to create (with page type and name)
  - Notes about each page (what content goes where)
  - Graph connections to establish

**Agent Task:**
1. Read the input content thoroughly
2. Read wiki structure definitions for ALL page types (Principle, Implementation, Environment, Heuristic)
3. Read page_connections.md to understand graph relationships
4. Analyze the content and match it to appropriate page types based on definitions
5. Identify distinct concepts/implementations in the content
6. Decide what pages to create based on content nature (NOT prescribed)
7. Write the plan file with reasoning for each page type decision

### Phase 2: Writing

**Goal:** Create wiki pages following section definitions.

**Input:**
- `_plan.md` from Phase 1
- Research content
- Wiki sections definitions (sections_definition.md for each page type)

**Output:**
- Wiki pages in appropriate subdirectories:
  - `principles/Research_{slug}.md`
  - `implementations/Research_{slug}.md`
  - `environments/Research_{slug}.md`
  - `heuristics/Research_{slug}.md`

**Agent Task:**
1. Read the plan file
2. For each planned page:
   a. Read the sections_definition.md for that page type
   b. Extract relevant content from research input
   c. Write the page following ALL required sections
   d. Establish graph connections using semantic wiki links

### Phase 3: Auditing

**Goal:** Validate pages against wiki schema and fix issues.

**Input:**
- Created wiki pages
- Wiki structure definitions
- Page connections schema

**Output:**
- Fixed pages (if issues found)
- `_audit_report.md` with validation results

**Agent Task:**
1. Read all created pages
2. Validate against wiki structure:
   - All required sections present?
   - Metadata block correct?
   - Page title format correct?
3. Validate graph connections:
   - All `[[implemented_by::...]]` links valid?
   - All `[[requires_env::...]]` links valid?
   - Principle pages have at least one implementation link?
4. Fix any issues found
5. Write audit report

## Source Type to Page Type Mapping

### Agent-Driven Page Type Decision

The agent is NOT prescribed what page types to create. Instead, it:

1. **Reads all wiki structure definitions** (Principle, Implementation, Environment, Heuristic)
2. **Analyzes the input content** to understand its nature
3. **Matches content to page types** based on definitions:
   - Content answering "What is this?" and "Why does it work?" → Principle
   - Content with code, APIs, signatures → Implementation
   - Content about dependencies, hardware, setup → Environment
   - Content with tips, trade-offs, debugging tactics → Heuristic

4. **Decides the page mix** based on content richness

### IdeaIngestor (source_type: "idea")

**Input:** `Idea` object with:
- `query`: Research query
- `source`: Source URL
- `content`: Structured content with sections (Description, How to Apply, When to Use, etc.)

**Agent Decision:** Analyzes content and creates appropriate page types. May create:
- Principle pages (if content is theoretical)
- Implementation pages (if content has code examples)
- Heuristic pages (if content has tips/trade-offs)
- Any combination based on content

### ImplementationIngestor (source_type: "implementation")

**Input:** `Implementation` object with:
- `query`: Research query
- `source`: Source URL
- `content`: Structured content with sections (Description, Code Snippet, Dependencies, etc.)

**Agent Decision:** Analyzes content and creates appropriate page types. May create:
- Implementation pages (if content has code/API docs)
- Principle pages (if content explains underlying concepts)
- Environment pages (if dependencies are significant)
- Heuristic pages (if content has tips/pitfalls)
- Any combination based on content

### ResearchReportIngestor (source_type: "researchreport")

**Input:** `ResearchReport` object with:
- `query`: Research query
- `content`: Full academic-style report (Abstract, Introduction, Literature Review, etc.)

**Agent Decision:** Analyzes the comprehensive report and extracts:
- Multiple Principle pages (key concepts)
- Implementation pages (code examples, if any)
- Heuristic pages (best practices, if any)
- The agent decides based on report content

## Agent Configuration

### Default Settings (Bedrock)

```python
agent_specific = {
    "allowed_tools": ["Read", "Write", "Edit", "Bash"],
    "timeout": 600,  # 10 minutes (shorter than repo ingestor)
    "planning_mode": True,
    "use_bedrock": True,  # Default to Bedrock
}

model = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # Sonnet for cost efficiency
```

### Configurable Parameters

```python
params = {
    "timeout": 600,           # Agent timeout in seconds
    "use_bedrock": True,      # Use AWS Bedrock (default: True)
    "aws_region": "us-east-1", # AWS region for Bedrock
    "model": None,            # Model override (uses default if None)
    "wiki_dir": "data/wikis", # Output directory
    "cleanup_staging": False, # Remove staging after ingest
}
```

## Wiki Structure Integration

### Loading Wiki Structures

The agent is provided with wiki structure definitions via the prompt:

```python
def load_wiki_structure(page_type: str) -> str:
    """Load page_definition.md + sections_definition.md for a page type."""
    wiki_structure_dir = Path(__file__).parents[3] / "wiki_structure"
    type_dir = wiki_structure_dir / f"{page_type.lower()}_page"
    # ... load and combine files
```

### Page Naming Convention

All pages follow WikiMedia naming:
- Format: `Research_{Source}_{Slug}.md`
- Example: `Research_Web_QLoRA_Fine_Tuning_Best_Practices.md`

Rules:
- First character capitalized
- Underscores only (no hyphens, no spaces)
- No forbidden characters: `# < > [ ] { } | + : /`

## Graph Connections

### Principle Pages Must Have

```mediawiki
== Related Pages ==
* [[implemented_by::Implementation:Research_{Slug}]]
```

### Implementation Pages May Have

```mediawiki
== Related Pages ==
* [[requires_env::Environment:Research_{Slug}]]
* [[uses_heuristic::Heuristic:Research_{Slug}]]
```

### Heuristic/Environment Pages Use Backlinks

```mediawiki
== Related Pages ==
* [[used_by::Implementation:Research_{Slug}]]
* [[required_by::Implementation:Research_{Slug}]]
```

## Error Handling

### Phase Failures

If a phase fails:
1. Log the error
2. Continue to next phase if possible
3. Return partial results (pages created before failure)

### Validation Failures

If audit finds issues:
1. Attempt to fix automatically
2. Log unfixable issues
3. Return pages anyway (with warnings)

## Usage Example

```python
from src.knowledge.learners.ingestors import IdeaIngestor
from src.knowledge.researcher import Idea

# Create ingestor with Bedrock (default)
ingestor = IdeaIngestor()

# Or with custom settings
ingestor = IdeaIngestor(params={
    "use_bedrock": False,  # Use direct Anthropic API
    "model": "claude-sonnet-4-20250514",
    "timeout": 300,
})

# Ingest an idea
idea = Idea(
    query="QLoRA fine-tuning best practices",
    source="https://example.com/qlora",
    content="..."
)

pages = ingestor.ingest(idea)
# Returns: [WikiPage(page_type="Principle", ...), ...]
```

## Comparison with Current Implementation

| Aspect | Current | New (Agentic) |
|--------|---------|---------------|
| Content Quality | Template-based, minimal | Agent-written, comprehensive |
| Section Compliance | Partial | Full (follows sections_definition.md) |
| Graph Connections | None | Proper semantic wiki links |
| Validation | None | Audit phase validates schema |
| Flexibility | Fixed output | Agent decides based on content |
| Cost | Free | ~$0.01-0.05 per ingest (Sonnet) |

## Implementation Plan

1. **Create directory structure** - `research_ingestor/` with all files
2. **Implement base class** - `ResearchIngestorBase` with shared logic
3. **Write prompt templates** - `planning.md`, `writing.md`, `auditing.md`
4. **Implement ingestors** - Each inherits from base, customizes prompts
5. **Update factory registration** - Register new ingestors
6. **Write tests** - Test each ingestor with sample inputs
7. **Update documentation** - Document new agentic behavior

## Open Questions

1. **Cost vs Quality Trade-off:** Should we use Sonnet (cheaper) or Opus (better quality)?
   - Recommendation: Sonnet by default, Opus as option for complex reports

2. **Multiple Pages from Single Input:** Should we always create multiple pages or keep it simple?
   - Recommendation: Let agent decide based on content richness

3. **Staging Directory:** Should we use staging like RepoIngestor?
   - Recommendation: Yes, for consistency and debugging
