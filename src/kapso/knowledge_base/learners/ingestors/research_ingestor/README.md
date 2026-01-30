# Research Ingestors

Agentic ingestors for converting research outputs (`Source.Idea`, `Source.Implementation`, `Source.ResearchReport`) into properly structured wiki pages.

## Overview

These ingestors use Claude Code to analyze research content and create wiki pages that conform to the Knowledge Graph schema. Unlike simple template-based converters, these agentic ingestors:

1. **Analyze content** to understand its nature
2. **Decide page types** based on wiki structure definitions
3. **Write comprehensive pages** following section definitions
4. **Validate output** against the wiki schema

## Architecture

```
ResearchIngestorBase (base class)
├── IdeaIngestor        (source_type: "idea")
├── ImplementationIngestor (source_type: "implementation")
└── ResearchReportIngestor (source_type: "researchreport")
```

All ingestors share a common three-phase pipeline:

1. **Planning Phase**: Analyze content and decide what pages to create
2. **Writing Phase**: Create wiki pages following section definitions
3. **Auditing Phase**: Validate pages and fix issues

## Usage

### Basic Usage

```python
from kapso.knowledge.learners.ingestors import IdeaIngestor
from kapso.knowledge.types import Source
from kapso.knowledge.researcher import Researcher

# Research
researcher = Researcher()
ideas = researcher.research(
    query="LoRA fine-tuning best practices",
    mode="idea",
    top_k=5,
)

# Ingest
ingestor = IdeaIngestor()
for idea in ideas:
    pages = ingestor.ingest(idea)
    print(f"Created {len(pages)} pages")
```

### With Custom Settings

```python
ingestor = IdeaIngestor(params={
    "use_bedrock": True,           # Use AWS Bedrock (default)
    "aws_region": "us-east-1",     # AWS region
    "model": None,                 # Use default model (Sonnet)
    "timeout": 600,                # Agent timeout in seconds
    "wiki_dir": "data/wikis",      # Output directory
    "cleanup_staging": False,      # Keep staging for debugging
})
```

### Using Direct Anthropic API

```python
ingestor = IdeaIngestor(params={
    "use_bedrock": False,
    "model": "claude-sonnet-4-20250514",
})
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | 600 | Agent timeout in seconds |
| `use_bedrock` | True | Use AWS Bedrock for Claude |
| `aws_region` | "us-east-1" | AWS region for Bedrock |
| `model` | None | Model override (uses Sonnet by default) |
| `wiki_dir` | "data/wikis" | Output directory for wiki pages |
| `staging_subdir` | "_staging" | Subdirectory for staging |
| `cleanup_staging` | False | Remove staging after ingest |

## Page Type Decision

The agent reads wiki structure definitions and decides what page types to create based on content:

| Content Nature | Page Type | Key Indicators |
|----------------|-----------|----------------|
| Theoretical concepts | **Principle** | Explains "what" and "why" |
| Code, APIs, functions | **Implementation** | Has code snippets, signatures |
| Dependencies, setup | **Environment** | Lists packages, versions |
| Tips, trade-offs | **Heuristic** | Contains advice, warnings |

A single input may produce multiple pages of different types.

## Output Structure

Pages are created in the staging directory:

```
{wiki_dir}/_staging/{source_type}_{slug}_{run_id}/
├── _plan.md              # Planning phase output
├── _audit_report.md      # Auditing phase output
├── principles/           # Principle pages
├── implementations/      # Implementation pages
├── environments/         # Environment pages
└── heuristics/           # Heuristic pages
```

## Testing

```bash
conda activate praxium_conda
cd /home/ubuntu/kapso
python tests/test_research_ingestors.py
```

## Files

```
src/knowledge/learners/ingestors/research_ingestor/
├── __init__.py              # Package exports
├── base.py                  # ResearchIngestorBase class
├── idea_ingestor.py         # IdeaIngestor
├── implementation_ingestor.py # ImplementationIngestor
├── research_report_ingestor.py # ResearchReportIngestor
├── utils.py                 # Shared utilities
├── design.md                # Design document
├── tasks.md                 # Implementation tasks
├── README.md                # This file
└── prompts/
    ├── planning.md          # Planning phase prompt
    ├── writing.md           # Writing phase prompt
    └── auditing.md          # Auditing phase prompt
```
