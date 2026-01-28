# Knowledge Merger Implementation Tasks

This document outlines the implementation tasks for the new hierarchical sub-graph-aware merge algorithm.

## Overview

The new design replaces the current page-by-page merge approach with a **fully agentic sub-graph-aware merge**. A single Claude Code agent receives all new wiki pages and a comprehensive instruction prompt, then executes the entire merge process using MCP tools.

## Architecture Change

### Current Architecture
```
KnowledgeMerger
├── For each page:
│   ├── Build prompt with handler instructions
│   ├── Agent searches, decides, executes
│   └── Parse result
└── Return MergeResult
```

### New Architecture
```
KnowledgeMerger
├── Build comprehensive instruction prompt
├── Serialize all new pages as context
├── Single agent call with MCP tools:
│   ├── Phase 1: Detect sub-graphs
│   ├── Phase 2: Plan (top-down)
│   ├── Phase 3: Execute (bottom-up)
│   ├── Phase 4: Audit
│   └── Phase 5: Finalize
└── Parse final result from plan.md
```

---

## Task 1: Create Merge Instruction Prompt

**File**: `src/knowledge/learners/prompts/hierarchical_merge.md`

Create a comprehensive instruction prompt for the agent. This is the core of the agentic approach.

### Prompt Structure

```markdown
# Hierarchical Knowledge Merge

You are a knowledge merge agent. Your task is to merge proposed wiki pages into an existing Knowledge Graph while respecting the graph hierarchy.

## Available MCP Tools

- `search_knowledge` - Search for similar pages in the main graph
- `get_wiki_page` - Read existing page content
- `kg_index` - Create new page in the graph
- `kg_edit` - Update existing page in the graph

## Wiki Directory

- Main KG Path: {wiki_dir}
- Plan Output: {wiki_dir}/_merge_plan.md

## Proposed Pages to Merge

{serialized_pages}

## Wiki Hierarchy (Top-Down DAG)

```
Principle (Core Node - The Theory)
├── implemented_by → Implementation (MANDATORY 1+)
└── uses_heuristic → Heuristic (optional)

Implementation (The Code)
├── requires_env → Environment (optional)
└── uses_heuristic → Heuristic (optional)

Environment (Leaf - Target Only)
Heuristic (Leaf - Target Only)
```

## Your Task

Execute the merge in 5 phases. Write your plan and progress to `{wiki_dir}/_merge_plan.md`.

### Phase 1: Sub-Graph Detection

1. Parse the proposed pages and their `outgoing_links`
2. Build the graph structure
3. Find root nodes (nodes with no incoming edges within proposed pages)
4. Group connected components into sub-graphs
5. Write to plan.md:

```markdown
# Merge Plan

## Phase 1: Sub-Graph Detection

### SubGraph 1
- Root: {page_id}
- Nodes: [list of page_ids]

### SubGraph 2
...
```

### Phase 2: Planning (Top-Down)

For each sub-graph, make merge decisions top-down:

1. **Root Decision**:
   - Use `search_knowledge` to find similar pages of same type
   - Decide: MERGE (with target) or CREATE_NEW
   
2. **Children Decisions** (recursive):
   - If parent = CREATE_NEW: All children = CREATE_NEW (inherited)
   - If parent = MERGE:
     - Get parent's target's children using `get_wiki_page`
     - For each child, search among those children only
     - Decide: MERGE or CREATE_NEW

3. **Heuristic with Multiple Parents**:
   - Use lowest parent (closest to leaves) for scoped search
   - If no match, escalate to next higher parent
   - If still no match, CREATE_NEW

4. **Compute Execution Order**:
   - Sort: Environment → Heuristic → Implementation → Principle
   
5. **Record Deferred Edges**:
   - For each node, record which parent should add an edge to it

6. Write to plan.md:

```markdown
## Phase 2: Planning

### SubGraph 1: {subgraph_id}

#### Root Decision
- Page: {page_id}
- Decision: {MERGE | CREATE_NEW}
- Target: {target_id or N/A}
- Reason: {explanation}

#### Execution Order
1. {page_id} → {decision} with {target}
2. ...

#### Node Plans

| Node | Decision | Target | Parent | Deferred Edge | Status |
|------|----------|--------|--------|---------------|--------|
| ... | ... | ... | ... | ... | PENDING |
```

### Phase 3: Execution (Bottom-Up)

For each sub-graph, execute in the planned order:

1. **For CREATE_NEW nodes**:
   - Update `outgoing_links` to use `result_page_id` of already-processed children
   - Call `kg_index` with page data
   - Record `result_page_id`
   - Update plan.md status to COMPLETED

2. **For MERGE nodes**:
   - Use `get_wiki_page` to fetch target content
   - Merge content intelligently (combine, don't duplicate)
   - Update `outgoing_links` ADDITIVELY:
     - Keep existing edges
     - Add new edges to children's `result_page_id`
     - Deduplicate same targets
   - Call `kg_edit` with merged content
   - Record `result_page_id = target_id`
   - Update plan.md status to COMPLETED

3. Write progress to plan.md after each node.

### Phase 4: Audit

After executing each sub-graph:

1. **Verify Nodes**:
   - For CREATE_NEW: Check page exists via `get_wiki_page`
   - For MERGE: Check target was updated

2. **Verify Edges**:
   - Check parent has edge to child's `result_page_id`

3. **Handle Failures**:
   - If audit fails, record reason in plan.md
   - Retry up to {max_retries} times
   - If still failing, mark as FAILED and continue

4. Write to plan.md:

```markdown
## Phase 4: Audit

### SubGraph 1
- Status: PASSED | FAILED
- Retry Count: 0
- Issues: (if any)
```

### Phase 5: Finalize

1. Collect all `result_page_id` values
2. Categorize as created or edited
3. Write final summary to plan.md:

```markdown
## Phase 5: Final Result

### Created Pages
- {page_id_1}
- {page_id_2}

### Edited Pages
- {page_id_1}
- {page_id_2}

### Status: SUCCESS | PARTIAL | FAILED
```

## Important Rules

1. **Always update plan.md** after each significant action
2. **Scoped search only**: Children search among parent's target's children, not globally
3. **Inherited CREATE_NEW**: If parent is CREATE_NEW, all children are CREATE_NEW
4. **Additive edges**: When merging, keep existing edges and add new ones
5. **Bottom-up execution**: Process leaves before parents
6. **No cross-type merges**: Principles only merge with Principles, etc.

## Output

When complete, ensure plan.md contains:
- All sub-graphs with their plans
- All node statuses (COMPLETED or FAILED)
- Final lists of created and edited pages
- Overall status

The orchestrator will parse plan.md to extract the final result.
```

---

## Task 2: Refactor KnowledgeMerger Class

**File**: `src/knowledge/learners/knowledge_merger.py`

### Changes Required

1. **Remove per-page processing loop**
2. **Remove MergeHandler usage** (instructions now in prompt)
3. **Add page serialization method**
4. **Add plan.md parsing method**
5. **Simplify to single agent call**

### New Class Structure

```python
class KnowledgeMerger:
    """
    Hierarchical sub-graph-aware knowledge merger.
    
    Uses a single Claude Code agent call with comprehensive instructions
    to merge proposed pages into the Knowledge Graph.
    
    Default configuration uses AWS Bedrock with Claude Opus 4.5.
    """
    
    def __init__(self, agent_config: Optional[Dict[str, Any]] = None):
        self._agent_config = agent_config or {}
        self._kg_index_path = self._agent_config.get("kg_index_path")
        self._agent = None
        self._max_retries = 3
    
    def merge(self, proposed_pages: List[WikiPage], wiki_dir: Path) -> MergeResult:
        """
        Main merge entry point.
        
        1. Check if index exists
        2. If no index: create all pages as new
        3. If index exists: run agentic hierarchical merge
        """
        ...
    
    def _serialize_pages(self, pages: List[WikiPage]) -> str:
        """Serialize pages to markdown for prompt context."""
        ...
    
    def _build_merge_prompt(self, pages: List[WikiPage], wiki_dir: Path) -> str:
        """Build the comprehensive merge instruction prompt."""
        ...
    
    def _parse_merge_plan(self, wiki_dir: Path) -> MergeResult:
        """Parse the plan.md file to extract results."""
        ...
    
    def _run_agentic_merge(self, pages: List[WikiPage], wiki_dir: Path) -> MergeResult:
        """Execute the single-agent hierarchical merge."""
        ...
```

---

## Task 3: Update MergeResult Data Structure

**File**: `src/knowledge/learners/knowledge_merger.py`

### Current Structure
```python
@dataclass
class MergeResult:
    total_proposed: int = 0
    created: List[str] = field(default_factory=list)
    merged: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
```

### New Structure (Enhanced)
```python
@dataclass
class MergeResult:
    """Result of hierarchical merge operation."""
    total_proposed: int = 0
    subgraphs_processed: int = 0
    created: List[str] = field(default_factory=list)      # New page IDs
    edited: List[str] = field(default_factory=list)       # Merged page IDs (renamed from 'merged')
    failed: List[str] = field(default_factory=list)       # Failed page IDs
    errors: List[str] = field(default_factory=list)       # Error messages
    plan_path: Optional[Path] = None                      # Path to plan.md
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.failed) == 0
```

---

## Task 4: Remove Merge Handlers

**Files to modify/delete**:
- `src/knowledge/learners/merge_handlers/` - **DELETE entire directory**
- `src/knowledge/learners/__init__.py` - Remove handler imports
- `src/knowledge/learners/merger_prompt.md` - **DELETE** (replaced by new prompt)

The merge instructions are now embedded in the comprehensive prompt, not in separate handler classes.

---

## Task 5: Add Page Serialization

**File**: `src/knowledge/learners/knowledge_merger.py`

Add method to serialize WikiPage objects for prompt context:

```python
def _serialize_pages(self, pages: List[WikiPage]) -> str:
    """
    Serialize pages to markdown format for prompt context.
    
    Format:
    ```
    ## Page: {page_id}
    
    - **Type**: {page_type}
    - **Title**: {page_title}
    - **Overview**: {overview}
    - **Domains**: {domains}
    - **Outgoing Links**:
      - {edge_type} → {target_type}:{target_id}
    
    ### Content
    {content}
    
    ---
    ```
    """
    parts = []
    for page in pages:
        parts.append(f"## Page: {page.id}\n")
        parts.append(f"- **Type**: {page.page_type}")
        parts.append(f"- **Title**: {page.page_title}")
        parts.append(f"- **Overview**: {page.overview}")
        parts.append(f"- **Domains**: {', '.join(page.domains) if page.domains else 'None'}")
        
        if page.outgoing_links:
            parts.append("- **Outgoing Links**:")
            for link in page.outgoing_links:
                edge = link.get('edge_type', 'related')
                target_type = link.get('target_type', '')
                target_id = link.get('target_id', '')
                parts.append(f"  - {edge} → {target_type}:{target_id}")
        
        parts.append("\n### Content")
        parts.append(page.content[:4000])  # Truncate if too long
        if len(page.content) > 4000:
            parts.append("\n... [content truncated]")
        
        parts.append("\n---\n")
    
    return "\n".join(parts)
```

---

## Task 6: Add Plan Parsing

**File**: `src/knowledge/learners/knowledge_merger.py`

Add method to parse the agent's plan.md output:

```python
def _parse_merge_plan(self, wiki_dir: Path) -> MergeResult:
    """
    Parse the plan.md file to extract merge results.
    
    Looks for:
    - ## Phase 5: Final Result
    - ### Created Pages
    - ### Edited Pages
    - ### Status
    """
    plan_path = wiki_dir / "_merge_plan.md"
    result = MergeResult(plan_path=plan_path)
    
    if not plan_path.exists():
        result.errors.append("Plan file not found")
        return result
    
    content = plan_path.read_text(encoding="utf-8")
    
    # Parse created pages
    created_match = re.search(
        r'### Created Pages\n(.*?)(?=###|\Z)', 
        content, 
        re.DOTALL
    )
    if created_match:
        for line in created_match.group(1).strip().split('\n'):
            if line.startswith('- '):
                result.created.append(line[2:].strip())
    
    # Parse edited pages
    edited_match = re.search(
        r'### Edited Pages\n(.*?)(?=###|\Z)', 
        content, 
        re.DOTALL
    )
    if edited_match:
        for line in edited_match.group(1).strip().split('\n'):
            if line.startswith('- '):
                result.edited.append(line[2:].strip())
    
    # Parse status
    status_match = re.search(r'### Status:\s*(\w+)', content)
    if status_match:
        status = status_match.group(1)
        if status == "FAILED":
            result.errors.append("Merge failed - see plan.md for details")
    
    # Count subgraphs
    subgraph_count = len(re.findall(r'### SubGraph \d+', content))
    result.subgraphs_processed = subgraph_count
    
    return result
```

---

## Task 7: Update Agent Initialization

**File**: `src/knowledge/learners/knowledge_merger.py`

Update agent initialization to use the wiki MCP server. **Default to Bedrock** if no config is provided (matching RepoIngestor pattern):

```python
def _initialize_agent(self, workspace: Path) -> None:
    """
    Initialize Claude Code agent with wiki MCP tools.
    
    Supports passing through agent_specific settings from agent_config:
        - use_bedrock: True to use AWS Bedrock instead of direct Anthropic API
        - aws_region: AWS region for Bedrock (default: "us-east-1")
        - model: Model name (required for Bedrock, e.g. "us.anthropic.claude-opus-4-5-20251101-v1:0")
    
    If no config is provided, defaults to Bedrock with Claude Opus 4.5.
    """
    agent_specific = {
        "allowed_tools": [
            "Read",
            "Write",
            "mcp__kg-graph-search__search_knowledge",
            "mcp__kg-graph-search__get_wiki_page",
            "mcp__kg-graph-search__kg_index",
            "mcp__kg-graph-search__kg_edit",
        ],
        "timeout": self._agent_config.get("timeout", 3600),  # 1 hour for complex merges
        "planning_mode": True,
    }
    
    # Model override - important for Bedrock which requires specific model IDs
    # Bedrock model IDs look like: us.anthropic.claude-opus-4-5-20251101-v1:0
    # Direct Anthropic API uses: claude-opus-4-5
    model = self._agent_config.get("model")
    
    # Default to Bedrock if use_bedrock is not explicitly set to False
    # This matches the RepoIngestor pattern
    use_bedrock = self._agent_config.get("use_bedrock", True)  # Default True
    
    if use_bedrock:
        agent_specific["use_bedrock"] = True
        # Allow aws_region override, default is handled by ClaudeCodeCodingAgent
        if self._agent_config.get("aws_region"):
            agent_specific["aws_region"] = self._agent_config["aws_region"]
        # Default Bedrock model if not specified (Claude Opus 4.5 on Bedrock)
        if not model:
            model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    
    # Pass KG index path to MCP server
    if self._kg_index_path:
        agent_specific["env_overrides"] = {
            "KG_INDEX_PATH": str(self._kg_index_path)
        }
    
    config = CodingAgentFactory.build_config(
        agent_type="claude_code",
        model=model,
        debug_model=model,  # Use same model for debug
        agent_specific=agent_specific,
    )
    
    self._agent = CodingAgentFactory.create(config)
    self._agent.initialize(str(workspace))
    logger.info(f"Initialized Claude Code agent for {workspace} (bedrock={use_bedrock})")
```

---

## Task 8: Implement Main Merge Method

**File**: `src/knowledge/learners/knowledge_merger.py`

```python
def merge(self, proposed_pages: List[WikiPage], wiki_dir: Path) -> MergeResult:
    """
    Main merge entry point using hierarchical sub-graph-aware algorithm.
    """
    wiki_dir = Path(wiki_dir).expanduser().resolve()
    result = MergeResult(total_proposed=len(proposed_pages))
    
    if not proposed_pages:
        logger.warning("No proposed pages to merge")
        return result
    
    try:
        # Check if index exists
        has_index = self._try_initialize_index()
        
        if not has_index:
            # No index - create all pages as new
            logger.info("No existing index. Creating all pages as new...")
            return self._create_all_pages(proposed_pages, wiki_dir, result)
        
        # Run agentic hierarchical merge
        return self._run_agentic_merge(proposed_pages, wiki_dir)
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        result.errors.append(str(e))
        return result

def _run_agentic_merge(
    self, 
    pages: List[WikiPage], 
    wiki_dir: Path
) -> MergeResult:
    """
    Execute the single-agent hierarchical merge.
    """
    # Initialize agent
    self._initialize_agent(wiki_dir)
    
    # Build comprehensive prompt
    prompt = self._build_merge_prompt(pages, wiki_dir)
    
    logger.info(f"Running hierarchical merge for {len(pages)} pages...")
    
    # Single agent call
    agent_result = self._agent.generate_code(prompt)
    
    if not agent_result.success:
        result = MergeResult(total_proposed=len(pages))
        result.errors.append(f"Agent failed: {agent_result.error}")
        return result
    
    # Parse results from plan.md
    result = self._parse_merge_plan(wiki_dir)
    result.total_proposed = len(pages)
    
    logger.info(
        f"Merge complete: {len(result.created)} created, "
        f"{len(result.edited)} edited, {len(result.errors)} errors"
    )
    
    return result
```

---

## Task 9: Create Prompts Directory

**Directory**: `src/knowledge/learners/prompts/`

Create the prompts directory structure:

```
src/knowledge/learners/prompts/
├── __init__.py
└── hierarchical_merge.md
```

The `__init__.py` should provide a helper to load prompts:

```python
# src/knowledge/learners/prompts/__init__.py

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

def load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")
```

---

## Task 10: Update __init__.py Exports

**File**: `src/knowledge/learners/__init__.py`

Remove merge handler exports, keep only:

```python
# Main pipeline
from src.knowledge.learners.knowledge_learner_pipeline import (
    KnowledgePipeline,
    PipelineResult,
)

# Source types
from src.knowledge.learners.sources import Source

# Merger
from src.knowledge.learners.knowledge_merger import (
    KnowledgeMerger,
    MergeResult,
)

# Ingestors
from src.knowledge.learners.ingestors import (
    Ingestor,
    IngestorFactory,
    register_ingestor,
    RepoIngestor,
    ExperimentIngestor,
    IdeaIngestor,
    ImplementationIngestor,
    ResearchReportIngestor,
)

__all__ = [
    "KnowledgePipeline",
    "PipelineResult",
    "Source",
    "KnowledgeMerger",
    "MergeResult",
    "Ingestor",
    "IngestorFactory",
    "register_ingestor",
    "RepoIngestor",
    "ExperimentIngestor",
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
```

---

## Implementation Order

1. **Task 9**: Create prompts directory structure
2. **Task 1**: Create hierarchical_merge.md prompt
3. **Task 3**: Update MergeResult data structure
4. **Task 5**: Add page serialization method
5. **Task 6**: Add plan parsing method
6. **Task 7**: Update agent initialization
7. **Task 8**: Implement main merge method
8. **Task 2**: Refactor KnowledgeMerger class (integrate all above)
9. **Task 4**: Remove merge handlers
10. **Task 10**: Update exports

---

## Testing

### Test Data

Three test directories are available in `data/wikis_test_research/_staging/`:

1. **idea_Best_practices_for_LLM_fine_tu_32be8ec18368/**
   - Principle: `LoRA_Adapter_Configuration`
   - Implementation: `PEFT_LoraConfig`
   - Environment: `PEFT_Training_Environment`
   - Heuristic: `LoRA_Hyperparameter_Selection`

2. **implementation_How_to_implement_RAG_with_Lang_4284e5484393/**
   - Principle: `RAG_Chain_Pattern`
   - Implementation: `LangChain_Create_Retrieval_Chain`
   - Environment: `LangChain_RAG_Environment`
   - Heuristic: `RAG_Configuration_Tips`

3. **researchreport_Comparison_of_LoRA_QLoRA_and_f_6f518d536006/**
   - Principles: `Full_Fine_Tuning_LLM`, `LoRA_Low_Rank_Adaptation`, `QLoRA_Quantized_Low_Rank_Adaptation`
   - Implementations: `BitsAndBytes_QLoRA_Configuration`, `HuggingFace_Trainer_Full_Finetuning`, `PEFT_LoRA_Configuration`
   - Environment: `PEFT_Training_Environment`
   - Heuristics: `LoRA_Hyperparameter_Selection`, `PEFT_Method_Selection_Framework`, `QLoRA_Memory_Optimization`

### Test Script

```python
from src.knowledge.learners import KnowledgeMerger
from src.knowledge.search.kg_graph_search import parse_wiki_directory
from pathlib import Path

# Load test pages from one of the staging directories
test_dir = Path("data/wikis_test_research/_staging/researchreport_Comparison_of_LoRA_QLoRA_and_f_6f518d536006")
pages = parse_wiki_directory(test_dir)

print(f"Loaded {len(pages)} pages from {test_dir.name}")
for page in pages:
    print(f"  - {page.id} ({page.page_type})")

# Run merge (uses Bedrock by default if no config provided)
merger = KnowledgeMerger(agent_config={"kg_index_path": "path/to/.index"})
result = merger.merge(pages, wiki_dir=Path("data/wikis"))

print(f"\nResults:")
print(f"  Created: {result.created}")
print(f"  Edited: {result.edited}")
print(f"  Plan: {result.plan_path}")
```
