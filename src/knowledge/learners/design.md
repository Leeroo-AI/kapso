# Knowledge Merger Design

This document describes the hierarchical sub-graph-aware merge algorithm for the Knowledge Merger.

## Overview

The Knowledge Merger intelligently merges proposed wiki pages into an existing Knowledge Graph (KG). Unlike simple page-by-page merging, this design considers the **graph structure** and processes pages as connected sub-graphs with proper hierarchy awareness.

### Key Principles

1. **Same-type search only**: Implementation searches only among Implementations, etc.
2. **Scoped search**: Children are compared only to children of the merged/matched parent
3. **No cross-Principle merges**: An Implementation under Principle_X' won't merge with an Implementation under Principle_Y
4. **Inherited CREATE_NEW**: When parent is CREATE_NEW, all children automatically become CREATE_NEW (no search needed)
5. **Heuristic tie-breaker**: If a Heuristic connects to multiple levels, use the lower position (closer to leaves), with escalation if no match found
6. **Conflict resolution**: Additive by default (keep both edges), agent decides if clear winner
7. **Bottom-up execution**: Process leaves first, then work up to roots

---

## Data Structures

```python
@dataclass
class NodePlan:
    """Plan for a single node in the sub-graph."""
    page_id: str                          # ID in new pages
    page_type: str                        # Principle, Implementation, etc.
    page: WikiPage                        # The actual page data
    decision: Literal["MERGE", "CREATE_NEW"]
    merge_target_id: Optional[str]        # Target page_id in main graph (if MERGE)
    best_match_score: Optional[float]     # Similarity score of best match
    parent_id: Optional[str]              # Parent node in new pages sub-graph
    deferred_edges: List[Dict]            # Edges to add when parent is processed
    status: Literal["PENDING", "COMPLETED", "FAILED"]
    result_page_id: Optional[str]         # Final page_id after execution (new or merged)

@dataclass
class SubGraphPlan:
    """Plan for an entire sub-graph."""
    subgraph_id: str
    root_id: str                          # Root node page_id
    nodes: Dict[str, NodePlan]            # page_id -> NodePlan
    execution_order: List[str]            # Bottom-up order of page_ids
    status: Literal["PLANNED", "EXECUTING", "COMPLETED", "FAILED", "RETRY"]
    audit_result: Optional[str]           # Audit feedback if failed
    retry_count: int = 0

@dataclass
class MergePlan:
    """Complete merge plan for all new pages."""
    subgraphs: List[SubGraphPlan]
    created_pages: List[str]              # Final list of created page_ids
    edited_pages: List[str]               # Final list of edited page_ids
```

---

## Algorithm

### Phase 1: Sub-Graph Detection

**Input**: `List[WikiPage]` (new pages)
**Output**: `List[SubGraph]` (connected components with roots identified)

1. Build adjacency graph from new pages using `outgoing_links`
2. Find root nodes (nodes with no incoming edges within new pages)
3. For each root, BFS/DFS to collect all descendants
4. Return list of SubGraphs

### Phase 2: Planning (per sub-graph, top-down)

For each SubGraph:

#### Step 2.1: Root Decision

- Search main graph for pages of same type as root
- Agent decides: `MERGE` (with target_id) or `CREATE_NEW`
- Record in NodePlan

#### Step 2.2: Children Decisions (recursive, level by level)

For each level (Principle → Implementation → Environment/Heuristic):

For each node at this level:

**If parent.decision == CREATE_NEW:**
- All children: decision = `CREATE_NEW` (inherited, no search needed)

**Else (parent.decision == MERGE):**
- Get parent's `merge_target` in main graph
- Get `merge_target`'s children of matching type
- Semantic search among those children only
- Agent decides: `MERGE` or `CREATE_NEW`

**Special Case - Heuristic with multiple parents:**
1. Use lowest parent (closest to leaves) for scoped search
2. If no match found, escalate to next higher parent
3. If still no match at any level, `CREATE_NEW`

#### Step 2.3: Compute Execution Order

- Sort nodes by level: Environment → Heuristic → Implementation → Principle
- Within same level, any order is fine
- Store in `SubGraphPlan.execution_order`

#### Step 2.4: Compute Deferred Edges

For each node:
- Identify parent node in sub-graph
- Record edge info in `node.deferred_edges`:

```python
{
    "parent_id": "parent's page_id in new pages",
    "edge_type": "implemented_by | requires_env | uses_heuristic",
    "edge_metadata": {}  # any additional edge data
}
```

### Phase 3: Execution (per sub-graph, bottom-up)

For each SubGraph (following `execution_order`):

For each `node_id` in `execution_order`:

#### Case A: CREATE_NEW

1. Prepare page content from `node.page`
2. Update `outgoing_links` to point to `result_page_ids` of children:
   - For each child that was processed:
   - Replace child's original `page_id` with `child.result_page_id`
3. Call `kg_index(page_data)`
4. Store result in `node.result_page_id`
5. Mark `node.status = COMPLETED`

#### Case B: MERGE

1. Fetch existing page content from `merge_target_id`
2. Merge content (agent decides how to combine)
3. Update `outgoing_links` **ADDITIVELY**:
   - Keep all existing edges from target
   - Add new edges pointing to `result_page_ids` of children
   - Deduplicate if same target (keep richer metadata)
4. Call `kg_edit(merge_target_id, merged_content)`
5. Store `node.result_page_id = merge_target_id`
6. Mark `node.status = COMPLETED`

### Phase 4: Audit (per sub-graph)

After executing each SubGraph:

#### 4.1 Verify Nodes Exist

- For each node with `decision = CREATE_NEW`:
  - Check `node.result_page_id` exists in main graph
- For each node with `decision = MERGE`:
  - Check `merge_target_id` was updated

#### 4.2 Verify Edges

- For each node:
  - Check parent has edge to this node's `result_page_id`
  - Check edge type is correct

#### 4.3 Verify Content

- For merged nodes: key content from new page exists in merged result
- For created nodes: content matches what was submitted

#### 4.4 Handle Failures

If audit fails:
1. Record failure reason in `subgraph.audit_result`
2. Set `subgraph.status = RETRY`
3. Increment `retry_count`
4. If `retry_count < MAX_RETRIES`:
   - Re-execute Phase 3 with audit feedback
   - Resume from first `PENDING`/`FAILED` node in `execution_order`
5. Else:
   - Set `subgraph.status = FAILED`
   - Log error, continue with next subgraph

### Phase 5: Finalize

1. Collect all `result_page_ids` where `decision = CREATE_NEW` → `created_pages`
2. Collect all `result_page_ids` where `decision = MERGE` → `edited_pages`
3. Return `MergePlan` with final results

---

## Plan Document Structure (plan.md)

The agent writes a structured plan document for each merge operation:

```markdown
# Merge Plan

Generated: {timestamp}
Total SubGraphs: {count}

---

## SubGraph 1: {subgraph_id}

### Root
- **Page**: {root_page_id}
- **Type**: {page_type}
- **Decision**: {MERGE | CREATE_NEW}
- **Target**: {merge_target_id or "N/A"}
- **Score**: {similarity_score}

### Execution Order
1. Environment_E1' → MERGE with Environment_E1
2. Environment_E2' → CREATE_NEW
3. Heuristic_H1' → CREATE_NEW
4. Implementation_A' → MERGE with Implementation_A
5. Principle_X' → MERGE with Principle_X

### Node Plans

| Node | Decision | Target | Score | Parent | Deferred Edge | Status |
|------|----------|--------|-------|--------|---------------|--------|
| Principle_X' | MERGE | Principle_X | 0.92 | (root) | - | PENDING |
| Implementation_A' | MERGE | Implementation_A | 0.95 | Principle_X' | implemented_by | PENDING |
| ... | ... | ... | ... | ... | ... | ... |

### Audit Status
- **Status**: {PENDING | PASSED | FAILED}
- **Retry Count**: 0
- **Feedback**: (if failed)

---

## SubGraph 2: ...
```

---

## Example Walkthrough

### Input: New Pages

```
Principle_X' (overview: "QLoRA fine-tuning theory")
├── implemented_by → Implementation_A' (overview: "FastLanguageModel loader")
│   ├── requires_env → Environment_E1' (overview: "CUDA 11.8 + PyTorch 2.0")
│   └── requires_env → Environment_E2' (overview: "16GB VRAM minimum")
├── implemented_by → Implementation_C' (overview: "Custom LoRA config")
│   └── requires_env → Environment_E3' (overview: "bitsandbytes 0.41+")
└── uses_heuristic → Heuristic_H1' (overview: "Memory optimization tips")
```

### Main Graph (existing)

```
Principle_X (overview: "QLoRA parameter-efficient fine-tuning")
├── implemented_by → Implementation_A (overview: "FastLanguageModel.from_pretrained")
│   └── requires_env → Environment_E1 (overview: "CUDA 11.x + PyTorch")
├── implemented_by → Implementation_B (overview: "get_peft_model for LoRA")
│   └── requires_env → Environment_E4 (overview: "PEFT library 0.5+")
└── uses_heuristic → Heuristic_H2 (overview: "Batch size tuning for QLoRA")

(Also exists elsewhere in graph):
Environment_E5 (overview: "bitsandbytes 0.40+") - under some other Implementation
```

### Phase 1 Result

1 sub-graph detected with root = `Principle_X'`

### Phase 2 Result (Planning)

| Node | Decision | Target | Score | Reason |
|------|----------|--------|-------|--------|
| Principle_X' | MERGE | Principle_X | 0.92 | High similarity |
| Implementation_A' | MERGE | Implementation_A | 0.95 | Scoped search in [A, B] |
| Implementation_C' | CREATE_NEW | - | 0.45 | No good match in [A, B] |
| Heuristic_H1' | CREATE_NEW | - | 0.35 | No match in [H2] |
| Environment_E1' | MERGE | Environment_E1 | 0.88 | Scoped search in [E1] |
| Environment_E2' | CREATE_NEW | - | 0.25 | No match in [E1] |
| Environment_E3' | CREATE_NEW | - | - | Inherited from parent (Implementation_C' is CREATE_NEW) |

### Phase 3 Result (Execution)

Execution order: E1' → E2' → E3' → H1' → A' → C' → X'

1. **Environment_E1'** → MERGE into E1 (content updated)
2. **Environment_E2'** → CREATE_NEW as E2
3. **Environment_E3'** → CREATE_NEW as E3 (inherited from parent)
4. **Heuristic_H1'** → CREATE_NEW as H1
5. **Implementation_A'** → MERGE into A (adds edge to E2)
6. **Implementation_C'** → CREATE_NEW as C (edge points to E3)
7. **Principle_X'** → MERGE into X (adds edges to C and H1)

### Final Main Graph

```
Principle_X (MERGED)
├── implemented_by → Implementation_A (MERGED)
│   ├── requires_env → Environment_E1 (MERGED)
│   └── requires_env → Environment_E2 (NEW)
├── implemented_by → Implementation_B (unchanged)
│   └── requires_env → Environment_E4 (unchanged)
├── implemented_by → Implementation_C (NEW)
│   └── requires_env → Environment_E3 (NEW - inherited CREATE_NEW)
├── uses_heuristic → Heuristic_H1 (NEW)
└── uses_heuristic → Heuristic_H2 (unchanged)
```

---

## Edge Cases Handled

| Scenario | Handling |
|----------|----------|
| Basic merge/create | Top-down planning, bottom-up execution |
| Scoped search | Children search among parent's target's children |
| Inherited CREATE_NEW | When parent is CREATE_NEW, all children inherit CREATE_NEW |
| Multi-parent Heuristics | Lowest-first with escalation |
| Deferred edges | Recorded in plan, applied during parent execution |
| Edge deduplication | Additive with dedupe on same target |
| Partial failure | Status tracking enables resume from failed node |
| Audit & retry | Verify nodes, edges, content; retry on failure |

---

## Wiki Hierarchy Reference

The merge algorithm respects the Knowledge Graph hierarchy:

```
Workflow (Entry Point)
    │
    └── github_url → GitHub Repository
    
Principle (Core Node - The Theory)
    │
    ├── implemented_by → Implementation (MANDATORY 1+)
    └── uses_heuristic → Heuristic (optional)
    
Implementation (The Code)
    │
    ├── requires_env → Environment (optional)
    └── uses_heuristic → Heuristic (optional)
    
Environment (Leaf - Target Only)
Heuristic (Leaf - Target Only)
```

**Processing Order** (bottom-up): Environment → Heuristic → Implementation → Principle → Workflow

---

## Configuration

```python
# Maximum retry attempts for failed sub-graphs
MAX_RETRIES = 3
```
