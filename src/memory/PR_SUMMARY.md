# Pull Request: Cognitive Memory System

## Overview

This PR introduces the **Cognitive Memory System** - a workflow-aware context management architecture that enables the Expert agent to leverage the Knowledge Graph (KG) structure for intelligent code generation.

## Key Features

1. **Tiered Knowledge Retrieval**
   - **TIER 1**: Exact workflow match → graph traversal for all linked knowledge
   - **TIER 2**: No match → synthesize workflow from relevant Principles
   - **TIER 3**: On error → LLM infers failing step → find alternative implementations

2. **Episodic Memory**
   - LLM-governed insight extraction (generalizes errors into reusable lessons)
   - LLM-governed retrieval (queries episodic memory based on current context)
   - Weaviate-based semantic search with JSON fallback

3. **LLM-Based Decisions**
   - RETRY: Try again with same workflow (agent sees error feedback)
   - PIVOT: Switch to different workflow from KG
   - COMPLETE: Goal achieved

---

## How to Run

### Starting the Expert Agent

```python
from src.expert import Expert

# Create Expert with cognitive mode enabled
expert = Expert(domain="ml_finetuning")

# Run with a goal
solution = expert.build(
    problem="Fine-tune LLaMA with LoRA adapter",
    max_iterations=5,
    coding_agent_name="claude_code",  # or "gemini", "aider"
    evaluator_name="llm_judge",
    stop_threshold=0.7,
)

# Solution contains the generated code
print(f"Code at: {solution.code_dir}")
print(f"Score: {solution.experiments[-1].score}")
```

### From Command Line

```bash
# Start KG infrastructure first
./start_infra.sh  # Starts Neo4j, Weaviate, indexes KG

# Run with cognitive context manager (set in config.yaml)
PYTHONPATH=. python -m src.runner \
    -p "Fine-tune a language model using LoRA"
```

---

## Switching Between Legacy and Cognitive Mode

### Automatic: Enable KG

The simplest way - just enable KG and cognitive context is used automatically:

```python
from src.expert import Expert

expert = Expert(domain="ml_finetuning")
expert.knowledge_search = KnowledgeSearchFactory.create("kg_graph_search")
# Now cognitive context manager is used automatically!
```

**Rule**: When `knowledge_search.is_enabled() == True`, the orchestrator automatically uses `cognitive` context manager.

### Manual: Config Change

Or explicitly set in `src/config.yaml`:

```yaml
# Enable cognitive context manager
context_manager:
  type: "cognitive"

# Also enable KG for workflow retrieval
knowledge_search:
  type: "kg_graph_search"
  enabled: true
```

### Legacy Mode (MLE/ALE)

Legacy mode uses `kg_enriched` context manager (no workflow tracking):

```yaml
context_manager:
  type: "kg_enriched"  # or omit - this is the default
knowledge_search:
  enabled: false
```

---

## Impact on Existing Benchmarks (MLE/ALE)

**No impact.** The cognitive system is:
- Opt-in via config mode
- Completely isolated from legacy code paths
- Default mode (`GENERIC`) uses legacy `token_efficient` context manager

MLE/ALE runs will continue to use:
- `token_efficient` context manager
- No KG dependency
- No episodic memory

---

## Running Tests

### Prerequisites

```bash
# Start KG infrastructure
./start_infra.sh

# This starts:
# - Neo4j (bolt://localhost:7687)
# - Weaviate (http://localhost:8080)
# - Indexes KG wiki pages
```

### Main E2E Test (Real KG, No Mocking)

```bash
cd /home/ubuntu/praxium
PYTHONPATH=. python tests/test_expert_full_e2e.py

# Expected output:
# - TIER 1 workflow retrieval
# - Graph traversal (6 implementations, heuristics per step)
# - Episodic memory query
# - Agent execution
# - Score: ~0.9
```

### Other Tests

```bash
# Multi-iteration scenarios (uses mocking for results only)
PYTHONPATH=. python tests/test_cognitive_multi_iteration.py

# Real KG connectivity test
PYTHONPATH=. python tests/test_cognitive_real_kg.py
```

---

## Understanding the Logs

### Key Log Sections

1. **Goal Initialization**
```
🎯 GOAL INITIALIZATION
  Goal: Fine-tune a language model using LoRA...
  Type: ml_training | Source: user

📚 KG Retrieval: EXACT_WORKFLOW          ← TIER 1 triggered
   Workflow: huggingface peft LoRA Fine Tuning
   Confidence: 84%
   Steps (6):
     1. Load Base Model → impl: AutoModelForCausalLM | heuristics: 7
     2. Configure LoRA Adapter → impl: LoraConfig init | heuristics: 15
     ...
```

2. **Briefing Preparation**
```
📋 PREPARING BRIEFING
  💭 Episodic Memory: 1 relevant insights retrieved  ← LLM-governed
  ✓ Briefing ready for iteration 1
    Step 1/6: Load Base Model
    Status: in_progress | Attempts: 0
    Implementation: huggingface peft AutoModelForCausalLM from pretrained
    Heuristics (7):
      • Guidelines for selecting which modules...
```

3. **Context Sent to Agent**
```
📤 Context prepared for agent:
   Problem: 545 chars
   Workflow guidance: 12,874 chars    ← Full workflow + heuristics
   Implementation code: 3,828 chars   ← Code snippets from KG
```

4. **LLM Decision**
```
⚙️ PROCESSING EXPERIMENT RESULT
  ✅ Iteration 1 | Score: 0.90

🧠 LLM Decision: COMPLETE
   Confidence: 85%
   Reasoning: Goal achieved with satisfactory score...
```

### Log Location

```
/home/ubuntu/praxium/logs/
├── expert_e2e_YYYYMMDD_HHMMSS.log      # Full execution log
├── expert_e2e_results_*.json            # Structured results
└── expert_e2e_report_*.html             # Visual report
```

---

## File Changes Summary

### New Files
```
src/memory/
├── cognitive_controller.py    # Main orchestrator
├── context.py                 # CognitiveContext + WorkflowState
├── decisions.py               # LLM-based action decisions
├── knowledge_retriever.py     # Tiered KG retrieval (TIER 1/2/3)
├── episodic.py               # Weaviate-based insight storage
├── episodic_retriever.py     # LLM-governed episodic queries
├── insight_extractor.py      # LLM-based insight generalization
├── objective.py              # Structured goal representation
├── types.py                  # Core data types
├── config.py                 # Configuration loader
├── cognitive_memory.yaml     # Default config
└── prompts/                  # LLM prompts
    ├── decide_action.md
    ├── extract_error_insight.md
    ├── extract_success_insight.md
    ├── episodic_retrieval_query.md
    ├── infer_failing_step.md     # NEW: For TIER 3
    └── synthesize_plan.md
```

### Modified Files
```
src/config.yaml                              # Documents context_manager.type options
src/execution/context_manager/
└── cognitive_context_manager.py             # Bridge to CognitiveController
src/knowledge/search/kg_graph_search.py      # Fixed uses_heuristic edge direction
```

---

## Known Issues

### KG Data Quality

Some `uses_heuristic` links in the wiki are reversed (Heuristic → Principle instead of Principle → Heuristic). 

**Workaround**: The indexer (`kg_graph_search.py`) now detects and reverses these backlinks during graph creation.

**Permanent fix**: The KG wiki pages should be corrected to use proper linking direction.

---

## Configuration Reference

### Cognitive Memory Config (`cognitive_memory.yaml`)

```yaml
defaults:
  episodic:
    embedding_model: "text-embedding-3-small"
    retrieval_top_k: 5
    min_confidence: 0.5
    
  controller:
    llm_model: "gpt-4o-mini"
    
  briefing:
    max_episodic_insights: 5
```

### Environment Variable Overrides

```bash
# Use better embeddings
export COGNITIVE_MEMORY_EPISODIC_EMBEDDING_MODEL=text-embedding-3-large

# Use different LLM for decisions
export COGNITIVE_MEMORY_CONTROLLER_LLM_MODEL=gpt-4-turbo
```

---

## Test Results

| Test | Status | Details |
|------|--------|---------|
| `test_expert_full_e2e.py` | ✅ PASS | Score: 0.90, 1 iteration |
| `test_cognitive_multi_iteration.py` | ✅ PASS | 4/4 tests pass |
| `test_cognitive_real_kg.py` | ✅ PASS | KG connectivity verified |

---

## Checklist

- [x] TIER 1 workflow retrieval works
- [x] Graph traversal finds implementations + heuristics
- [x] Episodic memory storage and retrieval works
- [x] LLM decisions (RETRY/PIVOT/COMPLETE) work
- [x] Legacy mode unaffected
- [x] Tests pass without mocking core components
- [x] Documentation updated

