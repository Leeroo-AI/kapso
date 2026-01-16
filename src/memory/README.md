# Cognitive Memory Architecture

A memory system for the Kapso agent that enables learning from past experiments and providing contextual briefings.

## Overview

The Cognitive Memory Architecture addresses "context stuffing" by intelligently managing what information gets passed to the agent. Instead of dumping all available context, it:

1. **Learns** - Extracts insights from errors and successes
2. **Remembers** - Stores insights in a searchable vector database
3. **Retrieves** - Finds relevant knowledge based on current goal
4. **Synthesizes** - Creates focused briefings for the agent

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CognitiveController                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Meta-Cognition Loop                      │   │
│  │   Reflect → Generate Query → Retrieve → Synthesize   │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│   │ EpisodicStore│ │  KG Search   │ │   Context    │       │
│   │  (Weaviate)  │ │(Neo4j+Weav.) │ │   (State)    │       │
│   └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### EpisodicStore (`episodic.py`)
Stores learned insights from past experiments.

- **Primary**: Weaviate vector database for semantic search
- **Fallback**: JSON file for persistence without Weaviate
- **Features**:
  - Client-side embedding generation (OpenAI)
  - Duplicate detection before insertion
  - Confidence-based filtering
  - Automatic pruning when max_insights exceeded

### CognitiveController (`controller.py`)
Orchestrates the memory system.

- **Briefing Generation**: Creates context packets for the agent
- **Insight Extraction**: Uses LLM to generalize rules from errors
- **State Management**: Persists working memory to file
- **Fallback Support**: Retries with fallback models on failure

### Types (`types.py`)
Core data structures:

- `Insight`: A learned rule with confidence and source tracking
- `Briefing`: Synthesized context packet for the agent
- `Goal`: Parsed goal with type classification
- `ExperimentResultProtocol`: Interface for experiment results

### Config (`config.py`, `cognitive_memory.yaml`)
YAML-based configuration with presets.

## Quick Start

```python
from src.memory import CognitiveController, Goal
from src.knowledge.search import KnowledgeSearchFactory

# Initialize with KG search
kg = KnowledgeSearchFactory.create("kg_graph_search")
controller = CognitiveController(knowledge_search=kg)

# Initialize goal (triggers KG retrieval for workflow)
goal = Goal.from_string("Fine-tune LLaMA with LoRA")
controller.initialize_goal(goal)

# Get briefing for agent (includes workflow, heuristics, code patterns)
briefing = controller.prepare_briefing()
print(briefing.to_string())

# Process experiment result
action, meta = controller.process_result(
    success=False,
    error_message="CUDA OOM: reduce batch size",
    score=0.3,
    feedback="Out of memory error"
)
# Returns: action="retry", meta={"reasoning": "..."}

# Clean up
controller.close()
```

## Configuration

### Default Config (`cognitive_memory.yaml`)

```yaml
defaults:
  episodic:
    embedding_model: "text-embedding-3-small"
    retrieval_top_k: 5
    min_confidence: 0.5
    max_insights: 1000
    
  controller:
    llm_model: "gpt-4o-mini"
    fallback_models: ["gpt-4.1-mini"]
    max_error_length: 1000
    
  insight_extraction:
    enabled: true
    max_insight_length: 500
    default_confidence: 0.8
    
  # Context size controlled implicitly via KG graph structure:
  # - ALL heuristics linked to steps (via USES_HEURISTIC edges)
  # - ALL implementations linked to steps (via IMPLEMENTED_BY edges)
  # - ALL environments linked to implementations (via REQUIRES_ENV edges)
  # Well-curated KG = well-bounded context. No arbitrary truncation.
  briefing:
    max_episodic_insights: 5  # Only episodic is limited
```

### Presets

| Preset | Use Case |
|--------|----------|
| `minimal` | Resource-constrained (100 insights, 10K context) |
| `high_quality` | Better accuracy (large embeddings, 5K insights) |
| `local` | Local development (localhost) |
| `docker` | Docker deployment (container names) |

```python
from src.memory.config import CognitiveMemoryConfig

# Load with preset
config = CognitiveMemoryConfig.load(preset="high_quality")
controller = CognitiveController(config=config)
```

### Environment Variables

Override any setting with environment variables:

```bash
export COGNITIVE_MEMORY_CONTROLLER_LLM_MODEL=gpt-4-turbo
export COGNITIVE_MEMORY_EPISODIC_EMBEDDING_MODEL=text-embedding-3-large
```

## Switching Between Legacy and Cognitive Mode

The cognitive system is **opt-in** - just change ONE config line.

### In `src/config.yaml`

```yaml
# Legacy (default for MLE/ALE)
context_manager:
  type: "token_efficient"

# Cognitive (new system)
context_manager:
  type: "cognitive"
```

That's it. No new mode needed. Also enable KG if you want workflow retrieval:

```yaml
knowledge_search:
  type: "kg_graph_search"
  enabled: true
```

## Running Tests

### Prerequisites

```bash
# Start KG infrastructure (Neo4j + Weaviate)
./start_infra.sh
```

### Main E2E Test (Real KG, No Mocking)

```bash
PYTHONPATH=. python tests/test_expert_full_e2e.py

# Expected: TIER 1 workflow retrieval, score ~0.9
```

### Other Tests

```bash
# Multi-iteration scenarios
PYTHONPATH=. python tests/test_cognitive_multi_iteration.py

# Quick KG connectivity test
PYTHONPATH=. python tests/test_cognitive_real_kg.py
```

### Understanding Logs

Logs are written to `/home/ubuntu/kapso/logs/`:
- `expert_e2e_*.log` - Full execution log
- `expert_e2e_results_*.json` - Structured results

Key log sections:
```
🎯 GOAL INITIALIZATION  → Shows TIER 1/2/3 retrieval
📋 PREPARING BRIEFING   → Shows context sent to agent
⚙️ PROCESSING RESULT    → Shows experiment outcome
🧠 LLM DECISION         → Shows RETRY/PIVOT/COMPLETE decision
```

## Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_cognitive_memory.py` | Unit tests for types and EpisodicStore |
| `test_cognitive_integration.py` | Integration with CognitiveController |
| `test_cognitive_e2e.py` | Basic E2E smoke tests |
| `test_cognitive_comprehensive.py` | Semantic relevance, persistence, learning |

## Dependencies

- `weaviate-client>=4.0` - Vector database
- `openai>=1.0` - Embeddings
- `pyyaml` - Config loading

## Files

```
src/memory/
├── __init__.py              # Module exports
├── types.py                 # Data types (Insight, Briefing, Goal)
├── episodic.py              # EpisodicStore (Weaviate + JSON)
├── controller.py            # CognitiveController
├── config.py                # Config loader
├── cognitive_memory.yaml    # Default configuration
└── README.md                # This file
```
