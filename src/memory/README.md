# Cognitive Memory Architecture

A memory system for the Praxium agent that enables learning from past experiments and providing contextual briefings.

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
│   │ EpisodicStore│ │  KG Search   │ │WorkingMemory │       │
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
- `WorkingMemory`: Current goal, plan, and facts
- `Briefing`: Synthesized context packet for the agent
- `ExperimentResultProtocol`: Interface for experiment results

### Config (`config.py`, `cognitive_memory.yaml`)
YAML-based configuration with presets.

## Quick Start

```python
from src.memory import CognitiveController, WorkingMemory
from src.knowledge.search import KnowledgeSearchFactory

# Initialize with KG search
kg = KnowledgeSearchFactory.create("kg_graph_search")
controller = CognitiveController(knowledge_search=kg)

# Create working memory
memory = WorkingMemory(
    current_goal="Fine-tune LLaMA with LoRA",
    active_plan=["Load model", "Configure LoRA", "Train"],
    facts={"model": "llama-7b"}
)

# Get briefing for agent
briefing = controller.prepare_briefing(memory, last_error=None)
print(briefing.to_string())

# Process experiment result
class Result:
    run_had_error = True
    error_details = "CUDA OOM: reduce batch size"

new_memory, insight = controller.process_result(Result(), memory)

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
    
  briefing:
    max_kg_context: 30000
    max_insights: 10
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

## Running Tests

```bash
# Unit tests
pytest tests/test_cognitive_memory.py -v

# Integration tests (requires Weaviate)
pytest tests/test_cognitive_integration.py -v

# E2E tests (requires full infrastructure)
./start_infra.sh
python tests/test_cognitive_e2e.py
python tests/test_cognitive_comprehensive.py
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
├── types.py                 # Data types (Insight, WorkingMemory, Briefing)
├── episodic.py              # EpisodicStore (Weaviate + JSON)
├── controller.py            # CognitiveController
├── config.py                # Config loader
├── cognitive_memory.yaml    # Default configuration
└── README.md                # This file
```
