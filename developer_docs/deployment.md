# Deployment System

Deploy solutions to various targets (local, Docker, Modal, BentoML, LangGraph).

---

## User Flow

```python
from src.expert import Expert, DeployStrategy

expert = Expert(domain="ml")
solution = expert.build(goal="Sentiment API", output_path="./repo")

# Deploy and run
software = expert.deploy(solution, strategy=DeployStrategy.LOCAL)
result = software.run({"text": "I love this!"})
software.stop()
```

---

## Architecture

```
expert.deploy(solution, strategy)
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DeploymentFactory                        │
│                                                             │
│  Phase 1: Select Strategy                                   │
│  ┌─────────────┐                                            │
│  │  Selector   │ ← reads selector_instruction.md            │
│  │  Agent      │   from each strategy                       │
│  └─────────────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  Phase 2: Adapt & Deploy                                    │
│  ┌─────────────┐                                            │
│  │  Adapter    │ ← reads adapter_instruction.md             │
│  │  Agent      │   creates deployment files, runs deploy    │
│  └─────────────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  Phase 3: Create Runner                                     │
│  ┌─────────────┐                                            │
│  │   Runner    │ ← strategy-specific execution              │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
         Software (unified interface)
```

---

## Available Strategies

| Strategy | Description | Interface | Use Case |
|----------|-------------|-----------|----------|
| `LOCAL` | Direct Python import | function | Development, testing |
| `DOCKER` | Container-based | http | Isolation, reproducibility |
| `MODAL` | Serverless GPU cloud | http | GPU workloads, scaling |
| `BENTOML` | ML model serving | http | Production ML APIs |
| `LANGGRAPH` | LangGraph Platform | langgraph | Conversational AI agents |

---

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Software` | `base.py` | Abstract interface users interact with |
| `DeployedSoftware` | `software.py` | Unified wrapper around any runner |
| `DeploymentFactory` | `factory.py` | Orchestrates selection → adaptation → runner |
| `SelectorAgent` | `selector/agent.py` | LLM picks best strategy |
| `AdapterAgent` | `adapter/agent.py` | LLM adapts code, runs deployment |
| `Runner` | `strategies/base.py` | Abstract runner interface |
| `StrategyRegistry` | `strategies/base.py` | Auto-discovers strategies from directories |

---

## Strategy Structure

Each strategy in `strategies/{name}/` contains:

```
strategies/local/
├── config.yaml              # Strategy config (interface, provider, run_interface)
├── selector_instruction.md  # When to choose this strategy
├── adapter_instruction.md   # How to adapt and deploy
└── runner.py               # Runtime execution class
```

---

## Software Interface

All strategies return a `Software` with the same interface:

```python
software.run(inputs)      # Execute with inputs → {"status": "success", "output": ...}
software.stop()           # Cleanup resources
software.start()          # Restart after stop
software.is_healthy()     # Check if running
software.logs()           # Get execution logs
software.name             # Strategy name
```

Response format is always:
```python
{"status": "success", "output": <result>}
{"status": "error", "error": <message>}
```

---

## Test

**Location:** `tests/deployment_examples/test_unified_deployment.py`

Tests the full deployment flow with sample repos:

```bash
python tests/deployment_examples/test_unified_deployment.py              # All repos, LOCAL
python tests/deployment_examples/test_unified_deployment.py sentiment    # Specific repo
python tests/deployment_examples/test_unified_deployment.py --strategy auto  # AUTO selection
```

Sample repos in `tests/deployment_examples/input_repos/`:
- `sentiment` — TextBlob sentiment API
- `image` — Pillow image processing
- `embeddings` — sentence-transformers
- `qa` — transformers QA pipeline
- `classifier` — ML text classification
- `chatbot` — LangGraph conversational AI

---

## Adding a New Strategy

1. Create `strategies/mycloud/` directory
2. Add `config.yaml`, `selector_instruction.md`, `adapter_instruction.md`
3. Create `runner.py` with class inheriting from `Runner`
4. Strategy auto-discovered via `StrategyRegistry`

---

## File Locations

```
src/deployment/
├── base.py                  # Software, DeployStrategy, DeployConfig
├── factory.py               # DeploymentFactory
├── software.py              # DeployedSoftware implementation
├── selector/
│   ├── agent.py             # SelectorAgent
│   └── selection_prompt.md  # Prompt template
├── adapter/
│   ├── agent.py             # AdapterAgent
│   └── adaptation_prompt.txt
└── strategies/
    ├── base.py              # Runner, StrategyRegistry
    ├── local/               # Local Python execution
    ├── docker/              # Docker container
    ├── modal/               # Modal serverless
    ├── bentoml/             # BentoML serving
    └── langgraph/           # LangGraph Platform
```

