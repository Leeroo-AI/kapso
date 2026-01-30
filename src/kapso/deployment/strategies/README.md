# Deployment Strategies

Each subdirectory is a self-contained deployment strategy package.

## Structure

```
strategies/
├── base.py                 # StrategyRegistry (auto-discovers strategies)
├── local/
│   ├── selector_instruction.md   # When to choose this strategy
│   ├── adapter_instruction.md    # How to adapt code for deployment
│   └── runner.py                 # Runtime execution class
├── docker/
│   └── ...
├── modal/
│   └── ...
├── bentoml/
│   └── ...
└── langgraph/
    └── ...
```

## Adding a New Strategy

1. Create a new directory: `strategies/mycloud/`

2. Create `selector_instruction.md`:
```markdown
# MyCloud

## Summary
One-line description for strategy selection.

## Best For
- Use case 1
- Use case 2

## Not For
- Anti-pattern 1

## Resources
Describe resource requirements (or "No resource specification needed").
Default: gpu=T4, memory=16Gi

## Interface
http (or function, modal, bentocloud, langgraph)

## Provider
mycloud (or None)
```

3. Create `adapter_instruction.md`:
```markdown
# MyCloud Deployment Instructions

## DEPLOY COMMAND

\`\`\`bash
mycloud deploy app.py
\`\`\`

## RUN INTERFACE
- type: http
- endpoint: from deployment output
- path: /predict

## Required Structure

Describe required files and structure.

## Code Examples

Provide templates for main files.
```

4. Create `runner.py`:
```python
from kapso.deployment.strategies.base import Runner

class MyCloudRunner(Runner):
    def __init__(self, endpoint: str, code_path: str = None):
        self.endpoint = endpoint
        self.code_path = code_path
    
    def run(self, inputs):
        # Make request to endpoint
        ...
    
    def stop(self):
        pass
    
    def is_healthy(self):
        return True
    
    def get_logs(self):
        return ""
```

5. Create `__init__.py`:
```python
from kapso.deployment.strategies.mycloud.runner import MyCloudRunner
__all__ = ["MyCloudRunner"]
```

6. Update `DeploymentFactory._create_runner()` to handle the new strategy.

## Usage

```python
from kapso.deployment.strategies import StrategyRegistry

# Get registry (auto-discovers all strategies)
registry = StrategyRegistry.get()

# List available strategies
strategies = registry.list_strategies()  # ['bentoml', 'docker', 'langgraph', 'local', 'modal']

# Filter to specific strategies
strategies = registry.list_strategies(allowed=['local', 'modal'])  # ['local', 'modal']

# Get instructions
selector_md = registry.get_selector_instruction('modal')
adapter_md = registry.get_adapter_instruction('modal')
```

## Flow

```
User Request
    │
    ▼
┌──────────────┐
│   Selector   │ ◄── reads selector_instruction.md from all strategies
│   Agent      │     to determine best deployment option
└──────────────┘
    │
    ▼
┌──────────────┐
│   Adapter    │ ◄── reads adapter_instruction.md for selected strategy
│   Agent      │     to adapt code and deploy
└──────────────┘
    │
    ▼
┌──────────────┐
│   Runner     │ ◄── uses runner.py from selected strategy
│              │     to execute deployed software
└──────────────┘
    │
    ▼
User Gets Result
```

