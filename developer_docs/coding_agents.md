# Coding Agents

Pluggable architecture for integrating coding agents (Aider, Gemini, Claude Code, OpenHands).

## Quick Start

```python
from src.execution.coding_agents import CodingAgentFactory, CodingAgentConfig

# List available agents
print(CodingAgentFactory.list_available())
# Output: ['aider', 'claude_code', 'gemini', 'openhands']

# Create agent with defaults from agents.yaml
config = CodingAgentFactory.build_config(agent_type="aider")
agent = CodingAgentFactory.create(config)

# Or with custom settings
config = CodingAgentConfig(
    agent_type="gemini",
    model="gemini-2.5-pro",
    debug_model="gemini-2.5-flash",
    workspace="/path/to/workspace",
    agent_specific={"temperature": 0.5}
)
agent = CodingAgentFactory.create(config)
```

## Available Agents

| Agent | Description | Env Var |
|-------|-------------|---------|
| aider | Git-centric diff-based editing | `OPENAI_API_KEY` |
| gemini | Google Gemini SDK | `GOOGLE_API_KEY` |
| claude_code | Anthropic Claude Code CLI | `ANTHROPIC_API_KEY` |
| openhands | Sandboxed execution | `OPENAI_API_KEY` |

## Example: Claude Code Usage

```python
from src.execution.coding_agents import CodingAgentFactory, CodingAgentConfig

# Create Claude Code config with custom options
config = CodingAgentConfig(
    agent_type="claude_code",
    model="claude-sonnet-4-20250514",
    debug_model="claude-sonnet-4-20250514",
    workspace="/path/to/project",
    agent_specific={
        "timeout": 3600,              # 1 hour timeout
        "streaming": True,            # Live output to terminal
        "allowed_tools": ["Edit", "Read", "Write", "Bash"],
        # For AWS Bedrock (optional):
        # "use_bedrock": True,
        # "aws_region": "us-east-1",
    }
)

# Create and initialize the agent
agent = CodingAgentFactory.create(config)
agent.initialize("/path/to/project")

# Generate code
result = agent.generate_code(
    prompt="Refactor the authentication module to use JWT tokens",
    debug_mode=False
)

# Check result
if result.success:
    print(f"Files changed: {result.files_changed}")
    print(f"Cost: ${result.cost:.4f}")
else:
    print(f"Error: {result.error}")

# Cleanup when done
agent.cleanup()
```

## Architecture

```
src/execution/coding_agents/
├── __init__.py              # Package exports
├── agents.yaml              # Central agent registry
├── base.py                  # CodingAgentInterface, CodingResult, CodingAgentConfig
├── factory.py               # CodingAgentFactory (auto-discovery)
├── commit_message_generator.py
└── adapters/
    ├── TEMPLATE.py          # Template for new agents
    ├── aider_agent.py
    ├── gemini_agent.py
    ├── claude_code_agent.py
    └── openhands_agent.py
```

## Key Interfaces

### CodingAgentInterface

All agents implement this abstract class:

```python
class CodingAgentInterface(ABC):
    def initialize(self, workspace: str) -> None: ...
    def generate_code(self, prompt: str, debug_mode: bool) -> CodingResult: ...
    def cleanup(self) -> None: ...
    def supports_native_git(self) -> bool: ...
    def get_cumulative_cost(self) -> float: ...
```

### CodingResult

```python
@dataclass
class CodingResult:
    success: bool                    # Whether generation succeeded
    output: str                      # Agent's response text
    files_changed: List[str] = []    # Modified file paths
    error: Optional[str] = None      # Error message if failed
    cost: float = 0.0                # API cost in dollars
    commit_message: Optional[str] = None
```

### CodingAgentConfig

```python
@dataclass
class CodingAgentConfig:
    agent_type: str           # "aider", "gemini", etc.
    model: str                # Primary model
    debug_model: str          # Debug/fix model
    workspace: str = ""       # Working directory
    use_git: bool = True
    agent_specific: Dict[str, Any] = {}  # Agent-specific options
```

## Integration with ExperimentSession

**Key Design**: `ExperimentSession` owns git operations. Agents only generate code.

```python
# ExperimentSession creates and manages coding agent
session = ExperimentSession(
    main_repo=repo,
    session_folder=folder,
    coding_agent_config=config,
    parent_branch_name="main",
    branch_name="experiment-1"
)

# Unified code generation interface
result = session.generate_code(prompt="Implement feature X")

# ExperimentSession handles commits for non-git agents
# (if agent.supports_native_git() returns False)
```

## Adding a New Agent

### Step 1: Create Adapter

Copy template and implement:

```bash
cp src/execution/coding_agents/adapters/TEMPLATE.py \
   src/execution/coding_agents/adapters/my_agent_agent.py
```

Implement required methods:
- `initialize(workspace)` - Setup client (no git operations)
- `generate_code(prompt, debug_mode)` - Generate code, return `CodingResult`
- `cleanup()` - Release resources

### Step 2: Register in agents.yaml

```yaml
my_agent:
  description: "My custom coding agent"
  adapter_class: "MyAgentCodingAgent"
  adapter_module: "src.execution.coding_agents.adapters.my_agent_agent"
  supports_native_git: false
  default_model: "my-model-v1"
  default_debug_model: "my-model-mini"
  env_vars:
    - "MY_API_KEY"
  install_command: "pip install my-agent-sdk"
  agent_specific:
    timeout: 300
```

Agent is auto-registered on import.

## Configuration via YAML

In benchmark/experiment configs:

```yaml
coding_agent:
  type: gemini
  model: "gemini-2.5-pro"
  agent_specific:
    temperature: 0.3
```

## Factory Methods

| Method | Description |
|--------|-------------|
| `CodingAgentFactory.create(config)` | Create agent from config |
| `CodingAgentFactory.build_config(agent_type, ...)` | Build config with defaults |
| `CodingAgentFactory.list_available()` | List registered agents |
| `CodingAgentFactory.get_default_config(agent_type)` | Get agent defaults |
| `CodingAgentFactory.print_agents_info()` | Print all agents info |

## Design Principles

1. **Separation of Concerns**: Agents generate code. `ExperimentSession` handles git.
2. **Commit Messages**: `CommitMessageGenerator` creates commits from diffs (for non-git agents).
3. **Auto-Discovery**: Factory auto-registers agents from `agents.yaml`.
4. **Graceful Fallback**: Missing dependencies don't break other agents.

