# Pluggable Coding Agents

This module provides a **plug-and-play architecture** for integrating different coding agents into the ML Kapso system. The orchestrator can seamlessly switch between agents like Aider, Gemini, Claude Code, and OpenHands.

## Quick Start

### Using an Agent

```python
from kapso.agents.coding_agents import CodingAgentFactory, CodingAgentConfig

# List available agents
print(CodingAgentFactory.list_available())
# Output: ['aider', 'claude_code', 'gemini', 'openhands']

# Create agent with defaults from agents.yaml
config = CodingAgentFactory.build_config(agent_type="aider")
agent = CodingAgentFactory.create(config)

# Or specify custom settings
config = CodingAgentConfig(
    agent_type="gemini",
    model="gemini-2.5-pro",
    debug_model="gemini-2.5-flash",
    workspace="/path/to/workspace",
    agent_specific={"temperature": 0.5}
)
agent = CodingAgentFactory.create(config)
```

### From Command Line

```bash
# Run with specific agent
python -m benchmarks.mle.runner -c competition-name -d aider
python -m benchmarks.mle.runner -c competition-name -d gemini
python -m benchmarks.mle.runner -c competition-name -d claude_code

# List available agents
python -m benchmarks.mle.runner --list-agents
```

## Available Agents

| Agent | Description | Native Git | Install |
|-------|-------------|------------|---------|
| **aider** | Git-centric pair programming with diff editing | ✅ | `pip install aider-chat` |
| **gemini** | Google Gemini SDK for code generation | ❌ | `pip install google-generativeai` |
| **claude_code** | Anthropic Claude Code CLI for complex refactoring | ❌ | `npm install -g @anthropic-ai/claude-code` |
| **openhands** | OpenHands agent with sandboxed execution | ❌ | `pip install openhands-ai litellm` |

## Architecture

```
src/agents/coding_agents/
├── __init__.py              # Package exports
├── agents.yaml              # ⭐ Central agent registry
├── base.py                  # CodingAgentInterface, CodingResult, CodingAgentConfig
├── factory.py               # CodingAgentFactory with auto-discovery
├── commit_message_generator.py  # LLM-based commit messages
├── README.md                # This file
└── adapters/
    ├── __init__.py
    ├── TEMPLATE.py          # ⭐ Template for new agents
    ├── aider_agent.py
    ├── gemini_agent.py
    ├── claude_code_agent.py
    └── openhands_agent.py
```

## Adding a New Agent

### Step 1: Create Adapter

Copy the template and implement your agent:

```bash
cd src/agents/coding_agents/adapters
cp TEMPLATE.py my_agent_agent.py
```

Edit `my_agent_agent.py`:

```python
from kapso.agents.coding_agents.base import (
    CodingAgentInterface,
    CodingAgentConfig,
    CodingResult,
)

class MyAgentCodingAgent(CodingAgentInterface):
    """My custom coding agent."""
    
    def __init__(self, config: CodingAgentConfig):
        super().__init__(config)
        self.workspace = None
        # Extract agent-specific config
        self._my_option = config.agent_specific.get("my_option", "default")
    
    def initialize(self, workspace: str) -> None:
        """Set up client/connection. Do NOT do git operations here."""
        self.workspace = workspace
        # Initialize your client
        self.client = MyClient(api_key=os.getenv("MY_API_KEY"))
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """Generate code. Do NOT commit - ExperimentSession handles that."""
        model = self.config.debug_model if debug_mode else self.config.model
        
        try:
            response = self.client.generate(prompt, model=model)
            files = self._write_files(response)
            
            self._cumulative_cost += response.cost
            
            return CodingResult(
                success=True,
                output=response.text,
                files_changed=files,
                cost=response.cost,
            )
        except Exception as e:
            return CodingResult(success=False, output="", error=str(e))
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.client = None
        self.workspace = None
    
    def supports_native_git(self) -> bool:
        """Return False - let ExperimentSession handle commits."""
        return False
```

### Step 2: Register in agents.yaml

Add entry to `src/agents/coding_agents/agents.yaml`:

```yaml
my_agent:
  description: "My custom coding agent"
  adapter_class: "MyAgentCodingAgent"
  adapter_module: "kapso.agents.coding_agents.adapters.my_agent_agent"
  supports_native_git: false
  default_model: "my-model-v1"
  default_debug_model: "my-model-v1-mini"
  env_vars:
    - "MY_API_KEY"
  install_command: "pip install my-agent-sdk"
  documentation_url: "https://docs.my-agent.com"
  agent_specific:
    my_option: "default_value"
    timeout: 300
```

### Step 3: Done!

Your agent is now auto-registered and available:

```bash
python -m benchmarks.mle.runner -c competition -d my_agent
```

## Key Interfaces

### CodingAgentInterface

All agents must implement this abstract class:

```python
class CodingAgentInterface(ABC):
    def __init__(self, config: CodingAgentConfig): ...
    
    @abstractmethod
    def initialize(self, workspace: str) -> None: ...
    
    @abstractmethod
    def generate_code(self, prompt: str, debug_mode: bool) -> CodingResult: ...
    
    @abstractmethod
    def cleanup(self) -> None: ...
    
    def supports_native_git(self) -> bool: ...
    def get_cumulative_cost(self) -> float: ...
    def get_capabilities(self) -> Dict[str, bool]: ...
```

### CodingResult

Standardized result from code generation:

```python
@dataclass
class CodingResult:
    success: bool                    # Whether generation succeeded
    output: str                      # Agent's response text
    files_changed: List[str] = []    # Modified file paths
    error: Optional[str] = None      # Error message if failed
    cost: float = 0.0                # API cost in dollars
    commit_message: Optional[str] = None  # Suggested commit message
    metadata: Dict[str, Any] = {}    # Agent-specific data
```

### CodingAgentConfig

Configuration passed to agents:

```python
@dataclass
class CodingAgentConfig:
    agent_type: str        # "aider", "gemini", etc.
    model: str             # Primary model
    debug_model: str       # Debug/fix model
    workspace: str         # Working directory
    use_git: bool = True   # Enable git integration
    agent_specific: Dict[str, Any] = {}  # Agent-specific options
```

## Configuration

### In Benchmark Config (benchmarks/*/config.yaml)

```yaml
modes:
  MY_MODE:
    # ... other settings ...
    
    # Specify agent (uses defaults from agents.yaml)
    coding_agent:
      type: aider
    
    # Or override defaults
    coding_agent:
      type: gemini
      model: "gemini-2.5-pro"
      agent_specific:
        temperature: 0.3
```

### Environment Variables

Each agent requires specific environment variables:

| Agent | Required Variables |
|-------|-------------------|
| aider | `OPENAI_API_KEY` |
| gemini | `GOOGLE_API_KEY` |
| claude_code | `ANTHROPIC_API_KEY` |
| openhands | `OPENAI_API_KEY` |

## Design Principles

### 1. Agents Only Generate Code

Agents should **NOT** handle git operations. The `ExperimentSession` class manages:
- Cloning repos
- Creating branches
- Committing changes
- Pushing to remote

Exception: Aider has `supports_native_git() = True` because it auto-commits.

### 2. Commit Messages are Generated

For agents without native git, `CommitMessageGenerator` uses an LLM to create meaningful commit messages from:
- Code diff
- Solution summary
- Agent's optional suggestion

### 3. Factory Pattern with Auto-Discovery

The factory automatically discovers agents from `agents.yaml`:
- No manual registration needed
- Missing dependencies are handled gracefully
- Custom agents can still register at runtime

## Troubleshooting

### Agent Not Available

```
[CodingAgentFactory] gemini not available (import): No module named 'google.generativeai'
```

**Fix:** Install the required package:
```bash
pip install google-generativeai
```

### Unknown Agent Type

```
ValueError: Unknown coding agent: 'my_agent'. Available agents: aider, claude_code, gemini, openhands
```

**Fix:** Check `agents.yaml` entry and adapter file exist.

### API Key Missing

```
ValueError: GOOGLE_API_KEY environment variable not set
```

**Fix:** Set the environment variable in `.env`:
```
GOOGLE_API_KEY=your_key_here
```

