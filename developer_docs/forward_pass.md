# Forward Pass - Execution Engine

The execution engine runs experiments to solve problems through iterative code generation.

## Overview

```
OrchestratorAgent.solve()
    │
    ├── ContextManager.get_context()     ─► Gather problem + history + knowledge
    │
    └── SearchStrategy.run()             ─► Generate solutions, implement, evaluate
            │
            ├── expand()                 ─► Generate new solution candidates (LLM)
            ├── select()                 ─► Pick best nodes to experiment (LLM)
            └── _implement_n_debug()     ─► Code generation loop
                    │
                    ├── ExperimentSession.generate_code()
                    │       └── CodingAgent.generate_code()
                    │
                    └── ProblemHandler.run()  ─► Evaluate and score
```

## Core Components

| Component | Role |
|-----------|------|
| `OrchestratorAgent` | Main coordinator. Loads config, creates components, runs solve loop. |
| `SearchStrategy` | Explores solution space. Tree-based with LLM guidance. |
| `ContextManager` | Gathers context from problem, history, and knowledge graph. |
| `ExperimentWorkspace` | Manages git workspace and experiment sessions. |
| `ExperimentSession` | Isolated environment for single experiment (git branch + agent). |
| `CodingAgent` | Generates code (Aider, Gemini, Claude Code, OpenHands). |
| `ProblemHandler` | Runs and evaluates solutions. Returns score. |

## Solve Loop

```python
# OrchestratorAgent.solve()
for iteration in range(max_iterations):
    # 1. Check stopping conditions
    if problem_handler.stop_condition() or budget >= 100:
        break
    
    # 2. Gather context
    context = context_manager.get_context(budget_progress)
    
    # 3. Run one search iteration
    search_strategy.run(context, budget_progress)
    
    # 4. Save checkpoint
    search_strategy.export_checkpoint()
```

## Search Strategy: LLM Tree Search

The default strategy (`llm_tree_search`) uses a tree to explore solutions:

### One Iteration

```
run(context, budget_progress)
    │
    ├── prune_bad_solutions()      # Remove unpromising nodes (if budget > 20%)
    │
    ├── expand(context)            # Generate new solution candidates
    │       │
    │       ├── select()           # Pick nodes to expand (LLM-guided)
    │       └── solution_generation()  # Generate child solutions (LLM)
    │
    ├── select()                   # Pick best nodes to experiment
    │
    └── [parallel] _run_for_node() # Run experiments for each selected node
            │
            └── _implement_n_debug()
```

### Tree Structure

```
Root (empty)
├── Node 1: "Use XGBoost with feature engineering"
│   ├── Node 4: "Add target encoding"
│   └── Node 5: "Add time-based features"
├── Node 2: "Use neural network approach"
│   └── Node 6: "Add dropout regularization"
└── Node 3: "Ensemble of multiple models"
```

Each node has:
- `solution`: Description of the approach
- `experiment_result`: Score, error, output from execution
- `is_terminated`: Pruned as unpromising
- `branch_name`: Git branch with implementation

## Experiment Execution

### _implement_n_debug Loop

```python
def _implement_n_debug(solution, context, code_debug_tries, branch_name, parent_branch):
    # Create isolated session (clones repo, creates branch)
    session = workspace.create_experiment_session(branch_name, parent_branch)
    
    # Generate initial implementation
    result = implement_solution(solution, context, session)
    
    # Debug loop
    for i in range(code_debug_tries):
        if result.run_had_error and result.continue_debugging:
            result = debug_solution(solution, context, result.error_details, session)
        else:
            break
    
    # Cleanup
    workspace.finalize_session(session)
    return result
```

### ExperimentSession

Each experiment runs in an isolated git branch:

```python
session = ExperimentSession(
    main_repo=repo,
    session_folder="/tmp/sessions/experiment_0",
    coding_agent_config=config,
    parent_branch_name="experiment_3",  # Inherit parent's code
    branch_name="experiment_7"
)

# Generate code
result = session.generate_code(prompt)

# Session handles:
# - Git clone to isolated folder
# - Branch creation from parent
# - Coding agent initialization
# - Commit handling (for non-git agents)
# - Push and cleanup on close
```

## Context Gathering

`ContextManager` builds context for solution generation:

```python
context = ContextData(
    problem="...",           # From ProblemHandler
    additional_info="...",   # Experiment history (top + recent)
    kg_results="...",        # Knowledge graph search results
    kg_code_results="..."    # Code snippets from KG
)
```

Sources:
1. **Problem**: From `ProblemHandler.get_problem_context()`
2. **History**: Top experiments (by score) + recent experiments
3. **Knowledge**: From `KnowledgeSearch.search()` (KG, RAG, etc.)

## Configuration

### Via OrchestratorAgent

```python
orchestrator = OrchestratorAgent(
    problem_handler=handler,
    config_path="benchmarks/mle/config.yaml",
    mode="PRODUCTION",
    coding_agent="claude_code",
    is_kg_active=True,
)

result = orchestrator.solve(
    experiment_max_iter=20,
    time_budget_minutes=60,
    cost_budget=50.0
)
```

### Config YAML Structure

```yaml
modes:
  PRODUCTION:
    # Search strategy
    search_strategy:
      type: llm_tree_search
      params:
        code_debug_tries: 5
        node_expansion_limit: 2
        exploration_budget_percent: 30
    
    # Coding agent
    coding_agent:
      type: claude_code
      model: claude-sonnet-4-20250514
    
    # Context manager
    context_manager:
      type: kg_enriched
      params:
        max_experiment_history_count: 5
    
    # Knowledge search
    knowledge_search:
      type: kg_llm_navigation
      enabled: true
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     OrchestratorAgent.solve()                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ ContextMgr   │───►│ SearchStrategy │───►│ ExperimentResult │  │
│  │              │    │                │    │                  │  │
│  │ - problem    │    │ - expand()     │    │ - solution       │  │
│  │ - history    │    │ - select()     │    │ - score          │  │
│  │ - kg_results │    │ - experiment() │    │ - branch_name    │  │
│  └──────────────┘    └───────────────┘    └──────────────────┘  │
│         ▲                    │                                   │
│         │                    ▼                                   │
│         │           ┌───────────────┐                            │
│         │           │ ExperimentSession                          │
│         │           │                │                           │
│         │           │ ┌─────────────┐│                           │
│         │           │ │CodingAgent  ││                           │
│         │           │ │ generate()  ││                           │
│         │           │ └─────────────┘│                           │
│         │           │       │        │                           │
│         │           │       ▼        │                           │
│         │           │ ┌─────────────┐│                           │
│         └───────────┤ │ProblemHandler                            │
│                     │ │ run()       ││                           │
│                     │ └─────────────┘│                           │
│                     └───────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `src/execution/orchestrator.py` | Main OrchestratorAgent class |
| `src/execution/search_strategies/llm_tree_search.py` | Tree-based search |
| `src/execution/search_strategies/base.py` | SearchStrategy base + ExperimentResult |
| `src/execution/context_manager/kg_enriched_context_manager.py` | Context gathering |
| `src/execution/experiment_workspace/experiment_workspace.py` | Git workspace management |
| `src/execution/experiment_workspace/experiment_session.py` | Single experiment session |
| `src/execution/coding_agents/` | Pluggable coding agents |

## Budget & Stopping

The loop tracks three budget dimensions:

```python
budget_progress = max(
    time_elapsed / time_budget,      # Time budget
    iteration / max_iterations,       # Iteration budget
    cumulative_cost / cost_budget     # Cost budget
) * 100

if budget_progress >= 100 or problem_handler.stop_condition():
    break
```

Budget affects behavior:
- `budget < 20%`: Exploration phase (expand diverse solutions)
- `budget >= 20%`: Start pruning unpromising solutions
- `budget >= exploration_budget_percent`: Switch to exploitation (expand top solutions only)

