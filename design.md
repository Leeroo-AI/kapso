# Kapso Evolve System - Design Architecture

## Overview

The Evolve system iteratively builds solutions for a given goal using a developer agent (coding agent) that autonomously implements, evaluates, and improves code based on feedback.

## API

```python
kapso.evolve(
    goal: str,                      # High-level objective
    eval_dir: Optional[str],        # User-provided evaluation files
    data_dir: Optional[str],        # User-provided datasets
    initial_repo: Optional[str],    # User-provided starting repository
    max_iterations: int = 10,       # Maximum iteration limit
)
```

## Architecture Flow

```
evolve(goal, eval_dir?, data_dir?, initial_repo?)
    │
    ├─► 1. INITIALIZE REPO
    │       if initial_repo provided:
    │           use it as base
    │       else:
    │           workflow_search(goal) → find related workflow
    │           if workflow found: init repo from workflow
    │           else: create empty repo
    │
    ├─► 2. SETUP DIRECTORIES
    │       Copy eval_dir → repo/kapso_evaluation/
    │       Copy data_dir → repo/kapso_datasets/
    │       Build repo_memory for workspace understanding
    │
    ├─► 3. DEVELOPER AGENT LOOP
    │       for each iteration:
    │           │
    │           ├─► Developer agent (coding agent):
    │           │     - Reads repo_memory to understand codebase
    │           │     - Implements solution for the goal
    │           │     - Builds evaluation in kapso_evaluation/
    │           │     - Runs evaluation
    │           │     - Reports result
    │           │     - Handles retries if evaluation crashes
    │           │
    │           └─► Feedback generator:
    │                 Input: idea, implementation, evaluation_result
    │                 Actions:
    │                   - Validates evaluation is fair/correct
    │                   - If goal reached: STOP
    │                   - Else: generate feedback for next iteration
    │
    └─► 4. RETURN RESULT
            Final repo with solution
```

## Directory Structure

After setup, the workspace looks like:

```
repo/
  ├── kapso_evaluation/     # Evaluation code and assets
  │     ├── (user files)    # Copied from eval_dir if provided
  │     └── (agent files)   # Agent builds evaluation here
  │
  ├── kapso_datasets/       # Data files
  │     └── (user files)    # Copied from data_dir if provided
  │
  ├── .kapso/
  │     └── repo_memory.json  # Codebase understanding
  │
  └── (solution code)       # From workflow or agent-built
```

## Components

### 1. Initialize Repo

**Purpose:** Create the starting point for development.

**Logic:**
```
if initial_repo is provided:
    clone/copy initial_repo as workspace
else:
    search for relevant workflow using workflow_search(goal)
    if workflow found:
        initialize repo from workflow template
    else:
        create empty repo
```

**Workflow Search:** Uses `src/knowledge/search/workflow_search.py` to find domain-relevant workflows (e.g., "triton_kernel_dev", "dpo_alignment", "langgraph_agents").

### 2. Setup Directories

**Purpose:** Prepare evaluation and data directories.

**Actions:**
- Create `kapso_evaluation/` directory
- If `eval_dir` provided: copy contents to `kapso_evaluation/`
- Create `kapso_datasets/` directory
- If `data_dir` provided: copy contents to `kapso_datasets/`
- Build initial `repo_memory` for the workspace

### 3. Developer Agent

**Purpose:** Implement solution and evaluation, run evaluation, report results.

**Type:** Coding agent (e.g., Claude Code, Aider)

**Responsibilities:**
1. Read `repo_memory` to understand workspace structure
2. Implement solution code for the goal
3. Build evaluation pipeline in `kapso_evaluation/`
4. Run evaluation and capture results
5. Report structured evaluation result
6. Handle evaluation crashes with retries (instruction-limited)

**Instruction Prompt Guidelines:**
- Build solution incrementally
- Create fair, correct evaluation
- Run evaluation and report result before completing iteration
- Retry evaluation if it crashes (max N attempts)
- Don't run excessive evaluations per iteration
- Be aware of `kapso_datasets/` location for data access

**Node Completion:** A node/iteration is complete when the agent has run evaluation and reported the result.

### 4. Feedback Generator

**Purpose:** Validate evaluation, decide stop/continue, generate improvement feedback.

**Type:** LLM-based coding agent with instruction prompt

**Input:**
```python
{
    "goal": str,                    # Original goal
    "idea": str,                    # Solution approach for this iteration
    "implementation": str,          # Code changes made
    "evaluation_code": str,         # Evaluation pipeline code
    "evaluation_result": dict,      # Result from running evaluation
}
```

**Responsibilities:**
1. **Validate evaluation:** Check if evaluation is fair and correct
   - If evaluation is flawed: generate feedback to fix evaluation
2. **Check goal completion:** Determine if goal is achieved
   - If goal reached: return STOP signal
3. **Generate feedback:** If not done, provide actionable feedback for next iteration

**Output:**
```python
{
    "stop": bool,                   # True if goal achieved
    "evaluation_valid": bool,       # True if evaluation is sound
    "feedback": str,                # Improvement suggestions (if not stopping)
}
```

## Iteration Flow Detail

```
┌─────────────────────────────────────────────────────────────┐
│                     ITERATION N                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Developer Agent receives:                                │
│     - Goal                                                   │
│     - Previous feedback (if N > 1)                          │
│     - repo_memory                                           │
│                                                              │
│  2. Developer Agent actions:                                 │
│     - Analyze feedback                                       │
│     - Modify/create solution code                           │
│     - Modify/create evaluation in kapso_evaluation/         │
│     - Run evaluation                                         │
│     - Report: {metrics, logs, artifacts}                    │
│                                                              │
│  3. Feedback Generator receives:                             │
│     - Goal                                                   │
│     - Implementation diff                                    │
│     - Evaluation code                                        │
│     - Evaluation result                                      │
│                                                              │
│  4. Feedback Generator decides:                              │
│     - Is evaluation valid? (if no → feedback to fix eval)   │
│     - Is goal achieved? (if yes → STOP)                     │
│     - Generate feedback for iteration N+1                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Example Use Cases

### Use Case 1: GPU Kernel Optimization

```python
kapso.evolve(
    goal="Optimize PyTorch MHA kernel using Triton. Target: 3x speedup.",
    eval_dir="./kernelbench_eval/",
    data_dir="./kernelbench_ops/",
)
```

**Flow:**
1. Init: workflow_search → "triton_kernel_dev" workflow
2. Setup: Copy benchmark harness to `kapso_evaluation/`, ops to `kapso_datasets/`
3. Iteration 1: Agent writes Triton kernel, builds speedup benchmark, reports 1.3x
4. Feedback: "Try larger tile sizes"
5. Iteration 2: Agent adjusts tiles, reports 2.4x
6. Feedback: "Optimize thread configuration"
7. Iteration 3: Agent optimizes threads, reports 3.6x
8. Feedback: "3.6x > 3x target. STOP."

### Use Case 2: LLM Fine-tuning (DPO)

```python
kapso.evolve(
    goal="Fine-tune Llama-3.1-8B with DPO for legal risk detection. Target: 40% improvement.",
    eval_dir="./legal_eval/",
    data_dir="./cuad_dataset/",
)
```

**Flow:**
1. Init: workflow_search → "dpo_alignment" workflow
2. Setup: Copy ContractEval to `kapso_evaluation/`, CUAD to `kapso_datasets/`
3. Iteration 1: Agent sets up DPO training, builds evaluation, reports 15% improvement
4. Feedback: "Add precision metric to evaluation, increase epochs"
5. Iteration 2: Agent adjusts, reports 28% improvement
6. Feedback: "Try adjusting DPO beta parameter"
7. Iteration 3: Agent adjusts beta, reports 44% improvement
8. Feedback: "44% > 40% target. STOP."

### Use Case 3: Multi-Agent System

```python
kapso.evolve(
    goal="Build multi-agent recruitment system with HITL and trace visualization.",
    eval_dir="./recruitment_eval/",
    data_dir="./matrix_profiles/",
)
```

**Flow:**
1. Init: workflow_search → "langgraph_agents" workflow
2. Setup: Copy test scenarios to `kapso_evaluation/`, profiles to `kapso_datasets/`
3. Iteration 1: Agent builds 3 agents, evaluation checks functionality, reports partial success
4. Feedback: "Fix coordinator, add HITL"
5. Iteration 2: Agent fixes coordinator, adds HITL, reports progress
6. Feedback: "Add trace visualization"
7. Iteration 3: Agent adds tracing, reports all requirements met
8. Feedback: "All requirements met. STOP."

## Error Handling

### Evaluation Crashes
- Developer agent retries evaluation (instruction-limited, e.g., max 3 retries)
- If still failing: report error as evaluation result
- Feedback generator can provide guidance to fix evaluation

### No Workflow Found
- Falls back to empty repo
- Developer agent builds everything from scratch

### Goal Not Achievable
- After max_iterations: return best result achieved
- Feedback generator can signal "FAIL" if evaluation is fundamentally broken

## Key Design Decisions

1. **Agent-built evaluation:** No predefined evaluators. Agent creates domain-appropriate evaluation.

2. **Feedback generator validates evaluation:** Prevents gaming (e.g., `print("SCORE: 1.0")`).

3. **Flat directory structure:** `kapso_evaluation/` and `kapso_datasets/` as direct subdirs of repo.

4. **repo_memory for context:** Agent uses repo_memory to understand large codebases/datasets.

5. **LLM-based feedback generator:** Handles diverse domains without domain-specific rules.

6. **Workflow bootstrap:** Gives agent a head start with relevant templates.

## Future Considerations

- **Evaluation rejection state:** Force agent to rebuild if feedback generator rejects evaluation
- **Dataset README requirement:** Require `kapso_datasets/README.md` for large datasets
- **Resource constraints:** Handle GPU/memory requirements in execution environment
- **Parallel iterations:** Run multiple solution branches concurrently
