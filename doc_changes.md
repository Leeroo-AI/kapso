# Documentation Changes Required

This document outlines all documentation changes needed to align with the new design in `design.md` and the completed tasks in the `tasks/` directory.

---

## Summary of Design Changes

Based on `design.md` and tasks 01-09, the key architectural changes are:

1. **New API**: `kapso.evolve()` now uses `initial_repo`, `eval_dir`, `data_dir` parameters
2. **Agent-Built Evaluation**: Developer agent builds evaluation dynamically in `kapso_evaluation/`
3. **Feedback Generator**: Replaces predefined evaluators and stop conditions
4. **Removed Components**: `src/environment/evaluators/` and `src/environment/stop_conditions/` deleted
5. **Structured JSON Output**: Developer agent returns structured JSON with results
6. **Feedback in Search Strategy**: Feedback generation moved from orchestrator to search strategy
7. **Benchmark Backward Compatibility**: New `benchmark_tree_search` strategy for MLE/ALE benchmarks

---

## Files to Update

### 1. `docs/evolve/overview.mdx`

**Priority: HIGH**

**Changes Required:**

- [ ] Update the mermaid diagram to remove `Evaluator` and `StopCondition` boxes
- [ ] Replace with `FeedbackGenerator` in the flow
- [ ] Update "How It Works" section:
  - Remove "4. Evaluating results: With configurable evaluators"
  - Remove "5. Iterating: Until stop conditions are met"
  - Add "4. Running evaluation: Agent builds and runs evaluation"
  - Add "5. Generating feedback: Feedback generator validates and decides stop"
- [ ] Remove CardGroup links to `evaluators` and `stop-conditions` pages
- [ ] Add CardGroup link to new `feedback-generator` page
- [ ] Update "Basic Usage" code example:
  ```python
  # OLD
  solution = kapso.evolve(
      goal="...",
      evaluator="regex_pattern",
      evaluator_params={"pattern": r"Accuracy: ([\d.]+)"},
      stop_condition="threshold",
      stop_condition_params={"threshold": 0.9},
  )
  
  # NEW
  solution = kapso.evolve(
      goal="Build a random forest classifier for Iris with accuracy > 0.9",
      eval_dir="./my_evaluation/",  # Optional: user-provided evaluation
      data_dir="./my_data/",        # Optional: user-provided data
      initial_repo=None,            # Optional: starting repo or GitHub URL
      max_iterations=10,
  )
  ```
- [ ] Update "The Experimentation Loop" section to reflect new flow:
  1. Select → Choose which solution candidates to explore
  2. Expand → Generate new variations using the coding agent
  3. Execute → Agent implements solution AND runs evaluation
  4. Feedback → Feedback generator validates and provides guidance
  5. Check → Feedback generator decides stop/continue
- [ ] Remove "Budget Tracking" section about cost-based stop conditions (now handled by feedback generator)

---

### 2. `docs/evolve/architecture.mdx`

**Priority: HIGH**

**Changes Required:**

- [ ] Update main architecture mermaid diagram:
  - Remove `EVAL["Evaluator"]` and `STOP["StopCondition"]` from Environment subgraph
  - Add `FG["FeedbackGenerator"]` to Execution subgraph
  - Update connections to show feedback generator in the flow
- [ ] Update "Component Responsibilities" section
- [ ] Update "Pluggable Components" table:
  - Remove `Evaluator` and `StopCondition` rows
  - Add `FeedbackGenerator` row
- [ ] Update "Configuration Flow" mermaid diagram:
  - Remove `EV["evaluator"]` and `SC["stop_condition"]`
  - Remove `EVF["EvaluatorFactory.create()"]` and `SCF["StopConditionFactory.create()"]`
- [ ] Update "Data Flow" section:
  - Change "6. Problem Handler evaluates code and returns scores" to "6. Developer agent runs evaluation and returns structured JSON"
  - Add "7. Feedback generator validates evaluation and decides stop/continue"
- [ ] Update "Directory Structure" section:
  - Remove `├── evaluators/` and `└── stop_conditions/` from `src/environment/`
  - Add `├── feedback_generator/` to `src/execution/`
  - Add `│   ├── feedback_generator.py`
  - Add `│   └── prompts/feedback_generator.md`
- [ ] Update "Configuration Modes" YAML example to remove evaluator/stop_condition

---

### 3. `docs/evolve/execution-flow.mdx`

**Priority: HIGH**

**Changes Required:**

- [ ] Update "The Big Picture" mermaid diagram:
  - Remove `D[ProblemHandler]` and `E{StopCondition}` 
  - Add `D[Agent Evaluation]` and `E[FeedbackGenerator]`
- [ ] Update "Step 1: Initialize Components" code example:
  ```python
  # OLD
  solution = kapso.evolve(
      goal="Build a classifier with 95% accuracy",
      evaluator="regex_pattern",
      evaluator_params={"pattern": r"Accuracy: ([\d.]+)"},
      stop_condition="threshold",
      stop_condition_params={"threshold": 0.95},
  )
  
  # NEW
  solution = kapso.evolve(
      goal="Build a classifier with 95% accuracy",
      eval_dir="./evaluation/",
      data_dir="./data/",
      max_iterations=10,
  )
  ```
- [ ] Update component table to remove Evaluator/StopCondition, add FeedbackGenerator
- [ ] Update "Step 3: The Solve Loop" code to show new flow with feedback generator
- [ ] Update "Step 5: Write Code" section:
  - Agent now also builds evaluation in `kapso_evaluation/`
  - Agent runs evaluation and returns structured JSON
  - Remove `return self.problem_handler.run(session.session_folder)`
- [ ] Update "Step 6: Run and Evaluate" section:
  - Rename to "Step 6: Agent Runs Evaluation"
  - Show that agent builds and runs evaluation, not ProblemHandler
  - Show structured JSON output format
- [ ] Add new "Step 7: Feedback Generation" section:
  - Explain FeedbackGenerator role
  - Show FeedbackResult structure
  - Explain validation, stop decision, and feedback generation
- [ ] Update "Step 8: Check Stop Condition" to "Step 8: Check Feedback Result"
  - Remove StopCondition table
  - Show `if feedback_result.stop: break` logic
- [ ] Update "Error Handling" section:
  - Update `ProblemRunResult` to `SearchNode` dataclass
  - Show new error fields

---

### 4. `docs/evolve/evaluators.mdx`

**Priority: HIGH - DEPRECATE OR REMOVE**

**Changes Required:**

- [ ] **Option A (Recommended)**: Delete this file entirely
- [ ] **Option B**: Convert to a "Legacy Evaluators" page with deprecation notice:
  - Add prominent deprecation warning at top
  - Explain that evaluators are only used by benchmarks (MLE-Bench, ALE-Bench)
  - Point users to new Feedback Generator documentation
  - Keep content for benchmark users who need it

---

### 5. `docs/evolve/stop-conditions.mdx`

**Priority: HIGH - DEPRECATE OR REMOVE**

**Changes Required:**

- [ ] **Option A (Recommended)**: Delete this file entirely
- [ ] **Option B**: Convert to a "Legacy Stop Conditions" page with deprecation notice:
  - Add prominent deprecation warning at top
  - Explain that stop conditions are only used by benchmarks
  - Point users to new Feedback Generator documentation
  - Keep content for benchmark users who need it

---

### 6. `docs/evolve/orchestrator.mdx`

**Priority: MEDIUM**

**Changes Required:**

- [ ] Update "The Solve Loop" code example:
  ```python
  # OLD
  if self.problem_handler.stop_condition() or budget_progress >= 100:
      break
  
  # NEW (simplified - feedback handled in search strategy)
  node = self.search_strategy.run(context, budget_progress)
  if node and node.should_stop:
      break
  ```
- [ ] Update "Component Creation" section:
  - Remove evaluator/stop_condition creation
  - Add feedback generator creation
- [ ] Update "Constructor Parameters" table:
  - Add `use_feedback_generator: bool` parameter
  - Add `feedback_generator: FeedbackGenerator` parameter
- [ ] Update configuration YAML example to remove evaluator/stop_condition

---

### 7. `docs/evolve/search-strategies.mdx`

**Priority: MEDIUM**

**Changes Required:**

- [ ] Update "Node Structure" to show new `SearchNode` dataclass:
  ```python
  @dataclass
  class SearchNode:
      node_id: int
      parent_node_id: Optional[int] = None
      solution: str = ""
      branch_name: str = ""
      code_changes_summary: str = ""
      evaluation_script_path: str = ""
      evaluation_output: str = ""
      feedback: str = ""
      score: Optional[float] = None
      should_stop: bool = False
      evaluation_valid: bool = True
      had_error: bool = False
      error_message: str = ""
  ```
- [ ] Update "Shared Implementation" section:
  - Remove `return self.problem_handler.run(session.session_folder)` from `implement_solution`
  - Show that agent returns structured JSON
  - Add `_extract_agent_result()` method description
  - Add `_generate_feedback()` method description
- [ ] Add new "Benchmark Tree Search" section:
  - Explain `benchmark_tree_search` strategy for MLE/ALE benchmarks
  - Show that it uses `handler.run()` for evaluation
  - Explain backward compatibility approach

---

### 8. `docs/evolve/coding-agents.mdx`

**Priority: LOW**

**Changes Required:**

- [ ] Add note that coding agents now also serve as feedback generators
- [ ] Update "Agent Interface" to mention structured JSON output requirement
- [ ] Add section on "Structured Output Format":
  ```json
  {
      "code_changes_summary": "Brief description of changes made",
      "evaluation_script_path": "kapso_evaluation/evaluate.py",
      "evaluation_output": "Full output from running evaluation"
  }
  ```

---

### 9. `docs/reference/kapso-api.mdx`

**Priority: HIGH**

**Changes Required:**

- [ ] Update `evolve()` method signature:
  ```python
  def evolve(
      self,
      goal: str,
      eval_dir: Optional[str] = None,      # NEW
      data_dir: Optional[str] = None,      # CHANGED behavior
      initial_repo: Optional[str] = None,  # RENAMED from starting_repo_path
      max_iterations: int = 10,
      # Configuration
      mode: Optional[str] = None,
      coding_agent: Optional[str] = None,
      # REMOVED: language, main_file, timeout
      # REMOVED: evaluator, evaluator_params
      # REMOVED: stop_condition, stop_condition_params
  ) -> SolutionResult
  ```
- [ ] Update parameters table:
  - Add `eval_dir` parameter
  - Update `data_dir` description (now copied to workspace)
  - Rename `starting_repo_path` to `initial_repo`
  - Add note that `initial_repo` accepts GitHub URLs
  - Remove `language`, `main_file`, `timeout` parameters
  - Remove `evaluator`, `evaluator_params` parameters
  - Remove `stop_condition`, `stop_condition_params` parameters
- [ ] Update `SolutionResult` dataclass:
  ```python
  @dataclass
  class SolutionResult:
      goal: str
      code_path: str
      experiment_logs: List[str]
      final_feedback: Optional[FeedbackResult]  # NEW
      metadata: Dict[str, Any]
      
      @property
      def succeeded(self) -> bool:
          """True if goal was achieved."""
      
      @property
      def final_score(self) -> Optional[float]:
          """Final evaluation score if available."""
  ```
- [ ] Update example code to use new API

---

### 10. `docs/quickstart.mdx`

**Priority: HIGH**

**Changes Required:**

- [ ] Update "Run Your First Experiment" Python example:
  ```python
  # OLD
  solution = kapso.evolve(
      goal="Build a random forest classifier for the Iris dataset with accuracy > 0.9",
      output_path="./models/iris_v1",
      evaluator="regex_pattern",
      evaluator_params={"pattern": r"Accuracy: ([\d.]+)"},
      stop_condition="threshold",
      stop_condition_params={"threshold": 0.9},
  )
  
  # NEW
  solution = kapso.evolve(
      goal="Build a random forest classifier for the Iris dataset with accuracy > 0.9",
      max_iterations=10,
  )
  ```
- [ ] Update CLI example to remove `--evaluator` and `--stop-condition` flags
- [ ] Update "Expected Output" section to reflect new flow
- [ ] Update "Understanding the Output" section to show `final_feedback` field

---

### 11. NEW: `docs/evolve/feedback-generator.mdx`

**Priority: HIGH - CREATE NEW FILE**

**Content to Include:**

```markdown
---
title: "Feedback Generator"
description: "How solutions are validated and feedback is generated"
---

## Overview

The Feedback Generator is an LLM-based component that validates evaluation results and decides whether to continue or stop the experimentation loop.

## Responsibilities

1. **Validate Evaluation**: Check if the agent-built evaluation is fair and correct
2. **Check Goal Completion**: Determine if the goal has been achieved
3. **Extract Score**: Parse evaluation output to extract numeric scores
4. **Generate Feedback**: Provide actionable feedback for the next iteration

## FeedbackResult

\`\`\`python
@dataclass
class FeedbackResult:
    stop: bool                      # Whether to stop iteration
    evaluation_valid: bool          # Whether evaluation is fair/correct
    feedback: str                   # Actionable feedback for next iteration
    score: Optional[float] = None   # Extracted evaluation score
\`\`\`

## How It Works

[Diagram and explanation of feedback generation flow]

## Configuration

The feedback generator uses a coding agent (default: claude_code) with a specialized prompt.

## Integration with Search Strategy

Feedback generation happens within the search strategy, not the orchestrator...
```

---

### 12. `docs/benchmarks/mle-bench.mdx` and `docs/benchmarks/ale-bench.mdx`

**Priority: MEDIUM**

**Changes Required:**

- [ ] Add note about `benchmark_tree_search` strategy
- [ ] Explain that benchmarks use legacy evaluation flow with `handler.run()`
- [ ] Update config examples to show `type: "benchmark_tree_search"`

---

## Files to Delete (or Mark as Deprecated)

1. `docs/evolve/evaluators.mdx` - No longer applicable to main API
2. `docs/evolve/stop-conditions.mdx` - No longer applicable to main API

---

## Navigation Updates

### `docs.json` or Navigation Config

- [ ] Remove `evaluators` and `stop-conditions` from evolve section
- [ ] Add `feedback-generator` to evolve section
- [ ] Update order of pages to reflect new flow

---

## Code Examples to Update Throughout

All documentation code examples using the old API pattern need updating:

**Old Pattern:**
```python
kapso.evolve(
    goal="...",
    evaluator="regex_pattern",
    evaluator_params={...},
    stop_condition="threshold",
    stop_condition_params={...},
)
```

**New Pattern:**
```python
kapso.evolve(
    goal="...",
    eval_dir="./evaluation/",  # Optional
    data_dir="./data/",        # Optional
    initial_repo=None,         # Optional: path or GitHub URL
    max_iterations=10,
)
```

---

## Diagrams to Update

1. **Overview flow diagram** - Remove Evaluator/StopCondition, add FeedbackGenerator
2. **Architecture diagram** - Update component relationships
3. **Execution flow diagram** - Show new agent-evaluation-feedback flow
4. **Search strategy diagrams** - Show feedback integration

---

## Terminology Changes

| Old Term | New Term |
|----------|----------|
| `evaluator` | Agent-built evaluation |
| `stop_condition` | Feedback generator decision |
| `starting_repo_path` | `initial_repo` |
| `ProblemRunResult` | `SearchNode` |
| `ExperimentResult` | `SearchNode` |

---

## Priority Order for Updates

1. **HIGH**: `quickstart.mdx`, `kapso-api.mdx`, `overview.mdx` - User-facing entry points
2. **HIGH**: `execution-flow.mdx`, `architecture.mdx` - Core understanding
3. **HIGH**: Create `feedback-generator.mdx` - New component documentation
4. **HIGH**: Delete/deprecate `evaluators.mdx`, `stop-conditions.mdx`
5. **MEDIUM**: `orchestrator.mdx`, `search-strategies.mdx` - Implementation details
6. **MEDIUM**: Benchmark documentation updates
7. **LOW**: `coding-agents.mdx` - Minor updates

---

## Testing Documentation Changes

After updates, verify:

1. All code examples are syntactically correct
2. All mermaid diagrams render properly
3. All internal links work
4. Navigation reflects new structure
5. No references to removed components (evaluators, stop_conditions)
