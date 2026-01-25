# Task 3: Developer Agent Loop

## Design Reference
From `design.md` lines 37-53:
```
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
```

## Current Implementation

### What Exists

1. **`src/execution/orchestrator.py` - `solve()` method (lines 275-349)**
   - Main loop: `for i in range(experiment_max_iter)`
   - Checks `problem_handler.stop_condition()` - predefined stop logic
   - Checks `context_manager.should_stop()` - LLM decision
   - Calls `search_strategy.run()` for each iteration

2. **`src/execution/search_strategies/linear_search.py` - `run()` method**
   - `_generate_solution()` - ideates solution
   - `_implement_n_debug()` - implements and debugs
   - Stores `ExperimentResult` with score

3. **`src/execution/search_strategies/base.py`**
   - `implement_solution()` - generates code via coding agent
   - `debug_solution()` - fixes errors
   - `_implement_n_debug()` - full implement + debug loop
   - Uses `problem_handler.run()` to execute and evaluate

4. **`src/environment/handlers/generic.py` - `GenericProblemHandler`**
   - `run()` - executes code and calls evaluator
   - `stop_condition()` - checks predefined stop condition
   - Uses `Evaluator` and `StopCondition` classes

5. **`src/environment/evaluators/`**
   - `ScriptEvaluator` - runs agent-written evaluate.py
   - `RegexPatternEvaluator`, `LLMJudgeEvaluator`, etc.
   - Factory pattern for creating evaluators

6. **`src/environment/stop_conditions/`**
   - `ThresholdStopCondition`, `FromEvalStopCondition`, etc.
   - Factory pattern for creating stop conditions

7. **`src/execution/prompts/coding_agent_implement.md`**
   - Current prompt for coding agent
   - Tells agent to implement solution

### Changes Required

#### DELETE

1. **`src/environment/evaluators/`** - ENTIRE DIRECTORY
   - No longer needed - agent builds evaluation dynamically
   - Files: `base.py`, `builtin.py`, `factory.py`, `__init__.py`

2. **`src/environment/stop_conditions/`** - ENTIRE DIRECTORY
   - No longer needed - feedback generator decides stop
   - Files: `base.py`, `builtin.py`, `factory.py`, `__init__.py`

3. **`src/environment/handlers/generic.py`**
   - Remove evaluator integration
   - Remove stop_condition integration
   - Simplify to just execute code

#### MODIFY

1. **`src/execution/orchestrator.py` - `solve()` method**
   - Remove `problem_handler.stop_condition()` check
   - Remove `context_manager.should_stop()` check
   - Add feedback generator call after each iteration
   - New loop structure:
     ```python
     for i in range(experiment_max_iter):
         # Developer agent iteration
         result = self.search_strategy.run(context, budget_progress)
         
         # Feedback generator
         feedback_result = self.feedback_generator.generate(
             goal=self.goal,
             idea=result.solution,
             implementation=result.code_diff,
             evaluation_result=result.evaluation_output,
         )
         
         if feedback_result.stop:
             break
         
         # Pass feedback to next iteration
         context.feedback = feedback_result.feedback
     ```

2. **`src/execution/search_strategies/base.py`**
   - Remove `problem_handler.run()` calls for evaluation
   - Agent is now responsible for running evaluation
   - `implement_solution()` should:
     - Generate code for solution AND evaluation
     - Run evaluation (agent does this)
     - Return evaluation output

3. **`src/execution/search_strategies/linear_search.py`**
   - Update `run()` to match new flow
   - Return evaluation result from agent

4. **`src/execution/prompts/coding_agent_implement.md`**
   - Major rewrite needed
   - New responsibilities:
     - Implement solution
     - Build evaluation in `kapso_evaluation/`
     - Run evaluation
     - Report result in structured format
   - Include retry instructions for evaluation crashes

5. **`src/environment/handlers/generic.py`**
   - Simplify significantly
   - Remove evaluator/stop_condition logic
   - Keep only: problem description, execution utilities

6. **`src/environment/handlers/base.py`**
   - Remove `stop_condition()` abstract method
   - Simplify interface

#### ADD

1. **`src/execution/feedback_generator.py`** - NEW FILE
   ```python
   class FeedbackGenerator:
       """
       LLM-based feedback generator.
       
       Validates evaluation and decides stop/continue.
       """
       
       def __init__(self, coding_agent_config: CodingAgentConfig):
           self.agent = CodingAgentFactory.create(coding_agent_config)
       
       def generate(
           self,
           goal: str,
           idea: str,
           implementation: str,
           evaluation_code: str,
           evaluation_result: dict,
       ) -> FeedbackResult:
           """
           Generate feedback for the iteration.
           
           Returns:
               FeedbackResult with stop flag and feedback text
           """
           # Use coding agent (like Claude Code) with instruction prompt
           # to analyze and generate feedback
           pass
   ```

2. **`src/execution/prompts/feedback_generator.md`** - NEW FILE
   - Instruction prompt for feedback generator
   - Responsibilities:
     - Validate evaluation is fair/correct
     - Check if goal is achieved
     - Generate actionable feedback

3. **`src/execution/prompts/coding_agent_implement.md`** - REWRITE
   - New prompt structure for developer agent
   - Include:
     - Goal and context
     - Previous feedback (if any)
     - Instructions to build evaluation
     - Instructions to run evaluation
     - Retry logic for crashes
     - Output format for evaluation result

### New Iteration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     ITERATION N                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Developer Agent (coding agent like Claude Code):         │
│     Input:                                                   │
│       - Goal                                                 │
│       - repo_memory                                          │
│       - Previous feedback (if N > 1)                        │
│       - kapso_evaluation/ contents                          │
│       - kapso_datasets/ location                            │
│                                                              │
│     Actions:                                                 │
│       - Implement/modify solution code                      │
│       - Build/modify evaluation in kapso_evaluation/        │
│       - Run evaluation (with retries if crash)              │
│       - Output structured result                            │
│                                                              │
│     Output:                                                  │
│       - Code changes (git diff)                             │
│       - Evaluation code                                      │
│       - Evaluation result (structured JSON)                 │
│                                                              │
│  2. Feedback Generator (LLM-based):                         │
│     Input:                                                   │
│       - Goal                                                 │
│       - Idea/solution description                           │
│       - Implementation (code diff)                          │
│       - Evaluation code                                      │
│       - Evaluation result                                    │
│                                                              │
│     Actions:                                                 │
│       - Validate evaluation fairness                        │
│       - Check goal completion                               │
│       - Generate feedback                                    │
│                                                              │
│     Output:                                                  │
│       - stop: bool                                          │
│       - evaluation_valid: bool                              │
│       - feedback: str                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Files to Touch

**Delete:**
- `src/environment/evaluators/base.py`
- `src/environment/evaluators/builtin.py`
- `src/environment/evaluators/factory.py`
- `src/environment/evaluators/__init__.py`
- `src/environment/stop_conditions/base.py`
- `src/environment/stop_conditions/builtin.py`
- `src/environment/stop_conditions/factory.py`
- `src/environment/stop_conditions/__init__.py`

**Modify:**
- `src/execution/orchestrator.py`
- `src/execution/search_strategies/base.py`
- `src/execution/search_strategies/linear_search.py`
- `src/execution/search_strategies/llm_tree_search.py`
- `src/environment/handlers/generic.py`
- `src/environment/handlers/base.py`
- `src/environment/handlers/__init__.py`
- `src/execution/prompts/coding_agent_implement.md`
- `src/execution/prompts/coding_agent_debug.md`

**Add:**
- `src/execution/feedback_generator.py`
- `src/execution/prompts/feedback_generator.md`

### Cross-References
- Depends on: `01_initialize_repo.md`, `02_setup_directories.md`
- Related to: `04_return_result.md`

### Testing

#### Test Data
Continues from `01_initialize_repo.md` using: `data/wikis_llm_finetuning_test/`

This contains:
- Workflow: `Jaymody_PicoGPT_Text_Generation.md`
- GitHub URL: `https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation`
- Domains: LLMs, Inference, Education
- Task: Text generation using GPT-2 with NumPy

#### Test Scenario: End-to-End PicoGPT Enhancement

**Goal:** Enhance PicoGPT to support temperature-based sampling instead of greedy decoding.

This is a good test because:
- Clear, measurable goal
- Requires understanding existing code (repo_memory)
- Requires building evaluation (compare greedy vs sampling output diversity)
- Has clear success criteria

#### Test Cases

##### Test 1: Developer Agent Builds Correct Evaluation

```python
# Setup: Continue from 01_initialize_repo.md test
# Workspace already has PicoGPT cloned and kapso_evaluation/ created

# Goal for this test
goal = """
Enhance PicoGPT to support temperature-based sampling.
Currently it uses greedy decoding (argmax). Add a temperature parameter
that controls randomness in token selection.

Success criteria:
- temperature=0.0 should behave like greedy (deterministic)
- temperature=1.0 should produce varied outputs for same prompt
- Measure output diversity across 5 runs with same prompt
"""

# Run one iteration
result = orchestrator.search_strategy.run(context, budget_progress=0)

# Verify: Agent should have created evaluation in kapso_evaluation/
eval_dir = os.path.join(workspace, "kapso_evaluation")
assert os.path.exists(os.path.join(eval_dir, "evaluate.py")) or \
       os.path.exists(os.path.join(eval_dir, "run_eval.py"))

# Verify: Evaluation code should measure diversity
eval_code = open(os.path.join(eval_dir, "evaluate.py")).read()
assert "diversity" in eval_code.lower() or "unique" in eval_code.lower()

# Verify: Agent ran evaluation and reported result
assert result.evaluation_output is not None
assert "diversity" in result.evaluation_output or "score" in result.evaluation_output
```

##### Test 2: Developer Agent Implements Solution Correctly

```python
# Verify: Agent modified gpt2.py or created new file with temperature support
gpt2_code = open(os.path.join(workspace, "gpt2.py")).read()

# Should have temperature parameter
assert "temperature" in gpt2_code

# Should have sampling logic (not just argmax)
assert "softmax" in gpt2_code or "multinomial" in gpt2_code or "random" in gpt2_code
```

##### Test 3: Feedback Generator Validates Evaluation

```python
# Mock a bad evaluation that always returns perfect score
bad_eval_result = {
    "score": 1.0,
    "diversity_score": 1.0,
    "message": "Perfect!"
}

bad_eval_code = """
# Bad evaluation - doesn't actually test anything
print('{"score": 1.0, "diversity_score": 1.0}')
"""

feedback = feedback_generator.generate(
    goal=goal,
    idea="Add temperature sampling",
    implementation="def generate(..., temperature=1.0): ...",
    evaluation_code=bad_eval_code,
    evaluation_result=bad_eval_result,
)

# Feedback generator should catch this
assert feedback.evaluation_valid == False
assert "doesn't actually test" in feedback.feedback.lower() or \
       "not measuring" in feedback.feedback.lower()
```

##### Test 4: Feedback Generator Stops When Goal Reached

```python
# Good evaluation result showing temperature works
good_eval_result = {
    "greedy_outputs": ["The cat sat on the mat"] * 5,  # All same (deterministic)
    "temp_1_outputs": [
        "The cat sat on the mat",
        "The cat jumped over the fence", 
        "The cat ran through the garden",
        "The cat slept by the window",
        "The cat played with yarn"
    ],  # All different (diverse)
    "greedy_diversity": 0.0,  # No diversity
    "temp_1_diversity": 1.0,  # Full diversity
    "score": 1.0,
}

good_eval_code = """
# Run generation 5 times with greedy and temp=1.0
# Measure unique outputs / total outputs
greedy_outputs = [generate(prompt, temperature=0.0) for _ in range(5)]
temp_outputs = [generate(prompt, temperature=1.0) for _ in range(5)]

greedy_diversity = len(set(greedy_outputs)) / len(greedy_outputs)
temp_diversity = len(set(temp_outputs)) / len(temp_outputs)

# Success: greedy should be deterministic, temp should be diverse
score = 1.0 if greedy_diversity < 0.2 and temp_diversity > 0.6 else 0.0
print(json.dumps({"score": score, ...}))
"""

feedback = feedback_generator.generate(
    goal=goal,
    idea="Add temperature sampling with softmax",
    implementation="...",  # Actual implementation diff
    evaluation_code=good_eval_code,
    evaluation_result=good_eval_result,
)

# Should stop - goal achieved
assert feedback.stop == True
assert feedback.evaluation_valid == True
assert "goal achieved" in feedback.feedback.lower() or \
       "success" in feedback.feedback.lower()
```

##### Test 5: Feedback Generator Provides Actionable Feedback

```python
# Partial success - temperature works but diversity is low
partial_eval_result = {
    "greedy_diversity": 0.0,
    "temp_1_diversity": 0.4,  # Some diversity but not enough
    "score": 0.4,
}

feedback = feedback_generator.generate(
    goal=goal,
    idea="Add temperature sampling",
    implementation="...",
    evaluation_code="...",
    evaluation_result=partial_eval_result,
)

# Should NOT stop - goal not fully achieved
assert feedback.stop == False
assert feedback.evaluation_valid == True

# Should provide actionable feedback
assert len(feedback.feedback) > 50  # Substantial feedback
# Should mention what to improve
assert "diversity" in feedback.feedback.lower() or \
       "temperature" in feedback.feedback.lower() or \
       "sampling" in feedback.feedback.lower()
```

##### Test 6: Retry Logic When Evaluation Crashes

```python
# Simulate evaluation that crashes on first run
crash_count = 0

def mock_run_evaluation():
    global crash_count
    crash_count += 1
    if crash_count < 3:
        raise RuntimeError("CUDA out of memory")
    return {"score": 0.8, "diversity": 0.7}

# Developer agent should retry up to N times
result = developer_agent.run_with_retries(
    max_retries=3,
    run_fn=mock_run_evaluation,
)

# Should succeed after retries
assert result is not None
assert result["score"] == 0.8
assert crash_count == 3  # Took 3 attempts
```

##### Test 7: Full Iteration Loop End-to-End

```python
# Full test: Initialize repo → Setup dirs → Run iterations → Get result

# Step 1: Initialize (from 01_initialize_repo.md)
kapso = Kapso()
kapso.index_kg(wiki_dir="data/wikis_llm_finetuning_test", save_to="test.index")
kapso = Kapso(kg_index="test.index")

# Step 2 & 3: Evolve with the goal
solution = kapso.evolve(
    goal="""
    Enhance PicoGPT to support temperature-based sampling.
    Add a temperature parameter (default 1.0) that controls randomness.
    temperature=0.0 should be deterministic (greedy).
    temperature>0 should produce varied outputs.
    Target: diversity score > 0.6 with temperature=1.0
    """,
    initial_repo=None,  # Should find PicoGPT workflow
    max_iterations=5,
)

# Verify: Solution achieved goal or made progress
assert solution.code_path is not None
assert len(solution.experiment_logs) > 0

# Verify: Final code has temperature support
final_code = open(os.path.join(solution.code_path, "gpt2.py")).read()
assert "temperature" in final_code

# Verify: Evaluation was built
assert os.path.exists(os.path.join(solution.code_path, "kapso_evaluation"))
```

#### Expected Iteration Flow for Test Scenario

```
Iteration 1:
  Developer Agent:
    - Reads repo_memory → understands PicoGPT structure
    - Modifies gpt2.py: adds temperature parameter to generate()
    - Creates kapso_evaluation/evaluate.py: measures output diversity
    - Runs evaluation: {"greedy_diversity": 0.0, "temp_diversity": 0.3, "score": 0.3}
  
  Feedback Generator:
    - Evaluation valid: Yes (actually measures diversity)
    - Goal reached: No (diversity 0.3 < 0.6 target)
    - Feedback: "Temperature sampling implemented but diversity is low. 
                 Consider using proper softmax with temperature scaling 
                 instead of just adding noise."

Iteration 2:
  Developer Agent:
    - Reads feedback
    - Improves sampling: proper softmax(logits/temperature)
    - Runs evaluation: {"greedy_diversity": 0.0, "temp_diversity": 0.5, "score": 0.5}
  
  Feedback Generator:
    - Evaluation valid: Yes
    - Goal reached: No (diversity 0.5 < 0.6 target)
    - Feedback: "Better! Diversity improved. Try using top-k or top-p 
                 sampling in addition to temperature for more variety."

Iteration 3:
  Developer Agent:
    - Adds top-k sampling with temperature
    - Runs evaluation: {"greedy_diversity": 0.0, "temp_diversity": 0.75, "score": 0.75}
  
  Feedback Generator:
    - Evaluation valid: Yes
    - Goal reached: Yes (diversity 0.75 > 0.6 target)
    - Feedback: "Goal achieved! Temperature-based sampling now produces 
                 diverse outputs (0.75 diversity score)."
    - STOP
```

### Testing Considerations Summary
- Test developer agent builds evaluation correctly
- Test developer agent runs evaluation and reports result
- Test feedback generator validates evaluation (catches bad evals)
- Test feedback generator stops when goal reached
- Test feedback generator provides useful feedback (actionable)
- Test retry logic when evaluation crashes
- Test full iteration loop end-to-end with real workflow
