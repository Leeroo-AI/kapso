# Task 5: Cleanup and Dependencies

## Overview

This task covers cleanup of removed modules and updating all import dependencies.

## Modules Being Removed

### 1. Evaluators (`src/environment/evaluators/`)
- `base.py` - Evaluator base class, EvaluationResult
- `builtin.py` - All built-in evaluators (ScriptEvaluator, RegexPatternEvaluator, etc.)
- `factory.py` - EvaluatorFactory
- `__init__.py` - Exports

### 2. Stop Conditions (`src/environment/stop_conditions/`)
- `base.py` - StopCondition base class, StopDecision
- `builtin.py` - All built-in stop conditions
- `factory.py` - StopConditionFactory
- `__init__.py` - Exports

## Files That Import Removed Modules

### Must Update

1. **`src/environment/handlers/generic.py`**
   - Lines 23-24: Imports Evaluator, EvaluatorFactory, StopCondition, StopConditionFactory
   - Line 63: Uses EvaluatorFactory
   - **Action:** Remove all evaluator/stop_condition logic

2. **`src/cli.py`**
   - Lines 33-34: Imports EvaluatorFactory, StopConditionFactory
   - **Action:** Remove CLI options for evaluator/stop_condition selection

3. **`src/environment/__init__.py`**
   - Lines 18-26: Re-exports evaluators and stop_conditions
   - **Action:** Remove these exports

### Internal (Will Be Deleted)

These files import from each other but will all be deleted:
- `src/environment/evaluators/factory.py`
- `src/environment/evaluators/builtin.py`
- `src/environment/evaluators/__init__.py`
- `src/environment/stop_conditions/factory.py`
- `src/environment/stop_conditions/builtin.py`
- `src/environment/stop_conditions/__init__.py`

## Other Cleanup

### Context Managers

Check if context managers reference evaluators/stop_conditions:

1. **`src/execution/context_manager/`**
   - May reference stop conditions in decision logic
   - Review and update as needed

### Config Files

1. **`src/config.yaml`**
   - May have evaluator/stop_condition defaults
   - Remove or update

2. **`src/execution/search_strategies/strategies.yaml`**
   - Check for evaluator references

### Tests

1. **`tests/`**
   - Find and update/remove tests for evaluators
   - Find and update/remove tests for stop_conditions

## Cleanup Checklist

- [ ] Delete `src/environment/evaluators/` directory
- [ ] Delete `src/environment/stop_conditions/` directory
- [ ] Update `src/environment/handlers/generic.py`
- [ ] Update `src/environment/handlers/base.py`
- [ ] Update `src/environment/__init__.py`
- [ ] Update `src/cli.py`
- [ ] Check and update `src/config.yaml`
- [ ] Check and update context managers
- [ ] Update/remove related tests
- [ ] Run full test suite to catch any missed imports

## Files to Touch

**Delete (entire directories):**
- `src/environment/evaluators/`
- `src/environment/stop_conditions/`

**Modify:**
- `src/environment/handlers/generic.py`
- `src/environment/handlers/base.py`
- `src/environment/handlers/__init__.py`
- `src/environment/__init__.py`
- `src/cli.py`
- `src/config.yaml` (if needed)

## Cross-References
- Related to: `03_developer_agent_loop.md` (main changes)
