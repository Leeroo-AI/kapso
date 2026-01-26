# Task 08: Developer Agent Returns Structured JSON

## Issue Description

Currently, after implementation, we read evaluation results from a file on the filesystem:

```python
# src/execution/search_strategies/linear_search.py lines 109-122
result_json_path = os.path.join(self.workspace_dir, "kapso_evaluation", "result.json")
if os.path.exists(result_json_path):
    try:
        with open(result_json_path, 'r') as f:
            eval_result = json.load(f)
        evaluation_output = eval_result.get("evaluation_output", result.output)
        evaluation_script_path = eval_result.get("evaluation_script_path", "")
    except Exception as e:
        print(f"[LinearSearch] Warning: Could not read result.json: {e}")
```

This approach is **fragile** because:
1. Relies on file existence at a specific path
2. Agent might not create the file
3. File format might vary
4. No guarantee agent writes all required fields

## Proposed Solution

Instead of file-based extraction, instruct the developer agent to return a **structured JSON** in its final response. Then extract this JSON directly from the agent's output.

## Required JSON Structure

The developer agent should return:

```json
{
    "code_changes_summary": "Brief summary of what was implemented/changed",
    "evaluation_script_path": "kapso_evaluation/evaluate.py",
    "evaluation_output": "Full output from running the evaluation script"
}
```

## Files to Modify

### 1. `src/execution/prompts/coding_agent_implement.md`

**Changes:**
- Add clear instructions for the agent to return structured JSON when done
- Specify the exact JSON format required
- Emphasize this is mandatory for task completion

**Add to prompt:**
```markdown
## IMPORTANT: Final Output Format

When you have completed the implementation and evaluation, you MUST return a JSON object with the following structure:

```json
{
    "code_changes_summary": "Brief description of changes made",
    "evaluation_script_path": "path/to/evaluation/script.py",
    "evaluation_output": "Full output from running evaluation"
}
```

This JSON should be the LAST thing in your response, wrapped in ```json code blocks.

Requirements:
- `code_changes_summary`: 2-5 sentences describing what you implemented
- `evaluation_script_path`: Relative path to the evaluation script you created
- `evaluation_output`: Complete stdout/stderr from running the evaluation
```

### 2. `src/execution/prompts/coding_agent_debug.md`

**Changes:**
- Same JSON output requirement for debug mode
- Agent should return updated results after fixing errors

### 3. `src/execution/search_strategies/base.py`

**Changes:**
- Add `_extract_agent_result()` method to parse JSON from agent output
- Update `implement_solution()` to return parsed result instead of calling `problem_handler.run()`
- Handle cases where JSON is missing or malformed

**New method:**
```python
def _extract_agent_result(self, agent_output: str) -> dict:
    """
    Extract structured JSON result from agent output.
    
    Looks for JSON block at the end of the output:
    ```json
    {"code_changes_summary": "...", ...}
    ```
    
    Returns:
        dict with keys: code_changes_summary, evaluation_script_path, evaluation_output
        Returns empty dict if extraction fails
    """
    import re
    import json
    
    # Look for JSON in code blocks
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, agent_output, re.DOTALL)
    
    if matches:
        try:
            # Take the last JSON block (final result)
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find raw JSON object
    try:
        # Find last occurrence of {...}
        start = agent_output.rfind('{')
        end = agent_output.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(agent_output[start:end])
    except json.JSONDecodeError:
        pass
    
    return {}
```

### 4. `src/execution/search_strategies/linear_search.py`

**Changes:**
- Remove file-based result extraction (lines 109-122)
- Use `_extract_agent_result()` from base class
- Populate `SearchNode` fields from extracted JSON

### 5. `src/execution/search_strategies/llm_tree_search.py`

**Changes:**
- Same updates as linear_search
- Use extracted JSON to populate node fields

## Error Handling

If JSON extraction fails:
1. Log a warning
2. Set `had_error = True` on the node
3. Use empty strings for missing fields
4. Feedback generator will see the error and provide appropriate feedback

## Example Agent Output

```
I've implemented the GPT-2 text generation solution. Here's what I did:

1. Created main.py with the transformer implementation
2. Added utils.py for tokenization
3. Created kapso_evaluation/evaluate.py to test generation

Running evaluation...
[output from evaluation]

```json
{
    "code_changes_summary": "Implemented GPT-2 text generation with transformer blocks, attention mechanism, and greedy decoding. Created evaluation script that tests token generation count and output coherence.",
    "evaluation_script_path": "kapso_evaluation/evaluate.py",
    "evaluation_output": "Loading model...\nGenerating 40 tokens...\nGenerated text: 'Alan Turing theorized...'\nScore: 95/100\nAll tests passed."
}
```
```

## Dependencies

- None (this task can be done independently)

## Testing

After changes:
1. Run `tests/demo_task1_initialize_repo.py`
2. Verify agent returns JSON in expected format
3. Verify JSON is correctly extracted
4. Verify `SearchNode` fields are populated from JSON
5. Test error handling when JSON is missing/malformed
