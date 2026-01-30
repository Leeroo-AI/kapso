You are a world class developer. Debug the Implemented <solution> for <problem>.

## Repository Memory

{{repo_memory_brief}}

{{repo_memory_detail_access_instructions}}

OBSERVABILITY REQUIREMENT (do not skip):
- If you consulted repo memory sections, record which ones in `changes.log`.
- Add a line exactly like:
  RepoMemory sections consulted: core.architecture, core.where_to_edit
- If you did not consult repo memory, write:
  RepoMemory sections consulted: none

## Problem

<problem>
{{problem}}
</problem>

## Solution

<solution>
{{solution}}
</solution>

## Current Error

{{error_details}}

## Debug Requirements

- Read the code line by line and understand the logic.
- Make sure every part of the <solution> is implemented correctly.
  - Read sections and steps of <solution> carefully and implement them exactly.
- Do not propose a new solution or drift away from the current implementation.
- Write clean, functional code that can be improved iteratively later.
- Output code and format must be as mentioned in the problem statement.
- Do not add fallback logic or discard functionality to avoid the error. Fix the error directly.
- Never use try-except blocks to hide errors. Fix the root cause.
- Check other parts of the code to ensure they will run correctly.
- Do not change hyperparameters or solution logic to fix the error.

## Evaluation

If the error is in the evaluation code (`kapso_evaluation/`):
- Fix the evaluation script to run correctly
- Ensure evaluation still tests what it claims to test
- Re-run evaluation after fixing

## CRITICAL: Final Output Format

When you have fixed the error and re-run evaluation, you MUST return your results using these XML tags as the LAST thing in your response:

<code_changes_summary>
Brief description of what you fixed (2-5 sentences)
</code_changes_summary>

<evaluation_script_path>
kapso_evaluation/evaluate.py
</evaluation_script_path>

<evaluation_output>
Full stdout/stderr output from running the evaluation script
</evaluation_output>

<score>
0.95
</score>

**Requirements:**
- `<code_changes_summary>`: 2-5 sentences describing what you fixed
- `<evaluation_script_path>`: Relative path to the evaluation script
- `<evaluation_output>`: Complete stdout/stderr from running the evaluation
- `<score>`: Numeric score from evaluation (use 0 if no score available, or "null" if evaluation failed)

**These tags are MANDATORY. The system extracts results from these tags.**

Do not ask any questions. Fix the error and re-run if needed.
