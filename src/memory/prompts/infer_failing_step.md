# Infer Failing Step Prompt

You are analyzing an error to identify which step(s) in a workflow are most likely causing the failure.

## Current Context

{context}

## Error Message

{error}

## Workflow Steps

{workflow_steps}

## Task

Analyze the error message and determine which workflow step(s) are most likely responsible.
Consider:
- Error type (OOM, TypeError, ValueError, etc.)
- Stack trace clues (function names, variable names)
- Which step's operations would cause this type of error

## Response Format

Respond with a JSON object:
```json
{
  "failing_steps": [
    {
      "step_number": 1,
      "principle_id": "Principle/Example_Principle",
      "confidence": 0.8,
      "reasoning": "Brief explanation"
    }
  ],
  "error_category": "memory|type|value|import|runtime|unknown",
  "suggested_search_terms": ["term1", "term2"]
}
```

Include 1-3 most likely failing steps, ordered by confidence.
Respond ONLY with the JSON object, no other text.

