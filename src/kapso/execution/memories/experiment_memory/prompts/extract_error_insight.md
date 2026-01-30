# Extract Error Insight
#
# Variables: {context}, {error_message}
#
# This prompt extracts a generalized lesson from a coding error.

You are extracting reusable lessons from coding errors.

## Context
{context}

## Error
{error_message}

## Task
Extract a GENERALIZED, REUSABLE lesson from this error.
Don't just repeat the error - explain what went wrong and how to prevent it.

Respond in JSON:
```json
{{
  "lesson": "A general principle that applies beyond this specific case",
  "trigger_conditions": "When/where this issue typically occurs",
  "suggested_fix": "Actionable steps to fix or prevent this",
  "confidence": 0.0-1.0,
  "tags": ["keyword1", "keyword2", "keyword3"]
}}
```

Make the lesson USEFUL for future similar problems.
Respond ONLY with JSON.
