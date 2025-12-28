# Error Response Decision

Decide how to respond to the error.

## Context
{context}

## Error
{error_message}

## Responses: retry_same, apply_fix, try_alternative, escalate

## Response Format (JSON only)
```json
{"response": "retry_same|apply_fix|try_alternative|escalate", "params": {}, "confidence": 0.0-1.0, "reasoning": "explanation"}
```
