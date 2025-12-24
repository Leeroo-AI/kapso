# Extract Learning Insight

Extract a generalizable insight from the experiment.

## Result
Success: {success}
Error: {error_message}
Code: {code_summary}
Step: {workflow_step}

## Response Format (JSON only)
```json
{"insight": "rule under 200 chars", "type": "critical_error|best_practice|workaround", "confidence": 0.0-1.0}
```
