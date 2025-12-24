# Should Consult Knowledge Graph

Determine whether to consult the Knowledge Graph.

## Context
{context}

## Response Format (JSON only)
```json
{"should_consult": true/false, "reason": "new_workflow|error_occurred|progress_stalled|periodic_check|null", "confidence": 0.0-1.0, "reasoning": "explanation"}
```
