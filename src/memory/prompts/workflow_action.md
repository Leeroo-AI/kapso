# Workflow Action Decision

Decide what action to take on the workflow.

## Context
{context}

## Actions: continue, retry, advance, modify_step, pivot, complete

## Response Format (JSON only)
```json
{"action": "continue|retry|advance|modify_step|pivot|complete", "params": {}, "confidence": 0.0-1.0, "reasoning": "explanation"}
```
