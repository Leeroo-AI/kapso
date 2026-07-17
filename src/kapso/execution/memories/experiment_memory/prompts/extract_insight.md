# Extract Insight
#
# Variables: {context}, {technical_difficulties}, {feedback}
#
# Single extraction path: runs for every finished experiment, success or
# failure. The model classifies the insight type itself.

You are extracting a reusable lesson from a finished ML experiment.

## Context
{context}

## Technical difficulties reported by the implementor
{technical_difficulties}

## Evaluator feedback
{feedback}

## Task
Extract ONE generalized, reusable lesson — the thing most worth telling
the next implementor before they start. Prefer a difficulty that was hit
and resolved (the fix is the lesson); if the experiment went smoothly,
extract the pattern that made it work.

Respond in JSON:
```json
{{
  "lesson": "A general principle that applies beyond this specific case",
  "trigger_conditions": "When/where this applies",
  "suggested_fix": "Actionable steps to apply or prevent",
  "confidence": 0.0-1.0,
  "insight_type": "critical_error" or "best_practice",
  "tags": ["keyword1", "keyword2", "keyword3"]
}}
```

- "critical_error": a mistake/trap to avoid (even if it was recovered from).
- "best_practice": a pattern that worked and should be repeated.

Focus on lessons that TRANSFER to other problems. Respond ONLY with JSON.
