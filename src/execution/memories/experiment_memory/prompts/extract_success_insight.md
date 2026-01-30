# Extract Success Insight
#
# Variables: {context}, {feedback}
#
# This prompt extracts a best practice from successful code solutions.

You are extracting best practices from successful code solutions.

## Context
{context}

## Evaluator Feedback
{feedback}

## Task
Extract a REUSABLE best practice from this success.
What made this solution work well? What pattern should be repeated?

Respond in JSON:
```json
{{
  "lesson": "A best practice or pattern that worked well",
  "trigger_conditions": "When to apply this pattern",
  "suggested_fix": "How to implement this pattern",
  "confidence": 0.0-1.0,
  "tags": ["keyword1", "keyword2", "keyword3"]
}}
```

Focus on PATTERNS that transfer to other problems.
Respond ONLY with JSON.
