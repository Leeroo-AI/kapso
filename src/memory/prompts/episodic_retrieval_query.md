# Episodic Memory Retrieval Query
#
# Variables: {goal}, {current_step}, {last_error}, {available_insights}
#
# This prompt helps the LLM generate a better query for episodic memory retrieval
# and filter which insights are actually relevant.

You are helping retrieve relevant past learnings from episodic memory.

## Current Task
Goal: {goal}
Current Step: {current_step}
Last Error (if any): {last_error}

## Available Insights
{available_insights}

## Task
1. Generate a refined search query that captures what we need to learn
2. Select which insights (by index) are most relevant to the current situation
3. For each selected insight, explain WHY it's relevant

Respond in JSON:
```json
{{
  "refined_query": "A focused query to find relevant past experiences",
  "selected_indices": [0, 2, 5],
  "relevance_reasons": [
    "This insight addresses the same type of error we're seeing",
    "This pattern applies to our current step",
    "This best practice would help here"
  ],
  "confidence": 0.0-1.0
}}
```

Only select insights that are DIRECTLY relevant. If none are relevant, return empty lists.
Respond ONLY with JSON.



