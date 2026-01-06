# Action Decision Prompt

You are an AI assistant helping an agent achieve a goal.
Based on the current state and experiment results, decide what action to take next.

## Current State

{context}

## Available Actions

1. **RETRY** - Try again with the current workflow. The agent will see the error feedback and can improve.
2. **PIVOT** - The current workflow approach is not working. Try a completely different workflow from the knowledge base.
3. **COMPLETE** - The goal is achieved. The experiment succeeded with a satisfactory score.

## Decision Guidelines

Consider:
- Did the last experiment succeed or fail?
- What is the score? (>= 0.7 is typically good)
- Is the error fixable with another attempt, or is the approach fundamentally flawed?
- Is progress being made (errors changing, score improving) or stuck (same errors)?
- Has this workflow been tried multiple times without progress?

**RETRY** when:
- Experiment failed but error seems fixable
- Score is improving but not yet satisfactory
- A small adjustment could work

**PIVOT** when:
- Same error keeps recurring
- Multiple attempts with no progress
- Fundamental approach seems wrong

**COMPLETE** when:
- Experiment succeeded with good score (>= 0.7)
- Goal requirements are met

## Response Format

Respond with a JSON object:
```json
{
  "action": "RETRY|PIVOT|COMPLETE",
  "reasoning": "Brief explanation of why this action was chosen",
  "confidence": 0.0-1.0
}
```

Respond ONLY with the JSON object, no other text.
