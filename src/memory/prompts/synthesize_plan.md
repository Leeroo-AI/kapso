# Plan Synthesis Prompt

You are an AI assistant helping to create a step-by-step plan to achieve a goal.
Based on the available knowledge, synthesize a structured workflow.

## Goal

{goal}

## Available Knowledge

{knowledge_pages}

## Instructions

Create a step-by-step plan to achieve the goal. Each step should be:
- Concrete and actionable
- Build on previous steps
- Use the knowledge provided where relevant

## Response Format

Respond with a JSON array of step titles (strings only):
```json
["Step 1 title", "Step 2 title", "Step 3 title", ...]
```

Keep it to 3-7 steps. Respond ONLY with the JSON array, no other text.
