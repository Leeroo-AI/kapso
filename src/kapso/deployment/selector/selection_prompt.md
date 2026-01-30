# Strategy Selection Task

Analyze this project and select the best deployment strategy.

## Goal
{goal}

## Available Strategies

{strategy_descriptions}

## Your Task

Based on the goal, select the most appropriate deployment strategy.
Consider:
- Complexity of the solution
- Resource requirements (GPU, memory)
- Whether it's an agent/LangGraph application
- Production vs development needs

## Output Format

Return ONLY a JSON object:
```json
{{
    "reasoning": "<brief explanation>",
    "strategy": "<strategy_name>",
    "resources": {{"gpu": "...", "memory": "..."}}
}}
```

The strategy MUST be one of: {allowed_strategies}

