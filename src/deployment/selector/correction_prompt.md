# JSON Extraction Prompt

Extract the deployment configuration from the text below and return ONLY valid JSON.

## Text to Parse
{text}

## Required Output Format

Return a JSON object with these exact fields:
- `strategy`: one of "local", "docker", "modal", "bentoml", "langgraph"
- `resources`: object with cpu/memory/gpu (can be empty {{}})
- `reasoning`: brief explanation of the choice

## Example Output

```json
{{
  "strategy": "modal",
  "resources": {{"gpu": "T4", "memory": "16Gi"}},
  "reasoning": "GPU workload detected"
}}
```

Output ONLY the JSON. No other text, no explanation, no markdown.

