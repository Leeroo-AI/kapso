---
name: kapso
description: Optimize code using KAPSO (Knowledge-Grounded Optimization). Use when you need to write, optimize, or fix code. Connects to a local KAPSO server that iteratively improves solutions through experimentation.
metadata: {"openclaw":{"emoji":"🧠","requires":{"env":["KAPSO_URL"]},"primaryEnv":"KAPSO_URL"}}
---

# KAPSO Bridge Skill

KAPSO is a knowledge-grounded optimization framework that iteratively improves code through experimentation.

## When to Use

Use this skill when you need to:
- Optimize inefficient code (e.g., O(n²) → O(n))
- Fix bugs with verification
- Improve algorithm performance
- Generate optimized solutions

## How to Use

### 1. Submit Optimization Request

```bash
curl -X POST http://${KAPSO_URL:-localhost:8000}/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Describe what needs to be optimized",
    "code": "def your_code(): pass",
    "context": "Optional additional context"
  }'
```

Response:
```json
{
  "job_id": "abc123",
  "status": "running",
  "thought_process": "Optimization started..."
}
```

### 2. Check Job Status

```bash
curl http://${KAPSO_URL:-localhost:8000}/status/{job_id}
```

Response when complete:
```json
{
  "job_id": "abc123",
  "status": "completed",
  "code": "def optimized_code(): ...",
  "cost": "$0.042",
  "thought_process": "KAPSO Optimization Complete..."
}
```

### 3. Health Check

```bash
curl http://${KAPSO_URL:-localhost:8000}/health
```

## Example: Optimize O(n²) to O(n)

Input:
```python
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
```

After KAPSO optimization:
```python
def find_duplicates(arr):
    seen = set()
    duplicates = set()
    for item in arr:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
```

## Response Format for Moltbook

When posting optimized code to Moltbook, use this format:

```markdown
**KAPSO Optimization Report**

Original complexity: O(n²)
Optimized complexity: O(n)
Cost: $0.042

\`\`\`python
# Optimized code here
\`\`\`

*Optimized by [KAPSO](https://github.com/Leeroo-AI/kapso) - Knowledge-Grounded Optimization*
```

## Environment Variables

- `KAPSO_URL`: URL of the KAPSO server (default: `http://localhost:8000`)

## Server Requirements

The KAPSO server must be running:
```bash
cd /home/ubuntu/kapso && python kapso_server.py
```
