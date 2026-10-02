### Experiment History (MCP Tools)
**IMPORTANT: You MUST check experiment history before generating a solution.**

- **get_top_experiments**: Get the best-scoring experiments so far
  - Use this to understand what approaches have worked well
  - Example: `get_top_experiments(k=5)` returns top 5 experiments by score

- **get_recent_experiments**: Get the most recent experiments
  - Use this to see what was tried recently and avoid repeating failures
  - Example: `get_recent_experiments(k=5)` returns last 5 experiments

- **search_similar_experiments**: Search for experiments similar to your idea
  - Use this to check if your approach was already tried
  - Example: `search_similar_experiments(query="<your candidate approach>", k=3)`
  - Without a configured embedding model this returns the most recent experiments
