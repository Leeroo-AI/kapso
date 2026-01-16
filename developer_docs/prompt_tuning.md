# Prompt Tuning (externalized templates)
#
# This project externalizes large prompt strings into files so they can be tuned
# without touching python logic. This is critical for RepoMemory quality and for
# agent prompt iteration.

## Where prompts live

### RepoMemory builders

- `src/repo_memory/prompts/plan_files_to_read.md`
- `src/repo_memory/prompts/infer_repo_model_initial.md`
- `src/repo_memory/prompts/infer_repo_model_retry.md`
- `src/repo_memory/prompts/infer_repo_model_update.md`

### Execution prompts (agents + ideation)

- `src/execution/prompts/coding_agent_implement.md`
- `src/execution/prompts/coding_agent_debug.md`
- `src/execution/prompts/ideation_solution_react.md`

## How prompts are loaded

All prompt loading is centralized in:

- `src/core/prompt_loader.py`

It provides:

- `load_prompt("execution/prompts/coding_agent_implement.md")`
- `render_prompt(template, {"var": "value"})` using `{{var}}` placeholders

## Override prompts without changing code

Set an environment variable:

```bash
export TINKERER_PROMPTS_DIR=/path/to/your/prompts_root
```

Then a call like:

```python
load_prompt("execution/prompts/coding_agent_implement.md")
```

will read from:

`/path/to/your/prompts_root/execution/prompts/coding_agent_implement.md`

This is useful when you want to tune prompts in a separate directory (e.g., mounted
volume or a different git branch) while keeping the engine code unchanged.

