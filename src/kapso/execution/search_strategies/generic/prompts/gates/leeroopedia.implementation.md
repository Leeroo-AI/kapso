### Leeroopedia (MCP Tools) — curated ML/AI framework knowledge
A hosted knowledge base of how ML/AI frameworks, libraries and tools actually
behave: documented APIs, config formats, recommended values, known failure
modes. Answers carry `[PageID]` citations. Each call takes about a minute and
spends credits, so use it at decision points, not continuously.

- **build_plan**: before you touch code, when the solution brings in a
  framework or library this repository does not already use
  - Example: `build_plan(goal="<what you are implementing>", constraints="<hardware, time, data>")`

- **search_knowledge**: a config format, an API contract or expected behavior
  you would otherwise guess at
  - Example: `search_knowledge(query="PEFT LoraConfig target_modules for Llama", context="transformers 4.x")`

- **verify_code_math**: a math-heavy or algorithmic change, before you run it
  - Example: `verify_code_math(code_snippet="<the function>", concept_name="LoRA low-rank update")`

- **diagnose_failure**: the first time a run fails in a way you do not
  immediately understand — pass the symptoms and the logs
  - Example: `diagnose_failure(symptoms="loss goes NaN at step 100", logs="<the traceback or log tail>")`

- **query_hyperparameter_priors**: documented ranges before you set a value
  the solution leaves open

- **get_page**: the full page behind a `[PageID]` citation

When an answer shaped what you built, cite its `[PageID]` in a code comment at
that site — cite real use only.
