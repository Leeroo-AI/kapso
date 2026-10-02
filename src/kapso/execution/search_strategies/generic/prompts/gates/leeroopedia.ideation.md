### Leeroopedia (MCP Tools) — curated ML/AI framework knowledge
A hosted knowledge base of how ML/AI frameworks, libraries and tools actually
behave: documented APIs, config formats, recommended values, known failure
modes. Answers are synthesized with `[PageID]` citations. Each call takes
about a minute and spends credits, so call it at decision points — one or two
targeted calls, several in parallel when the questions are independent — not
on every step.

- **search_knowledge**: how a framework or technique actually works, before
  you propose around it
  - Example: `search_knowledge(query="QLoRA learning-rate scaling with rank", context="7B model, single GPU")`

- **propose_hypothesis**: ranked approaches when you are choosing between
  directions; pass what has already been tried
  - Example: `propose_hypothesis(current_status="<where the campaign stands>", recent_experiments="<what was tried and how it scored>")`

- **query_hyperparameter_priors**: documented ranges for every numeric knob
  your proposal fixes (learning rate, batch size, rank, epochs)
  - Example: `query_hyperparameter_priors(query="LoRA rank and alpha for a 7B instruction model")`

- **review_plan**: your final proposal against documented behavior and known
  pitfalls, before you emit it
  - Example: `review_plan(proposal="<your plan>", goal="<the GOAL>")`

- **get_page**: the full page behind a `[PageID]` citation you want to read

Carry the citations into your solution's rationale as `[PageID]` wherever a
step rests on them — cite real use only.
